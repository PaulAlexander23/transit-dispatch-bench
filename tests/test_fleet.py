"""The fleet env: the Python reference, JAX against it, and the heuristics."""

import itertools
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from transit_bench import make
from transit_bench.core.policy import RandomPolicy, make_rollout
from transit_bench.envs.fleet import PRESETS, FleetConfig, FleetEnv
from transit_bench.envs.fleet.heuristic import Matching, NearestIdle, Rebalance, Stay
from transit_bench.envs.fleet.sim import (
    EAST,
    NORTH,
    STAY,
    TAKE,
    TO_DROPOFF,
    TO_PICKUP,
    Exogenous,
    FleetSim,
    auction,
    matching,
    nearest_idle,
    rebalance,
    stay,
)


@pytest.fixture
def x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def none():
    return Exogenous(np.zeros(0, int), np.zeros(0, int))


def one(origin, dest):
    return Exogenous(np.array([origin]), np.array([dest]))


# -- the Python reference ------------------------------------------------------


def test_a_request_is_taken_picked_up_and_delivered_in_manhattan_time():
    c = replace(FleetConfig(), n_vehicles=1)  # the vehicle starts at node 0
    sim = FleetSim(c)
    sim.step([STAY], one(12, 14))  # (2, 2) to (4, 2)
    assert sim.valid[0] and sim.available(0)
    _, reward, _, _, info = sim.step([TAKE + 0], none())
    assert sim.phase[0] == TO_PICKUP and not sim.available(0) and sim.pos[0] == 1
    assert reward == pytest.approx(-1.0)  # one request waited a minute
    for _ in range(3):  # 4 edges to (2, 2), one already done
        _, _, _, _, info = sim.step([STAY], none())
    # Picked up in the vehicle's turn of the 4th tick, before requests age:
    # it waited 3 full ticks, as the reward counted.
    assert info["picked_up"] == 1 and info["wait_ticks"] == 3
    assert sim.phase[0] == TO_DROPOFF and not sim.valid[0]
    sim.step([STAY], none())
    _, reward, _, _, info = sim.step([STAY], none())
    assert info["delivered"] == 1 and sim.pos[0] == 14 and reward == 0.0


def test_an_unassigned_request_abandons_after_its_patience():
    c = replace(FleetConfig(), n_vehicles=1, patience=3)
    sim = FleetSim(c)
    sim.step([STAY], one(5, 6))
    for _ in range(2):
        _, _, _, _, info = sim.step([STAY], none())
        assert info["abandoned"] == 0
    _, reward, _, _, info = sim.step([STAY], none())
    assert info["abandoned"] == 1 and not sim.valid[0]
    assert reward == pytest.approx(-(1.0 + c.abandon_penalty))


def test_moves_reposition_and_stop_at_the_edge_of_the_grid():
    sim = FleetSim(replace(FleetConfig(), n_vehicles=1))
    sim.step([NORTH], none())
    assert sim.pos[0] == 5
    sim.pos[0] = 4  # the south-east corner
    sim.step([EAST], none())
    assert sim.pos[0] == 4


def test_two_vehicles_taking_one_request_the_earlier_gets_it():
    sim = FleetSim(replace(FleetConfig(), n_vehicles=2))
    sim.step([STAY, STAY], one(7, 8))
    sim.step([TAKE + 0, TAKE + 0], none())
    assert sim.taken[0] == 0 or not sim.valid[0]
    assert sim.phase[1] == 0


def test_a_full_buffer_rejects_arrivals():
    c = replace(FleetConfig(), max_requests=2)
    sim = FleetSim(c)
    _, reward, _, _, info = sim.step(stay(sim), Exogenous(np.array([1, 2, 3]), np.array([4, 5, 6])))
    assert info["rejected"] == 1 and sum(sim.valid) == 2
    assert reward == pytest.approx(-c.abandon_penalty)


def brute_force(cost):
    """The minimum total cost over assignments of every bidder to a
    distinct allowed object."""
    bidders = [i for i, row in enumerate(cost) if any(x is not None for x in row)]
    best = None
    for objs in itertools.permutations(range(len(cost[0])), len(bidders)):
        if any(cost[i][j] is None for i, j in zip(bidders, objs, strict=True)):
            continue
        total = sum(cost[i][j] for i, j in zip(bidders, objs, strict=True))
        best = total if best is None else min(best, total)
    return best


def test_the_auction_finds_the_optimal_assignment():
    rng = np.random.default_rng(0)
    for _ in range(60):
        rows, cols = rng.integers(1, 5), rng.integers(4, 7)
        cost = rng.integers(0, 9, (rows, cols)).tolist()
        for i in range(rows):
            for j in range(cols):
                if rng.random() < 0.2 and j < cols - rows:  # leave each row options
                    cost[i][j] = None
        if rng.random() < 0.3:
            cost[0] = [None] * cols  # a non-bidder
        assigned = auction(cost)
        used = [j for j in assigned if j >= 0]
        assert len(used) == len(set(used))
        total = sum(cost[i][j] for i, j in enumerate(assigned) if j >= 0)
        assert total == brute_force(cost)


@pytest.mark.parametrize("preset", PRESETS)
def test_matching_beats_nearest_idle_beats_staying(preset):
    def run(rule):
        total = 0.0
        for seed in range(4):
            sim = FleetSim(preset, max_steps=120)
            sim.reset(seed)
            done = False
            while not done:
                _, reward, _, done, _ = sim.step(rule(sim))
                total += reward
        return total

    assert run(matching) > run(nearest_idle) > run(stay)


# -- JAX against Python --------------------------------------------------------


def replay(preset, policy, seed, max_steps=120):
    sim = FleetSim(preset, max_steps=max_steps)
    env = FleetEnv(preset, max_steps=max_steps)
    obs = sim.reset(seed)
    state, ts = env.reset(jax.random.key(0))
    np.testing.assert_array_equal(ts.observation, obs)
    step = jax.jit(env.step)
    rng = np.random.default_rng(seed)
    for t in range(max_steps):
        actions = policy(sim, rng)
        exo = sim.draw()
        obs, reward, _, truncated, info = sim.step(actions, exo)
        state, ts = step(state, jnp.asarray(actions), env.exogenous(exo.origin, exo.dest))
        np.testing.assert_allclose(ts.observation, obs, rtol=0, atol=1e-6, err_msg=f"step {t}")
        assert float(ts.reward) == pytest.approx(reward, abs=1e-9)
        assert bool(ts.truncated) == truncated
        for name, value in info.items():
            assert int(ts.info[name]) == value, name


@pytest.mark.usefixtures("x64")
@pytest.mark.parametrize("preset", PRESETS)
def test_jax_matches_python_step_for_step(preset):
    n = FleetEnv(preset).n_actions
    for seed in range(2):
        replay(preset, lambda sim, rng: rng.integers(n, size=sim.config.n_vehicles), seed)
        replay(preset, lambda sim, rng: matching(sim), seed)
        replay(preset, lambda sim, rng: rebalance(sim), seed)


@pytest.mark.usefixtures("x64")
@pytest.mark.parametrize("preset", PRESETS)
def test_jax_heuristics_match_python(preset):
    env = FleetEnv(preset)
    for policy, reference in [
        (env.heuristic(), rebalance),
        (Matching(env), matching),
        (NearestIdle(env), nearest_idle),
        (Rebalance(env), rebalance),
        (Stay(env), stay),
    ]:
        sim = FleetSim(preset)
        sim.reset(3)
        state, _ = env.reset(jax.random.key(0))
        act, step = jax.jit(policy.act), jax.jit(env.step)
        carry = policy.init()
        # Drive with nearest idle so both see plenty of busy and idle vehicles.
        for t in range(120):
            expected = reference(sim)
            carry, actions = act(carry, state)
            np.testing.assert_array_equal(actions, expected, err_msg=f"step {t}")
            drive = nearest_idle(sim)
            exo = sim.draw()
            sim.step(drive, exo)
            state, _ = step(state, jnp.asarray(drive), env.exogenous(exo.origin, exo.dest))


def test_draws_match_python_in_distribution():
    env = FleetEnv("commute")
    keys = jax.random.split(jax.random.key(0), 4000)
    draws = jax.jit(jax.vmap(env.draw))(keys)
    count = np.asarray(draws.count)
    assert count.mean() == pytest.approx(env.config.request_rate, rel=0.05)
    used = np.arange(env.arrival_slots)[None, :] < count[:, None]
    x = np.asarray(draws.origin)[used] % env.config.grid
    sim = FleetSim("commute", rng=np.random.default_rng(0))
    px = np.concatenate([sim.draw().origin for _ in range(4000)]) % env.config.grid
    assert x.mean() == pytest.approx(px.mean(), abs=0.1)  # western origins
    assert (np.asarray(draws.origin)[used] != np.asarray(draws.dest)[used]).all()


def test_rollout_and_mappo_run():
    from transit_bench.algos.ppo import PPOConfig, train

    env = make("fleet/grid-v0")
    rewards, _ = jax.jit(make_rollout(env, RandomPolicy(env), 4, 20))(jax.random.key(0))
    assert rewards.shape == (20, 4)
    config = PPOConfig(n_envs=8, n_steps=8, batch_size=64, n_epochs=1, algo="mappo")
    params, _ = train("fleet/grid-v0", 128, config, verbose=False)
    assert params is not None
