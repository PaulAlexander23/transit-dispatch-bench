"""The train env: infrastructure, the Python reference, and JAX against it."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from transit_bench import make
from transit_bench.core.policy import RandomPolicy, make_rollout
from transit_bench.envs.train import PRESETS, TrainConfig, TrainEnv
from transit_bench.envs.train.heuristic import FirstComeFirstServed
from transit_bench.envs.train.infra import EAST, WEST, build
from transit_bench.envs.train.sim import (
    GO,
    WAIT,
    Exogenous,
    TrainSim,
    first_come_first_served,
    section_reservation,
)


@pytest.fixture
def x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def calm(config, n_trains):
    """No noise, no incidents and no disruptions."""
    n_blocks = build(config).n_blocks
    return Exogenous(np.ones(n_trains), np.zeros(n_trains), np.ones(n_blocks), np.zeros(n_blocks))


# -- infrastructure ------------------------------------------------------------


def test_routes_visit_every_station_and_section_in_order():
    infra = build(PRESETS["line"])
    for i in range(len(infra.route)):
        stations = [s for s in infra.station_of[infra.route[i]] if s >= 0]
        expected = list(range(5)) if infra.direction[i] == EAST else list(range(4, -1, -1))
        assert stations == expected
    # Double track: the two directions share no block.
    east = set(infra.route[infra.direction == EAST].ravel())
    west = set(infra.route[infra.direction == WEST].ravel())
    assert not east & west


def test_single_track_shares_sections_but_not_platforms():
    infra = build(PRESETS["single"])
    east = set(infra.route[infra.direction == EAST].ravel())
    west = set(infra.route[infra.direction == WEST].ravel())
    shared = east & west
    assert shared and all(infra.section_of[b] >= 0 for b in shared)


def test_express_trains_overtake_at_loops():
    infra = build(PRESETS["passing"])
    stopping = np.flatnonzero(~infra.express & (infra.direction == EAST))[0]
    express = np.flatnonzero(infra.express & (infra.direction == EAST))[0]
    middle_s = set(infra.route[stopping]) - set(infra.route[express])
    # The stopping train uses loop platforms the express does not.
    assert middle_s and all(infra.station_of[b] > 0 for b in middle_s)
    assert infra.sched[express, -1] - infra.sched[express, 0] < (
        infra.sched[stopping, -1] - infra.sched[stopping, 0]
    )


def test_timetable_includes_slack():
    c = replace(PRESETS["line"], slack=0.0)
    tight, padded = build(c), build(replace(c, slack=0.1))
    run_tight = tight.sched[:, -1] - tight.sched[:, 0]
    run_padded = padded.sched[:, -1] - padded.sched[:, 0]
    np.testing.assert_allclose(run_padded, 1.1 * run_tight)


# -- the Python reference ------------------------------------------------------


def test_a_calm_train_runs_to_time():
    c = replace(PRESETS["line"], trains_per_direction=1, slack=0.0)
    sim = TrainSim(c)
    T = len(sim.pos)
    done = False
    while not done:
        _, _, terminated, truncated, info = sim.step([GO] * T, calm(c, T))
        done = terminated or truncated
    assert terminated and info["finished"] == T
    assert sim.arrival_delay == [0.0] * T


def test_holding_a_train_makes_it_late_and_the_return_is_the_delay():
    c = replace(PRESETS["line"], trains_per_direction=1, slack=0.0)
    sim = TrainSim(c)
    T = len(sim.pos)
    total, done, held = 0.0, False, 0
    while not done:
        # Hold train 0 for its first five decisions.
        actions = [GO] * T
        if sim.deciding(0) and held < 5:
            actions[0] = WAIT
            held += 1
        _, reward, terminated, truncated, _ = sim.step(actions, calm(c, T))
        total += reward
        done = terminated or truncated
    step = c.step_seconds
    assert sim.arrival_delay[0] == pytest.approx(5 * step)
    # The rewards telescope to minus the total arrival delay, in minutes.
    assert total == pytest.approx(-sum(sim.arrival_delay) / 60)


def test_blocking_keeps_one_train_per_block():
    sim = TrainSim("passing")
    sim.reset(0)
    for _ in range(sim.max_steps):
        sim.step(first_come_first_served(sim))
        held = [b for b in sim.occupant if b >= 0]
        assert len(held) == len(set(held))
        for i, p in enumerate(sim.pos):
            if 0 <= p < sim.infra.route_len[i]:
                assert sim.occupant[int(sim.infra.route[i][p])] == i


def test_head_on_trains_on_single_track_deadlock_without_reservation():
    c = replace(
        PRESETS["single"],
        n_stations=2,
        blocks_per_section=2,
        trains_per_direction=1,
        running_cv=0.0,
        incident_probability=0.0,
    )
    for policy, expect in [(first_come_first_served, True), (section_reservation, False)]:
        sim = TrainSim(c)
        deadlocked = False
        for _ in range(60):
            _, _, terminated, _, info = sim.step(policy(sim))
            deadlocked |= info["deadlocked"] == 2
            if terminated:
                break
        assert deadlocked == expect, policy.__name__
        assert terminated != expect


def test_closed_block_stops_trains():
    c = replace(PRESETS["line"], trains_per_direction=1)
    sim = TrainSim(c)
    first = int(sim.infra.route[0][1])
    sim.block_open[first] = False
    T = len(sim.pos)
    for _ in range(30):
        sim.step([GO] * T, calm(c, T))
    assert sim.pos[0] == 0


@pytest.mark.parametrize("preset", PRESETS)
def test_heuristics_beat_random(preset):
    def run(policy):
        total = 0.0
        for seed in range(5):
            sim = TrainSim(preset)
            sim.reset(seed)
            rng = np.random.default_rng(seed)
            done = False
            while not done:
                _, reward, terminated, truncated, _ = sim.step(policy(sim, rng))
                total += reward
                done = terminated or truncated
        return total / 5

    random = run(lambda sim, rng: rng.integers(2, size=len(sim.pos)))
    reserve = run(lambda sim, rng: section_reservation(sim))
    assert reserve > random


# -- JAX against Python --------------------------------------------------------


def replay(preset, policy, seed):
    sim = TrainSim(preset)
    env = TrainEnv(preset)
    obs = sim.reset(seed)
    state, ts = env.reset(jax.random.key(0))
    np.testing.assert_array_equal(ts.observation, obs)
    step = jax.jit(env.step)
    deciding = jax.jit(env.deciding)
    rng = np.random.default_rng(seed)
    for t in range(env.max_steps):
        np.testing.assert_array_equal(
            deciding(state), [sim.deciding(i) for i in range(len(sim.pos))]
        )
        actions = policy(sim, rng)
        exo = sim.draw()
        obs, reward, terminated, truncated, info = sim.step(actions, exo)
        state, ts = step(state, jnp.asarray(actions), exo)
        np.testing.assert_array_equal(ts.observation, obs, err_msg=f"step {t}")
        assert float(ts.reward) == pytest.approx(reward, rel=1e-12, abs=1e-12)
        assert bool(ts.terminated) == terminated and bool(ts.truncated) == truncated
        for name, value in info.items():
            assert float(ts.info[name]) == pytest.approx(value, rel=1e-12), name
        if terminated or truncated:
            break


@pytest.mark.usefixtures("x64")
@pytest.mark.parametrize("preset", PRESETS)
def test_jax_matches_python_step_for_step(preset):
    for seed in range(2):
        replay(preset, lambda sim, rng: rng.integers(2, size=len(sim.pos)), seed)
        replay(preset, lambda sim, rng: first_come_first_served(sim), seed)
        replay(preset, lambda sim, rng: section_reservation(sim), seed)


@pytest.mark.usefixtures("x64")
@pytest.mark.parametrize("preset", PRESETS)
def test_jax_heuristics_match_python(preset):
    env = TrainEnv(preset)
    for policy, reference in [
        (env.heuristic(), section_reservation),
        (FirstComeFirstServed(env), first_come_first_served),
    ]:
        sim = TrainSim(preset)
        sim.reset(3)
        state, _ = env.reset(jax.random.key(0))
        act, step = jax.jit(policy.act), jax.jit(env.step)
        carry = policy.init()
        for t in range(env.max_steps):
            expected = reference(sim)
            carry, actions = act(carry, state)
            np.testing.assert_array_equal(actions, expected, err_msg=f"step {t}")
            exo = sim.draw()
            _, _, terminated, _, _ = sim.step(expected, exo)
            state, _ = step(state, actions, exo)
            if terminated:
                break


def test_deadlock_detection_finds_exactly_the_cycle():
    env = TrainEnv("single")
    state = env.initial_state(jax.random.key(0))
    infra = env.infra
    # Trains 0 and 1 face each other in one section; train 2 queues behind 0.
    east = np.flatnonzero(infra.direction == EAST)
    west = np.flatnonzero(infra.direction == WEST)
    a, b, behind = int(east[0]), int(west[0]), int(east[1])
    k = env.config.blocks_per_section
    pos_a = 1  # first section block eastwards from station 0
    blk_a = int(infra.route[a][pos_a])
    pos_b = int(np.flatnonzero(infra.route[b] == infra.route[a][pos_a + 1])[0])
    assert k == 2
    pos_c = 0  # station 0 platform, waiting for blk_a
    state = state._replace(
        pos=state.pos.at[a].set(pos_a).at[b].set(pos_b).at[behind].set(pos_c),
        occupant=state.occupant.at[blk_a]
        .set(a)
        .at[int(infra.route[b][pos_b])]
        .set(b)
        .at[int(infra.route[behind][pos_c])]
        .set(behind),
        t=jnp.asarray(1000, jnp.int32),
    )
    stuck = np.asarray(env.deadlocked(state))
    assert stuck[a] and stuck[b] and not stuck[behind]
    assert stuck.sum() == 2


@pytest.mark.usefixtures("x64")
def test_draws_match_python_in_distribution():
    env = TrainEnv("line")
    keys = jax.random.split(jax.random.key(0), 20_000)
    exo = jax.jit(jax.vmap(env.draw))(keys)
    noise, incident = np.asarray(exo.noise), np.asarray(exo.incident)
    c = env.config
    assert noise.mean() == pytest.approx(1.0, abs=0.005)
    assert noise.std() == pytest.approx(c.running_cv, rel=0.05)
    hit = incident > 0
    assert hit.mean() == pytest.approx(c.incident_probability, rel=0.1)
    assert incident[hit].mean() == pytest.approx(c.incident_seconds, rel=0.1)


def test_rollout_and_mappo_run():
    from transit_bench.algos.ppo import PPOConfig, make_train

    env = make("train/single-v0")
    rewards, info = jax.jit(make_rollout(env, RandomPolicy(env), 8, env.max_steps))(
        jax.random.key(0)
    )
    assert np.isfinite(np.asarray(rewards)).all()
    config = PPOConfig(n_envs=4, n_steps=8, batch_size=16, net_arch=(16,), algo="mappo")
    init, update = make_train(TrainEnv("single", max_steps=20), config)
    runner, metrics = jax.jit(update)(init(jax.random.key(0)))
    assert np.isfinite(float(metrics["value_loss"]))


def test_config_validation():
    with pytest.raises(ValueError):
        TrainConfig(single_track=True, passing_loops=True)


def test_observation_shows_an_oncoming_rival_for_the_section():
    from transit_bench.envs.train.sim import BLOCK_FIELDS, HEAD_FIELDS, LOOKAHEAD

    c = replace(
        PRESETS["single"],
        n_stations=2,
        trains_per_direction=1,
        running_cv=0.0,
        incident_probability=0.0,
    )
    sim = TrainSim(c)
    sim.step([GO, GO])  # both enter their terminal platforms
    while not (sim.deciding(0) and sim.deciding(1)):
        sim.step([WAIT, WAIT])
    width = HEAD_FIELDS + BLOCK_FIELDS * LOOKAHEAD
    obs = sim.observe().reshape(2, width)
    entering, inside, before, after = obs[:, 9], obs[:, 10], obs[:, 11], obs[:, 12]
    assert list(entering) == [1, 1]
    assert list(inside) == [0, 0]
    assert list(before) == [0, 1]  # train 1 sees train 0 go first
    assert list(after) == [1, 0]
