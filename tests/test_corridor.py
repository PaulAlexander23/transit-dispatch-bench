"""The corridor env: the coupling in Python, JAX against it, the rules."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from transit_bench import make
from transit_bench.core.policy import RandomPolicy, make_rollout
from transit_bench.envs.corridor import PRESETS, CorridorConfig, CorridorEnv
from transit_bench.envs.corridor.heuristic import BusAware, BusBlind, NoTermination
from transit_bench.envs.corridor.sim import (
    CorridorSim,
    bus_aware,
    bus_blind,
    no_termination,
)
from transit_bench.envs.train.infra import EAST
from transit_bench.envs.train.infra import PRESETS as TRAIN_PRESETS
from transit_bench.envs.train.sim import TERMINATE, WAIT


@pytest.fixture
def x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


# No bus demand and no blockages: only what the test does costs anything.
QUIET = replace(
    CorridorConfig(),
    arrival_rate=0.0,
    travel_cv=0.0,
    train=replace(TRAIN_PRESETS["single"], terminate_penalty=5.0),
)


def test_the_bus_route_runs_out_and_back_along_the_line():
    c = CorridorConfig()
    n = c.train.n_stations
    assert c.n_stops == 2 * (n - 1)
    stop_of = c.stop_of()
    assert list(stop_of[EAST]) == list(range(n))
    assert list(stop_of[1 - EAST]) == [0, 7, 6, 5, 4]  # the ends are one stop each
    bus = c.bus_config()
    assert bus.travel_time == c.road_seconds == 1.5 * 2 * 90
    assert bus.step_seconds == c.train.step_seconds


def test_the_config_needs_terminate_and_no_bridging_buses():
    with pytest.raises(ValueError, match="TERMINATE"):
        CorridorConfig(train=TRAIN_PRESETS["single"])
    with pytest.raises(ValueError, match="bridge_buses"):
        CorridorConfig(train=TRAIN_PRESETS["bridge"])


def first_train(sim, direction):
    infra = sim.train.infra
    return min(
        range(sim.n_trains), key=lambda i: (infra.direction[i] != direction, infra.sched[i][0])
    )


@pytest.mark.parametrize("direction", [0, 1])
def test_a_cancelled_trains_riders_wait_for_the_bus_to_their_calls(direction):
    sim = CorridorSim(QUIET)
    sim.reset(0)
    i = first_train(sim, direction)
    hold = [WAIT] * sim.n_trains + [1] * QUIET.n_buses
    while not sim.train.can_terminate(i):
        sim.step(hold)
    actions = list(hold)
    actions[i] = TERMINATE

    def by_destination():  # bus passengers waiting or on board, per destination stop
        return np.array(sim.bus.waiting).sum(axis=0) + np.array(sim.bus.onboard).sum(axis=0)

    start = by_destination()
    _, _, _, _, info = sim.step(actions)
    n, r = QUIET.train.n_stations, QUIET.riders_per_call
    assert info["terminated"] == 1 and info["transferred"] == r * (n - 1)
    origin = 0 if direction == EAST else n - 1
    expected = np.zeros(QUIET.n_stops, int)
    for s in range(n):
        if s != origin:
            expected[QUIET.stop_of()[direction][s]] = r
    np.testing.assert_array_equal(by_destination() - start, expected)
    # Not yet boarded: waiting at the origin's stop.
    waiting = np.array(sim.bus.waiting)
    assert waiting.sum() == waiting[QUIET.stop_of()[direction][origin]].sum()


def test_transferred_riders_cost_nothing_while_their_credit_runs():
    """With no other passengers, a transferred rider's bus time and its
    credit cancel tick for tick until the credit ends or it arrives."""
    sim = CorridorSim(QUIET)
    sim.reset(0)
    i = first_train(sim, EAST)
    hold = [WAIT] * sim.n_trains + [1] * QUIET.n_buses
    while not sim.train.can_terminate(i):
        sim.step(hold)
    actions = list(hold)
    actions[i] = TERMINATE
    _, _, _, _, info = sim.step(actions)
    # Hold everything else: no other train moves, so no rail cost but the waits.
    for _ in range(3):
        _, _, _, _, info = sim.step(hold)
        assert info["bus_cost"] == pytest.approx(info["credit"])
        assert info["bus_cost"] > 0


def test_a_late_train_charges_each_rider_on_board():
    sim = CorridorSim(QUIET)
    sim.reset(0)
    i = first_train(sim, EAST)
    hold = [WAIT] * sim.n_trains + [1] * QUIET.n_buses
    while not sim.train.ready(i):
        sim.step(hold)
    _, _, _, _, info = sim.step(hold)  # due, and held
    dt, r, calls = QUIET.train.step_seconds, QUIET.riders_per_call, QUIET.train.n_stations - 1
    late = sum(sim.train.ready(j) for j in range(sim.n_trains))  # all due trains wait
    assert info["rail_cost"] == pytest.approx(late * r * calls * dt / 60)
    assert info["rail_cost"] > 0


def test_the_transfer_cost_is_the_wait_plus_the_ride_less_the_timetable():
    sim = CorridorSim(QUIET)
    sim.reset(0)
    i = first_train(sim, EAST)
    for b in range(QUIET.n_buses):  # every bus just past the origin's stop
        sim.bus.stop[b] = 1
    road, infra = QUIET.road_seconds, sim.train.infra
    assert sim.bus_wait(0) == pytest.approx((QUIET.n_stops - 1) * road)
    calls = [q for q in range(1, int(infra.route_len[i])) if infra.dwell[i][q] > 0]
    extra = [
        sim.bus_wait(0) + road * int(infra.station_of[infra.route[i][q]]) - infra.sched[i][q]
        for q in calls
    ]
    assert sim.transfer_cost(i) == pytest.approx(np.mean(extra))
    sim.bus.stop[0] = 0  # one bus at the origin: no wait, and a bus beats the timetable
    assert sim.transfer_cost(i) == 0.0


def test_terminating_when_stuck_beats_never_terminating():
    def run(rule, preset):
        total = 0.0
        for seed in range(6):
            sim = CorridorSim(preset)
            sim.reset(seed)
            for _ in range(sim.max_steps):
                _, reward, _, _, _ = sim.step(rule(sim))
                total += reward
        return total

    for preset in PRESETS.values():
        assert run(bus_aware, preset) > run(no_termination, preset)


# -- JAX against Python --------------------------------------------------------


def replay(preset, policy, seed, max_steps=180):
    sim = CorridorSim(preset, max_steps=max_steps)
    env = CorridorEnv(preset, max_steps=max_steps)
    obs = sim.reset(seed)
    state, ts = env.reset(jax.random.key(0), sim.train.trip)
    np.testing.assert_allclose(ts.observation, obs, rtol=0, atol=1e-6)
    step = jax.jit(env.step)
    rng = np.random.default_rng(seed)
    for t in range(max_steps):
        actions = policy(sim, rng)
        exo = sim.draw()
        obs, reward, terminated, truncated, info = sim.step(actions, exo)
        state, ts = step(state, jnp.asarray(actions), env.exogenous(exo))
        np.testing.assert_allclose(ts.observation, obs, rtol=0, atol=1e-6, err_msg=f"step {t}")
        assert float(ts.reward) == pytest.approx(reward, rel=1e-9, abs=1e-9)
        assert bool(ts.terminated) == terminated and bool(ts.truncated) == truncated
        for name, value in info.items():
            assert float(ts.info[name]) == pytest.approx(value, rel=1e-9, abs=1e-9), name


@pytest.mark.usefixtures("x64")
@pytest.mark.parametrize("preset", PRESETS)
def test_jax_matches_python_step_for_step(preset):
    n = CorridorEnv(preset).n_agents
    for seed in range(2):
        # Mostly GO, so trains run, block and deadlock; some TERMINATE.
        replay(preset, lambda sim, rng: rng.choice(3, size=n, p=[0.8, 0.1, 0.1]), seed)
        replay(preset, lambda sim, rng: bus_aware(sim), seed)


@pytest.mark.usefixtures("x64")
@pytest.mark.parametrize("preset", PRESETS)
def test_jax_heuristics_match_python(preset):
    env = CorridorEnv(preset)
    for policy, reference in [
        (env.heuristic(), bus_aware),
        (BusAware(env), bus_aware),
        (BusBlind(env), bus_blind),
        (NoTermination(env), no_termination),
    ]:
        sim = CorridorSim(preset)
        sim.reset(3)
        state, _ = env.reset(jax.random.key(0), sim.train.trip)
        act, step = jax.jit(policy.act), jax.jit(env.step)
        carry = policy.init()
        for t in range(env.max_steps):
            expected = reference(sim)
            carry, actions = act(carry, state)
            np.testing.assert_array_equal(actions, expected, err_msg=f"step {t}")
            exo = sim.draw()
            sim.step(expected, exo)
            state, _ = step(state, jnp.asarray(expected), env.exogenous(exo))


def test_active_agents_are_the_deciding_trains_and_buses():
    env = CorridorEnv()
    state, _ = env.reset(jax.random.key(0))
    active = env.active_agents(state)
    assert active.shape == (env.n_agents,)
    assert active[env.n_trains :].all()  # every bus starts at a stop with nobody about


def test_rollout_and_mappo_run():
    from transit_bench.algos.ppo import PPOConfig, train

    env = make("corridor/corridor-v0")
    assert env.observation_size == env.n_agents * env.block
    rewards, _ = jax.jit(make_rollout(env, RandomPolicy(env), 4, 20))(jax.random.key(0))
    assert rewards.shape == (20, 4)
    config = PPOConfig(n_envs=8, n_steps=8, batch_size=64, n_epochs=1, algo="mappo")
    params, _ = train("corridor/corridor-v0", 128, config, verbose=False)
    assert params is not None
