"""The bus env: the Python reference's behaviour, and JAX against it."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from transit_bench import make
from transit_bench.core.policy import RandomPolicy, make_rollout
from transit_bench.envs.bus import PRESETS, BusConfig, BusEnv
from transit_bench.envs.bus.sim import (
    GO,
    HOLD,
    BusSim,
    Exogenous,
    arrival_rates,
    headway_holding,
)


@pytest.fixture
def x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def quiet(**changes):
    """No arrivals and no noise: every passenger is placed by the test."""
    return replace(BusConfig(), arrival_rate=0.0, target_headway=100.0, **changes)


def no_draw(config):
    return Exogenous(np.zeros((config.n_stops, config.n_stops), int), np.ones(config.n_buses))


# -- the Python reference ------------------------------------------------------


def test_bus_runs_round_the_loop_in_nominal_time():
    c = quiet(n_buses=1, n_stops=4)
    sim = BusSim(c)
    ticks = int(c.travel_time / c.step_seconds)
    stops = []
    for _ in range(4 * (ticks + 1)):
        sim.step([GO], no_draw(c))
        if sim.at_stop[0]:
            stops.append(sim.stop[0])
    # Each edge takes `ticks` ticks, plus one tick deciding at each stop.
    assert stops == [1, 2, 3, 0]


def test_boarding_and_alighting_take_their_seconds():
    c = quiet(n_buses=1, n_stops=4)
    sim = BusSim(c)
    sim.waiting[0][2] = 7
    # 10 s ticks, 2 s per boarding: 5 board in the first tick, 2 in the next.
    _, _, _, info = sim.step([GO], no_draw(c))
    assert info["boarded"] == 5 and sim.at_stop[0]
    _, _, _, info = sim.step([GO], no_draw(c))
    assert info["boarded"] == 2 and sum(sim.onboard[0]) == 7
    assert sim.deciding(0)
    sim.step([GO], no_draw(c))
    assert not sim.at_stop[0]


def test_boarding_takes_the_nearest_destinations_first():
    c = quiet(n_buses=1, n_stops=5)
    sim = BusSim(c)
    sim.waiting[3][1] = 4  # 3 stops ahead of stop 3
    sim.waiting[3][4] = 4  # 1 stop ahead
    sim.stop[0] = 3
    sim.step([GO], no_draw(c))
    assert sim.onboard[0][4] == 4 and sim.onboard[0][1] == 1


def test_hold_keeps_a_deciding_bus_and_counts():
    c = quiet(n_buses=1)
    sim = BusSim(c)
    for _ in range(3):
        _, _, _, info = sim.step([HOLD], no_draw(c))
        assert info["held"] == 1 and sim.at_stop[0]
    assert sim.hold[0] == 3
    sim.step([GO], no_draw(c))
    assert not sim.at_stop[0] and sim.hold[0] == 0


def test_closed_edge_holds_buses_at_the_stop_before_it():
    c = quiet(n_buses=1)
    sim = BusSim(c)
    sim.edge_open[0] = False
    for _ in range(3):
        sim.step([GO], no_draw(c))
    assert sim.at_stop[0] and sim.stop[0] == 0
    sim.edge_open[0] = True
    sim.step([GO], no_draw(c))
    assert not sim.at_stop[0]


@pytest.mark.usefixtures("x64")
def test_a_broken_down_bus_freezes_until_repaired():
    from transit_bench.core.disruption import DisruptionConfig

    fail = DisruptionConfig(mode="markov", rate_per_hour=1.0, mean_seconds=60.0)
    c = quiet(n_buses=2, breakdown=fail)
    env, sim = BusEnv(c), BusSim(c)
    state, _ = env.reset(jax.random.key(0))
    arrivals = np.zeros((c.n_stops, c.n_stops), int)
    # Bus 0 breaks down at the end of the first tick, for 60 s; bus 1 never.
    breaks = Exogenous(arrivals, np.ones(2), None, None, np.array([0.0, 1.0]), np.full(2, 60.0))
    calm = Exogenous(arrivals, np.ones(2), None, None, np.ones(2), np.zeros(2))
    for t in range(12):
        exo = breaks if t == 0 else calm
        sim.step([HOLD, GO] if t == 0 else [GO, GO], exo)
        state, _ = env.step(state, jnp.array([HOLD, GO] if t == 0 else [GO, GO]), exo)
        if 1 <= t < 6:
            assert sim.broken(0) and not sim.deciding(0)
            assert sim.at_stop[0] and sim.stop[0] == 0  # frozen at its stop
        np.testing.assert_array_equal(state.stop, sim.stop)
        np.testing.assert_array_equal(state.at_stop, sim.at_stop)
    assert not sim.broken(0) and not sim.at_stop[0]  # repaired and gone


def test_a_scripted_road_closure_holds_buses_then_lets_them_go():
    from transit_bench.core.disruption import DisruptionConfig

    d = DisruptionConfig(mode="scripted", scripted=((0, 0.0, 60.0),))
    c = quiet(n_buses=1, disruption=d)
    sim = BusSim(c)
    sim.step([HOLD], no_draw(c))  # the closure takes effect at the end of this tick
    obs = sim.step([GO], no_draw(c))[0]
    assert sim.at_stop[0] and sim.stop[0] == 0 and not sim.edge_open[0]
    # The first stop's fields (from offset HEAD_FIELDS): waiting, on board, since, down.
    from transit_bench.envs.bus.sim import HEAD_FIELDS

    assert obs[HEAD_FIELDS + 3] == 1.0
    for _ in range(6):
        sim.step([GO], no_draw(c))
    assert not sim.at_stop[0]  # reopened after 60 s


def test_departure_gaps_are_recorded_at_each_stop():
    c = quiet(n_buses=2, n_stops=4)
    sim = BusSim(c)
    sim.stop = [0, 0]
    _, _, _, info = sim.step([GO, HOLD], no_draw(c))
    assert info["departures"] == 1 and info["gap_sum"] == 0  # the first has no gap
    sim.step([GO, HOLD], no_draw(c))
    _, _, _, info = sim.step([GO, GO], no_draw(c))
    assert info["departures"] == 1 and info["gap_sum"] == 2 * c.step_seconds
    assert info["bunched"] == 1  # 20 s against a 100 s target


def test_reward_is_minus_passenger_minutes():
    c = quiet(n_buses=1, n_stops=4)
    sim = BusSim(c)
    sim.waiting[2][3] = 6
    _, reward, _, info = sim.step([GO], no_draw(c))
    assert info["waiting"] == 6
    assert reward == pytest.approx(-6 * c.step_seconds / 60 / c.n_stops)


def test_target_headway_is_the_steady_state():
    c = BusConfig(n_stops=10, n_buses=3, travel_time=60, arrival_rate=1 / 30)
    # H = Sτ / (B − Sλ(α+β)) = 600 / (3 − 10/30 × 3) = 300 s.
    assert c.headway == pytest.approx(300.0)
    with pytest.raises(ValueError, match="demand"):
        _ = replace(c, arrival_rate=1 / 5).headway


@pytest.mark.parametrize("preset", PRESETS)
def test_holding_beats_always_go_on_headway_regularity(preset):
    """The heuristic's point: even departures. Over 3 seeds its headway CV
    is well below always-go's, and it never bunches."""

    def run(policy):
        gaps, bunched, departures = [], 0, 0
        for seed in range(3):
            sim = BusSim(preset)
            sim.reset(seed)
            done = False
            while not done:
                _, _, done, info = sim.step(policy(sim))
                bunched += info["bunched"]
                departures += info["departures"]
                if info["departures"]:
                    gaps.append((info["gap_sum"], info["gap_sq_sum"], info["departures"]))
        s, sq, n = map(sum, zip(*gaps, strict=True))
        mean = s / n
        return np.sqrt(sq / n - mean**2) / mean, bunched / departures

    cv_go, _ = run(lambda sim: [GO] * sim.config.n_buses)
    cv_hold, bunched = run(headway_holding)
    assert cv_hold < 0.8 * cv_go
    assert bunched < 0.01


# -- JAX against Python --------------------------------------------------------


def replay(preset, policy, seed, max_steps=120):
    """Run Python and JAX side by side on Python's draws; check every step."""
    sim = BusSim(preset, max_steps=max_steps)
    env = BusEnv(preset, max_steps=max_steps)
    obs = sim.reset(seed)
    state, ts = env.reset(jax.random.key(0))
    np.testing.assert_array_equal(ts.observation, obs)
    step = jax.jit(env.step)
    rng = np.random.default_rng(seed)
    for t in range(max_steps):
        actions = policy(sim, rng)
        exo = sim.draw()
        obs, reward, truncated, info = sim.step(actions, exo)
        state, ts = step(state, jnp.asarray(actions), exo)
        np.testing.assert_array_equal(ts.observation, obs, err_msg=f"step {t}")
        assert float(ts.reward) == pytest.approx(reward, rel=1e-12, abs=1e-12)
        assert bool(ts.truncated) == truncated
        for name, value in info.items():
            assert float(ts.info[name]) == pytest.approx(value, rel=1e-12), name
    return state


@pytest.mark.usefixtures("x64")
@pytest.mark.parametrize("preset", PRESETS)
def test_jax_matches_python_step_for_step(preset):
    for seed in range(2):
        replay(preset, lambda sim, rng: rng.integers(2, size=sim.config.n_buses), seed)
        replay(preset, lambda sim, rng: headway_holding(sim), seed)


@pytest.mark.usefixtures("x64")
@pytest.mark.parametrize("preset", ["noisy", "full"])
def test_jax_heuristic_matches_python(preset):
    sim = BusSim(preset)
    env = BusEnv(preset)
    sim.reset(0)
    state, _ = env.reset(jax.random.key(0))
    policy = env.heuristic()
    act, step = jax.jit(policy.act), jax.jit(env.step)
    carry = policy.init()
    for t in range(env.max_steps):
        expected = headway_holding(sim)
        carry, actions = act(carry, state)
        np.testing.assert_array_equal(actions, expected, err_msg=f"step {t}")
        exo = sim.draw()
        sim.step(expected, exo)
        state, _ = step(state, actions, exo)


@pytest.mark.usefixtures("x64")
def test_arrivals_match_python_in_distribution():
    env = BusEnv("peak")
    draws, fraction = 20_000, 0.4
    keys = jax.random.split(jax.random.key(0), draws)
    exo = jax.jit(jax.vmap(lambda k: env.draw(k, fraction)))(keys)
    counts = np.asarray(exo.arrivals)
    expected = arrival_rates(env.config, fraction)
    error = np.sqrt(expected / draws) + 1e-12
    assert np.all(np.abs(counts.mean(axis=0) - expected) < 5 * error)
    np.testing.assert_allclose(counts.var(axis=0), expected, rtol=0.15, atol=1e-3)
    assert counts[:, np.arange(10), np.arange(10)].sum() == 0  # nobody rides to their own stop
    noise = np.asarray(exo.noise)
    assert noise.mean() == pytest.approx(1.0, abs=0.01)
    assert noise.std() == pytest.approx(env.config.travel_cv, rel=0.05)


def test_heuristic_beats_random_in_jax():
    env = make("bus/noisy-v0")
    returns = {}
    for name, policy in [("random", RandomPolicy(env)), ("holding", env.heuristic())]:
        rewards, info = jax.jit(make_rollout(env, policy, 32, env.max_steps))(jax.random.key(0))
        returns[name] = float(np.asarray(rewards).sum(axis=0).mean())
    assert returns["holding"] > returns["random"]


def test_bus_env_trains_with_mappo():
    from transit_bench.algos.ppo import PPOConfig, make_train

    env = BusEnv("loop", max_steps=20)
    config = PPOConfig(n_envs=4, n_steps=10, batch_size=20, net_arch=(16,), algo="mappo")
    init, update = make_train(env, config)
    runner, metrics = jax.jit(update)(init(jax.random.key(0)))
    assert np.isfinite(float(metrics["value_loss"]))
