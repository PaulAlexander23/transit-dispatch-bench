"""The runway env: the Python reference, JAX against it, the rules and the bound."""

import itertools
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from transit_bench import make
from transit_bench.core.disruption import DisruptionConfig
from transit_bench.core.policy import make_rollout
from transit_bench.envs.runway import PRESETS, RunwayConfig, RunwayEnv
from transit_bench.envs.runway.bound import Scenario, cost_of, run, scenarios, solve
from transit_bench.envs.runway.heuristic import (
    ConstrainedPositionShifting,
    FirstComeFirstServed,
    Wait,
)
from transit_bench.envs.runway.sim import (
    LAND,
    WAIT,
    Exogenous,
    RunwaySim,
    cps,
    fcfs,
    overtakes,
    shifts,
    wait,
)

HEAVY, LARGE, SMALL = 0, 1, 2


@pytest.fixture
def x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def none():
    return Exogenous(np.zeros(0, int))


def arrive(*classes):
    return Exogenous(np.array(classes, int))


# -- the Python reference ------------------------------------------------------


def test_separation_rounds_up_to_ticks():
    sep = RunwayConfig().separation_ticks
    assert sep[HEAVY][SMALL] == 20 and sep[SMALL][HEAVY] == 6 and sep[HEAVY][LARGE] == 16


def test_an_aircraft_lands_at_its_eta_and_the_next_waits_for_separation():
    sim = RunwaySim(RunwayConfig())
    sim.step([WAIT], arrive(HEAVY, SMALL))  # announced in tick 0, E = 1
    assert sim.eta[:2] == [1, 1] and sim.seq[:2] == [0, 1]
    _, reward, _, _, info = sim.step([LAND + 0], none())
    assert info["landed"] == 1 and info["delay_ticks"] == 0
    assert reward == pytest.approx(-10 / 60)  # the small one waited a tick
    for _ in range(19):  # heavy then small: 20 ticks
        _, _, _, _, info = sim.step([LAND + 1], none())
        assert info["landed"] == 0
    _, _, _, _, info = sim.step([LAND + 1], none())
    assert info["landed"] == 1 and info["delay_ticks"] == 20


def test_an_aircraft_cannot_land_before_its_eta():
    sim = RunwaySim(replace(RunwayConfig(), notice=3))
    sim.step([WAIT], arrive(LARGE))
    for _ in range(3):
        _, reward, _, _, info = sim.step([LAND + 0], none())
        assert info["landed"] == 0 and reward == 0.0  # not delayed before its ETA
    _, _, _, _, info = sim.step([LAND + 0], none())
    assert info["landed"] == 1 and info["delay_ticks"] == 0


def test_landing_out_of_order_counts_overtakes():
    sim = RunwaySim(RunwayConfig())
    sim.step([WAIT], arrive(HEAVY, LARGE, SMALL))
    sim.step([LAND + 2], none())
    assert sim.overtaken[:2] == [1, 1]


def test_a_closed_runway_lands_nothing():
    closed = DisruptionConfig(mode="scripted", scripted=((0, 0.0, 50.0),))
    sim = RunwaySim(replace(RunwayConfig(), disruption=closed))
    sim.step([WAIT], arrive(LARGE))  # closed from the end of tick 0
    for _ in range(4):
        _, _, _, _, info = sim.step([LAND + 0], none())
        assert info["landed"] == 0 and info["closed"] == 1
    _, _, _, _, info = sim.step([LAND + 0], none())
    assert info["landed"] == 1 and info["delay_ticks"] == 4


def test_a_full_buffer_rejects_arrivals():
    c = replace(RunwayConfig(), max_aircraft=2)
    sim = RunwaySim(c)
    _, reward, _, _, info = sim.step([WAIT], arrive(LARGE, LARGE, LARGE))
    assert info["rejected"] == 1 and info["announced"] == 2
    assert reward == pytest.approx(-c.reject_penalty)


def test_shift_orders_and_overtakes():
    assert shifts(3, 0) == [(0, 1, 2)]
    assert shifts(3, 1) == [(0, 1, 2), (0, 2, 1), (1, 0, 2)]
    assert all(max(abs(i - n) for n, i in enumerate(o)) <= 2 for o in shifts(5, 2))
    assert overtakes((2, 0, 1)) == [1, 1, 0]


def test_cps_lands_the_small_aircraft_before_the_heavy():
    # A large lands at tick 1; a heavy (slot 0) and a small (slot 1) are
    # due at tick 2, and by tick 15 both may land. FCFS lands the heavy now
    # and the small 20 ticks later (delays 13 + 33); small first, the heavy
    # follows 6 ticks later (13 + 19).
    sim = RunwaySim(RunwayConfig())
    sim.step([WAIT], arrive(LARGE))
    sim.step([LAND + 0], arrive(HEAVY, SMALL))
    for _ in range(13):
        sim.step([WAIT], none())
    assert sim.t == 15 and sim.can_land(0) and sim.can_land(1)
    assert fcfs(sim) == [LAND + 0]
    assert cps(sim) == [LAND + 1]
    assert cps(sim, k=0) == fcfs(sim)


def test_cps_with_k_0_is_fcfs():
    for seed in range(3):
        sim = RunwaySim("peak")
        sim.reset(seed)
        done = False
        while not done:
            assert cps(sim, k=0) == fcfs(sim)
            _, _, _, done, _ = sim.step(cps(sim))  # overtakes happen


def total_return(preset, rule, seeds=8):
    total = 0.0
    for seed in range(seeds):
        sim = RunwaySim(preset)
        sim.reset(seed)
        done = False
        while not done:
            _, reward, _, done, _ = sim.step(rule(sim))
            total += reward
    return total


@pytest.mark.parametrize("preset", ["mixed", "peak"])
def test_cps_beats_fcfs_beats_waiting(preset):
    assert total_return(preset, cps) > total_return(preset, fcfs) > total_return(preset, wait)


# -- JAX against Python --------------------------------------------------------


def replay(preset, policy, seed, max_steps=360):
    sim = RunwaySim(preset, max_steps=max_steps)
    env = RunwayEnv(preset, max_steps=max_steps)
    obs = sim.reset(seed)
    state, ts = env.reset(jax.random.key(0))
    np.testing.assert_array_equal(ts.observation, obs)
    step = jax.jit(env.step)
    rng = np.random.default_rng(seed)
    for t in range(max_steps):
        actions = policy(sim, rng)
        exo = sim.draw()
        obs, reward, _, truncated, info = sim.step(actions, exo)
        state, ts = step(state, jnp.asarray(actions), env.exogenous(exo))
        np.testing.assert_allclose(ts.observation, obs, rtol=0, atol=1e-6, err_msg=f"step {t}")
        assert float(ts.reward) == pytest.approx(reward, abs=1e-9)
        assert bool(ts.truncated) == truncated
        for name, value in info.items():
            assert int(ts.info[name]) == value, name


def random_landing(sim, rng):
    """Mostly tries to land an occupied slot, so landings happen."""
    occupied = [r for r in range(sim.config.max_aircraft) if sim.valid[r]]
    if occupied and rng.random() < 0.7:
        return [LAND + int(rng.choice(occupied))]
    return [int(rng.integers(sim.config.n_actions))]


@pytest.mark.usefixtures("x64")
@pytest.mark.parametrize("preset", PRESETS)
def test_jax_matches_python_step_for_step(preset):
    for seed in range(2):
        replay(preset, random_landing, seed)
        replay(preset, lambda sim, rng: cps(sim), seed)


@pytest.mark.usefixtures("x64")
@pytest.mark.parametrize("preset", PRESETS)
def test_jax_rules_match_python(preset):
    env = RunwayEnv(preset)
    for policy, reference in [
        (env.heuristic(), cps),
        (ConstrainedPositionShifting(env, k=1, window=4), lambda s: cps(s, k=1, window=4)),
        (FirstComeFirstServed(env), fcfs),
        (Wait(env), wait),
    ]:
        sim = RunwaySim(preset)
        sim.reset(3)
        state, _ = env.reset(jax.random.key(0))
        act, step = jax.jit(policy.act), jax.jit(env.step)
        carry = policy.init()
        rng = np.random.default_rng(1)
        # Drive with a mix of the rule and random landings, so the rules
        # see queues, overtakes and all buffer states.
        for t in range(360):
            expected = reference(sim)
            carry, actions = act(carry, state)
            np.testing.assert_array_equal(actions, expected, err_msg=f"step {t}")
            drive = expected if rng.random() < 0.7 else random_landing(sim, rng)
            exo = sim.draw()
            sim.step(drive, exo)
            state, _ = step(state, jnp.asarray(drive), env.exogenous(exo))


def test_draws_match_python_in_distribution():
    env = RunwayEnv("mixed")
    keys = jax.random.split(jax.random.key(0), 20000)
    draws = jax.jit(jax.vmap(lambda k: env.draw(k, 0)))(keys)
    count = np.asarray(draws.count)
    assert count.mean() == pytest.approx(36 / 360, rel=0.05)
    used = np.arange(env.arrival_slots)[None, :] < count[:, None]
    share = np.bincount(np.asarray(draws.cls)[used], minlength=3) / used.sum()
    np.testing.assert_allclose(share, env.config.mix, atol=0.03)


def test_peak_profile_rates():
    rate = PRESETS["peak"].rates(360) * 360  # per hour
    assert rate[0] == pytest.approx(54) and rate[179] == pytest.approx(54)
    assert rate[180] == pytest.approx(18)


def test_rollout_and_ppo_run():
    from transit_bench.algos.ppo import PPOConfig, train

    env = make("runway/mixed-v0")
    policy = env.heuristic()
    rewards, info = jax.jit(make_rollout(env, policy, 4, env.max_steps))(jax.random.key(0))
    assert rewards.shape == (env.max_steps, 4) and int(info["landed"].sum()) > 0
    config = PPOConfig(n_envs=8, n_steps=8, batch_size=64, n_epochs=1, algo="ppo")
    params, _ = train("runway/mixed-v0", 128, config, verbose=False)
    assert params is not None


# -- the hindsight bound -------------------------------------------------------


def brute_force(config, scenario, horizon):
    """The least total delay over every order of the due aircraft, each
    landing as early as allowed."""
    sep = config.separation_ticks
    due = [a for a in range(len(scenario.eta)) if scenario.eta[a] < horizon]
    best = None
    for order in itertools.permutations(due):
        last, at, total = None, 0, 0
        for a in order:
            t = (
                int(scenario.eta[a])
                if last is None
                else max(scenario.eta[a], at + sep[last][scenario.cls[a]])
            )
            while t < horizon and scenario.closed[t]:
                t += 1
            total += min(t, horizon) - scenario.eta[a]
            last, at = scenario.cls[a], t
        best = total if best is None else min(best, total)
    return best * config.step_seconds / 60


def test_the_bound_matches_brute_force_on_small_instances():
    config = RunwayConfig()
    rng = np.random.default_rng(0)
    for _ in range(40):
        n = int(rng.integers(1, 7))
        horizon = 60
        eta = np.sort(rng.integers(1, 30, n))
        closed = np.zeros(horizon, bool)
        if rng.random() < 0.5:
            s = int(rng.integers(0, 40))
            closed[s : s + int(rng.integers(1, 15))] = True
        sc = Scenario(rng.integers(0, 3, n), eta, closed)
        bound = solve(config, sc, horizon)
        assert bound.cost == pytest.approx(brute_force(config, sc, horizon))
        assert cost_of(config, sc, bound.landing, horizon) == pytest.approx(bound.cost)


def check_schedule(config, scenario, landing, horizon):
    """The bound's schedule is one the env allows."""
    sep = config.separation_ticks
    due = np.flatnonzero(scenario.eta < horizon)
    order = due[np.argsort(landing[due], kind="stable")]
    for n, a in enumerate(order):
        t = landing[a]
        assert t >= scenario.eta[a]
        if t < horizon:
            assert not scenario.closed[t]
        if n:
            b = order[n - 1]
            assert t - landing[b] >= sep[scenario.cls[b]][scenario.cls[a]]


@pytest.mark.parametrize("preset", PRESETS)
def test_the_bound_is_below_every_episode(preset):
    env = RunwayEnv(preset)
    keys = jax.random.split(jax.random.key(1), 6)
    found = scenarios(env, keys)
    for policy in (FirstComeFirstServed(env), env.heuristic()):
        episode = jax.jit(jax.vmap(lambda k, policy=policy: run(env, policy, k)))
        delay, _, rejected = episode(keys)
        for b, sc in enumerate(found):
            assert int(rejected[b]) == 0
            bound = solve(env.config, sc, env.max_steps)
            check_schedule(env.config, sc, bound.landing, env.max_steps)
            assert bound.cost <= float(delay[b]) + 1e-4
