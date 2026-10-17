"""The disruption process (core.disruption) and its use in the train env."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from transit_bench.core import disruption as dis
from transit_bench.core.disruption import DisruptionConfig


@pytest.fixture
def x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def run_np(config, n_edges, ticks, dt=10.0, seed=0, eligible=None):
    rng = np.random.default_rng(seed)
    eligible = np.ones(n_edges, bool) if eligible is None else eligible
    state = dis.initial_np(n_edges)
    history = []
    for t in range(ticks):
        draw = dis.draw_np(config, rng, n_edges)
        state = dis.step_np(config, state, draw, eligible, t * dt, dt)
        history.append(state)
    return history


def outage_lengths(history, dt):
    """Lengths in seconds of completed outages, from a history of states."""
    down = np.array([s.down for s in history])  # (ticks, edges)
    lengths = []
    for e in range(down.shape[1]):
        run = 0
        for x in down[:, e]:
            if x:
                run += 1
            elif run:
                lengths.append(run * dt)
                run = 0
    return np.array(lengths)


def test_markov_outages_have_the_configured_rate_and_mean():
    c = DisruptionConfig(mode="markov", rate_per_hour=2.0, mean_seconds=600.0)
    dt = 10.0
    history = run_np(c, n_edges=200, ticks=2000, dt=dt)
    lengths = outage_lengths(history, dt)
    # Rounded up to whole ticks, so about dt / 2 longer on average.
    assert lengths.mean() == pytest.approx(600.0 + dt / 2, rel=0.05)
    assert lengths.std() / lengths.mean() == pytest.approx(1.0, abs=0.1)  # exponential
    # Fraction of time down: λm / (1 + λm), with λm = 2/h × 10 min = 1/3.
    down = np.mean([s.down.mean() for s in history[200:]])
    assert down == pytest.approx((1 / 3) / (1 + 1 / 3), rel=0.08)


def test_semi_markov_durations_are_lognormal_with_the_configured_cv():
    c = DisruptionConfig(mode="semi_markov", rate_per_hour=2.0, mean_seconds=600.0, cv=0.3)
    lengths = outage_lengths(run_np(c, n_edges=200, ticks=2000), 10.0)
    assert lengths.mean() == pytest.approx(605.0, rel=0.05)
    assert lengths.std() / lengths.mean() == pytest.approx(0.3, abs=0.05)


def test_elapsed_and_remaining_add_up_to_the_duration():
    c = DisruptionConfig(mode="semi_markov", rate_per_hour=6.0, mean_seconds=300.0, cv=0.5)
    history = run_np(c, n_edges=20, ticks=500)
    for prev, state in zip(history, history[1:], strict=False):
        going = prev.down & state.down
        np.testing.assert_allclose(
            (state.elapsed + state.remaining)[going], (prev.elapsed + prev.remaining)[going]
        )
        assert (state.elapsed[~state.down] == 0).all()


def test_only_eligible_edges_go_down():
    c = DisruptionConfig(mode="markov", rate_per_hour=20.0)
    eligible = np.arange(10) % 2 == 0
    history = run_np(c, 10, 300, eligible=eligible)
    down = np.array([s.down for s in history])
    assert down[:, eligible].any() and not down[:, ~eligible].any()


def test_scripted_windows():
    c = DisruptionConfig(mode="scripted", scripted=((1, 100.0, 300.0),))
    history = run_np(c, 3, 40)
    times = 10.0 * (np.arange(40) + 1)  # the state after tick t is at (t + 1) dt
    down = np.array([s.down[1] for s in history])
    np.testing.assert_array_equal(down, (times >= 100) & (times < 300))
    i = int(np.flatnonzero(times == 150)[0])
    assert history[i].elapsed[1] == 50 and history[i].remaining[1] == 150


def test_slow_running_keeps_edges_open():
    c = DisruptionConfig(mode="scripted", scripted=((0, 0.0, 100.0),), speed=2.0)
    state = run_np(c, 2, 1)[0]
    is_open, speed = dis.edge_effect_np(c, state)
    assert is_open.all() and list(speed) == [2.0, 1.0]
    with pytest.raises(ValueError):
        DisruptionConfig(speed=0.5)


@pytest.mark.usefixtures("x64")
@pytest.mark.parametrize(
    "config",
    [
        DisruptionConfig(mode="markov", rate_per_hour=5.0, mean_seconds=200.0),
        DisruptionConfig(mode="semi_markov", rate_per_hour=5.0, mean_seconds=200.0, cv=0.4),
        DisruptionConfig(mode="scripted", scripted=((0, 50.0, 120.0), (2, 0.0, 30.0))),
    ],
    ids=["markov", "semi_markov", "scripted"],
)
def test_jax_matches_numpy_given_the_same_draws(config):
    n, dt = 6, 10.0
    rng = np.random.default_rng(0)
    eligible = np.array([True, True, False, True, True, True])
    s_np, s_jax = dis.initial_np(n), dis.initial(n)
    step = jax.jit(lambda s, d, now: dis.step(config, s, d, eligible, now, dt))
    for t in range(300):
        draw = dis.draw_np(config, rng, n)
        s_np = dis.step_np(config, s_np, draw, eligible, t * dt, dt)
        s_jax = step(s_jax, dis.DisruptionDraw(*map(jnp.asarray, draw)), t * dt)
        for a, b in zip(s_np, s_jax, strict=True):
            np.testing.assert_array_equal(a, np.asarray(b))


def test_jax_draws_match_numpy_in_distribution():
    c = DisruptionConfig(mode="semi_markov", mean_seconds=600.0, cv=0.5)
    d = jax.jit(lambda k: dis.draw(c, k, 100_000))(jax.random.key(0))
    duration = np.asarray(d.duration)
    assert duration.mean() == pytest.approx(600.0, rel=0.02)
    assert duration.std() / duration.mean() == pytest.approx(0.5, rel=0.05)


# -- in the train env ----------------------------------------------------------


def test_a_blocked_section_holds_trains_until_it_reopens():
    from transit_bench.envs.train import PRESETS
    from transit_bench.envs.train.sim import GO, TrainSim

    base = replace(PRESETS["line"], trains_per_direction=1, running_cv=0, incident_probability=0)
    sim = TrainSim(base)
    first_section = int(sim.infra.route[0][1])
    window = ((first_section, 0.0, 600.0),)
    c = replace(base, disruption=DisruptionConfig(mode="scripted", scripted=window))
    sim = TrainSim(c)
    T = len(sim.pos)
    entered = None
    for t in range(120):
        sim.step([GO] * T)
        if entered is None and sim.pos[0] >= 1:
            entered = t
    # It waits on the platform until the block reopens at 600 s (tick 60).
    assert entered is not None and entered >= 60
    # About 500 s late when the block reopens (it was due in at 94.5 s), then
    # slack in the timetable wins a little back.
    now = sim.t * c.step_seconds
    assert 450 <= sim.projected_delay(0, now) <= 510


def test_remaining_time_is_observed_only_when_configured():
    from transit_bench.envs.train import PRESETS
    from transit_bench.envs.train.sim import BLOCK_FIELDS, HEAD_FIELDS, TrainSim

    base = replace(PRESETS["line"], trains_per_direction=1)
    block = int(TrainSim(base).infra.route[0][1])
    fields = {}
    for observe in (False, True):
        d = DisruptionConfig(
            mode="scripted", scripted=((block, 0.0, 600.0),), observe_remaining=observe
        )
        sim = TrainSim(replace(base, disruption=d))
        sim.step([0] * len(sim.pos))
        obs = sim.observe().reshape(len(sim.pos), -1)[0]
        # Train 0 is on its first platform: its first look-ahead block is route[1].
        assert sim.pos[0] == 0
        fields[observe] = obs[HEAD_FIELDS : HEAD_FIELDS + BLOCK_FIELDS]
    assert fields[False][3] == 0 and fields[False][4] > 0 and fields[False][5] == 0
    assert fields[True][5] > 0
