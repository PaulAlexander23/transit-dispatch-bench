"""The train env: infrastructure, the Python reference, and JAX against it."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from transit_bench import make
from transit_bench.core.policy import RandomPolicy, make_rollout
from transit_bench.envs.train import PRESETS, TrainConfig, TrainEnv
from transit_bench.envs.train.heuristic import FirstComeFirstServed, SectionReservation
from transit_bench.envs.train.infra import EAST, WEST, build
from transit_bench.envs.train.sim import (
    DIVERT,
    GO,
    TERMINATE,
    WAIT,
    Exogenous,
    TrainSim,
    Trip,
    first_come_first_served,
    give_way_to_express,
    heuristic,
    reroute,
    route_setting,
    section_reservation,
    terminate_when_stuck,
)


@pytest.fixture
def x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def calm(sim):
    """No noise and no incidents on `sim`'s trip; returns a step's draws
    without disruptions."""
    shape = sim.infra.route.shape
    sim.trip = Trip(np.ones(shape), np.zeros(shape))
    n_blocks = sim.infra.n_blocks
    return Exogenous(np.ones(n_blocks), np.zeros(n_blocks))


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
    exo = calm(sim)
    while not done:
        _, _, terminated, truncated, info = sim.step([GO] * T, exo)
        done = terminated or truncated
    assert terminated and info["finished"] == T
    assert sim.arrival_delay == [0.0] * T


def test_holding_a_train_makes_it_late_and_the_return_is_the_delay():
    c = replace(PRESETS["line"], trains_per_direction=1, slack=0.0)
    sim = TrainSim(c)
    T = len(sim.pos)
    total, done, held = 0.0, False, 0
    exo = calm(sim)
    while not done:
        # Hold train 0 for its first five decisions.
        actions = [GO] * T
        if sim.deciding(0) and held < 5:
            actions[0] = WAIT
            held += 1
        _, reward, terminated, truncated, _ = sim.step(actions, exo)
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
    sim = TrainSim(replace(PRESETS["line"], trains_per_direction=1))
    first = int(sim.infra.route[0][1])
    sim.block_open[first] = False
    T = len(sim.pos)
    exo = calm(sim)
    for _ in range(30):
        sim.step([GO] * T, exo)
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
    state, ts = env.reset(jax.random.key(0), sim.trip)
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
        # Four actions everywhere: TERMINATE without a terminate_penalty, and
        # DIVERT away from a junction, hold as WAIT does.
        replay(preset, lambda sim, rng: rng.integers(4, size=len(sim.pos)), seed)
        replay(preset, lambda sim, rng: first_come_first_served(sim), seed)
        replay(preset, lambda sim, rng: section_reservation(sim), seed)
        replay(preset, lambda sim, rng: route_setting(sim), seed)
        replay(preset, lambda sim, rng: terminate_when_stuck(sim), seed)
        replay(preset, lambda sim, rng: reroute(sim), seed)
        replay(preset, lambda sim, rng: heuristic(sim), seed)


@pytest.mark.usefixtures("x64")
@pytest.mark.parametrize("preset", PRESETS)
def test_jax_heuristics_match_python(preset):
    from transit_bench.envs.train.heuristic import GiveWayToExpress, Reroute, RouteSetting

    env = TrainEnv(preset)
    for policy, reference in [
        (env.heuristic(), heuristic),
        (RouteSetting(env), route_setting),
        (GiveWayToExpress(env), give_way_to_express),
        (Reroute(env), reroute),
        (SectionReservation(env), section_reservation),
        (FirstComeFirstServed(env), first_come_first_served),
    ]:
        sim = TrainSim(preset)
        sim.reset(3)
        state, _ = env.reset(jax.random.key(0), sim.trip)
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
    keys = jax.random.split(jax.random.key(0), 2_000)
    trip = jax.jit(jax.vmap(env.draw_trip))(keys)
    noise, incident = np.asarray(trip.noise), np.asarray(trip.incident)
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
    entering, inside, before, after = obs[:, HEAD_FIELDS - 4 : HEAD_FIELDS].T
    assert list(entering) == [1, 1]
    assert list(inside) == [0, 0]
    assert list(before) == [0, 1]  # train 1 sees train 0 go first
    assert list(after) == [1, 0]


def test_route_setting_avoids_the_platform_cycle_deadlocks():
    """Over 100 seeds on `single`, section reservation deadlocks a few
    times; route setting never does, and is not later on average."""

    def run(policy):
        deadlocks, total = 0, 0.0
        for seed in range(100):
            sim = TrainSim("single")
            sim.reset(seed)
            done, stuck = False, 0
            while not done:
                _, reward, terminated, truncated, info = sim.step(policy(sim))
                total += reward
                stuck = max(stuck, info["deadlocked"])
                done = terminated or truncated
            deadlocks += stuck > 0
        return deadlocks, total

    reserve_dl, reserve_return = run(section_reservation)
    route_dl, route_return = run(route_setting)
    assert reserve_dl > 0 and route_dl == 0
    assert route_return > reserve_return


def test_shield_vetoes_unsafe_go_and_makes_always_go_route_setting():
    env = TrainEnv("single", shield=True)
    plain = TrainEnv("single")
    rule = plain.heuristic()
    always_go = FirstComeFirstServed(env)
    for seed in range(3):
        shielded, _ = jax.jit(make_rollout(env, always_go, 16, env.max_steps))(jax.random.key(seed))
        reference, _ = jax.jit(make_rollout(plain, rule, 16, plain.max_steps))(jax.random.key(seed))
        np.testing.assert_allclose(np.asarray(shielded), np.asarray(reference))
    state, _ = env.reset(jax.random.key(0))
    active = env.active_agents(state)
    assert not (active & env.vetoed(state)).any()


def test_route_setting_lets_trains_follow_on_double_track():
    """Only shared (single-track) sections need a free far platform; on a
    double-track line route setting is first come, first served."""
    assert not build(PRESETS["line"]).section_shared.any()
    assert build(PRESETS["single"]).section_shared.all()
    sim = TrainSim("passing")
    sim.reset(0)
    done = False
    while not done:
        assert route_setting(sim) == first_come_first_served(sim)
        _, _, terminated, truncated, _ = sim.step(route_setting(sim))
        done = terminated or truncated


# -- response actions ------------------------------------------------------------


def responsive(**changes):
    """One train each way on the double-track line, calm, TERMINATE enabled."""
    return replace(
        PRESETS["line"],
        trains_per_direction=1,
        running_cv=0,
        incident_probability=0,
        terminate_penalty=5.0,
        **changes,
    )


def test_cancelling_a_train_before_departure_costs_every_call():
    sim = TrainSim(responsive())
    assert TrainEnv(responsive()).n_actions == 3 and TrainEnv("line").n_actions == 2
    # Train 0 is due out now; train 1 (westbound) only at 150 s.
    assert sim.can_terminate(0) and not sim.can_terminate(1)
    _, reward, _, _, info = sim.step([TERMINATE, GO])
    assert sim.finished(0) and info["terminated"] == 1 and info["missed"] == 5
    assert reward == pytest.approx(-5 * 5.0)  # 5 calls x 5 minutes, and no delay yet


def test_short_turning_frees_the_platform_and_misses_the_calls_beyond():
    c = responsive()
    sim = TrainSim(c)
    infra = sim.infra
    total = 0.0
    turned = None
    for _ in range(200):
        on_platform = sim.pos[0] >= 0 and infra.station_of[sim.here(0)] == 2
        turn = on_platform and sim.ready(0)
        if turn:
            turned = sim.here(0)
            assert sim.can_terminate(0) and sim.missed_calls(0) == 2  # stations 3 and 4
        _, reward, terminated, _, info = sim.step([TERMINATE if turn else GO, GO])
        total += reward
        if turn:
            assert info["missed"] == 2 and sim.occupant[turned] == -1
        if terminated:
            break
    assert turned is not None and terminated
    # The return still telescopes to minus the trains' final lateness, which
    # for the turned train includes the penalty.
    assert sim.arrival_delay[0] >= 2 * 5.0 * 60
    assert total == pytest.approx(-sum(sim.arrival_delay) / 60)


def test_a_train_held_by_a_closed_block_can_terminate_and_is_active():
    from transit_bench.core.disruption import DisruptionConfig

    base = responsive()
    first_section = int(build(base).route[0][1])
    window = ((first_section, 0.0, 3000.0),)
    d = DisruptionConfig(mode="scripted", scripted=window, observe_remaining=True)
    env = TrainEnv(replace(base, disruption=d))
    state, _ = env.reset(jax.random.key(0))
    step = jax.jit(env.step)
    while not (state.pos[0] == 0 and env.ready(state)[0]):  # onto the platform, dwell
        state, _ = step(state, jnp.array([GO, GO]))
    assert not env.deciding(state)[0] and env.can_terminate(state)[0]
    assert env.active_agents(state)[0]
    # The rule terminates it: 50 minutes of outage left outweighs 4 calls x 5 minutes.
    _, actions = env.heuristic().act(env.heuristic().init(), state)
    assert int(actions[0]) == TERMINATE


def test_the_trip_is_the_same_whatever_the_policy():
    """Common random numbers: a seed fixes each train's running times."""
    env = TrainEnv("single")
    a, _ = env.reset(jax.random.key(5))
    b, _ = env.reset(jax.random.key(5))
    np.testing.assert_array_equal(a.trip.noise, b.trip.noise)
    assert a.trip.noise.shape == env.infra.route.shape
    assert (np.asarray(a.trip.noise) != 1).all()


def test_giving_way_to_the_express_reaches_the_bound_when_deterministic():
    """On `passing` without noise or incidents, the hindsight-optimal
    schedule holds a stopping train at the last loop before the terminus
    for the express behind it; the rule makes exactly those holds."""
    pytest.importorskip("highspy")
    from transit_bench.envs.train.bound import run, scenario, solve
    from transit_bench.envs.train.heuristic import GiveWayToExpress, RouteSetting

    c = replace(PRESETS["passing"], running_cv=0.0, incident_probability=0.0)
    env = TrainEnv(c, max_steps=720)
    key = jax.random.key(0)
    bound = solve(env, scenario(env, key)).cost
    give_way, finished = run(env, GiveWayToExpress(env), key)
    assert finished and float(give_way) == pytest.approx(bound, abs=1e-6)
    route, _ = run(env, RouteSetting(env), key)
    assert float(route) > bound + 5  # 12.2 vs 3.6 minutes of lateness in total


# -- route choice ------------------------------------------------------------------


def relieved(**changes):
    """One train each way on the single-track line, calm, with a relief track
    alongside section 1 (stations 1 to 2)."""
    return replace(
        PRESETS["single"],
        trains_per_direction=1,
        running_cv=0,
        incident_probability=0,
        relief_sections=(1,),
        **changes,
    )


def test_relief_track_is_an_alternative_for_one_section_both_ways():
    infra = build(relieved())
    k = relieved().blocks_per_section
    assert TrainEnv(relieved()).n_actions == 4
    for i in range(2):
        alt = infra.alt_route[i]
        assert (alt >= 0).sum() == k  # the one section, in either direction
        positions = np.flatnonzero(alt >= 0)
        assert set(infra.section_of[alt[positions]]) == {infra.section_of.max()}
        assert (infra.alt_run[i, positions] == 1.25 * infra.run[i, positions]).all()
    # Both directions use it, so route setting treats it as single track.
    assert infra.section_shared[infra.section_of.max()]


def run_train_zero(sim, choose):
    exo = calm(sim)
    blocks = set()
    while not sim.finished(0):
        sim.step([choose(sim), WAIT], exo)
        if not sim.finished(0):
            blocks.add(sim.here(0))
    return blocks, sim.arrival_delay[0]


def test_diverting_takes_the_relief_track_to_the_end_of_the_section():
    main_blocks, _ = run_train_zero(TrainSim(relieved()), lambda sim: GO)
    sim = TrainSim(relieved())
    relief_blocks, late = run_train_zero(sim, lambda sim: DIVERT if sim.alt_block(0) >= 0 else GO)
    alt = sim.infra.alt_route[0]
    assert set(alt[alt >= 0]) <= relief_blocks and not set(alt[alt >= 0]) & main_blocks
    # 2 blocks x 90 s x 1.25 = 2 x 12 ticks instead of 2 x 9: 60 s later,
    # less the timetable's slack (5% of the 1170 s nominal run).
    infra = sim.infra
    slack = infra.sched[0, -1] - 10.0 * infra.cum_ticks[0, -1]
    assert late == pytest.approx(60 - slack)


def test_reroute_goes_around_a_closed_block_where_route_setting_waits():
    from transit_bench.core.disruption import DisruptionConfig
    from transit_bench.envs.train.heuristic import Reroute, RouteSetting

    infra = build(relieved())
    closed = int(infra.route[0][np.flatnonzero(infra.alt_route[0] >= 0)[0]])
    window = ((closed, 0.0, 1800.0),)
    c = relieved(disruption=DisruptionConfig(mode="scripted", scripted=window))
    env = TrainEnv(c, max_steps=720)
    from transit_bench.envs.train.bound import run

    key = jax.random.key(0)
    waited, done_wait = run(env, RouteSetting(env), key)
    diverted, done_divert = run(env, Reroute(env), key)
    assert done_wait and done_divert
    assert float(diverted) < float(waited) - 20  # the eastbound train waits out ~25 minutes
