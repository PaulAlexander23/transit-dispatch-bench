from collections import deque
from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from transit_bench.envs.elevator.gym_env import OBS_TYPES, ElevatorGymEnv, Shaping
from transit_bench.envs.elevator.sim import (
    DOORS,
    DOWN,
    IDLE,
    MOVING,
    PRESETS,
    SERVE,
    SERVE_DOWN,
    SERVE_UP,
    UP,
    ElevatorConfig,
    ElevatorSim,
)


def quiet(config=None, **changes):
    """A config with no arrivals, so every passenger is placed by the test."""
    return replace(config or ElevatorConfig(), arrival_probability=0.0, **changes)


def make_sim(config):
    return ElevatorSim(config, rng=np.random.default_rng(0))


@pytest.mark.parametrize("obs_type", OBS_TYPES)
@pytest.mark.parametrize("preset", PRESETS)
def test_presets_pass_gymnasium_checks_and_stay_in_space(preset, obs_type):
    env = ElevatorGymEnv(preset, obs_type=obs_type, reward_shaping=True)
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    env.action_space.seed(0)
    for _ in range(env.max_steps):
        assert env.observation_space.contains(obs), obs
        obs, _, _, _, info = env.step(env.action_space.sample())
    assert info["waiting"] >= 0


def test_presets_add_one_feature_each():
    names = list(PRESETS)
    for before, after in zip(names, names[1:], strict=False):
        a, b = PRESETS[before], PRESETS[after]
        changed = [f for f in a.__dataclass_fields__ if getattr(a, f) != getattr(b, f)]
        assert len(changed) == 1, (before, after, changed)


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        ElevatorConfig(traffic="rush")
    with pytest.raises(ValueError):
        ElevatorConfig(n_lifts=0)
    with pytest.raises(ValueError):
        ElevatorGymEnv(obs_type="multi_binary")


def test_lower_numbered_lift_boards_first():
    sim = make_sim(quiet(n_lifts=2, lift_capacity=2))
    sim.state().floor_queues[0].extend([3, 4, 5])
    delivered, boarded, served = sim.apply_actions([SERVE, SERVE])
    assert boarded == [2, 1] and served == [True, True]
    assert [lift.passengers for lift in sim.state().lifts] == [[3, 4], [5]]


def test_lifts_move_independently():
    sim = make_sim(quiet(n_lifts=3))
    sim.apply_actions([UP, UP, SERVE])
    sim.apply_actions([UP, DOWN, UP])
    assert [lift.floor for lift in sim.state().lifts] == [2, 0, 1]


def test_hall_calls_board_one_direction_in_arrival_order():
    sim = make_sim(quiet(hall_calls=True, lift_capacity=2))
    lift = sim.state().lifts[0]
    lift.position = lift.target = 5
    sim.state().floor_queues[5] = deque([2, 8, 1, 9, 7])

    _, boarded, _ = sim.apply_actions([SERVE_UP])
    assert boarded == [2] and lift.passengers == [8, 9]
    assert list(sim.state().floor_queues[5]) == [2, 1, 7]

    lift.passengers = []
    _, boarded, _ = sim.apply_actions([SERVE_DOWN])
    assert lift.passengers == [2, 1]
    assert list(sim.state().floor_queues[5]) == [7]


def test_hall_call_observation_splits_up_and_down():
    env = ElevatorGymEnv(quiet(hall_calls=True), obs_type="custom")
    env.reset(seed=0)
    env.sim.state().floor_queues[3].extend([0, 5, 6, 1, 9])
    obs = env._observation()
    n = env.config.n_floors
    assert obs[2 + 3] == 3  # going up
    assert obs[2 + n + 3] == 2  # going down


def run_until_idle(sim, max_steps=100):
    lift = sim.state().lifts[0]
    for step in range(1, max_steps + 1):
        sim.apply_actions([SERVE])  # ignored while the lift is busy
        if lift.status == IDLE:
            return step
    raise AssertionError("lift never came to rest")


def test_one_floor_trip_takes_the_kinematic_time():
    config = quiet(kinematics=True)
    sim = make_sim(config)
    lift = sim.state().lifts[0]
    sim.apply_actions([UP])
    assert lift.status == MOVING and lift.target == 1
    # Triangular profile: 2 * sqrt(3.5 m / 1 m/s^2) = 3.74 s, so 4 one-second steps.
    steps = 1
    while lift.status != IDLE:
        sim.apply_actions([DOWN])  # a reversal mid-flight is ignored
        steps += 1
    assert steps == 4
    assert lift.position == 1.0 and lift.velocity == 0.0


def test_moving_lift_can_extend_its_trip_and_respects_max_speed():
    config = quiet(kinematics=True)
    sim = make_sim(config)
    lift = sim.state().lifts[0]
    max_speed = config.max_speed / config.floor_height
    top_speed = 0.0
    for _ in range(12):
        sim.apply_actions([UP])
        assert lift.target - lift.position <= 2
        top_speed = max(top_speed, abs(lift.velocity))
    assert lift.target > 3
    assert top_speed == pytest.approx(max_speed)


def test_doors_block_the_lift_for_door_time():
    config = quiet(kinematics=True)
    sim = make_sim(config)
    lift = sim.state().lifts[0]
    lift.passengers = [0, 4]
    delivered, _, served = sim.apply_actions([SERVE])
    assert delivered == [1] and served == [True]
    assert lift.status == DOORS
    steps = 1
    while lift.status != IDLE:
        _, _, served = sim.apply_actions([UP])
        assert served == [False]
        steps += 1
    assert steps == int(config.door_time / config.step_seconds)
    assert lift.floor == 0
    sim.apply_actions([UP])
    assert lift.status == MOVING


def test_kinematic_shaping_ignores_busy_lifts():
    env = ElevatorGymEnv(quiet(kinematics=True), reward_shaping=True)
    env.reset(seed=0)
    _, reward, _, _, _ = env.step([SERVE])
    assert reward == -0.5
    _, reward, _, _, _ = env.step([SERVE])  # doors still open: no new serve
    assert reward == 0


@pytest.mark.parametrize("traffic", ["up_peak", "down_peak", "lunch", "day"])
def test_arrival_model_is_normalised(traffic):
    config = ElevatorConfig(n_floors=12, traffic=traffic)
    sim = make_sim(config)
    for fraction in (0.0, 0.4, 0.8, 1.0):
        rates, destinations = sim.arrival_model(fraction)
        assert rates.sum() == pytest.approx(config.arrival_probability * config.n_floors)
        np.testing.assert_allclose(destinations.sum(axis=1), 1.0)
        assert np.all(np.diag(destinations)[rates > 0] == 0)


def test_up_peak_starts_at_the_lobby_and_day_turns_to_down_peak():
    sim = make_sim(ElevatorConfig(traffic="day"))
    morning, _ = sim.arrival_model(0.0)
    evening, destinations = sim.arrival_model(1.0)
    assert morning[0] > morning[1:].sum()
    assert evening[0] < evening[1:].sum()
    assert destinations[5, 0] > 0.5


def test_profile_arrivals_are_valid():
    config = ElevatorConfig(n_floors=15, traffic="day")
    sim = make_sim(config)
    for step in range(200):
        sim.sample_passengers(step / 200)
    total = 0
    for floor, queue in enumerate(sim.state().floor_queues):
        total += len(queue)
        assert all(0 <= d < config.n_floors and d != floor for d in queue)
    # About 1.5 arrivals per step on average.
    assert 200 < total < 400


def test_human_render_prints_every_floor(capsys):
    env = ElevatorGymEnv("full", render_mode="human")
    env.reset(seed=0)
    env.step(env.action_space.sample())
    env.render()
    assert capsys.readouterr().out.count("|") == env.config.n_floors


def test_shaping_none_rewards_only_deliveries():
    env = ElevatorGymEnv(quiet(), reward_shaping="none")
    env.reset(seed=0)
    env.sim.state().floor_queues[0].append(3)
    _, reward, _, _, info = env.step([SERVE])
    assert reward == 0 and info["boarded"] == 1
    _, reward, _, _, _ = env.step([SERVE])  # empty serve, no penalty
    assert reward == 0


def test_progress_shaping_telescopes_to_the_potential_change():
    shaping = Shaping(pickup=0.0, empty_serve=0.0, progress=1.0, gamma=1.0)
    env = ElevatorGymEnv(quiet(kinematics=True), reward_shaping=shaping)
    env.reset(seed=0)
    env.sim.state().floor_queues[0].extend([4, 4])
    start = env._potential()
    total = 0.0
    rewards = []
    for action in [SERVE] * 4 + [UP] * 14:
        _, reward, _, _, _ = env.step([action])
        rewards.append(reward)
        total += reward
    # Boarding adds 2 x 4 floors still to go; travelling removes them again.
    assert min(rewards) < 0 < max(rewards)
    assert total == pytest.approx(env._potential() - start)


def test_progress_shaping_rewards_moving_towards_destinations():
    env = ElevatorGymEnv(quiet(), reward_shaping="progress")
    env.reset(seed=0)
    env.sim.state().lifts[0].passengers = [5]
    env._last_potential = env._potential()
    _, up_reward, _, _, _ = env.step([UP])
    _, down_reward, _, _, _ = env.step([DOWN])
    assert up_reward > 0 > down_reward


@pytest.mark.parametrize("preset", ["original", "multi", "full"])
def test_collective_policy_beats_random(preset):
    from transit_bench.envs.elevator.heuristic import CollectivePolicy

    env = ElevatorGymEnv(preset, obs_type="custom")
    policy = CollectivePolicy(preset)
    returns = []
    for seed in range(3):
        obs, _ = env.reset(seed=seed)
        policy.reset()
        total, truncated = 0.0, False
        while not truncated:
            (action,), _ = policy.predict([obs])
            obs, reward, _, truncated, _ = env.step(action)
            total += reward
        returns.append(total)
    mean = np.mean(returns)
    # Random policies deliver under 20 per episode on every preset.
    assert mean > 50, mean


def test_target_mode_drives_to_the_floor_then_serves():
    env = ElevatorGymEnv(quiet(), action_mode="target")
    env.reset(seed=0)
    lift = env.sim.state().lifts[0]
    lift.passengers = [3]
    assert env._observation()[-1] == 1  # free
    for _ in range(3):
        _, reward, _, _, _ = env.step([3])
        assert reward == 0
    assert lift.floor == 3 and env._observation()[-1] == 0  # still has its goal
    _, reward, _, _, _ = env.step([7])  # ignored: the lift is busy with floor 3
    assert reward == 1 and lift.passengers == []
    assert env._observation()[-1] == 1
    env.step([5])
    assert lift.floor == 4


def test_target_mode_with_hall_calls_picks_the_direction():
    env = ElevatorGymEnv(quiet(hall_calls=True), action_mode="target")
    env.reset(seed=0)
    n = env.config.n_floors
    assert env.action_space.nvec.tolist() == [2 * n]
    env.sim.state().floor_queues[0].extend([4])
    env.sim.state().lifts[0].position = env.sim.state().lifts[0].target = 2
    env.sim.state().floor_queues[2].extend([0, 6])
    _, _, _, _, info = env.step([n + 2])  # serve down at floor 2
    assert info["boarded"] == 1 and env.sim.state().lifts[0].passengers == [0]


def test_target_mode_with_kinematics_delivers():
    env = ElevatorGymEnv(quiet(kinematics=True), action_mode="target")
    env.reset(seed=0)
    lift = env.sim.state().lifts[0]
    lift.passengers = [2]
    delivered = 0
    for _ in range(20):
        _, _, _, _, info = env.step([2])
        delivered += info["delivered"]
    assert delivered == 1 and lift.floor == 2


def test_invalid_action_mode_is_rejected():
    with pytest.raises(ValueError):
        ElevatorGymEnv(action_mode="teleport")


def test_relative_observation_is_centred_on_each_lift():
    from transit_bench.envs.elevator.gym_env import RELATIVE_FLOOR_FIELDS, RELATIVE_LIFT_FIELDS

    env = ElevatorGymEnv(quiet(n_lifts=2), obs_type="relative")
    env.reset(seed=0)
    n = env.config.n_floors
    lifts = env.sim.state().lifts
    lifts[0].position = lifts[0].target = 3
    lifts[1].position = lifts[1].target = 7
    lifts[0].passengers = [8]
    env.sim.state().floor_queues[5].extend([1, 2])
    obs = env._observation()
    block = RELATIVE_LIFT_FIELDS + (2 * n - 1) * RELATIVE_FLOOR_FIELDS
    assert obs.shape == (2 * block,)

    def cell(lift, offset):
        start = lift * block + RELATIVE_LIFT_FIELDS + (n - 1 + offset) * RELATIVE_FLOOR_FIELDS
        return obs[start : start + RELATIVE_FLOOR_FIELDS]

    inside, wanted, up, down, others = cell(0, 2)  # floor 5, seen from floor 3
    assert inside == 1 and up == pytest.approx(2 / 8) and wanted == 0
    assert cell(0, 5)[1] == pytest.approx(1 / 8)  # floor 8 is wanted by lift 0
    assert cell(0, 4)[4] == 1  # the other lift is at floor 7
    assert cell(1, -2)[2] == pytest.approx(2 / 8)  # floor 5, seen from floor 7
    assert cell(0, -4)[0] == 0  # below the ground floor
