"""Collective control in JAX against the Python reference, action for action."""

import jax
import numpy as np
import pytest

from transit_bench import make
from transit_bench.core.policy import RandomPolicy, make_rollout
from transit_bench.envs.elevator.gym_env import ElevatorGymEnv
from transit_bench.envs.elevator.heuristic import CollectiveControl, CollectivePolicy
from transit_bench.envs.elevator.jax_env import Arrivals, ElevatorEnv, arrivals_from_list


@pytest.fixture
def x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


@pytest.mark.usefixtures("x64")
@pytest.mark.parametrize("preset", ["original", "tall", "multi", "kinematic", "hall_calls", "full"])
def test_jax_heuristic_matches_python_action_for_action(preset):
    env = ElevatorGymEnv(preset, obs_type="custom")
    jenv = ElevatorEnv(preset)
    n = env.config.n_floors
    drawn = []
    draw = env.sim.draw_arrivals
    env.sim.draw_arrivals = lambda fraction: drawn.append(draw(fraction)) or drawn[-1]
    python, control = CollectivePolicy(preset), CollectiveControl(jenv)
    act, step = jax.jit(control.act), jax.jit(jenv.step)

    for episode in range(2):
        drawn.clear()
        obs, _ = env.reset(seed=episode)
        rounds = [arrivals_from_list(pairs, n) for pairs in drawn]
        warmup = Arrivals(*(np.stack(field) for field in zip(*rounds, strict=True)))
        state, _ = jenv.reset(jax.random.key(0), warmup)
        python.reset()
        carry = control.init()
        for t in range(env.max_steps):
            (action,), _ = python.predict([obs])
            carry, jaction = act(carry, state)
            np.testing.assert_array_equal(jaction, action, err_msg=f"step {t}")
            np.testing.assert_array_equal(carry, python.directions)
            drawn.clear()
            obs, *_ = env.step(action)
            state, _ = step(state, jaction, arrivals_from_list(drawn[0], n))


def test_heuristic_beats_random_on_full():
    env = make("elevator/full-v0")
    returns = {}
    for name, policy in [("random", RandomPolicy(env)), ("heuristic", CollectiveControl(env))]:
        rewards, _ = jax.jit(make_rollout(env, policy, 32, env.max_steps))(jax.random.key(0))
        returns[name] = float(np.asarray(rewards).sum(axis=0).mean())
    # Old repo: random about 7, collective control 69 delivered per episode.
    assert returns["random"] < 15
    assert returns["heuristic"] > 55
