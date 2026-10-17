"""One-off check: the ported elevator env against elevator-dispatching-rl's.

Imports the old repo's package from its source tree (default
~/Repositories/elevator-dispatching-rl, checked out at 206b740) and runs both
side by side:

- Python reference: the same seed must give the same trajectory, since the
  sims draw from NumPy in the same order. Observations (on the obs types both
  have), rewards and info must match exactly.
- JAX env: the same key must give the same trajectory, including the
  arrivals drawn on the device, since the draw is the same code moved into
  `core.demand`.

    uv run python scripts/parity_old_repo.py [--old-repo PATH]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np

PRESETS = ["original", "tall", "multi", "kinematic", "hall_calls", "full"]
OPTIONS = [
    {"obs_type": "relative", "reward_shaping": True},
    {"obs_type": "custom", "reward_shaping": "progress_waiting", "observe_direction": True},
    {"obs_type": "relative", "action_mode": "target"},
]


def python_parity(old, new, preset, options, episodes=3, max_steps=200):
    env_a = old.BuildingEnv(preset, max_steps=max_steps, **options)
    env_b = new.ElevatorGymEnv(preset, max_steps=max_steps, **options)
    rng = np.random.default_rng(0)
    steps = 0
    for episode in range(episodes):
        obs_a, _ = env_a.reset(seed=episode)
        obs_b, _ = env_b.reset(seed=episode)
        np.testing.assert_array_equal(obs_a, obs_b)
        truncated = False
        while not truncated:
            action = rng.integers(env_a.action_space.nvec[0], size=env_a.config.n_lifts)
            obs_a, r_a, _, truncated, info_a = env_a.step(action)
            obs_b, r_b, _, truncated_b, info_b = env_b.step(action)
            np.testing.assert_array_equal(obs_a, obs_b)
            assert r_a == r_b and truncated == truncated_b and info_a == info_b
            steps += 1
    return steps


def jax_parity(old, new, preset, options, max_steps=200):
    env_a = old.JaxBuildingEnv(preset, max_steps=max_steps, **options)
    env_b = new.ElevatorEnv(preset, max_steps=max_steps, **options)
    key = jax.random.key(0)
    state_a, obs_a = jax.jit(env_a.reset)(key)
    state_b, ts = jax.jit(env_b.reset)(key)
    np.testing.assert_array_equal(obs_a, ts.observation)
    step_a, step_b = jax.jit(env_a.step), jax.jit(env_b.step)
    rng = np.random.default_rng(1)
    for _ in range(max_steps):
        action = rng.integers(env_a.n_actions, size=env_a.config.n_lifts)
        state_a, obs_a, r_a, truncated, info = step_a(state_a, action)
        state_b, ts = step_b(state_b, action)
        np.testing.assert_array_equal(obs_a, ts.observation)
        assert float(r_a) == float(ts.reward) and bool(truncated) == bool(ts.truncated)
        assert int(info["delivered"]) == int(ts.info["delivered"])
    return max_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--old-repo", type=Path, default=Path.home() / "Repositories/elevator-dispatching-rl"
    )
    args = parser.parse_args()
    commit = subprocess.run(
        ["git", "-C", args.old_repo, "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    sys.path.insert(0, str(args.old_repo / "src"))
    from elevator_rl import building_env as old_gym
    from elevator_rl import jax_env as old_jax

    from transit_bench.envs.elevator import gym_env as new_gym
    from transit_bench.envs.elevator import jax_env as new_jax

    print(f"old repo at {commit}; JAX on {jax.devices()[0].platform}")
    for preset in PRESETS:
        for options in OPTIONS:
            n_py = python_parity(old_gym, new_gym, preset, options)
            n_jax = jax_parity(old_jax, new_jax, preset, options)
            print(f"{preset:>10}  {options}: python {n_py} steps, jax {n_jax} steps match")
    print("all match")


if __name__ == "__main__":
    main()
