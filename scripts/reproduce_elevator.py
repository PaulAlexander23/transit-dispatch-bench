"""Reproduce the old repo's reference numbers on elevator/full-v0.

Old repo (206b740): random about 7, collective control 69, tuned PPO after
an imitation warm start 73.6 ± 2.0 at 20M steps and 77.0 ± 2.4 at 50M
(1,024 envs × 8 steps, minibatch 2,048, 10 epochs, lr 5.2e-5, 256 × 256).

    uv run python scripts/reproduce_elevator.py --timesteps 10000000
"""

import argparse
import time

import jax
import numpy as np

from transit_bench import make
from transit_bench.algos.imitate import warm_start
from transit_bench.algos.ppo import PPOConfig, train
from transit_bench.core.policy import RandomPolicy, make_rollout

ENV = "elevator/full-v0"


def baseline(env, policy, n_episodes, seed=0):
    rollout = jax.jit(make_rollout(env, policy, n_episodes, env.max_steps))
    rewards, _ = rollout(jax.random.key(seed))
    returns = np.asarray(rewards).sum(axis=0)
    return returns.mean(), returns.std() / np.sqrt(len(returns))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--pretrain", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--episodes", type=int, default=1024, help="for the baselines")
    args = parser.parse_args()
    print(f"devices: {jax.devices()}")

    env = make(ENV)
    for name, policy in [("random", RandomPolicy(env)), ("heuristic", env.heuristic())]:
        mean, sem = baseline(env, policy, args.episodes)
        print(f"{name:>9}: {mean:.1f} ± {sem:.1f} (s.e.) delivered per episode")

    config = PPOConfig(
        n_envs=1024,
        n_steps=8,
        batch_size=2048,
        n_epochs=10,
        learning_rate=5.2e-5,
        net_arch=(256, 256),
    )
    shaped = make(ENV, reward_shaping=True)
    finals = []
    for seed in args.seeds:
        t0 = time.perf_counter()
        params, stats = warm_start(shaped, config, args.pretrain, seed=seed)
        print(
            f"seed {seed}: imitation {time.perf_counter() - t0:.1f}s, "
            f"log-likelihood {stats['log_likelihood']:.3f}"
        )
        _, history = train(
            ENV,
            args.timesteps,
            config,
            seed=seed,
            eval_freq=max(args.timesteps // 10, 1),
            n_eval_episodes=64,
            params=params,
        )
        finals.append(history[-1]["eval_mean"])
    print(f"PPO after {args.timesteps:,} steps: {np.mean(finals):.1f} ± {np.std(finals):.1f}")


if __name__ == "__main__":
    main()
