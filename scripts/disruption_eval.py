"""Evaluate train dispatching policies with and without blockages.

Each policy runs on train/single-v0 (no blockages) and train/blockage-v0
(random section closures), on the same seeds. Reported per env: mean delay
per train, CVaR at 10% (the mean delay per train over the worst 10% of
episodes), deadlock rate and trains finished. Policies:

- first come, first served, section reservation and route setting (the
  rules);
- a clone of route setting (the env's heuristic) trained on `single` only: the
  distribution-shift test (trained without blockages, tested with them);
- the clone fine-tuned with MAPPO on `blockage` (in distribution), if
  `--timesteps` > 0.

    uv run python scripts/disruption_eval.py --timesteps 5000000
"""

import argparse

import jax
import numpy as np

from transit_bench import make
from transit_bench.algos.imitate import warm_start
from transit_bench.algos.ppo import PPOConfig, policy, train
from transit_bench.core.policy import make_rollout
from transit_bench.envs.train.heuristic import FirstComeFirstServed, SectionReservation

ENVS = ("train/single-v0", "train/blockage-v0")


class Greedy:
    """PPO params as a policy for make_rollout (reads the observation)."""

    def __init__(self, env, params):
        self.env, self.params = env, params

    def init(self):
        return jax.numpy.zeros((), jax.numpy.int32)

    def act(self, carry, state, key):
        logits, _ = policy(self.params, self.env.observe(state), self.env.n_agents)
        return carry, logits.argmax(axis=-1).astype(jax.numpy.int32)


def evaluate(env, pol, n_episodes, seed=7):
    rewards, info = jax.jit(make_rollout(env, pol, n_episodes, env.max_steps))(jax.random.key(seed))
    delay = -np.asarray(rewards).sum(axis=0) / env.n_trains  # minutes per train, per episode
    worst = np.sort(delay)[-max(1, n_episodes // 10) :]
    return {
        "delay": delay.mean(),
        "cvar10": worst.mean(),
        "deadlocks": (np.asarray(info["deadlocked"]).max(axis=0) > 0).mean(),
        "finished": np.asarray(info["finished"]).max(axis=0).mean() / env.n_trains,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--timesteps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(f"devices: {jax.devices()}")
    envs = {env_id: make(env_id) for env_id in ENVS}
    config = PPOConfig(
        n_envs=512,
        n_steps=16,
        batch_size=2048,
        n_epochs=4,
        learning_rate=5e-5,
        net_arch=(128, 128),
        algo="mappo",
    )
    clone, _ = warm_start(make(ENVS[0], reward_shaping=True), config, 100_000, seed=args.seed)
    policies = {
        "fcfs": lambda env: FirstComeFirstServed(env),
        "reservation": lambda env: SectionReservation(env),
        "route setting": lambda env: env.heuristic(),
        "clone (single)": lambda env: Greedy(env, clone),
    }
    if args.timesteps:
        tuned, _ = train(
            ENVS[1], args.timesteps, config, seed=args.seed, params=clone, verbose=False
        )
        policies["clone + mappo (blockage)"] = lambda env: Greedy(env, tuned)
    print(f"{'policy':>26} | " + " | ".join(f"{e.split('/')[1]:^34}" for e in ENVS))
    print(f"{'':>26} | " + " | ".join(f"{'delay  cvar10  deadlk  finish':^34}" for _ in ENVS))
    for name, build in policies.items():
        cells = []
        for env in envs.values():
            m = evaluate(env, build(env), args.episodes)
            row = (m["delay"], m["cvar10"], m["deadlocks"], m["finished"])
            cells.append("{:5.2f}  {:6.2f}  {:6.3f}  {:6.3f}".format(*row))
        print(f"{name:>26} | " + " | ".join(f"{c:^34}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
