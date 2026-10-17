"""Disruption response on train/response-v0: when to cancel or short-turn.

The preset is the single-track line with 20-minute outages and the
TERMINATE action (5 minutes per missed call). Cost per train = arrival
delay + penalties, in minutes. Policies:

- route setting (never terminates) and `TerminateWhenStuck` (route
  setting, plus terminate a train whose next block is closed when
  `factor` x the expected outage time left exceeds its penalty), for a
  few factors;
- with `--timesteps`, MAPPO under the safety shield (so GO/WAIT cannot
  deadlock and the learner's real choice is when to terminate), from
  scratch and from a clone of the rule.

    uv run python scripts/response_eval.py --timesteps 5000000
"""

import argparse

import jax
import numpy as np

from transit_bench import make
from transit_bench.algos.imitate import warm_start
from transit_bench.algos.ppo import PPOConfig, train
from transit_bench.core.policy import make_rollout
from transit_bench.envs.train.heuristic import RouteSetting, TerminateWhenStuck

from disruption_eval import Greedy  # noqa: E402  (scripts/ is on sys.path when run)

ENV = "train/response-v0"


def evaluate(env, pol, n_episodes, seed=7):
    rewards, info = jax.jit(make_rollout(env, pol, n_episodes, env.max_steps))(jax.random.key(seed))
    cost = -np.asarray(rewards).sum(axis=0) / env.n_trains
    worst = np.sort(cost)[-max(1, n_episodes // 10) :]
    return {
        "cost": cost.mean(),
        "se": cost.std() / np.sqrt(n_episodes),
        "cvar10": worst.mean(),
        "deadlocks": (np.asarray(info["deadlocked"]).max(axis=0) > 0).mean(),
        "terminated": np.asarray(info["terminated"]).sum(axis=0).mean(),
        "missed": np.asarray(info["missed"]).sum(axis=0).mean(),
    }


def show(name, m):
    print(
        f"{name:>28}: cost/train {m['cost']:5.2f} ± {m['se']:.2f}  CVaR10 {m['cvar10']:5.2f}  "
        f"deadlocks {m['deadlocks']:.3f}  terminated/ep {m['terminated']:.2f}  "
        f"missed calls/ep {m['missed']:.2f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--timesteps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(f"devices: {jax.devices()}")
    env = make(ENV)
    shielded = make(ENV, shield=True)
    show("route setting", evaluate(env, RouteSetting(env), args.episodes))
    for factor in (1.0, 2.0):
        pol = TerminateWhenStuck(env, factor=factor)
        show(f"terminate when stuck (x{factor:g})", evaluate(env, pol, args.episodes))
    if not args.timesteps:
        return
    config = PPOConfig(
        n_envs=512,
        n_steps=16,
        batch_size=2048,
        n_epochs=4,
        learning_rate=3e-4,
        net_arch=(128, 128),
        algo="mappo",
    )
    params, _ = train(
        ENV,
        args.timesteps,
        config,
        env_kwargs={"shield": True},
        seed=args.seed,
        verbose=False,
    )
    show("shielded mappo", evaluate(shielded, Greedy(shielded, params), args.episodes))
    clone, stats = warm_start(shielded, config, 200_000, seed=args.seed)
    show("clone of the rule (shielded)", evaluate(shielded, Greedy(shielded, clone), args.episodes))
    tuned, _ = train(
        ENV,
        args.timesteps,
        config._replace(learning_rate=5e-5),
        env_kwargs={"shield": True},
        seed=args.seed,
        params=clone,
        verbose=False,
    )
    show("clone + shielded mappo", evaluate(shielded, Greedy(shielded, tuned), args.episodes))


if __name__ == "__main__":
    main()
