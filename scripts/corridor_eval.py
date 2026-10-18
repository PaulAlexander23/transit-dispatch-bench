"""Coupled corridor: when should a blocked train hand its riders to the bus?

On corridor/corridor-v0 and busy-v0 (a single-track line with 10-minute
blockages, and a bus route along it), compares:

- no termination (route setting + headway holding);
- bus-blind: terminate when `factor` x the outage's expected time left
  exceeds the train env's fixed 5 minutes per missed call;
- bus-aware: the same rule, with the cost read from the buses
  (`transfer_cost`: the wait for the next bus, plus the bus ride, minus
  the timetable).

Every policy sees the same scenarios (the same seed), so differences
against bus-blind are paired; their standard errors are of the
per-episode difference. Costs are passenger-minutes per episode: rail
riders' lateness, and bus riders' time less the transfer credit.

    uv run python scripts/corridor_eval.py
"""

import argparse

import jax
import numpy as np

from transit_bench import make
from transit_bench.core.policy import make_rollout
from transit_bench.envs.corridor.heuristic import BusAware, BusBlind, NoTermination


def evaluate(env, pol, n_episodes, seed=7):
    rewards, info = jax.jit(make_rollout(env, pol, n_episodes, env.max_steps))(jax.random.key(seed))
    total = lambda k: np.asarray(info[k], np.float64).sum(axis=0)  # noqa: E731
    return {
        "cost": -np.asarray(rewards, np.float64).sum(axis=0),
        "rail": total("rail_cost"),
        "bus": total("bus_cost") - total("credit"),
        "terminated": total("terminated"),
        "transferred": total("transferred"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--envs", nargs="+", default=["corridor/corridor-v0", "corridor/busy-v0"])
    args = parser.parse_args()
    for env_id in args.envs:
        env = make(env_id)
        print(f"\n{env_id} ({args.episodes} episodes; passenger-minutes per episode)")
        policies = {"no termination": NoTermination(env)}
        for factor in (1.0, 2.0, 4.0):
            policies[f"bus-blind x{factor:g}"] = BusBlind(env, factor)
            policies[f"bus-aware x{factor:g}"] = BusAware(env, factor)
        runs = {name: evaluate(env, pol, args.episodes) for name, pol in policies.items()}
        base = runs["bus-blind x2"]["cost"]
        for name, m in runs.items():
            diff = m["cost"] - base
            se = diff.std() / np.sqrt(len(diff))
            mean = {k: v.mean() for k, v in m.items()}
            print(
                f"{name:>16}: cost {mean['cost']:8.1f}  vs blind x2 {diff.mean():+7.1f} ± {se:4.1f}"
                f"  rail {mean['rail']:6.1f}  bus {mean['bus']:8.1f}"
                f"  terminated {mean['terminated']:.2f}  transferred {mean['transferred']:5.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
