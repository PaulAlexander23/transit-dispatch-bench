"""Each rule's gap to the hindsight bound on the train presets.

For each episode (seed), solve the hindsight MILP (envs/train/bound.py) and
run each rule on the same scenario, with a long horizon so that trains
finish. Reported per preset and policy, over the episodes where the policy
finished every train: mean delay per train, the bound on those episodes,
the gap (policy − bound, minutes per train) and the share of the policy's
delay above the bound. Needs `highspy` (`uv sync --extra bound`).

    uv run python scripts/bound_eval.py --presets single blockage --episodes 200
"""

import argparse
import time

import jax
import numpy as np

from transit_bench.envs.train import TrainEnv
from transit_bench.envs.train.bound import run, scenarios, solve
from transit_bench.envs.train.heuristic import (
    FirstComeFirstServed,
    GiveWayToExpress,
    Reroute,
    RouteSetting,
    SectionReservation,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--presets",
        nargs="+",
        default=["line", "passing", "single", "blockage", "junction", "branch", "branch_outage"],
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=720)
    parser.add_argument("--time-limit", type=float, default=30.0)
    args = parser.parse_args()
    print(f"devices: {jax.devices()}")
    for preset in args.presets:
        env = TrainEnv(preset, max_steps=args.max_steps)
        keys = jax.random.split(jax.random.key(11), args.episodes)
        t0 = time.perf_counter()
        bounds = [solve(env, s, args.time_limit) for s in scenarios(env, keys)]
        lower = np.array([b.cost for b in bounds]) / env.n_trains
        optimal = np.mean([b.optimal for b in bounds])
        solve_s = np.array([b.seconds for b in bounds])
        print(
            f"{preset}: bound {lower.mean():.2f} min/train, optimal {optimal:.0%}, "
            f"solve {solve_s.mean():.2f} s mean / {solve_s.max():.1f} s max "
            f"({time.perf_counter() - t0:.0f} s total)",
            flush=True,
        )
        for name, policy in [
            ("first come first served", FirstComeFirstServed(env)),
            ("section reservation", SectionReservation(env)),
            ("route setting", RouteSetting(env)),
            ("give way to express", GiveWayToExpress(env)),
            ("reroute", Reroute(env)),
        ]:
            episode = jax.jit(jax.vmap(lambda k, env=env, policy=policy: run(env, policy, k)))
            cost, finished = map(np.asarray, episode(keys))
            cost = cost / env.n_trains
            f = finished.astype(bool)
            gap = cost[f] - lower[f]
            share = gap.sum() / max(cost[f].sum(), 1e-9)
            print(
                f"  {name:>24}: finished {f.mean():.3f}  delay {cost[f].mean():5.2f}  "
                f"bound {lower[f].mean():5.2f}  gap {gap.mean():5.2f} min/train "
                f"({share:.0%} of delay above the bound)  min gap {gap.min():+.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
