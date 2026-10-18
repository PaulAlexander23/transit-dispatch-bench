"""The runway rules against the hindsight bound, on every runway preset.

For each episode (seed), solve the hindsight DP (envs/runway/bound.py) and
run each rule on the same scenario. Reported per preset and rule: the mean
delay per aircraft due in the episode (minutes), the bound's, the share of
the rule's delay above the bound, and how much of FCFS's excess over the
bound the rule recovers. Episodes with rejections (a full buffer) are left
out, since the bound must land everyone; the count is reported.

    uv run python scripts/runway_eval.py --episodes 200
"""

import argparse
import time

import jax
import numpy as np

from transit_bench.envs.runway import PRESETS, RunwayEnv
from transit_bench.envs.runway.bound import run, scenarios, solve
from transit_bench.envs.runway.heuristic import ConstrainedPositionShifting, FirstComeFirstServed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--presets", nargs="+", default=list(PRESETS))
    parser.add_argument("--episodes", type=int, default=200)
    args = parser.parse_args()
    print(f"devices: {jax.devices()}")
    for preset in args.presets:
        env = RunwayEnv(preset)
        keys = jax.random.split(jax.random.key(11), args.episodes)
        t0 = time.perf_counter()
        bounds = [solve(env.config, s, env.max_steps) for s in scenarios(env, keys)]
        lower = np.array([b.cost for b in bounds])
        seconds = np.array([b.seconds for b in bounds])
        states = np.array([b.states for b in bounds])
        policies = {"FCFS": FirstComeFirstServed(env)}
        for k, window in [(1, 5), (2, 5), (3, 6)]:
            policies[f"CPS k={k} m={window}"] = ConstrainedPositionShifting(env, k, window)
        results = {}
        for name, policy in policies.items():
            episode = jax.jit(jax.vmap(lambda key, env=env, policy=policy: run(env, policy, key)))
            results[name] = [np.asarray(x) for x in episode(keys)]
        due = results["FCFS"][1].astype(float)
        ok = np.all([r[2] == 0 for r in results.values()], axis=0)
        print(
            f"{preset}: {due.mean():.1f} aircraft due, bound {lower[ok].sum() / due[ok].sum():.2f}"
            f" min/aircraft; DP {seconds.mean():.2f} s mean, {states.mean():.0f} states;"
            f" {(~ok).sum()} episodes with rejections left out"
            f" ({time.perf_counter() - t0:.0f} s)",
            flush=True,
        )
        fcfs = results["FCFS"][0]
        for name, (delay, _, _) in results.items():
            d, b, f = delay[ok], lower[ok], fcfs[ok]
            above = (d - b).sum() / d.sum()
            recovered = (f - d).sum() / max((f - b).sum(), 1e-9)
            diff = f - d
            se = diff.std() / np.sqrt(len(diff))
            print(
                f"  {name:>14}: delay {d.sum() / due[ok].sum():5.2f} min/aircraft"
                f"  {above:4.0%} above the bound"
                f"  recovers {recovered:4.0%} of FCFS's excess"
                f"  (saves {diff.mean():5.1f} ± {se:.1f} min/episode)",
                flush=True,
            )


if __name__ == "__main__":
    main()
