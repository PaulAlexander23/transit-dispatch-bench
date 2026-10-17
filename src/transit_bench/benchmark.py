"""Benchmark harness: baselines on every env, as one results table.

For each env id, runs each baseline policy for `--episodes` episodes in one
jitted rollout and reports the mean return, the normalised score (random =
0, the env's heuristic = 1) and the env family's domain metrics. Writes a
Markdown table to stdout, and with `--write PATH` into that file between
`<!-- results -->` markers (or as the whole file if it has none).

    uv run python -m transit_bench.benchmark --write docs/results.md
"""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from transit_bench.core.policy import RandomPolicy, make_rollout
from transit_bench.registry import make, registered

START, END = "<!-- results -->", "<!-- /results -->"


class Constant:
    """The same action for every agent, every step (e.g. bus 'always go')."""

    def __init__(self, env, action):
        self.env, self.action = env, action

    def init(self):
        return jnp.zeros((), jnp.int32)

    def act(self, carry, state, key):
        return carry, jnp.full(self.env.n_agents, self.action, jnp.int32)


def baselines(env_id, env):
    """Named policies for an env: random, family extras, then the heuristic."""
    out = {"random": RandomPolicy(env)}
    family = env_id.split("/")[0]
    if family == "bus":
        out["always go"] = Constant(env, 0)
    if family == "train":
        from transit_bench.envs.train.heuristic import (
            FirstComeFirstServed,
            RouteSetting,
            SectionReservation,
        )

        out["first come first served"] = FirstComeFirstServed(env)
        out["section reservation"] = SectionReservation(env)
        if env.n_actions > 2 or env.infra.express.any():  # the heuristic adds to it
            out["route setting"] = RouteSetting(env)
    if family == "fleet":
        from transit_bench.envs.fleet.heuristic import Matching, NearestIdle

        out["nearest idle"] = NearestIdle(env)
        out["matching"] = Matching(env)
    out["heuristic"] = env.heuristic()
    return out


def domain_metrics(env_id, env, rewards, info):
    """Per-family metrics as (name, formatted value) pairs."""
    family = env_id.split("/")[0]
    info = {k: np.asarray(v, np.float64) for k, v in info.items() if k != "terminal_obs"}
    if family == "elevator":
        return [("delivered/ep", f"{info['delivered'].sum(axis=0).mean():.1f}")]
    if family == "bus":
        n = info["departures"].sum()
        mean_gap = info["gap_sum"].sum() / n
        cv = np.sqrt(max(info["gap_sq_sum"].sum() / n - mean_gap**2, 0.0)) / mean_gap
        wait = info["waiting"].sum() * env.config.step_seconds / max(info["boarded"].sum(), 1)
        return [
            ("mean wait", f"{wait:.0f} s"),
            ("headway CV", f"{cv:.2f}"),
            ("bunching", f"{info['bunched'].sum() / n:.3f}"),
        ]
    if family == "train":
        delay = -np.asarray(rewards).sum(axis=0) / env.n_trains
        worst = np.sort(delay)[-max(1, len(delay) // 10) :]
        deadlocks = (info["deadlocked"].max(axis=0) > 0).mean()
        penalty = env.config.terminate_penalty > 0
        out = [
            ("cost/train" if penalty else "delay/train", f"{delay.mean():.2f} min"),
            ("CVaR10", f"{worst.mean():.2f} min"),
            ("deadlocks", f"{deadlocks:.3f}"),
        ]
        if env.has_relief:
            out.append(("diverted/ep", f"{info['diverted'].sum(axis=0).mean():.2f}"))
        if penalty:
            out.append(("terminated/ep", f"{info['terminated'].sum(axis=0).mean():.2f}"))
        return out
    if family == "fleet":
        picked = info["picked_up"].sum(axis=0)
        lost = info["abandoned"].sum(axis=0) + info["rejected"].sum(axis=0)
        wait = info["wait_ticks"].sum() * env.config.step_seconds / max(picked.sum(), 1)
        return [
            ("served/ep", f"{picked.mean():.1f}"),
            ("lost", f"{lost.sum() / max(picked.sum() + lost.sum(), 1):.3f}"),
            ("mean wait", f"{wait / 60:.1f} min"),
        ]
    return []


def run(env_ids, n_episodes, seed=0):
    """Rows of (env, policy, mean return, s.e., score, metrics)."""
    rows = []
    for env_id in env_ids:
        env = make(env_id)
        results = {}
        for name, policy in baselines(env_id, env).items():
            rollout = jax.jit(make_rollout(env, policy, n_episodes, env.max_steps))
            rewards, info = rollout(jax.random.key(seed))
            returns = np.asarray(rewards).sum(axis=0)
            metrics = domain_metrics(env_id, env, rewards, info)
            results[name] = (returns.mean(), returns.std() / np.sqrt(n_episodes), metrics)
        low, high = results["random"][0], results["heuristic"][0]
        for name, (mean, se, metrics) in results.items():
            score = (mean - low) / (high - low) if high != low else float("nan")
            rows.append((env_id, name, mean, se, score, metrics))
    return rows


def markdown(rows, n_episodes):
    lines = [
        f"Baselines over {n_episodes} episodes each (seed 0). Score: random = 0, "
        "the env's heuristic = 1. Return ± standard error.",
        "",
        "| Env | Policy | Return | Score | Domain metrics |",
        "|---|---|---|---|---|",
    ]
    for env_id, name, mean, se, score, metrics in rows:
        domain = ", ".join(f"{k} {v}" for k, v in metrics)
        lines.append(f"| `{env_id}` | {name} | {mean:.1f} ± {se:.1f} | {score:.2f} | {domain} |")
    return "\n".join(lines)


def write(path, table):
    path = Path(path)
    text = path.read_text() if path.exists() else ""
    if START in text and END in text:
        head, rest = text.split(START, 1)
        _, tail = rest.split(END, 1)
        text = f"{head}{START}\n{table}\n{END}{tail}"
    else:
        text = f"# Results\n\n{START}\n{table}\n{END}\n"
    path.write_text(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--envs", nargs="+", default=registered())
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write", help="Markdown file to put the table in")
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    rows = run(args.envs, args.episodes, args.seed)
    table = markdown(rows, args.episodes)
    print(table)
    print(f"\n({time.perf_counter() - t0:.0f} s on {jax.devices()[0].platform})")
    if args.write:
        write(args.write, table)


if __name__ == "__main__":
    main()
