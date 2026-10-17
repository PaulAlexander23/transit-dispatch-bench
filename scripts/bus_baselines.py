"""Bus env baselines and short PPO / MAPPO runs, with domain metrics.

For each preset: random, always-go and headway holding over many episodes
(return, mean wait, headway coefficient of variation, bunching rate), then
optionally PPO or MAPPO for a few million steps, from scratch or after
imitating the heuristic.

    uv run python scripts/bus_baselines.py --presets noisy full --timesteps 5000000
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from transit_bench import make
from transit_bench.algos.imitate import warm_start
from transit_bench.algos.ppo import PPOConfig, policy, train
from transit_bench.core.policy import RandomPolicy, make_rollout
from transit_bench.core.vec import make_vec_env


class AlwaysGo:
    def __init__(self, env):
        self.env = env

    def init(self):
        return jnp.zeros((), jnp.int32)

    def act(self, carry, state, key):
        return carry, jnp.zeros(self.env.n_agents, jnp.int32)


def metrics(env, rewards, info):
    """Per-episode return and pooled domain metrics from a rollout's arrays."""
    info = {k: np.asarray(v, np.float64) for k, v in info.items()}
    # Departures without a previous bus at that stop add 0 to the gap sums;
    # there are at most n_stops of them per episode, a small bias.
    n = info["departures"].sum()
    mean_gap = info["gap_sum"].sum() / n
    cv = np.sqrt(max(info["gap_sq_sum"].sum() / n - mean_gap**2, 0.0)) / mean_gap
    wait = info["waiting"].sum() * env.config.step_seconds / max(info["boarded"].sum(), 1)
    return {
        "return": float(np.asarray(rewards).sum(axis=0).mean()),
        "wait_s": float(wait),
        "headway_cv": float(cv),
        "bunching": float(info["bunched"].sum() / n),
        "held": float(info["held"].sum() / rewards.shape[1]),
    }


def evaluate_params(env, params, n_episodes, key):
    """Greedy rollout of PPO params, returning (rewards, info) like make_rollout."""
    reset, step = make_vec_env(env, n_episodes)

    def run(key):
        vec_state, ts = reset(key)

        def body(carry, _):
            vec_state, obs = carry
            logits, _ = policy(params, obs, env.n_agents)
            vec_state, ts = step(vec_state, logits.argmax(axis=-1))
            return (vec_state, ts.observation), (ts.reward, ts.info)

        _, out = jax.lax.scan(body, (vec_state, ts.observation), None, length=env.max_steps)
        return out

    return jax.jit(run)(key)


def show(name, m):
    print(
        f"  {name:>14}: return {m['return']:8.1f}  wait {m['wait_s']:5.0f}s  "
        f"cv {m['headway_cv']:.2f}  bunching {m['bunching']:.3f}  "
        f"holds/ep {m['held']:.0f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--presets", nargs="+", default=["loop", "noisy", "busy", "peak", "full"])
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=0, help="0: baselines only")
    parser.add_argument("--algo", choices=("ppo", "mappo"), default="mappo")
    parser.add_argument("--pretrain", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(f"devices: {jax.devices()}")
    for preset in args.presets:
        env_id = f"bus/{preset}-v0"
        env = make(env_id)
        print(f"{env_id}: H = {env.headway:.0f}s")
        for name, pol in [
            ("random", RandomPolicy(env)),
            ("always go", AlwaysGo(env)),
            ("holding", env.heuristic()),
        ]:
            rollout = jax.jit(make_rollout(env, pol, args.episodes, env.max_steps))
            show(name, metrics(env, *rollout(jax.random.key(1))))
        if not args.timesteps:
            continue
        config = PPOConfig(
            n_envs=512,
            n_steps=16,
            batch_size=2048,
            n_epochs=4,
            learning_rate=args.learning_rate,
            net_arch=(128, 128),
            algo=args.algo,
        )
        params = None
        if args.pretrain:
            params, stats = warm_start(env, config, args.pretrain, seed=args.seed)
            print(f"  imitation log-likelihood {stats['log_likelihood']:.3f}")
            show("imitation", metrics(env, *evaluate_params(env, params, 256, jax.random.key(2))))
        params, history = train(
            env_id,
            args.timesteps,
            config,
            seed=args.seed,
            eval_freq=max(args.timesteps // 5, 1),
            n_eval_episodes=64,
            params=params,
            verbose=False,
        )
        label = f"{args.algo}{'+bc' if args.pretrain else ''}"
        rate = history[-1]["steps_per_second"]
        show(label, metrics(env, *evaluate_params(env, params, 256, jax.random.key(2))))
        print(f"  ({args.timesteps:,} steps at {rate:,.0f} steps/s)")


if __name__ == "__main__":
    main()
