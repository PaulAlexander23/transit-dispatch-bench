"""Train env baselines and short MAPPO runs, with domain metrics.

For each preset: random, first come first served and section reservation
over many episodes (return = minus total delay in minutes, mean arrival
delay per train, deadlock rate, trains finished), then optionally MAPPO
for a few million steps, from scratch or after imitating the heuristic.

    uv run python scripts/train_baselines.py --presets passing single --timesteps 5000000
"""

import argparse

import jax
import numpy as np

from transit_bench import make
from transit_bench.algos.imitate import warm_start
from transit_bench.algos.ppo import PPOConfig, policy, train
from transit_bench.core.policy import RandomPolicy, make_rollout
from transit_bench.core.vec import make_vec_env
from transit_bench.envs.train.heuristic import FirstComeFirstServed


def metrics(env, rewards, info):
    rewards = np.asarray(rewards)
    finished = np.asarray(info["finished"]).max(axis=0)
    deadlocked = np.asarray(info["deadlocked"]).max(axis=0) > 0
    return {
        "return": float(rewards.sum(axis=0).mean()),
        "delay_min": float(-rewards.sum(axis=0).mean() / env.n_trains),
        "deadlock_rate": float(deadlocked.mean()),
        "finished": float(finished.mean() / env.n_trains),
    }


def evaluate_params(env, params, n_episodes, key):
    """Greedy rollout of PPO params, first episode per env, like make_rollout."""
    reset, step = make_vec_env(env, n_episodes)

    def run(key):
        vec_state, ts = reset(key)

        def body(carry, _):
            vec_state, obs, alive = carry
            logits, _ = policy(params, obs, env.n_agents)
            vec_state, ts = step(vec_state, logits.argmax(axis=-1))
            info = {k: jax.numpy.where(alive, ts.info[k], 0) for k in ("finished", "deadlocked")}
            reward = jax.numpy.where(alive, ts.reward, 0.0)
            return (vec_state, ts.observation, alive & ~ts.done), (reward, info)

        alive = jax.numpy.ones(n_episodes, bool)
        init = (vec_state, ts.observation, alive)
        _, out = jax.lax.scan(body, init, None, length=env.max_steps)
        return out

    return jax.jit(run)(key)


def show(name, m):
    print(
        f"  {name:>14}: return {m['return']:7.1f}  delay/train {m['delay_min']:5.2f} min  "
        f"deadlocks {m['deadlock_rate']:.3f}  finished {m['finished']:.3f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--presets", nargs="+", default=["line", "passing", "single"])
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--timesteps", type=int, default=0, help="0: baselines only")
    parser.add_argument("--algo", choices=("ppo", "mappo"), default="mappo")
    parser.add_argument("--pretrain", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(f"devices: {jax.devices()}")
    for preset in args.presets:
        env_id = f"train/{preset}-v0"
        env = make(env_id)
        print(f"{env_id}: {env.n_trains} trains, {env.n_blocks} blocks")
        for name, pol in [
            ("random", RandomPolicy(env)),
            ("fcfs", FirstComeFirstServed(env)),
            ("reservation", env.heuristic()),
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
            show("imitation", metrics(env, *evaluate_params(env, params, 512, jax.random.key(2))))
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
        show(label, metrics(env, *evaluate_params(env, params, 512, jax.random.key(2))))
        print(f"  ({args.timesteps:,} steps at {history[-1]['steps_per_second']:,.0f} steps/s)")


if __name__ == "__main__":
    main()
