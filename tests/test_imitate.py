"""Imitation warm start for PPO (algos.imitate)."""

import jax
import numpy as np

from transit_bench.algos.imitate import collect, pretrain, warm_start
from transit_bench.algos.ppo import PPOConfig, init_params, train
from transit_bench.envs.elevator import ElevatorEnv
from transit_bench.envs.elevator.gym_env import ElevatorGymEnv
from transit_bench.envs.elevator.heuristic import CollectivePolicy


def test_collect_records_whole_heuristic_episodes():
    env = ElevatorEnv("multi", reward_shaping=True, max_steps=50)
    observations, actions, returns = collect(env, 120, gamma=0.9)
    assert observations.shape == (120, env.observation_size)
    assert actions.shape == (120, 4)
    assert returns.shape == (120,)
    # Episode by episode: the last step of each episode is its last reward.
    first = returns[:50]
    assert np.isfinite(first).all()


def test_collected_returns_match_the_python_heuristic_in_distribution():
    """Mean unshaped return of collected episodes against the Python heuristic."""
    env = ElevatorEnv("multi", max_steps=100)
    _, _, returns = collect(env, 100 * 64, gamma=1.0)
    episode_returns = returns.reshape(64, 100)[:, 0]  # return from the first step
    python = ElevatorGymEnv("multi", obs_type="custom", max_steps=100)
    policy = CollectivePolicy("multi")
    totals = []
    for seed in range(30):
        obs, _ = python.reset(seed=seed)
        policy.reset()
        total, truncated = 0.0, False
        while not truncated:
            (action,), _ = policy.predict([obs])
            obs, reward, _, truncated, _ = python.step(action)
            total += reward
        totals.append(total)
    error = np.hypot(np.std(episode_returns) / 8, np.std(totals) / np.sqrt(30))
    assert abs(episode_returns.mean() - np.mean(totals)) < 4 * error


def test_pretrain_fits_the_heuristic():
    env = ElevatorEnv("multi", reward_shaping=True)
    config = PPOConfig(net_arch=(32, 32))
    observations, actions, returns = collect(env, 2048)
    params = init_params(jax.random.key(0), env, config)
    _, before, _ = pretrain(params, observations, actions, returns, 4, epochs=1, lr=0.0)
    _, after, _ = pretrain(params, observations, actions, returns, 4, epochs=5)
    assert after > before + 1.0  # mean log-likelihood of the heuristic's 4-lift actions


def test_warm_start_feeds_ppo():
    env = ElevatorEnv("multi", reward_shaping=True)
    config = PPOConfig(n_envs=8, n_steps=16, batch_size=64, net_arch=(16, 16))
    params, stats = warm_start(env, config, n_samples=512, epochs=2)
    assert np.isfinite(stats["value_loss"])
    _, history = train(
        "elevator/multi-v0", 256, config, params=params, eval_freq=128, verbose=False
    )
    assert history[0]["timesteps"] == 0  # the warm start's own eval comes first
    assert len(history) == 2  # the first update is compilation, so not evaluated
