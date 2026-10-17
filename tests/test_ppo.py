"""PPO and MAPPO in JAX (algos.ppo)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from transit_bench.algos.ppo import (
    PPOConfig,
    entropy,
    init_params,
    log_prob,
    make_train,
    mlp,
    policy,
    train,
)
from transit_bench.envs.elevator import ElevatorEnv


def test_multidiscrete_log_prob_and_entropy():
    # Two lifts, three actions: uniform logits for lift 0, certain for lift 1.
    logits = jnp.array([[[0.0, 0.0, 0.0], [0.0, -1e9, -1e9]]])
    actions = jnp.array([[2, 0]])
    assert float(log_prob(logits, actions)[0]) == pytest.approx(np.log(1 / 3))
    assert float(entropy(logits)[0]) == pytest.approx(np.log(3))


def test_policy_shapes():
    env = ElevatorEnv("full")
    params = init_params(jax.random.key(0), env, PPOConfig())
    obs = jnp.zeros((5, env.observation_size))
    logits, value = policy(params, obs, env.n_agents)
    assert logits.shape == (5, 4, env.n_actions)
    assert value.shape == (5,)


def test_update_runs_and_changes_the_policy():
    env = ElevatorEnv("multi", reward_shaping=True, max_steps=20)
    config = PPOConfig(n_envs=8, n_steps=10, batch_size=40)
    init, update = make_train(env, config)
    runner = init(jax.random.key(0))
    before = runner.params
    for _ in range(2):  # 20 steps: the second update ends an episode (bootstrapping)
        runner, metrics = jax.jit(update)(runner)
    assert np.isfinite(float(metrics["value_loss"]))
    assert int(metrics["episodes"]) == 8
    moved = jax.tree.map(lambda a, b: bool((a != b).any()), before, runner.params)
    assert all(jax.tree.leaves(moved))


@pytest.mark.parametrize(
    "options",
    [
        {"obs_type": "custom"},
        {"action_mode": "target"},
        {"obs_type": "custom", "action_mode": "target"},
    ],
    ids=["custom", "target", "custom-target"],
)
def test_update_runs_with_target_actions_and_custom_obs(options):
    env = ElevatorEnv("hall_calls", reward_shaping=True, max_steps=20, **options)
    config = PPOConfig(n_envs=4, n_steps=10, batch_size=20, net_arch=(16, 16))
    init, update = make_train(env, config)
    runner = init(jax.random.key(0))
    if env.obs_type == "custom":
        assert runner.params["pi"][0]["w"].shape[0] == env.observation_nvec.sum()
    assert runner.params["pi"][-1]["w"].shape[1] == env.n_agents * env.n_actions
    runner, metrics = jax.jit(update)(runner)
    assert np.isfinite(float(metrics["value_loss"]))


def test_agent_encoder_shares_weights_across_lifts():
    env = ElevatorEnv("full")
    config = PPOConfig(net_arch=(32,), agent_encoder=8)
    params = init_params(jax.random.key(0), env, config)
    block = env.observation_size // env.n_agents
    encoder, hidden, head = params["pi"]
    assert encoder["w"].shape == (block, 8)
    assert hidden["w"].shape == (4 * 8, 32)
    obs = jax.random.normal(jax.random.key(1), (3, env.observation_size))
    # Lift by lift: the same encoder on each block, outputs side by side.
    encoded = jnp.concatenate(
        [
            jnp.tanh(obs[:, i * block : (i + 1) * block] @ encoder["w"] + encoder["b"])
            for i in range(env.n_agents)
        ],
        axis=-1,
    )
    expected = jnp.tanh(encoded @ hidden["w"] + hidden["b"]) @ head["w"] + head["b"]
    np.testing.assert_allclose(mlp(params["pi"], obs), expected, rtol=1e-5, atol=1e-6)
    logits, value = policy(params, obs, env.n_agents)
    assert logits.shape == (3, 4, env.n_actions)
    assert value.shape == (3,)


def test_agent_encoder_trains_and_warm_starts():
    from transit_bench.algos.imitate import warm_start

    env = ElevatorEnv("multi", reward_shaping=True, max_steps=20)
    config = PPOConfig(n_envs=8, n_steps=10, batch_size=40, net_arch=(16,), agent_encoder=8)
    params, stats = warm_start(env, config, n_samples=512, epochs=2)
    assert np.isfinite(stats["log_likelihood"])
    init, update = make_train(env, config)
    runner = init(jax.random.key(0), params)
    runner, metrics = jax.jit(update)(runner)
    assert np.isfinite(float(metrics["value_loss"]))


def test_agent_encoder_needs_the_relative_observation():
    with pytest.raises(ValueError, match="relative"):
        init_params(
            jax.random.key(0), ElevatorEnv("multi", obs_type="custom"), PPOConfig(agent_encoder=8)
        )


def test_minibatch_must_divide_the_rollout():
    with pytest.raises(ValueError, match="minibatches"):
        make_train(ElevatorEnv("multi"), PPOConfig(n_envs=8, n_steps=10, batch_size=64))


@pytest.mark.slow
def test_ppo_learns_the_original_preset():
    # SB3 with the same settings reaches about 134 in 200k steps (the
    # collective heuristic delivers 136, a random policy about 7).
    _, history = train(
        "elevator/original-v0",
        200_000,
        PPOConfig(net_arch=(256, 256)),
        eval_freq=200_000,
        verbose=False,
    )
    assert history[-1]["eval_mean"] > 100


@pytest.mark.parametrize("agent_id", [True, False])
def test_mappo_log_probs_match_a_lift_by_lift_reference(agent_id):
    env = ElevatorEnv("full")
    config = PPOConfig(net_arch=(16,), algo="mappo", agent_id=agent_id)
    params = init_params(jax.random.key(0), env, config)
    assert "pi" not in params
    block = env.observation_size // 4
    assert params["actor"][0]["w"].shape[0] == block + (4 if agent_id else 0)
    obs = jax.random.normal(jax.random.key(1), (3, env.observation_size))
    actions = jax.random.randint(jax.random.key(2), (3, 4), 0, env.n_actions)
    logits, value = policy(params, obs, 4)
    assert logits.shape == (3, 4, env.n_actions)
    assert value.shape == (3,)
    expected = 0.0
    for lift in range(4):
        x = obs[:, lift * block : (lift + 1) * block]
        if agent_id:
            x = jnp.concatenate([x, jnp.broadcast_to(jnp.eye(4)[lift], (3, 4))], axis=-1)
        log_p = jax.nn.log_softmax(mlp(params["actor"], x))
        expected += log_p[jnp.arange(3), actions[:, lift]]
    np.testing.assert_allclose(log_prob(logits, actions), expected, rtol=1e-5, atol=1e-5)


def test_mappo_trains_and_warm_starts():
    from transit_bench.algos.imitate import warm_start

    env = ElevatorEnv("multi", reward_shaping=True, max_steps=20)
    config = PPOConfig(n_envs=8, n_steps=10, batch_size=40, net_arch=(16,), algo="mappo")
    params, stats = warm_start(env, config, n_samples=512, epochs=2)
    assert np.isfinite(stats["log_likelihood"])
    init, update = make_train(env, config)
    runner = init(jax.random.key(0), params)
    runner, metrics = jax.jit(update)(runner)
    assert np.isfinite(float(metrics["value_loss"]))
    moved = jax.tree.map(lambda a, b: bool((a != b).any()), params, runner.params)
    assert all(jax.tree.leaves(moved))


@pytest.mark.parametrize("agent_id", [True, False])
@pytest.mark.parametrize("n_agents", [2, 6])
def test_mappo_actor_transfers_across_lift_counts(n_agents, agent_id):
    from dataclasses import replace

    from transit_bench.algos.ppo import make_evaluate, transfer
    from transit_bench.envs.elevator import PRESETS

    config = PPOConfig(net_arch=(16,), algo="mappo", agent_id=agent_id)
    params = init_params(jax.random.key(0), ElevatorEnv("full"), config)
    env = ElevatorEnv(replace(PRESETS["full"], n_lifts=n_agents), max_steps=20)
    moved = transfer(params, env, config, jax.random.key(1))
    logits, value = policy(moved, jnp.zeros((2, env.observation_size)), n_agents)
    assert logits.shape == (2, n_agents, env.n_actions)
    # The shared layers, and the ids of the lifts both buildings have, carry over.
    np.testing.assert_array_equal(moved["actor"][-1]["w"], params["actor"][-1]["w"])
    block = env.observation_size // n_agents
    np.testing.assert_array_equal(moved["actor"][0]["w"][:block], params["actor"][0]["w"][:block])
    if agent_id:
        kept = min(n_agents, 4)
        np.testing.assert_array_equal(
            moved["actor"][0]["w"][block : block + kept],
            params["actor"][0]["w"][block : block + kept],
        )
    rewards = jax.jit(make_evaluate(env, 2))(moved, jax.random.key(2))
    assert np.isfinite(np.asarray(rewards)).all()


def test_ppo_params_do_not_transfer():
    from transit_bench.algos.ppo import transfer

    env = ElevatorEnv("full")
    with pytest.raises(ValueError, match="mappo"):
        transfer(init_params(jax.random.key(0), env, PPOConfig()), env, PPOConfig(), None)


def test_inactive_agents_add_nothing_to_log_prob_or_entropy():
    logits = jnp.array([[[0.0, 0.0, 0.0], [0.0, -1e9, -1e9]]])
    actions = jnp.array([[2, 0]])
    active = jnp.array([[False, True]])
    assert float(log_prob(logits, actions, active)[0]) == pytest.approx(0.0)
    assert float(entropy(logits, active)[0]) == pytest.approx(0.0)
    active = jnp.array([[True, False]])
    assert float(log_prob(logits, actions, active)[0]) == pytest.approx(np.log(1 / 3))
