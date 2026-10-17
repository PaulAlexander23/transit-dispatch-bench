"""Warm-start PPO by imitating the env's heuristic (behaviour cloning).

Ported from elevator-dispatching-rl (`jax_imitate.py` at 206b740). The old
version ran the heuristic on the Python env (about a minute per 100k samples,
hence a disk cache); here the heuristic is a JAX policy on the env state
(`env.heuristic()`), so collection is one jitted rollout of many envs and
takes seconds. The policy MLP is fitted to the heuristic's actions
(cross-entropy) and the value MLP to its discounted (shaped) returns.

    params, stats = warm_start(env, config, n_samples=100_000)
    ppo.train(..., params=params)
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax

from transit_bench.algos.ppo import PPOConfig, init_params, log_prob, policy
from transit_bench.core.vec import make_vec_env


def collect(env, n_samples, gamma=0.99, seed=0, heuristic=None):
    """Run the heuristic for whole episodes: observations, actions, returns.

    One episode in each of ceil(n_samples / max_steps) envs, all in one
    `jit`; the first `n_samples` (observation, action, discounted return)
    triples are kept. Returns use `env`'s reward, so pass the shaped env the
    PPO will train on.
    """
    heuristic = heuristic or env.heuristic()
    n_envs = math.ceil(n_samples / env.max_steps)
    reset, step = make_vec_env(env, n_envs)
    act = jax.vmap(heuristic.act)

    @jax.jit
    def run(key):
        k_reset, k_run = jax.random.split(key)
        vec_state, ts = reset(k_reset)
        carry = jax.vmap(lambda _: heuristic.init())(jnp.arange(n_envs))

        def body(loop, key):
            vec_state, carry, obs = loop
            carry, action = act(carry, vec_state.env, jax.random.split(key, n_envs))
            vec_state, ts = step(vec_state, action)
            return (vec_state, carry, ts.observation), (obs, action, ts.reward)

        keys = jax.random.split(k_run, env.max_steps)
        _, (obs, actions, rewards) = jax.lax.scan(body, (vec_state, carry, ts.observation), keys)

        def back(g, reward):
            g = reward + gamma * g
            return g, g

        _, returns = jax.lax.scan(back, jnp.zeros(n_envs), rewards, reverse=True)

        # (steps, envs, ...) -> episode by episode, as the Python collector
        def flat(x):
            return jnp.swapaxes(x, 0, 1).reshape(n_envs * env.max_steps, *x.shape[2:])

        return flat(obs), flat(actions), flat(returns)

    observations, actions, returns = run(jax.random.key(seed))
    n = n_samples
    return (
        np.asarray(observations[:n]),
        np.asarray(actions[:n]),
        np.asarray(returns[:n], np.float32),
    )


def pretrain(params, observations, actions, returns, n_agents, epochs=15, batch_size=256, lr=1e-3,
             seed=0):  # fmt: skip
    """Fit `params` to the data; return (params, final epoch's log-likelihood, value loss)."""
    optimizer = optax.adam(lr)
    n_batches = len(actions) // batch_size
    data = (
        jnp.asarray(observations, jnp.float32),
        jnp.asarray(actions, jnp.int32),
        jnp.asarray(returns, jnp.float32),
    )

    def loss_fn(params, batch):
        obs, act, ret = batch
        logits, value = policy(params, obs, n_agents)
        log_likelihood = log_prob(logits, act).mean()
        value_loss = ((value - ret) ** 2).mean()
        return -log_likelihood + 0.5 * value_loss, (log_likelihood, value_loss)

    @jax.jit
    def epoch(carry, key):
        params, opt_state = carry
        order = jax.random.permutation(key, len(data[1]))[: n_batches * batch_size]
        batches = jax.tree.map(
            lambda x: x[order].reshape(n_batches, batch_size, *x.shape[1:]), data
        )

        def step(carry, batch):
            params, opt_state = carry
            grads, aux = jax.grad(loss_fn, has_aux=True)(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            return (optax.apply_updates(params, updates), opt_state), aux

        (params, opt_state), (ll, vl) = jax.lax.scan(step, (params, opt_state), batches)
        return (params, opt_state), (ll.mean(), vl.mean())

    carry = (params, optimizer.init(params))
    for key in jax.random.split(jax.random.key(seed), epochs):
        carry, (log_likelihood, value_loss) = epoch(carry, key)
    return carry[0], float(log_likelihood), float(value_loss)


def warm_start(env, config: PPOConfig, n_samples=100_000, epochs=15, seed=0):
    """Fresh PPO params for `env`, cloned from its heuristic.

    Returns (params, {"log_likelihood", "value_loss"}).
    """
    observations, actions, returns = collect(env, n_samples, config.gamma, seed)
    params = init_params(jax.random.key(seed), env, config)
    params, log_likelihood, value_loss = pretrain(
        params, observations, actions, returns, env.n_agents, epochs=epochs, seed=seed
    )
    return params, {"log_likelihood": log_likelihood, "value_loss": value_loss}
