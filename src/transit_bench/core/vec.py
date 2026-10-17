"""Batched envs with auto-reset, as pure functions for `jit`."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from transit_bench.core.types import Env, TimeStep


class VecState(NamedTuple):
    env: Any  # the env's state, batched: a leading axis of n_envs on every field
    key: jax.Array  # (n_envs,) keys for the next resets
    episode_return: jax.Array  # (n_envs,) float
    episode_length: jax.Array  # (n_envs,) int


def make_vec_env(env: Env, n_envs: int):
    """(reset, step) over `n_envs` copies of `env`, resetting finished ones.

    step returns (vec_state, timestep). Where an env finished, the
    timestep's observation is already the next episode's first one, and
    info["terminal_obs"] is the observation it finished on (as SB3's
    terminal_observation), so a learner can bootstrap through a time limit.
    info also carries the running episode_return and episode_length, read
    where done for completed episodes.
    """
    v_reset = jax.vmap(env.reset)
    v_step = jax.vmap(env.step)

    def reset(key):
        k_env, k_next = jax.random.split(key)
        states, timestep = v_reset(jax.random.split(k_env, n_envs))
        vec_state = VecState(
            states,
            jax.random.split(k_next, n_envs),
            jnp.zeros(n_envs),
            jnp.zeros(n_envs, jnp.int32),
        )
        return vec_state, timestep

    def step(vec_state, actions):
        states, ts = v_step(vec_state.env, actions)
        done = ts.done
        episode_return = vec_state.episode_return + ts.reward
        episode_length = vec_state.episode_length + 1
        info = dict(
            ts.info,
            terminal_obs=ts.observation,
            episode_return=episode_return,
            episode_length=episode_length,
        )

        def reset_done(args):
            states, obs, keys = args
            split = jax.vmap(jax.random.split)(keys)
            fresh, fresh_ts = v_reset(split[:, 0])

            def pick(new, old):
                mask = done.reshape(done.shape + (1,) * (old.ndim - 1))
                return jnp.where(mask, new, old)

            keys = pick(split[:, 1], keys)
            return jax.tree.map(pick, fresh, states), pick(fresh_ts.observation, obs), keys

        # Envs with a fixed time limit that start together finish together, so
        # resets come in whole batches; the cond skips the reset work on every
        # other step.
        states, obs, keys = jax.lax.cond(
            done.any(), reset_done, lambda args: args, (states, ts.observation, vec_state.key)
        )
        vec_state = VecState(
            states,
            keys,
            jnp.where(done, 0.0, episode_return),
            jnp.where(done, 0, episode_length),
        )
        return vec_state, TimeStep(obs, ts.reward, ts.terminated, ts.truncated, info)

    return reset, step
