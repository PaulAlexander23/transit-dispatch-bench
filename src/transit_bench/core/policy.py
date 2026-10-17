"""Baseline policies that act on the env state inside `jit`.

A policy is `init() -> carry` and `act(carry, state, key) -> (carry, action)`.
The carry holds any memory the policy keeps between steps (collective
control's sweep directions). Heuristics may read the full env state: they
are reference points, not agents under the observation's limits. Learned
policies (`algos`) read the observation instead.
"""

from typing import Any, Protocol

import jax
import jax.numpy as jnp


class Policy(Protocol):
    def init(self) -> Any: ...

    def act(self, carry: Any, state: Any, key: jax.Array) -> tuple[Any, jax.Array]: ...


class RandomPolicy:
    """Uniform random actions: the zero of the normalised score."""

    def __init__(self, env):
        self.env = env

    def init(self):
        return jnp.zeros((), jnp.int32)

    def act(self, carry, state, key):
        return carry, jax.random.randint(key, (self.env.n_agents,), 0, self.env.n_actions)


def make_rollout(env, policy, n_envs, n_steps):
    """rollout(key) -> per-step (reward, info) of `n_envs` envs under `policy`.

    Runs inside one `jit`: (n_steps, n_envs) rewards and info arrays.
    `n_steps` is usually env.max_steps, for one episode per env. Only each
    env's first episode counts: rewards and info after it ends (an env that
    terminates early resets and carries on) are zeroed, so sums over the
    steps are per-episode totals.
    """
    from transit_bench.core.vec import make_vec_env

    reset, step = make_vec_env(env, n_envs)
    act = jax.vmap(policy.act)

    def rollout(key):
        k_reset, k_run = jax.random.split(key)
        vec_state, _ = reset(k_reset)
        carry = jax.vmap(lambda _: policy.init())(jnp.arange(n_envs))

        def body(loop, key):
            vec_state, carry, alive = loop
            carry, action = act(carry, vec_state.env, jax.random.split(key, n_envs))
            vec_state, ts = step(vec_state, action)
            # A finished env starts afresh, and so does its policy's memory.
            fresh = jax.vmap(lambda _: policy.init())(jnp.arange(n_envs))

            def pick(new, old):
                return jnp.where(ts.done.reshape((-1,) + (1,) * (old.ndim - 1)), new, old)

            carry = jax.tree.map(pick, fresh, carry)

            def mask(x):
                return jnp.where(alive.reshape((-1,) + (1,) * (x.ndim - 1)), x, 0).astype(x.dtype)

            info = jax.tree.map(mask, ts.info)
            reward = jnp.where(alive, ts.reward, 0.0)
            return (vec_state, carry, alive & ~ts.done), (reward, info)

        alive = jnp.ones(n_envs, bool)
        keys = jax.random.split(k_run, n_steps)
        _, out = jax.lax.scan(body, (vec_state, carry, alive), keys)
        return out

    return rollout
