"""Observation helpers shared by envs and learners."""

import jax
import jax.numpy as jnp
import numpy as np


def one_hot(obs, nvec):
    """SB3's preprocessing of a MultiDiscrete observation: one one-hot per entry.

    `obs` (..., len(nvec)) of integers -> (..., sum(nvec)) float32.
    """
    offsets = jnp.asarray(np.concatenate([[0], np.cumsum(nvec)[:-1]]))
    index = obs.astype(jnp.int32) + offsets

    def encode(index):
        return jnp.zeros(int(np.sum(nvec)), jnp.float32).at[index].set(1.0)

    flat = index.reshape(-1, index.shape[-1])
    return jax.vmap(encode)(flat).reshape(*obs.shape[:-1], -1)
