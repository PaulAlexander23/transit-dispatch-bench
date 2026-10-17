"""Passenger demand: a time-varying Poisson origin–destination process.

Every env draws arrivals the same way. Independent Poisson counts per origin
are one Poisson total with each arrival's origin drawn in proportion to the
rates (Poisson splitting). So a step's arrivals are drawn into a fixed number
of slots: the total, then an origin and a destination per slot, all by
inverse CDF from uniforms. There is no rejection loop (each iteration of
which is a kernel launch on a GPU) and no data-dependent shape.
"""

import math

import jax
import jax.numpy as jnp


def arrival_slots(mean_total):
    """Slots for one step's arrivals given the expected total per step.

    Mean plus 10 standard deviations plus 10, rounded up to a multiple of 8:
    32 for a mean of 3, which Poisson(3) exceeds about once in 1e23 steps.
    """
    return max(8, 8 * math.ceil((mean_total + 10 * math.sqrt(mean_total) + 10) / 8))


def inverse_cdf(weights, u):
    """For each u in [0, 1), the index it falls in under `weights` (last axis).

    `weights` is unnormalised, one row or one row per u. Zero-weight indices
    are never chosen, even when rounding puts u at the very end.
    """
    cdf = jnp.cumsum(weights, axis=-1)
    index = (u[:, None] * cdf[..., -1:] >= cdf).sum(axis=-1)
    last = weights.shape[-1] - 1 - jnp.argmax(weights[..., ::-1] > 0, axis=-1)
    return jnp.minimum(index, last)


def poisson_total(mean, u, k):
    """A Poisson(mean) count from one uniform, as an index in 0..k+1.

    k + 1 means "more than k". In float32 the cumulative sum reaches 1 once
    the tail is under 6e-8, so very large totals come out a little short.
    """
    n = jnp.arange(k + 1, dtype=u.dtype)
    log_pmf = n * jnp.log(jnp.maximum(mean, 1e-30)) - mean - jax.lax.lgamma(n + 1)
    log_pmf = jnp.where(n == 0, -mean, log_pmf)
    return (u >= jnp.cumsum(jnp.exp(log_pmf))).sum()


def draw_od(key, rates, joint, k):
    """Up to `k` Poisson arrivals, in queueing order.

    rates: (origins,) expected arrivals per step at each origin.
    joint: (origins, destinations) each origin's destination distribution
    (rows may be unnormalised).

    Returns per slot the origin and destination, which slots are used, and
    the count drawn beyond the slots (lost, for the env to report).
    """
    u = jax.random.uniform(key, (2 * k + 1,), dtype=rates.dtype)
    total = poisson_total(rates.sum(), u[0], k)
    origin = inverse_cdf(rates, u[1 : k + 1])
    dest = inverse_cdf(joint[origin], u[k + 1 :]).astype(jnp.int32)
    used = jnp.arange(k) < total
    overflow = jnp.maximum(total - k, 0).astype(jnp.int32)
    return origin, dest, used, overflow


def slot_rank(origin, used):
    """Each slot's place in its origin's queue order: the number of earlier
    used slots from the same origin."""
    slot = jnp.arange(origin.shape[0])
    earlier = (origin[:, None] == origin[None, :]) & used[None, :] & (slot[None, :] < slot[:, None])
    return earlier.sum(axis=1)
