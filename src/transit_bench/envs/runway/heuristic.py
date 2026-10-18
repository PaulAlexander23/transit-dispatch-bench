"""Runway sequencing rules in JAX, reading the env state.

The Python references are `sim.fcfs`, `sim.cps` and `sim.wait`.
"""

import jax.numpy as jnp

from transit_bench.envs.runway.sim import LAND, NONE, WAIT, overtakes, shifts


class Wait:
    """Never land: every aircraft waits (the floor)."""

    def __init__(self, env):
        self.env = env

    def init(self):
        return jnp.zeros((), jnp.int32)

    def act(self, carry, state, key=None):
        return carry, jnp.full(1, WAIT, jnp.int32)


class FirstComeFirstServed(Wait):
    """Land the earliest arrival as soon as it is allowed."""

    def act(self, carry, state, key=None):
        head = self.env.fcfs_order(state)[0]
        go = state.valid[head] & self.env.can_land(state)[head]
        return carry, jnp.where(go, LAND + head, WAIT).astype(jnp.int32)[None]


class ConstrainedPositionShifting(Wait):
    """`sim.cps` in JAX: plan the first `window` aircraft in FCFS order over
    the orders that move none by more than k places and overtake no
    aircraft beyond k times; land the cheapest order's first aircraft
    if it can go now.

    The orders are a constant table of permutations of `window`, in
    lexicographic order. With m < `window` aircraft waiting, only the
    orders that leave positions m and beyond in place are allowed: they are
    the orders of m, in the same relative order, so the first cheapest one
    is the one the Python reference picks.
    """

    def __init__(self, env, k=2, window=5):
        super().__init__(env)
        self.k, self.window = k, window
        orders = shifts(window, k)
        self.orders = jnp.asarray(orders, jnp.int32)  # (P, window): FCFS index per position
        self.before = jnp.asarray([overtakes(o) for o in orders], jnp.int32)  # (P, window)

    def act(self, carry, state, key=None):
        env, W = self.env, self.window
        sep = jnp.asarray(env.sep, jnp.int32)
        n = state.valid.sum(dtype=jnp.int32)
        m = jnp.minimum(n, W)
        slots = env.fcfs_order(state)[:W]
        position = jnp.arange(W)
        fixed = (position[None, :] >= m) & (self.orders != position[None, :])
        present = position < m
        over = state.overtaken[slots][None, :] + self.before > self.k
        within = ~(over & (self.before > 0) & present[None, :])
        allowed = ~fixed.any(axis=1) & within.all(axis=1)
        P = self.orders.shape[0]
        last = jnp.full(P, state.last_class)
        at = jnp.full(P, state.last_time)
        total = jnp.zeros(P, jnp.int32)
        first = jnp.zeros(P, jnp.int32)
        for p in range(W):
            r = slots[self.orders[:, p]]
            k = state.cls[r]
            after = at + sep[jnp.maximum(last, 0), k]
            t = jnp.maximum(state.t, state.eta[r])
            t = jnp.where(last == NONE, t, jnp.maximum(t, after))
            here = present[p]
            total = total + jnp.where(here, t, 0)
            first = first if p else t
            last = jnp.where(here, k, last)
            at = jnp.where(here, t, at)
        cost = total + (n - m) * at
        cost = jnp.where(allowed, cost, jnp.iinfo(jnp.int32).max)
        b = jnp.argmin(cost)  # the first cheapest
        r = slots[self.orders[b, 0]]
        go = (n > 0) & (first[b] == state.t)
        return carry, jnp.where(go, LAND + r, WAIT).astype(jnp.int32)[None]
