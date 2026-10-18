"""Hindsight lower bound for the runway env: the best landing sequence, knowing everything.

After an episode every random draw is known: each aircraft's class and ETA
E, and when the runway was closed (neither depends on the controller). The
best any controller could have done is then a single-machine scheduling
problem with release times and sequence-dependent setups: land every
aircraft at a tick t >= E, one at a time, with t - s >= S[i, j] after the
previous landing (class i at tick s), never in a closed tick, minimising
sum(min(t, T) - E) over aircraft with E < T (the env's total delay;
aircraft with E >= T cost nothing and are left out).

It is solved exactly by dynamic programming:

- **Same-class aircraft land in ETA order.** Two aircraft of one class have
  the same separations to and from everything, so swapping them keeps the
  schedule feasible (the earlier ETA takes the earlier tick) and the cost
  the same. An optimal sequence is then fixed by the order of the *classes*:
  after landing n_k aircraft of each class k, the ones landed are the first
  n_k of each class (Psaraftis 1980).
- **State**: (n_heavy, n_large, n_small, class of the last landing). Given
  the order, landing each aircraft at the earliest tick allowed is optimal,
  since the cost only grows with a landing's tick.
- **Release times break the plain DP**: two partial sequences reaching the
  same state can differ in both the tick of the last landing and the cost
  so far, and neither need be better. The cost still to come is a
  non-decreasing function of the last tick, so a partial sequence is
  dominated only if another is no later *and* no costlier. Each state
  keeps the Pareto front of (last tick, cost so far), which makes the DP
  exact. The fronts stay short (a few entries), and an hour's traffic has
  a few thousand states.

Every episode without rejections is a feasible sequence with the same cost,
so the optimum is a lower bound on any controller's cost for that scenario
(tested), and it is attained: the gap to it is what a controller loses by
not knowing the future and by deciding greedily.
"""

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from transit_bench.envs.runway.sim import NONE, WAIT


class Scenario(NamedTuple):
    """Everything random about one episode."""

    cls: np.ndarray  # (aircraft,) int, in arrival order
    eta: np.ndarray  # (aircraft,) int
    closed: np.ndarray  # (ticks,) bool: runway closed in the tick


class Bound(NamedTuple):
    cost: float  # minutes of total delay: the optimum
    landing: np.ndarray  # (aircraft,) the optimal landing tick of each (-1 if E >= T)
    states: int  # DP states visited
    seconds: float


def scenarios(env, keys):
    """The scenarios that episodes started from `keys` (`env.reset(key)`)
    face, as a list of `Scenario`; one jitted, vmapped run of the draws
    `env.step` would make."""
    wait = jnp.full(1, WAIT, jnp.int32)

    def one(key):
        state, _ = env.reset(key)

        def body(state, _):
            key, sub = jax.random.split(state.key)
            exo = env.draw(sub, state.t)
            closed = state.closure.down[0]
            state, _ = env.step(state._replace(key=key), wait, exo)
            return state, (exo.cls, exo.count, closed)

        _, out = jax.lax.scan(body, state, None, length=env.max_steps)
        return out

    cls, count, closed = (np.asarray(x) for x in jax.jit(jax.vmap(one))(keys))
    out = []
    notice = env.config.notice
    for b in range(len(keys)):
        ticks = np.repeat(np.arange(env.max_steps), count[b])
        k = np.concatenate([cls[b, t, : count[b, t]] for t in range(env.max_steps)])
        out.append(Scenario(k.astype(int), ticks + 1 + notice, closed[b].astype(bool)))
    return out


def solve(config, scenario, horizon):
    """The optimal landing sequence for `scenario` over `horizon` ticks (the
    env's max_steps), as a `Bound`."""
    t0 = time.perf_counter()
    sep = config.separation_ticks
    K = config.n_classes
    T = horizon
    due = scenario.eta < T
    # The next open tick at or after each tick (ticks from T on are open).
    next_open = np.arange(T + 1)
    closed = np.zeros(T, bool)
    closed[: min(T, len(scenario.closed))] = scenario.closed[:T]
    for t in range(T - 1, -1, -1):
        if closed[t]:
            next_open[t] = next_open[t + 1]
    members = [np.flatnonzero(due & (scenario.cls == k)) for k in range(K)]  # ETA order
    sizes = tuple(len(m) for m in members)

    def land(t):
        return int(next_open[t]) if t < T else t

    # front[(counts, last)] = [(tick, cost, back)], back = (previous entry, aircraft).
    layer = {((0,) * K, NONE): [(0, 0, None)]}
    states = 1
    for _ in range(sum(sizes)):
        nxt = {}
        for (counts, last), front in layer.items():
            for k in range(K):
                if counts[k] == sizes[k]:
                    continue
                a = members[k][counts[k]]
                e = int(scenario.eta[a])
                gap = int(sep[last][k]) if last != NONE else 0
                key = (counts[:k] + (counts[k] + 1,) + counts[k + 1 :], k)
                bucket = nxt.setdefault(key, [])
                for entry in front:
                    t = land(max(e, entry[0] + gap) if last != NONE else e)
                    bucket.append((t, entry[1] + min(t, T) - e, (entry, a)))
        layer = {key: pareto(bucket) for key, bucket in nxt.items()}
        states += len(layer)
    best = min((entry for front in layer.values() for entry in front), key=lambda e: e[1])
    landing = np.full(len(scenario.eta), -1)
    entry = best  # walk back: each entry holds the tick of the aircraft it added
    while entry[2] is not None:
        prev, a = entry[2]
        landing[a] = entry[0]
        entry = prev
    minutes = best[1] * config.step_seconds / 60
    return Bound(minutes, landing, states, time.perf_counter() - t0)


def pareto(entries):
    """The entries no other is both no later and no costlier than."""
    entries.sort(key=lambda e: (e[0], e[1]))
    out = []
    for e in entries:
        if not out or e[1] < out[-1][1]:
            out.append(e)
    return out


def cost_of(config, scenario, landing, horizon):
    """The env's total delay (minutes) of a landing schedule."""
    due = scenario.eta < horizon
    ticks = np.minimum(landing[due], horizon) - scenario.eta[due]
    return float(ticks.sum()) * config.step_seconds / 60


def run(env, policy, key):
    """One episode from `env.reset(key)` under `policy`: (total delay in
    minutes, aircraft due before the end, rejections). Jit and vmap it."""
    state, _ = env.reset(key)

    def body(loop, _):
        state, carry = loop
        carry, action = policy.act(carry, state)
        state, ts = env.step(state, action)
        return (state, carry), (ts.info["waiting"], ts.info["due"], ts.info["rejected"])

    loop = (state, policy.init())
    _, (waiting, due, rejected) = jax.lax.scan(body, loop, None, length=env.max_steps)
    delay = waiting.sum() * env.config.step_seconds / 60
    return delay, due.sum(), rejected.sum()
