"""Hindsight lower bound for the train env: the best schedule, knowing everything.

After an episode, every random draw is known: each train's running times
(the `Trip`, drawn at reset) and when each block was closed (the
disruption process does not depend on the trains, so any policy sees the
same closures). The best any policy could have done on that scenario is
the optimum of a job shop with blocking (docs/formulation.md, "Train"),
written as a MILP over integer entry ticks and solved with HiGHS:

- `x[i, p]`: the tick train i enters route position p; `a[i]`: the tick it
  leaves its last block (arrives).
- Route order: `x[i, p+1] ≥ x[i, p] + d[i, p]`, where `d` is the env's
  time in the block (running time from the trip, rounded up to ticks, plus
  dwell); `a[i] ≥ x[i, L-1] + d[i, L-1]`. Staying longer is allowed: that
  is a train held, or blocked.
- Departure: `x[i, 0] ≥` the first tick at or after its scheduled time.
- Blocking: for each pair of trains sharing a block, one leaves it before
  the other enters: `x[j, q] ≥ leave[i] + 1` or `x[i, p] ≥ leave[j] + 1`,
  with a binary choosing the order (big-M). `leave[i]` is when i enters
  its next block (or arrives); the +1 is the env's rule that a block freed
  during a tick can be entered from the next tick.
- Closures: a train may not enter a block in a tick it is closed: for each
  closed window `[s, e)` of ticks, `x ≤ s − 1` or `x ≥ e` (a binary each).
- Objective: total lateness `Σ_i max(0, a[i] − scheduled arrival)`.

Every env trajectory in which all trains finish is a feasible solution with
the same cost, so the optimum is a lower bound on any policy's cost for
that scenario (tested). It is a relaxation in a few ways (no acting order
within a tick, no TERMINATE), so the bound can be loose, but never above
what is achievable. Supported: closures (disruption `speed` 0); slow running
changes the running time with the entry tick, which a fixed `d` cannot
express.

Needs the optional `highspy` dependency (`transit-bench[bound]`).
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from transit_bench.envs.train.sim import WAIT


class Scenario(NamedTuple):
    """Everything random about one episode."""

    noise: np.ndarray  # (trains, L) running-time factors (the trip)
    incident: np.ndarray  # (trains, L) incident seconds (the trip)
    closed: np.ndarray  # (ticks, blocks) bool: block closed at the start of the tick


class Bound(NamedTuple):
    cost: float  # minutes of total lateness: the lower bound (the dual bound if not optimal)
    optimal: bool
    seconds: float


def scenarios(env, keys, n_steps=None):
    """The scenarios that episodes started from `keys` (`env.reset(key)`)
    face, as a list of `Scenario`; one jitted, vmapped rollout."""
    n_steps = n_steps or env.max_steps
    wait = jnp.full(env.n_agents, WAIT, jnp.int32)

    def one(key):
        state, _ = env.reset(key)

        def body(state, _):
            closed = ~state.blocks.open
            state, _ = env.step(state, wait)
            return state, closed

        _, closed = jax.lax.scan(body, state, None, length=n_steps)
        return state.trip.noise, state.trip.incident, closed

    noise, incident, closed = map(np.asarray, jax.jit(jax.vmap(one))(keys))
    return [Scenario(*parts) for parts in zip(noise, incident, closed, strict=True)]


def scenario(env, key, n_steps=None):
    """The scenario an episode started from `key` faces."""
    return scenarios(env, key[None], n_steps)[0]


def durations(config, infra, noise, incident):
    """(trains, L) ticks in each block, as the env computes them on entry."""
    seconds = infra.run * noise + incident
    ticks = np.maximum(1, np.ceil(seconds / config.step_seconds - 1e-9)).astype(int)
    return ticks + infra.dwell


def windows(closed):
    """Per block, the [start, end) tick windows it is closed."""
    out = []
    for b in range(closed.shape[1]):
        col = np.concatenate([[False], closed[:, b], [False]])
        edges = np.flatnonzero(col[1:] != col[:-1])
        out.append(list(zip(edges[::2], edges[1::2], strict=True)))
    return out


def solve(env, scen, time_limit=30.0, horizon=None):
    """Solve the hindsight MILP for one scenario; returns a `Bound`."""
    import highspy

    c, infra = env.config, env.infra
    if c.disruption.mode != "none" and c.disruption.speed != 0:
        raise ValueError("the bound supports closures only (disruption speed 0)")
    T, L = infra.route.shape
    dt = c.step_seconds
    d = durations(c, infra, scen.noise, scen.incident)
    horizon = horizon or 4 * max(env.max_steps, len(scen.closed))
    M = 2 * horizon

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", 1e-6)
    start = [math.ceil(infra.sched[i, 0] / dt - 1e-9) for i in range(T)]
    x = [[h.addIntegral(lb=start[i], ub=horizon) for _ in range(L)] for i in range(T)]
    a = [h.addIntegral(lb=start[i], ub=horizon) for i in range(T)]
    late = [h.addVariable(lb=0.0) for _ in range(T)]
    for i in range(T):
        for p in range(L - 1):
            h.addConstr(x[i][p + 1] - x[i][p] >= int(d[i, p]))
        h.addConstr(a[i] - x[i][L - 1] >= int(d[i, L - 1]))
        h.addConstr(late[i] - dt * a[i] >= -float(infra.sched[i, L]))

    def leave(i, p):
        return x[i][p + 1] if p + 1 < L else a[i]

    # Blocking: one binary per pair of trains sharing a block.
    where = [{int(b): p for p, b in enumerate(infra.route[i])} for i in range(T)]
    for i in range(T):
        for j in range(i + 1, T):
            for b, p in where[i].items():
                q = where[j].get(b)
                if q is None:
                    continue
                first = h.addBinary()  # 1: i uses b first
                h.addConstr(x[j][q] - leave(i, p) >= 1 - M * (1 - first))
                h.addConstr(x[i][p] - leave(j, q) >= 1 - M * first)

    # Closures: no entry while closed.
    for b, spans in enumerate(windows(scen.closed)):
        for i in range(T):
            p = where[i].get(b)
            if p is None:
                continue
            for s, e in spans:
                before = h.addBinary()
                h.addConstr(x[i][p] <= int(s) - 1 + M * (1 - before))
                h.addConstr(x[i][p] >= int(e) - M * before)

    h.minimize(sum(late))
    info = h.getInfo()
    optimal = h.getModelStatus() == highspy.HighsModelStatus.kOptimal
    value = info.objective_function_value if optimal else info.mip_dual_bound
    return Bound(float(value) / 60, bool(optimal), float(h.getRunTime()))


def run(env, policy, key, n_steps=None):
    """One episode of `policy` from `env.reset(key)`, the scenario `scenario`
    returns: (total cost in minutes, every train finished)."""
    n_steps = n_steps or env.max_steps
    state, _ = env.reset(key)

    def body(carry, k):
        state, inner = carry
        inner, actions = policy.act(inner, state, k)
        state, ts = env.step(state, actions)
        return (state, inner), ts.reward

    keys = jax.random.split(jax.random.fold_in(key, 1), n_steps)
    (state, _), rewards = jax.lax.scan(body, (state, policy.init()), keys)
    return -rewards.sum(), env.finished(state).all()
