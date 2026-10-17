"""The fleet env in JAX: the Python reference (`sim`) as pure array functions.

Same dynamics, observation and reward as `sim.FleetSim`; given the same
`Exogenous` draws, trajectories match exactly (tests/test_fleet.py; all
quantities are integers except the observation and reward). Vehicles act
one after another in index order, through a loop over vehicles that is
unrolled when traced; requests age, abandon and arrive vectorised. New
requests are drawn with the core's loop-free Poisson OD draw
(`core.demand.draw_od`) into a fixed number of slots, and go into the
free buffer slots in order: the k-th arrival into the k-th free slot.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from transit_bench.core import demand
from transit_bench.core.types import TimeStep
from transit_bench.envs.fleet.sim import (
    HEAD_FIELDS,
    IDLE,
    NODE_FIELDS,
    PRESETS,
    SLOT_FIELDS,
    TAKE,
    TO_DROPOFF,
    TO_PICKUP,
    FleetConfig,
    demand_weights,
)

INFO_INT = ("picked_up", "delivered", "abandoned", "rejected", "wait_ticks", "moved", "waiting")


class Exogenous(NamedTuple):
    """One step's new requests, in fixed slots: the first `count` are used."""

    origin: jax.Array  # (slots,) int
    dest: jax.Array  # (slots,) int
    count: jax.Array  # () int


class FleetState(NamedTuple):
    pos: jax.Array  # (vehicles,) int: node
    phase: jax.Array  # (vehicles,) int: IDLE, TO_PICKUP or TO_DROPOFF
    goal: jax.Array  # (vehicles,) int: node it is heading for
    job: jax.Array  # (vehicles,) int: the slot it will pick up from, or -1
    valid: jax.Array  # (requests,) bool: a request is waiting in the slot
    origin: jax.Array  # (requests,) int
    dest: jax.Array  # (requests,) int
    age: jax.Array  # (requests,) int: ticks waited
    taken: jax.Array  # (requests,) int: the vehicle that took it, or -1
    t: jax.Array  # () int
    key: jax.Array


class FleetEnv:
    """On-demand vehicles on a grid, as pure JAX functions.

    `n_agents` vehicles each choose STAY (0), a move north, east, south or
    west (1–4), or TAKE request slot r (5 + r); only idle vehicles'
    choices take effect (`active_agents`). The observation is one block
    of features per vehicle (`per_agent_obs`), as `sim.FleetSim.observe`.
    """

    def __init__(self, config=None, max_steps=240, reward_shaping=False):
        if isinstance(config, str):
            config = PRESETS[config]
        self.config = c = config if config is not None else FleetConfig()
        # The reward (minus waiting minutes and lost requests) is dense.
        self.reward_shaping = reward_shaping
        self.max_steps = max_steps
        self.n_agents = c.n_vehicles
        self.n_actions = c.n_actions
        self.per_agent_obs = True
        self.observation_nvec = None
        self.observation_size = c.n_vehicles * (
            HEAD_FIELDS + SLOT_FIELDS * c.max_requests + NODE_FIELDS * c.n_nodes
        )
        origin, joint = demand_weights(c)
        self.rates = origin / origin.sum() * c.request_rate
        self.joint = joint
        self.arrival_slots = demand.arrival_slots(c.request_rate)

    def heuristic(self):
        """Per-tick bipartite matching, with idle vehicles rebalancing to
        their homes (`heuristic.Rebalance` over `heuristic.Matching`)."""
        from transit_bench.envs.fleet.heuristic import Rebalance

        return Rebalance(self)

    # -- state ---------------------------------------------------------------

    def initial_state(self, key):
        c = self.config
        V, R = c.n_vehicles, c.max_requests
        i32 = jnp.int32
        return FleetState(
            pos=jnp.array([v * c.n_nodes // V for v in range(V)], i32),
            phase=jnp.full(V, IDLE, i32),
            goal=jnp.zeros(V, i32),
            job=jnp.full(V, -1, i32),
            valid=jnp.zeros(R, bool),
            origin=jnp.zeros(R, i32),
            dest=jnp.zeros(R, i32),
            age=jnp.zeros(R, i32),
            taken=jnp.full(R, -1, i32),
            t=jnp.zeros((), i32),
            key=key,
        )

    def _info(self, **values):
        info = {k: jnp.zeros((), jnp.int32) for k in INFO_INT}
        for k, v in values.items():
            info[k] = jnp.asarray(v).astype(jnp.int32)
        return info

    def reset(self, key):
        state = self.initial_state(key)
        f = jnp.zeros(()).dtype
        false = jnp.zeros((), bool)
        ts = TimeStep(self.observe(state), jnp.zeros((), f), false, false, self._info())
        return state, ts

    def draw(self, key):
        """This step's `Exogenous` draws, as `sim.FleetSim.draw` in distribution."""
        f = jnp.zeros(()).dtype
        rates = jnp.asarray(self.rates, f)
        origin, dest, used, _ = demand.draw_od(
            key, rates, jnp.asarray(self.joint, f), self.arrival_slots
        )
        return Exogenous(origin.astype(jnp.int32), dest, used.sum(dtype=jnp.int32))

    def exogenous(self, origin, dest):
        """A Python `sim.Exogenous` as this env's fixed-slot draws (for replays)."""
        k = self.arrival_slots
        n = len(origin)
        pad = np.zeros(k, np.int32)
        o, d = pad.copy(), pad.copy()
        o[:n], d[:n] = origin, dest
        return Exogenous(jnp.asarray(o), jnp.asarray(d), jnp.asarray(n, jnp.int32))

    # -- queries -------------------------------------------------------------

    def deciding(self, state):
        return state.phase == IDLE

    def active_agents(self, state):
        """Vehicles whose action takes effect this step: the idle ones."""
        return self.deciding(state)

    def available(self, state):
        return state.valid & (state.taken < 0)

    def distance(self, a, b):
        g = self.config.grid
        return jnp.abs(a % g - b % g) + jnp.abs(a // g - b // g)

    def toward(self, a, b):
        g = self.config.grid
        ax, ay, bx, by = a % g, a // g, b % g, b // g
        step_x = jnp.where(bx > ax, 1, -1)
        step_y = jnp.where(by > ay, g, -g)
        return jnp.where(ax != bx, a + step_x, jnp.where(ay != by, a + step_y, a))

    # -- dynamics ------------------------------------------------------------

    def step(self, state, actions, exo=None):
        """One tick: (state, TimeStep). No auto-reset (see `core.vec`)."""
        c = self.config
        g = c.grid
        if exo is None:
            key, sub = jax.random.split(state.key)
            state = state._replace(key=key)
            exo = self.draw(sub)
        actions = jnp.asarray(actions, jnp.int32)
        deciding = self.deciding(state)
        pos, phase, goal, job = state.pos, state.phase, state.goal, state.job
        valid, taken, age = state.valid, state.taken, state.age
        counts = {
            k: jnp.zeros((), jnp.int32) for k in ("picked_up", "wait_ticks", "delivered", "moved")
        }
        dx = jnp.array([0, 0, 1, 0, -1], jnp.int32)
        dy = jnp.array([0, 1, 0, -1, 0], jnp.int32)
        R = c.max_requests
        for v in range(c.n_vehicles):  # in index order: earlier vehicles first
            a = actions[v]
            p = pos[v]
            is_move = deciding[v] & (a >= 1) & (a < TAKE)
            m = jnp.clip(a, 0, TAKE - 1)
            x, y = p % g + dx[m], p // g + dy[m]
            inside = (x >= 0) & (x < g) & (y >= 0) & (y < g)
            moved = is_move & inside
            p = jnp.where(moved, y * g + x, p)
            r = jnp.clip(a - TAKE, 0, R - 1)
            take = deciding[v] & (a >= TAKE) & valid[r] & (taken[r] < 0)
            taken = taken.at[r].set(jnp.where(take, jnp.int32(v), taken[r]))
            ph = jnp.where(take, TO_PICKUP, phase[v])
            gl = jnp.where(take, state.origin[r], goal[v])
            jb = jnp.where(take, r, job[v])
            busy = ph != IDLE
            p = jnp.where(busy, self.toward(p, gl), p)
            arrived = busy & (p == gl)
            pickup = arrived & (ph == TO_PICKUP)
            dropoff = arrived & (ph == TO_DROPOFF)
            rj = jnp.maximum(jb, 0)
            counts["picked_up"] += pickup
            counts["wait_ticks"] += jnp.where(pickup, age[rj], 0)
            counts["delivered"] += dropoff
            counts["moved"] += moved
            valid = valid.at[rj].set(jnp.where(pickup, False, valid[rj]))
            taken = taken.at[rj].set(jnp.where(pickup, -1, taken[rj]))
            pos = pos.at[v].set(p)
            phase = phase.at[v].set(jnp.where(pickup, TO_DROPOFF, jnp.where(dropoff, IDLE, ph)))
            goal = goal.at[v].set(jnp.where(pickup, state.dest[rj], gl))
            job = job.at[v].set(jnp.where(pickup, -1, jb))
        waiting = valid.sum(dtype=jnp.int32)
        age = jnp.where(valid, age + 1, age)
        abandon = valid & (taken < 0) & (age >= c.patience)
        valid = valid & ~abandon
        # The k-th new request goes into the k-th free slot.
        free = ~valid
        rank = jnp.cumsum(free) - 1
        k = jnp.clip(rank, 0, exo.origin.shape[0] - 1)
        fill = free & (rank < exo.count)
        origin = jnp.where(fill, exo.origin[k], state.origin)
        dest = jnp.where(fill, exo.dest[k], state.dest)
        age = jnp.where(fill, 0, age)
        taken = jnp.where(fill, -1, taken)
        valid = valid | fill
        rejected = jnp.maximum(exo.count - free.sum(dtype=jnp.int32), 0)
        state = FleetState(
            pos, phase, goal, job, valid, origin, dest, age, taken, state.t + 1, state.key
        )
        f = jnp.zeros(()).dtype
        lost = abandon.sum(dtype=jnp.int32) + rejected
        reward = -(waiting.astype(f) * c.step_seconds / 60 + c.abandon_penalty * lost)
        info = self._info(**counts, abandoned=abandon.sum(), rejected=rejected, waiting=waiting)
        truncated = state.t >= self.max_steps
        terminated = jnp.zeros((), bool)
        return state, TimeStep(self.observe(state), reward, terminated, truncated, info)

    # -- observation ---------------------------------------------------------

    def observe(self, state):
        c = self.config
        g, span = c.grid, c.grid - 1
        diameter = 2 * span
        f = jnp.zeros(()).dtype  # computed in the default float, as Python, then cast

        def fl(x):
            return x.astype(f)

        idle = state.phase == IDLE
        avail = self.available(state)
        idle_at = jnp.zeros(c.n_nodes, f).at[state.pos].add(idle.astype(f))
        waiting_at = jnp.zeros(c.n_nodes, f).at[state.origin].add(avail.astype(f))
        p = state.pos
        head = jnp.stack(
            [
                fl(p % g) / span,
                fl(p // g) / span,
                idle,
                state.phase == TO_PICKUP,
                state.phase == TO_DROPOFF,
                jnp.where(idle, 0.0, fl(self.distance(p, state.goal)) / diameter),
            ],
            axis=1,
        ).astype(f)
        o = state.origin[None, :]  # (1, R)
        pv = p[:, None]  # (V, 1)
        slots = jnp.stack(
            [
                jnp.ones_like(o * pv, dtype=f),
                fl(o % g - pv % g) / span,
                fl(o // g - pv // g) / span,
                fl(self.distance(pv, o)) / diameter,
                jnp.broadcast_to(
                    fl(self.distance(state.origin, state.dest)) / diameter,
                    (c.n_vehicles, c.max_requests),
                ),
                jnp.broadcast_to(fl(state.age) / c.patience, (c.n_vehicles, c.max_requests)),
            ],
            axis=2,
        ).astype(f)
        slots = jnp.where(avail[None, :, None], slots, 0.0)
        own = jax.nn.one_hot(p, c.n_nodes, dtype=f) * idle[:, None]
        others = idle_at[None, :] - own
        nodes = jnp.stack(
            [others / c.n_vehicles, jnp.broadcast_to(waiting_at / c.max_requests, others.shape)],
            axis=2,
        )
        out = [head, slots.reshape(c.n_vehicles, -1), nodes.reshape(c.n_vehicles, -1)]
        return jnp.concatenate(out, axis=1).astype(jnp.float32).ravel()


__all__ = ["Exogenous", "FleetEnv", "FleetState"]
