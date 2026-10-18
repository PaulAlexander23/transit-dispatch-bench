"""The runway env in JAX: the Python reference (`sim`) as pure array functions.

Same dynamics, observation and reward as `sim.RunwaySim`; given the same
`Exogenous` draws, trajectories match exactly (tests/test_runway.py; all
quantities are integers except the observation, the reward and the
disruption's clock). New aircraft are drawn with the core's loop-free
Poisson draw (`core.demand.draw_od`, with the classes as origins) into a
fixed number of slots, and go into the free buffer slots in order.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from transit_bench.core import demand, disruption
from transit_bench.core.types import TimeStep
from transit_bench.envs.runway.sim import (
    GLOBAL_FIELDS,
    LAND,
    NONE,
    PRESETS,
    SLOT_FIELDS,
    RunwayConfig,
)

INFO_INT = ("landed", "delay_ticks", "rejected", "announced", "due", "closed", "waiting")


class Exogenous(NamedTuple):
    """One step's draws: new arrivals in fixed slots (the first `count` are
    used) and the disruption draw."""

    cls: jax.Array  # (slots,) int
    count: jax.Array  # () int
    onset: jax.Array  # (1,)
    duration: jax.Array  # (1,)


class RunwayState(NamedTuple):
    valid: jax.Array  # (slots,) bool: an aircraft is waiting in the slot
    cls: jax.Array  # (slots,) int: its wake class
    eta: jax.Array  # (slots,) int: its earliest landing tick
    seq: jax.Array  # (slots,) int: its arrival number (FCFS order)
    overtaken: jax.Array  # (slots,) int: later arrivals that landed before it
    arrived: jax.Array  # () int: arrival numbers handed out
    last_class: jax.Array  # () int: class of the last landing, or NONE
    last_time: jax.Array  # () int: its tick
    closure: disruption.DisruptionState  # the runway as one edge
    t: jax.Array  # () int
    key: jax.Array


class RunwayEnv:
    """One runway's landing sequence, as pure JAX functions.

    One agent (the controller) chooses WAIT (0) or LAND slot r (1 + r) each
    tick. The observation is a global block then one block per slot, as
    `sim.RunwaySim.observe`.
    """

    def __init__(self, config=None, max_steps=360, reward_shaping=False):
        if isinstance(config, str):
            config = PRESETS[config]
        self.config = c = config if config is not None else RunwayConfig()
        self.reward_shaping = reward_shaping  # the delay reward is dense already
        self.max_steps = max_steps
        self.n_agents = 1
        self.n_actions = c.n_actions
        self.per_agent_obs = False
        self.observation_nvec = None
        self.observation_size = GLOBAL_FIELDS + SLOT_FIELDS * c.max_aircraft
        self.sep = c.separation_ticks
        self.rate = c.rates(max_steps)
        self.mix = np.asarray(c.mix) / sum(c.mix)
        self.arrival_slots = demand.arrival_slots(float(self.rate.max()))

    def heuristic(self):
        """Constrained position shifting, k = 2 over a window of 5."""
        from transit_bench.envs.runway.heuristic import ConstrainedPositionShifting

        return ConstrainedPositionShifting(self)

    # -- state ---------------------------------------------------------------

    def initial_state(self, key):
        R = self.config.max_aircraft
        i32 = jnp.int32
        zeros = jnp.zeros(R, i32)
        return RunwayState(
            valid=jnp.zeros(R, bool),
            cls=zeros,
            eta=zeros,
            seq=zeros,
            overtaken=zeros,
            arrived=jnp.zeros((), i32),
            last_class=jnp.full((), NONE, i32),
            last_time=jnp.zeros((), i32),
            closure=disruption.initial(1),
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

    def draw(self, key, t):
        """Tick t's `Exogenous` draws, as `sim.RunwaySim.draw` in distribution."""
        f = jnp.zeros(()).dtype
        k_arrive, k_close = jax.random.split(key)
        rate = jnp.asarray(self.rate, f)[jnp.minimum(t, self.max_steps - 1)]
        rates = jnp.asarray(self.mix, f) * rate
        one = jnp.ones((self.config.n_classes, 1), f)
        cls, _, used, _ = demand.draw_od(k_arrive, rates, one, self.arrival_slots)
        d = disruption.draw(self.config.disruption, k_close, 1, f)
        count = used.sum(dtype=jnp.int32)
        return Exogenous(cls.astype(jnp.int32), count, d.onset, d.duration)

    def exogenous(self, exo):
        """A Python `sim.Exogenous` as this env's fixed-slot draws (for replays)."""
        k = self.arrival_slots
        n = len(exo.cls)
        cls = np.zeros(k, np.int32)
        cls[:n] = exo.cls
        f = jnp.zeros(()).dtype
        onset = jnp.ones(1, f) if exo.onset is None else jnp.asarray(exo.onset, f)
        duration = jnp.zeros(1, f) if exo.duration is None else jnp.asarray(exo.duration, f)
        return Exogenous(jnp.asarray(cls), jnp.asarray(n, jnp.int32), onset, duration)

    # -- queries -------------------------------------------------------------

    def active_agents(self, state):
        return jnp.ones(1, bool)

    def is_open(self, state):
        return ~state.closure.down[0]

    def free(self, state):
        """(classes,) the first tick the separation allows each class."""
        last = jnp.maximum(state.last_class, 0)
        after = state.last_time + jnp.asarray(self.sep, jnp.int32)[last]
        return jnp.where(state.last_class == NONE, 0, after)

    def can_land(self, state):
        """(slots,) may land now, apart from the runway being open."""
        free = self.free(state)[state.cls]
        return state.valid & (state.eta <= state.t) & (free <= state.t)

    def fcfs_order(self, state):
        """Slots in FCFS order, occupied ones first."""
        big = jnp.iinfo(jnp.int32).max
        return jnp.argsort(jnp.where(state.valid, state.seq, big), stable=True)

    # -- dynamics ------------------------------------------------------------

    def step(self, state, actions, exo=None):
        """One tick: (state, TimeStep). No auto-reset (see `core.vec`)."""
        c = self.config
        R = c.max_aircraft
        if exo is None:
            key, sub = jax.random.split(state.key)
            state = state._replace(key=key)
            exo = self.draw(sub, state.t)
        a = jnp.asarray(actions, jnp.int32).reshape(-1)[0]
        t = state.t
        r = jnp.clip(a - LAND, 0, R - 1)
        closed = ~self.is_open(state)
        land = (a >= LAND) & ~closed & self.can_land(state)[r]
        valid, seq = state.valid, state.seq
        overtaken = state.overtaken + (land & valid & (seq < seq[r]))
        valid = valid.at[r].set(valid[r] & ~land)
        last_class = jnp.where(land, state.cls[r], state.last_class)
        last_time = jnp.where(land, t, state.last_time)
        delay = jnp.where(land, t - state.eta[r], 0)
        waiting = (valid & (state.eta <= t)).sum(dtype=jnp.int32)
        # The k-th new aircraft goes into the k-th free slot.
        free = ~valid
        rank = jnp.cumsum(free) - 1
        k = jnp.clip(rank, 0, exo.cls.shape[0] - 1)
        fill = free & (rank < exo.count)
        eta_new = t + 1 + c.notice
        cls = jnp.where(fill, exo.cls[k], state.cls)
        eta = jnp.where(fill, eta_new, state.eta)
        seq = jnp.where(fill, state.arrived + rank, seq)
        overtaken = jnp.where(fill, 0, overtaken)
        valid = valid | fill
        n_free = free.sum(dtype=jnp.int32)
        announced = jnp.minimum(exo.count, n_free)
        rejected = jnp.maximum(exo.count - n_free, 0)
        closure = self._advance_closure(state, exo)
        state = RunwayState(
            valid, cls, eta, seq, overtaken, state.arrived + announced,
            last_class, last_time, closure, t + 1, state.key,
        )  # fmt: skip
        f = jnp.zeros(()).dtype
        reward = -(waiting.astype(f) * c.step_seconds / 60 + c.reject_penalty * rejected)
        info = self._info(
            landed=land,
            delay_ticks=delay,
            rejected=rejected,
            announced=announced,
            due=jnp.where(eta_new < self.max_steps, announced, 0),
            closed=closed,
            waiting=waiting,
        )
        truncated = state.t >= self.max_steps
        terminated = jnp.zeros((), bool)
        return state, TimeStep(self.observe(state), reward, terminated, truncated, info)

    def _advance_closure(self, state, exo):
        """As sim.RunwaySim._advance_closure: at the end of the tick."""
        c = self.config.disruption
        if c.mode == "none":
            return state.closure
        f = state.closure.elapsed.dtype
        dt = self.config.step_seconds
        draw = disruption.DisruptionDraw(jnp.asarray(exo.onset, f), jnp.asarray(exo.duration, f))
        now = state.t.astype(f) * dt
        return disruption.step(c, state.closure, draw, jnp.ones(1, bool), now, dt)

    # -- observation ---------------------------------------------------------

    def observe(self, state):
        c = self.config
        K, R = c.n_classes, c.max_aircraft
        f = jnp.zeros(()).dtype
        span = float(self.sep.max())
        t = state.t
        last = jax.nn.one_hot(state.last_class, K, dtype=f)  # zeros for NONE
        free = self.free(state)
        head = jnp.concatenate(
            [
                jnp.stack([t.astype(f) / self.max_steps, self.is_open(state).astype(f)]),
                last,
                jnp.maximum(free - t, 0).astype(f) / span,
                (state.valid.sum() / R).astype(f)[None],
            ]
        )
        earlier = state.valid[None, :] & (state.seq[None, :] < state.seq[:, None])
        rank = earlier.sum(axis=1)
        ready = jnp.maximum(state.eta, free[state.cls]) - t
        slots = jnp.concatenate(
            [
                jnp.ones((R, 1), f),
                jax.nn.one_hot(state.cls, K, dtype=f),
                jnp.stack(
                    [
                        jnp.maximum(state.eta - t, 0).astype(f) / 60,
                        jnp.maximum(t - state.eta, 0).astype(f) / 60,
                        rank.astype(f) / R,
                        jnp.maximum(ready, 0).astype(f) / span,
                    ],
                    axis=1,
                ),
            ],
            axis=1,
        )
        slots = jnp.where(state.valid[:, None], slots, 0.0)
        return jnp.concatenate([head, slots.ravel()]).astype(jnp.float32)


__all__ = ["Exogenous", "RunwayEnv", "RunwayState"]
