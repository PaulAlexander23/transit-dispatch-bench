"""The bus env in JAX: the Python reference (`sim`) as pure array functions.

Same dynamics, observation and reward as `sim.BusSim`; given the same
`Exogenous` draws (and jax_enable_x64) trajectories match exactly
(tests/test_bus.py). What changes:

- Arrivals are drawn with the core's loop-free Poisson OD draw
  (`core.demand.draw_od`) into a fixed number of slots, then counted per
  (origin, destination); the counts match NumPy's per-cell Poisson draw in
  distribution.
- Buses still act one after another (a lower-numbered bus boards first),
  through a loop over buses that is unrolled when traced. Boarding "nearest
  destination first" is a cumulative sum over the stop's row of waiting
  counts, rolled to start after the stop.
- Edges carry a `core.network.EdgeState` (open, speed) in the state, the
  hook for disruptions.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from transit_bench.core import demand
from transit_bench.core.network import EdgeState, nominal
from transit_bench.core.types import TimeStep
from transit_bench.envs.bus.sim import (
    _PEAK_KEYFRAMES,
    GO,
    HEAD_FIELDS,
    HOLD,
    NEVER,
    PRESETS,
    STOP_FIELDS,
    BusConfig,
    Exogenous,
    lognormal_params,
)

INFO_INT = ("delivered", "boarded", "departures", "bunched", "held", "waiting", "onboard")
INFO_FLOAT = ("gap_sum", "gap_sq_sum")


class BusState(NamedTuple):
    stop: jax.Array  # (buses,) int: the stop it is at, or the edge it is on
    at_stop: jax.Array  # (buses,) bool
    remaining: jax.Array  # (buses,) int: ticks left on the current edge
    trip: jax.Array  # (buses,) int: ticks the current edge takes in all
    hold: jax.Array  # (buses,) int: ticks held at the current stop
    onboard: jax.Array  # (buses, stops) int: passengers per destination
    waiting: jax.Array  # (stops, stops) int: passengers per origin, destination
    last_depart: jax.Array  # (stops,) float: seconds, NEVER if no bus has left yet
    edges: EdgeState
    t: jax.Array  # () int
    key: jax.Array


class BusEnv:
    """Headway control on a bus loop, as pure JAX functions.

    `n_agents` buses each choose GO (0) or HOLD (1); only deciding buses'
    choices take effect. The observation is one block per bus, centred on
    it (`per_agent_obs`), as `sim.BusSim.observe`.
    """

    def __init__(self, config=None, max_steps=360, reward_shaping=False):
        if isinstance(config, str):
            config = PRESETS[config]
        self.config = c = config if config is not None else BusConfig()
        # No shapings yet: the reward (minus passenger-minutes) is dense.
        # Accepted for the common env interface; any value trains unshaped.
        self.reward_shaping = reward_shaping
        self.max_steps = max_steps
        self.n_agents = c.n_buses
        self.n_actions = 2
        self.per_agent_obs = True
        self.observation_nvec = None
        self.observation_size = c.n_buses * (HEAD_FIELDS + STOP_FIELDS * c.n_stops)
        self.network = c.network()
        self.travel_time = self.network.travel_time
        self.offset = np.concatenate([[0.0], np.cumsum(self.travel_time)[:-1]])
        self.length = float(self.travel_time.sum())
        self.headway = c.headway
        peak = max(v for _, v in _PEAK_KEYFRAMES) if c.demand == "peak" else 1.0
        mean_total = c.n_stops * c.arrival_rate * c.step_seconds * peak
        self.arrival_slots = demand.arrival_slots(mean_total)
        self.max_alight = int(c.step_seconds // c.alight_seconds)

    def heuristic(self):
        from transit_bench.envs.bus.heuristic import HeadwayHolding

        return HeadwayHolding(self)

    # -- state ---------------------------------------------------------------

    def _float(self):
        return jnp.zeros(()).dtype

    def initial_state(self, key):
        c = self.config
        f = self._float()
        buses, stops = c.n_buses, c.n_stops
        return BusState(
            stop=jnp.array([b * stops // buses for b in range(buses)], jnp.int32),
            at_stop=jnp.ones(buses, bool),
            remaining=jnp.zeros(buses, jnp.int32),
            trip=jnp.ones(buses, jnp.int32),
            hold=jnp.zeros(buses, jnp.int32),
            onboard=jnp.zeros((buses, stops), jnp.int32),
            waiting=jnp.zeros((stops, stops), jnp.int32),
            last_depart=jnp.full(stops, NEVER, f),
            edges=nominal(self.network, f),
            t=jnp.zeros((), jnp.int32),
            key=key,
        )

    def _info(self, **values):
        f = self._float()
        info = {k: jnp.zeros((), jnp.int32) for k in INFO_INT}
        info |= {k: jnp.zeros((), f) for k in INFO_FLOAT}
        for k, v in values.items():
            info[k] = v.astype(info[k].dtype)
        return info

    def reset(self, key):
        state = self.initial_state(key)
        f = self._float()
        false = jnp.zeros((), bool)
        ts = TimeStep(self.observe(state), jnp.zeros((), f), false, false, self._info())
        return state, ts

    # -- randomness ----------------------------------------------------------

    def demand_factor(self, episode_fraction):
        if self.config.demand == "flat":
            return jnp.ones(())
        times, values = zip(*_PEAK_KEYFRAMES, strict=True)
        t = jnp.clip(episode_fraction, 0.0, 1.0)
        return jnp.interp(t, jnp.array(times), jnp.array(values))

    def draw(self, key, episode_fraction):
        """This step's `Exogenous` draws, as `sim.BusSim.draw` in distribution."""
        c = self.config
        n = c.n_stops
        f = self._float()
        k_arrive, k_noise = jax.random.split(key)
        per_stop = c.arrival_rate * c.step_seconds * self.demand_factor(episode_fraction)
        rates = jnp.full(n, per_stop, f)
        joint = 1.0 - jnp.eye(n, dtype=f)  # uniform over the other stops
        origin, dest, used, _ = demand.draw_od(k_arrive, rates, joint, self.arrival_slots)
        arrivals = jnp.zeros((n, n), jnp.int32).at[origin, dest].add(used.astype(jnp.int32))
        if c.travel_cv > 0:
            mu, sigma = lognormal_params(c.travel_cv)
            noise = jnp.exp(mu + sigma * jax.random.normal(k_noise, (c.n_buses,), f))
        else:
            noise = jnp.ones(c.n_buses, f)
        return Exogenous(arrivals, noise)

    # -- dynamics ------------------------------------------------------------

    def deciding(self, state):
        """(buses,) bool: at a stop with nobody to drop off or pick up."""
        c = self.config
        buses = jnp.arange(c.n_buses)
        alightable = state.onboard[buses, state.stop]
        space = c.capacity - state.onboard.sum(axis=1) + alightable
        boardable = jnp.minimum(space, state.waiting[state.stop].sum(axis=1))
        return state.at_stop & (alightable == 0) & (boardable == 0)

    def active_agents(self, state):
        """Buses whose action takes effect this step: the deciding ones."""
        return self.deciding(state)

    def _bus(self, state, b, action, noise, now, info):
        """One bus's turn, as the body of `sim.BusSim.step`'s loop."""
        c = self.config
        n = c.n_stops
        f = state.last_depart.dtype
        at = state.at_stop[b]
        s = state.stop[b]

        # On an edge: count down, arrive at the next stop when it runs out.
        rem = state.remaining[b] - 1
        arrive = rem <= 0
        moved_stop = jnp.where(arrive, (s + 1) % n, s)
        moved_at = arrive
        moved_rem = jnp.where(arrive, 0, rem)

        # At a stop.
        row = state.onboard[b]
        alightable = row[s]
        space = c.capacity - row.sum(dtype=jnp.int32) + alightable
        boardable = jnp.minimum(space, state.waiting[s].sum(dtype=jnp.int32))
        deciding = at & (alightable == 0) & (boardable == 0)
        go = deciding & (action == GO) & state.edges.open[s]
        held = deciding & ~go
        work = at & ~deciding

        seconds = jnp.asarray(self.travel_time, f)[s] * state.edges.speed[s] * noise
        ticks = jnp.maximum(1, jnp.ceil(seconds / c.step_seconds - 1e-9).astype(jnp.int32))
        has_last = state.last_depart[s] > NEVER
        gap = now - state.last_depart[s]
        counted = go & has_last

        n_alight = jnp.minimum(alightable, self.max_alight)
        budget = c.step_seconds - n_alight * c.alight_seconds
        row_after = row.at[s].add(-n_alight)
        space_now = c.capacity - row_after.sum(dtype=jnp.int32)
        max_board = jnp.floor_divide(budget, c.board_seconds).astype(jnp.int32)
        n_waiting = state.waiting[s].sum(dtype=jnp.int32)
        n_board = jnp.minimum(jnp.minimum(space_now, n_waiting), max_board)
        order = (s + jnp.arange(1, n)) % n  # nearest destination first
        w = state.waiting[s, order]
        before = jnp.cumsum(w, dtype=jnp.int32) - w
        take = jnp.clip(n_board - before, 0, w)
        boarded_row = row_after.at[order].add(take)
        waiting_row = state.waiting[s].at[order].add(-take)
        n_alight = jnp.where(work, n_alight, 0)
        n_board = jnp.where(work, n_board, 0)

        state = state._replace(
            stop=state.stop.at[b].set(jnp.where(at, s, moved_stop)),
            at_stop=state.at_stop.at[b].set(jnp.where(at, ~go, moved_at)),
            remaining=state.remaining.at[b].set(jnp.where(go, ticks, jnp.where(at, 0, moved_rem))),
            trip=state.trip.at[b].set(jnp.where(go, ticks, state.trip[b])),
            hold=state.hold.at[b].set(jnp.where(go, 0, state.hold[b] + held)),
            onboard=state.onboard.at[b].set(jnp.where(work, boarded_row, row)),
            waiting=state.waiting.at[s].set(jnp.where(work, waiting_row, state.waiting[s])),
            last_depart=state.last_depart.at[s].set(jnp.where(go, now, state.last_depart[s])),
        )
        info = dict(info)
        info["delivered"] += n_alight
        info["boarded"] += n_board
        info["departures"] += go
        info["held"] += held
        info["gap_sum"] += jnp.where(counted, gap, 0.0)
        info["gap_sq_sum"] += jnp.where(counted, gap * gap, 0.0)
        info["bunched"] += counted & (gap < 0.25 * self.headway)
        return state, info

    def step(self, state, actions, exo=None):
        """One tick: (state, TimeStep). No auto-reset (see `core.vec`)."""
        c = self.config
        if exo is None:
            key, sub = jax.random.split(state.key)
            state = state._replace(key=key)
            exo = self.draw(sub, state.t / self.max_steps)
        state = state._replace(waiting=state.waiting + jnp.asarray(exo.arrivals, jnp.int32))
        now = (state.t * c.step_seconds).astype(state.last_depart.dtype)
        noise = jnp.asarray(exo.noise, state.last_depart.dtype)
        info = self._info()
        for b in range(c.n_buses):  # in bus order: lower buses board first
            state, info = self._bus(state, b, actions[b], noise[b], now, info)
        state = state._replace(t=state.t + 1)
        waiting = state.waiting.sum()
        onboard = state.onboard.sum()
        info["waiting"] = waiting.astype(jnp.int32)
        info["onboard"] = onboard.astype(jnp.int32)
        f = self._float()
        reward = -(waiting + onboard).astype(f) * c.step_seconds / 60 / c.n_stops
        truncated = state.t >= self.max_steps
        terminated = jnp.zeros((), bool)
        return state, TimeStep(self.observe(state), reward, terminated, truncated, info)

    # -- observation ---------------------------------------------------------

    def positions(self, state):
        f = state.last_depart.dtype
        offset = jnp.asarray(self.offset, f)[state.stop]
        progress = 1.0 - state.remaining.astype(f) / state.trip
        edge = jnp.asarray(self.travel_time, f)[state.stop]
        return jnp.where(state.at_stop, offset, offset + progress * edge)

    def observe(self, state):
        c = self.config
        h, cap, n = self.headway, c.capacity, c.n_stops
        now = state.t * c.step_seconds
        p = self.positions(state)
        gaps = jnp.mod(p[None, :] - p[:, None], self.length)  # [b, o]: o ahead of b
        eye = jnp.eye(c.n_buses, dtype=bool)
        ahead = jnp.where(eye, self.length, gaps).min(axis=1)
        behind = jnp.where(eye, self.length, gaps.T).min(axis=1)
        last = state.last_depart
        since = jnp.where(last <= NEVER, 1.0, jnp.minimum((now - last) / h, 2.0) / 2)
        f = p.dtype
        progress = jnp.where(state.at_stop, 0.0, 1.0 - state.remaining.astype(f) / state.trip)
        head = jnp.stack(
            [
                state.at_stop.astype(p.dtype),
                self.deciding(state).astype(p.dtype),
                state.onboard.sum(axis=1).astype(f) / cap,
                jnp.minimum(ahead / h, 2.0) / 2,
                jnp.minimum(behind / h, 2.0) / 2,
                jnp.minimum(state.hold.astype(f) * c.step_seconds / h, 1.0),
                progress,
            ],
            axis=1,
        ).astype(jnp.float32)
        ref = jnp.where(state.at_stop, state.stop, (state.stop + 1) % n)
        q = (ref[:, None] + jnp.arange(n)) % n  # (buses, stops)
        waiting = state.waiting.sum(axis=1)
        buses = jnp.arange(c.n_buses)[:, None]
        stops = jnp.stack(
            [
                jnp.minimum(waiting[q].astype(f) / cap, 1.0),
                jnp.minimum(state.onboard[buses, q].astype(f) / cap, 1.0),
                since[q],
            ],
            axis=2,
        ).astype(jnp.float32)
        return jnp.concatenate([head, stops.reshape(c.n_buses, -1)], axis=1).ravel()


__all__ = ["GO", "HOLD", "BusEnv", "BusState"]
