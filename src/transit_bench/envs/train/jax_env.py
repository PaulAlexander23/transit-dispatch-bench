"""The train env in JAX: the Python reference (`sim`) as pure array functions.

Same dynamics, observation and reward as `sim.TrainSim`; given the same
`Trip` and `Exogenous` draws (and jax_enable_x64) trajectories match exactly
(tests/test_train.py). Trains act one after another in index order, through
a loop over trains that is unrolled when traced; everything else is
vectorised. Blocks carry a `core.network.EdgeState` (open, speed) in the
state, the hook for disruptions.

Deadlock detection has no data-dependent loop: the wait-for relation is an
N×N boolean matrix A (train i waits for a block train j holds), and
squaring R ← R ∨ R·R doubles the length of the paths R covers, so after
⌈log₂ N⌉ squarings R is reachability. Train i is deadlocked when it can
reach itself, R[i, i].
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from transit_bench.core import disruption
from transit_bench.core.network import EdgeState
from transit_bench.core.types import TimeStep
from transit_bench.envs.train.infra import PRESETS, TrainConfig, build
from transit_bench.envs.train.sim import (
    BLOCK_FIELDS,
    DELAY_SCALE,
    DIVERT,
    GO,
    HEAD_FIELDS,
    LOOKAHEAD,
    NOT_STARTED,
    TERMINATE,
    WAIT,
    Exogenous,
    Trip,
    lognormal_params,
)

INFO_INT = (
    "entered",
    "arrived",
    "held",
    "blocked",
    "terminated",
    "missed",
    "diverted",
    "deadlocked",
    "finished",
)


class TrainState(NamedTuple):
    pos: jax.Array  # (trains,) int: index on the route, -1 before it, route_len after
    relief: jax.Array  # (trains, L) bool: on the relief track at that route position
    remaining: jax.Array  # (trains,) int: ticks left in the current block
    arrival_delay: jax.Array  # (trains,) float: seconds late at the end, once arrived
    occupant: jax.Array  # (blocks,) int: the train holding each block, or -1
    blocks: EdgeState  # per block: open, speed factor
    disruption: disruption.DisruptionState  # per block: down, elapsed, remaining
    potential: jax.Array  # () float: total projected delay after the last step
    trip: Trip  # (trains, L) each: running-time noise and incident seconds, from reset
    t: jax.Array  # () int
    key: jax.Array


class TrainEnv:
    """Train dispatching on blocks of track, as pure JAX functions.

    `n_agents` trains each choose GO (0) or WAIT (1), TERMINATE (2) if
    the config has a `terminate_penalty`, and DIVERT (3) if it has relief
    tracks (then there are 4 actions, and TERMINATE without a penalty
    holds, as WAIT does); only deciding trains' choices, and terminating
    trains', take effect (`active_agents`). The observation is one block of
    features per train (`per_agent_obs`), as `sim.TrainSim.observe`.

    With `shield=True` a safety shield vetoes any GO that route setting
    (`heuristic.RouteSetting`) would not allow: the train waits instead, and
    it does not count as active, since it had no choice. Exploration can
    then never walk into the deadlocks route setting avoids, and "always
    GO" under the shield is route setting itself. Without the shield the
    env matches the Python reference step for step.
    """

    def __init__(self, config=None, max_steps=360, reward_shaping=False, shield=False):
        if isinstance(config, str):
            config = PRESETS[config]
        self.config = c = config if config is not None else TrainConfig()
        # The reward is already dense (projected lateness); no shapings yet.
        self.reward_shaping = reward_shaping
        self.shield = shield
        self.max_steps = max_steps
        self.infra = infra = build(c)
        self.n_trains = T = len(infra.route)
        self.n_agents = T
        self.has_relief = bool((infra.alt_route >= 0).any())
        self.n_actions = 4 if self.has_relief else 3 if c.terminate_penalty > 0 else 2
        # span[i, q, p]: diverting into route position q puts train i on the
        # relief track at p (the rest of that section).
        T_, L = infra.route.shape
        span = np.zeros((T_, L, L), bool)
        for i in range(T_):
            for q in range(L):
                p = q
                while p < L and infra.alt_route[i, p] >= 0:
                    span[i, q, p] = True
                    p += 1
        self.span = span
        self.per_agent_obs = True
        self.observation_nvec = None
        self.observation_size = T * (HEAD_FIELDS + BLOCK_FIELDS * LOOKAHEAD)
        self.n_blocks = infra.n_blocks
        self.n_sections = int(infra.section_of.max()) + 1
        self.n_squarings = max(1, math.ceil(math.log2(T)))

    def heuristic(self):
        """The strongest rules combined (`sim.heuristic`): route setting;
        `GiveWayToExpress` where there are expresses; `Reroute` where there
        are relief tracks; `TerminateWhenStuck` where the config allows
        TERMINATE."""
        from transit_bench.envs.train.heuristic import (
            GiveWayToExpress,
            Reroute,
            RouteSetting,
            TerminateWhenStuck,
        )

        rule = RouteSetting(self)
        if self.infra.express.any():
            rule = GiveWayToExpress(self, rule)
        elif self.has_relief:
            rule = Reroute(self, rule)
        if self.config.terminate_penalty > 0:
            rule = TerminateWhenStuck(self, rule)
        return rule

    # -- state ---------------------------------------------------------------

    def _float(self):
        return jnp.zeros(()).dtype

    def _arrays(self):
        """The infrastructure as JAX arrays (constants under jit)."""
        f, infra = self._float(), self.infra
        return (
            jnp.asarray(infra.route, jnp.int32),
            jnp.asarray(infra.route_len, jnp.int32),
            jnp.asarray(infra.run, f),
            jnp.asarray(infra.dwell, jnp.int32),
            jnp.asarray(infra.sched, f),
            jnp.asarray(infra.direction, jnp.int32),
        )

    def initial_state(self, key, trip=None):
        """The state at t = 0; draws the `Trip` from `key` unless given."""
        f = self._float()
        T, B = self.n_trains, self.n_blocks
        if trip is None:
            key, sub = jax.random.split(key)
            trip = self.draw_trip(sub)
        trip = Trip(jnp.asarray(trip.noise, f), jnp.asarray(trip.incident, f))
        return TrainState(
            pos=jnp.full(T, NOT_STARTED, jnp.int32),
            relief=jnp.zeros(self.infra.route.shape, bool),
            remaining=jnp.zeros(T, jnp.int32),
            arrival_delay=jnp.zeros(T, f),
            occupant=jnp.full(B, -1, jnp.int32),
            blocks=EdgeState(open=jnp.ones(B, bool), speed=jnp.ones(B, f)),
            disruption=disruption.initial(B, f),
            potential=jnp.zeros((), f),
            trip=trip,
            t=jnp.zeros((), jnp.int32),
            key=key,
        )

    def _info(self, **values):
        info = {k: jnp.zeros((), jnp.int32) for k in INFO_INT}
        info["delay"] = jnp.zeros((), self._float())
        for k, v in values.items():
            info[k] = v.astype(info[k].dtype)
        return info

    def reset(self, key, trip=None):
        state = self.initial_state(key, trip)
        false = jnp.zeros((), bool)
        zero = jnp.zeros((), self._float())
        return state, TimeStep(self.observe(state), zero, false, false, self._info())

    def draw_trip(self, key):
        """The episode's `Trip`, as `sim.TrainSim.draw_trip` in distribution."""
        c, f = self.config, self._float()
        shape = self.infra.route.shape
        k_noise, k_hit, k_size = jax.random.split(key, 3)
        if c.running_cv > 0:
            mu, sigma = lognormal_params(c.running_cv)
            noise = jnp.exp(mu + sigma * jax.random.normal(k_noise, shape, f))
        else:
            noise = jnp.ones(shape, f)
        hit = jax.random.uniform(k_hit, shape, f) < c.incident_probability
        size = jax.random.exponential(k_size, shape, f) * c.incident_seconds
        return Trip(noise, jnp.where(hit, size, 0.0))

    def draw(self, key):
        """This step's `Exogenous` draws, as `sim.TrainSim.draw` in distribution."""
        d = disruption.draw(self.config.disruption, key, self.n_blocks, self._float())
        return Exogenous(d.onset, d.duration)

    # -- queries -------------------------------------------------------------

    def finished(self, state):
        return state.pos >= jnp.asarray(self.infra.route_len)

    def route_now(self, state):
        """(trains, L) each train's route, on the track it takes."""
        route = jnp.asarray(self.infra.route, jnp.int32)
        return jnp.where(state.relief, jnp.asarray(self.infra.alt_route, jnp.int32), route)

    def next_block(self, state):
        """(trains,) the block each train would enter next (on the main track
        at a junction), or -1 from its last."""
        route_len = jnp.asarray(self.infra.route_len, jnp.int32)
        route = self.route_now(state)
        nxt = state.pos + 1
        trains = jnp.arange(self.n_trains)
        block = route[trains, jnp.clip(nxt, 0, route.shape[1] - 1)]
        return jnp.where(nxt < route_len, block, -1)

    def ready(self, state):
        sched = self._arrays()[4]
        now = state.t * self.config.step_seconds
        started = state.pos != NOT_STARTED
        due = jnp.where(started, state.remaining == 0, now >= sched[:, 0])
        return ~self.finished(state) & due

    def alt_block(self, state):
        """(trains,) at a junction, the relief block each train could enter
        instead; else -1."""
        alt = jnp.asarray(self.infra.alt_route, jnp.int32)
        L = alt.shape[1]
        trains = jnp.arange(self.n_trains)
        nxt = state.pos + 1
        a = alt[trains, jnp.clip(nxt, 0, L - 1)]
        here = alt[trains, jnp.clip(state.pos, 0, L - 1)]
        at = (nxt < jnp.asarray(self.infra.route_len)) & ((state.pos == NOT_STARTED) | (here < 0))
        return jnp.where(at & ~self.finished(state) & (a >= 0), a, -1)

    def free(self, state, b):
        """Elementwise: block b exists, is unoccupied and open."""
        safe = jnp.maximum(b, 0)
        return (b >= 0) & (state.occupant[safe] < 0) & state.blocks.open[safe]

    def deciding(self, state):
        b = self.next_block(state)
        either = self.free(state, b) | self.free(state, self.alt_block(state))
        return self.ready(state) & (b >= 0) & either

    def can_terminate(self, state):
        """(trains,) bool: ready, not started or on a platform, not at the
        last block (and the config allows TERMINATE)."""
        if self.config.terminate_penalty <= 0:
            return jnp.zeros(self.n_trains, bool)
        station_of = jnp.asarray(self.infra.station_of, jnp.int32)
        started = state.pos != NOT_STARTED
        on_platform = station_of[jnp.maximum(self.here(state), 0)] >= 0
        return self.ready(state) & (self.next_block(state) >= 0) & (~started | on_platform)

    def here(self, state):
        """(trains,) the block each train holds, or -1."""
        route = self.route_now(state)
        block = route[jnp.arange(self.n_trains), jnp.clip(state.pos, 0, route.shape[1] - 1)]
        return jnp.where((state.pos != NOT_STARTED) & ~self.finished(state), block, -1)

    def missed_calls(self, state):
        """(trains,) calls each train would miss if it terminated now."""
        calls_from = jnp.asarray(self.infra.calls_from, jnp.int32)
        index = jnp.clip(state.pos + 1, 0, calls_from.shape[1] - 1)
        return calls_from[jnp.arange(self.n_trains), index]

    def active_agents(self, state):
        """Trains whose action takes effect this step: the deciding ones
        (less those the shield holds), and those that can terminate."""
        deciding = self.deciding(state)
        if self.shield:
            deciding = deciding & ~self.vetoed(state)
        return deciding | self.can_terminate(state)

    def vetoed(self, state):
        """(trains,) bool: deciding trains route setting would hold."""
        from transit_bench.envs.train.heuristic import RouteSetting

        _, rule = RouteSetting(self).act(jnp.zeros((), jnp.int32), state)
        return self.deciding(state) & (rule == WAIT)

    def projected_delay(self, state, now):
        route, route_len, _, _, sched, _ = self._arrays()
        trains = jnp.arange(self.n_trains)
        started = state.pos != NOT_STARTED
        nxt = jnp.clip(state.pos + 1, 0, sched.shape[1] - 1)
        exit_time = now + state.remaining * self.config.step_seconds
        running = jnp.maximum(0.0, exit_time - sched[trains, nxt])
        waiting = jnp.maximum(0.0, now - sched[:, 0])
        delay = jnp.where(started, running, waiting)
        return jnp.where(self.finished(state), state.arrival_delay, delay)

    def deadlocked(self, state):
        """(trains,) bool: on a cycle of waits (see the module docstring)."""
        T = self.n_trains
        b = self.next_block(state)
        started = state.pos != NOT_STARTED
        waiting = started & self.ready(state) & (b >= 0)
        waiting = waiting & ~self.free(state, self.alt_block(state))  # see sim.waits_for
        holder = jnp.where(waiting, state.occupant[jnp.maximum(b, 0)], -1)
        reach = (holder[:, None] == jnp.arange(T)[None, :]) & (holder[:, None] >= 0)
        for _ in range(self.n_squarings):
            two_steps = (reach.astype(jnp.int32) @ reach.astype(jnp.int32)) > 0
            reach = reach | two_steps
        return jnp.diagonal(reach)

    # -- dynamics ------------------------------------------------------------

    def step(self, state, actions, exo=None):
        """One tick: (state, TimeStep). No auto-reset (see `core.vec`)."""
        c = self.config
        route, route_len, run, dwell, sched, _ = self._arrays()
        f = self._float()
        if exo is None:
            key, sub = jax.random.split(state.key)
            state = state._replace(key=key)
            exo = self.draw(sub)
        if self.shield:
            actions = jnp.where(self.vetoed(state), WAIT, actions)
        now = (state.t * c.step_seconds).astype(f)
        ready = self.ready(state)
        deciding = self.deciding(state)
        terminate = self.can_terminate(state) & (actions == TERMINATE)
        frozen = self.projected_delay(state, now) + 60 * c.terminate_penalty * self.missed_calls(
            state
        )
        nxt_block = self.next_block(state)
        alt_block = self.alt_block(state)
        free_main = self.free(state, nxt_block)
        free_alt = self.free(state, alt_block)
        alt_route = jnp.asarray(self.infra.alt_route, jnp.int32)
        alt_run = jnp.asarray(self.infra.alt_run, f)
        span = jnp.asarray(self.span)
        B = self.n_blocks
        claimed = jnp.zeros(B, bool)
        keys = ("entered", "arrived", "held", "blocked", "terminated", "missed", "diverted")
        counts = {k: jnp.zeros((), jnp.int32) for k in keys}
        missed = self.missed_calls(state)
        pos, remaining = state.pos, state.remaining
        occupant, arrival_delay, relief = state.occupant, state.arrival_delay, state.relief
        L = route.shape[1]
        for i in range(self.n_trains):  # in index order: earlier trains first
            divert = (actions[i] == DIVERT) & (alt_block[i] >= 0)
            b = jnp.where(divert, alt_block[i], nxt_block[i])
            safe = jnp.maximum(b, 0)
            p = pos[i]
            cp = jnp.clip(p, 0, L - 1)
            here = jnp.where(relief[i, cp], alt_route[i, cp], route[i, cp])
            stop = terminate[i]
            leave = ready[i] & (nxt_block[i] < 0)
            chosen_free = jnp.where(divert, free_alt[i], free_main[i])
            can = ready[i] & (b >= 0) & deciding[i] & chosen_free & ~claimed[safe] & ~stop
            moving = (actions[i] == GO) | divert
            enter = can & moving
            held = can & ~moving
            blocked = ready[i] & (nxt_block[i] >= 0) & ~can & ~stop
            vacate = leave | ((stop | enter) & (p != NOT_STARTED))
            occupant = occupant.at[jnp.where(vacate, here, B)].set(-1, mode="drop")
            occupant = occupant.at[jnp.where(enter, b, B)].set(i, mode="drop")
            claimed = claimed.at[jnp.where(enter, b, B)].set(True, mode="drop")
            new_p = jnp.clip(p + 1, 0, L - 1)
            relief = relief.at[i].set(relief[i] | (enter & divert & span[i, new_p]))
            noise, incident = state.trip.noise[i, new_p], state.trip.incident[i, new_p]
            base = jnp.where(relief[i, new_p], alt_run[i, new_p], run[i, new_p])
            seconds = base * state.blocks.speed[safe] * noise + incident
            ticks = jnp.maximum(1, jnp.ceil(seconds / c.step_seconds - 1e-9).astype(jnp.int32))
            remaining = remaining.at[i].set(jnp.where(enter, ticks + dwell[i, new_p], remaining[i]))
            late = jnp.maximum(0.0, now - sched[i, route_len[i]])
            late = jnp.where(stop, frozen[i], late)
            arrival_delay = arrival_delay.at[i].set(jnp.where(leave | stop, late, arrival_delay[i]))
            pos = pos.at[i].set(jnp.where(leave | stop, route_len[i], jnp.where(enter, p + 1, p)))
            counts["terminated"] += stop
            counts["missed"] += jnp.where(stop, missed[i], 0)
            counts["entered"] += enter
            counts["diverted"] += enter & divert
            counts["arrived"] += leave
            counts["held"] += held
            counts["blocked"] += blocked
        state = state._replace(
            pos=pos,
            relief=relief,
            remaining=jnp.maximum(remaining - 1, 0),
            arrival_delay=arrival_delay,
            occupant=occupant,
            t=state.t + 1,
        )
        state = self._advance_disruption(state, exo, now)
        later = (state.t * c.step_seconds).astype(f)
        total = self.projected_delay(state, later).sum()
        reward = -(total - state.potential) / 60
        state = state._replace(potential=total)
        finished = self.finished(state).sum(dtype=jnp.int32)
        info = self._info(
            **counts,
            delay=total,
            deadlocked=self.deadlocked(state).sum(dtype=jnp.int32),
            finished=finished,
        )
        terminated = finished == self.n_trains
        truncated = state.t >= self.max_steps
        return state, TimeStep(self.observe(state), reward, terminated, truncated, info)

    def _advance_disruption(self, state, exo, now):
        """As sim.TrainSim._advance_disruption: at the end of the tick."""
        c = self.config.disruption
        if c.mode == "none":
            return state
        eligible = jnp.asarray(self.infra.section_of >= 0)
        draw = disruption.DisruptionDraw(exo.onset, exo.duration)
        d = disruption.step(c, state.disruption, draw, eligible, now, self.config.step_seconds)
        is_open, speed = disruption.edge_effect(c, d)
        return state._replace(disruption=d, blocks=EdgeState(open=is_open, speed=speed))

    # -- observation ---------------------------------------------------------

    def joining(self, state):
        """(trains,) the section each train is deciding to enter from outside, or -1."""
        section_of = jnp.asarray(self.infra.section_of, jnp.int32)
        target = section_of[jnp.maximum(self.next_block(state), 0)]
        here_block = self.here(state)
        here = jnp.where(here_block >= 0, section_of[jnp.maximum(here_block, 0)], -1)
        joins = self.deciding(state) & (target >= 0) & (here != target)
        return jnp.where(joins, target, -1)

    def _section_fields(self, state):
        """For each train's next section: whether its next block enters it,
        an oncoming train inside it, and oncoming trains deciding to enter
        it before and after this one."""
        infra = self.infra
        S, T = self.n_sections, self.n_trains
        direction = jnp.asarray(infra.direction, jnp.int32)
        section_of = jnp.asarray(infra.section_of, jnp.int32)
        next_section = jnp.asarray(infra.next_section, jnp.int32)
        index = jnp.minimum(state.pos + 1, next_section.shape[1] - 1)
        target = jnp.where(self.finished(state), -1, next_section[jnp.arange(T), index])
        holder = state.occupant
        in_section = (holder >= 0) & (section_of >= 0)
        rows = jnp.where(in_section, section_of, S)
        cols = direction[jnp.maximum(holder, 0)]
        occupied = jnp.zeros((S + 1, 2), bool).at[rows, cols].max(in_section)
        safe = jnp.where(target >= 0, target, S)
        inside = occupied[safe, 1 - direction] & (target >= 0)
        joining = self.joining(state)
        rival = (
            (joining[None, :] == target[:, None])
            & (target[:, None] >= 0)
            & (direction[None, :] != direction[:, None])
        )  # [i, j]: j is an oncoming train joining i's next section
        order = jnp.arange(T)
        before = (rival & (order[None, :] < order[:, None])).any(axis=1)
        after = (rival & (order[None, :] > order[:, None])).any(axis=1)
        b = self.next_block(state)
        now_entering = (target >= 0) & (b >= 0) & (section_of[jnp.maximum(b, 0)] == target)
        return now_entering, inside, before, after

    def observe(self, state):
        c = self.config
        _, route_len, _, _, _, direction = self._arrays()
        route = self.route_now(state)
        f = self._float()
        now = (state.t * c.step_seconds).astype(f)
        started = state.pos != NOT_STARTED
        finished = self.finished(state)
        section_fields = self._section_fields(state)
        head = jnp.stack(
            [
                ~started,
                started & ~finished & (state.remaining > 0),
                self.deciding(state),
                finished,
                (state.pos + 1).astype(f) / route_len,
                jnp.minimum(state.remaining.astype(f) * c.step_seconds / DELAY_SCALE, 1.0),
                jnp.minimum(self.projected_delay(state, now) / DELAY_SCALE, 1.0),
                jnp.asarray(self.infra.express),
                direction.astype(bool),
                self.can_terminate(state),
                jnp.where(finished, 0, self.missed_calls(state)).astype(f) / c.n_stations,
                self.alt_block(state) >= 0,
                self.free(state, self.alt_block(state)),
                *section_fields,
            ],
            axis=1,
        ).astype(jnp.float32)
        p = state.pos[:, None] + jnp.arange(1, LOOKAHEAD + 1)[None, :]  # (trains, K)
        on_route = p < route_len[:, None]
        trains = jnp.arange(self.n_trains)[:, None]
        b = route[trains, jnp.clip(p, 0, route.shape[1] - 1)]
        j = state.occupant[b]
        occupied = on_route & (j >= 0)
        same = occupied & (direction[jnp.maximum(j, 0)] == direction[:, None])
        is_open = on_route & state.blocks.open[b]
        d = state.disruption
        down = on_route & d.down[b]
        elapsed = jnp.where(down, jnp.minimum(d.elapsed[b] / DELAY_SCALE, 1.0), 0.0)
        left = jnp.minimum(d.remaining[b] / DELAY_SCALE, 1.0)
        left = jnp.where(down & self.config.disruption.observe_remaining, left, 0.0)
        ahead = jnp.stack([on_route, occupied, same, is_open, elapsed, left], axis=2).astype(
            jnp.float32
        )
        return jnp.concatenate([head, ahead.reshape(self.n_trains, -1)], axis=1).ravel()


__all__ = ["DIVERT", "GO", "TERMINATE", "WAIT", "TrainEnv", "TrainState"]
