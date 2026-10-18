"""The corridor env in JAX: the Python reference (`sim`) as pure array
functions, composed from the train and bus envs.

The state is a `TrainState`, a `BusState` and the transferred riders'
credit ends. A step is the train env's step, the transfers (a scatter-add
into the bus stops' waiting counts), then the bus env's step: the coupling
lives between the two, so each env is used unchanged. Given the same
`Trip` and `Exogenous` draws (and jax_enable_x64) trajectories match
`sim.CorridorSim` exactly (tests/test_corridor.py).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from transit_bench.core.types import TimeStep
from transit_bench.envs.bus.jax_env import BusEnv, BusState
from transit_bench.envs.bus.sim import GO as BUS_GO
from transit_bench.envs.bus.sim import HEAD_FIELDS as BUS_HEAD
from transit_bench.envs.bus.sim import HOLD as BUS_HOLD
from transit_bench.envs.bus.sim import STOP_FIELDS
from transit_bench.envs.corridor.sim import (
    COUPLING_FIELDS,
    PRESETS,
    CorridorConfig,
    Exogenous,
    last_platform,
)
from transit_bench.envs.train.jax_env import TrainEnv, TrainState
from transit_bench.envs.train.sim import (
    BLOCK_FIELDS,
    DELAY_SCALE,
    LOOKAHEAD,
    TERMINATE,
)
from transit_bench.envs.train.sim import HEAD_FIELDS as TRAIN_HEAD

INFO_INT = ("terminated", "deadlocked", "transferred", "delivered", "departures", "bunched")
INFO_FLOAT = ("rail_cost", "bus_cost", "credit")


class CorridorState(NamedTuple):
    train: TrainState
    bus: BusState
    # (trains, L) seconds: until when each transferred rider group (train,
    # call) earns its credit; 0 for none.
    credit_end: jax.Array


class CorridorEnv:
    """A train line and a bus route along it, as pure JAX functions.

    Agents are the `n_trains` trains, then the buses. Each picks one of the
    train env's actions (GO, WAIT, TERMINATE); a bus reads 0 as GO and
    anything else as HOLD. The observation is one block per agent
    (`per_agent_obs`), as `sim.CorridorSim.observe`.
    """

    def __init__(self, config=None, max_steps=360, reward_shaping=False):
        if isinstance(config, str):
            config = PRESETS[config]
        self.config = c = config if config is not None else CorridorConfig()
        self.reward_shaping = reward_shaping  # the reward is dense; no shapings
        self.max_steps = max_steps
        self.train_env = TrainEnv(c.train, max_steps)
        self.bus_env = BusEnv(c.bus_config(), max_steps)
        self.n_trains = T = self.train_env.n_trains
        self.n_buses = B = c.n_buses
        self.n_agents = T + B
        self.n_actions = self.train_env.n_actions
        self.per_agent_obs = True
        self.observation_nvec = None
        self.train_width = TRAIN_HEAD + BLOCK_FIELDS * LOOKAHEAD
        self.bus_width = BUS_HEAD + STOP_FIELDS * c.n_stops
        self.width = max(self.train_width, self.bus_width)
        self.block = 1 + self.width + COUPLING_FIELDS
        self.observation_size = self.n_agents * self.block
        infra = self.train_env.infra
        station = infra.station_of[np.maximum(infra.route, 0)]
        stop_of = c.stop_of()
        # (trains, L): the bus stop at each route position (0 off platforms).
        self.stops = np.where(station >= 0, stop_of[infra.direction[:, None], station], 0)
        self.calls = infra.dwell > 0
        self.platform = last_platform(infra)

    def heuristic(self):
        from transit_bench.envs.corridor.heuristic import BusAware

        return BusAware(self)

    # -- state ---------------------------------------------------------------

    def _float(self):
        return jnp.zeros(()).dtype

    def _info(self, **values):
        f = self._float()
        info = {k: jnp.zeros((), jnp.int32) for k in INFO_INT}
        info |= {k: jnp.zeros((), f) for k in INFO_FLOAT}
        for k, v in values.items():
            info[k] = jnp.asarray(v).astype(info[k].dtype)
        return info

    def reset(self, key, trip=None):
        k_train, k_bus = jax.random.split(key)
        train, _ = self.train_env.reset(k_train, trip)
        bus, _ = self.bus_env.reset(k_bus)
        credit = jnp.zeros(self.train_env.infra.route.shape, self._float())
        state = CorridorState(train, bus, credit)
        false = jnp.zeros((), bool)
        zero = jnp.zeros((), self._float())
        return state, TimeStep(self.observe(state), zero, false, false, self._info())

    # -- coupling ------------------------------------------------------------

    def origin(self, state):
        """(trains,) route position of each train's station now
        (`sim.CorridorSim.origin`)."""
        L = self.platform.shape[1]
        p = jnp.clip(state.train.pos, 0, L - 1)
        return jnp.asarray(self.platform, jnp.int32)[jnp.arange(self.n_trains), p]

    def onboard_calls(self, state):
        calls_from = jnp.asarray(self.train_env.infra.calls_from, jnp.int32)
        rows = jnp.arange(self.n_trains)
        index = jnp.clip(jnp.maximum(state.train.pos, 0) + 1, 0, calls_from.shape[1] - 1)
        left = calls_from[rows, index]
        return jnp.where(self.train_env.finished(state.train), 0, left)

    def bus_wait(self, state, stop):
        """Elementwise over `stop`: nominal seconds until the next working
        bus reaches it."""
        env = self.bus_env
        p = env.positions(state.bus)
        target = jnp.asarray(env.offset, p.dtype)[stop]
        gaps = jnp.mod(target[..., None] - p, env.length)
        gaps = jnp.where(state.bus.failure.down, env.length, gaps)
        return gaps.min(axis=-1)

    def transfer_cost(self, state):
        """(trains,) seconds, as `sim.CorridorSim.transfer_cost`."""
        env, infra = self.bus_env, self.train_env.infra
        f = self._float()
        T, L = infra.route.shape
        rows = jnp.arange(T)
        p = self.origin(state)
        stops = jnp.asarray(self.stops, jnp.int32)
        o = stops[rows, p]
        wait = self.bus_wait(state, o)
        offset = jnp.asarray(env.offset, f)
        ride = jnp.mod(offset[stops] - offset[o][:, None], env.length)
        sched = jnp.asarray(infra.sched[:, :L], f)
        rail = sched - sched[rows, p][:, None]
        mask = (jnp.arange(L)[None, :] > p[:, None]) & jnp.asarray(self.calls)
        total = jnp.where(mask, wait[:, None] + ride - rail, 0.0).sum(axis=1)
        n = mask.sum(axis=1)
        cost = jnp.maximum(0.0, total / jnp.maximum(n, 1))
        return jnp.where(self.train_env.finished(state.train) | (n == 0), 0.0, cost)

    def active_agents(self, state):
        return jnp.concatenate(
            [self.train_env.active_agents(state.train), self.bus_env.active_agents(state.bus)]
        )

    # -- dynamics ------------------------------------------------------------

    def step(self, state, actions, exo=None):
        """One tick: (state, TimeStep). No auto-reset (see `core.vec`)."""
        c, tenv = self.config, self.train_env
        infra, dt, r = tenv.infra, c.train.step_seconds, c.riders_per_call
        f = self._float()
        T, L = infra.route.shape
        rows = jnp.arange(T)
        actions = jnp.asarray(actions, jnp.int32)
        train_exo, bus_exo = (None, None) if exo is None else exo
        now = (state.train.t * dt).astype(f)
        before = tenv.projected_delay(state.train, now)
        weight = r * self.onboard_calls(state)
        ends = tenv.can_terminate(state.train) & (actions[:T] == TERMINATE)
        origin = self.origin(state)
        train, tts = tenv.step(state.train, actions[:T], train_exo)
        later = (train.t * dt).astype(f)
        change = tenv.projected_delay(train, later) - before
        rail = jnp.where(ends, 0.0, weight * change).sum()
        # Transfers: the riders of each terminated train, per later call.
        stops = jnp.asarray(self.stops, jnp.int32)
        mask = ends[:, None] & (jnp.arange(L)[None, :] > origin[:, None]) & jnp.asarray(self.calls)
        o = jnp.broadcast_to(stops[rows, origin][:, None], (T, L))
        S = c.n_stops
        moved = jnp.zeros((S, S), jnp.int32).at[o, stops].add(jnp.where(mask, r, 0))
        sched = jnp.asarray(infra.sched[:, :L], f)
        credit_end = jnp.where(mask, sched + before[:, None], state.credit_end)
        bus = state.bus._replace(waiting=state.bus.waiting + moved)
        bus_actions = jnp.where(actions[T:] == 0, BUS_GO, BUS_HOLD).astype(jnp.int32)
        bus, bts = self.bus_env.step(bus, bus_actions, bus_exo)
        left = jnp.clip(credit_end - now, 0.0, dt)
        credit = r * jnp.where(credit_end > 0, left, 0.0).sum()
        bus_cost = (bts.info["waiting"] + bts.info["onboard"]).astype(f) * dt
        reward = -(rail + bus_cost - credit) / 60
        state = CorridorState(train, bus, credit_end)
        info = self._info(
            terminated=tts.info["terminated"],
            deadlocked=tts.info["deadlocked"],
            transferred=moved.sum(),
            delivered=bts.info["delivered"],
            departures=bts.info["departures"],
            bunched=bts.info["bunched"],
            rail_cost=rail / 60,
            bus_cost=bus_cost / 60,
            credit=credit / 60,
        )
        false = jnp.zeros((), bool)
        return state, TimeStep(self.observe(state), reward, false, bts.truncated, info)

    def exogenous(self, exo):
        """A `sim.Exogenous` from the Python reference, as arrays."""
        return Exogenous(exo.train, exo.bus)

    # -- observation ---------------------------------------------------------

    def observe(self, state):
        T, B = self.n_trains, self.n_buses
        trains = self.train_env.observe(state.train).reshape(T, -1)
        buses = self.bus_env.observe(state.bus).reshape(B, -1)
        f = trains.dtype
        done = self.train_env.finished(state.train)
        stops = jnp.asarray(self.stops, jnp.int32)[jnp.arange(T), self.origin(state)]
        wait = jnp.where(done, 0.0, self.bus_wait(state, stops))
        coupling = jnp.stack(
            [
                jnp.minimum(wait / DELAY_SCALE, 1.0),
                jnp.minimum(self.transfer_cost(state) / DELAY_SCALE, 1.0),
            ],
            axis=1,
        ).astype(f)
        train_rows = jnp.concatenate(
            [
                jnp.ones((T, 1), f),
                trains,
                jnp.zeros((T, self.width - self.train_width), f),
                coupling,
            ],
            axis=1,
        )
        bus_rows = jnp.concatenate(
            [
                jnp.zeros((B, 1), f),
                buses.astype(f),
                jnp.zeros((B, self.width - self.bus_width + COUPLING_FIELDS), f),
            ],
            axis=1,
        )
        return jnp.concatenate([train_rows, bus_rows]).ravel()
