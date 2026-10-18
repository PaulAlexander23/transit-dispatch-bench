"""The corridor env's Python reference sim: a train line and a bus route
along it, coupled through one clock, one network and one passenger model.

The slow, readable ground truth that the JAX env is tested against step for
step (given the same random draws). It holds a `TrainSim` and a `BusSim`
and couples them between their steps; neither env's dynamics change.

- **Network.** The bus route is a loop of 2(n − 1) stops along the line's n
  stations, out and back: stop s is station s eastbound, stop 2(n − 1) − s
  is station s westbound (the two end stations are one stop each). Each
  edge takes the road time of a section, `bus_slowdown` x a stopping
  train's running time (`TrainConfig.road_ticks`, in seconds).
- **Clock.** One tick: the trains act, then the riders of any train that
  terminated join the bus stops, then the buses act.
- **Rail riders.** Each train leaves its origin with `riders_per_call`
  riders for each later call. Every rider still on board pays the train's
  projected lateness: each tick costs r x (calls left) x the change in the
  train's projected delay. A train that terminates (cancels, or
  short-turns at a station) costs nothing more itself. Its riders join
  the bus stop of that station in its direction, each bound for the stop
  of their own call, and pay bus passenger-minutes from then on.
- **Transfer credit.** So that a transferred rider costs its lateness, not
  its whole bus journey, it earns 1 minute a minute from the transfer
  until `credit_end`: its call's scheduled time plus the projected
  lateness already charged. Over the episode the rider costs its bus
  arrival minus its scheduled train arrival, and the credit comes per
  tick, so terminating does not look good at once and get billed later.

Reward: minus the passenger-minutes this tick: rail riders' lateness, plus
everyone waiting for or riding a bus, less the transfer credit. Agents
are the trains then the buses. Actions are the train's (GO, WAIT,
TERMINATE); for a bus, 0 is GO and anything else HOLD.
"""

from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np

from transit_bench.core.disruption import DisruptionConfig
from transit_bench.envs.bus.sim import GO as BUS_GO
from transit_bench.envs.bus.sim import HOLD as BUS_HOLD
from transit_bench.envs.bus.sim import BusConfig, BusSim, headway_holding
from transit_bench.envs.train.infra import PRESETS as TRAIN_PRESETS
from transit_bench.envs.train.infra import TrainConfig
from transit_bench.envs.train.sim import (
    DELAY_SCALE,
    TERMINATE,
    TrainSim,
    route_setting,
    terminate_when_stuck,
)

COUPLING_FIELDS = 2  # per agent: bus wait at the train's station, transfer cost


@dataclass(frozen=True)
class CorridorConfig:
    # The rail line; it needs TERMINATE (a penalty, which only the
    # bus-blind rule reads) and no bridging buses of its own. The default
    # is the `blockage` line (single track, 10-minute outages) with
    # TERMINATE at 5 minutes a call: long enough outages that ending a
    # trip can pay, short enough that whether it does depends on the bus.
    train: TrainConfig = field(
        default_factory=lambda: replace(TRAIN_PRESETS["blockage"], terminate_penalty=5.0)
    )
    n_buses: int = 4
    capacity: int = 80
    arrival_rate: float = 1 / 60  # bus passengers per second per stop
    travel_cv: float = 0.25
    demand: str = "flat"
    road_disruption: DisruptionConfig = DisruptionConfig()
    riders_per_call: int = 10

    def __post_init__(self):
        if self.train.terminate_penalty <= 0:
            raise ValueError("the corridor's trains need TERMINATE (terminate_penalty > 0)")
        if self.train.bridge_buses:
            raise ValueError("the corridor's buses are the bus route's: bridge_buses must be 0")
        if self.riders_per_call < 0:
            raise ValueError("riders_per_call must be >= 0")

    @property
    def n_stops(self):
        return 2 * (self.train.n_stations - 1)

    @property
    def road_seconds(self):
        t = self.train
        return t.bus_slowdown * t.blocks_per_section * t.section_block_seconds

    def bus_config(self):
        return BusConfig(
            n_stops=self.n_stops,
            n_buses=self.n_buses,
            capacity=self.capacity,
            travel_time=self.road_seconds,
            travel_cv=self.travel_cv,
            arrival_rate=self.arrival_rate,
            demand=self.demand,
            step_seconds=self.train.step_seconds,
            disruption=self.road_disruption,
        )

    def stop_of(self):
        """(2, stations): the bus stop for each direction and station."""
        n, S = self.train.n_stations, self.n_stops
        return np.array([list(range(n)), [(S - s) % S for s in range(n)]], dtype=int)


# `corridor`: the `blockage` line with TERMINATE, and a four-bus route at
# about the train's 10-minute headway. `busy`: twice
# the bus demand, so buses run fuller and transfers find less room.
PRESETS = {"corridor": CorridorConfig()}
PRESETS["busy"] = replace(PRESETS["corridor"], arrival_rate=1 / 30)


def last_platform(infra):
    """(trains, L): per route position, the latest position at or before it
    on a platform (the station a train between stations last left)."""
    T, L = infra.route.shape
    out = np.zeros((T, L), dtype=int)
    for i in range(T):
        last = 0
        for q in range(L):
            if infra.route[i][q] >= 0 and infra.station_of[infra.route[i][q]] >= 0:
                last = q
            out[i][q] = last
    return out


class Exogenous(NamedTuple):
    train: object  # train.sim.Exogenous
    bus: object  # bus.sim.Exogenous


class CorridorSim:
    def __init__(self, config=None, max_steps=360, rng=None):
        if isinstance(config, str):
            config = PRESETS[config]
        self.config = c = config if config is not None else CorridorConfig()
        self.max_steps = max_steps
        self.rng = rng if rng is not None else np.random.default_rng()
        self.train = TrainSim(c.train, max_steps, rng=self.rng)
        self.bus = BusSim(c.bus_config(), max_steps, rng=self.rng)
        self.stop_of = c.stop_of()
        self.platform = last_platform(self.train.infra)
        self.reset()

    @property
    def n_trains(self):
        return len(self.train.pos)

    def reset(self, seed=None, trip=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.train.rng = self.bus.rng = self.rng
        self.train.reset(trip=trip)
        self.bus.reset()
        self.credit_end = np.zeros(self.train.infra.route.shape)  # seconds, 0 for none
        self.t = 0
        return self.observe()

    def draw(self):
        return Exogenous(self.train.draw(), self.bus.draw())

    # -- coupling ------------------------------------------------------------

    def origin(self, i):
        """Route position of train i's station now: where it stands, or the
        last it left (its first before it departs). A train can only
        terminate on a platform, so there this is where it stands."""
        p = min(max(self.train.pos[i], 0), self.platform.shape[1] - 1)
        return int(self.platform[i][p])

    def onboard_calls(self, i):
        """Calls train i's riders are still bound for (0 once it is done)."""
        if self.train.finished(i):
            return 0
        return int(self.train.infra.calls_from[i][max(self.train.pos[i], 0) + 1])

    def stop_at(self, i, q):
        """The bus stop at train i's route position q, in its direction."""
        infra = self.train.infra
        station = int(infra.station_of[infra.route[i][q]])
        return int(self.stop_of[infra.direction[i]][station])

    def bus_wait(self, stop):
        """Nominal seconds until the next working bus reaches `stop`."""
        bus = self.bus
        target = bus.offset[stop]
        gaps = [
            (target - bus.position(b)) % bus.length
            for b in range(bus.config.n_buses)
            if not bus.broken(b)
        ]
        return min(gaps, default=bus.length)

    def transfer_cost(self, i):
        """Seconds: the mean extra journey time by bus over the train for
        train i's riders if it terminated now (at least 0). The wait for
        the next bus, plus the nominal ride to each call, less the
        timetabled time to it."""
        tr, infra, bus = self.train, self.train.infra, self.bus
        if tr.finished(i):
            return 0.0
        p = self.origin(i)
        o = self.stop_at(i, p)
        wait = self.bus_wait(o)
        total, n = 0.0, 0
        for q in range(p + 1, int(infra.route_len[i])):
            if infra.dwell[i][q] == 0:
                continue
            ride = (bus.offset[self.stop_at(i, q)] - bus.offset[o]) % bus.length
            total += wait + ride - (infra.sched[i][q] - infra.sched[i][p])
            n += 1
        return max(0.0, total / n) if n else 0.0

    # -- dynamics ------------------------------------------------------------

    def step(self, actions, exo=None):
        """One tick: (observation, reward, terminated, truncated, info)."""
        c, tr = self.config, self.train
        infra, dt, r = tr.infra, c.train.step_seconds, c.riders_per_call
        if exo is None:
            exo = self.draw()
        T = self.n_trains
        now = self.t * dt
        before = [tr.projected_delay(i, now) for i in range(T)]
        weight = [r * self.onboard_calls(i) for i in range(T)]
        ends = [tr.can_terminate(i) and int(actions[i]) == TERMINATE for i in range(T)]
        origin = [self.origin(i) for i in range(T)]
        _, _, _, _, tinfo = tr.step(actions[:T], exo.train)
        later = tr.t * dt
        rail = sum(
            weight[i] * (tr.projected_delay(i, later) - before[i]) for i in range(T) if not ends[i]
        )
        transferred = 0
        for i in range(T):
            if not ends[i]:
                continue
            o = self.stop_at(i, origin[i])
            for q in range(origin[i] + 1, int(infra.route_len[i])):
                if infra.dwell[i][q] > 0:
                    self.bus.waiting[o][self.stop_at(i, q)] += r
                    self.credit_end[i][q] = infra.sched[i][q] + before[i]
                    transferred += r
        bus_actions = [BUS_GO if int(a) == 0 else BUS_HOLD for a in actions[T:]]
        _, _, truncated, binfo = self.bus.step(bus_actions, exo.bus)
        left = np.clip(self.credit_end - now, 0.0, dt)
        credit = r * float(np.where(self.credit_end > 0, left, 0.0).sum())
        bus_cost = (binfo["waiting"] + binfo["onboard"]) * dt
        self.t += 1
        reward = -(rail + bus_cost - credit) / 60
        info = {
            "terminated": tinfo["terminated"],
            "deadlocked": tinfo["deadlocked"],
            "transferred": transferred,
            "delivered": binfo["delivered"],
            "departures": binfo["departures"],
            "bunched": binfo["bunched"],
            "rail_cost": rail / 60,
            "bus_cost": bus_cost / 60,
            "credit": credit / 60,
        }
        return self.observe(), reward, False, truncated, info

    # -- observation ---------------------------------------------------------

    def observe(self):
        """One block per agent, trains then buses: is a train, the agent's
        own env features (`TrainSim.observe` or `BusSim.observe`), padded
        with zeros to the longer of the two, then for trains: the wait for
        the next bus at its station in its direction, and `transfer_cost`
        (over DELAY_SCALE, capped at 1; 0 for buses and finished trains)."""
        T, B = self.n_trains, self.bus.config.n_buses
        trains = self.train.observe().reshape(T, -1)
        buses = self.bus.observe().reshape(B, -1)
        width = max(trains.shape[1], buses.shape[1])
        rows = []
        for i in range(T):
            done = self.train.finished(i)
            wait = 0.0 if done else self.bus_wait(self.stop_at(i, self.origin(i)))
            coupling = [min(wait / DELAY_SCALE, 1.0), min(self.transfer_cost(i) / DELAY_SCALE, 1.0)]
            pad = [0.0] * (width - trains.shape[1])
            rows.append([1.0, *trains[i], *pad, *coupling])
        for b in range(B):
            pad = [0.0] * (width - buses.shape[1])
            rows.append([0.0, *buses[b], *pad, 0.0, 0.0])
        return np.array(rows, dtype=np.float32).ravel()


# -- heuristics ----------------------------------------------------------------


def no_termination(sim):
    """Route setting for the trains, headway holding for the buses."""
    return route_setting(sim.train) + headway_holding(sim.bus)


def bus_blind(sim, factor=2.0):
    """`terminate_when_stuck` with the train env's fixed penalty per missed
    call, and headway holding: each mode's own best rule, side by side."""
    return terminate_when_stuck(sim.train, factor=factor) + headway_holding(sim.bus)


def bus_aware(sim, factor=2.0):
    """`terminate_when_stuck` with the cost read from the buses
    (`transfer_cost`), and headway holding: the joint rule."""
    cost = [sim.transfer_cost(i) for i in range(sim.n_trains)]
    return terminate_when_stuck(sim.train, factor=factor, cost=cost) + headway_holding(sim.bus)
