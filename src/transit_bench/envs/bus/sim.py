"""The bus env's Python reference sim: buses on a loop of stops.

The slow, readable ground truth that the JAX env is tested against step for
step (given the same random draws).

Model (see docs/formulation.md, "Bus"):
- Stops on a one-way loop (`core.network.loop`); edge s runs from stop s to
  stop s + 1. A trip's running time is drawn when the bus leaves a stop:
  nominal time x edge speed factor x lognormal noise (mean 1, CV
  `travel_cv`), rounded up to whole ticks. A closed edge holds buses at the
  stop before it.
- Passengers arrive at each stop as a Poisson process (rate scaled over the
  episode by the demand profile), to a destination uniform over the other
  stops. Waiting passengers are counts per (stop, destination); boarding
  takes the nearest destinations first.
- Time moves in fixed ticks of `step_seconds`. Each tick a bus at a stop
  spends its seconds on alighting (`alight_seconds` each) and then boarding
  (`board_seconds` each). That is the bunching mechanism: a late bus finds
  more people waiting, dwells longer and falls further behind.
- Once a bus has nobody to drop off and nobody it can pick up, it is
  *deciding*: action GO leaves for the next stop, HOLD waits one more tick
  (still picking up anyone who arrives). Actions of buses that are not
  deciding are ignored.
- Buses act in index order within a tick (a lower-numbered bus boards
  first) and may overtake each other.

The reward is minus the passenger-minutes spent this tick (waiting plus on
board) over the number of stops, so holding trades waiting time against
in-vehicle time.

Randomness is drawn up front for each step as `Exogenous` (Powell's W_{t+1}:
the arrivals and one noise factor per bus), so a replay can hand the same
draws to the JAX env.
"""

import math
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np

from transit_bench.core.network import loop

GO, HOLD = 0, 1
DEMAND_PROFILES = ("flat", "peak")
# "peak": keyframes of (episode fraction, demand factor), linearly interpolated.
_PEAK_KEYFRAMES = ((0.0, 0.5), (0.5, 1.5), (1.0, 0.5))
NEVER = -1e9  # last_depart before any bus has left a stop
HEAD_FIELDS = 7  # per bus in the observation
STOP_FIELDS = 3  # per stop offset in the observation


@dataclass(frozen=True)
class BusConfig:
    n_stops: int = 10
    n_buses: int = 3
    capacity: int = 80
    travel_time: float = 60.0  # nominal seconds per edge
    travel_cv: float = 0.0  # lognormal noise on each trip's running time
    arrival_rate: float = 1 / 60  # passengers per second per stop, before the profile
    demand: str = "flat"
    alight_seconds: float = 1.0
    board_seconds: float = 2.0
    step_seconds: float = 10.0
    # The heuristic's target and its cap on holding, in seconds; the target
    # defaults to an estimate of the loop time over the number of buses.
    target_headway: float | None = None
    max_hold_seconds: float = 120.0

    def __post_init__(self):
        if self.demand not in DEMAND_PROFILES:
            raise ValueError(f"demand must be one of {DEMAND_PROFILES}, got {self.demand!r}")
        if self.n_stops < 2 or self.n_buses < 1 or self.capacity < 1:
            raise ValueError("need at least 2 stops, 1 bus and a capacity of 1")

    @property
    def headway(self):
        """The target headway H: given, or the evenly spaced steady state.

        With even headways each stop sees λH new passengers per bus, and as
        many get off, so a bus dwells λH(α + β) per stop and a loop takes
        Sτ + SλH(α + β) = BH. Solving: H = Sτ / (B − Sλ(α + β)), finite
        while the buses can keep up with demand.
        """
        if self.target_headway is not None:
            return self.target_headway
        work = self.n_stops * self.arrival_rate * (self.alight_seconds + self.board_seconds)
        if work >= self.n_buses:
            raise ValueError("demand exceeds what the buses can board; set target_headway")
        return self.n_stops * self.travel_time / (self.n_buses - work)

    def network(self):
        return loop(np.full(self.n_stops, self.travel_time))


# Each preset adds one feature to the one before.
PRESETS = {"loop": BusConfig()}
PRESETS["noisy"] = replace(PRESETS["loop"], travel_cv=0.25)
PRESETS["busy"] = replace(PRESETS["noisy"], arrival_rate=1 / 30)
PRESETS["peak"] = replace(PRESETS["busy"], demand="peak")
# A longer route: twice the stops and buses, half the demand per stop, so
# each bus carries about as many as on `peak` (up to ~60 of 80 at the peak).
PRESETS["full"] = replace(PRESETS["peak"], n_stops=20, n_buses=6, arrival_rate=1 / 60)


class Exogenous(NamedTuple):
    """One step's random draws."""

    arrivals: np.ndarray  # (stops, stops) int: new passengers per origin, destination
    noise: np.ndarray  # (buses,) float: running-time factor if the bus leaves this tick


def demand_factor(demand, episode_fraction):
    if demand == "flat":
        return 1.0
    t = min(max(episode_fraction, 0.0), 1.0)
    times, values = zip(*_PEAK_KEYFRAMES, strict=True)
    return float(np.interp(t, times, values))


def arrival_rates(config, episode_fraction):
    """Expected arrivals per step, (origin, destination)."""
    c = config
    per_stop = c.arrival_rate * c.step_seconds * demand_factor(c.demand, episode_fraction)
    rates = np.full((c.n_stops, c.n_stops), per_stop / (c.n_stops - 1))
    np.fill_diagonal(rates, 0.0)
    return rates


def lognormal_params(cv):
    """(mu, sigma) of a lognormal with mean 1 and coefficient of variation cv."""
    sigma2 = math.log1p(cv * cv)
    return -sigma2 / 2, math.sqrt(sigma2)


class BusSim:
    def __init__(self, config=None, max_steps=360, rng=None):
        if isinstance(config, str):
            config = PRESETS[config]
        self.config = config if config is not None else BusConfig()
        self.max_steps = max_steps
        self.rng = rng if rng is not None else np.random.default_rng()
        network = self.config.network()
        self.travel_time = network.travel_time
        # Position of each stop along the loop, in nominal seconds.
        self.offset = np.concatenate([[0.0], np.cumsum(self.travel_time)[:-1]])
        self.length = float(self.travel_time.sum())
        self.edge_open = np.ones(network.n_edges, bool)
        self.edge_speed = np.ones(network.n_edges)
        self.reset()

    # -- state ---------------------------------------------------------------

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.config
        self.t = 0
        # Evenly spaced, at stops.
        self.stop = [b * c.n_stops // c.n_buses for b in range(c.n_buses)]
        self.at_stop = [True] * c.n_buses
        self.remaining = [0] * c.n_buses  # ticks left on the current edge
        self.trip = [1] * c.n_buses  # ticks the current edge takes in all
        self.hold = [0] * c.n_buses  # ticks held at the current stop
        self.onboard = [[0] * c.n_stops for _ in range(c.n_buses)]
        self.waiting = [[0] * c.n_stops for _ in range(c.n_stops)]
        self.last_depart = [NEVER] * c.n_stops
        return self.observe()

    def draw(self):
        """This step's `Exogenous` draws."""
        c = self.config
        arrivals = self.rng.poisson(arrival_rates(c, self.t / self.max_steps))
        if c.travel_cv > 0:
            mu, sigma = lognormal_params(c.travel_cv)
            noise = self.rng.lognormal(mu, sigma, c.n_buses)
        else:
            noise = np.ones(c.n_buses)
        return Exogenous(arrivals, noise)

    # -- dynamics ------------------------------------------------------------

    def _alightable(self, b):
        return self.onboard[b][self.stop[b]]

    def _boardable(self, b):
        s = self.stop[b]
        space = self.config.capacity - sum(self.onboard[b]) + self._alightable(b)
        return min(space, sum(self.waiting[s]))

    def deciding(self, b):
        return self.at_stop[b] and self._alightable(b) == 0 and self._boardable(b) == 0

    def step(self, actions, exo=None):
        """One tick: (observation, reward, truncated, info)."""
        c = self.config
        if exo is None:
            exo = self.draw()
        for s in range(c.n_stops):
            for d in range(c.n_stops):
                self.waiting[s][d] += int(exo.arrivals[s][d])
        now = self.t * c.step_seconds
        info = dict.fromkeys(("delivered", "boarded", "departures", "bunched", "held"), 0) | {
            "gap_sum": 0.0,
            "gap_sq_sum": 0.0,
        }
        for b in range(c.n_buses):
            if not self.at_stop[b]:
                self.remaining[b] -= 1
                if self.remaining[b] <= 0:
                    self.at_stop[b] = True
                    self.stop[b] = (self.stop[b] + 1) % c.n_stops
                    self.remaining[b] = 0
                continue
            s = self.stop[b]
            if self.deciding(b):
                if int(actions[b]) == GO and self.edge_open[s]:
                    seconds = self.travel_time[s] * self.edge_speed[s] * exo.noise[b]
                    ticks = max(1, math.ceil(seconds / c.step_seconds - 1e-9))
                    self.trip[b] = self.remaining[b] = ticks
                    self.at_stop[b] = False
                    self.hold[b] = 0
                    if self.last_depart[s] > NEVER:
                        gap = now - self.last_depart[s]
                        info["gap_sum"] += gap
                        info["gap_sq_sum"] += gap * gap
                        info["bunched"] += gap < 0.25 * c.headway
                    info["departures"] += 1
                    self.last_depart[s] = now
                else:
                    self.hold[b] += 1
                    info["held"] += 1
                continue
            # Alight, then board the nearest destinations first, within the tick.
            budget = c.step_seconds
            n_alight = min(self._alightable(b), int(budget // c.alight_seconds))
            budget -= n_alight * c.alight_seconds
            self.onboard[b][s] -= n_alight
            space = c.capacity - sum(self.onboard[b])
            n_board = min(space, sum(self.waiting[s]), int(budget // c.board_seconds))
            info["delivered"] += n_alight
            info["boarded"] += n_board
            for k in range(1, c.n_stops):
                d = (s + k) % c.n_stops
                take = min(n_board, self.waiting[s][d])
                self.waiting[s][d] -= take
                self.onboard[b][d] += take
                n_board -= take
        self.t += 1
        waiting = sum(map(sum, self.waiting))
        onboard = sum(map(sum, self.onboard))
        info["waiting"] = waiting
        info["onboard"] = onboard
        reward = -(waiting + onboard) * c.step_seconds / 60 / c.n_stops
        return self.observe(), reward, self.t >= self.max_steps, info

    # -- observation ---------------------------------------------------------

    def position(self, b):
        """Bus b's position along the loop, in nominal seconds."""
        s = self.stop[b]
        if self.at_stop[b]:
            return self.offset[s]
        progress = 1.0 - self.remaining[b] / self.trip[b]
        return self.offset[s] + progress * self.travel_time[s]

    def observe(self):
        """One block per bus, centred on it (its stop, or the next one).

        Head: at a stop, deciding, load, gap to the bus ahead and behind (in
        headways, capped at 2, halved), time held (in headways, capped at 1),
        progress along the current edge. Then for each stop from that one on
        round the loop: people waiting there and on board for there (over the
        capacity, capped at 1), and time since a bus last left it (in
        headways, capped at 2, halved; 1 if none has).
        """
        c = self.config
        h, cap = c.headway, c.capacity
        now = self.t * c.step_seconds
        positions = [self.position(b) for b in range(c.n_buses)]
        since = [
            1.0 if last <= NEVER else min((now - last) / h, 2.0) / 2 for last in self.last_depart
        ]
        waiting = [sum(row) for row in self.waiting]
        blocks = []
        for b in range(c.n_buses):
            others = [positions[o] for o in range(c.n_buses) if o != b]
            ahead = min(((p - positions[b]) % self.length for p in others), default=self.length)
            behind = min(((positions[b] - p) % self.length for p in others), default=self.length)
            progress = 0.0 if self.at_stop[b] else 1.0 - self.remaining[b] / self.trip[b]
            head = [
                float(self.at_stop[b]),
                float(self.deciding(b)),
                sum(self.onboard[b]) / cap,
                min(ahead / h, 2.0) / 2,
                min(behind / h, 2.0) / 2,
                min(self.hold[b] * c.step_seconds / h, 1.0),
                progress,
            ]
            ref = self.stop[b] if self.at_stop[b] else (self.stop[b] + 1) % c.n_stops
            stops = []
            for k in range(c.n_stops):
                q = (ref + k) % c.n_stops
                stops += [min(waiting[q] / cap, 1.0), min(self.onboard[b][q] / cap, 1.0), since[q]]
            blocks.append(head + stops)
        return np.array(blocks, dtype=np.float32).ravel()


def headway_holding(sim):
    """The heuristic on the Python sim: hold a deciding bus until a target
    headway has passed since the bus ahead left this stop, for at most
    `max_hold_seconds`."""
    c = sim.config
    now = sim.t * c.step_seconds
    actions = []
    for b in range(c.n_buses):
        early = now - sim.last_depart[sim.stop[b]] < c.headway
        room = sim.hold[b] * c.step_seconds < c.max_hold_seconds
        actions.append(HOLD if sim.deciding(b) and early and room else GO)
    return actions
