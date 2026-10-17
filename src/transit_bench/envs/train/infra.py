"""Rail infrastructure and timetables for the train env.

The network is split into track *blocks*, each a resource that holds at most
one train (κ = 1); the train env is a job shop with blocking over them (see
docs/formulation.md, "Train"). This module builds, from a `TrainConfig`, a
line of stations joined by sections of blocks, the trains' fixed routes
through it and their timetable, as plain NumPy arrays:

- A *station* has a main platform block per direction, plus a loop
  platform where the config has passing loops. On a single-track line the
  two directions' platforms are the crossing loop: trains can only pass
  each other there.
- A *section* between stations is `blocks_per_section` blocks, one track
  per direction on a double-track line, shared on a single-track line.
- Trains run east (station 0 to the last) or west. Stopping trains dwell at
  every station and use the loop platform where there is one, so express
  trains can overtake them there; express trains run faster and only stop
  at the ends.
- A *relief track* (`relief_sections`) is a second track alongside a
  section, `relief_slowdown` times slower, used in both directions. A
  train standing at the station before such a section chooses which track
  to take (GO for the main one, DIVERT for the relief): the network's
  route choice. Both tracks have `blocks_per_section` blocks, so a train's
  route positions, timetable and calls are the same whichever it takes;
  `alt_route` and `alt_run` give the relief alternative per position.

Times are in ticks of `step_seconds`. The timetable gives each train the
time it should enter each block of its route, from its departure time and
the nominal running and dwell times, padded by `slack` (recovery time).
"""

import math
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np

from transit_bench.core.disruption import DisruptionConfig

EAST, WEST = 0, 1


@dataclass(frozen=True)
class TrainConfig:
    n_stations: int = 5
    blocks_per_section: int = 2
    single_track: bool = False
    passing_loops: bool = False  # a loop platform per direction at inner stations
    # Each direction: trains depart every `headway` seconds, `trains_per_direction`
    # of them; `express_every` > 0 makes every n-th train an express.
    trains_per_direction: int = 6
    headway: float = 300.0
    west_offset: float = 150.0  # seconds after the first eastbound departure
    express_every: int = 0
    section_block_seconds: float = 90.0  # stopping train, per section block
    express_speedup: float = 1.5  # express running times are divided by this
    platform_seconds: float = 30.0  # running through a station block
    dwell_seconds: float = 60.0  # stopping trains, at each station
    slack: float = 0.05  # timetable padding on running and dwell times
    step_seconds: float = 10.0
    # Randomness: lognormal noise on each block's running time, and with
    # probability `incident_probability` per block entry an extra dwell of
    # mean `incident_seconds` (exponential), the primary delays.
    running_cv: float = 0.1
    incident_probability: float = 0.02
    incident_seconds: float = 120.0
    # Blockages of section blocks (see core.disruption); none by default.
    disruption: DisruptionConfig = DisruptionConfig()
    # > 0 enables the TERMINATE response action (cancel a trip before it
    # departs, or short-turn it at a platform); each scheduled call the
    # train then misses costs this many minutes.
    terminate_penalty: float = 0.0
    # Sections (index s joins stations s and s + 1) with a relief track, and
    # its running times relative to the main track's.
    relief_sections: tuple = ()
    relief_slowdown: float = 1.25

    def __post_init__(self):
        if self.n_stations < 2 or self.blocks_per_section < 1:
            raise ValueError("need at least 2 stations and 1 block per section")
        if self.terminate_penalty < 0:
            raise ValueError("terminate_penalty must be >= 0")
        if self.single_track and self.passing_loops:
            raise ValueError("single-track lines pass at their crossing loops only")
        if any(not 0 <= s < self.n_stations - 1 for s in self.relief_sections):
            raise ValueError("relief_sections must name sections 0 .. n_stations - 2")
        if self.relief_slowdown < 1:
            raise ValueError("relief_slowdown must be >= 1")

    @property
    def n_trains(self):
        return 2 * self.trains_per_direction


# Each preset adds one hard feature to the one before.
PRESETS = {"line": TrainConfig()}
PRESETS["passing"] = replace(PRESETS["line"], passing_loops=True, express_every=3)
PRESETS["single"] = replace(
    PRESETS["line"], single_track=True, trains_per_direction=4, headway=600.0, west_offset=0.0
)
# Section blocks close at random: about 0.8 outages an episode over the 8
# section blocks, lasting 10 minutes on average (exponential, memoryless).
PRESETS["blockage"] = replace(
    PRESETS["single"],
    disruption=DisruptionConfig(mode="markov", rate_per_hour=0.1, mean_seconds=600.0),
)
# Longer outages (20 minutes on average) and the TERMINATE response: cancel
# or short-turn a train, at 5 minutes per missed call (the mean passenger
# wait for the next train at a 10-minute headway).
PRESETS["response"] = replace(
    PRESETS["blockage"],
    disruption=DisruptionConfig(mode="markov", rate_per_hour=0.1, mean_seconds=1200.0),
    terminate_penalty=5.0,
)
# Route choice: the blockage line with a relief track alongside the two
# middle sections, 25% slower. A train can divert onto it around a closed
# block, or to cross an oncoming train between stations.
PRESETS["junction"] = replace(PRESETS["blockage"], relief_sections=(1, 2))


class Infra(NamedTuple):
    n_blocks: int
    station_of: np.ndarray  # (blocks,) int: station index, -1 for section blocks
    section_of: (
        np.ndarray
    )  # (blocks,) int: section (stretch of track between stations), -1 at stations
    route: np.ndarray  # (trains, L) int: block ids, padded with -1
    route_len: np.ndarray  # (trains,) int
    run: np.ndarray  # (trains, L) float: nominal running seconds per route step
    dwell: np.ndarray  # (trains, L) int: dwell ticks per route step
    sched: np.ndarray  # (trains, L + 1) float: scheduled entry seconds; [len] = arrival
    direction: np.ndarray  # (trains,) int: EAST or WEST
    express: np.ndarray  # (trains,) bool
    # (trains, L + 1): at route index pos + 1, the next section the train
    # will enter from outside it (-1 if none left).
    next_section: np.ndarray
    # (trains, L + 1): at route index q, the first platform after q (-1 if none):
    # where a train entering a section at q will next stop.
    far_platform: np.ndarray
    # (sections,) bool: used by trains in both directions (single track)
    section_shared: np.ndarray
    # (trains, L + 1): calls (stops, dwell > 0) at route index q or later.
    # A train terminated on the block at p misses calls_from[p + 1]; before
    # its departure, calls_from[0].
    calls_from: np.ndarray
    # (trains, L + 1): nominal ticks (no noise or incidents) from entering
    # route index 0 to entering index q; the difference of two entries is
    # the nominal time between them.
    cum_ticks: np.ndarray
    # (blocks,) bool: a loop platform, where a following train can overtake
    loop_platform: np.ndarray
    # (trains, L): the relief-track block a train can take at each route
    # position instead (-1 if none), and its nominal running seconds there.
    alt_route: np.ndarray
    alt_run: np.ndarray


def build(config: TrainConfig) -> Infra:
    c = config
    n_st, k = c.n_stations, c.blocks_per_section
    blocks = []  # (kind, station or section, detail) per block id

    def add(kind, where):
        blocks.append((kind, where))
        return len(blocks) - 1

    # Platforms: [direction][station] -> main, and loop where present.
    main = [[add("station", s) for s in range(n_st)] for _ in range(2)]
    has_loop = [c.passing_loops and 0 < s < n_st - 1 for s in range(n_st)]
    loop = [[add("station", s) if has_loop[s] else -1 for s in range(n_st)] for _ in range(2)]
    # Sections: [direction][section][j], j counted eastwards; shared if single track.
    # Section ids name a stretch of physical track: one per stretch on a
    # single-track line, one per stretch and direction on a double-track one.
    if c.single_track:
        shared = [[add("section", s) for _ in range(k)] for s in range(n_st - 1)]
        section = [shared, shared]
    else:
        section = [
            [[add("section", d * (n_st - 1) + s) for _ in range(k)] for s in range(n_st - 1)]
            for d in range(2)
        ]

    # Relief tracks: one id per stretch, after the main sections' ids.
    first_relief = 1 + max(w for kind, w in blocks if kind == "section")
    relief = {
        s: [add("section", first_relief + n) for _ in range(k)]
        for n, s in enumerate(sorted(set(c.relief_sections)))
    }

    n_blocks = len(blocks)
    station_of = np.array([w if kind == "station" else -1 for kind, w in blocks])
    section_of = np.array([w if kind == "section" else -1 for kind, w in blocks])

    trains = []
    for d in (EAST, WEST):
        for n in range(c.trains_per_direction):
            express = c.express_every > 0 and n % c.express_every == c.express_every - 1
            depart = n * c.headway + (c.west_offset if d == WEST else 0.0)
            trains.append((d, express, depart))
    # Index order is departure order: on a tie for a block, the lower index
    # (the earlier-scheduled train) goes first.
    trains.sort(key=lambda tr: tr[2])

    L = n_st + (n_st - 1) * k
    T = len(trains)
    route = np.full((T, L), -1, np.int64)
    run = np.zeros((T, L))
    dwell = np.zeros((T, L), np.int64)
    sched = np.zeros((T, L + 1))
    alt_route = np.full((T, L), -1, np.int64)
    alt_run = np.zeros((T, L))
    for i, (d, express, depart) in enumerate(trains):
        order = range(n_st) if d == EAST else range(n_st - 1, -1, -1)
        steps = []
        stations = list(order)
        for pos, s in enumerate(stations):
            stop = (not express) or pos in (0, len(stations) - 1)
            platform = loop[d][s] if (has_loop[s] and not express) else main[d][s]
            steps.append((platform, c.platform_seconds, stop, -1))
            if pos < len(stations) - 1:
                nxt = stations[pos + 1]
                sec = section[d][min(s, nxt)]
                js = range(k) if d == EAST else range(k - 1, -1, -1)
                seconds = c.section_block_seconds / (c.express_speedup if express else 1.0)
                alt = relief.get(min(s, nxt), [-1] * k)
                steps += [(sec[j], seconds, False, alt[j]) for j in js]
        t = depart
        for j, (block, seconds, stop, alt) in enumerate(steps):
            route[i, j] = block
            alt_route[i, j] = alt
            alt_run[i, j] = seconds * c.relief_slowdown if alt >= 0 else 0.0
            run[i, j] = seconds
            dwell[i, j] = math.ceil(c.dwell_seconds / c.step_seconds) if stop else 0
            sched[i, j] = t
            ticks = math.ceil(seconds / c.step_seconds) + dwell[i, j]
            t += ticks * c.step_seconds * (1 + c.slack)
        sched[i, len(steps)] = t
    direction = np.array([d for d, _, _ in trains])
    express = np.array([e for _, e, _ in trains])
    next_section = np.full((T, L + 1), -1, np.int64)
    for i in range(T):
        for pos in range(-1, L):
            here = section_of[route[i, pos]] if pos >= 0 else -1
            for p in range(pos + 1, L):
                sec = section_of[route[i, p]]
                if sec >= 0 and sec != here:
                    next_section[i, pos + 1] = sec
                    break
    far_platform = np.full((T, L + 1), -1, np.int64)
    for i in range(T):
        for q in range(L):
            for p in range(q + 1, L):
                if station_of[route[i, p]] >= 0:
                    far_platform[i, q] = route[i, p]
                    break
    n_sections = int(section_of.max()) + 1
    used = np.zeros((n_sections, 2), bool)
    for i in range(T):
        for r in (route[i], alt_route[i][alt_route[i] >= 0]):
            secs = section_of[r]
            used[secs[secs >= 0], direction[i]] = True
    section_shared = used.all(axis=1)
    calls = np.concatenate([dwell > 0, np.zeros((T, 1), bool)], axis=1)
    calls_from = np.cumsum(calls[:, ::-1], axis=1)[:, ::-1]
    nominal = np.maximum(1, np.ceil(run / c.step_seconds - 1e-9)).astype(np.int64) + dwell
    cum_ticks = np.concatenate([np.zeros((T, 1), np.int64), np.cumsum(nominal, axis=1)], axis=1)
    loop_platform = np.zeros(n_blocks, bool)
    loop_platform[[b for row in loop for b in row if b >= 0]] = True
    return Infra(
        n_blocks,
        station_of,
        section_of,
        route,
        np.full(T, L),
        run,
        dwell,
        sched,
        direction,
        express,
        next_section,
        far_platform,
        section_shared,
        calls_from,
        cum_ticks,
        loop_platform,
        alt_route,
        alt_run,
    )
