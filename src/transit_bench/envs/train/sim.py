"""The train env's Python reference sim: dispatching on blocks of track.

The slow, readable ground truth that the JAX env is tested against step for
step (given the same random draws). The infrastructure, routes and
timetable come from `infra.build`.

Dynamics, in ticks of `step_seconds`:
- A train occupies one block at a time and holds it until it enters the
  next (blocking). Entering a block takes its running time (nominal x the
  block's speed factor x lognormal noise, plus any incident delay) plus
  the dwell if the train stops there, rounded up to whole ticks.
- The noise and incidents are drawn per train and route position at
  reset (a `Trip`), not per tick: each train enters each block once, so
  the distribution is the same, but the scenario no longer depends on
  when a train enters a block. Every policy then faces the same scenario
  given the seed (common random numbers), and the best schedule in
  hindsight is well defined (`bound`).
- A train is *ready* when its time in the current block is up (or, before
  its first block, from its scheduled departure). A ready train in its last
  block leaves the network. Otherwise, if its next block is free and open,
  it is *deciding*: GO enters the block, WAIT stays. A train whose next
  block is taken waits whatever it chose.
- Ready and deciding are judged once, at the start of the tick, from the
  state the policy observed; then trains act in index (departure) order.
  So if two deciding trains claim the same block the earlier one gets it,
  and a block freed during a tick can be entered from the next tick on
  (like a signal clearing). Time left in blocks counts down at the end of
  the tick. These synchronous rules make "deciding" exactly the set of
  trains whose action matters (`active_agents` in the JAX env).
- No train enters its first block before its scheduled departure.
- With `terminate_penalty` > 0 there is a third action, TERMINATE: a ready
  train that has not started, or stands on a platform with blocks still
  ahead, ends its trip there. It leaves the network at once (freeing its
  block, which others can enter from the next tick), and its lateness is
  frozen at its projected lateness then, plus `terminate_penalty` minutes
  for each call it misses. Before departure this is a cancellation; at an
  intermediate station it is the outbound half of a short-turn (the return
  working needs vehicle circulation, which the env does not model). Such
  a train *can terminate* and is active even if its next block is taken
  or closed, which is when terminating is worth considering.

Reward: minus the change in total *projected* lateness, in minutes. A
train's projected lateness is how late it will be at its next scheduled
event if nothing else delays it: before departure, max(0, now − departure);
in a block, max(0, now + time left − scheduled entry to the next block);
after leaving, its arrival delay. The rewards telescope, so an episode's
return is minus the total arrival delay of all trains (plus whatever is
still projected at the time limit): the weighted-lateness objective of the
job-shop formulation, made dense.

A *deadlock* is a cycle of ready trains each waiting for a block held by
the next. Trains in one never move again; `info["deadlocked"]` counts them.
"""

import math
from typing import NamedTuple

import numpy as np

from transit_bench.core import disruption
from transit_bench.envs.train.infra import PRESETS, TrainConfig, build

GO, WAIT, TERMINATE = 0, 1, 2
NOT_STARTED = -1
HEAD_FIELDS = 15
LOOKAHEAD = 4  # blocks ahead on the route in each train's observation
BLOCK_FIELDS = 6
DELAY_SCALE = 600.0  # seconds: observation scale for delays and times


class Trip(NamedTuple):
    """Each train's random running times, drawn at reset."""

    noise: np.ndarray  # (trains, L) float: running-time factor entering route[i, p]
    incident: np.ndarray  # (trains, L) float: extra seconds then (mostly 0)


class Exogenous(NamedTuple):
    """One step's random draws."""

    onset: np.ndarray  # (blocks,) float: disruption onset draws (core.disruption)
    duration: np.ndarray  # (blocks,) float: outage seconds if one starts


def lognormal_params(cv):
    sigma2 = math.log1p(cv * cv)
    return -sigma2 / 2, math.sqrt(sigma2)


class TrainSim:
    def __init__(self, config=None, max_steps=360, rng=None):
        if isinstance(config, str):
            config = PRESETS[config]
        self.config = config if config is not None else TrainConfig()
        self.infra = build(self.config)
        self.max_steps = max_steps
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    # -- state ---------------------------------------------------------------

    def reset(self, seed=None, trip=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        T = len(self.infra.route)
        self.trip = trip if trip is not None else self.draw_trip()
        self.t = 0
        self.pos = [NOT_STARTED] * T
        self.remaining = [0] * T
        self.arrival_delay = [0.0] * T
        self.occupant = [-1] * self.infra.n_blocks
        self.block_open = [True] * self.infra.n_blocks
        self.block_speed = [1.0] * self.infra.n_blocks
        self.disruption = disruption.initial_np(self.infra.n_blocks)
        self.potential = 0.0
        return self.observe()

    def draw_trip(self):
        c = self.config
        shape = self.infra.route.shape
        if c.running_cv > 0:
            mu, sigma = lognormal_params(c.running_cv)
            noise = self.rng.lognormal(mu, sigma, shape)
        else:
            noise = np.ones(shape)
        hit = self.rng.random(shape) < c.incident_probability
        incident = np.where(hit, self.rng.exponential(c.incident_seconds, shape), 0.0)
        return Trip(noise, incident)

    def draw(self):
        d = disruption.draw_np(self.config.disruption, self.rng, self.infra.n_blocks)
        return Exogenous(d.onset, d.duration)

    # -- queries -------------------------------------------------------------

    def finished(self, i):
        return self.pos[i] >= self.infra.route_len[i]

    def next_block(self, i):
        """The block train i would enter next, or -1 from its last block."""
        nxt = self.pos[i] + 1
        return int(self.infra.route[i][nxt]) if nxt < self.infra.route_len[i] else -1

    def ready(self, i):
        if self.finished(i):
            return False
        if self.pos[i] == NOT_STARTED:
            return self.t * self.config.step_seconds >= self.infra.sched[i][0]
        return self.remaining[i] == 0

    def deciding(self, i):
        b = self.next_block(i)
        return self.ready(i) and b >= 0 and self.occupant[b] < 0 and self.block_open[b]

    def can_terminate(self, i):
        """Ready, not started or on a platform, and not at its last block."""
        if self.config.terminate_penalty <= 0 or not self.ready(i) or self.next_block(i) < 0:
            return False
        return self.pos[i] == NOT_STARTED or self.infra.station_of[self.here(i)] >= 0

    def here(self, i):
        """The block train i holds, or -1."""
        if self.pos[i] == NOT_STARTED or self.finished(i):
            return -1
        return int(self.infra.route[i][self.pos[i]])

    def missed_calls(self, i):
        """Calls train i would miss if it terminated now."""
        return int(self.infra.calls_from[i][self.pos[i] + 1])

    def projected_delay(self, i, now):
        sched = self.infra.sched[i]
        if self.finished(i):
            return self.arrival_delay[i]
        if self.pos[i] == NOT_STARTED:
            return max(0.0, now - sched[0])
        exit_time = now + self.remaining[i] * self.config.step_seconds
        return max(0.0, exit_time - sched[self.pos[i] + 1])

    def waits_for(self):
        """Per train, the train holding the block it is ready to enter, or -1."""
        out = []
        for i in range(len(self.pos)):
            b = self.next_block(i)
            started = self.pos[i] != NOT_STARTED
            waiting = started and self.ready(i) and b >= 0
            out.append(self.occupant[b] if waiting else -1)
        return out

    def deadlocked_trains(self):
        """Per train, whether it is on a cycle of waits: each waits for a
        block the next holds."""
        waits = self.waits_for()
        out = []
        for i in range(len(waits)):
            j = waits[i]
            for _ in range(len(waits)):
                if j < 0 or j == i:
                    break
                j = waits[j]
            out.append(j == i)
        return out

    def deadlocked(self):
        return sum(self.deadlocked_trains())

    # -- dynamics ------------------------------------------------------------

    def step(self, actions, exo=None):
        """One tick: (observation, reward, terminated, truncated, info)."""
        c, infra = self.config, self.infra
        if exo is None:
            exo = self.draw()
        now = self.t * c.step_seconds
        keys = ("entered", "arrived", "held", "blocked", "terminated", "missed")
        info = dict.fromkeys(keys, 0)
        T = len(self.pos)
        ready = [self.ready(i) for i in range(T)]
        deciding = [self.deciding(i) for i in range(T)]
        can_terminate = [self.can_terminate(i) for i in range(T)]
        claimed = set()
        for i in range(T):
            if not ready[i]:
                continue
            if can_terminate[i] and int(actions[i]) == TERMINATE:
                missed = self.missed_calls(i)
                late = self.projected_delay(i, now)
                if self.pos[i] != NOT_STARTED:
                    self.occupant[self.here(i)] = -1
                self.pos[i] = int(infra.route_len[i])
                self.arrival_delay[i] = late + 60 * c.terminate_penalty * missed
                info["terminated"] += 1
                info["missed"] += missed
                continue
            b = self.next_block(i)
            if b < 0:  # leave the network from the last block
                self.occupant[int(infra.route[i][self.pos[i]])] = -1
                self.pos[i] = int(infra.route_len[i])
                self.arrival_delay[i] = max(0.0, now - infra.sched[i][self.pos[i]])
                info["arrived"] += 1
                continue
            if not deciding[i] or b in claimed:
                info["blocked"] += 1
                continue
            if int(actions[i]) != GO:
                info["held"] += 1
                continue
            if self.pos[i] != NOT_STARTED:
                self.occupant[int(infra.route[i][self.pos[i]])] = -1
            self.pos[i] += 1
            self.occupant[b] = i
            claimed.add(b)
            p = self.pos[i]
            noise, incident = self.trip.noise[i][p], self.trip.incident[i][p]
            seconds = infra.run[i][p] * self.block_speed[b] * noise + incident
            ticks = max(1, math.ceil(seconds / c.step_seconds - 1e-9))
            self.remaining[i] = ticks + int(infra.dwell[i][p])
            info["entered"] += 1
        for i in range(T):
            if self.remaining[i] > 0:
                self.remaining[i] -= 1
        self._advance_disruption(exo, now)
        self.t += 1
        later = self.t * c.step_seconds
        total = sum(self.projected_delay(i, later) for i in range(len(self.pos)))
        reward = -(total - self.potential) / 60
        self.potential = total
        info["delay"] = total
        info["deadlocked"] = self.deadlocked()
        info["finished"] = sum(self.finished(i) for i in range(len(self.pos)))
        terminated = info["finished"] == len(self.pos)
        truncated = self.t >= self.max_steps
        return self.observe(), reward, terminated, truncated, info

    # -- observation ---------------------------------------------------------

    def observe(self):
        """One block of features per train.

        Head: not started, running (time left in its block), deciding,
        finished, progress along its route, time left in its block and
        projected delay (over DELAY_SCALE, capped at 1), express, westbound,
        can terminate, and the calls it would miss if it did (over the
        number of stations);
        then, for the next section it will enter: its next block is in it
        (it would enter the section now); a train going the other way is in
        it; one is deciding to enter it this tick ahead of this train in the
        acting order; one is deciding to enter it after.
        Then for each of the next LOOKAHEAD blocks on its route: on the
        route, occupied, occupied by a train going the same way, open, time
        since it went down and time until it comes back (over DELAY_SCALE,
        capped at 1; the second is 0 unless the disruption config makes it
        observable).

        The section fields expose the local conflict a single-track
        dispatcher must resolve: an oncoming train waiting at the far end of
        the section stands on a platform that is not on this train's route.
        """
        infra = self.infra
        now = self.t * self.config.step_seconds
        T = len(self.pos)
        target = self._next_sections()
        joining = self._joining()
        out = []
        for i in range(T):
            n = int(infra.route_len[i])
            started = self.pos[i] != NOT_STARTED
            head = [
                float(not started),
                float(started and not self.finished(i) and self.remaining[i] > 0),
                float(self.deciding(i)),
                float(self.finished(i)),
                (self.pos[i] + 1) / n,
                min(self.remaining[i] * self.config.step_seconds / DELAY_SCALE, 1.0),
                min(self.projected_delay(i, now) / DELAY_SCALE, 1.0),
                float(infra.express[i]),
                float(infra.direction[i]),
                float(self.can_terminate(i)),
                self.missed_calls(i) / self.config.n_stations if not self.finished(i) else 0.0,
            ]
            sec, d = target[i], infra.direction[i]
            inside = sec >= 0 and any(
                j >= 0 and infra.section_of[blk] == sec and infra.direction[j] != d
                for blk, j in enumerate(self.occupant)
            )
            rivals = [
                j for j in range(T) if sec >= 0 and joining[j] == sec and infra.direction[j] != d
            ]
            b = self.next_block(i) if not self.finished(i) else -1
            now_entering = sec >= 0 and b >= 0 and infra.section_of[b] == sec
            head += [float(now_entering), float(inside), float(any(j < i for j in rivals))]
            head += [float(any(j > i for j in rivals))]
            ahead = []
            for k in range(1, LOOKAHEAD + 1):
                p = self.pos[i] + k
                if p < n:
                    b = int(infra.route[i][p])
                    j = self.occupant[b]
                    same = j >= 0 and infra.direction[j] == infra.direction[i]
                    ahead += [1.0, float(j >= 0), float(same), float(self.block_open[b])]
                    ahead += self._outage(b)
                else:
                    ahead += [0.0] * BLOCK_FIELDS
            out.append(head + ahead)
        return np.array(out, dtype=np.float32).ravel()

    def _outage(self, b):
        d, c = self.disruption, self.config.disruption
        if not d.down[b]:
            return [0.0, 0.0]
        left = min(d.remaining[b] / DELAY_SCALE, 1.0) if c.observe_remaining else 0.0
        return [min(d.elapsed[b] / DELAY_SCALE, 1.0), left]

    def _advance_disruption(self, exo, now):
        """Blockages move on at the end of the tick, so the next decisions see
        the state they will act under. Only section blocks can be disrupted."""
        c = self.config.disruption
        if c.mode == "none":
            return
        eligible = self.infra.section_of >= 0
        draw = disruption.DisruptionDraw(exo.onset, exo.duration)
        step = self.config.step_seconds
        self.disruption = disruption.step_np(c, self.disruption, draw, eligible, now, step)
        is_open, speed = disruption.edge_effect_np(c, self.disruption)
        self.block_open = [bool(x) for x in is_open]
        self.block_speed = [float(x) for x in speed]

    def _next_sections(self):
        """Per train, the next section it will enter from outside (-1 if none)."""
        table = self.infra.next_section
        width = table.shape[1]
        return [int(table[i][p + 1]) if p + 1 < width else -1 for i, p in enumerate(self.pos)]

    def _joining(self):
        """Per train, the section it is deciding to enter from outside, or -1."""
        infra = self.infra
        out = []
        for i in range(len(self.pos)):
            if not self.deciding(i):
                out.append(-1)
                continue
            sec = int(infra.section_of[self.next_block(i)])
            here = int(infra.route[i][self.pos[i]]) if self.pos[i] != NOT_STARTED else -1
            if sec >= 0 and here >= 0 and infra.section_of[here] == sec:
                sec = -1  # already inside
            out.append(sec)
        return out


# -- heuristics ----------------------------------------------------------------


def first_come_first_served(sim):
    """Every train goes as soon as it can; ties go to the earlier train."""
    return [GO] * len(sim.pos)


def section_reservation(sim):
    """First come, first served, but never into a section against the flow.

    A train may enter a section (the blocks between two stations) only if
    no train going the other way is in it, or about to enter it this tick
    ahead of this one in the acting order. On a single-track line this
    rules out the head-on deadlock that first come, first served walks into
    (two trains entering a section from both ends). Deadlocks spanning
    several sections and platforms are still possible.
    """
    infra = sim.infra
    actions = []
    entering = {}  # section -> directions entering it this tick, so far in order
    for i in range(len(sim.pos)):
        b = sim.next_block(i)
        if not sim.deciding(i) or infra.section_of[b] < 0:
            actions.append(GO)
            continue
        section = int(infra.section_of[b])
        here = int(infra.route[i][sim.pos[i]]) if sim.pos[i] != NOT_STARTED else -1
        if here >= 0 and infra.section_of[here] == section:
            actions.append(GO)  # already inside: carry on
            continue
        d = infra.direction[i]
        against = any(
            j >= 0 and infra.section_of[blk] == section and infra.direction[j] != d
            for blk, j in enumerate(sim.occupant)
        )
        against |= any(other != d for other in entering.get(section, []))
        if against:
            actions.append(WAIT)
        else:
            actions.append(GO)
            entering.setdefault(section, []).append(d)
    return actions


def route_setting(sim):
    """Section reservation, and set the route through to the next platform.

    As `section_reservation`, but before a section that carries both
    directions (single track) a train also waits until the platform at its
    far end is free: station-to-station route setting. On one-way sections
    trains may follow each other in, as on a double-track line.
    Section reservation alone still deadlocks when trains bunch: an
    eastbound train on a platform waits for the next section, held by a
    westbound train waiting for the westbound platform, held by a train
    waiting for the previous section, held by an eastbound train waiting
    for the first train's platform. Never entering a section without
    somewhere to go at the end breaks every such cycle on this line.
    """
    infra = sim.infra
    actions = section_reservation(sim)
    for i, action in enumerate(actions):
        if action == WAIT or not sim.deciding(i):
            continue
        b = sim.next_block(i)
        section = int(infra.section_of[b])
        here = int(infra.route[i][sim.pos[i]]) if sim.pos[i] != NOT_STARTED else -1
        if section < 0 or (here >= 0 and infra.section_of[here] == section):
            continue
        if not infra.section_shared[section]:
            continue
        far = int(infra.far_platform[i][sim.pos[i] + 1])
        if far >= 0 and sim.occupant[far] >= 0:
            actions[i] = WAIT
    return actions


def terminate_when_stuck(sim, rule=route_setting, factor=2.0):
    """`rule`, plus TERMINATE where waiting looks dearer than the missed calls.

    A train that can terminate does so when its next block is closed and
    `factor` x the outage's expected time left exceeds `terminate_penalty`
    x the calls it would miss. The factor is there because waiting costs
    more than the blocked train's own wait: the trains queued behind it
    wait too. On `response`, factor 2 beats 1 (3.70 vs 3.80 minutes per
    train) and larger factors change nothing (docs/plan.md, Phase 4).
    The expected time left is the observed one if the config shows it,
    else the mean outage length: exact for Markov (memoryless)
    outages, whose time left does not depend on how long they have lasted,
    and a rough guess for semi-Markov ones. It also breaks deadlocks: of the
    deadlocked trains that can terminate, the one missing fewest calls
    (lowest index on a tie) terminates; one per tick.
    """
    actions = list(rule(sim))
    c = sim.config
    T = len(sim.pos)
    for i in range(T):
        if not sim.can_terminate(i):
            continue
        b = sim.next_block(i)
        if sim.block_open[b]:
            continue
        d = sim.disruption
        left = d.remaining[b] if c.disruption.observe_remaining else c.disruption.mean_seconds
        if factor * left > 60 * c.terminate_penalty * sim.missed_calls(i):
            actions[i] = TERMINATE
    stuck = sim.deadlocked_trains()
    options = [(sim.missed_calls(i), i) for i in range(T) if stuck[i] and sim.can_terminate(i)]
    if options:
        actions[min(options)[1]] = TERMINATE
    return actions
