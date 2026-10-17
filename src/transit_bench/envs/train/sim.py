"""The train env's Python reference sim: dispatching on blocks of track.

The slow, readable ground truth that the JAX env is tested against step for
step (given the same random draws). The infrastructure, routes and
timetable come from `infra.build`.

Dynamics, in ticks of `step_seconds`:
- A train occupies one block at a time and holds it until it enters the
  next (blocking). Entering a block takes its running time (nominal x the
  block's speed factor x lognormal noise, plus any incident delay drawn
  for this entry) plus the dwell if the train stops there, rounded up to
  whole ticks.
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

GO, WAIT = 0, 1
NOT_STARTED = -1
HEAD_FIELDS = 13
LOOKAHEAD = 4  # blocks ahead on the route in each train's observation
BLOCK_FIELDS = 6
DELAY_SCALE = 600.0  # seconds: observation scale for delays and times


class Exogenous(NamedTuple):
    """One step's random draws."""

    noise: np.ndarray  # (trains,) float: running-time factor if the train enters a block
    incident: np.ndarray  # (trains,) float: extra seconds if it does (mostly 0)
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

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        T = len(self.infra.route)
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

    def draw(self):
        c = self.config
        T = len(self.infra.route)
        if c.running_cv > 0:
            mu, sigma = lognormal_params(c.running_cv)
            noise = self.rng.lognormal(mu, sigma, T)
        else:
            noise = np.ones(T)
        hit = self.rng.random(T) < c.incident_probability
        incident = np.where(hit, self.rng.exponential(c.incident_seconds, T), 0.0)
        d = disruption.draw_np(c.disruption, self.rng, self.infra.n_blocks)
        return Exogenous(noise, incident, d.onset, d.duration)

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

    def deadlocked(self):
        """Trains on a cycle of waits: each waits for a block the next holds."""
        waits = self.waits_for()
        stuck = 0
        for i in range(len(waits)):
            j = waits[i]
            for _ in range(len(waits)):
                if j < 0 or j == i:
                    break
                j = waits[j]
            stuck += j == i
        return stuck

    # -- dynamics ------------------------------------------------------------

    def step(self, actions, exo=None):
        """One tick: (observation, reward, terminated, truncated, info)."""
        c, infra = self.config, self.infra
        if exo is None:
            exo = self.draw()
        now = self.t * c.step_seconds
        info = dict.fromkeys(("entered", "arrived", "held", "blocked"), 0)
        T = len(self.pos)
        ready = [self.ready(i) for i in range(T)]
        deciding = [self.deciding(i) for i in range(T)]
        claimed = set()
        for i in range(T):
            if not ready[i]:
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
            seconds = infra.run[i][p] * self.block_speed[b] * exo.noise[i] + exo.incident[i]
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
        projected delay (over DELAY_SCALE, capped at 1), express, westbound;
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
