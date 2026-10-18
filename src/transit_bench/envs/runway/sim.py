"""The runway env's Python reference sim: aircraft landing on one runway.

The slow, readable ground truth that the JAX env is tested against step for
step (given the same random draws). See docs/formulation.md, "Runway".

Model, in ticks of `step_seconds`:
- Aircraft arrive as a Poisson process (`arrivals_per_hour`, or a piecewise
  `profile`), each with a wake class drawn from `mix`. An aircraft that
  arrives in tick t is announced at the end of it, with an earliest landing
  tick (its ETA) E = t + 1 + `notice`, into a buffer of `max_aircraft`
  slots (the k-th arrival into the k-th free slot). Arrivals that find the
  buffer full are rejected (diverted).
- Every aircraft gets a sequence number in arrival order. E grows with it,
  so first come, first served (FCFS) is the order of sequence numbers.
- Separation: landing an aircraft of class j at tick t after one of class i
  landed at tick s needs t - s >= S[i, j], the `separation` in seconds
  rounded up to ticks.
- One controller acts each tick: WAIT (0), or LAND slot r (1 + r). A LAND
  is carried out if the slot holds an aircraft, its E has come, the
  separation from the last landing has passed and the runway is open;
  otherwise it is a WAIT. Landing an aircraft adds one to the `overtaken`
  count of every waiting aircraft that arrived before it (for the
  constrained-position-shifting rule).
- The runway is one edge of the disruption process (`core.disruption`):
  while down it is closed. It is advanced at the end of each tick, so
  whether the runway is open in tick t is known at the start of it.

Reward: minus the aircraft-minutes of delay this tick (aircraft whose E has
come and that have not landed by the end of the controller's turn), and
`reject_penalty` minutes per rejected arrival. An episode's return is then
minus the total delay sum(min(land, T) - E) over aircraft with E < T, less
the penalties: the hindsight bound (`bound`) minimises the same sum.
"""

import itertools
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np

from transit_bench.core import disruption
from transit_bench.core.disruption import DisruptionConfig

WAIT = 0
LAND = 1  # action LAND + r lands slot r
CLASSES = ("heavy", "large", "small")
# Minimum time between landings, in seconds: leader in the row, follower in
# the column. Wake turbulence behind a heavy aircraft separates a small one
# by over three minutes, a heavy behind a small by one. The values used in
# Balakrishnan and Chandran (2010), from memory (see the plan's references
# item).
SEPARATION = ((96.0, 157.0, 196.0), (60.0, 69.0, 131.0), (60.0, 69.0, 82.0))
NONE = -1  # no aircraft has landed yet
GLOBAL_FIELDS = 9
SLOT_FIELDS = 8


@dataclass(frozen=True)
class RunwayConfig:
    step_seconds: float = 10.0
    arrivals_per_hour: float = 36.0
    # ((end_seconds, arrivals_per_hour), ...): the rate until each end, in
    # turn; after the last, `arrivals_per_hour`. Empty: constant.
    profile: tuple = ()
    mix: tuple = (0.2, 0.7, 0.1)  # share of heavy, large and small aircraft
    separation: tuple = SEPARATION
    notice: int = 0  # ticks between an aircraft's announcement and its ETA
    max_aircraft: int = 24
    reject_penalty: float = 30.0  # minutes per rejected arrival
    disruption: DisruptionConfig = field(default_factory=DisruptionConfig)

    def __post_init__(self):
        k = len(CLASSES)
        if len(self.mix) != k or min(self.mix) < 0 or sum(self.mix) <= 0:
            raise ValueError(f"mix needs {k} non-negative shares")
        if np.shape(self.separation) != (k, k) or np.min(self.separation) <= 0:
            raise ValueError(f"separation must be a {k} x {k} matrix of positive seconds")
        if self.notice < 0 or self.max_aircraft < 1:
            raise ValueError("need notice >= 0 and at least 1 buffer slot")
        if self.disruption.speed != 0:
            raise ValueError("the runway is closed while down (disruption speed 0)")

    @property
    def n_classes(self):
        return len(CLASSES)

    @property
    def n_actions(self):
        return LAND + self.max_aircraft

    @property
    def separation_ticks(self):
        """S in ticks, rounded up: (classes, classes) int."""
        s = np.asarray(self.separation) / self.step_seconds
        return np.ceil(s - 1e-9).astype(int)

    def rates(self, n_ticks):
        """Expected arrivals in each tick: (n_ticks,)."""
        seconds = np.arange(n_ticks) * self.step_seconds
        per_hour = np.full(n_ticks, self.arrivals_per_hour)
        start = 0.0
        for end, rate in self.profile:
            per_hour[(seconds >= start) & (seconds < end)] = rate
            start = end
        return per_hour * self.step_seconds / 3600.0


PRESETS = {"mixed": RunwayConfig()}
# An arrival bank: 54 aircraft an hour for 30 minutes (above any
# sequence's capacity), then 18 an hour. The queue builds and drains.
PRESETS["peak"] = replace(PRESETS["mixed"], profile=((1800.0, 54.0),), arrivals_per_hour=18.0)
# Aircraft announced 10 minutes before their ETA, so the controller can
# plan around aircraft that have not reached the runway yet.
PRESETS["lookahead"] = replace(PRESETS["mixed"], notice=60)
# The runway closes about once an hour for 5 minutes on average.
PRESETS["closure"] = replace(
    PRESETS["mixed"],
    disruption=DisruptionConfig(mode="markov", rate_per_hour=1.0, mean_seconds=300.0),
)


class Exogenous(NamedTuple):
    """One step's random draws: the new arrivals' classes and the disruption draw."""

    cls: np.ndarray  # (n,) int, in arrival order
    onset: np.ndarray | None = None  # (1,)
    duration: np.ndarray | None = None  # (1,)


class RunwaySim:
    def __init__(self, config=None, max_steps=360, rng=None):
        if isinstance(config, str):
            config = PRESETS[config]
        self.config = config if config is not None else RunwayConfig()
        self.max_steps = max_steps
        self.rng = rng if rng is not None else np.random.default_rng()
        self.sep = self.config.separation_ticks
        self.rate = self.config.rates(max_steps)
        self.reset()

    # -- state ---------------------------------------------------------------

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        R = self.config.max_aircraft
        self.t = 0
        self.valid = [False] * R
        self.cls = [0] * R
        self.eta = [0] * R
        self.seq = [0] * R  # arrival number: FCFS order
        self.overtaken = [0] * R  # aircraft that arrived later but landed first
        self.arrived = 0  # sequence numbers handed out
        self.last_class = NONE
        self.last_time = 0
        self.closure = disruption.initial_np(1)
        return self.observe()

    def draw(self):
        c = self.config
        n = self.rng.poisson(self.rate[min(self.t, self.max_steps - 1)])
        mix = np.asarray(c.mix) / sum(c.mix)
        cls = self.rng.choice(c.n_classes, size=n, p=mix).astype(int)
        d = disruption.draw_np(c.disruption, self.rng, 1)
        return Exogenous(cls, d.onset, d.duration)

    # -- queries -------------------------------------------------------------

    @property
    def open(self):
        return not bool(self.closure.down[0])

    def free(self, k):
        """The first tick the separation allows an aircraft of class k."""
        if self.last_class == NONE:
            return 0
        return self.last_time + int(self.sep[self.last_class][k])

    def can_land(self, r):
        """Slot r may land now, apart from the runway being open."""
        if not self.valid[r]:
            return False
        return self.eta[r] <= self.t and self.free(self.cls[r]) <= self.t

    def queue(self):
        """Occupied slots in FCFS order."""
        return sorted(
            (r for r in range(self.config.max_aircraft) if self.valid[r]), key=lambda r: self.seq[r]
        )

    # -- dynamics ------------------------------------------------------------

    def step(self, action, exo=None):
        """One tick: (observation, reward, terminated, truncated, info)."""
        c = self.config
        if exo is None:
            exo = self.draw()
        info = dict.fromkeys(("landed", "delay_ticks", "rejected", "announced", "due", "closed"), 0)
        info["closed"] = int(not self.open)
        a = int(np.asarray(action).reshape(-1)[0])
        if a >= LAND and self.open and self.can_land(a - LAND):
            r = a - LAND
            for q in range(c.max_aircraft):
                if self.valid[q] and self.seq[q] < self.seq[r]:
                    self.overtaken[q] += 1
            self.valid[r] = False
            self.last_class, self.last_time = self.cls[r], self.t
            info["landed"] = 1
            info["delay_ticks"] = self.t - self.eta[r]
        waiting = sum(self.valid[r] and self.eta[r] <= self.t for r in range(c.max_aircraft))
        free = [r for r in range(c.max_aircraft) if not self.valid[r]]
        for n, k in enumerate(exo.cls):
            if n >= len(free):
                info["rejected"] += 1
                continue
            r = free[n]
            self.valid[r], self.cls[r], self.eta[r] = True, int(k), self.t + 1 + c.notice
            self.seq[r], self.overtaken[r] = self.arrived, 0
            self.arrived += 1
            info["announced"] += 1
            info["due"] += self.eta[r] < self.max_steps
        self._advance_closure(exo)
        self.t += 1
        reward = -(waiting * c.step_seconds / 60 + c.reject_penalty * info["rejected"])
        info["waiting"] = waiting
        truncated = self.t >= self.max_steps
        return self.observe(), reward, False, truncated, info

    def _advance_closure(self, exo):
        c = self.config.disruption
        if c.mode == "none":
            return
        onset = np.ones(1) if exo.onset is None else exo.onset
        duration = np.zeros(1) if exo.duration is None else exo.duration
        draw = disruption.DisruptionDraw(onset, duration)
        dt = self.config.step_seconds
        self.closure = disruption.step_np(c, self.closure, draw, np.ones(1, bool), self.t * dt, dt)

    # -- observation ---------------------------------------------------------

    def observe(self):
        """Global fields, then one block per slot.

        Global: the time (over max_steps), the runway open, the last landing's
        class (one-hot, zeros before the first), the ticks until the
        separation allows each class (over the largest separation), and the
        occupied slots (over the buffer size).

        Per slot: occupied, its class (one-hot), the ticks until its ETA and
        the ticks it has been delayed (over 60 ticks), its place in FCFS
        order (over the buffer size), and the ticks until it could land
        (over the largest separation, ignoring closures); zeros if empty.
        """
        c = self.config
        K, R = c.n_classes, c.max_aircraft
        span = float(self.sep.max())
        last = [0.0] * K
        if self.last_class != NONE:
            last[self.last_class] = 1.0
        wait = [max(self.free(k) - self.t, 0) / span for k in range(K)]
        out = [self.t / self.max_steps, float(self.open), *last, *wait, sum(self.valid) / R]
        rank = {r: n for n, r in enumerate(self.queue())}
        for r in range(R):
            if not self.valid[r]:
                out += [0.0] * SLOT_FIELDS
                continue
            onehot = [float(self.cls[r] == k) for k in range(K)]
            ready = max(self.eta[r], self.free(self.cls[r])) - self.t
            out += [
                1.0,
                *onehot,
                max(self.eta[r] - self.t, 0) / 60,
                max(self.t - self.eta[r], 0) / 60,
                rank[r] / R,
                max(ready, 0) / span,
            ]
        return np.array(out, dtype=np.float32)


# -- heuristics ----------------------------------------------------------------


def fcfs(sim):
    """First come, first served: land the earliest arrival as soon as allowed."""
    queue = sim.queue()
    if queue and sim.can_land(queue[0]):
        return [LAND + queue[0]]
    return [WAIT]


def shifts(m, k):
    """The orders of m aircraft (FCFS index per position) that move none by
    more than k places, in lexicographic order (so FCFS comes first)."""
    return [
        p for p in itertools.permutations(range(m)) if all(abs(i - n) <= k for n, i in enumerate(p))
    ]


def overtakes(order):
    """For each FCFS index, how many later aircraft come before it in `order`."""
    before = [0] * len(order)
    for n, i in enumerate(order):
        before[i] = sum(j > i for j in order[:n])
    return before


def plan_cost(sim, slots, order, tail):
    """(cost, first landing tick) of landing `slots` in `order`, each as early
    as the ETA and separation allow (closures ignored). The cost is the sum of
    landing ticks plus `tail` times the last: the aircraft behind the window
    wait for it."""
    last, at = sim.last_class, sim.last_time
    total, first = 0, None
    for i in order:
        r = slots[i]
        k = sim.cls[r]
        t = max(sim.t, sim.eta[r])
        if last != NONE:
            t = max(t, at + int(sim.sep[last][k]))
        total += t
        last, at = k, t
        first = t if first is None else first
    return total + tail * at, first


def cps(sim, k=2, window=5):
    """Constrained position shifting, online: plan the first `window`
    aircraft in FCFS order, over the orders that move none by more than k
    places and overtake no aircraft beyond k times in all; take the cheapest
    (`plan_cost`; FCFS on a tie) and land its first aircraft if it can go
    now. FCFS order is always allowed (it overtakes nobody), even when
    another controller has overtaken aircraft more. k = 0 is FCFS."""
    queue = sim.queue()
    if not queue:
        return [WAIT]
    slots = queue[:window]
    m = len(slots)
    tail = len(queue) - m
    best = None
    for order in shifts(m, k):
        before = overtakes(order)
        if any(before[i] and sim.overtaken[slots[i]] + before[i] > k for i in range(m)):
            continue
        cost, first = plan_cost(sim, slots, order, tail)
        if best is None or cost < best[0]:
            best = (cost, first, slots[order[0]])
    _, first, r = best
    return [LAND + r] if first == sim.t else [WAIT]


def wait(sim):
    return [WAIT]


__all__ = ["PRESETS", "RunwayConfig", "RunwaySim", "cps", "fcfs", "wait"]


if __name__ == "__main__":  # a quick look: python -m transit_bench.envs.runway.sim
    for name in PRESETS:
        for rule in (fcfs, cps):
            total = 0.0
            for seed in range(20):
                sim = RunwaySim(name)
                sim.reset(seed)
                done = False
                while not done:
                    _, reward, _, done, _ = sim.step(rule(sim))
                    total += reward
            print(f"{name:10s} {rule.__name__:5s} {total / 20:8.1f}")
