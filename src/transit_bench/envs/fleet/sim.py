"""The fleet env's Python reference sim: on-demand vehicles on a grid.

The slow, readable ground truth that the JAX env is tested against step for
step (given the same random draws). See docs/formulation.md, "Fleet".

Model, in ticks of `step_seconds` (one grid edge per tick):
- Nodes of a `grid` x `grid` lattice, numbered y * grid + x. Travel time is
  the Manhattan distance; a vehicle moves along x first, then y.
- Requests arrive as a Poisson process (`request_rate` per tick in all),
  with an origin and a destination drawn from the demand pattern, into a
  buffer of `max_requests` slots (in slot order; arrivals that find the
  buffer full are rejected). A request waits until a vehicle picks it up;
  one that is still unassigned after `patience` ticks abandons.
- Each vehicle carries one request at a time. An idle vehicle is
  *deciding*: it may STAY, MOVE one node north, east, south or west
  (repositioning; a move off the grid stays), or TAKE request slot r
  (action 5 + r), if that request is waiting and nobody has taken it.
  Vehicles act in index order, so if two take the same request the earlier
  one gets it and the later one stays. A taken request no longer abandons.
- A vehicle with a job moves one node a tick towards its goal: the
  request's origin, then (once it has picked up there, in the tick it
  arrives) its destination, where it drops off and becomes idle. A vehicle
  that takes a request at its own node picks up at once.

Reward: minus the passenger-minutes spent waiting this tick (requests
waiting at the end of the vehicles' turns) and `abandon_penalty` minutes
for each request that abandons or is rejected. In-vehicle time is fixed
once a request is taken (the Manhattan distance), so only waiting and
losing requests are priced.
"""

from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np

STAY, NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3, 4
TAKE = 5  # action TAKE + r takes request slot r
IDLE, TO_PICKUP, TO_DROPOFF = 0, 1, 2
MOVES = {NORTH: (0, 1), EAST: (1, 0), SOUTH: (0, -1), WEST: (-1, 0)}
DEMANDS = ("uniform", "commute")
HEAD_FIELDS = 6
SLOT_FIELDS = 6
NODE_FIELDS = 2


@dataclass(frozen=True)
class FleetConfig:
    grid: int = 5
    n_vehicles: int = 4
    step_seconds: float = 60.0
    request_rate: float = 0.5  # requests per tick, over the whole grid
    demand: str = "uniform"
    patience: int = 8  # ticks an unassigned request waits before abandoning
    max_requests: int = 16  # request buffer slots
    abandon_penalty: float = 10.0  # minutes per abandoned or rejected request

    def __post_init__(self):
        if self.demand not in DEMANDS:
            raise ValueError(f"demand must be one of {DEMANDS}, got {self.demand!r}")
        if self.grid < 2 or self.n_vehicles < 1 or self.max_requests < 1:
            raise ValueError("need a grid of at least 2, 1 vehicle and 1 request slot")
        if self.patience < 1:
            raise ValueError("patience must be at least 1 tick")

    @property
    def n_nodes(self):
        return self.grid * self.grid

    @property
    def n_actions(self):
        return TAKE + self.max_requests


PRESETS = {"grid": FleetConfig()}
# Commuting: trips start mostly in the west and end mostly in the east, so
# vehicles drift east with their passengers and must come back empty: the
# case for repositioning.
PRESETS["commute"] = replace(PRESETS["grid"], demand="commute")
# A bigger city: an 8 x 8 grid, 10 vehicles, 1.2 requests a tick.
PRESETS["city"] = replace(
    PRESETS["commute"], grid=8, n_vehicles=10, request_rate=1.2, max_requests=32
)


def demand_weights(config):
    """(origin weights (nodes,), destination weights (nodes, nodes)), unnormalised.

    "uniform": any origin, any other node as the destination. "commute":
    origins weighted by grid - x (west), destinations by x + 1 (east).
    """
    g = config.grid
    x = np.arange(config.n_nodes) % g
    if config.demand == "uniform":
        origin = np.ones(config.n_nodes)
        dest = np.ones(config.n_nodes)
    else:
        origin = (g - x).astype(float)
        dest = (x + 1).astype(float)
    joint = np.tile(dest, (config.n_nodes, 1))
    np.fill_diagonal(joint, 0.0)
    return origin, joint


def distance(config, a, b):
    """Manhattan distance between nodes a and b (arrays or ints)."""
    g = config.grid
    return abs(a % g - b % g) + abs(a // g - b // g)


def toward(config, a, b):
    """The node after a on the way to b: along x first, then y."""
    g = config.grid
    ax, ay, bx, by = a % g, a // g, b % g, b // g
    if ax != bx:
        return a + (1 if bx > ax else -1)
    if ay != by:
        return a + (g if by > ay else -g)
    return a


class Exogenous(NamedTuple):
    """One step's random draws: the new requests, in arrival order."""

    origin: np.ndarray  # (n,) int
    dest: np.ndarray  # (n,) int


class FleetSim:
    def __init__(self, config=None, max_steps=240, rng=None):
        if isinstance(config, str):
            config = PRESETS[config]
        self.config = config if config is not None else FleetConfig()
        self.max_steps = max_steps
        self.rng = rng if rng is not None else np.random.default_rng()
        self.origin_weights, self.joint = demand_weights(self.config)
        self.reset()

    # -- state ---------------------------------------------------------------

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.config
        V, R = c.n_vehicles, c.max_requests
        self.t = 0
        # Spread over the grid in node order.
        self.pos = [v * c.n_nodes // V for v in range(V)]
        self.phase = [IDLE] * V
        self.goal = [0] * V
        self.job = [-1] * V  # the slot it is going to pick up from
        self.valid = [False] * R
        self.origin = [0] * R
        self.dest = [0] * R
        self.age = [0] * R  # ticks waited
        self.taken = [-1] * R  # the vehicle that took it, or -1
        return self.observe()

    def draw(self):
        c = self.config
        n = self.rng.poisson(c.request_rate)
        p = self.origin_weights / self.origin_weights.sum()
        origin = self.rng.choice(c.n_nodes, size=n, p=p)
        dest = np.array(
            [self.rng.choice(c.n_nodes, p=self.joint[o] / self.joint[o].sum()) for o in origin],
            dtype=int,
        )
        return Exogenous(origin.astype(int), dest)

    # -- queries -------------------------------------------------------------

    def deciding(self, v):
        return self.phase[v] == IDLE

    def available(self, r):
        return self.valid[r] and self.taken[r] < 0

    # -- dynamics ------------------------------------------------------------

    def step(self, actions, exo=None):
        """One tick: (observation, reward, terminated, truncated, info)."""
        c = self.config
        if exo is None:
            exo = self.draw()
        info = dict.fromkeys(
            ("picked_up", "delivered", "abandoned", "rejected", "wait_ticks", "moved"), 0
        )
        deciding = [self.deciding(v) for v in range(c.n_vehicles)]
        for v in range(c.n_vehicles):
            a = int(actions[v])
            if deciding[v] and a in MOVES:
                dx, dy = MOVES[a]
                x, y = self.pos[v] % c.grid + dx, self.pos[v] // c.grid + dy
                if 0 <= x < c.grid and 0 <= y < c.grid:
                    self.pos[v] = y * c.grid + x
                    info["moved"] += 1
            elif deciding[v] and a >= TAKE and self.available(a - TAKE):
                r = a - TAKE
                self.taken[r] = v
                self.job[v] = r
                self.phase[v] = TO_PICKUP
                self.goal[v] = self.origin[r]
            if self.phase[v] == IDLE:
                continue
            self.pos[v] = toward(c, self.pos[v], self.goal[v])
            if self.pos[v] != self.goal[v]:
                continue
            if self.phase[v] == TO_PICKUP:
                r = self.job[v]
                info["picked_up"] += 1
                info["wait_ticks"] += self.age[r]
                self.valid[r] = False
                self.taken[r] = -1
                self.job[v] = -1
                self.phase[v] = TO_DROPOFF
                self.goal[v] = self.dest[r]
            else:
                info["delivered"] += 1
                self.phase[v] = IDLE
        waiting = sum(self.valid)
        for r in range(c.max_requests):
            if not self.valid[r]:
                continue
            self.age[r] += 1
            if self.taken[r] < 0 and self.age[r] >= c.patience:
                self.valid[r] = False
                info["abandoned"] += 1
        free = [r for r in range(c.max_requests) if not self.valid[r]]
        for n, (o, d) in enumerate(zip(exo.origin, exo.dest, strict=True)):
            if n >= len(free):
                info["rejected"] += 1
                continue
            r = free[n]
            self.valid[r], self.origin[r], self.dest[r] = True, int(o), int(d)
            self.age[r], self.taken[r] = 0, -1
        self.t += 1
        lost = info["abandoned"] + info["rejected"]
        reward = -(waiting * c.step_seconds / 60 + c.abandon_penalty * lost)
        info["waiting"] = waiting
        truncated = self.t >= self.max_steps
        return self.observe(), reward, False, truncated, info

    # -- observation ---------------------------------------------------------

    def observe(self):
        """One block per vehicle.

        Head: x and y (over grid - 1), idle, going to a pickup, going to a
        dropoff, distance to its goal (over the grid's diameter, 0 if idle).
        Then per request slot: available (waiting and not taken), the
        origin's offset from the vehicle in x and y (over grid - 1), the
        distance to it and the trip's length (over the diameter), and the
        time it has waited (over patience); zeros if not available. Then per
        node: other idle vehicles there (over the fleet size) and available
        requests starting there (over the buffer size).
        """
        c = self.config
        g, span = c.grid, c.grid - 1
        diameter = 2 * span
        idle_at = np.zeros(c.n_nodes)
        for v in range(c.n_vehicles):
            if self.phase[v] == IDLE:
                idle_at[self.pos[v]] += 1
        waiting_at = np.zeros(c.n_nodes)
        for r in range(c.max_requests):
            if self.available(r):
                waiting_at[self.origin[r]] += 1
        out = []
        for v in range(c.n_vehicles):
            p = self.pos[v]
            busy = self.phase[v] != IDLE
            head = [
                (p % g) / span,
                (p // g) / span,
                float(self.phase[v] == IDLE),
                float(self.phase[v] == TO_PICKUP),
                float(self.phase[v] == TO_DROPOFF),
                distance(c, p, self.goal[v]) / diameter if busy else 0.0,
            ]
            slots = []
            for r in range(c.max_requests):
                if not self.available(r):
                    slots += [0.0] * SLOT_FIELDS
                    continue
                o = self.origin[r]
                slots += [
                    1.0,
                    (o % g - p % g) / span,
                    (o // g - p // g) / span,
                    distance(c, p, o) / diameter,
                    distance(c, o, self.dest[r]) / diameter,
                    self.age[r] / c.patience,
                ]
            others = idle_at.copy()
            if self.phase[v] == IDLE:
                others[p] -= 1
            nodes = np.stack([others / c.n_vehicles, waiting_at / c.max_requests], axis=1)
            out.append(np.concatenate([head, slots, nodes.ravel()]))
        return np.array(out, dtype=np.float32).ravel()


# -- heuristics ----------------------------------------------------------------


def nearest_idle(sim):
    """Each waiting request, longest-waiting first (lowest slot on a tie),
    takes the nearest idle vehicle not yet chosen (lowest index on a tie).
    Unchosen vehicles stay."""
    c = sim.config
    actions = [STAY] * c.n_vehicles
    free = [v for v in range(c.n_vehicles) if sim.deciding(v)]
    order = sorted((-sim.age[r], r) for r in range(c.max_requests) if sim.available(r))
    for _, r in order:
        if not free:
            break
        v = min(free, key=lambda v: (distance(c, sim.pos[v], sim.origin[r]), v))
        actions[v] = TAKE + r
        free.remove(v)
    return actions


def assignment_costs(config, pos, deciding, origin, available):
    """Integer costs for the per-tick matching: vehicles x (requests + dummies).

    A real request costs the pickup distance; an unavailable one is
    forbidden (None); each of the n_vehicles + 1 dummy objects ("stay")
    costs more than any distance, so the matching serves as many requests
    as it can and then minimises the total pickup distance.
    """
    big = 2 * (config.grid - 1) + 1
    V, R = config.n_vehicles, config.max_requests
    cost = [[None] * (R + V + 1) for _ in range(V)]
    for v in range(V):
        if not deciding[v]:
            continue
        for r in range(R):
            if available[r]:
                cost[v][r] = int(distance(config, pos[v], origin[r]))
        for j in range(R, R + V + 1):
            cost[v][j] = big
    return cost


def auction(cost):
    """Minimum-cost assignment of bidders (rows) to objects (columns) by
    Bertsekas' auction algorithm, in integers.

    Rows that are all None do not bid. Benefits are -cost x (n + 1) for n
    bidders, and every bid raises a price by at least 1, which is less than
    1/n in cost units, so the assignment is optimal (Bertsekas 1988). Each
    round, every unassigned bidder bids for its best object at the price
    that makes it indifferent with its second best, plus 1; each object
    goes to its highest bidder (lowest index on a tie), and the bidder it
    had is unassigned. Returns each row's object (-1 for non-bidders).
    """
    rows = len(cost)
    bidders = [i for i in range(rows) if any(c is not None for c in cost[i])]
    scale = len(bidders) + 1
    cols = len(cost[0]) if rows else 0
    price = [0] * cols
    owner = [-1] * cols
    assigned = [-1] * rows
    while any(assigned[i] < 0 for i in bidders):
        bids = {}  # object -> (bid, bidder)
        for i in bidders:
            if assigned[i] >= 0:
                continue
            values = [
                (-cost[i][j] * scale - price[j], j) for j in range(cols) if cost[i][j] is not None
            ]
            values.sort(key=lambda vj: (-vj[0], vj[1]))
            best, j = values[0]
            second = values[1][0]
            bid = price[j] + best - second + 1
            if j not in bids or bid > bids[j][0]:
                bids[j] = (bid, i)
        for j, (bid, i) in bids.items():
            if owner[j] >= 0:
                assigned[owner[j]] = -1
            owner[j], price[j], assigned[i] = i, bid, j
    return assigned


def matching(sim):
    """Per-tick bipartite matching: idle vehicles to waiting requests,
    serving as many as possible at the least total pickup distance (an
    exact assignment by `auction`). Vehicles matched to nothing stay."""
    c = sim.config
    deciding = [sim.deciding(v) for v in range(c.n_vehicles)]
    available = [sim.available(r) for r in range(c.max_requests)]
    cost = assignment_costs(c, sim.pos, deciding, sim.origin, available)
    assigned = auction(cost)
    return [TAKE + j if 0 <= j < c.max_requests else STAY for j in assigned]


def homes(config):
    """Each vehicle's home node: the origin-weight quantiles (v + 1/2) / V, so
    homes spread over the grid in proportion to where trips start."""
    weights, _ = demand_weights(config)
    cdf = np.cumsum(weights) / weights.sum()
    V = config.n_vehicles
    return [int(np.searchsorted(cdf, (v + 0.5) / V)) for v in range(V)]


def rebalance(sim, rule=matching):
    """`rule`, plus: an idle vehicle that `rule` leaves where it is heads one
    node towards its home (`homes`). Static rebalancing to where demand
    starts, the simplest repositioning policy."""
    c = sim.config
    home = homes(c)
    actions = list(rule(sim))
    step = {1: EAST, -1: WEST, c.grid: NORTH, -c.grid: SOUTH}
    for v, a in enumerate(actions):
        if a == STAY and sim.deciding(v) and sim.pos[v] != home[v]:
            actions[v] = step[toward(c, sim.pos[v], home[v]) - sim.pos[v]]
    return actions


def stay(sim):
    return [STAY] * sim.config.n_vehicles


__all__ = ["FleetConfig", "FleetSim", "PRESETS", "matching", "nearest_idle", "rebalance", "stay"]
