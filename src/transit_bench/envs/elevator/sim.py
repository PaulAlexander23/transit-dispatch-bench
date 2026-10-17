"""The elevator env's Python reference sim: a configurable multi-lift building.

Ported from elevator-dispatching-rl (`building.py` at 206b740) unchanged in
behaviour: with the same seed it draws the same random numbers and produces
the same trajectory (scripts/parity_old_repo.py). It is the slow, readable
ground truth that the JAX env is tested against step for step.

`ElevatorConfig` switches each feature on or off. The default config is 10
floors, one lift and uniform arrivals.

Features:
- `n_floors`, `n_lifts`: building size. Lifts act in index order within a
  step, so a lower-numbered lift boards passengers first.
- `hall_calls`: waiting passengers press an up or down button. A lift serves
  one direction at a time ("serve up" / "serve down"), and only passengers
  going that way board.
- `traffic`: arrival pattern. "uniform" is the original model; the others mix
  up-peak (lobby to upper floors), interfloor and down-peak (upper floors to
  the lobby) traffic, and "day" moves from morning up-peak through lunch to
  the evening down-peak over one episode.
- `kinematics`: lifts accelerate, cruise and brake between floors, and doors
  take time to open and close. Each step is `step_seconds` long and is
  integrated in `substeps`. A lift only takes a new action when it is idle
  (stopped with its doors closed), except that a lift already travelling can
  be told to carry on one floor further in the same direction.
"""

from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

TRAFFIC_PROFILES = ("uniform", "up_peak", "down_peak", "lunch", "day")

# Mixture weights over the (up-peak, interfloor, down-peak) patterns.
_PROFILE_WEIGHTS = {
    "uniform": (0.0, 1.0, 0.0),
    "up_peak": (0.8, 0.2, 0.0),
    "down_peak": (0.0, 0.2, 0.8),
    "lunch": (0.4, 0.2, 0.4),
}
# "day": keyframes of (episode fraction, weights), linearly interpolated.
_DAY_KEYFRAMES = (
    (0.0, _PROFILE_WEIGHTS["up_peak"]),
    (0.3, _PROFILE_WEIGHTS["up_peak"]),
    (0.5, _PROFILE_WEIGHTS["lunch"]),
    (0.7, (0.0, 1.0, 0.0)),
    (0.9, _PROFILE_WEIGHTS["down_peak"]),
    (1.0, _PROFILE_WEIGHTS["down_peak"]),
)

# A door timer this fraction of a substep from zero counts as zero.
DOOR_SNAP = 1e-3
# Lift status, as reported in observations.
IDLE, MOVING, DOORS = 0, 1, 2
# Actions per lift. SERVE opens the doors for everyone; with hall calls it is
# replaced by SERVE_UP and SERVE_DOWN.
UP, DOWN, SERVE = 0, 1, 2
SERVE_UP, SERVE_DOWN = 2, 3


@dataclass(frozen=True)
class ElevatorConfig:
    n_floors: int = 10
    n_lifts: int = 1
    lift_capacity: int = 8
    # Mean arrivals per floor per step; the building total is this x n_floors.
    arrival_probability: float = 0.1
    traffic: str = "uniform"
    hall_calls: bool = False
    kinematics: bool = False
    # Kinematics only:
    floor_height: float = 3.5  # m
    max_speed: float = 2.5  # m/s
    acceleration: float = 1.0  # m/s^2, also used for braking
    door_time: float = 4.0  # s, open + dwell + close
    step_seconds: float = 1.0
    substeps: int = 10

    def __post_init__(self):
        if self.traffic not in TRAFFIC_PROFILES:
            raise ValueError(f"traffic must be one of {TRAFFIC_PROFILES}, got {self.traffic!r}")
        if self.n_floors < 2 or self.n_lifts < 1 or self.lift_capacity < 1:
            raise ValueError("need at least 2 floors, 1 lift and a capacity of 1")
        if self.substeps < 1:
            raise ValueError("substeps must be at least 1")

    @property
    def n_actions(self):
        """Actions per lift."""
        return 4 if self.hall_calls else 3


# Each preset adds one feature to the one before, matching the sub-phases of
# the plan, so the cost of each feature can be measured on its own.
PRESETS = {"original": ElevatorConfig()}
PRESETS["tall"] = replace(PRESETS["original"], n_floors=30)
PRESETS["multi"] = replace(PRESETS["tall"], n_lifts=4)
PRESETS["kinematic"] = replace(PRESETS["multi"], kinematics=True)
PRESETS["hall_calls"] = replace(PRESETS["kinematic"], hall_calls=True)
PRESETS["full"] = replace(PRESETS["hall_calls"], traffic="day")


@dataclass
class Lift:
    # Kinematic quantities are in floors and seconds; without kinematics the
    # position is always a whole floor and the velocity is zero.
    position: float = 0.0
    velocity: float = 0.0
    target: int = 0
    door_timer: float = 0.0
    # Destination floor of each passenger on board.
    passengers: list[int] = field(default_factory=list)

    @property
    def floor(self):
        """The floor the lift is at or nearest to."""
        return int(round(self.position))

    @property
    def status(self):
        if self.door_timer > 0:
            return DOORS
        if self.position != self.target or self.velocity != 0:
            return MOVING
        return IDLE


@dataclass
class ElevatorState:
    lifts: list[Lift]
    # Destination floor of each waiting passenger, in order of arrival.
    floor_queues: list[deque[int]]


class ElevatorSim:
    def __init__(self, config=None, rng=None):
        self.config = config if config is not None else ElevatorConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        self._patterns = _traffic_patterns(self.config.n_floors)
        c = self.config
        # Kinematic limits in floors and seconds.
        self._max_speed = c.max_speed / c.floor_height
        self._acceleration = c.acceleration / c.floor_height
        self._dt = c.step_seconds / c.substeps
        self._state = None
        self.reset()

    def reset(self):
        c = self.config
        self._state = ElevatorState(
            lifts=[Lift() for _ in range(c.n_lifts)],
            floor_queues=[deque() for _ in range(c.n_floors)],
        )

    def state(self):
        return self._state

    def sample_passengers(self, episode_fraction=0.0):
        """Add this step's arrivals. `episode_fraction` drives the "day" profile."""
        self.add_arrivals(self.draw_arrivals(episode_fraction))

    def draw_arrivals(self, episode_fraction=0.0):
        """This step's arrivals as (floor, destination) pairs, in queueing order."""
        c = self.config
        arrivals = []
        if c.traffic == "uniform":
            for floor in range(c.n_floors):
                if self.rng.random() < c.arrival_probability:
                    destination = int(self.rng.integers(c.n_floors - 1))
                    if destination >= floor:
                        destination += 1
                    arrivals.append((floor, destination))
            return arrivals

        # Poisson counts, since a busy lobby can see several arrivals a step.
        rates, destinations = self.arrival_model(episode_fraction)
        counts = self.rng.poisson(rates)
        for floor in np.flatnonzero(counts):
            chosen = self.rng.choice(c.n_floors, size=counts[floor], p=destinations[floor])
            arrivals.extend((int(floor), int(d)) for d in chosen)
        return arrivals

    def add_arrivals(self, arrivals):
        queues = self._state.floor_queues
        for floor, destination in arrivals:
            queues[floor].append(destination)

    def arrival_model(self, episode_fraction=0.0):
        """Expected arrivals per floor this step, and each floor's destination distribution."""
        c = self.config
        weights = np.asarray(_traffic_weights(c.traffic, episode_fraction))
        origins, destinations = self._patterns
        total = c.arrival_probability * c.n_floors
        # mix[k, f]: share of all arrivals that come from pattern k at floor f
        mix = weights[:, None] * origins
        rates = total * mix.sum(axis=0)
        joint = (mix[:, :, None] * destinations).sum(axis=0)
        sums = joint.sum(axis=1, keepdims=True)
        joint = np.divide(joint, sums, out=np.full_like(joint, 1 / c.n_floors), where=sums > 0)
        return rates, joint

    def apply_actions(self, actions):
        """Apply one action per lift, in lift order, then advance time.

        Returns per-lift (delivered, boarded) counts and whether each lift
        opened its doors this step.
        """
        c = self.config
        delivered = [0] * c.n_lifts
        boarded = [0] * c.n_lifts
        served = [False] * c.n_lifts
        for i, (lift, action) in enumerate(zip(self._state.lifts, actions, strict=True)):
            action = int(action)
            if action in (UP, DOWN):
                self._move(lift, 1 if action == UP else -1)
            elif not c.kinematics or lift.status == IDLE:
                direction = 0
                if c.hall_calls:
                    direction = 1 if action == SERVE_UP else -1
                delivered[i], boarded[i] = self._serve(lift, direction)
                served[i] = True
                if c.kinematics:
                    lift.door_timer = c.door_time
        if c.kinematics:
            for _ in range(c.substeps):
                for lift in self._state.lifts:
                    self._integrate(lift)
        return delivered, boarded, served

    def _move(self, lift, direction):
        top = self.config.n_floors - 1
        if not self.config.kinematics:
            lift.position = float(min(max(lift.floor + direction, 0), top))
            lift.target = lift.floor
            return
        status = lift.status
        if status == IDLE:
            lift.target = min(max(lift.floor + direction, 0), top)
        elif status == MOVING:
            # Carry on one floor further, but only while the current target is
            # the next floor ahead, and never reverse mid-flight.
            ahead = (lift.target - lift.position) * direction
            if 0 < ahead <= 1:
                lift.target = min(max(lift.target + direction, 0), top)

    def _serve(self, lift, direction=0):
        """Drop off at this floor, then board in arrival order until full.

        direction 0 boards everyone; +1 / -1 boards only those going up / down.
        """
        floor = lift.floor
        n_before = len(lift.passengers)
        lift.passengers = [d for d in lift.passengers if d != floor]
        delivered = n_before - len(lift.passengers)

        queue = self._state.floor_queues[floor]
        space = self.config.lift_capacity - len(lift.passengers)
        if direction == 0:
            n_board = min(space, len(queue))
            lift.passengers.extend(queue.popleft() for _ in range(n_board))
            return delivered, n_board

        staying = deque()
        n_board = 0
        while queue:
            destination = queue.popleft()
            if n_board < space and (destination - floor) * direction > 0:
                lift.passengers.append(destination)
                n_board += 1
            else:
                staying.append(destination)
        self._state.floor_queues[floor] = staying
        return delivered, n_board

    def _integrate(self, lift):
        """Advance one lift by one substep: doors, then a bang-bang profile."""
        dt = self._dt
        if lift.door_timer > 0:
            lift.door_timer = _count_down(lift.door_timer, dt)
            return
        distance = lift.target - lift.position
        if distance == 0 and lift.velocity == 0:
            return
        sign = 1.0 if distance > 0 else -1.0
        distance = abs(distance)
        if distance < 1e-9:
            lift.position = float(lift.target)
            lift.velocity = 0.0
            return
        speed = lift.velocity * sign
        braking_distance = speed * speed / (2 * self._acceleration)
        if distance <= braking_distance:
            # Constant deceleration that stops exactly on the target.
            deceleration = speed * speed / (2 * distance)
            new_speed = max(speed - deceleration * dt, 0.0)
        else:
            new_speed = min(speed + self._acceleration * dt, self._max_speed)
        travelled = (speed + new_speed) / 2 * dt
        if travelled >= distance or new_speed == 0:
            lift.position = float(lift.target)
            lift.velocity = 0.0
        else:
            lift.position += sign * travelled
            lift.velocity = sign * new_speed

    def render(self):
        print(self)

    def __repr__(self):
        c = self.config
        state = self._state
        cells = []
        for lift in state.lifts:
            cell = "[" + " ".join(str(d) for d in lift.passengers) + "]"
            cells.append(cell)
        width = max(len(cell) for cell in cells)
        lines = []
        for floor in range(c.n_floors - 1, -1, -1):
            row = [
                cell.ljust(width) if lift.floor == floor else " " * width
                for lift, cell in zip(state.lifts, cells, strict=True)
            ]
            waiting = " ".join(str(d) for d in reversed(state.floor_queues[floor]))
            lines.append(f"{floor:3d} " + "  ".join(row) + " | " + waiting)
        return "\n".join(lines)


def _count_down(timer, dt):
    """A timer one substep later, snapped to zero within a thousandth of `dt`.

    door_time / dt substeps should empty it exactly, but dt (0.1 s) is not
    exact in binary: in float32 four seconds less forty 0.1s leaves 1.6e-6,
    which held the doors one whole step too long (5 steps per stop instead
    of 4) and cost the JAX env about 7% of its deliveries on `full`. In
    float64 the remainder is negative and clipped, so the snap changes
    nothing there.
    """
    timer = max(timer - dt, 0.0)
    return 0.0 if timer < DOOR_SNAP * dt else timer


def _traffic_patterns(n_floors):
    """Origin and destination distributions of the three canonical patterns.

    origins[k, f]: chance an arrival of pattern k starts at floor f.
    destinations[k, f, g]: chance it goes from f to g. Floor 0 is the lobby.
    """
    origins = np.zeros((3, n_floors))
    destinations = np.zeros((3, n_floors, n_floors))
    upper = 1 / (n_floors - 1)
    # Up-peak: lobby to any upper floor.
    origins[0, 0] = 1.0
    destinations[0, 0, 1:] = upper
    # Interfloor: any floor to any other floor.
    origins[1, :] = 1 / n_floors
    destinations[1] = (1 - np.eye(n_floors)) * upper
    # Down-peak: any upper floor to the lobby.
    origins[2, 1:] = upper
    destinations[2, 1:, 0] = 1.0
    return origins, destinations


def _traffic_weights(traffic, episode_fraction):
    if traffic != "day":
        return _PROFILE_WEIGHTS[traffic]
    t = min(max(episode_fraction, 0.0), 1.0)
    for (t0, w0), (t1, w1) in zip(_DAY_KEYFRAMES, _DAY_KEYFRAMES[1:], strict=False):
        if t <= t1:
            a = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
            return tuple((1 - a) * x + a * y for x, y in zip(w0, w1, strict=True))
    return _DAY_KEYFRAMES[-1][1]
