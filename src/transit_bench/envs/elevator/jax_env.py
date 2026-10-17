"""The elevator env in JAX: pure functions over fixed-shape arrays.

Ported from elevator-dispatching-rl (`jax_env.py` at 206b740) onto the
suite's API (`core.types`): `reset(key) -> (state, TimeStep)` and
`step(state, action) -> (state, TimeStep)`. Arrivals use the core's Poisson
draw (`core.demand`), batching the core's `make_vec_env`. The "box"
observation was dropped.

`ElevatorEnv` holds the static settings (config, shaping, observation
type); its methods are pure functions of an `EnvState` and a PRNG key, so
they can be `jit`-compiled and `vmap`-ped over thousands of envs, and fused
with the policy and the PPO update into one XLA program (see `algos.ppo`).

What differs from the Python reference (`sim`, `gym_env`):
- Floor queues are `(n_floors, MAX_QUEUE)` int8 arrays of destinations, in
  arrival order, with a length per floor; arrivals beyond MAX_QUEUE are
  dropped and counted. The queues are most of the state (15 KB of the full
  preset's 16 KB per env), so a step touches only the rows it changes, and
  the number going up per floor is kept as a running count rather than
  recounted from the queues for every observation. Each step adds at most
  MAX_ARRIVALS per floor (a Poisson count above that, about 1e-10 likely at
  the busiest lobby, is clipped).
- A lift holds a count of passengers per destination floor instead of a
  list: their order never affects observations or rewards.
- No data-dependent control flow: boarding "in arrival order until full, only
  those going my way" is a cumulative sum over the queue, and the passengers
  left behind are compacted with a scatter. Lifts still act one after the
  other (a lower-numbered lift boards first), through a loop over lifts that
  is unrolled when traced.
- Random draws use `jax.random`, so they match NumPy only in distribution;
  given the same arrivals the dynamics match Python (tests/test_jax_env.py).

Floats are float32 unless `jax_enable_x64` is on. The parity tests turn it on:
in float32 the kinematics round differently, so trajectories drift from
Python's while staying statistically the same.

Both action modes ("step" and "target") and both observations ("custom",
"relative") are ported. The "custom" observation is
returned as float32 holding small integers; a learner one-hot encodes it
(`one_hot`), as SB3 does for a MultiDiscrete space.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from transit_bench.core import demand
from transit_bench.core.obs import one_hot  # noqa: F401 (re-exported)
from transit_bench.core.types import TimeStep
from transit_bench.envs.elevator.gym_env import (
    RELATIVE_FLOOR_FIELDS,
    RELATIVE_LIFT_FIELDS,
    SHAPINGS,
)
from transit_bench.envs.elevator.sim import (
    _DAY_KEYFRAMES,
    _PROFILE_WEIGHTS,
    DOOR_SNAP,
    DOORS,
    DOWN,
    IDLE,
    MOVING,
    PRESETS,
    SERVE,
    SERVE_DOWN,
    SERVE_UP,
    UP,
    ElevatorConfig,
    _traffic_patterns,
)

MAX_QUEUE = 512  # as the C++ port's kMaxQueue
MAX_ARRIVALS = 16  # per floor per step
WARMUP_ROUNDS = 10  # arrival rounds drawn at reset, as ElevatorGymEnv.reset
OBS_TYPES = ("custom", "relative")
ACTION_MODES = ("step", "target")


class EnvState(NamedTuple):
    position: jax.Array  # (lifts,) float, in floors
    velocity: jax.Array  # (lifts,) float, in floors per second
    target: jax.Array  # (lifts,) int
    door_timer: jax.Array  # (lifts,) float, seconds
    passengers: jax.Array  # (lifts, floors) int: passengers on board per destination
    queue: jax.Array  # (floors, MAX_QUEUE) int8: waiting passengers' destinations
    queue_len: jax.Array  # (floors,) int
    queue_up: jax.Array  # (floors,) int: how many in each queue are going up
    direction: jax.Array  # (lifts,) int: last direction of travel, 0 / 1 / -1
    has_goal: jax.Array  # (lifts,) bool: target mode, a goal not yet served
    goal_floor: jax.Array  # (lifts,) int: target mode, the floor to serve
    goal_dir: jax.Array  # (lifts,) int: target mode, serve going up (1) or down (-1)
    steps: jax.Array  # () int
    last_potential: jax.Array  # () float
    dropped: jax.Array  # () int: arrivals lost to full queues or the per-step cap
    key: jax.Array  # PRNG key for this env's arrivals


class Arrivals(NamedTuple):
    """One step's arrivals: per floor, `count` destinations in queueing order."""

    destination: jax.Array  # (floors, MAX_ARRIVALS) int
    count: jax.Array  # (floors,) int
    overflow: jax.Array = 0  # () int: arrivals drawn past the step's slots, lost


def arrivals_from_list(pairs, n_floors):
    """`Arrivals` from (floor, destination) pairs, as ElevatorSim.draw_arrivals gives."""
    destination = np.zeros((n_floors, MAX_ARRIVALS), np.int32)
    count = np.zeros(n_floors, np.int32)
    for floor, dest in pairs:
        destination[floor, count[floor]] = dest
        count[floor] += 1
    return Arrivals(jnp.asarray(destination), jnp.asarray(count), jnp.zeros((), jnp.int32))


def arrival_slots(config):
    """Slots for one step's arrivals, over all floors, under Poisson traffic.

    The expected total is arrival_probability * n_floors whatever the mix
    (each pattern's origins sum to one, as do the mix's weights): 32 slots on
    `full` (a mean of 3).
    """
    return demand.arrival_slots(config.arrival_probability * config.n_floors)


def _rounded(product):
    """A product rounded before it is added to anything.

    XLA (through LLVM) fuses `a + b * c` into one fused multiply-add, which
    skips the rounding of `b * c` and so differs from NumPy in the last bit,
    enough to flip a kinematics branch later on (the C++ port builds with
    -ffp-contract=off for the same reason). The barrier keeps the two apart.
    """
    return jax.lax.optimization_barrier(product)


class ElevatorEnv:
    """ElevatorGymEnv's dynamics, observations and rewards, as pure JAX functions.

    Takes the same arguments as `ElevatorGymEnv`. The methods take and return
    `EnvState`; nothing is stored on the object. `n_actions` is the number
    of choices per lift in the chosen action mode; `n_agents` is the number
    of lifts.
    """

    def __init__(
        self,
        config=None,
        reward_shaping=False,
        obs_type="relative",
        max_steps=200,
        observe_direction=False,
        action_mode="step",
    ):
        if isinstance(config, str):
            config = PRESETS[config]
        self.config = c = config if config is not None else ElevatorConfig()
        if obs_type not in OBS_TYPES:
            raise ValueError(f"the JAX env supports obs_type {OBS_TYPES}, got {obs_type!r}")
        if action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}, got {action_mode!r}")
        self.action_mode = action_mode
        if reward_shaping is True:
            reward_shaping = SHAPINGS["default"]
        elif isinstance(reward_shaping, str):
            reward_shaping = SHAPINGS[reward_shaping]
        self.reward_shaping = reward_shaping or None
        self.obs_type = obs_type
        self.max_steps = max_steps
        self.n_agents = c.n_lifts
        self.observe_direction = observe_direction
        # The "relative" observation is one block per lift (for MAPPO).
        self.per_agent_obs = obs_type == "relative"
        self.observation_nvec = None  # set below for "custom"
        if action_mode == "target":
            self.n_actions = c.n_floors * (2 if c.hall_calls else 1)
        else:
            self.n_actions = c.n_actions
        # Kinematic limits in floors and seconds, as ElevatorSim.
        self._max_speed = c.max_speed / c.floor_height
        self._acceleration = c.acceleration / c.floor_height
        self._dt = c.step_seconds / c.substeps
        origins, destinations = _traffic_patterns(c.n_floors)
        self._origins = origins
        self._destinations = destinations
        self.arrival_slots = arrival_slots(c)
        if obs_type == "relative":
            self.observation_size = c.n_lifts * (
                RELATIVE_LIFT_FIELDS + (2 * c.n_floors - 1) * RELATIVE_FLOOR_FIELDS
            )
        else:
            lift_fields = 4 if c.kinematics else 2
            n_waiting = (2 if c.hall_calls else 1) * c.n_floors
            n_free = c.n_lifts if action_mode == "target" else 0
            n_dir = c.n_lifts if observe_direction else 0
            self.observation_size = (
                c.n_lifts * lift_fields + n_waiting + c.n_lifts * c.n_floors + n_free + n_dir
            )
            # Choices per entry, as ElevatorGymEnv's MultiDiscrete observation_space.nvec.
            per_lift = [c.n_floors, c.lift_capacity + 1]
            if c.kinematics:
                per_lift += [c.n_floors, 3]
            nvec = np.full(self.observation_size, c.lift_capacity + 1)
            nvec[: c.n_lifts * lift_fields] = np.tile(per_lift, c.n_lifts)
            if n_free:
                nvec[len(nvec) - n_dir - n_free : len(nvec) - n_dir] = 2
            if n_dir:
                nvec[-n_dir:] = 3
            self.observation_nvec = nvec

    # -- state ---------------------------------------------------------------

    def _float(self):
        return jnp.zeros(()).dtype  # float64 with jax_enable_x64, else float32

    def empty_state(self, key):
        c, f = self.config, self._float()
        lifts = c.n_lifts
        return EnvState(
            position=jnp.zeros(lifts, f),
            velocity=jnp.zeros(lifts, f),
            target=jnp.zeros(lifts, jnp.int32),
            door_timer=jnp.zeros(lifts, f),
            passengers=jnp.zeros((lifts, c.n_floors), jnp.int32),
            queue=jnp.zeros((c.n_floors, MAX_QUEUE), jnp.int8),
            queue_len=jnp.zeros(c.n_floors, jnp.int32),
            queue_up=jnp.zeros(c.n_floors, jnp.int32),
            direction=jnp.zeros(lifts, jnp.int32),
            has_goal=jnp.zeros(lifts, bool),
            goal_floor=jnp.zeros(lifts, jnp.int32),
            goal_dir=jnp.zeros(lifts, jnp.int32),
            steps=jnp.zeros((), jnp.int32),
            last_potential=jnp.zeros((), f),
            dropped=jnp.zeros((), jnp.int32),
            key=key,
        )

    def reset(self, key, warmup=None):
        """A fresh episode with WARMUP_ROUNDS of arrivals, and its first TimeStep.

        `warmup` (an `Arrivals` with a leading axis of WARMUP_ROUNDS) replaces
        the draws, to replay a Python trajectory.
        """
        state = self.empty_state(key)
        if warmup is None:

            def draw(state, _):
                key, sub = jax.random.split(state.key)
                state = state._replace(key=key)
                return self.add_arrivals(state, self.draw_arrivals(sub, 0.0)), None

            state, _ = jax.lax.scan(draw, state, None, length=WARMUP_ROUNDS)
        else:
            state, _ = jax.lax.scan(lambda s, a: (self.add_arrivals(s, a), None), state, warmup)
        # Nobody is on board yet, so the potential starts at zero.
        f = self._float()
        zero = jnp.zeros((), jnp.int32)
        info = {"delivered": zero, "boarded": zero, "waiting": state.queue_len.sum()}
        false = jnp.zeros((), bool)
        return state, TimeStep(self.observe(state), jnp.zeros((), f), false, false, info)

    # -- arrivals ------------------------------------------------------------

    def arrival_model(self, episode_fraction):
        """Expected arrivals per floor, and each floor's destination distribution."""
        c = self.config
        if c.traffic == "day":
            times = jnp.array([t for t, _ in _DAY_KEYFRAMES])
            keyframes = np.array([w for _, w in _DAY_KEYFRAMES])
            t = jnp.clip(episode_fraction, 0.0, 1.0)
            weights = jnp.stack([jnp.interp(t, times, keyframes[:, k]) for k in range(3)])
        else:
            weights = jnp.asarray(_PROFILE_WEIGHTS[c.traffic])
        mix = weights[:, None] * self._origins
        rates = c.arrival_probability * c.n_floors * mix.sum(axis=0)
        joint = (mix[:, :, None] * self._destinations).sum(axis=0)
        sums = joint.sum(axis=1, keepdims=True)
        joint = jnp.where(sums > 0, joint / jnp.where(sums > 0, sums, 1), 1 / c.n_floors)
        return rates, joint

    def draw_arrivals(self, key, episode_fraction):
        c = self.config
        n = c.n_floors
        k_count, k_dest = jax.random.split(key)
        floors = jnp.arange(n)
        if c.traffic == "uniform":
            # One Bernoulli arrival per floor, to any other floor.
            count = (jax.random.uniform(k_count, (n,)) < c.arrival_probability).astype(jnp.int32)
            dest = jax.random.randint(k_dest, (n,), 0, n - 1, dtype=jnp.int32)
            dest = dest + (dest >= floors)
            destination = jnp.zeros((n, MAX_ARRIVALS), jnp.int32).at[:, 0].set(dest)
            return Arrivals(destination, count)
        origin, dest, used, overflow = self._draw_slots(key, episode_fraction)
        # Lay the slots out per floor, in slot order, for add_arrivals. (Writing
        # each slot straight into its queue, or laying them out with a one-hot
        # product instead of a scatter, made the GPU step slower at 16,384 envs.)
        count = (jax.nn.one_hot(origin, n, dtype=jnp.int32) * used[:, None]).sum(axis=0)
        destination = (
            jnp.zeros((n, MAX_ARRIVALS), jnp.int32)
            .at[jnp.where(used, origin, n), demand.slot_rank(origin, used)]
            .set(dest, mode="drop")
        )
        return Arrivals(destination, count.astype(jnp.int32), overflow)

    def _draw_slots(self, key, episode_fraction):
        """Poisson traffic as up to `arrival_slots` arrivals, in queueing order
        (`core.demand.draw_od`): origin and destination per slot, which slots
        are used, and the count drawn beyond the slots (counted in `dropped`).
        """
        rates, joint = self.arrival_model(episode_fraction)
        return demand.draw_od(key, rates, joint, self.arrival_slots)

    def add_arrivals(self, state, arrivals):
        """Append each floor's arrivals to its queue, dropping any past the caps."""
        n = self.config.n_floors
        slot = jnp.arange(MAX_ARRIVALS)
        count = jnp.minimum(arrivals.count, MAX_ARRIVALS)
        position = state.queue_len[:, None] + slot  # (floors, MAX_ARRIVALS)
        keep = (slot < count[:, None]) & (position < MAX_QUEUE)
        rows = jnp.broadcast_to(jnp.arange(n)[:, None], position.shape)
        queue = state.queue.at[rows, jnp.where(keep, position, MAX_QUEUE)].set(
            arrivals.destination.astype(jnp.int8), mode="drop"
        )
        queue_len = jnp.minimum(state.queue_len + count, MAX_QUEUE)
        going_up = keep & (arrivals.destination > jnp.arange(n)[:, None])
        queue_up = state.queue_up + going_up.sum(axis=1, dtype=jnp.int32)
        lost = arrivals.count.sum() - (queue_len - state.queue_len).sum() + arrivals.overflow
        # sum() widens int32 to int64 under jax_enable_x64; keep the state's dtype.
        dropped = state.dropped + lost.astype(jnp.int32)
        return state._replace(queue=queue, queue_len=queue_len, queue_up=queue_up, dropped=dropped)

    # -- lifts ---------------------------------------------------------------

    @staticmethod
    def _floor(position):
        # Python's round() and jnp.round both round half to even.
        return jnp.round(position).astype(jnp.int32)

    @staticmethod
    def status(state):
        moving = (state.position != state.target) | (state.velocity != 0)
        return jnp.where(state.door_timer > 0, DOORS, jnp.where(moving, MOVING, IDLE))

    def _move(self, state, i, direction, active):
        """ElevatorSim._move for lift i (direction +1 or -1), where `active`."""
        c = self.config
        top = c.n_floors - 1
        floor = self._floor(state.position[i])
        target = state.target[i]
        if not c.kinematics:
            new = jnp.where(active, jnp.clip(floor + direction, 0, top), target)
            return state._replace(
                position=state.position.at[i].set(new.astype(state.position.dtype)),
                target=state.target.at[i].set(new),
            )
        status = self.status(state)[i]
        ahead = (target - state.position[i]) * direction
        carry_on = (status == MOVING) & (ahead > 0) & (ahead <= 1)
        new_target = jnp.where(
            status == IDLE,
            jnp.clip(floor + direction, 0, top),
            jnp.where(carry_on, jnp.clip(target + direction, 0, top), target),
        )
        return state._replace(target=state.target.at[i].set(jnp.where(active, new_target, target)))

    def _serve(self, state, i, direction, active):
        """ElevatorSim._serve for lift i, where `active`: drop off, then board.

        Boarding is in arrival order until full; direction 0 boards everyone,
        +1 / -1 only those going up / down. Only lift i's row and its floor's
        queue are rewritten, so the rest of the state is never copied.
        """
        c = self.config
        floor = self._floor(state.position[i])
        on_board = state.passengers[i]
        delivered = on_board[floor]
        on_board = on_board.at[floor].set(0)
        space = c.lift_capacity - on_board.sum()

        queue = state.queue[floor].astype(jnp.int32)
        length = state.queue_len[floor]
        waiting = jnp.arange(MAX_QUEUE) < length
        going_up = queue > floor
        if c.hall_calls:
            eligible = waiting & jnp.where(direction > 0, going_up, queue < floor)
        else:
            eligible = waiting
        # The first `space` eligible passengers, in arrival order, board.
        board = eligible & (jnp.cumsum(eligible) <= space)
        n_board = board.sum(dtype=jnp.int32)
        n_up = (board & going_up).sum(dtype=jnp.int32)
        # Count the boarders per destination with a compare-and-sum rather
        # than a scatter-add, which on a GPU means atomics.
        floors = jnp.arange(c.n_floors)[:, None]
        on_board += (board & (queue == floors)).sum(axis=1, dtype=jnp.int32)
        # Everyone else keeps their place in the queue, moved up to fill gaps.
        stay = waiting & ~board
        slot = jnp.where(stay, jnp.cumsum(stay) - 1, MAX_QUEUE)
        queue = jnp.zeros_like(queue).at[slot].set(queue, mode="drop")

        def where(new, old):
            return jnp.where(active, new, old)

        state = state._replace(
            passengers=state.passengers.at[i].set(where(on_board, state.passengers[i])),
            queue=state.queue.at[floor].set(where(queue.astype(jnp.int8), state.queue[floor])),
            queue_len=state.queue_len.at[floor].set(where(length - n_board, length)),
            queue_up=state.queue_up.at[floor].add(where(-n_up, 0)),
        )
        if c.kinematics:
            door = state.door_timer
            state = state._replace(door_timer=door.at[i].set(where(c.door_time, door[i])))
        return state, where(delivered, 0), where(n_board, 0)

    def _integrate(self, state):
        """One substep for every lift: doors, then a bang-bang profile.

        ElevatorSim._integrate with every branch computed and selected.
        """
        dt, accel = self._dt, self._acceleration
        f = state.position.dtype
        door = state.door_timer > 0
        target = state.target.astype(f)
        distance = target - state.position
        at_rest = (distance == 0) & (state.velocity == 0)
        sign = jnp.where(distance > 0, 1.0, -1.0).astype(f)
        distance = jnp.abs(distance)
        snap = distance < 1e-9
        speed = state.velocity * sign
        braking_distance = speed * speed / (2 * accel)
        safe_distance = jnp.where(snap, 1.0, distance)
        deceleration = speed * speed / (2 * safe_distance)
        new_speed = jnp.where(
            distance <= braking_distance,
            jnp.maximum(speed - _rounded(deceleration * dt), 0.0),
            jnp.minimum(speed + accel * dt, self._max_speed),
        )
        travelled = (speed + new_speed) / 2 * dt
        arrive = snap | (travelled >= distance) | (new_speed == 0)
        moving = ~door & ~at_rest
        position = jnp.where(arrive, target, state.position + _rounded(sign * travelled))
        velocity = jnp.where(arrive, 0.0, sign * new_speed)
        # Snapped to zero near the end, as sim._count_down: float32 would
        # otherwise leave the doors open for one step too many.
        timer = jnp.maximum(state.door_timer - dt, 0.0)
        timer = jnp.where(timer < DOOR_SNAP * dt, 0.0, timer).astype(f)
        return state._replace(
            door_timer=jnp.where(door, timer, state.door_timer),
            position=jnp.where(moving, position, state.position),
            velocity=jnp.where(moving, velocity, state.velocity).astype(f),
        )

    def apply_actions(self, state, actions):
        """ElevatorSim.apply_actions: per-lift (delivered, boarded, served)."""
        c = self.config
        delivered, boarded, served = [], [], []
        for i in range(c.n_lifts):  # in lift order: lower lifts board first
            action = actions[i]
            is_move = (action == UP) | (action == DOWN)
            can_serve = ~is_move
            if c.kinematics:
                can_serve &= self.status(state)[i] == IDLE
            state = self._move(state, i, jnp.where(action == UP, 1, -1), is_move)
            direction = jnp.where(action == SERVE_UP, 1, -1) if c.hall_calls else 0
            state, d, b = self._serve(state, i, direction, can_serve)
            delivered.append(d)
            boarded.append(b)
            served.append(can_serve)
        if c.kinematics:
            state = jax.lax.fori_loop(0, c.substeps, lambda _, s: self._integrate(s), state)
        return state, jnp.stack(delivered), jnp.stack(boarded), jnp.stack(served)

    def primitive_actions(self, state, actions):
        """ElevatorGymEnv._primitive_actions: each lift's goal as this step's action.

        A lift without a goal takes `actions[i]` as its new one; returns the
        state with the goals recorded, and the primitive actions.
        """
        c = self.config
        n = c.n_floors
        new = ~state.has_goal
        goal_floor = jnp.where(new, actions % n, state.goal_floor)
        down = (actions >= n) if c.hall_calls else jnp.zeros_like(new)
        goal_dir = jnp.where(new, jnp.where(down, -1, 1), state.goal_dir)
        serve = jnp.where(goal_dir > 0, SERVE_UP, SERVE_DOWN) if c.hall_calls else SERVE
        status = self.status(state)
        floor = self._floor(state.position)
        heading = jnp.where(state.target > state.position, UP, DOWN)
        # Carry on while the goal lies beyond the current target.
        beyond = (goal_floor - state.target) * jnp.where(heading == UP, 1, -1) > 0
        primitive = jnp.where(
            status == DOORS,
            serve,  # ignored until the doors close
            jnp.where(
                status == MOVING,
                jnp.where(beyond, heading, serve),
                jnp.where(floor != goal_floor, jnp.where(goal_floor > floor, UP, DOWN), serve),
            ),
        )
        state = state._replace(
            has_goal=jnp.ones_like(state.has_goal), goal_floor=goal_floor, goal_dir=goal_dir
        )
        return state, primitive

    # -- env API -------------------------------------------------------------

    def heuristic(self):
        """Collective control (`heuristic.CollectiveControl`), as a JAX policy."""
        from transit_bench.envs.elevator.heuristic import CollectiveControl

        return CollectiveControl(self)

    def potential(self, state):
        shaping = self.reward_shaping
        if shaping is None or not shaping.progress:
            return jnp.zeros((), state.position.dtype)
        floors = jnp.arange(self.config.n_floors)
        remaining = (state.passengers * jnp.abs(floors - state.position[:, None])).sum()
        return -shaping.progress * remaining / (self.config.n_floors - 1)

    def step(self, state, actions, arrivals=None):
        """One step: (state, TimeStep). No auto-reset (see `core.vec`).

        `arrivals` replaces this step's draw, to replay a Python trajectory.
        """
        c = self.config
        if arrivals is None:
            key, sub = jax.random.split(state.key)
            state = state._replace(key=key)
            arrivals = self.draw_arrivals(sub, state.steps / self.max_steps)
        state = self.add_arrivals(state, arrivals)
        before = state.position
        if self.action_mode == "target":
            state, actions = self.primitive_actions(state, actions)
        state, delivered, boarded, served = self.apply_actions(state, actions)
        if self.action_mode == "target":
            reached = served & (actions != UP) & (actions != DOWN)
            state = state._replace(has_goal=state.has_goal & ~reached)
        moved = state.position != before
        state = state._replace(
            direction=jnp.where(moved, jnp.where(state.position > before, 1, -1), state.direction)
        )

        f = state.position.dtype
        reward = delivered.sum().astype(f)
        shaping = self.reward_shaping
        if shaping is not None:
            if shaping.pickup:
                reward += shaping.pickup * boarded.sum()
            if shaping.empty_serve:
                empty = served & (delivered == 0) & (boarded == 0)
                reward -= shaping.empty_serve * empty.sum()
            if shaping.progress:
                potential = self.potential(state)
                reward += shaping.gamma * potential - state.last_potential
                state = state._replace(last_potential=potential)
            if shaping.waiting:
                reward -= shaping.waiting * state.queue_len.sum() / c.n_floors

        state = state._replace(steps=state.steps + 1)
        truncated = state.steps >= self.max_steps
        info = {
            "delivered": delivered.sum(dtype=jnp.int32),
            "boarded": boarded.sum(dtype=jnp.int32),
            "waiting": state.queue_len.sum(),
        }
        terminated = jnp.zeros((), bool)
        return state, TimeStep(self.observe(state), reward, terminated, truncated, info)

    def _waiting(self, state):
        """Waiting counts: per floor, or up then down per floor with hall calls."""
        c = self.config
        if not c.hall_calls:
            return state.queue_len
        return jnp.concatenate([state.queue_up, state.queue_len - state.queue_up])

    def _free(self, state):
        """Per lift, 1 when it will take a new target (target action mode only)."""
        return [] if self.action_mode != "target" else [~state.has_goal]

    def observe(self, state):
        if self.obs_type == "relative":
            return self._relative(state)
        return self._custom(state)

    def _custom(self, state):
        """The MultiDiscrete "custom" observation, as float32 integers."""
        c = self.config
        lifts = [self._floor(state.position), state.passengers.sum(axis=1)]
        if c.kinematics:
            lifts += [state.target, self.status(state)]
        parts = [
            jnp.stack(lifts, axis=1).ravel(),
            jnp.minimum(self._waiting(state), c.lift_capacity),
            state.passengers.ravel(),
            *(free.astype(jnp.int32) for free in self._free(state)),
        ]
        if self.observe_direction:
            parts.append(jnp.where(state.direction < 0, 2, state.direction))  # 0, up 1, down 2
        return jnp.concatenate(parts).astype(jnp.float32)

    def _relative(self, state):
        """ElevatorGymEnv._relative_observation: one window per lift, centred on it."""
        c = self.config
        n, cap, lifts = c.n_floors, c.lift_capacity, c.n_lifts
        f = state.position.dtype
        waiting = jnp.minimum(self._waiting(state).astype(jnp.float32) / cap, 1)
        up, down = (waiting[:n], waiting[n:]) if c.hall_calls else (waiting, waiting)
        wanted = jnp.minimum(state.passengers.astype(jnp.float32) / cap, 1)
        floors = self._floor(state.position)
        here = jax.nn.one_hot(floors, n, dtype=jnp.float32)
        others = (here.sum(axis=0) - here) / max(lifts - 1, 1)
        status = self.status(state)
        door = state.door_timer / c.door_time if c.kinematics else jnp.zeros(lifts, f)
        head = jnp.stack(
            [
                state.position / (n - 1),
                state.passengers.sum(axis=1) / cap,
                state.direction.astype(f),
                (status == IDLE).astype(f),
                (status == MOVING).astype(f),
                (status == DOORS).astype(f),
                (state.target - state.position) / (n - 1),
                state.velocity / self._max_speed,
                door,
                (~state.has_goal).astype(f) if self.action_mode == "target" else jnp.ones(lifts, f),
            ],
            axis=1,
        ).astype(jnp.float32)

        def window(wanted_i, others_i, floor):
            body = jnp.stack(
                [jnp.ones(n, jnp.float32), wanted_i, up, down, others_i], axis=1
            )  # (n, RELATIVE_FLOOR_FIELDS)
            grid = jnp.zeros((3 * n - 2, RELATIVE_FLOOR_FIELDS), jnp.float32)
            grid = grid.at[n - 1 : 2 * n - 1].set(body)
            # Index n - 1 of the window is the lift's own floor.
            return jax.lax.dynamic_slice_in_dim(grid, floor, 2 * n - 1).ravel()

        windows = jax.vmap(window)(wanted, others, floors)
        return jnp.concatenate([head, windows], axis=1).ravel()
