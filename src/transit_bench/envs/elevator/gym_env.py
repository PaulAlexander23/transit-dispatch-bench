"""The elevator env as a Gymnasium env over the Python reference sim.

Ported from elevator-dispatching-rl (`building_env.py` at 206b740), without
the "box" observation. It is the reference the JAX env (`jax_env`) is tested
against, and the env Stable-Baselines3 can train on directly.
"""

from dataclasses import dataclass
from time import sleep

import numpy as np
from gymnasium import Env, spaces

from transit_bench.envs.elevator.sim import (
    DOORS,
    DOWN,
    MOVING,
    PRESETS,
    SERVE,
    SERVE_DOWN,
    SERVE_UP,
    UP,
    ElevatorConfig,
    ElevatorSim,
)

OBS_TYPES = ("custom", "relative")
# Features per lift, and per floor offset, of the "relative" observation.
RELATIVE_LIFT_FIELDS = 10
RELATIVE_FLOOR_FIELDS = 5
ACTION_MODES = ("step", "target")


@dataclass(frozen=True)
class Shaping:
    """Training-only reward terms, added to the +1 per passenger delivered.

    - `pickup`: per passenger boarded.
    - `empty_serve`: penalty per lift that opens its doors and moves nobody.
    - `progress`: potential-based shaping (Ng et al., 1999) with potential
      -progress x (floors still to travel by everyone on board) / (n_floors - 1).
      It rewards carrying passengers towards their floors, and being
      potential-based it leaves the best policy unchanged.
    - `waiting`: penalty per passenger waiting per step, divided by n_floors.
      Unlike the others it changes the objective, towards short waits.
    - `gamma`: the discount used by the progress term; match the agent's.
    """

    pickup: float = 1.0
    empty_serve: float = 0.5
    progress: float = 0.0
    waiting: float = 0.0
    gamma: float = 0.99


# reward_shaping=True means "default", the original shaping.
SHAPINGS = {
    "none": Shaping(pickup=0.0, empty_serve=0.0),
    "default": Shaping(),
    "no_penalty": Shaping(empty_serve=0.0),
    "progress": Shaping(empty_serve=0.0, progress=1.0),
    "progress_waiting": Shaping(empty_serve=0.0, progress=1.0, waiting=0.1),
}


class ElevatorGymEnv(Env):
    """Multi-lift dispatching with one central controller.

    The action is one choice per lift (`MultiDiscrete`): 0 = up, 1 = down,
    2 = serve; with hall calls, 2 = serve up and 3 = serve down. The reward is
    summed over lifts. `config` (a `ElevatorConfig` or a name from `PRESETS`)
    selects the features.

    `reward_shaping` adds training-only terms: True for the original shaping
    (+1 per pickup, -0.5 per empty serve), a name from `SHAPINGS`, or a
    `Shaping`. Info always reports the unshaped counts.

    Action modes (``action_mode``):
    - "step": the primitive actions above, one floor or one serve at a time.
    - "target": each lift is given a floor to go to, and opens its doors when
      it gets there; with hall calls the action also picks the direction to
      serve (actions 0..n-1 serve up at floor a, n..2n-1 serve down at floor
      a - n). A lift takes a new target only once it has served the last one,
      so one decision covers a whole trip; actions for busy lifts are ignored.
      The observation gains a final flag per lift: 1 when it is free.

    With ``observe_direction=True`` the observation ends with each lift's last
    direction of travel (custom: 0 none yet, 1 up, 2 down), as
    a direction lantern shows. Without it a lift sweeping up and one sweeping
    down look the same, so a sweeping strategy cannot be learned.

    Observation types (``obs_type``), lift fields first, then floors:
    - "custom" (`MultiDiscrete`): per lift its floor and passenger count, plus
      its target floor and status (idle, moving, doors) with kinematics; then
      waiting counts per floor (per floor and direction with hall calls),
      capped at the lift capacity; then, per lift, passengers wanting each floor.
    - "relative" (`Box`): one block per lift, as seen from that lift: its
      position, load, last direction, status (3 flags), offset to its target,
      velocity, door timer and free flag; then for each floor offset from
      -(n-1) to n-1: inside the building, passengers on board wanting that
      floor, people waiting to go up, people waiting to go down (the same
      count twice without hall calls) and other lifts there. Counts are
      scaled by the capacity (or the number of lifts) and capped at 1. "Is
      anyone waiting ahead of me?" becomes a sum over half a block, instead
      of a comparison the network must learn for every floor.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        config=None,
        render_mode=None,
        reward_shaping=False,
        obs_type="custom",
        max_steps=200,
        action_mode="step",
        observe_direction=False,
    ):
        super().__init__()
        if isinstance(config, str):
            config = PRESETS[config]
        self.config = config if config is not None else ElevatorConfig()
        if obs_type not in OBS_TYPES:
            raise ValueError(f"obs_type must be one of {OBS_TYPES}, got {obs_type!r}")
        if action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}, got {action_mode!r}")
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.action_mode = action_mode
        self._goals = [None] * self.config.n_lifts
        self.observe_direction = observe_direction
        self._directions = [0] * self.config.n_lifts
        if reward_shaping is True:
            reward_shaping = SHAPINGS["default"]
        elif isinstance(reward_shaping, str):
            reward_shaping = SHAPINGS[reward_shaping]
        self.reward_shaping = reward_shaping or None
        self.max_steps = max_steps
        self.steps = 0
        self.sim = ElevatorSim(self.config)

        c = self.config
        if action_mode == "target":
            per_lift = c.n_floors * (2 if c.hall_calls else 1)
            self.action_space = spaces.MultiDiscrete([per_lift] * c.n_lifts)
        else:
            self.action_space = spaces.MultiDiscrete([c.n_actions] * c.n_lifts)
        n_free = c.n_lifts if action_mode == "target" else 0
        n_dir = c.n_lifts if observe_direction else 0
        # With car failures, one more entry per lift: out of service.
        self.has_failure = c.failure.mode != "none"
        n_down = c.n_lifts if self.has_failure else 0
        lift_fields = 4 if c.kinematics else 2
        n_waiting = (2 if c.hall_calls else 1) * c.n_floors
        if obs_type == "custom":
            per_lift = [c.n_floors, c.lift_capacity + 1]
            if c.kinematics:
                per_lift += [c.n_floors, 3]
            nvec = np.full(
                c.n_lifts * lift_fields
                + n_waiting
                + c.n_lifts * c.n_floors
                + n_free
                + n_dir
                + n_down,
                c.lift_capacity + 1,
            )
            nvec[: c.n_lifts * lift_fields] = np.tile(per_lift, c.n_lifts)
            if n_free:
                nvec[len(nvec) - n_dir - n_free : len(nvec) - n_dir] = 2
            if n_dir:
                nvec[len(nvec) - n_down - n_dir : len(nvec) - n_down] = 3
            if n_down:
                nvec[-n_down:] = 2
            self.observation_space = spaces.MultiDiscrete(nvec)
        else:
            head = RELATIVE_LIFT_FIELDS + (1 if self.has_failure else 0)
            size = c.n_lifts * (head + (2 * c.n_floors - 1) * RELATIVE_FLOOR_FIELDS)
            self.observation_space = spaces.Box(-1, 1, shape=(size,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.sim.rng = self.np_random

        self.steps = 0
        self.sim.reset()
        self._goals = [None] * self.config.n_lifts
        self._directions = [0] * self.config.n_lifts
        for _ in range(10):
            self.sim.sample_passengers(0.0)
        self._last_potential = self._potential()

        return self._observation(), {}

    def _primitive_actions(self, action):
        """Turn each lift's target (kept until served) into this step's action."""
        c = self.config
        n = c.n_floors
        primitive = []
        for i, lift in enumerate(self.sim.state().lifts):
            if self._goals[i] is None:
                a = int(action[i])
                self._goals[i] = (a % n, -1 if c.hall_calls and a >= n else 1)
            floor, direction = self._goals[i]
            serve = (SERVE_UP if direction > 0 else SERVE_DOWN) if c.hall_calls else SERVE
            status = lift.status
            if status == DOORS:
                primitive.append(serve)  # ignored until the doors close
            elif status == MOVING:
                # Ask to carry on while the goal lies beyond the current target.
                heading = UP if lift.target > lift.position else DOWN
                beyond = (floor - lift.target) * (1 if heading == UP else -1) > 0
                primitive.append(heading if beyond else serve)
            elif lift.floor != floor:
                primitive.append(UP if floor > lift.floor else DOWN)
            else:
                primitive.append(serve)
        return primitive

    def _potential(self):
        """Minus the floors still to travel by everyone on board, scaled."""
        shaping = self.reward_shaping
        if shaping is None or not shaping.progress:
            return 0.0
        # A plain loop, not sum(): from Python 3.12, sum() of floats is
        # compensated, so it rounds differently from the C++ port and 3.11.
        remaining = 0.0
        for lift in self.sim.state().lifts:
            for destination in lift.passengers:
                remaining += abs(destination - lift.position)
        return -shaping.progress * remaining / (self.config.n_floors - 1)

    def step(self, action):
        self.sim.sample_passengers(self.steps / self.max_steps)
        before = [lift.position for lift in self.sim.state().lifts]
        if self.action_mode == "target":
            primitive = self._primitive_actions(action)
            delivered, boarded, served = self.sim.apply_actions(primitive)
            for i, (a, s) in enumerate(zip(primitive, served, strict=True)):
                if s and a not in (UP, DOWN):
                    self._goals[i] = None
        else:
            delivered, boarded, served = self.sim.apply_actions(action)
        for i, (lift, position) in enumerate(zip(self.sim.state().lifts, before, strict=True)):
            if lift.position != position:
                self._directions[i] = 1 if lift.position > position else -1
        if self.has_failure:
            now = self.steps * self.config.step_seconds
            self.sim.advance_failure(self.sim.draw_failure(), now)

        reward = float(sum(delivered))
        shaping = self.reward_shaping
        if shaping is not None:
            if shaping.pickup:
                reward += shaping.pickup * sum(boarded)
            if shaping.empty_serve:
                reward -= shaping.empty_serve * sum(
                    s and d == 0 and b == 0
                    for s, d, b in zip(served, delivered, boarded, strict=True)
                )
            if shaping.progress:
                potential = self._potential()
                reward += shaping.gamma * potential - self._last_potential
                self._last_potential = potential
            if shaping.waiting:
                waiting = sum(len(queue) for queue in self.sim.state().floor_queues)
                reward -= shaping.waiting * waiting / self.config.n_floors

        self.steps += 1
        truncated = self.steps >= self.max_steps
        info = {
            "delivered": sum(delivered),
            "boarded": sum(boarded),
            "waiting": sum(len(queue) for queue in self.sim.state().floor_queues),
        }
        return self._observation(), reward, False, truncated, info

    def _observation(self):
        c = self.config
        state = self.sim.state()
        n = c.n_floors

        if c.hall_calls:
            up = [sum(d > f for d in queue) for f, queue in enumerate(state.floor_queues)]
            down = [len(queue) - u for queue, u in zip(state.floor_queues, up, strict=True)]
            waiting = up + down
        else:
            waiting = [len(queue) for queue in state.floor_queues]
        wanted = []
        for lift in state.lifts:
            counts = [0] * n
            for destination in lift.passengers:
                counts[destination] += 1
            wanted += counts

        if self.obs_type == "relative":
            return self._relative_observation(waiting, wanted)

        obs = []
        for lift in state.lifts:
            obs += [lift.floor, len(lift.passengers)]
            if c.kinematics:
                obs += [lift.target, lift.status]
        obs += [min(w, c.lift_capacity) for w in waiting] + wanted + self._free()
        if self.observe_direction:
            obs += [{0: 0, 1: 1, -1: 2}[d] for d in self._directions]
        if self.has_failure:
            obs += [int(self.sim.out_of_service(i)) for i in range(c.n_lifts)]
        return np.array(obs, dtype=np.int64)

    def _relative_observation(self, waiting, wanted):
        c = self.config
        n, cap = c.n_floors, c.lift_capacity
        lifts = self.sim.state().lifts
        waiting = np.minimum(np.asarray(waiting, dtype=np.float32) / cap, 1)
        up, down = (waiting[:n], waiting[n:]) if c.hall_calls else (waiting, waiting)
        wanted = np.minimum(np.asarray(wanted, dtype=np.float32).reshape(c.n_lifts, n) / cap, 1)
        floors = np.array([lift.floor for lift in lifts])
        here = np.zeros((c.n_lifts, n), dtype=np.float32)
        here[np.arange(c.n_lifts), floors] = 1
        others = (here.sum(axis=0) - here) / max(c.n_lifts - 1, 1)
        free = self._free() or [1] * c.n_lifts
        speed = c.max_speed / c.floor_height
        blocks = []
        for i, lift in enumerate(lifts):
            status = lift.status
            head = [
                lift.position / (n - 1),
                len(lift.passengers) / cap,
                self._directions[i],
                status == 0,
                status == 1,
                status == 2,
                (lift.target - lift.position) / (n - 1),
                lift.velocity / speed,
                lift.door_timer / c.door_time if c.kinematics else 0.0,
                free[i],
            ]
            if self.has_failure:
                head.append(float(self.sim.out_of_service(i)))
            # Floor arrays padded by n - 1 on each side, then a window centred
            # on this lift: index n - 1 of the window is the lift's own floor.
            grid = np.zeros((3 * n - 2, RELATIVE_FLOOR_FIELDS), dtype=np.float32)
            body = grid[n - 1 : 2 * n - 1]
            body[:, 0] = 1
            body[:, 1] = wanted[i]
            body[:, 2] = up
            body[:, 3] = down
            body[:, 4] = others[i]
            window = grid[floors[i] : floors[i] + 2 * n - 1]
            blocks.append(np.concatenate([np.array(head, dtype=np.float32), window.ravel()]))
        return np.concatenate(blocks)

    def _free(self):
        """Per lift, 1 when it will take a new target (target action mode only)."""
        if self.action_mode != "target":
            return []
        return [int(goal is None) for goal in self._goals]

    def render(self):
        if self.render_mode == "human":
            print(self.steps)
            for i, lift in enumerate(self.sim.state().lifts):
                if lift.status == DOORS:
                    print(f"lift {i}: doors open at floor {lift.floor}")
            self.sim.render()
            sleep(1 / self.metadata["render_fps"])
        return None
