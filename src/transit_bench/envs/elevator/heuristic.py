"""The elevator heuristic: collective control.

`CollectivePolicy` is the Python reference (ported from elevator-dispatching-rl's
`baselines.py`), reading the "custom" observation through the Stable-Baselines3
`predict` interface. `CollectiveControl` is the same policy in JAX, reading
the env state, so it runs inside `jit` over thousands of envs (for the
benchmark harness and for collecting imitation data).
"""

import jax.numpy as jnp

from transit_bench.envs.elevator.sim import DOWN, IDLE, MOVING, SERVE, SERVE_DOWN, SERVE_UP, UP


class CollectivePolicy:
    """Collective control for `ElevatorGymEnv` with obs_type="custom".

    The classic elevator algorithm: each lift sweeps in one direction,
    stopping where someone on board wants to get off or someone is waiting to
    go its way, and turns round when there is nothing more ahead. With no work
    at all it heads for the lobby. Lifts do not coordinate, so several may
    chase the same call; it is a reference point, not a strong dispatcher.
    """

    def __init__(self, config):
        from transit_bench.envs.elevator.sim import PRESETS, ElevatorConfig

        if isinstance(config, str):
            config = PRESETS[config]
        self.config = config if config is not None else ElevatorConfig()
        self.directions = [1] * self.config.n_lifts

    def reset(self):
        self.directions = [1] * self.config.n_lifts

    def predict(self, observations, state=None, episode_start=None, deterministic=False):
        if episode_start is not None and episode_start[0]:
            self.reset()
        return [self._act(obs) for obs in observations], state

    def _parse(self, obs):
        c = self.config
        n, fields = c.n_floors, 4 if c.kinematics else 2
        lifts = [obs[i * fields : (i + 1) * fields] for i in range(c.n_lifts)]
        k = c.n_lifts * fields
        if c.hall_calls:
            up, down = obs[k : k + n], obs[k + n : k + 2 * n]
            k += 2 * n
        else:
            up = down = obs[k : k + n]
            k += n
        wanted = [obs[k + i * n : k + (i + 1) * n] for i in range(c.n_lifts)]
        return lifts, up, down, wanted

    def _act(self, obs):
        lifts, up, down, wanted = self._parse(obs)
        return [
            self._act_lift(i, lift, up, down, want)
            for i, (lift, want) in enumerate(zip(lifts, wanted, strict=True))
        ]

    def _act_lift(self, i, lift, up, down, want):
        from transit_bench.envs.elevator.sim import (
            DOWN,
            IDLE,
            MOVING,
            SERVE,
            SERVE_DOWN,
            SERVE_UP,
            UP,
        )

        c = self.config
        n = c.n_floors
        floor, load = int(lift[0]), int(lift[1])
        room = load < c.lift_capacity

        def stop_at(f, direction):
            calls = up[f] if direction > 0 or not c.hall_calls else down[f]
            return want[f] > 0 or (room and calls > 0)

        def work_beyond(f, direction):
            floors = range(f + 1, n) if direction > 0 else range(f - 1, -1, -1)
            return any(want[g] > 0 or up[g] > 0 or down[g] > 0 for g in floors)

        def serve(direction):
            if not c.hall_calls:
                return SERVE
            return SERVE_UP if direction > 0 else SERVE_DOWN

        def go(direction):
            return UP if direction > 0 else DOWN

        d = self.directions[i]
        if c.kinematics and int(lift[3]) != IDLE:
            # Carry on past the target only if there is no reason to stop there.
            target = int(lift[2])
            if int(lift[3]) == MOVING and not stop_at(target, d) and work_beyond(target, d):
                return go(d)
            return serve(d)  # ignored while busy
        if stop_at(floor, d):
            return serve(d)
        if work_beyond(floor, d):
            return go(d)
        if stop_at(floor, -d) or work_beyond(floor, -d):
            self.directions[i] = d = -d
            return serve(d) if stop_at(floor, d) else go(d)
        self.directions[i] = -1
        return DOWN  # park at the lobby


class CollectiveControl:
    """`CollectivePolicy` in JAX, for the "step" action mode.

    A `core.types.Policy`: `init()` gives the carry (each lift's sweep
    direction) and `act(carry, state, key)` the next carry and one action per
    lift. Lifts decide independently from the same state, so the per-lift
    loop of the Python version becomes array operations over lifts and floors.
    """

    def __init__(self, env):
        if env.action_mode != "step":
            raise ValueError("collective control acts in the 'step' action mode")
        self.env = env

    def init(self):
        return jnp.ones(self.env.config.n_lifts, jnp.int32)

    def act(self, directions, state, key=None):
        env = self.env
        c = env.config
        n = c.n_floors
        floor = env._floor(state.position)  # (lifts,)
        want = state.passengers > 0  # (lifts, floors)
        if c.hall_calls:
            up = state.queue_up > 0
            down = (state.queue_len - state.queue_up) > 0
        else:
            up = down = state.queue_len > 0
        room = state.passengers.sum(axis=1) < c.lift_capacity  # (lifts,)
        work = want | up | down  # (lifts, floors)
        floors = jnp.arange(n)
        lifts = jnp.arange(c.n_lifts)

        def stop_at(f, d):
            calls = jnp.where((d > 0) | (not c.hall_calls), up[f], down[f])
            return want[lifts, f] | (room & calls)

        def work_beyond(f, d):
            ahead = jnp.where(d[:, None] > 0, floors > f[:, None], floors < f[:, None])
            return (work & ahead).any(axis=1)

        def serve(d):
            if not c.hall_calls:
                return jnp.full_like(d, SERVE)
            return jnp.where(d > 0, SERVE_UP, SERVE_DOWN)

        def go(d):
            return jnp.where(d > 0, UP, DOWN)

        d = directions
        turn = stop_at(floor, -d) | work_beyond(floor, -d)
        turned = -d
        idle_action = jnp.where(
            stop_at(floor, d),
            serve(d),
            jnp.where(
                work_beyond(floor, d),
                go(d),
                jnp.where(
                    turn,
                    jnp.where(stop_at(floor, turned), serve(turned), go(turned)),
                    DOWN,  # park at the lobby
                ),
            ),
        )
        new_d = jnp.where(stop_at(floor, d) | work_beyond(floor, d), d, jnp.where(turn, turned, -1))
        if not c.kinematics:
            return new_d, idle_action
        # A busy lift carries on past its target only if there is no reason
        # to stop there; otherwise it asks to serve, which is ignored while busy.
        status = env.status(state)
        target = state.target
        carry_on = (status == MOVING) & ~stop_at(target, d) & work_beyond(target, d)
        busy_action = jnp.where(carry_on, go(d), serve(d))
        busy = status != IDLE
        return jnp.where(busy, d, new_d), jnp.where(busy, busy_action, idle_action)
