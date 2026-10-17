"""The env API shared by every env in the suite.

An env is an object holding static settings (its config); its methods are
pure functions of a state `NamedTuple` and a PRNG key, so they can be
`jit`-compiled, `vmap`-ped over thousands of envs and fused with a learner
into one XLA program:

    state, timestep = env.reset(key)
    state, timestep = env.step(state, action)

`TimeStep.info` holds the env's domain metrics for the step (delivered,
boarded, ...). Its keys and shapes are the same at reset and at every step,
so it can be carried through `lax.scan`.
"""

from typing import Any, NamedTuple, Protocol

import jax


class TimeStep(NamedTuple):
    observation: jax.Array
    reward: jax.Array  # () float
    terminated: jax.Array  # () bool: the episode reached a terminal state
    truncated: jax.Array  # () bool: the episode hit its time limit
    info: dict[str, jax.Array]

    @property
    def done(self):
        return self.terminated | self.truncated


class Env(Protocol):
    """What the vectoriser, the learners and the benchmark harness rely on."""

    n_agents: int  # vehicles acting each step (actions have shape (n_agents,))
    n_actions: int  # choices per agent
    observation_size: int  # flat observation length
    max_steps: int

    def reset(self, key: jax.Array) -> tuple[Any, TimeStep]: ...

    def step(self, state: Any, action: jax.Array) -> tuple[Any, TimeStep]: ...
