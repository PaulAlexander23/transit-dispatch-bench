"""The bus heuristic: headway-based holding, in JAX.

The Python reference is `sim.headway_holding`. A deciding bus holds until
the target headway H has passed since the bus ahead left this stop, for at
most `max_hold_seconds`: the forward-headway rule of the bus-holding
literature, which damps bunching by evening out departures (see
docs/formulation.md, "Bus").
"""

import jax.numpy as jnp

from transit_bench.envs.bus.sim import GO, HOLD


class HeadwayHolding:
    """A `core.policy.Policy` reading the env state."""

    def __init__(self, env):
        self.env = env

    def init(self):
        return jnp.zeros((), jnp.int32)

    def act(self, carry, state, key=None):
        c = self.env.config
        now = state.t * c.step_seconds
        early = now - state.last_depart[state.stop] < self.env.headway
        room = state.hold * c.step_seconds < c.max_hold_seconds
        hold = self.env.deciding(state) & early & room
        return carry, jnp.where(hold, HOLD, GO).astype(jnp.int32)
