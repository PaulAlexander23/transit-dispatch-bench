"""Train dispatching heuristics in JAX, reading the env state.

The Python references are `sim.first_come_first_served`,
`sim.section_reservation` and `sim.route_setting`.
"""

import jax.numpy as jnp

from transit_bench.envs.train.sim import GO, NOT_STARTED, WAIT


class FirstComeFirstServed:
    """Every train goes as soon as it can; ties go to the earlier train."""

    def __init__(self, env):
        self.env = env

    def init(self):
        return jnp.zeros((), jnp.int32)

    def act(self, carry, state, key=None):
        return carry, jnp.full(self.env.n_trains, GO, jnp.int32)


class SectionReservation:
    """First come, first served, but never into a section against the flow.

    A deciding train about to enter a section (from outside it) waits if a
    train going the other way occupies the section, or enters it this tick
    ahead of it in the acting order. The occupancy per section and direction
    is one vectorised reduction; the "ahead of it this tick" part depends on
    earlier trains' choices, so it is a loop over trains, unrolled when traced.
    """

    def __init__(self, env):
        self.env = env

    def init(self):
        return jnp.zeros((), jnp.int32)

    def act(self, carry, state, key=None):
        env = self.env
        infra = env.infra
        S = env.n_sections
        section_of = jnp.asarray(infra.section_of, jnp.int32)
        direction = jnp.asarray(infra.direction, jnp.int32)
        route = jnp.asarray(infra.route, jnp.int32)
        # occupied[s, d]: a train going direction d is in section s.
        holder = state.occupant
        held_dir = direction[jnp.maximum(holder, 0)]
        in_section = (holder >= 0) & (section_of >= 0)
        occupied = jnp.zeros((S + 1, 2), bool)
        rows = jnp.where(in_section, section_of, S)
        occupied = occupied.at[rows, held_dir].max(in_section)[:S]
        deciding = env.deciding(state)
        nxt = env.next_block(state)
        target = section_of[jnp.maximum(nxt, 0)]
        here_block = route[jnp.arange(env.n_trains), jnp.clip(state.pos, 0, route.shape[1] - 1)]
        here = jnp.where(state.pos != NOT_STARTED, section_of[here_block], -1)
        entering = jnp.zeros((S, 2), bool)
        actions = []
        for i in range(env.n_trains):
            d = direction[i]
            s = jnp.maximum(target[i], 0)
            joining = deciding[i] & (target[i] >= 0) & (here[i] != target[i])
            against = occupied[s, 1 - d] | entering[s, 1 - d]
            wait = joining & against
            entering = entering.at[s, d].set(entering[s, d] | (joining & ~wait))
            actions.append(jnp.where(wait, WAIT, GO))
        return carry, jnp.stack(actions).astype(jnp.int32)


class RouteSetting(SectionReservation):
    """Section reservation, and a free platform at the far end of the section.

    `sim.route_setting` in JAX: a train joining a section also waits while
    the first platform after it is occupied.
    """

    def act(self, carry, state, key=None):
        carry, actions = super().act(carry, state, key)
        env = self.env
        far_table = jnp.asarray(env.infra.far_platform, jnp.int32)
        trains = jnp.arange(env.n_trains)
        index = jnp.minimum(state.pos + 1, far_table.shape[1] - 1)
        far = jnp.where(env.finished(state), -1, far_table[trains, index])
        taken = (far >= 0) & (state.occupant[jnp.maximum(far, 0)] >= 0)
        wait = (env.joining(state) >= 0) & taken
        return carry, jnp.where(wait, WAIT, actions).astype(jnp.int32)
