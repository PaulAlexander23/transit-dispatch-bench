"""Train dispatching heuristics in JAX, reading the env state.

The Python references are `sim.first_come_first_served`,
`sim.section_reservation`, `sim.route_setting` and `sim.terminate_when_stuck`.
"""

import jax.numpy as jnp

from transit_bench.envs.train.sim import GO, NOT_STARTED, TERMINATE, WAIT


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

    `sim.route_setting` in JAX: a train joining a section that carries both
    directions also waits while the first platform after it is occupied.
    """

    def act(self, carry, state, key=None):
        carry, actions = super().act(carry, state, key)
        env = self.env
        far_table = jnp.asarray(env.infra.far_platform, jnp.int32)
        trains = jnp.arange(env.n_trains)
        index = jnp.minimum(state.pos + 1, far_table.shape[1] - 1)
        far = jnp.where(env.finished(state), -1, far_table[trains, index])
        taken = (far >= 0) & (state.occupant[jnp.maximum(far, 0)] >= 0)
        joining = env.joining(state)
        shared = jnp.asarray(env.infra.section_shared)[jnp.maximum(joining, 0)]
        wait = (joining >= 0) & shared & taken
        return carry, jnp.where(wait, WAIT, actions).astype(jnp.int32)


class TerminateWhenStuck:
    """`rule` (route setting by default), plus TERMINATE where waiting looks
    dearer than the missed calls, and to break deadlocks.

    `sim.terminate_when_stuck` in JAX: a train that can terminate does so if
    its next block is closed and the outage's expected time left (observed,
    or the mean outage length) times `factor` exceeds the penalty for its
    missed calls; and
    of the deadlocked trains that can terminate, the one missing fewest
    calls (lowest index on a tie) terminates.
    """

    def __init__(self, env, rule=None, factor=2.0):
        self.env, self.factor = env, factor
        self.rule = rule if rule is not None else RouteSetting(env)

    def init(self):
        return self.rule.init()

    def act(self, carry, state, key=None):
        carry, actions = self.rule.act(carry, state, key)
        env, c = self.env, self.env.config
        T = env.n_trains
        can = env.can_terminate(state)
        b = jnp.maximum(env.next_block(state), 0)
        closed = ~state.blocks.open[b]
        if c.disruption.observe_remaining:
            left = state.disruption.remaining[b]
        else:
            left = jnp.full(T, c.disruption.mean_seconds)
        missed = env.missed_calls(state)
        stuck = can & closed & (self.factor * left > 60 * c.terminate_penalty * missed)
        actions = jnp.where(stuck, TERMINATE, actions)
        dead = env.deadlocked(state) & can
        score = jnp.where(dead, missed * T + jnp.arange(T), jnp.iinfo(jnp.int32).max)
        pick = jnp.argmin(score)
        actions = actions.at[pick].set(jnp.where(dead.any(), TERMINATE, actions[pick]))
        return carry, actions.astype(jnp.int32)


class GiveWayToExpress:
    """`rule` (route setting by default), plus: a stopping train on a loop
    platform waits for an express behind it when the express would lose
    more by following it through the next section (and, if there is no
    loop at its end, waiting while it stands at the platform) than it
    loses by waiting. `sim.give_way_to_express` in JAX, over all (train, express)
    pairs at once."""

    def __init__(self, env, rule=None):
        self.env = env
        self.rule = rule if rule is not None else RouteSetting(env)

    def init(self):
        return self.rule.init()

    def act(self, carry, state, key=None):
        carry, actions = self.rule.act(carry, state, key)
        env, infra = self.env, self.env.infra
        k, T = env.config.blocks_per_section, env.n_trains
        cum = jnp.asarray(infra.cum_ticks, jnp.int32)
        width = cum.shape[1]
        L = width - 1
        express = jnp.asarray(infra.express)
        direction = jnp.asarray(infra.direction, jnp.int32)
        route_len = jnp.asarray(infra.route_len, jnp.int32)
        loop = jnp.asarray(infra.loop_platform)[jnp.maximum(env.here(state), 0)]
        p = state.pos
        rows = jnp.arange(T)
        candidate = (
            (actions == GO)
            & env.deciding(state)
            & ~express
            & (p >= 0)
            & loop
            & (p + 1 + k < route_len)
        )
        q = jnp.clip(p + 1, 0, width - 1)
        nxt = jnp.clip(p + 1 + k, 0, width - 1)
        a_i = cum[rows, nxt] - cum[rows, q]  # (T,)
        route = jnp.asarray(infra.route, jnp.int32)
        next_loop = jnp.asarray(infra.loop_platform)[route[rows, jnp.minimum(nxt, L - 1)]]
        a_i = a_i + jnp.where(next_loop, 0, cum[rows, jnp.minimum(nxt + 1, L)] - cum[rows, nxt] + 1)
        # Pairs [i, j]: j an express behind i on the same line.
        behind = (p[None, :] <= p[:, None]) & ~env.finished(state)[None, :]
        pair = express[None, :] & (direction[None, :] == direction[:, None]) & behind
        dt = env.config.step_seconds
        depart = jnp.ceil(jnp.asarray(infra.sched[:, 0]) / dt - 1e-9).astype(jnp.int32)
        base = jnp.where(p == NOT_STARTED, jnp.maximum(0, depart - state.t), state.remaining)
        start = cum[rows, jnp.clip(p + 1, 0, width - 1)]  # (T,) j's next index
        cum_q = cum[rows[None, :], q[:, None]]  # [i, j]: cum[j, q_i]
        eta = base[None, :] + cum_q - start[None, :]
        t_j = cum[rows[None, :], nxt[:, None]] - cum_q
        loss = jnp.maximum(0, a_i[:, None] - (eta + t_j))
        first = cum[rows[None, :], jnp.clip(q + 1, 0, width - 1)[:, None]] - cum_q
        wait = eta + first + 1
        give_way = candidate & (pair & (loss > wait)).any(axis=1)
        return carry, jnp.where(give_way, WAIT, actions).astype(jnp.int32)
