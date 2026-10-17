"""Fleet dispatching heuristics in JAX, reading the env state.

The Python references are `sim.nearest_idle`, `sim.matching` (with
`sim.auction`), `sim.rebalance` and `sim.stay`.
"""

import jax
import jax.numpy as jnp

from transit_bench.envs.fleet.sim import EAST, NORTH, SOUTH, STAY, TAKE, WEST, homes


class Stay:
    """Every vehicle stays: nobody is ever picked up (the floor)."""

    def __init__(self, env):
        self.env = env

    def init(self):
        return jnp.zeros((), jnp.int32)

    def act(self, carry, state, key=None):
        return carry, jnp.full(self.env.n_agents, STAY, jnp.int32)


class NearestIdle(Stay):
    """Each waiting request, longest-waiting first (lowest slot on a tie),
    takes the nearest idle vehicle not yet chosen (lowest index on a tie);
    unchosen vehicles stay. A loop over requests in that order."""

    def act(self, carry, state, key=None):
        env, c = self.env, self.env.config
        V, R = c.n_vehicles, c.max_requests
        avail = env.available(state)
        order = jnp.argsort(jnp.where(avail, -state.age, c.patience + 1), stable=True)
        free = env.deciding(state)
        actions = jnp.full(V, STAY, jnp.int32)
        vehicles = jnp.arange(V)
        for n in range(R):
            r = order[n].astype(jnp.int32)
            dist = env.distance(state.pos, state.origin[r])
            score = jnp.where(free, dist * V + vehicles, jnp.iinfo(jnp.int32).max)
            v = jnp.argmin(score)
            ok = avail[r] & free.any()
            actions = actions.at[v].set(jnp.where(ok, TAKE + r, actions[v]))
            free = free.at[v].set(free[v] & ~ok)
        return carry, actions


class Matching(Stay):
    """Per-tick bipartite matching: as `sim.matching`, an exact assignment
    of idle vehicles to waiting requests by Bertsekas' auction algorithm,
    in integers (so it matches the Python reference exactly).

    Costs are the pickup distances, with n_vehicles + 1 dummy objects
    ("stay") that cost more than any distance. Benefits are -cost x (n + 1)
    for n bidders, and each bid adds at least 1 to a price, so the result
    is optimal. Rounds run in a `lax.while_loop` until every idle vehicle
    holds an object: in each, every unassigned bidder bids for its best
    object; each object goes to its highest bidder (lowest index on a tie).
    """

    def act(self, carry, state, key=None):
        env, c = self.env, self.env.config
        V, R = c.n_vehicles, c.max_requests
        cols = R + V + 1
        big = 2 * (c.grid - 1) + 1
        bidder = env.deciding(state)
        avail = env.available(state)
        dist = env.distance(state.pos[:, None], state.origin[None, :])  # (V, R)
        cost = jnp.concatenate([dist, jnp.full((V, V + 1), big, jnp.int32)], axis=1)
        allowed = jnp.concatenate([jnp.broadcast_to(avail, (V, R)), jnp.ones((V, V + 1), bool)], 1)
        allowed = allowed & bidder[:, None]
        scale = bidder.sum(dtype=jnp.int32) + 1
        lowest = jnp.iinfo(jnp.int32).min // 2
        benefit = jnp.where(allowed, -cost * scale, lowest)

        def unassigned(s):
            assigned = s[2]
            return (bidder & (assigned < 0)).any()

        def round_(s):
            price, owner, assigned = s
            value = benefit - price[None, :]
            best_j = jnp.argmax(value, axis=1)
            best = jnp.take_along_axis(value, best_j[:, None], axis=1)[:, 0]
            masked = value.at[jnp.arange(V), best_j].set(lowest)
            second = masked.max(axis=1)
            bids = price[best_j] + best - second + 1
            bidding = bidder & (assigned < 0)
            table = jnp.where(
                bidding[:, None] & (jnp.arange(cols)[None, :] == best_j[:, None]), bids[:, None], 0
            )  # (V, cols): bids are at least 1
            top = table.max(axis=0)
            winner = jnp.argmax(table, axis=0).astype(jnp.int32)  # lowest index on a tie
            won = top > 0
            # Previous owners of objects that changed hands lose them.
            outbid = won & (owner >= 0)
            lost = jnp.zeros(V + 1, bool).at[jnp.where(outbid, owner, V)].set(True)[:V]
            assigned = jnp.where(lost, -1, assigned)
            owner = jnp.where(won, winner, owner)
            price = jnp.where(won, top, price)
            gets = jnp.zeros(V, bool).at[jnp.where(won, winner, V)].set(True, mode="drop")
            obj = (
                jnp.zeros(V, jnp.int32)
                .at[jnp.where(won, winner, V)]
                .set(jnp.arange(cols, dtype=jnp.int32), mode="drop")
            )
            assigned = jnp.where(gets, obj, assigned)
            return price, owner, assigned

        start = (
            jnp.zeros(cols, jnp.int32),
            jnp.full(cols, -1, jnp.int32),
            jnp.full(V, -1, jnp.int32),
        )
        _, _, assigned = jax.lax.while_loop(unassigned, round_, start)
        take = (assigned >= 0) & (assigned < R)
        return carry, jnp.where(take, TAKE + assigned, STAY).astype(jnp.int32)


class Rebalance:
    """`rule` (matching by default), plus: an idle vehicle the rule leaves
    where it is heads one node towards its home, the origin-weight
    quantile for its index (`sim.homes`). `sim.rebalance` in JAX."""

    def __init__(self, env, rule=None):
        self.env = env
        self.rule = rule if rule is not None else Matching(env)
        self.home = jnp.asarray(homes(env.config), jnp.int32)

    def init(self):
        return self.rule.init()

    def act(self, carry, state, key=None):
        carry, actions = self.rule.act(carry, state, key)
        env, g = self.env, self.env.config.grid
        pos = state.pos
        diff = env.toward(pos, self.home) - pos
        move = jnp.where(
            diff == 1, EAST, jnp.where(diff == -1, WEST, jnp.where(diff == g, NORTH, SOUTH))
        )
        go = (actions == STAY) & env.deciding(state) & (pos != self.home)
        return carry, jnp.where(go, move, actions).astype(jnp.int32)
