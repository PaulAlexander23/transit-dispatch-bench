"""Corridor heuristics in JAX: a train rule and a bus rule side by side.

The Python references are `sim.no_termination`, `sim.bus_blind` and
`sim.bus_aware`. The bus rule is always headway holding; the train rules
differ in whether, and on what cost, a blocked train terminates.
"""

import jax.numpy as jnp

from transit_bench.envs.bus.heuristic import HeadwayHolding
from transit_bench.envs.train.heuristic import RouteSetting, TerminateWhenStuck


class Joint:
    """`train_rule` on the trains and headway holding on the buses. With
    `bus_cost`, the train rule's terminate cost is `env.transfer_cost`."""

    def __init__(self, env, train_rule, bus_cost=False):
        self.env, self.train_rule, self.bus_cost = env, train_rule, bus_cost
        self.bus_rule = HeadwayHolding(env.bus_env)

    def init(self):
        return self.train_rule.init(), self.bus_rule.init()

    def act(self, carry, state, key=None):
        train_carry, bus_carry = carry
        if self.bus_cost:
            cost = self.env.transfer_cost(state)
            train_carry, a = self.train_rule.act(train_carry, state.train, key, cost=cost)
        else:
            train_carry, a = self.train_rule.act(train_carry, state.train, key)
        bus_carry, b = self.bus_rule.act(bus_carry, state.bus, key)
        return (train_carry, bus_carry), jnp.concatenate([a, b]).astype(jnp.int32)


def NoTermination(env):
    """Route setting and headway holding: no trip ever ends early."""
    return Joint(env, RouteSetting(env.train_env))


def BusBlind(env, factor=2.0):
    """Terminate when stuck at the fixed penalty per missed call."""
    return Joint(env, TerminateWhenStuck(env.train_env, factor=factor))


def BusAware(env, factor=2.0):
    """Terminate when stuck at the cost read from the buses."""
    return Joint(env, TerminateWhenStuck(env.train_env, factor=factor), bus_cost=True)
