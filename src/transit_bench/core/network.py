"""Transport networks: a static graph and the edge state that changes over time.

`Network` is static config (Python/NumPy, fixed when the env is built):
nodes, directed edges and nominal travel times. `EdgeState` is part of an
env's state and so can change during an episode:

- `open` (κ_e > 0): whether a vehicle may enter the edge. A closed edge is
  a blockage; vehicles wait before it.
- `speed`: the factor σ_e on the nominal travel time (1 nominal, 2 twice as
  slow).

Disruptions (phase 4) are writes to `EdgeState`, so every env that moves
vehicles over a `Network` gets them through the same arrays.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class Network(NamedTuple):
    n_nodes: int
    src: np.ndarray  # (edges,) int
    dst: np.ndarray  # (edges,) int
    travel_time: np.ndarray  # (edges,) float, nominal seconds

    @property
    def n_edges(self):
        return len(self.src)


class EdgeState(NamedTuple):
    open: jax.Array  # (edges,) bool
    speed: jax.Array  # (edges,) float, multiplies the nominal travel time


def loop(travel_times):
    """A one-way loop: edge i runs from node i to node i + 1 (mod n)."""
    travel_times = np.asarray(travel_times, dtype=np.float64)
    n = len(travel_times)
    src = np.arange(n)
    return Network(n, src, (src + 1) % n, travel_times)


def nominal(network, dtype=None):
    """Every edge open at its nominal speed."""
    dtype = dtype or jnp.zeros(()).dtype
    return EdgeState(
        open=jnp.ones(network.n_edges, bool),
        speed=jnp.ones(network.n_edges, dtype),
    )
