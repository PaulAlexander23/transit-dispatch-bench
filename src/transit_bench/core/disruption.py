"""Disruptions: edges of a network going down and coming back.

Any env that moves vehicles over edges (`core.network`) can switch this on.
Each eligible edge is up or down; while down it is closed (κ_e = 0) or, with
`speed` > 1, slowed to that factor on the running time (σ_e). See
docs/formulation.md, "Disruptions as a stochastic process".

Modes:
- "markov": outages start at a constant rate and last an exponential time,
  so the process is memoryless: knowing how long an edge has been down says
  nothing about how long it will stay down.
- "semi_markov": the same onsets, but lognormal durations (coefficient of
  variation `cv`). Now the time already down predicts the time left, which
  an agent can use if it observes it.
- "scripted": fixed windows (edge, start, end) in seconds, for reproducible
  scenarios.

Both random modes draw an outage's whole duration at its onset and count it
down; for the exponential this is the same process as a per-tick repair
probability, and it lets one machinery serve both. `observe_remaining`
decides the information structure: the time left is in the observation, or
it is hidden (a partially observed problem).

There is a NumPy reference (`*_np`) for the Python sims and a JAX version;
given the same `DisruptionDraw` they agree exactly.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

MODES = ("none", "markov", "semi_markov", "scripted")


@dataclass(frozen=True)
class DisruptionConfig:
    mode: str = "none"
    rate_per_hour: float = 0.0  # outage onsets per eligible edge per hour
    mean_seconds: float = 600.0  # mean outage duration
    cv: float = 1.0  # semi_markov: coefficient of variation of the duration
    speed: float = 0.0  # 0: the edge is closed while down; > 1: slowed by this factor
    scripted: tuple = ()  # ((edge, start_seconds, end_seconds), ...)
    observe_remaining: bool = False

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.speed != 0 and self.speed < 1:
            raise ValueError("speed is 0 (closed) or a slow-down factor of at least 1")

    @property
    def random(self):
        return self.mode in ("markov", "semi_markov")


class DisruptionDraw(NamedTuple):
    """One tick's random draws for the disruption process."""

    onset: np.ndarray  # (edges,) float in [0, 1): an up edge goes down if below p
    duration: np.ndarray  # (edges,) float seconds: the outage's length if it starts


class DisruptionState(NamedTuple):
    down: jax.Array  # (edges,) bool
    elapsed: jax.Array  # (edges,) float seconds since the outage began
    remaining: jax.Array  # (edges,) float seconds left


def onset_probability(config, dt):
    """Chance an up edge goes down in one tick of `dt` seconds."""
    return 1.0 - math.exp(-config.rate_per_hour * dt / 3600.0)


def lognormal_params(mean, cv):
    sigma2 = math.log1p(cv * cv)
    return math.log(mean) - sigma2 / 2, math.sqrt(sigma2)


# -- NumPy reference -----------------------------------------------------------


def initial_np(n_edges):
    return DisruptionState(np.zeros(n_edges, bool), np.zeros(n_edges), np.zeros(n_edges))


def draw_np(config, rng, n_edges):
    if not config.random:
        return DisruptionDraw(np.ones(n_edges), np.zeros(n_edges))
    onset = rng.random(n_edges)
    if config.mode == "markov":
        duration = rng.exponential(config.mean_seconds, n_edges)
    else:
        mu, sigma = lognormal_params(config.mean_seconds, config.cv)
        duration = rng.lognormal(mu, sigma, n_edges)
    return DisruptionDraw(onset, duration)


def step_np(config, state, draw, eligible, now, dt):
    """The state at time `now` + dt, after one tick (`eligible`: (edges,) bool)."""
    down, elapsed, remaining = (np.array(x) for x in state)
    if config.mode == "scripted":
        t = now + dt
        down = np.zeros_like(down)
        elapsed = np.zeros_like(elapsed)
        remaining = np.zeros_like(remaining)
        for edge, start, end in config.scripted:
            if start <= t < end:
                down[edge] = True
                elapsed[edge] = t - start
                remaining[edge] = end - t
        return DisruptionState(down, elapsed, remaining)
    if not config.random:
        return DisruptionState(down, elapsed, remaining)
    # Outages in progress count down and may end; then up edges may fail.
    elapsed = np.where(down, elapsed + dt, 0.0)
    remaining = np.where(down, remaining - dt, 0.0)
    down = down & (remaining > 0)
    p = onset_probability(config, dt)
    start = ~down & eligible & (np.asarray(draw.onset) < p)
    down = down | start
    remaining = np.where(start, draw.duration, np.where(down, remaining, 0.0))
    elapsed = np.where(down, elapsed, 0.0)
    elapsed = np.where(start, 0.0, elapsed)
    return DisruptionState(down, elapsed, remaining)


def edge_effect_np(config, state):
    """(open, speed) per edge given the disruption state."""
    down = np.asarray(state.down)
    if config.speed == 0:
        return ~down, np.ones(len(down))
    return np.ones(len(down), bool), np.where(down, config.speed, 1.0)


# -- JAX -----------------------------------------------------------------------


def initial(n_edges, dtype=None):
    dtype = dtype or jnp.zeros(()).dtype
    zeros = jnp.zeros(n_edges, dtype)
    return DisruptionState(jnp.zeros(n_edges, bool), zeros, zeros)


def draw(config, key, n_edges, dtype=None):
    dtype = dtype or jnp.zeros(()).dtype
    if not config.random:
        return DisruptionDraw(jnp.ones(n_edges, dtype), jnp.zeros(n_edges, dtype))
    k_onset, k_duration = jax.random.split(key)
    onset = jax.random.uniform(k_onset, (n_edges,), dtype)
    if config.mode == "markov":
        duration = jax.random.exponential(k_duration, (n_edges,), dtype) * config.mean_seconds
    else:
        mu, sigma = lognormal_params(config.mean_seconds, config.cv)
        duration = jnp.exp(mu + sigma * jax.random.normal(k_duration, (n_edges,), dtype))
    return DisruptionDraw(onset, duration)


def step(config, state, draw, eligible, now, dt):
    """`step_np` in JAX: the state after one tick."""
    down, elapsed, remaining = state
    f = elapsed.dtype
    if config.mode == "scripted":
        t = now + dt
        down = jnp.zeros_like(down)
        elapsed = jnp.zeros_like(elapsed)
        remaining = jnp.zeros_like(remaining)
        for edge, start, end in config.scripted:
            active = (start <= t) & (t < end)
            down = down.at[edge].set(down[edge] | active)
            elapsed = elapsed.at[edge].set(jnp.where(active, t - start, elapsed[edge]))
            remaining = remaining.at[edge].set(jnp.where(active, end - t, remaining[edge]))
        return DisruptionState(down, elapsed.astype(f), remaining.astype(f))
    if not config.random:
        return state
    elapsed = jnp.where(down, elapsed + dt, 0.0)
    remaining = jnp.where(down, remaining - dt, 0.0)
    down = down & (remaining > 0)
    p = onset_probability(config, dt)
    start = ~down & jnp.asarray(eligible) & (jnp.asarray(draw.onset, f) < p)
    down = down | start
    duration = jnp.asarray(draw.duration, f)
    remaining = jnp.where(start, duration, jnp.where(down, remaining, 0.0))
    elapsed = jnp.where(down & ~start, elapsed, 0.0)
    return DisruptionState(down, elapsed.astype(f), remaining.astype(f))


def edge_effect(config, state):
    """(open, speed) per edge given the disruption state."""
    if config.speed == 0:
        return ~state.down, jnp.ones_like(state.elapsed)
    return jnp.ones_like(state.down), jnp.where(state.down, config.speed, 1.0).astype(
        state.elapsed.dtype
    )
