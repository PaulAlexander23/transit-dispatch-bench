"""A train line and a bus route along it, coupled: one clock, one network,
one passenger model (Phase 7)."""

from transit_bench.envs.corridor.jax_env import CorridorEnv
from transit_bench.envs.corridor.sim import PRESETS, CorridorConfig

__all__ = ["PRESETS", "CorridorConfig", "CorridorEnv"]
