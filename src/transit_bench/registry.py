"""Versioned env ids: `make("elevator/full-v0")`.

An id is `<env>/<preset>-v<version>`. A change to an env's dynamics,
observation or reward bumps its version, so published scores stay
comparable.
"""

from collections.abc import Callable

_REGISTRY: dict[str, Callable] = {}


def register(env_id: str, factory: Callable):
    if env_id in _REGISTRY:
        raise ValueError(f"{env_id!r} is already registered")
    _REGISTRY[env_id] = factory


def make(env_id: str, **kwargs):
    """The env for `env_id`; keyword arguments go to its constructor."""
    if env_id not in _REGISTRY:
        raise KeyError(f"unknown env {env_id!r}; registered: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[env_id](**kwargs)


def registered():
    return sorted(_REGISTRY)


def _register_builtin():
    from functools import partial

    from transit_bench.envs.elevator import PRESETS, ElevatorEnv

    for preset in PRESETS:
        register(f"elevator/{preset}-v0", partial(ElevatorEnv, preset))


_register_builtin()
