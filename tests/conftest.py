import pytest

from transit_bench.envs.elevator import gym_env


@pytest.fixture(autouse=True)
def no_render_sleep(monkeypatch):
    """Human rendering sleeps 1s per frame; skip that in tests."""
    monkeypatch.setattr(gym_env, "sleep", lambda seconds: None)
