"""The hindsight bound (envs/train/bound.py): never above what a policy achieves."""

from dataclasses import replace

import jax
import numpy as np
import pytest

pytest.importorskip("highspy")

from transit_bench.envs.train import PRESETS, TrainEnv  # noqa: E402
from transit_bench.envs.train.bound import run, scenario, solve, windows  # noqa: E402
from transit_bench.envs.train.heuristic import FirstComeFirstServed  # noqa: E402


def test_windows_finds_closed_spans():
    closed = np.array([[0, 1], [1, 1], [1, 0], [0, 0], [1, 0]], bool)
    assert windows(closed) == [[(1, 3), (4, 5)], [(0, 2)]]


def test_a_calm_timetable_has_a_zero_bound():
    c = replace(PRESETS["line"], running_cv=0.0, incident_probability=0.0)
    env = TrainEnv(c)
    assert solve(env, scenario(env, jax.random.key(0))).cost == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("preset", ["single", "blockage", "junction"])
def test_the_bound_is_below_every_finished_episode(preset):
    env = TrainEnv(preset, max_steps=720)  # long enough for everyone to finish
    episode = jax.jit(lambda k, policy: run(env, policy, k), static_argnums=1)
    gaps = []
    for seed in range(6):
        key = jax.random.key(seed)
        bound = solve(env, scenario(env, key))
        assert bound.optimal
        for policy in (env.heuristic(), FirstComeFirstServed(env)):
            cost, finished = episode(key, policy)
            if finished:
                assert bound.cost <= float(cost) + 1e-6
                gaps.append(float(cost) - bound.cost)
    assert gaps and min(gaps) >= -1e-6
