"""The benchmark harness."""

from transit_bench import benchmark


def test_harness_scores_random_zero_and_heuristic_one(tmp_path):
    envs = ["elevator/original-v0", "bus/loop-v0", "train/single-v0"]
    rows = benchmark.run(envs, n_episodes=4)
    by_policy = {(env, name): score for env, name, _, _, score, _ in rows}
    for env in envs:
        assert by_policy[(env, "random")] == 0.0
        assert by_policy[(env, "heuristic")] == 1.0
    table = benchmark.markdown(rows, 4)
    assert table.count("\n| `") == len(rows)
    path = tmp_path / "results.md"
    path.write_text(f"intro\n{benchmark.START}\nold\n{benchmark.END}\nouttro\n")
    benchmark.write(path, table)
    text = path.read_text()
    assert text.startswith("intro") and text.endswith("outtro\n") and "old" not in text


def test_harness_adds_a_trained_row():
    rows = benchmark.run(["fleet/grid-v0"], n_episodes=2, ppo_timesteps=8192)
    names = [name for _, name, *_ in rows]
    assert "mappo 8,192" in names
