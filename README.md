# Transit dispatch benchmarks

Reinforcement learning benchmark environments for passenger dispatch on
transport networks, built on one shared JAX core: elevators, buses and
trains so far, with on-demand fleets planned. Every env has a slow,
readable Python reference it is tested against step for step, a classical
heuristic, and a normalised score (random = 0, heuristic = 1).

| Env | Network | Decision | Heuristic |
|---|---|---|---|
| `elevator/*` | Line of floors | Move or serve, per lift | Collective control |
| `bus/*` | Loop of stops | Go or hold, per deciding bus | Headway-based holding |
| `train/*` | Blocks of track, double or single | Go or wait, per deciding train | Station-to-station route setting |

Disruptions (blocked or slowed edges; Markov, semi-Markov or scripted) are a
core process; `train/blockage-v0` switches them on, and `train/response-v0`
adds a TERMINATE action (cancel or short-turn a train, at a cost per missed
call).

```sh
uv sync --group jax-cuda            # or plain `uv sync` for CPU JAX
uv run pytest                       # tests, including step-for-step parity
uv run python -m transit_bench.benchmark --write docs/results.md
uv run --extra bound python scripts/bound_eval.py   # gaps to the hindsight-optimal schedule
uv run python -m transit_bench.algos.ppo --env elevator/full-v0 --pretrain 100000 \
    --n-envs 1024 --n-steps 8 --batch-size 2048 --n-epochs 10 \
    --learning-rate 5.2e-5 --timesteps 10000000
```

```python
import jax
from transit_bench import make

env = make("train/single-v0")
state, timestep = env.reset(jax.random.key(0))
state, timestep = env.step(state, jax.numpy.zeros(env.n_agents, int))
```

- [docs/plan.md](docs/plan.md): the plan, with results and lessons per phase.
- [docs/formulation.md](docs/formulation.md): the maths (sequential decision
  process, the job shop with blocking, disruption processes).
- [docs/related-work.md](docs/related-work.md): where this sits among
  Flatland, DISPLIB, Jumanji, RL4CO and the bus-holding literature.
- [docs/results.md](docs/results.md): the baseline table.
