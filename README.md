# Transit dispatch benchmarks

Reinforcement learning benchmark environments for passenger dispatch on
transport networks, built on one shared JAX core: elevators, buses and
trains, on-demand fleets, a train line coupled with a bus route, and a
runway's landing sequence. Every env has a slow,
readable Python reference it is tested against step for step, a classical
heuristic, and a normalised score (random = 0, heuristic = 1).

| Env | Network | Decision | Heuristic |
|---|---|---|---|
| `elevator/*` | Line of floors | Move or serve, per lift | Collective control |
| `bus/*` | Loop of stops | Go or hold, per deciding bus | Headway-based holding |
| `train/*` | Blocks of track, double or single, on a line or any network | Go or wait (or terminate, or divert), per deciding train | Route setting, giving way to expresses, rerouting |
| `fleet/*` | Grid of nodes | Stay, move, or take a waiting request, per idle vehicle | Per-tick matching (exact auction) with rebalancing |
| `corridor/*` | A single-track line and a bus loop along it | Train actions (incl. terminate) and bus go/hold, in one state | Terminate when stuck, at a cost read from the buses; headway holding |
| `runway/*` | One runway, with separations that depend on both aircraft | Wait, or land one waiting aircraft | Constrained position shifting |

Disruptions (blocked or slowed edges; Markov, semi-Markov or scripted) are a
core process (as are vehicle breakdowns, on `bus/breakdown-v0` and
`elevator/outage-v0`);
`bus/closure-v0` and `train/blockage-v0` switch them on, and `train/response-v0`
adds a TERMINATE action (cancel or short-turn a train, at a cost per missed
call). `train/branch-v0` runs on a network rather than a line (a trunk
splitting into two branches; any network can be given as a `Network`).
`train/junction-v0` adds route choice: relief tracks alongside two
sections, and a DIVERT action to take them. `train/bridge-v0` adds
replacement buses: BRIDGE ends a trip and sends a bus to serve its calls.
`corridor/*` couples the train and bus envs in one state. A terminated
train's riders move to the bus route, and everyone's cost is counted in
passenger-minutes. `runway/*` sequences aircraft landing on one runway,
where the gap between two landings depends on both (wake turbulence); its
hindsight bound is an exact dynamic programme.

Which other transport problems these models cover, and which would need a
new model: [docs/applications.md](docs/applications.md).

```sh
uv sync --group jax-cuda            # or plain `uv sync` for CPU JAX
uv run pytest                       # tests, including step-for-step parity
uv run python -m transit_bench.benchmark --write docs/results.md
uv run --extra bound python scripts/bound_eval.py   # gaps to the hindsight-optimal schedule
uv run python scripts/runway_eval.py                # the same for the runway rules
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
