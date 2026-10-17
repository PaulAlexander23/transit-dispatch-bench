# Plan: transit dispatch benchmarks

Started 2026-10-17. This is the live plan: tick items off, and add a short
**Result** under each phase as it finishes (what was built, numbers, what was
learned, and anything deferred to the backlog).

## Goal

A suite of RL benchmark environments for passenger dispatch on transport
networks: elevators, buses, trains and on-demand fleets. All of them are one
problem class, the **dynamic stochastic pickup-and-delivery problem**:
capacity-limited vehicles move on a network, passengers arrive at random with
an origin and a destination, and a controller decides in real time. In OR
terms this is the dynamic vehicle routing / dial-a-ride problem; elevator
group control is its special case on a line graph. With several vehicles it is
a cooperative multi-agent problem (Dec-POMDP).

What the suite offers that existing benchmarks (Flatland, Jumanji, RL4CO,
SUMO-RL, OR-Gym; check their current state in phase 0) don't: one shared,
fast JAX core across transport modes, with a classical heuristic and
normalised scores for every environment.

| Env | Network | Decisions | Heuristic baseline |
|---|---|---|---|
| `elevator` | Line (floors) | Move, serve, hall-call direction | Collective control |
| `bus` | Loop / route graph | Hold, skip stop, short-turn | Headway-based holding |
| `train` | Line / graph with track blocks | Dwell, reorder, reroute | Timetable + first-come-first-served |
| `fleet` | Grid / road graph | Assign vehicle to request, reposition | Nearest idle vehicle |

## Where this comes from

Learning from [elevator-dispatching-rl](https://github.com/PaulAlexander23/elevator-dispatching-rl)
(local: `~/Repositories/elevator-dispatching-rl`), a study in accelerating one
elevator env. Source refs: `main` at `66f747b`, and the `next-plan` branch at
`c4325a0` (draft PR #20), which alone has the multi-agent view and MAPPO.

Port by copying and adapting, not by depending on that repo. Worth porting:

| From | What | Notes |
|---|---|---|
| `building.py`, `building_env.py` | Python reference sim + Gymnasium env | Ground truth for parity tests |
| `jax_env.py` | JAX env | Inverse-CDF Poisson arrivals into fixed slots (`next-plan` version) |
| `jax_ppo.py` | JAX PPO + MAPPO (shared per-agent actor, central critic) | `next-plan` version |
| `jax_imitate.py` | Imitation warm start from the heuristic | |
| `baselines.py` | `CollectivePolicy` and random | |
| `marl_env.py` | Multi-agent view in Mava's layout | `next-plan` only |
| `conf/`, `tune*.py` | Hydra + Optuna tuning | Later, if needed |

Leave behind: the C++, nanobind, ctypes, EnvPool and Warp ports. They were the
point of the old repo, but one JAX backend per env is what scales to a suite.

Reference numbers to reproduce on the `full` preset (30 floors, 4 lifts,
kinematics, hall calls, day traffic): random ≈ 7, `CollectivePolicy` 69,
tuned PPO after warm start 77.0 ± 2.4 delivered per episode (50M steps,
~9.5 min on the GTX 1080). JAX PPO reached ~363k steps/s (on the Warp env;
the JAX env alone steps ~0.68M env steps/s).

## Lessons carried over

- **PPO from scratch fails on long horizons.** On 30 floors a delivery needs
  dozens of correct steps; exploration rarely finds one and the empty-serve
  penalty teaches "never open the doors". Imitate the heuristic first, then
  fine-tune with a low learning rate. Expect the same on bus/train envs.
- **Observation centred on the agent** (the `relative` obs) was what made
  imitation and fine-tuning work.
- **Parity tests catch real bugs.** Every fast implementation is checked step
  for step against a slow, readable Python reference with the same seed.
- **Add features one preset at a time** (`original → tall → multi →
  kinematic → hall_calls → full`), so a regression points at one feature.
- **JAX needs fixed shapes.** Pad to a max number of vehicles and passenger
  slots; use fixed time ticks rather than event-driven steps; avoid
  data-dependent loops (the Poisson draw became an inverse CDF for this).
- **The env must keep up with the learner.** End-to-end `jit` of rollout and
  update, with no host round-trips, is where the big speed-up came from.

## Design

- **Core** (`src/transit_bench/core/`): nodes, edges with travel times,
  vehicles (position, capacity, onboard passengers), passenger slots (origin,
  destination, arrival time, status), and a demand generator (time-varying
  Poisson OD matrix). Pure functions over `NamedTuple` state.
- **Envs** (`src/transit_bench/envs/<name>/`): each is a core config plus its
  own action set, constraints, observation and reward. Each has a Python
  reference sim, a JAX env, a heuristic and presets.
- **API:** functional JAX (`reset(key) -> state, timestep`;
  `step(state, action) -> state, timestep`), in the Gymnax/Jumanji style.
  Wrappers: Gymnasium (for SB3) and a multi-agent view (Mava layout).
- **Registry:** `make("bus/loop-v0")`. Versioned ids; a change in dynamics
  bumps the version so published scores stay comparable.
- **Metrics:** domain metrics alongside reward: mean wait, mean journey time,
  passengers served, headway variance (buses), delay (trains). Reward
  shapings are named and switchable.
- **Scores:** normalised per env, random = 0, heuristic = 1.
- **Algorithms** (`src/transit_bench/algos/`): JAX PPO, MAPPO, imitation.
  Kept small: the point is the envs, the algorithms are reference baselines.

## Phases

### Phase 0: skeleton

- [ ] Survey existing benchmarks (Flatland, Jumanji, RL4CO, SUMO-RL, OR-Gym,
      bus-holding RL papers) and record in a short `docs/related-work.md`
      what each covers, so the suite's niche is stated with evidence.
- [ ] `pyproject.toml` with uv: Python 3.12, `jax` / `jax-cuda` groups
      (CUDA 12 for the GTX 1080), optional `torch` cpu/cuda groups for the
      SB3 wrapper, `dev` group (pytest, ruff). Package `transit_bench`.
- [ ] pytest with a `slow` marker, ruff, GitHub Actions CI (CPU only).
- [ ] Decide the core's state layout and the env API; write them down here.

### Phase 1: elevator port (regression check)

- [ ] Port the Python reference (`building.py`) and the JAX env onto the core.
- [ ] Parity test: new JAX env against new Python reference, every preset.
- [ ] Parity test against the old repo's env (same seed, same trajectory),
      run once and recorded, so the port is known to match before diverging.
- [ ] Port `CollectivePolicy`, JAX PPO + MAPPO and imitation.
- [ ] Reproduce the reference numbers on `full` (heuristic 69; PPO ≥ 70
      after warm start is enough to call the port good; a full 50M-step
      run only with the user's go-ahead).

### Phase 2: bus loop

- [ ] Env: buses on a loop of stops, stochastic travel times, passenger
      arrivals per stop; action = holding time at a stop (discrete).
- [ ] Heuristic: headway-based holding (hold to target headway).
- [ ] Metrics: mean wait, headway variance, bunching rate.
- [ ] Presets: few stops/buses → realistic route; time-varying demand.
- [ ] Python reference + parity tests; PPO/MAPPO results vs heuristic.

### Phase 3: train line

- [ ] Env: one line, stations, track blocks (one train per block), dwell
      decisions; disruptions (delays) injected at random.
- [ ] Heuristic: keep to timetable, first-come-first-served at conflicts.
- [ ] Metrics: total delay, passengers served, conflicts resolved.

### Phase 4: on-demand fleet

- [ ] Env: grid graph, requests with OD, vehicle assignment + repositioning.
- [ ] Heuristic: nearest idle vehicle.

### Phase 5: benchmark harness

- [ ] One command runs random, heuristic and PPO on every env/preset over
      seeds and writes a results table (normalised scores + domain metrics),
      like the old repo's `benchmark.py --readme`.

## Backlog

- GTFS loader for real bus and train networks and timetables.
- Graph-network policies (the old repo has a GNN experiment in `scripts/gnn.py`).
- Semi-MDP / event-driven decisions, if fixed ticks prove too coarse.
- Publish to GitHub (`PaulAlexander23/transit-dispatch-bench`, public);
  confirm with the user first.

## Open questions

- Fixed tick length per env: one shared unit (seconds) or per-env?
- Single-agent central controller, multi-agent, or both as first-class?
  (The old repo had both; MAPPO was the newer path.)
