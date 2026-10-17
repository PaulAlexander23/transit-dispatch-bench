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
| `train` | Line / graph with track blocks | Order trains at conflicts, dwell, short-turn, reroute | Timetable + first-come-first-served; prioritised replanning |
| `fleet` | Grid / road graph | Assign vehicle to request, reposition | Nearest idle vehicle; per-tick matching |

Disruptions (closed edges, failed vehicles, demand surges) are a core process
that every env can switch on, not a train-only feature. The headline direction
is rail with disruptions: no JAX rail env turned up in the survey, and the
strongest rail baselines are planners, so the suite pairs a fast env with a
planning baseline and, on small presets, a hindsight-optimal bound.

The maths is in [formulation.md](formulation.md); the survey in
[related-work.md](related-work.md).

## Where this comes from

Learning from [elevator-dispatching-rl](https://github.com/PaulAlexander23/elevator-dispatching-rl)
(local: `~/Repositories/elevator-dispatching-rl`), a study in accelerating one
elevator env. Source ref: `main` at `206b740` (the merge of PR #20, which
brought in the multi-agent view, MAPPO and the loop-free arrival draw).

Port by copying and adapting, not by depending on that repo. Worth porting:

| From | What | Notes |
|---|---|---|
| `building.py`, `building_env.py` | Python reference sim + Gymnasium env | Ground truth for parity tests |
| `jax_env.py` | JAX env | Inverse-CDF Poisson arrivals into fixed slots |
| `jax_ppo.py` | JAX PPO + MAPPO (shared per-agent actor, central critic) | |
| `jax_imitate.py` | Imitation warm start from the heuristic | |
| `baselines.py` | `CollectivePolicy` and random | |
| `marl_env.py` | Multi-agent view in Mava's layout | |
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

- **Core** (`src/transit_bench/core/`): nodes, edges with travel times
  and a capacity and speed factor that can change over time (`κ_e(t)`,
  `σ_e(t)`: a disruption is a write to these), vehicles (position, capacity,
  onboard passengers, status), passenger slots (origin, destination, arrival
  time, status), a demand generator (time-varying Poisson OD matrix) and a
  disruption generator (Markov, semi-Markov or scripted outages). Pure
  functions over `NamedTuple` state.
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

- [x] Survey existing benchmarks (Flatland, Jumanji, RL4CO, SUMO-RL, OR-Gym,
      bus-holding RL papers) and record in a short `docs/related-work.md`
      what each covers, so the suite's niche is stated with evidence.
      Done: [related-work.md](related-work.md), and the maths in
      [formulation.md](formulation.md).
- [x] `pyproject.toml` with uv: Python 3.12, `jax-cuda` group
      (CUDA 12 for the GTX 1080), `dev` group (pytest, ruff, scipy). Package
      `transit_bench`. JAX and optax are core dependencies, since every env
      is JAX; the `torch` groups wait for the SB3 wrapper (backlog).
- [x] pytest with a `slow` marker, ruff, GitHub Actions CI (CPU only).
- [x] Decide the core's state layout and the env API; write them down here.

**Result.** The core is the API plus the shared machinery; each env keeps
its own state layout.

- **API** (`core/types.py`): `reset(key) -> (state, TimeStep)`,
  `step(state, action) -> (state, TimeStep)`, with
  `TimeStep(observation, reward, terminated, truncated, info)`. `info` holds
  domain metrics with the same keys at reset and every step, so it scans.
  Actions are `(n_agents,)` integers with `n_actions` choices each; the
  observation is flat (`observation_size`). Env objects hold only static
  config; all state is in a `NamedTuple`.
- **Batching** (`core/vec.py`): `make_vec_env(env, n)` with auto-reset and
  `terminal_obs`, written once for every env. Reset keys live in the
  `VecState`, not in each env's state.
- **Demand** (`core/demand.py`): the loop-free Poisson OD draw from the old
  repo (Poisson total by inverse CDF, then origin and destination per slot)
  as env-independent functions.
- **Registry** (`registry.py`): `make("elevator/full-v0")`.
- **State layout: per env, not one universal struct.** A shared
  vehicles-on-a-graph struct would have forced the elevator's per-lift
  kinematics and per-floor queues into a generic shape for no gain, and
  broken parity. The plan's `κ_e(t)` / `σ_e(t)` arrays arrive with the
  network module when the bus env needs edges (phase 2), and disruptions
  write to those (phase 4). Revisit once two graph envs exist and the
  common part is visible, rather than guessing it now.

### Phase 1: elevator port (regression check)

- [x] Port the Python reference (`building.py`) and the JAX env onto the core.
      `envs/elevator/{sim,gym_env,jax_env}.py`; the "box" observation was
      dropped (only "relative", used for learning, and "custom", read by the
      heuristic, remain).
- [x] Parity test: new JAX env against new Python reference, every preset.
      `tests/test_elevator_jax.py`: 5 presets × 5 option sets, 2 episodes
      each, exact observations in float64.
- [x] Parity test against the old repo's env (same seed, same trajectory),
      run once and recorded, so the port is known to match before diverging.
      `scripts/parity_old_repo.py`, run 2026-10-17 against 206b740 on CPU:
      all 6 presets × 3 option sets match exactly, both the Python reference
      (3 episodes, same NumPy seed) and the JAX env (same key, arrivals drawn
      on the device).
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

### Phase 3: train network

A job shop with blocking (see [formulation.md](formulation.md#train)). The
action is an ordering choice at a conflict: which train enters a contested
block first. Presets, one hard feature at a time:

- [ ] `line`: double track, stations, track blocks (capacity 1), timetable.
- [ ] `passing`: passing loops, mixed fast and slow trains (overtaking).
- [ ] `single`: single track with crossing points; deadlock becomes possible.
      Detect it in JAX by repeated squaring of the wait-for matrix.
- [ ] `junction`: a small network with route choice.
- [ ] Heuristics: keep to timetable with first-come-first-served at
      conflicts; prioritised replanning (the Flatland 2020 winners' approach),
      so RL is compared with a planner, not only a rule.
- [ ] Metrics: total and weighted delay, passengers served, deadlocks,
      conflicts resolved.
- [ ] Hindsight bound on small presets: solve the alternative-graph MILP
      after the episode with all disruptions known (e.g. HiGHS via `highspy`;
      check the dependency first). Report each policy's gap to it.
- [ ] Optional: load a few small DISPLIB instances as test networks.

### Phase 4: disruptions

- [ ] Core disruption process: edge outages (`κ_e = 0`), slow running
      (`σ_e > 1`), vehicle failures, demand surges. Markov, semi-Markov
      (lognormal durations, elapsed time observed) and scripted modes.
- [ ] Observation switch: remaining outage time visible or hidden.
- [ ] Train `blockage` preset with short-turn, cancel and reroute actions.
- [ ] Switch disruptions on for elevator (car out of service) and bus
      (road closure, breakdown).
- [ ] Evaluation: in-distribution, shift (trained without, tested with),
      and CVaR at 10% of delay alongside the mean.

### Phase 5: on-demand fleet

- [ ] Env: grid graph, requests with OD, vehicle assignment + repositioning.
- [ ] Heuristics: nearest idle vehicle; per-tick bipartite matching.

### Phase 6: benchmark harness

- [ ] One command runs random, heuristic and PPO on every env/preset over
      seeds and writes a results table (normalised scores + domain metrics),
      like the old repo's `benchmark.py --readme`.

### Phase 7: coupled modes (stretch)

- [ ] Bus bridging: when a rail block closes, buses run a replacement
      service between the affected stations. Train and bus envs share one
      core state. No paper found does short-turning and bridging jointly.

## Backlog

- GTFS loader for real bus and train networks and timetables.
- Graph-network policies (the old repo has a GNN experiment in `scripts/gnn.py`).
  Needed for rail generalisation: train on random networks (Flatland-style
  generator), test on unseen ones and on DISPLIB networks.
- Adversarial disruptions: a second agent picks which edge to block
  (attacker–defender), for a worst-case robustness score.
- RL proposes, a solver repairs: RL picks train orderings and a replanner
  makes them feasible.
- Semi-MDP / event-driven decisions, if fixed ticks prove too coarse.
- Publish to GitHub (`PaulAlexander23/transit-dispatch-bench`, public);
  confirm with the user first.

## Open questions

- Fixed tick length per env: one shared unit (seconds) or per-env?
  Leaning: one shared unit (seconds), each env sets its tick length; coupled
  modes (phase 7) need a common clock.
- Single-agent central controller, multi-agent, or both as first-class?
  (The old repo had both; MAPPO was the newer path.)
