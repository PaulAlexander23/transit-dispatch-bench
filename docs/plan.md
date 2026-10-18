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

How the suite grows: take an application, distil its hard core into a
mathematical model, and build an env only if that core is new. If it
matches an existing model, record it as an instance of the closest env
instead. [applications.md](applications.md) has the method and the
applications assessed so far (waterways, locks, mining, trams, metro,
shared lift shafts, airports, airlines, air traffic, shunting, robots,
electric buses), each with the complexity it would add.

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
- [x] Port `CollectivePolicy`, JAX PPO + MAPPO and imitation.
      Collective control is also written in JAX on the env state
      (`CollectiveControl`), tested action for action against the Python
      one, so imitation data is collected in one jitted rollout (16 s for
      100k samples, against about a minute on the Python env before).
- [x] Reproduce the reference numbers on `full` (heuristic 69; PPO ≥ 70
      after warm start is enough to call the port good; a full 50M-step
      run only with the user's go-ahead).

**Result.** Ported, with exact parity, and a bug found in the old envs.

- **A float32 bug in the old JAX/Warp envs.** On `full` the JAX heuristic
  delivered 62.3 ± 0.1 per episode against the Python reference's
  67.0 ± 0.4. In float64 JAX gave 67.0, so I compared float32 trajectories
  against Python step by step. They split at the first door cycle: a 4 s
  door timer counted down in forty 0.1 s substeps keeps 1.6e-6 s in
  float32 (0.1 is not exact in binary), so doors stayed open one extra step
  and every stop took 5 steps instead of 4. Fixed by snapping the timer to
  zero within a thousandth of a substep (`sim._count_down`). That changes
  nothing in float64, so the old-repo parity still holds there. Now float32
  follows Python step for step under the heuristic (regression test
  `test_float32_env_follows_python_under_the_heuristic`). Lesson: the
  old repo's "float32 drifts but stays statistically the same" was never
  tested statistically. A systematic bias hid behind it, and it made the
  old PPO numbers (77 at 50M steps, trained and evaluated in float32) look
  better against the heuristic (69, measured in float64) than they were.
- **Reference numbers, fixed env** (`scripts/reproduce_elevator.py`,
  GTX 1080). Random 6.2 ± 0.1, collective control 67.0 ± 0.1 (1,024
  episodes each). PPO with the old tuned settings (1,024 envs × 8 steps,
  minibatch 2,048, 10 epochs, lr 5.2e-5, 256 × 256) after imitating 100k
  heuristic samples: **80.6 ± 1.4 after 10M steps** (seeds 80.2, 82.5,
  79.1; 2 minutes each at 83k steps/s), so 20% above the heuristic. With
  the doors fixed the env is easier and PPO gets further in a fifth of the
  steps. The old repo's 69 for the heuristic is not reproduced: the Python
  reference gives 67.0 ± 0.4 over seeds 0–99 and JAX agrees, so 69 was
  probably a small-sample estimate.
- **Throughput.** JAX PPO at 83k steps/s on `full` with 1,024 envs; the
  old repo's 363k was on the Warp env, which was left behind.

### Phase 2: bus loop

- [x] Env: buses on a loop of stops, stochastic travel times, passenger
      arrivals per stop; action = holding time at a stop (discrete).
      Holding is a per-tick go/hold choice for a *deciding* bus (nobody
      left to drop off or pick up); other buses' actions are ignored.
- [x] Heuristic: headway-based holding (hold to target headway).
- [x] Metrics: mean wait, headway variance, bunching rate.
- [x] Presets: few stops/buses → realistic route; time-varying demand.
      `loop → noisy → busy → peak → full`.
- [x] Python reference + parity tests; PPO/MAPPO results vs heuristic.

**Result.** Env built and verified; RL does not beat the heuristic yet.

- **Model** (`envs/bus/`, maths in [formulation.md](formulation.md#bus)).
  10 s ticks; boarding 2 s and alighting 1 s per passenger, which gives the
  bunching feedback. Running times are lognormal, rounded up to whole ticks
  so no float accumulates. Reward is minus passenger-minutes (waiting plus
  on board) over the number of stops. The target headway is the
  steady-state fixed point `H = Sτ / (B − Sλ(α+β))`. Random draws are an
  explicit `Exogenous` per step (Powell's `W_{t+1}`), so the JAX env replays
  Python's draws: exact parity on every preset, under random and heuristic
  actions. Edges carry `core.network.EdgeState` (open, speed); a closed
  edge already holds buses (tested), which phase 4 builds on.
- **Bugs caught by parity.** In JAX, `int32 / int32` gives float32 even
  with x64 on, and `sum` widens int32 to int64 under x64. Both now cast
  explicitly.
- **Baselines** (`scripts/bus_baselines.py`, 256 episodes, return / mean
  wait / headway CV / bunching rate):

  | Preset | Random | Always go | Holding | MAPPO 5M | MAPPO + BC 5M |
  |---|---|---|---|---|---|
  | `noisy` | −668 / 239 s / 0.79 / 0.13 | −605 / 219 s / 0.78 / 0.15 | **−594** / 166 s / 0.46 / 0 | −605 (= always go) | −597 / 172 s / 0.50 / 0 |
  | `full` | −1066 / 257 s / 0.69 / 0.08 | −986 / 227 s / 0.71 / 0.11 | −986 / 180 s / 0.44 / 0 | −1806 (holds forever) | **−982** / 180 s / 0.44 / 0 |

- **What was learned.**
  - Holding evens headways (CV 0.78 → 0.46, no bunching) and cuts waits
    by 25%, but saves only about 2% of total passenger time. The time it
    saves at stops is spent on board. On `full` it ties with always-go.
  - **MAPPO from scratch converges to never holding.** A hold costs
    in-vehicle minutes immediately, while the benefit (less bunching) is
    delayed and spread over many passengers. P(hold | deciding) went to
    0.000. On `full` the greedy policy even held forever.
  - Masking inactive agents out of the PPO loss (`active_agents`, added
    to the PPO port for envs with sparse decisions) did not change this.
    It is right in principle and is now in place for rail.
  - **Imitation then fine-tuning** reproduces the heuristic (log-likelihood
    ≈ 0) and keeps it, but does not improve on it in 5M steps (~15 s at
    370k steps/s on `noisy`).
- **Backlog from this phase:** a two-way headway heuristic (balance the
  gaps ahead and behind); a named `headway` shaping (penalise uneven
  departures, as much of the bus-holding literature does); a hold
  *duration* action at the decision point (semi-MDP), so one decision
  covers a whole hold; longer runs.

### Phase 3: train network

A job shop with blocking (see [formulation.md](formulation.md#train)). The
action is an ordering choice at a conflict: which train enters a contested
block first. Presets, one hard feature at a time:

- [x] `line`: double track, stations, track blocks (capacity 1), timetable.
- [x] `passing`: passing loops, mixed fast and slow trains (overtaking).
- [x] `single`: single track with crossing points; deadlock becomes possible.
      Detect it in JAX by repeated squaring of the wait-for matrix.
- [x] `junction`: a small network with route choice. Done as relief
      tracks: a second, slower track alongside a section, and a DIVERT
      action at the station before it (Result below).
- [x] Heuristics: keep to timetable with first-come-first-served at
      conflicts; prioritised replanning (the Flatland 2020 winners' approach),
      so RL is compared with a planner, not only a rule.
      Done: first come, first served, and *section reservation* (never enter
      a section against an oncoming train), the interlocking principle
      single-track lines run on. Prioritised replanning is still to do
      (backlog).
- [x] Metrics: total and weighted delay, passengers served, deadlocks,
      conflicts resolved. Done: delay per train, deadlock rate, trains
      finished, held and blocked counts. Passengers are not modelled yet.
- [x] Hindsight bound on small presets: solve the alternative-graph MILP
      after the episode with all disruptions known (e.g. HiGHS via `highspy`;
      check the dependency first). Report each policy's gap to it.
      Done: `envs/train/bound.py`, `scripts/bound_eval.py`; result below.
- [ ] Optional: load a few small DISPLIB instances as test networks.

**Result.** Env built and verified; RL matches the rules where they are
optimal and copies section reservation, but does not yet improve on it.

- **Model** (`envs/train/`). `infra.build` lays out stations and
  sections of blocks (double or single track, optional loop platforms),
  routes and a timetable with 5% slack. Each tick a *ready* train with a
  free next block is *deciding*: GO enters, WAIT stays. Blocks hold one
  train, and a train holds its block until it enters the next (blocking).
  Running times are lognormal; incidents (2% of block entries) add
  exponential delays (mean 2 min), the primary delays. Reward is minus
  the change in total projected lateness, so the return telescopes to
  minus total arrival delay (tested exactly). Deadlock detection in JAX:
  ⌈log₂ N⌉ boolean matrix squarings of the wait-for relation (tested on a
  constructed head-on cycle with a third train queued behind).
- **Design fix found by testing the heuristic: synchronous ticks.** At
  first a train with one tick left became ready during the tick and acted
  on an action chosen before it was deciding. So two trains could enter
  a single-track section from both ends, and section reservation behaved
  exactly like first come, first served (20% deadlocks). Now ready and
  deciding are judged once at the start of the tick, a block freed during
  a tick opens on the next one (like a signal clearing), and countdowns
  happen at the end. `active_agents` is then exactly the set of trains
  whose action matters.
- **Baselines** (`scripts/train_baselines.py`, 512 episodes; delay per
  train in minutes, deadlock rate):

  | Preset | Random | FCFS | Reservation | MAPPO 5M | Clone | Clone + MAPPO 5M |
  |---|---|---|---|---|---|---|
  | `line` | 2.97 / 0 | 0.86 / 0 | 0.86 / 0 | 0.85 / 0 | | |
  | `passing` | 4.27 / 0 | 2.16 / 0 | 2.16 / 0 | 2.15 / 0 | 2.15 / 0 | 2.15 / 0 |
  | `single` | 9.57 / 0.39 | 4.92 / 0.22 | **1.48 / 0.025** | 4.68 / 0.20 | 1.63 / 0.029 | 1.87 / 0.041 |

  MAPPO trains at 175–235k steps/s (5M steps in under 30 s).
- **What was learned.**
  - **From scratch, MAPPO finds first come, first served and never
    discovers deadlock avoidance.** A deadlock is rare and catastrophic,
    and its cost arrives long after the decision that caused it, which
    16-step rollouts with GAE cannot credit. This is the Flatland lesson
    in miniature.
  - **Cloning the rule failed at first, for two reasons, each found by
    measuring accuracy on the rare WAIT decisions (0.09% of agent-actions)
    rather than the average log-likelihood (−0.008, which looked
    perfect).** (1) The observation lacked the conflict: an oncoming train
    waiting at the far end of a section stands on a platform that is not on
    this train's route. Added fields: next section's oncoming occupant, and
    oncoming trains about to enter it before or after this one. (2) "An
    oncoming train is in my next section" was also true for trains about
    to move onto a platform, not into the section, so a field for "my next
    block enters that section" was needed. After both, WAIT accuracy went
    from 60% to 100% and the clone matches the rule. Masking inactive
    agents out of the imitation loss (as in PPO) did not help by itself.
  - Fine-tuning the clone with PPO for 5M steps drifts slightly worse
    (1.63 → 1.87 min, deadlocks 2.9% → 4.1%): the same credit problem.
  - On `passing`, no rule or learner beats FCFS yet. Holding a stopping
    train in a loop for a following express is the decision that could
    help, and it needs look-ahead along the route behind the train.
- **Common random numbers.** Running-time noise and incidents were drawn
  per tick, so a train's running time depended on when it entered a
  block, and "the same scenario under another policy" was not defined.
  They are now drawn per train and route position at reset (a `Trip` in
  the state). The distribution is unchanged (each train enters each block
  once), but a seed now fixes the scenario for every policy, which also
  cuts the variance of policy comparisons. Numbers after this change
  differ from the tables above by sampling noise only.
- **Hindsight bound** (`envs/train/bound.py`). After an episode every
  draw is known: the trip, and the closures (the disruption process does
  not depend on the trains, so a rollout of all-WAIT recovers them). The
  best schedule for that scenario is a MILP over integer entry ticks: route
  order with the env's rounded block times, departures, one big-M
  disjunction per pair of trains sharing a block (with the env's +1 tick
  for a freed block), one per train and closure window, and total
  lateness as the objective. Any env trajectory in which all trains
  finish is feasible for it with the same cost, so its optimum is a lower
  bound (tested on `single` and `blockage`, and zero on a calm timetable).
  About 150 binaries on `single`; HiGHS solves it in 0.2–1.5 s on average.
  *Dependency check:* `highspy` 1.15.1 is MIT (licence expression in the
  wheel metadata), depends only on NumPy, ships cp312 manylinux wheels
  (about 7 MB); added as the optional extra `bound` and to the dev group,
  not as a core dependency.
- **Gaps to the bound** (`scripts/bound_eval.py`, 200 scenarios per
  preset, 720-tick horizon so trains finish; minutes per train, over the
  episodes where the policy finished every train):

  | Preset | Bound | FCFS | Section reservation | Route setting | Route setting above bound |
  |---|---|---|---|---|---|
  | `line` | 0.88 | 0.88 | 0.88 | 0.88 | 1% |
  | `passing` | 1.56 | 2.16 | 2.16 | 2.16 | **28%** |
  | `single` | 1.04 | 0.62\* | 0.99\* | 1.11 | 8% |
  | `blockage` | 2.50 | 0.90\* | 1.89\* | 2.66 | 12% |

  \* Over the episodes it finished (FCFS finishes 77% on `single`, 59% on
  `blockage`; route setting 99.5% and 98%), so these are the easy ones;
  compare each with the bound on the same episodes: FCFS is at the bound
  (gap 0.00) whenever it does not deadlock, section reservation within
  0.05 min. All solves were optimal except 2% on `blockage`, which hit the
  30 s limit and report the dual bound (still valid).
- **What the bound says.**
  - `line` and `single` leave almost nothing to learn: the rules are
    within 1–8% of a dispatcher who knows the future. On single track the
    whole problem is avoiding deadlocks; once a rule does that, little
    remains. This explains the RL results above better than the credit
    problem alone: there was no headroom to find.
  - **`passing` is where the headroom is**: 28% of the delay (0.6 min per
    train, at least 0.14 in every scenario) is avoidable, and no rule
    captures any of it. It comes from overtaking: holding a stopping train
    in a loop so an express can pass. That is the next target for a
    better rule, a look-ahead planner and RL.
  - The bound knows the future, so part of any gap could be the value of
    information, which no real-time policy can capture. Checked on
    `passing` (100 scenarios): without incidents the gap is 39%, and fully
    deterministic it is **70%** (route setting 1.01 vs 0.30 min per
    train). So the gap is policy, not information.
- **Rule: give way to the express** (`GiveWayToExpress`, now part of the
  env heuristic where there are expresses). Found by reading the bound's
  schedule on a deterministic scenario: the optimum holds a stopping train
  at the *last loop before the terminus* for 8 ticks, because the
  terminus has no loop and the express behind it would otherwise wait out
  its dwell there (15 ticks). The rule: a stopping train on a loop
  platform waits for an express behind it (or beside it on the main line)
  when, in nominal ticks, the express's loss from following it (through
  the section, plus the platform time if there is no loop at the end)
  exceeds its own wait. Two bugs on the way, both found by the rule
  never firing: pricing only the section (the gain is at the terminus),
  and requiring the express to be strictly behind (it is usually on the
  main platform of the same station when the stopping train decides).
  **On deterministic `passing` it equals the bound exactly** (tested).
  With noise and incidents (`scripts/bound_eval.py`, 200 scenarios):

  | Policy on `passing` | Delay/train | Above bound |
  |---|---|---|
  | Route setting (= FCFS here) | 2.16 | 28% |
  | **Give way to express** | **1.86** | **16%** |
  | Hindsight bound | 1.56 | |

  The remaining 16% is noise and incidents: the nominal-tick rule cannot
  see a late express or a long dwell coming, and some of it is the value
  of information. This is the headroom a learner or a stochastic
  look-ahead could go after on `passing`.
- **Route choice: relief tracks and DIVERT** (`TrainConfig.relief_sections`,
  `train/junction-v0`). A full graph with arbitrary alternative routes
  would change every per-train table (routes, timetable, calls, the
  bound's variables) into a per-route one. A *relief track* keeps them
  all: a second track alongside a section, with the same number of
  blocks, `relief_slowdown` (1.25) times slower, used in both directions.
  A train's route positions line up whichever track it takes, so the
  timetable, its calls, the trip's noise (per position) and the platforms
  are unchanged; only the block and its running time at those positions
  differ (`Infra.alt_route`, `alt_run`). The state gains a (trains, L)
  flag, on the relief track at each position.
  - **DIVERT** (action 3; the action space is 4 wherever there is a relief
    track) is legal *at a junction*, i.e. on the platform before such a
    section: the train enters the relief track's first block and follows
    it to the end of the section. A train at a junction is deciding if
    either first block is free and open, and blocked if the one it chose
    is not. Observation: two more head fields (at a junction; the relief
    block is free and open), so `HEAD_FIELDS` is now 17 on every preset.
  - **Deadlock detection becomes approximate there.** With two ways out,
    the wait-for relation is an AND-OR graph: a train at a junction is
    stuck only if both next blocks' holders are. Squaring the wait-for
    matrix handles single successors only, so a train whose relief block
    is free counts as waiting for no one, and one whose relief block is
    taken counts as waiting for the main block's holder only. So a policy
    that ignores the relief track can sit in a "deadlock" the metric does
    not report (section reservation on `junction`: 0 deadlocks, but delay
    as on `blockage`).
  - **`junction` preset**: `blockage` plus relief tracks alongside the two
    middle sections. The relief blocks are numbered after all the others
    and closures are drawn per block in one vector, so the main blocks'
    closures are the same as `blockage`'s for a seed: route setting, which
    never diverts, scores the same on both (2.96 vs 3.01 min per train).
  - **Rule: `Reroute`** (the `junction` heuristic): route setting, plus at
    a junction, if the main track's next block is taken or any of the
    section's main blocks is closed, divert when the relief track is free,
    open throughout, has no oncoming train in it (or diverting into it
    ahead in the acting order), and the platform at its far end is free
    (route setting for the relief track, which carries both directions).
  - **Hindsight bound with route choice.** One more binary per train and
    relief section; times in those blocks are `d + (d_relief − d) z`, and
    each blocking and closure constraint on a block is relaxed by M for a
    train that does not use it. Still below every finished episode
    (tested on `junction`), 99% solved to optimality, 0.8 s on average.
  - **Results** (harness, 512 episodes; `scripts/bound_eval.py`, 200
    scenarios; minutes per train):

    | Policy on `junction` | Delay | CVaR10 | Above bound |
    |---|---|---|---|
    | Route setting (never diverts) | 3.01 | 15.4 | 34% |
    | **Reroute** | **1.75** | **7.05** | **10%** |
    | Hindsight bound | | | (1.87 vs reroute's 2.09 on its scenarios) |

    Diverting 1.5 times an episode cuts the mean by 42% and halves the
    tail. Most of that is going around closures: on the calm
    single-track line with the same relief tracks (1,024 episodes) reroute
    gains only 1.16 → 0.98 min per train (0.9 diversions an episode, to
    cross or pass a train), against 2.93 → 1.75 with closures.
  - **What was learned.** Reroute is a greedy rule, yet it is within 10%
    of the hindsight optimum, closer than any rule on the other
    disruption presets. The remaining gap is the trade it never makes:
    waiting for a main block that will clear soon instead of taking the
    25% slower track, and diverting a train early to leave the main track
    to a later one.
### Phase 4: disruptions

- [x] Core disruption process: edge outages (`κ_e = 0`), slow running
      (`σ_e > 1`), vehicle failures, demand surges. Markov, semi-Markov
      (lognormal durations, elapsed time observed) and scripted modes.
      Done for edge outages and slow running (`core/disruption.py`), and
      vehicle failures by running the same process over vehicles (bus
      breakdowns, lift outages). Demand surges are still to do.
- [x] Observation switch: remaining outage time visible or hidden.
- [x] Train `blockage` preset with short-turn, cancel and reroute actions.
      Done: `blockage` (random closures on the single-track line) and
      `response` (longer closures plus TERMINATE, which cancels a trip
      before departure or short-turns it at a platform), and `junction`
      (reroute onto a relief track with DIVERT, Phase 3). The return half
      of a short-turn needs vehicle circulation (backlog).
- [x] Switch disruptions on for elevator (car out of service) and bus
      (road closure, breakdown). Done: `bus/closure-v0`,
      `bus/breakdown-v0` and `elevator/outage-v0` (Result below).
- [x] Evaluation: in-distribution, shift (trained without, tested with),
      and CVaR at 10% of delay alongside the mean.

**Result.** Disruptions work end to end on the train env; the best answer to
them so far is a better rule, not RL.

- **Process** (`core/disruption.py`). Each eligible edge is up or down;
  both random modes draw the outage's full duration at onset and count it
  down. For exponential durations this is the same process as a per-tick
  repair chance (memoryless), so one mechanism serves Markov and
  semi-Markov. Tested: outage rate, mean and CV of durations, and the
  stationary fraction down `λm/(1+λm)`. NumPy and JAX agree exactly given
  the same draws. The train env advances it at the *end* of each tick (so
  the policy sees the state it will act under), over section blocks only.
  It observes per look-ahead block how long the block has been down, and
  how long it has left only when `observe_remaining` is set.
- **`train/blockage-v0`**: `single` plus Markov closures (0.1 per section
  block per hour, mean 10 min: about 0.8 outages an episode).
- **A third rule, route setting, found by reading the deadlocks.** The
  deadlocks section reservation still allowed are four-train cycles
  through two sections and the two platforms of one station (described in
  `sim.route_setting`). Entering a section only when the platform at its
  far end is free, i.e. station-to-station route setting as real
  interlockings do, removes them on this line. It is now the train env's
  heuristic (`TrainEnv.heuristic()`).
- **Results** (`scripts/disruption_eval.py`, 1,024 episodes; delay per
  train in minutes, mean of the worst 10% of episodes, deadlock rate):

  | Policy | `single` | `blockage` |
  |---|---|---|
  | First come, first served | 5.24 / 30.5 / 0.23 | 9.01 / 34.6 / 0.39 |
  | Section reservation | 1.42 / 7.25 / 0.021 | 4.05 / 25.0 / 0.099 |
  | **Route setting** | **1.15 / 4.36 / 0** | **3.05 / 15.7 / 0.003** |
  | Clone of route setting, trained on `single` | 1.15 / 4.36 / 0 | 3.05 / 15.7 / 0.003 |
  | That clone + MAPPO 5M on `blockage` | 1.59 / 8.89 / 0.030 | 4.17 / 25.5 / 0.105 |
  | MAPPO 5M from scratch under the shield | 1.15 / 4.36 / 0 | 3.05 / 15.7 / 0.003 |

- **What was learned.**
  - Blockages roughly triple delay and quadruple the tail (CVaR 4.4 →
    15.7 min under route setting), and blocked trains bunch, which makes
    deadlock cycles more likely. The tail metric moves far more than the
    mean, which is why it belongs in the benchmark.
  - Shift: a clone that copies a rule exactly transfers exactly, because
    the rule never looked at the disruption fields. A learned policy that
    used them could not be tested this way; that needs a policy that
    beats the rules first.
  - **PPO fine-tuning destroys the deadlock avoidance it starts with**
    (deadlocks 0 → 3% on `single`, 0.3% → 10.5% on `blockage`, after 5M
    steps at lr 5e-5). The penalty for a deadlock arrives long after the
    decision that caused it, so the policy gradient does not protect the
    rare WAIT decisions that prevent it, and noise erodes them. Candidates:
    a deadlock penalty (named shaping), longer rollouts and higher γ, a
    KL penalty to the clone, or action masking that forbids unsafe GO
    (a safety shield, as in the 2026 shielding paper in related work).
  - **Tried the shield** (`TrainEnv(shield=True)`: any GO route setting
    forbids becomes WAIT, and a vetoed train is not an active agent).
    Under it, always-GO *is* route setting (tested), so the shield makes
    the rule the floor. MAPPO from scratch under the shield converges to
    exactly that floor in 5M steps (identical numbers to route setting):
    safe, but it found no extra holds that pay. Improving on the rule
    needs either decisions the rule does not make (the response actions
    below) or a signal for the holds that do pay (e.g. holding a train at
    a crossing loop for a late oncoming one).
  - With only go/wait, a dispatcher has few levers once a block is closed.
    The disruption *response* actions (short-turn, cancel, reroute) are
    where a learner could add value over the rules.
- **Response action: TERMINATE** (`TrainConfig.terminate_penalty` > 0,
  `train/response-v0`). A ready train that has not departed, or stands on
  a platform with blocks still ahead, may end its trip: it leaves the
  network at once, freeing its platform, and its lateness is frozen at
  its projected lateness plus 5 minutes per missed call (the mean
  passenger wait at a 10-minute headway). The return still telescopes to
  minus total cost (tested). Before departure it is a cancellation; at an
  intermediate station it is the outbound half of a short-turn. Such a
  train is an active agent even when its next block is taken or closed,
  since that is when terminating is worth considering. Observation: two
  more head fields (can terminate; calls it would miss). JAX matches
  Python step for step under random three-action play and the new rule.
- **Rule: `TerminateWhenStuck`** (the `response` heuristic): route
  setting, plus terminate a train whose next block is closed when
  `factor` × the expected outage time left exceeds its penalty, and break
  any deadlock by terminating the deadlocked train that misses fewest
  calls.
- **Results** (`scripts/response_eval.py`, 2,048 episodes, seed 7; cost
  per train in minutes = delay + penalties, mean ± s.e.; CVaR10;
  deadlock rate; trains terminated per episode):

  | Policy | Cost | CVaR10 | Deadlocks | Terminated |
  |---|---|---|---|---|
  | Route setting (never terminates) | 5.23 ± 0.18 | 26.4 | 0.010 | 0 |
  | Terminate when stuck, factor 1 | 3.80 ± 0.12 | 17.1 | 0.004 | 0.59 |
  | **Terminate when stuck, factor 2** | **3.70 ± 0.11** | **15.8** | **0.001** | 0.72 |
  | MAPPO 5M from scratch, shielded | 5.23 ± 0.18 | 26.4 | 0.010 | 0 |
  | Clone of the rule, shielded | 3.72 ± 0.11 | 16.0 | 0.001 | 0.72 |
  | Clone + MAPPO 5M, shielded | 3.72 ± 0.11 | 16.0 | 0.001 | 0.72 |

  A sweep over outage length (10–30 min), rate (0.1–0.2 per block-hour),
  penalty (2–10 min per call) and observability: the rule beats route
  setting everywhere, by 13–50% on the mean and up to half the tail.
- **What was learned.**
  - **Terminating is a tail-risk tool**: it cuts CVaR10 by 40% for a 30%
    mean saving, and deadlocks fall tenfold because a terminated train
    frees a platform that a cycle would have needed.
  - **The myopic threshold undervalues waiting.** With factor 1 (wait
    cost = the blocked train's own wait), knowing the true time left was
    *no better* than assuming the mean, and sometimes worse: told that an
    outage will be short, the rule waits, but the trains queued behind
    wait too. Factor 2 is better and factors 1.5–5 are identical (with
    the mean estimate, every blocked train with up to 5 missed calls then
    terminates). Seeing the true time left gains 0.07 min with factor 2,
    within noise. The queue behind the train is what a better rule (or a
    learner) should price.
  - **MAPPO from scratch never discovers TERMINATE.** It costs 5–25
    minutes at once, and the delay it saves arrives over the following
    tens of minutes, spread across other trains: the credit problem
    again, now with an immediate penalty in front of a delayed benefit.
    Starting from a clone keeps the rule's behaviour (the shield keeps
    GO/WAIT safe, so fine-tuning no longer erodes it), but 5M steps found
    no improvement.
- **Bus road closures** (`BusConfig.disruption`, `bus/closure-v0`). The
  bus env now runs the core process on its edges, as the train env does
  on blocks: it advances at the end of each tick, a closed edge holds
  buses at the stop before it, and a slowed one (speed > 1) stretches the
  running time drawn on departure. The disruption draws use a key folded
  from the step's key, so the arrival and noise draws of the other
  presets are unchanged (their rows in the results table are identical).
  Observation: per stop, the edge out of it is down, time since, and
  time left if observable (`STOP_FIELDS` 3 → 6 on every bus preset).
  JAX matches Python step for step on all presets. `closure` is `peak`
  plus Markov closures, 0.1 per edge-hour, 5 minutes on average (about
  one an episode). Harness, 512 episodes:

  | Policy | `peak` return / wait / headway CV | `closure` |
  |---|---|---|
  | Always go | −1582 / 347 s / 0.83 | −1617 / 368 s / 0.87 |
  | Headway holding | −1540 / 288 s / 0.46 | −1595 / 323 s / 0.52 |

  A closure costs headway holding 55 return and 35 s of mean wait, and
  its lead over always-go halves (42 → 22). A queue that forms behind a
  closure leaves bunched; the rule re-spaces it at the next stops, but
  only 120 s per stop (`max_hold_seconds`), and nothing in it
  anticipates the closure (e.g. holding buses upstream while the road is
  shut, so they do not all join the queue). That is headroom for a
  closure-aware rule or a learner.
- **Bus breakdowns** (`BusConfig.breakdown`, `bus/breakdown-v0`). Vehicle
  failures reuse the core process with buses in place of edges: a bus
  goes down at random and is repaired after a random time; while down it
  freezes (no running, boarding, alighting or departing) and is not
  deciding. Observation: three more head fields per bus (broken down,
  for how long, time to repair if observable), so `HEAD_FIELDS` 7 → 10.
  `breakdown` is `peak` with 0.3 failures per bus-hour, 5 minutes on
  average (about one an episode). Holding −1619 vs always-go −1633
  (512 episodes): the rule's lead falls from 42 on `peak` to 13. Variants
  that re-space for the buses still running (target headway × B / (B −
  broken)), stop holding during a breakdown, or allow longer holds are
  all within noise of it (1,024 episodes, ±4.5): with 3 buses a failure
  removes a third of the capacity, and no holding rule recovers that.
  The levers that would are out of this env's action space (a spare bus,
  short-turning the bus behind).
- **Lift cars out of service** (`ElevatorConfig.failure`,
  `elevator/outage-v0`). The same vehicle-failure use of the process: an
  out-of-service car ignores its actions, finishes any move or door cycle
  under way, and stands with its passengers until it is back. The
  observation gains one entry per lift (out of service) *only* when the
  config has failures, and the failure draws use a key folded from the
  step's, so the other presets' observations and draws are unchanged and
  the port still matches the old repo exactly. JAX matches Python step
  for step, including a test where every car fails and recovers within
  80 steps. `outage` is `full` with 3 failures per car-hour lasting a
  minute on average (about 0.7 in a 200 s episode across 4 cars).
  Collective control delivers 65.8 vs 67.1 on `full` (512 episodes): it
  does not coordinate the cars, so nothing is assigned to a dead car and
  the other three pick up its calls. Small at this rate; longer episodes
  or rarer, longer outages would weigh more.
### Phase 5: on-demand fleet

- [x] Env: grid graph, requests with OD, vehicle assignment + repositioning.
- [x] Heuristics: nearest idle vehicle; per-tick bipartite matching.

**Result.** `envs/fleet/`: a Python reference and a JAX env that match step
for step (all state is integer, so exactly), three presets, and three
rules, the strongest a third better than nearest idle.

- **Model.** Nodes of a G × G grid, one edge per tick (60 s), Manhattan
  travel (x first, then y). Requests arrive as a Poisson process with an
  origin and destination into a buffer of R slots (full: rejected), and
  abandon after `patience` ticks (8) unless taken. Each vehicle carries
  one request. An idle vehicle is the deciding agent: STAY, MOVE one node
  (repositioning), or TAKE request slot r (5 + r actions in all); then it
  drives itself to the pickup and on to the dropoff. Vehicles act in
  index order, so a request two vehicles take goes to the earlier one.
  Reward: minus waiting passenger-minutes, and 10 minutes per abandoned or
  rejected request. Observation per vehicle: its position and job; per
  request slot, the offset to the origin, pickup and trip distances and
  time waited; per node, other idle vehicles and waiting requests.
- **Action design.** "Take slot r" keeps the action space fixed (5 + R)
  and lets every rule be written as a policy over it: nearest idle and
  matching choose slots, repositioning chooses moves. The cost is that a
  learner must learn which slot is which from the slot features; a
  pointer-style policy over requests would be the natural next step.
- **Matching, exactly, in integers.** The matching rule is a minimum-cost
  assignment of idle vehicles to waiting requests (pickup distance), with
  V + 1 dummy "stay" objects costing more than any distance so it serves
  as many as it can. Solved by Bertsekas' auction algorithm: benefits
  −cost × (n + 1) for n bidders and price increments of at least 1 make
  it exactly optimal (tested against brute force on random instances),
  and integer arithmetic makes the JAX `while_loop` version match the
  Python one bit for bit. A Hungarian solver would not vectorise.
- **Rebalancing.** Each vehicle gets a home, the origin-weight quantile
  (v + ½)/V, and an idle vehicle that matching leaves alone heads home.
  The env heuristic is matching + rebalancing.
- **Presets and results** (harness, 512 episodes; return, lost share,
  mean wait):

  | Preset | Nearest idle | Matching | Matching + rebalancing |
  |---|---|---|---|
  | `grid` (5×5, 4 vehicles, 0.5/tick, uniform) | −395 / 2.2% / 3.0 min | −293 / 2.1% / 2.1 | **−286** / 2.0% / 2.1 |
  | `commute` (trips west → east) | −517 / 4.0% / 3.8 | −391 / 3.9% / 2.7 | **−331** / 3.1% / 2.3 |
  | `city` (8×8, 10 vehicles, 1.2/tick, commute) | −3005 / 22% / 8.6 | −1758 / 14% / 4.3 | **−1709** / 13% / 4.2 |

- **What was learned.**
  - Matching beats nearest idle by 25–40% on every preset, most on the
    overloaded `city` (lost requests 22% → 14%). The likely cause (not
    yet measured): serving the longest-waiting request first sends a
    vehicle across the grid while requests next to it wait, and greedy
    choices cannot be undone within the tick. This is the formulation's point that the per-tick
    assignment is a much stronger baseline than nearest idle.
  - Rebalancing matters where demand is asymmetric: 15% on `commute`
    (vehicles end trips in the east and must come back empty), 3% on
    `grid` and `city`. On `city` the fleet is too busy to rebalance: few
    vehicles are ever idle with nothing to take.
  - Where RL should look: anticipatory repositioning (going where
    requests will appear, not where they started on average), and
    holding a vehicle back from a far request for a likely near one,
    which per-tick matching never does.

### Phase 6: benchmark harness

- [x] One command runs random, heuristic and PPO on every env/preset over
      seeds and writes a results table (normalised scores + domain metrics),
      like the old repo's `benchmark.py --readme`. Done for the baselines:
      `python -m transit_bench.benchmark --write docs/results.md`. PPO
      rows: `--ppo-timesteps N` trains MAPPO (PPO where the observation is
      not per agent) on each env with one shared config and adds its
      greedy policy's row (`algos.ppo.Greedy`, now in the library). Not
      yet run for the table: 24 envs at a few million steps each is a long
      run, to be agreed first. Until then PPO numbers come from the
      per-env scripts (`reproduce_elevator.py`, `bus_baselines.py`,
      `train_baselines.py`, `disruption_eval.py`, `response_eval.py`).

**Result.** [results.md](results.md): 14 envs × 2–4 baselines × 512
episodes in about 2 minutes on the GTX 1080. Running every rule on every
preset caught a regression at once: route setting was the train heuristic,
and on double track it was *worse* than first come, first served (1.62 vs
0.87 min per train on `line`), because waiting for a free far platform
stops trains following each other into a one-way section. It now applies
only to sections used in both directions (tested: on double track it is
first come, first served); single-track numbers are unchanged. Lesson:
normalising by the heuristic makes the heuristic part of the benchmark's
definition, so it must be checked on every preset, not just the one it was
designed for.

### Phase 7: coupled modes (stretch)

- [x] Bus bridging: when a rail block closes, buses run a replacement
      service between the affected stations. Train and bus envs share one
      core state. No paper found does short-turning and bridging jointly.
      Done as a first step: buses as a shared resource inside the train
      env (`bridge_buses`, BRIDGE, `train/bridge-v0`), not a coupling of
      the two envs' states (Result below).

**Result.** Short-turning and bridging decided jointly, by one rule, on one
line; bridging pays when missed calls are dear.

- **Model.** A pool of replacement buses waits at the depot (the middle
  station). BRIDGE ends a trip as TERMINATE does and sends the
  lowest-numbered free bus: it drives from the depot to where the train
  stopped (its origin, for a cancellation), then through the train's
  remaining calls on a road alongside the line, `bus_slowdown` (1.5)
  times slower than a stopping train, dwelling at each. Each call costs
  the bus's lateness there against the train's timetable, capped at the
  missed-call penalty, and the bus is busy until it is back at the
  depot. Bus travel has no noise, so a bridge's cost and the bus's return
  are known when it is sent (`bridge_plan`). Observation: can bridge,
  and the bridge's cost (`HEAD_FIELDS` 17 → 19). JAX matches Python step
  for step under random five-action play and the rule.
- **Why not couple the bus env.** The bus env is a loop of stops with
  passengers and headway holding; a replacement service is a shuttle
  along the rail line, timed against the train timetable. Sharing one
  core state between the two envs would mean a common clock, passenger
  model and network, which is the real Phase 7 design question; the
  resource model gets the decision (when to end a trip, with or without
  a bus) without it.
- **Rule.** `TerminateWhenStuck` now ends a trip the cheaper way: BRIDGE
  if a bus is free and bridging costs less than the penalty, else
  TERMINATE; free buses go to trains in index order, and the stopping
  threshold uses the cheaper cost.
- **Results** (2,048 episodes, seed 7; cost per train in minutes; the
  same scenarios for every row, since only the bus pool differs):

  | Policy | Penalty 5 min/call | Penalty 15 (`bridge`) |
  |---|---|---|
  | Route setting | 4.98 / CVaR 25.8 | 4.98 / 25.8 |
  | Terminate, no buses | 3.52 / 15.6 | 4.67 / 22.5 |
  | Terminate + 1 bus | 3.44 / 15.3 | 4.01 / 19.2 |
  | **Terminate + 2 buses** | 3.40 / 15.1 | **3.75 / 17.8** |
  | Terminate + 4 buses | 3.39 / 15.0 | 3.67 / 17.2 |

- **What was learned.**
  - Where a missed call is dear (a 30-minute headway), terminating alone
    barely helps (4.98 → 4.67): the penalty outweighs most waits. Two
    buses make it 20% better (3.75) and terminate more often (0.40 →
    0.53 trains per episode), because a bridged call costs
    the bus's lateness, not the full penalty: buses change *when* to
    short-turn, not just what it costs. That is the joint decision.
  - Returns diminish fast: the first bus saves 0.66 min per train, the
    second 0.26, the third and fourth 0.08 between them. Presumably
    because outages rarely overlap on one line (not measured).
  - At a 5-minute penalty buses add little (3.52 → 3.40): the cap means a
    bridged call can save at most 5 minutes, and a bus that has to come
    from the depot is often later than that.

- [x] Coupled corridor (`corridor/*`): the train line and a bus route
      along it in one state, with one clock, one network and one
      passenger model. A short-turned train's riders move to the bus.

**Design.** The three things the two envs have to share, and the choice
made for each:

- **Clock.** One tick for both (`step_seconds` must agree). Within a tick,
  the trains act first, then stranded riders join the bus stops, then the
  buses act. So a bus at that stop can board them in the same tick.
- **Network.** The bus route is built from the rail line: a loop of
  2(n − 1) stops, out along the stations and back. Stop s is station s
  eastbound, and stop 2(n − 1) − s is station s westbound (the two end
  stations are shared). Each bus edge takes the road time of one section
  (`bus_slowdown` x the stopping train's running time, as for bridging).
  Rail blockages and road closures stay independent processes: a
  correlated one (one flood closing both) is a backlog item.
- **Passengers.** Each train leaves its origin with `riders_per_call`
  riders for each later call. That is the simplest rail model with a
  destination per rider, which is what a transfer needs. While a train
  runs, every rider still on board pays its projected lateness: weight
  r x (calls left) on the change in the train's projected delay each
  tick. Summed, a rider's cost is about the train's lateness at their
  stop. If the train terminates, its riders get off at that station (or
  never left the origin). They join the bus stop for that station in the
  train's direction, each with the bus stop of their own call as
  destination. From then on they are ordinary bus passengers and pay bus
  passenger-minutes (waiting and riding). This replaces the train env's
  fixed missed-call penalty with a cost the bus state decides.
- **Common units.** The reward is minus passenger-minutes over everyone.
  For rail riders that means lateness; for bus riders, all their time.
  To put transferred riders on the lateness scale, each one earns a
  credit of 1 minute per minute from the transfer until its scheduled
  arrival plus the lateness already charged (`credit_end`). Over an
  episode a transferred rider then costs its bus arrival minus its
  scheduled train arrival. The credit is paid per tick rather than all
  at once: a lump at the transfer would make terminating look good
  straight away, followed by a slow bill (a credit-assignment trap for
  RL). Paid per tick, the reward comes to zero while the bus keeps them
  on time.
- **Agents.** The trains and then the buses, with the train's action set
  (GO, WAIT, TERMINATE). For a bus, 0 is GO and anything else is HOLD.
  The observation has one block per agent: a type flag, the agent's own
  env features padded to the longer of the two, and two coupling fields
  for trains. These are the expected wait for the next bus at its station
  in its direction, and the expected extra journey time by bus over the
  train for its riders (`transfer_cost`).
- **The joint rule.** `TerminateWhenStuck` with the bus in the loop. A
  train held at a closed block terminates when factor x the outage's
  expected time left exceeds `transfer_cost`, its riders' mean extra time
  by bus. That cost is: the wait until the next bus reaches the station,
  plus the nominal bus ride to each call, minus the train's timetabled
  time there. The fixed penalty x missed calls becomes a quantity read
  from where the buses are. Baselines: no termination (route setting +
  headway holding), and the bus-blind rule (the fixed 5-minute penalty).
- **Why composition.** The corridor sim holds a `TrainSim` and a
  `BusSim`, and the JAX env holds a `TrainState` and a `BusState`. The
  coupling sits between their steps (the bus stops' waiting counts, the
  credits), so neither env changes and both keep their parity tests.
  The cost is that the rail passenger model stays coarse, at the
  corridor level: a full one (per-station boarding, train capacity)
  belongs in the train env (backlog).

**Result.** `corridor/corridor-v0` and `busy-v0` (twice the bus demand).
The JAX env matches the Python reference step for step, under random play
and all three rules (`tests/test_corridor.py`). The rail line is
`blockage` with TERMINATE: single track, 10-minute outages. The bus route
is a loop of 8 stops with 4 buses, at about the train's 10-minute headway.

- **Results** (`scripts/corridor_eval.py`; 2,048 episodes, seed 7;
  passenger-minutes per episode; "vs blind" is the paired difference to
  the bus-blind rule at factor 2):

  | Policy | `corridor`: cost | vs blind | rail | bus | terminated/ep | `busy`: vs blind |
  |---|---|---|---|---|---|---|
  | No termination | 10,362 | +166 ± 16 | 675 | 9,687 | 0 | +150 ± 15 |
  | Bus-blind, factor 1 | 10,283 | +88 ± 11 | 585 | 9,698 | 0.14 | +72 ± 10 |
  | Bus-blind, factor 2 | 10,195 | 0 | 441 | 9,754 | 0.37 | 0 |
  | **Bus-aware, factor 2** | **10,175** | **−20 ± 5** | 382 | 9,793 | 0.45 | −6 ± 5 |
  | Bus-blind, factor 4 | 10,175 | −20 ± 5 | 382 | 9,793 | 0.45 | −6 ± 5 |

- **What was learned.**
  - Terminating a blocked train pays once a bus is there to take its
    riders: 166 passenger-minutes an episode (1.6% of the total, but
    25% of the rail riders' share once the bus's extra cost is netted off). Rail lateness falls from 675 to 382; the bus pays
    back 106 of that for about 10 transferred riders, roughly 10 minutes
    late each.
  - The bus-aware rule wins by terminating *more* (0.45 vs 0.37 trains per
    episode), not by timing it to where the buses are. Its estimated cost
    (a few minutes of extra journey) is nearly always below factor x the
    10-minute outage, so it makes the same choices as the bus-blind rule
    at factor 4, which is "terminate whenever blocked". At smaller
    factors, where the bus's position would tip decisions (0.25–0.75),
    the paired differences are within ±5 passenger-minutes, i.e. noise.
    The fixed 5-minute penalty overstates what a transfer costs when a
    bus runs alongside every 10 minutes.
  - Transfers cost more on a busy route: the gain over bus-blind drops
    from 20 to 6 (not significant), because transferred riders crowd the
    buses and slow boarding for everyone. `transfer_cost` ignores
    crowding, so a rule (or a learner) that priced it would do better
    there. That is a backlog item, and a place where coupling matters
    that a single-mode env cannot show.
  - The normalised score is dominated by the bus: random play holds
    buses (2 of its 3 actions are HOLD), which costs far more than any
    train decision. The rail rules sit between 0.95 and 1.00. Paired
    differences in passenger-minutes are the meaningful comparison here;
    the harness reports rail and bus costs separately for that reason.

### Phase 8: general rail networks

The train env on any network of stations and sections, not just a line.
This opens the backlog's random-network generalisation, and makes the
waterway and taxiing applications into presets (applications.md).

- [x] **Networks as config.** A `Network` gives the stations, the edges
      (single or double track) and the services (a station path, a
      number of trains, a headway and an offset). `infra.build` builds
      platforms, blocks, routes and the timetable from it. The line
      builder stays, so the line presets are unchanged.
- [x] **Heading instead of direction.** The conflict logic asks "is a
      train going the other way in this shared section?". On a line,
      "the other way" is east against west. On a network it is the
      direction each train traverses that section. A per-position
      `heading` table replaces `direction` in the section rules, the
      section observation fields and the "same direction" lookahead
      field. On a line, heading equals direction, so the line presets
      must give the same returns as before (checked against the results
      table).
- [x] **Variable route lengths.** Services have different lengths. The
      derived tables and the hindsight bound stop at each train's route
      length instead of the padded width.
- [x] **Presets.** `branch`: a double-track trunk splitting at a junction
      into two single-track branches, one service per branch. Trains
      merge at the junction and cross on the branches. `branch_outage`
      adds the blockage process. Rules, bound and harness rows for both.
- Deferred: relief tracks, passing loops and expresses on networks;
  networks drawn at random per episode (the infrastructure becomes part
  of the state rather than constants); route choice between paths.

**Result.** The train env runs on any network given as a `Network`. Its
step logic needed no change beyond headings: everything else already
worked on routes over blocks.

- **Line presets unchanged.** The line builder produces the same arrays as
  before, and all 40 train and corridor baselines return bit-identical
  means (64 episodes each). On a line, heading equals direction.
- **Parity.** JAX matches Python step for step on `branch`,
  `branch_outage` and a test network (`MIXED` in tests/test_train.py).
  The test network has a spur, both kinds of track, routes of two
  lengths, blockages and TERMINATE. The hindsight bound handles routes of
  different lengths, and stays below every finished episode there too.
- **Results** (bound: `scripts/bound_eval.py`, 100 scenarios; harness:
  512 episodes; minutes per train):

  | Preset | Bound | Section reservation | Route setting (the heuristic) | Route setting above bound | Section reservation deadlocks |
  |---|---|---|---|---|---|
  | `branch` | 1.06 | 1.16 | 1.38 | **23%** | 1.4% of episodes |
  | `branch_outage` | 2.34 | 2.70 | 3.17 | **26%** | 8.2% |

  (Each rule's delay is over the episodes it finished: route setting 98%
  and 95%, section reservation 97% and 91%. The bound column is over
  route setting's episodes. On `branch_outage`, 5% of solves hit the 20 s
  limit and report the dual bound, which is still valid.)
- **What was learned.**
  - The network gives the rules their largest gap to the bound so far:
    23–26%, against 8% on `single` and 12% on `blockage`. Alongside
    `passing` (28%), this is where a learner or a better rule has room.
  - Route setting is the only rule that never deadlocks, but it gives
    away more delay than on the line. Section reservation is 0.1–0.5 min
    per train faster on average, at the price of deadlocks. A probable
    cause (not measured) is the junction station: it has one platform
    per heading for both branches, and route setting holds a train
    until the platform at the far end of a single-track section is free,
    so trains from the two branches wait for each other there. Next
    steps: a rule that reserves platforms more finely, and an
    analysis of where the bound's schedules differ from route setting
    (as `give way to express` came out of the `passing` schedules).

## Backlog

- New cores from [applications.md](applications.md), in the recommended
  order:
  - runway sequencing (sequence-dependent separation times; exact bound,
    constrained position shifting as the rule);
  - electric bus charging (a charge state and charger resources, on `bus`
    or `fleet`);
  - lock scheduling (a batch server with packing and a direction setup);
  - capacity above 1 (cumulative constraints: air traffic flow, channels,
    depots);
  - last-in, first-out storage (depot shunting, container stacking).

  Extensions: a shared lift shaft on `elevator`, capacity-1 nodes on
  `fleet` (warehouse robots), and a throughput objective with cyclic jobs
  (mine haul roads).
- Paper references: check and complete the references in
  applications.md (cited from memory) and add them to related-work.md.
- When the backlog runs low, revisit [applications.md](applications.md):
  assess more applications and pick the next new core from it.

- Corridor: a crowding-aware `transfer_cost` (room on the next bus, the
  dwell the extra boarding adds for everyone on board); bus holding for
  transfers (connection protection); a rail passenger model inside the
  train env (boarding per station, capacity), so riders waiting at missed
  calls count too; correlated disruptions across modes (one event
  closing a rail block and the road beside it); MAPPO on the corridor,
  where trains and buses share one policy with a type flag.

- GTFS loader for real bus and train networks and timetables.
- Graph-network policies (the old repo has a GNN experiment in `scripts/gnn.py`).
  Needed for rail generalisation: train on random networks (Flatland-style
  generator), test on unseen ones and on DISPLIB networks.
- Train env: prioritised replanning baseline (space-time reservations
  along each train's route, replanned on delay); a look-behind observation
  (the next train behind on the same track, its speed and delay) for
  overtaking on `passing`; passengers and connections; longer PPO runs and
  longer rollouts for deadlock credit; a deadlock penalty or early
  termination as a named shaping.
- Safe RL for rail: mask GO actions that a deadlock check (route setting,
  or a look-ahead on the wait-for graph) rules out, so exploration cannot
  cause deadlocks; compare with a deadlock-penalty shaping.
- Adversarial disruptions: a second agent picks which edge to block
  (attacker–defender), for a worst-case robustness score.
- RL proposes, a solver repairs: RL picks train orderings and a replanner
  makes them feasible.
- Semi-MDP / event-driven decisions, if fixed ticks prove too coarse.
- Bound: a `--bound` option in the harness; a TERMINATE-aware MILP
  (optional calls with a penalty) for `response`; slow running (speed
  depends on the entry tick); a rolling-horizon MILP policy (DLA) using
  only what is known at decision time, to split the gap into
  value-of-information and policy suboptimality.
- Harder train presets with headroom: denser single-track timetables
  (more trains, shorter headways), and `passing` with a look-behind
  observation (the train behind and its speed).
- Train response actions: vehicle circulation (each eastbound trip's
  vehicle forms a westbound trip after a layover), so a short-turn also
  runs the return trip early from the turning station and gets credit
  for the calls it serves; partial cancellations that skip stops; a
  rule that prices the queue behind a blocked train (trains waiting for
  it × expected time left); a value baseline for TERMINATE from rollouts
  (DLA, Powell's fourth class).
- Train failures (a train stops in its block for a random time), as done
  for buses and lifts; demand surges in the core process. A closure-aware bus rule (hold
  upstream buses while a road is shut). Bus response actions for
  breakdowns: a spare bus, or short-turning the bus behind.
- Route choice: general alternative routes on a graph (relief tracks
  only cover a parallel track of equal length; a diversion via other
  stations, skipping calls, needs per-route tables); an exact AND-OR
  deadlock check for trains with two ways out; the shield and route
  setting aware of the relief track (the shield vetoes on the main
  track only); a reroute rule that weighs waiting for the main track
  (expected time left) against the slower relief run.
- CI: the workflow (`.github/workflows/ci.yml`) is not on GitHub yet: the
  gh token lacks the `workflow` scope (`gh auth refresh -s workflow`),
  so PR #1 went up without it. It is on the local `plan-rail-disruptions`
  branch.

## Open questions

- Fixed tick length per env: one shared unit (seconds) or per-env?
  Leaning: one shared unit (seconds), each env sets its tick length; coupled
  modes (phase 7) need a common clock.
- Single-agent central controller, multi-agent, or both as first-class?
  (The old repo had both; MAPPO was the newer path.)
