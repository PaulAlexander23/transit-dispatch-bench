# Formulation

The mathematical model behind the suite. Every env is a special case of one
sequential decision problem; the core's state layout follows from it.

## The general problem

A network `G = (V, E)`. Each edge `e` has a nominal travel time `τ_e` and a
capacity `κ_e` (how many vehicles may be on it at once). Vehicles `k` have a
capacity `Q_k`. Requests `r = (o_r, d_r, t_r)` appear at random: origin,
destination, arrival time.

In Powell's sequential decision framework:

- **State** `S_t = (R_t, D_t, Z_t)`
  - `R_t`: vehicles: position (node, or edge plus progress), load, onboard
    passengers and their destinations.
  - `D_t`: waiting requests.
  - `Z_t`: the **disruption state**: the current capacity `κ_e(t)` and speed
    factor `σ_e(t)` of each edge, and each vehicle's status (running, failed).
- **Decision** `x_t = X^π(S_t)`, from a policy `π`.
- **Exogenous information** `W_{t+1}`: new requests, travel-time noise, and
  changes to `Z`.
- **Transition** `S_{t+1} = f(S_t, x_t, W_{t+1})`.
- **Objective** `min_π E[ Σ_t C(S_t, x_t) ]`, e.g. passenger waiting plus
  in-vehicle time, or train delay.

With several vehicles each acting on its own observation this is a
decentralised POMDP (Dec-POMDP); with one central controller it is an MDP over
the joint action.

### Policy classes as baselines

Powell's four classes give one baseline family each:

| Class | Meaning | In the suite |
|---|---|---|
| PFA, policy function approximation | A rule mapping state to action | Heuristics (collective control, headway holding, nearest vehicle) |
| CFA, cost function approximation | A tuned or parameterised optimisation | Tuned dispatch rules |
| VFA, value function approximation | Act greedily on a learned value | RL critics (PPO's value head) |
| DLA, direct look-ahead | Plan over a horizon, replan | Rolling MILP / prioritised replanning |

A hindsight solve, with all of `W` known in advance, bounds every real-time
policy from above. The gap between a policy and that bound is the cost of not
knowing the future.

## The envs as special cases

### Elevator

`G` is a line graph (the floors); each car's shaft is its own set of edges.
Requests are hall calls; the decision per car is move up/down, open doors, and
the hall-call direction it serves.

### Bus

`G` is a loop of stops. The headway `h_k` (time between bus `k-1` and bus
`k` at a stop) is unstable: dwell time grows with the passengers waiting, which
grows with the headway, so a late bus falls further behind. With dwell
`β · λ · h` at each stop (`λ` the arrival rate, `β` boarding time per
passenger), a small delay `δ` grows by a factor `1 + βλ` per stop: bunching.
Holding the bus at a stop for `α (H − h_k)` (target headway `H`, gain `α`)
is feedback control that stabilises it (Daganzo 2009). That near
closed-form controller is the baseline RL has to beat.

### Train

A **job shop with blocking**, on the *alternative graph* (Mascis and
Pacciarelli 2002). The network is split into track blocks, each a resource of
capacity 1. An operation is train `i` occupying block `b`, starting at
`t_ib` for running time `p_ib`.

- Route order: `t_{i,b'} ≥ t_ib + p_ib` where `b'` follows `b` on `i`'s route.
- Timetable: `t_{i,s} ≥ dep_{i,s}` (no departure before the scheduled time).
- Blocking: if trains `i` and `j` both use block `b`, one must leave it
  before the other enters:

  `t_{j,b} ≥ t_{i,b⁺}`  **or**  `t_{i,b} ≥ t_{j,b⁺}`

  where `b⁺` is the block after `b` on each train's route. As a MILP: a binary
  `y_ijb` choosing the order, with a big-M on each side; Lamorgese and Mannino
  (2019) give a tighter noncompact formulation.
- Objective: weighted lateness `Σ_i w_i · max(0, t_i,end − sched_i)`, or
  passenger delay when passengers are modelled.

In the alternative graph each operation is a node, each fixed constraint an
arc, and each disjunction a pair of alternative arcs. A schedule chooses one
arc per pair; it is feasible if the chosen graph has no positive-length cycle,
and a **deadlock is such a cycle**.

So the natural RL action is **one ordering choice at a conflict: which train
goes first into a contested block**. This is a small discrete action with a
clear meaning, and matches what a dispatcher actually decides.

Disruptions enter as data:

- Block `b` closed on `[a, c]`: a fixed dummy operation occupies `b` for that
  window, or equivalently `κ_b(t) = 0` there.
- Slower running: `p_ib ← σ_b · p_ib`.
- Train failure: train `i` stays in its current block for a random time.

Disruption responses add actions: short-turn (reverse at an intermediate
station), cancel a service, reroute through an alternative block sequence.

### Fleet

Dynamic dial-a-ride on a grid or road graph. Each tick, assigning idle
vehicles to waiting requests is a bipartite matching (Hungarian algorithm or
an LP), which is a much stronger baseline than nearest idle vehicle. RL earns
its keep on **repositioning** idle vehicles ahead of demand.

## Disruptions as a stochastic process

Each edge has a state `z_e ∈ {up, down}`.

- **Markov (memoryless):** it fails at rate `λ_e` and is repaired at rate
  `μ_e`, so outages last `Exp(μ_e)`. Seeing that an edge is down says nothing
  about how long it will stay down.
- **Semi-Markov:** outage durations from a heavy-tailed law (lognormal). The
  time already spent down then predicts the time left, so the agent should see
  the elapsed outage time. More realistic, and harder.
- **Scripted:** a fixed scenario ("block 4–5 closed from t=600 to 1800 s"),
  for reproducible evaluation.

The **information structure** is part of the env: whether the agent sees the
remaining outage time (fully observed) or not (partially observed).

Vehicle malfunctions follow the same pattern per vehicle (Flatland uses a
per-step probability and a duration interval). Demand surges scale the OD
matrix for a window.

### Evaluation under disruption

- In-distribution: train and test with the same disruption process.
- Shift: train without disruptions, test with them.
- Tail metrics: CVaR at 10% of delay (the mean of the worst 10% of episodes)
  alongside the mean.
- Later, adversarial: a second agent picks which edge to block (an attacker–
  defender game), giving a worst-case robustness score and the network's weak
  points.

## JAX implementation notes

- **Fixed shapes.** Pad to max vehicles, max passenger slots, max edges;
  masks mark the live entries.
- **Capacities as one array.** `κ_e(t)` and `σ_e(t)` are arrays in the state;
  every disruption is a write to them, so all envs get disruptions from the
  core.
- **Deadlock detection without data-dependent loops.** With `N` trains, build
  the N×N wait-for matrix `A` (train `i` waits for a block train `j` holds).
  Square `(A ∨ I)` for `⌈log₂ N⌉` steps to get reachability; train `i` is in
  a cycle if it reaches itself through at least one edge (`(A · reach)_ii`).
  Fixed shape, works under `jit`.
- **Sparse decisions.** Rail decisions are rare relative to ticks; mask
  agents with no decision rather than going event-driven first, and measure
  the cost.
