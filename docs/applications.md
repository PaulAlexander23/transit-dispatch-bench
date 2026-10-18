# Applications

Which transport problems the suite's models cover, and which would need a
new one. The maths of the existing models is in
[formulation.md](formulation.md).

## Method

1. **Take an application** (a lock on a canal, a runway, a depot).
2. **Distil its hard core**: the mathematical structure that makes it
   difficult. This is often a familiar problem class (job shop with
   blocking, single-machine sequencing, flow over time), with whatever
   detail that structure doesn't depend on stripped away.
3. **Compare with the cores already modelled.**
   - If the core is the same, record the application here as an instance
     of the closest existing env, with how it maps. Don't build an
     identical model under a new name.
   - Build only for genuinely new structure. Where the new part is small,
     prefer extending an existing env (a preset, a config field) over
     writing a new one.
4. **Model it** as an env: a Python reference, a JAX env, parity tests, a
   heuristic, presets that add one feature at a time and, where small
   enough, a hindsight bound.
5. **Apply algorithms from each of Powell's policy classes** (rules,
   tuned rules, learned values, look-ahead planners; see formulation.md)
   and compare them with the bound.

## Cores already modelled

| Core | Env | The hard part |
|---|---|---|
| Dynamic assignment on a line | `elevator` | Assigning calls to cars online, with stochastic demand |
| Headway regulation | `bus` | Unstable dynamics: a late bus collects more passengers and falls further behind; holding is feedback control |
| Job shop with blocking | `train` | Ordering decisions at shared capacity-1 resources; deadlock as a cycle; route choice; stopping (terminate) |
| Dynamic matching and repositioning | `fleet` | Assignment each tick, and moving idle vehicles ahead of demand |
| Coupled modes | `corridor` | One mode's decision (ending a trip) sets the other's demand |
| Sequence-dependent setups | `runway` | A service's duration depends on the one before it |
| Disruptions (shared by all) | `core.disruption` | Markov, semi-Markov or scripted outages of edges and vehicles |

## How general the train model is

The train env's core is a general model: vehicles follow fixed routes
through resources that hold one vehicle each. A vehicle keeps its
resource until it can enter the next. Delays are measured against a
timetable, resources close at random, and the responses are cancel,
short-turn, reroute and bridge.

In the code, the step logic (`train/sim.py`, `train/jax_env.py`) already
works on generic arrays: routes, running times, dwells and the timetable.
The rail-specific parts are the infrastructure builder (`infra.build`
makes a line of stations) and a few rules, such as terminating only at a
platform. Two backlog items would open up most applications below:

- **General route graphs**: for airports, warehouses and branching
  networks.
- **Resources with capacity above 1**: for airspace sectors, wide
  channels and depots.

## Applications

Each entry gives the core, the closest existing model, the new complexity
it adds (if any) and a verdict:

- **note**: an instance of an existing model; recorded here, not built;
- **extend**: a small addition to an existing env;
- **new**: needs a new model.

### Summary

| Application | Closest model | New complexity | Verdict |
|---|---|---|---|
| Single-lane waterways (fairways, passing places) | `train/single` | None | note |
| Locks | `train` | Batch server, packing, direction setup | new |
| Mine haul roads | `train/single` + `fleet` | Closed population, throughput objective | extend |
| Tram and light rail | `train` + `bus` | Periodic resource availability | note |
| High-frequency metro | `bus` holding + `train` blocking | Headway control without overtaking | note |
| Several lift cars in one shaft | `elevator` | Fixed vehicle order on a shared track | extend |
| Airport taxiing | `train` + route graphs | None beyond route graphs | note |
| Runway sequencing | `runway` (built, Phase 9) | Sequence-dependent separation times | new |
| Airline disruption recovery | `train` TERMINATE + circulation (backlog) | Chains of trips per vehicle and crew | note |
| Air traffic flow management | `train` | Capacity above 1: cumulative constraints, flow over time | new |
| Depot shunting, container stacking | (none) | Last-in, first-out storage | new |
| Warehouse and terminal robots | `fleet` + `train` | Assignment costs that depend on congestion | extend |
| Electric bus charging | `bus` / `fleet` | A consumable vehicle state (charge) | extend |

### Single-lane waterways

Ships on a canal or river channel can only pass each other at wider
stretches. That is the single-track line exactly:
- channel stretches are sections;
- passing places are crossing loops;
- ships are trains;
- the risk is the same head-on deadlock.

Fog and low water are block closures and slowdowns. **Note**: an instance
of `train/single` and `train/blockage`.

### Locks

A lock is not a track block.
- **It is a batch server.** It takes several ships per cycle, and the cycle
  time is fixed whatever the batch.
- **Its batches must be packed.** Ships have to fit the chamber, which is a
  two-dimensional packing problem.
- **It has a setup that depends on its state.** The chamber is at one level
  and must cycle, empty if need be, to serve the other direction. So the
  cost of serving a ship depends on the direction last served.

The decisions are when to start a cycle (wait for more ships, or go now)
and which ships to take. This is batch scheduling with sequence-dependent
setups (the lock scheduling problem, e.g. Verstichel et al.). None of the
current envs batches at a resource. **New**. The fairways between locks
reuse the train model.

### Mine haul roads

Trucks shuttle between loaders and dumps on roads that are partly single
lane with passing bays. The blocking part is `train/single`. What is new:
- **No timetable.** The objective is throughput: tonnes moved, and loaders
  kept busy.
- **A closed population.** The same trucks cycle all shift, which makes it
  a closed queueing network with routing control.
- **The decision is which loader to send a truck to.** That is the
  fleet's assignment, made at the dump.

**Extend**: a throughput objective and cyclic jobs on the train model,
with assignment on top. Worth it only for the closed-population,
throughput objective, which no env has yet.

### Tram and light rail

Signalled sections like rail, plus street running, where a traffic signal
makes the next section unavailable on a fixed cycle. That is a periodic
scripted closure, which `core.disruption` already supports. Signal
priority (changing the signal for the tram) would couple to road traffic,
which is a different problem (SUMO-RL territory). **Note**: `train` with
scripted periodic closures; short-turning and bus bridging carry over.

### High-frequency metro

At headways of 2–3 minutes passengers ignore the timetable, so the
objective is regular headways, as on the bus loop. The difference is that
trains can't overtake: the bus env's holding control acts under the train
env's blocking constraints. **Note**: a combination of `bus` (objective
and holding) and `train` (blocking). Build it only if that interaction,
holding when the train ahead blocks you, becomes the question.

### Several lift cars in one shaft

Two cars share one shaft (twin systems), or ropeless cars circulate up one
shaft and down another. The cars can never pass, so their order is fixed:
the top car can't serve the lowest floors while the bottom car is above
them. Unlike the train env there is no ordering decision. The constraint
cuts down which assignments are feasible and adds a deadlock-like case
(each car waiting for the other to move). **Extend**: the elevator env
with a shared shaft. It is cheap because it reuses the elevator code and
its collective-control baseline, and it is the closest link between the
elevator and train models.

### Airport taxiing

Taxiway segments are blocks of capacity 1, aircraft are trains with
routes from gate to runway, and pushback slots are the timetable.
Taxiway deadlock and route choice are the train env's. **Note**: `train`
once it has general route graphs (backlog). It adds nothing beyond them.

### Runway sequencing

A runway is a single server, but the minimum gap between two aircraft
depends on both of them: wake turbulence separates a light aircraft
behind a heavy one by more than the reverse. That is single-machine
scheduling with *sequence-dependent setup times* and time windows: the
aircraft landing problem (Beasley et al. 2000). It is closer to an
asymmetric travelling salesman problem than to a job shop. In the train
env a train's time in a block depends only on that train, never on which
train went before. **New**. It is small and crisp, with strong bounds: an
exact MILP or DP for short horizons, and constrained position shifting
(each aircraft at most k places from first come, first served) as the
classical rule.

Built as `runway/*` (Phase 9 of the plan). The bound turned out to be an
exact DP rather than a MILP: grouping the aircraft by class shrinks the
state to the counts landed per class, with a Pareto front per state for
the release times. On 36 aircraft an hour, FCFS is 30% above the bound
and constrained position shifting (k = 2) 19%.

### Airline disruption recovery

Cancelling a flight is TERMINATE. What is new is that each aircraft and
crew flies a chain of trips, so cancelling one leg breaks its rotation:
the aircraft is in the wrong place for its next flight. Recovery swaps
aircraft or crews between chains (set partitioning or network flow over
the chains). That is the vehicle circulation item already on the train
backlog (a return working for each short-turn). **Note**: one model of
vehicle circulation would serve rail rolling stock and airlines; build
it there.

### Air traffic flow management

Airspace sectors take a limited number of aircraft per period (capacity
above 1), weather cuts that capacity partly rather than closing the
sector, and the decisions are ground delay (WAIT before departure) and
rerouting (DIVERT). Bertsimas and Stock Patterson (1998) give the
standard formulation. Capacity above 1 changes the mathematics:
- **The pairwise "who goes first" choice becomes a cumulative constraint**:
  at most κ aircraft in a sector at once.
- **Deadlock mostly disappears.**
- **The problem becomes rationing slots in a flow over time.** At scale,
  it is usually solved on aggregate flows rather than individual
  aircraft.

**New**: it shares the capacity-above-1 backlog item, and wide channels
and depot tracks would reuse it.

### Depot shunting and container stacking

A dead-end stabling track is a stack: the last unit in is the first out.
So the order units arrive in decides which departures they can serve
without moving others. The core is assigning units to departures under
last-in, first-out storage (train unit shunting, Freling et al. 2005).
Container yards have the same stacks, where the cost is the reshuffles
needed to reach a container (the block relocation problem, Caserta et al.
2012). No current resource remembers the order it was filled in. **New**:
one model would cover both. It is more planning than real-time dispatch,
so it fits the suite less naturally than the others.

### Warehouse and terminal robots

Robots pick up and deliver tasks on a grid where every cell holds one
robot: multi-agent pickup and delivery (Ma et al. 2017). It is `fleet`'s
assignment combined with `train`'s blocking on a dense graph where routes
are not fixed. What the combination adds is that assignment costs depend
on congestion: sending a robot down an aisle delays the others, so the
cheapest matching on travel distance may not be the best. **Extend**:
`fleet` with capacity-1 nodes and collision rules. Worth building when
that interaction is the question; otherwise each half is already
modelled.

### Electric bus charging

The vehicle gains a state that is used up along its trips and restored
only at chargers: its charge. A bus can do its 5th trip only if its
earlier trips and charging left enough. That couples a vehicle's whole
day, and the chargers are shared resources with queues. It is the
electric vehicle scheduling problem (e.g. Wen et al. 2016). **Extend**:
a charge state and charger resources on `bus` or `fleet` (robotaxis
need the same). It is the smallest of these additions that adds a
genuinely new kind of state.

## What the new structures add

Of the applications above, these bring structure no current env has:

| Structure | Where it appears | What it changes |
|---|---|---|
| Sequence-dependent setups | Runways, lock direction | A service's cost depends on the one before it: ATSP-like rather than a job shop |
| Batch service with packing | Locks | The server's decision is when to start and what to take |
| Capacity above 1 | Air traffic flow, channels, depots | Pairwise orders become cumulative constraints: congestion, not deadlock |
| Last-in, first-out storage | Shunting, container yards | The order of arrivals limits the order of departures |
| Consumable vehicle state | Electric buses, robotaxis | Decisions now affect what the same vehicle can do later in the day |
| Closed population, throughput objective | Mining | No timetable; maximise flow through shared servers |
| Congestion-dependent assignment | Warehouse robots | Assignment and routing can no longer be solved separately |

**Recommendation.**
- **Next, for the train work:** general route graphs, already on the
  backlog. They turn waterways and taxiing into presets.
- **First new model: runway sequencing.** It is the smallest env with the
  most distinct mathematics, and has an exact bound and a classical rule
  to compare RL against.
- **Most useful extension: electric bus charging.** It adds a state that
  the bus and fleet envs could both use.

The references here are from memory, not re-checked this session; check
them against [related-work.md](related-work.md) before relying on them.
