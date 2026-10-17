# Related work

Surveyed 2026-10-17 from web searches; papers were read at abstract level
unless noted. The aim is to state the suite's niche with evidence.

## Summary

| Work | Covers | Speed / stack | Gap this suite fills |
|---|---|---|---|
| [Flatland-RL](https://github.com/flatland-association/flatland-rl) ([paper](https://arxiv.org/abs/2012.05893)) | Multi-agent trains on a grid of rail tiles, random malfunctions (probability + duration) | Python, v4.1 on PyPI, run by the Flatland Association | No passengers (goal is reaching a target), tile grid not a track graph, slow |
| [DISPLIB](https://arxiv.org/abs/2509.12254) (2025) | Real train dispatching instances from several infrastructure managers, a verifier and a reference MILP | Static instances | Not sequential; usable as test networks and for hindsight bounds |
| [Jumanji](https://arxiv.org/abs/2306.09884) | 22 JAX envs incl. CVRP, multi-CVRP, TSP | JAX | Routing envs are static (all customers known up front) |
| [RL4CO](https://arxiv.org/abs/2306.17100) (KDD '25) | 27 combinatorial-optimisation envs, 23 baselines | PyTorch/TorchRL | Static construction problems, not real-time dispatch |
| [MAEnvs4VRP](https://arxiv.org/abs/2411.14411) | Multi-agent VRP envs | PyTorch | Static VRP |
| [EURO meets NeurIPS 2022](https://euro-neurips-vrp-2022.challenges.ortec.com/) ([report](https://proceedings.mlr.press/v220/kool23a/kool23a.pdf)) | Static and dynamic VRPTW on real road times; requests revealed in hourly epochs | Competition code (HGS solver) | One mode, epoch-based; the precedent for `fleet` |
| [BusEnv](https://researchcommons.waikato.ac.nz/items/a37e4c88-4581-4659-a514-76873089101f) (AAMAS 2026) | MARL bus holding, Salvador (Brazil) data, 9 MARLlib baselines | Python | Bus only; licence not confirmed |
| [Bus holding platform](https://arxiv.org/abs/2410.10212) (IEEE OJ-ITS 2024) | Open-source bus holding testbed, model-based and model-free control | Python | Bus only |
| SUMO-RL, OR-Gym | Traffic signals; small OR envs (inventory, knapsack, VRP) | Python | Neither covers passenger dispatch on networks |

No JAX rail environment turned up in the searches; check GitHub directly before
claiming there is none.

## Rail

- **Flatland.** The 2020 challenge was won by a planning-and-replanning method
  ([Li et al.](https://people.eng.unimelb.edu.au/pstuckey/papers/flatland.pdf)),
  which beat every RL entry. Lesson: a rail benchmark needs a planning
  baseline, not only first-come-first-served, or RL looks better than it is.
  Recent RL work: Maze-Flatland, a semi-hierarchical RL method that splits
  dispatching from routing ([arXiv 2605.10257](https://arxiv.org/abs/2605.10257)).
- **DISPLIB.** Competition ran Oct 2024 – Apr 2025; won by a custom
  branch-and-bound (Stärk and Scholl,
  [ZIB](https://www.zib.de/news/luka-stark-wins-displib-2025-competition)).
  Many instances still lack proven optima.
- **Formulation.** Train dispatching is a job shop with blocking and no-wait
  constraints, solved on the *alternative graph* (Mascis and Pacciarelli 2002).
  Lamorgese and Mannino give an exact decomposition (Oper. Res. 2015) and a
  noncompact MILP stronger than big-M (Oper. Res. 2019,
  [summary](https://academicnewsletter.sufe.edu.cn/info/382914)); see the
  [SINTEF/Gurobi webinar](https://cdn.gurobi.com/wp-content/uploads/SINTEF_Webinar.pdf?x85428).
- **Safe multi-agent RL on Flatland.** Contract-based compositional
  shielding ([arXiv 2606.14130](https://arxiv.org/abs/2606.14130)): a
  shield vetoes unsafe joint actions so exploration cannot cause
  collisions or deadlocks; it treats single-track Flatland as
  deterministic once generated. The suite's `shield` option on the train
  env is the simplest version of this idea.
- **RL for rescheduling.** Yue et al. ([arXiv 2401.06952](https://arxiv.org/abs/2401.06952),
  T-ITS 2024): timetable as a directed graph, GNN state, curriculum over delay
  levels plus local search. Agasucci et al.
  ([arXiv 2009.00433](https://arxiv.org/abs/2009.00433)): decentralised and
  centralised DQN on US freight data. Lövétei et al. (Appl. Sci. 2022,
  [DOAJ](https://doaj.org/article/b19c78faaa9d407a8b57a3d9d477cacb)): how to
  represent rail infrastructure for RL.

## Disruptions

- **Short-turning under blockage.** Ying et al. (TR-B 2024,
  [WTR](https://www.worldtransitresearch.info/research/10584)): multi-agent
  deep RL, one agent per direction, complete blockage on a double-track
  corridor. The closest work to this suite's `blockage` preset.
- **Bus bridging.** Stochastic-demand bus bridging for metro disruptions with
  an RL-guided genetic algorithm ([PMC12503319](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12503319/));
  timetable rescheduling plus response vehicles (Wang et al., TR-E 2024,
  [RePEc](https://ideas.repec.org/a/eee/transe/v188y2024ics1366554524001960.html)).
  None found that does short-turning and bus bridging jointly: that is the
  multimodal gap.

## Bus

- Event-driven multi-agent RL for bus holding
  ([arXiv 1709.06656](https://arxiv.org/abs/1709.06656)): decisions only at
  stop arrivals, the semi-MDP case.
- Cooperative holding and stop-skipping with deep RL
  ([MIT](https://mobility.mit.edu/biblio/rodriguez-cooperative-bus-holding-and-stop-skipping-deep-reinforcement-learning)).
- Classical baseline: headway-based holding as feedback control on an unstable
  headway system (Daganzo 2009).

## The niche

1. One fast JAX core over transport modes, with passengers (not only vehicles
   reaching goals), so the same policy code runs on lifts, buses, trains and
   fleets.
2. Disruptions as a first-class, switchable process in every env, with
   evaluation under distribution shift and tail metrics.
3. Baselines from each policy class (rule, planner/look-ahead, RL), and on
   small presets a hindsight-optimal bound from the alternative-graph MILP.
4. Later, coupled modes (bus bridging for rail blockages), which none of the
   above can express.
