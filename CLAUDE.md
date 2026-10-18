# Notes for Claude

The plan is [docs/plan.md](docs/plan.md): work through it in order, tick items
off and add a **Result** under each phase as it finishes.

This repo follows on from `~/Repositories/elevator-dispatching-rl`; code is
ported from there (see the plan for which commits and files).

## Working preferences

- Commit as each step finishes; open one PR at the end of the session. Branch
  off `main` first.
- No long training runs without asking; short benchmark and parity runs are fine.
- Document findings in the plan as you go: what was built, numbers, what was
  learned, and backlog items for anything deferred.
- When asked for a recommendation, give one with its reasoning. Check the
  risks of a big new dependency before adopting it.
- This is a learning exercise: explain trade-offs and implementation details.
- New applications: distil the hard core into a mathematical model first.
  If it matches an existing env, record it as an instance of the closest
  one in docs/applications.md rather than building a duplicate model.

## Machine

Ryzen 7 3700X (8 cores / 16 threads), 31 GB RAM, GTX 1080 (Pascal, sm_61,
8 GB), Ubuntu 24.04.

- CUDA 13 builds don't support Pascal: use CUDA 12 builds (JAX `cuda12`,
  torch cu126).
- Use uv-managed Python 3.12.
- If JAX falls back to the CPU in an old shell, run with `env -u LD_LIBRARY_PATH`.
  The user manages their dotfiles: don't remind them about them.

## Pitfalls

- Never `pkill -f <pattern>` in a command containing the pattern: it kills
  the shell. Kill by PID.
- Each JAX process preallocates 75% of the GPU, so a second concurrent GPU run
  fails ("BlasLt is unavailable"). Serialise GPU jobs, or set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- Background jobs import from the working tree: don't switch branches under a
  running job; use `git worktree add`.
