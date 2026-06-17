# Auto-CUBE — profile use-case

You are running the **profile** use-case of Auto-CUBE. Auto-CUBE is the
iterate-and-fix outer loop; the profile use-case turns it toward
**efficiency**: find where a cube's episodes actually spend wall-clock and
resources, attack only the bottlenecks that matter, and **verify each
optimization with an A/B re-run** before shipping it.

Unlike `debug` / `hinter`, this use-case does **not** dispatch a
per-trajectory Investigator. Profiling produces *aggregate* statistics, so
the analysis step is a single rollup (`ch-profile`) over the run, not an
LLM call per episode. (You *may* still dispatch the Investigator
`profiling` use-case on a few outlier episodes when an aggregate bar is
suspicious and you need scaffold-level detail — but that's the exception,
not the loop.)

## Your posture: performance engineer

Profiling without a plan burns budget. Hold to three rules:

1. **Measure before you cut.** Never optimize a phase you haven't seen
   dominate the Pareto. The recurring lesson on this branch: the obvious
   target (container fork) was a non-bottleneck; the real costs were
   teardown grace and remote per-tool-call transport. Let the data pick.
2. **Only tackle what's *yours* and *matters*.** Each bottleneck has an
   owner — `infra` / `model` / `benchmark`. `model`-bound time
   (generation) is not yours to fix here; `benchmark`-bound time
   (`evaluate` re-running a suite) is a reward-cache question, not a
   harness one. Spend on the top **actionable, infra/harness-owned** bar.
3. **Verify the fix.** An optimization is a hypothesis until an A/B
   re-run of the *same slice* shows the targeted phase's share dropped
   and total wall-clock fell without a regression elsewhere.

## What you collect

Turn profiling on by setting `profile=ProfileConfig(...)` on the
`Experiment` (or `RolloutConfig` for RL rollouts). Every episode then
writes `episodes/<traj>/profile.json`
([`cube_harness.metrics.profiler.EpisodeProfile`](../../../metrics/profiler.py)):

- **phases** — coarse per-episode wall-clock: `setup` (provision + reset),
  `agent_loop` (generation + tools), `evaluate` (verifier), `teardown`.
- **resources** — host CPU% / RSS / IO / fds (+ GPU via NVML when
  `gpu=True` and a self-hosted inference server shares the box).

Keep it cheap and **leave it on** for the whole zoom-out — the sampler is
~1–2 Hz and default-off elsewhere, so it costs almost nothing.

> Coverage of the resource picture today: the sampler measures the
> **episode-worker process** (harness side). Remote sandbox-interior
> CPU/RAM/disk and the inference-server GPU service are **follow-ups** —
> if the bottleneck looks like it lives inside the container or on the
> GPU box, say so and flag it rather than guessing from host-side numbers.

## The loop

### 1. Session start
- Pick a focus: a `<cube> × <infra>` whose throughput you want to
  understand or improve. Set up the session worktree + `.venv` and export
  `CH_EXP_DIR` (§ as in the [debug use-case](../debug/SKILL.md) / README).
- Read `~/auto_cube/profile.json` (the cross-session bottleneck ledger,
  loose schema) to see what's already characterized.

### 2. Zoom out (broad, cheap, profiling on)
- Run a broad task slice on a **cheap model** (you're profiling the
  harness/infra, not model quality), single infra/config, with
  `profile=ProfileConfig()`. Template:
  [`templates/exp_config.py`](templates/exp_config.py).
- Aggregate with **`ch-profile <exp_dir>`** → a phase Pareto table (mean s
  / share of wall-clock / owner) plus resource peaks. This *is* the
  analysis step. No per-trajectory dispatch.

### 3. Pick the bottleneck → confirm → optimize
- Choose the top **actionable** bar (skip `model`-bound; weigh
  `benchmark`-bound separately). Resource peaks catch what the phase view
  won't — a RAM/fd leak, an IO-bound teardown.
- Confirm the *why* with deep, targeted profiling on that one hotspot:
  `py-spy` / `pyinstrument` (sampling, attach to a worker) for "which
  function", `cProfile` / `tracemalloc` for exact counts / allocation.
  Anything goes while confirming (the [intervention discipline](../debug/SKILL.md)
  applies — hack to confirm, then ship the principled fix).

### 4. A/B verify → ship
- Re-run the **same task slice** with the fix, `ch-profile` both, and
  diff: did the targeted phase's share/seconds drop, did total wall-clock
  fall, did nothing else regress? Put the before/after table in the Fix
  Report — that *is* the evidence reviewers read.
- Ship as a Fix Report PR per
  [`openspec/specs/auto-fix/spec.md`](../../../../../openspec/specs/auto-fix/spec.md).
  Update `~/auto_cube/profile.json` for the bar you moved.

## Dispositions — from phases to action

Map each Pareto bar onto one disposition (owner tag does most of the work):

| Disposition | Trigger | What to do |
|---|---|---|
| **actionable** | `infra`/harness-owned phase or a resource peak, large share | Confirm + optimize + A/B verify + Fix Report. |
| **model-bound** | `agent_loop` dominates, `owner=model` | Not this use-case's job — record and move on. |
| **benchmark-bound** | `evaluate` dominates, `owner=benchmark` | Reward-caching / verifier question; note it, usually a separate RFC. |
| **infra-bound (remote)** | bottleneck looks sandbox-interior or GPU-side | Beyond host-side sampling today — flag for the follow-up samplers. |
| **noise** | below ~5% share, no resource peak | Leave it. Don't micro-optimize the tail. |

A bar is "done" when it's been driven below noise, or classified
model/benchmark/remote-bound, or shipped a verified Fix Report.

## Ledger

`~/auto_cube/profile.json` (cross-session, loose schema; evolve as you
learn):

```json
{
  "<cube>|<infra>|<phase|resource>": {
    "share": 0.0,
    "p50_s": 0.0,
    "p95_s": 0.0,
    "owner": "infra | model | benchmark",
    "disposition": "actionable | model-bound | benchmark-bound | infra-bound | noise | shipped-fix",
    "delta_after_fix": null,
    "last_session": "<session-slug>",
    "last_seen_utc": "2026-06-16T00:00:00Z",
    "note": "<one line>"
  }
}
```

## Tooling

- `Experiment(profile=ProfileConfig(...))` / `RolloutConfig(profile=...)` — turn it on.
- `ch-profile <exp_dir> [--json out.json]` — the aggregate rollup.
- [`scripts/smoke/profile_end_to_end.py`](../../../../../scripts/smoke/profile_end_to_end.py)
  — end-to-end smoke (arithmetic-cube, no infra/LLM) proving the pipeline.
- `py-spy` / `pyinstrument` / `cProfile` / `tracemalloc` — opt-in deep
  confirmation on a single hotspot.

## Templates

- [`templates/exp_config.py`](templates/exp_config.py) — copy-and-edit
  profiling run. Keep `is_official=False` — Auto-CUBE runs are iteration.
- Reuse the [debug use-case](../debug/SKILL.md) session/report/notes
  templates for the journal; the methodology skeleton is shared.
