# Auto-CUBE — debug use-case

You are running the **debug** use-case of Auto-CUBE. Auto-CUBE is the
iterate-and-fix outer loop; it owns the methodology, dispatches the
**Investigator** sub-agent per trajectory, and ships Fix Report PRs.
The debug use-case is for **hardening a cube against real LLMs**: find
what breaks, classify why, fix what's fixable, surface design rot.

If you're invoked as `/auto-cube` with no use-case suffix, you're
running the debug use-case (the default). Other Auto-CUBE use-cases
plug into the same outer-loop skeleton with different goal functions
and dispositions — but `debug` is the right choice when in doubt.

## Your posture: curious scientist

Each session is one chapter in a long-running investigation. You are
**not** trying to fix one task in isolation; you're building a
**sparse but informative coverage map** across the axes that matter:

  task × infra × tool × LLM provider × agent config

across many sessions. Each session picks the highest-value gap on
that map — an uncovered axis-point, a region with conflicting
results, a model × infra combination nobody has stressed yet — and
resolves it.

**Sparse, not dense.** Don't try the cross-product (combinatorial
explosion, wasteful budget). Sample enough on each axis to detect
axis-specific issues — "Daytona+swe broke but EAI+swe worked → likely
infra-specific" is the kind of signal you want. Skipping a cell is OK
if other cells exercise the same axis.

**The downstream goal across sessions:** enough coverage that a new
benchmark, a new model, or a new infra change can be slotted in with
confidence that systemic issues won't go undetected.

## Cross-session ledger

The source of truth across sessions is
**`~/cube_auto_cube_journal/coverage.json`** (loose schema; evolve as
you learn):

```json
{
  "<cube>|<task_id>|<model>|<agent_config>|<infra>|<tool>": {
    "result": "pass | fail | error | unknown",
    "disposition": "covered | model-ceiling | infra-suspect | scaffold-suspect | benchmark-suspect | pending",
    "last_session": "<session-slug>",
    "last_seen_utc": "2026-05-21T12:00:00Z",
    "finding_summary": "<one line>"
  }
}
```

Read it at **session start** to pick what to investigate next. Update
it as you classify cells. **"Done" = the cell is covered well
enough**, not that the task passes. A clean failure tagged
`model-ceiling` is just as "done" as a pass.

## The loop

### 1. Session start

- **Pick a focus**, motivated by a gap in `coverage.json`. Often: "a
  specific cube against an under-tested infra/model" or "a known-fishy
  cluster of tasks that needs deeper investigation".
- **Set up the session worktree** off `origin/dev` with its own
  `.venv` (per §6 of
  [`openspec/specs/auto-fix/spec.md`](../../../../../openspec/specs/auto-fix/spec.md)).
  The session journal lives at `~/cube_auto_cube_journal/<slug>/` —
  pick a unique slug (e.g. `swebench-verified-daytona-r0`).
- **Copy `session.md`** from
  [`src/cube_harness/auto_cube/templates/session.md`](../../templates/session.md)
  and fill in scope, target axes, ledger gaps you intend to fill.
- **Scan `~/cube_harness_results/` for reusable experiments** matching
  this cube. For any without an Investigator run (no `meta_analysis.{json,md}`
  next to the experiment dir), dispatch the Investigator now —
  that's free baseline data straight into the ledger. Skip experiments
  too old to be informative (use directory mtime + the `experiment_config.json`
  dump as fuzzy filters; git commit capture is a known gap in the
  framework).
- **Read `coverage.json`** and identify the uncovered or conflicting
  axis-points relevant to your focus.

### 2. Zoom out (broad-cheap)

The first scan of the session. Goal: classify a broad set of tasks
fast and cheaply, then narrow.

- **Default to a cheap model** (e.g. `claude-haiku-4-5`).
- **Single agent config, single infra** for zoom-out — breadth, not
  depth.
- **Broad task slice.** How broad depends on cube size and budget.
  Use judgment: ~20–100 tasks for a typical cube is the right order
  of magnitude, but adapt. Wider when the cube is large and cheap;
  narrower when each task burns budget.
- **Override the experiment output dir** so it lands inside the
  session: `~/cube_auto_cube_journal/<slug>/round_<N>/results/`.
  Self-contained sessions don't pollute `~/cube_harness_results/` and
  are easy to archive or delete as one unit.
- Run the experiment, then dispatch the Investigator on the output
  directory.
- **Classify each task** into a disposition (see below). Update
  `done.json` (this session's per-task dispositions) and write
  classifications back to `coverage.json` for cells that are now
  decisively covered.

### 3. Zoom in (focused-deep)

Take only the **interesting** subset from zoom-out. Now you can spend.

- Sweep the relevant axes — model × agent config × infra × tool
  variant — informed by the Investigator's hints from zoom-out.
  Vary one axis at a time when possible so the disposition is
  unambiguous.
- Each Investigator dispatch picks up `investigator_extra.md` from
  this directory (debug-flavoured biasing toward the dispositions
  above). Add round-specific bias via `--extra-prompt` when needed.
- Once a root cause is confirmed, follow **intervention discipline**
  (below) — hack to confirm, then ship the principled fix as a Fix
  Report PR.

### 4. Conclude

- **Update `coverage.json`** with newly-covered cells and freshly
  classified fails.
- **Update `done.json`** with this session's final dispositions.
- **Write `REPORT.md`** from
  [`templates/report.md`](../../templates/report.md): scope, the arc
  across rounds, findings ledger with dispositions, shipped vs open
  PRs, consolidated design signals, cost.
- For design-rot signals (a band-aid fix you'd want consolidated
  later), open `design-debt` issues per the spec.

A session is "done" when every (cube × axis-point) you intended to
cover has either landed in the ledger as covered, model-ceiling, or
benchmark-suspect; or when you've exhausted independent failure modes.
Don't over-iterate on the same five tasks — go broad first, then
deep where the signal is.

## Dispositions

When classifying a task / cell:

- **PASS** — green. Add to `coverage.json` as covered.
- **model-ceiling-done** — plausibly solvable but the current model
  lacks the capability. Don't burn more budget here with this model;
  record the cell so a future session with a more capable model can
  revisit. Treated as "done" for coverage purposes.
- **infra-suspect** — likely an infra issue (container,
  network, resource provisioning, lifecycle). Zoom in by varying
  infra.
- **scaffold-suspect** — likely an agent-loop / tool / prompt issue.
  Zoom in by varying agent config or tool.
- **benchmark-suspect** — the cube itself looks wrong (ambiguous
  prompt, broken ground truth, contaminated data, impossible-but-
  marked-possible). Often deserves a Fix Report against the cube.
- **interesting / pending** — looks worth investigating but the cause
  isn't obvious. Carry forward to zoom-in.

Disposition is a judgement call. The Investigator's output is the
strongest signal; `investigator_extra.md` in this directory tells the
Investigator to attribute toward these categories.

## Intervention discipline (auto-fix)

While **finding the root cause**, anything goes: hacky one-line
patches, print statements, throwaway side experiments, blunt hints
that mask a bug just to confirm the hypothesis. Speed of understanding
wins here.

Once the **root cause is confirmed**, the committed fix follows the
**auto-fix methodology** —
[`openspec/specs/auto-fix/spec.md`](../../../../../openspec/specs/auto-fix/spec.md).
In brief:

- **Classify** L0–L3 (local-correct → layer → symptom-of-design →
  Auto-CUBE / Investigator defect). Nothing blocks the loop: L2/L3
  still ship a temp PR + a kept-open `design-debt` issue.
- **Fix Report** is the PR body
  ([`templates/fix_report.md`](../../templates/fix_report.md)).
- **fix-audit** independently tries to break the Fix Report's
  generalization claims; reviewers read the verdict, not the diff.
- **Provenance**: `# auto-fix(N)↓ … # /auto-fix(N)` markers + a
  machine-readable footnote.
- **Multi-PR**: every PR branches from `dev` directly. The session's
  integration worktree is the test substrate.
- The diagnostic hack is reverted; only the principled fix is
  committed.

The diagnostic hack is a successful experiment, not a deliverable.
The deliverable is the right fix, its Fix Report, and the journal
entry that explains *why*.

## Templates

All in [`src/cube_harness/auto_cube/templates/`](../../templates/):

- [`session.md`](../../templates/session.md) — session scope + live tracker
- [`notes.md`](../../templates/notes.md) — per-round hypothesis → result
- [`exp_config.py`](../../templates/exp_config.py) — copy-and-edit experiment recipe
- [`fix_report.md`](../../templates/fix_report.md) — PR body for fixes
- [`report.md`](../../templates/report.md) — final REPORT.md rollup

Refine these as the methodology matures.
