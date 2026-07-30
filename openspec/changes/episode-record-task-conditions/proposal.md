# RFC: `EpisodeRecord.conditions` — persist the experimental condition an episode ran under

**Status:** DRAFT
**Author:** Auto-CUBE (session `timewarp-oracle-sweep-r1`)
**Date:** 2026-07-30
**Origin:** finding 14, `~/auto_cube/timewarp-oracle-sweep-r1/journal/REPORT.md`

---

## Problem

`EpisodeRecord` (`src/cube_harness/eval_log.py`) is the record designed for **cross-run
comparison** — it is what ATLAS ingests and what `M[agent, task] = reward` is built from. Its
field list is closed: `from_view` reads `task_id`, `action_schemas`, `summary_stats` and
`reward_info`, and nothing else.

That is a problem for any benchmark whose point is an **independent variable other than the
task**. TimeWarp is the clean example: it exists to measure agent robustness across historical
UI eras, selected by `ui_version`. Today that value survives in exactly one place —
`episodes/<id>/episode.metadata.json` → `.metadata.ui_version` — and it reaches:

| artifact | carries the applied condition? |
|---|---|
| `episode.metadata.json` | yes (via `reset()` info) |
| `episode_record.json` (`EpisodeRecord`) | **no** |
| `experiment_config.json` | only the *requested* `benchmark_config.ui_version` |
| `experiment_summary.json` | no |

The requested value is precisely the one that may not have been applied: when auto mode reuses
externally-started servers, the cube correctly records `ui_version: None` because the era of a
server it did not launch is unknowable — but `experiment_config.json` still says `1`. So the only
honest record of the condition is the one artifact that cross-run tooling does not read.

Two runs that differ *only* in the experimental condition are therefore indistinguishable in the
record built for comparing runs. This is not TimeWarp-specific: any benchmark with a rendering
mode, a difficulty tier, a locale, a seed sweep, or a tool-variant axis has the same gap.

## Why not just add `ui_version`

Adding one cube's field to a shared record is the "additive isn't free" anti-pattern the
[Design Philosophy](https://the-ai-alliance.github.io/cube-standard/design-philosophy) warns
about, and it does not generalise — the next cube brings `locale`, then `difficulty`, and the
record accretes a union of every cube's private vocabulary.

## Proposed change

One optional, free-form field on `EpisodeRecord`:

```python
conditions: dict[str, Any] = Field(default_factory=dict)
"""The experimental condition this episode actually ran under — the independent variable(s),
as applied rather than as requested. Populated by the task from its reset() info; empty for
benchmarks that vary nothing but the task."""
```

Populated in `EpisodeRecord.from_view` from a reserved key in the trajectory metadata, so no
task-side API changes and no per-cube special-casing in the harness:

```python
conditions=view.metadata.get("conditions", {}),
```

A task opts in by returning `{"conditions": {...}}` from `reset()` — the same channel that
already carries `goal` and `sites`. TimeWarp becomes:

```python
info = {..., "conditions": {"ui_version": self.tw_ui_version}}
```

`None` stays meaningful: `{"ui_version": None}` says "this episode ran at an unknown era",
which is different from `{}` ("this benchmark has no conditions") and from an absent key.

## Alternatives considered

1. **Read `episode.metadata.json` in downstream tooling.** Works today, but pushes per-cube
   knowledge into every consumer and leaves `EpisodeRecord` — the thing we tell people to
   consume — quietly wrong.
2. **Promote `ui_version` to a first-class field.** Rejected: see above.
3. **Reuse `summary_stats`.** It is typed as run *statistics* (tokens, cost, timings); a
   condition is an input, not a measurement. Mixing them makes both harder to consume.

## Scope

- `eval_log.py`: one field, one line in `from_view`.
- `episode/spec.md`, `eval_log/spec.md`: document the reserved `conditions` key in `reset()` info.
- `timewarp_cube.task`: emit `conditions`.
- Backward compatible: absent in old records, `{}` for benchmarks that never set it.

## Open questions

- Should the harness **also** copy `conditions` into `experiment_summary.json`, so a run is
  self-describing without opening an episode? Probably yes when every episode agrees, but that
  needs a rule for the mixed case.
- Should `conditions` be validated as JSON-scalar-valued to keep it indexable downstream?
