# Deltas — `EpisodeRecord.conditions`

Changes against current specs in `openspec/specs/`.

---

## `eval_log/spec.md`

### ADDED — `EpisodeRecord.conditions`

```python
conditions: dict[str, Any] = Field(default_factory=dict)
```

The experimental condition the episode actually ran under — the benchmark's independent
variable(s) **as applied**, not as requested. Empty for benchmarks that vary nothing but the task.

Populated in `EpisodeRecord.from_view` from the reserved `conditions` key in trajectory metadata:

```python
conditions=view.metadata.get("conditions", {}),
```

Invariants:

- **As-applied, not as-requested.** A config value that did not take effect must not appear here.
  Where the applied value is unknowable, the key is present with value `None`.
- `{}` (no conditions declared) and `{"k": None}` (declared, unknown) are distinct and both valid.
- Backward compatible: records written before this change deserialize with `conditions={}`.

---

## `episode/spec.md`

### MODIFIED — reserved keys in `Task.reset()` info

`reset()`'s info dict gains one reserved key, `conditions`, alongside the existing harness-owned
keys. `Episode` spreads reset info into `TrajectoryMetadata.metadata` as it does today; no new
plumbing.

```python
def reset(self) -> tuple[Observation, dict[str, Any]]:
    ...
    return obs, {..., "conditions": {"ui_version": self.tw_ui_version}}
```

Tasks that vary only by task id omit the key.

---

## Non-normative: consumer guidance

`experiment_config.json` records the **requested** benchmark config and remains the right place
to read intent. `EpisodeRecord.conditions` records what an episode **actually ran under**, and is
the field cross-run comparison should key on. When they disagree, the record is authoritative and
the disagreement is itself the finding.
