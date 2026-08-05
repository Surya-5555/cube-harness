"""Shared accessors for the shipped task data, and the guard that they agree with each other.

Two datasets meet here: the raw browsergym-timewarp task configs (``data/test.raw.json``, from
the installed upstream package) and this cube's own ``task_metadata.json`` (committed, generated
from a specific upstream release). Both the debug suite, the metadata generator, the benchmark
and the task layer read one or the other, so the path, the parse and the consistency check live
in one place — and, being below both ``benchmark`` and ``task``, this module is importable from
either without a cycle.
"""

from __future__ import annotations

import functools
import importlib.resources
import json


@functools.cache
def load_raw_tasks() -> list[dict]:
    """Load and parse the browsergym-timewarp raw task configs. Treat the result as read-only."""
    raw = importlib.resources.files("browsergym.timewarp").joinpath("data/test.raw.json").read_text()
    return json.loads(raw)


@functools.cache
def load_task_metadata() -> list[dict]:
    """Load this cube's shipped task metadata. Treat the result as read-only."""
    raw = importlib.resources.files("timewarp_cube").joinpath("task_metadata.json").read_text()
    return json.loads(raw)


@functools.cache
def verify_upstream_data() -> None:
    """Fail fast when the installed browsergym-timewarp scores tasks differently from the release
    this cube was built against.

    Called from ``TimeWarpBenchmark._setup`` on the driver *and* from ``TimeWarpTaskConfig.make``
    in every Ray worker — workers build tasks from a pickled ``TaskConfig`` and never run
    ``_setup``, so a driver-only check would leave the process that actually scores unguarded.
    Cached, so the repeat calls cost nothing.

    Scope, stated precisely: this compares ``eval_types`` per task and nothing else. It catches a
    release that moves tasks between scoring *mechanisms* — the failure below — but not one that
    keeps the mechanisms and edits the answer specs (``reference_answers``, guarded upstream by
    ``eval.revision``). It is a tripwire for the catastrophic case, not a conformance check.

    Why a data check and not a version check. The failure this guards against is silent and
    total: browsergym-timewarp 0.1.0 scores *every* task with an LLM judge, while 0.2.0 scores
    229 of 231 deterministically. Run the cube against 0.1.0 without a judge key and 0.1.0
    swallows the resulting error into ``score = 0.0`` — so a whole sweep comes back COMPLETED,
    ``error_type: null``, reward 0.0. A uniform, clean, entirely fictitious 0%, indistinguishable
    from an agent that is simply bad at TimeWarp.

    A version comparison cannot catch it. ``importlib.metadata.requires("timewarp-cube")`` reads
    the *installed* dist-info, which goes stale the moment pyproject.toml's pin is bumped without
    a reinstall — exactly the state that produces this bug in the first place. The shipped
    task_metadata.json, by contrast, is committed alongside the source and regenerated
    deliberately, so it is the honest record of what this code expects.
    """
    # frozenset, not tuple: EvaluatorComb multiplies its evaluators, so eval_types order carries
    # no meaning and a pure reordering must not block a run.
    expected = {str(t["id"]): frozenset(t["eval_types"]) for t in load_task_metadata()}
    try:
        actual = {str(t["task_id"]): frozenset(t["eval"]["eval_types"]) for t in load_raw_tasks()}
    except (KeyError, TypeError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"Cannot read scoring types from the installed browsergym-timewarp ({exc!r}) — its task "
            f"data does not have the shape this cube expects. Reinstall the pinned version:\n"
            f"  uv pip install --active -e cubes/timewarp"
        ) from exc
    drifted = sorted(tid for tid, types in expected.items() if actual.get(tid) != types)
    if not drifted:
        return
    sample = ", ".join(
        f"{tid}: expected {sorted(expected[tid])}, installed {sorted(actual.get(tid) or [])}" for tid in drifted[:3]
    )
    raise RuntimeError(
        f"Installed browsergym-timewarp does not match the release this cube was built against: "
        f"{len(drifted)} of {len(expected)} tasks disagree on how they are scored ({sample}). "
        f"Running anyway would produce silently wrong rewards. Reinstall the pinned version:\n"
        f"  uv pip install --active -e cubes/timewarp"
    )
