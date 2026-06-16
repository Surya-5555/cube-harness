"""Deterministic debug agent for bfcl-cube (no LLM).

Replays the *gold* function call(s) for a representative task in each category,
then ``final_step``. Gold calls are derived from BFCL ground truth (first
concrete acceptable value per parameter), so every debug task reaches
``reward == 1.0``:

- AST categories  → the ground-truth call(s), then final_step.
- (live_)irrelevance → final_step only (correct = abstain).
- live_relevance   → one call to any offered function, then final_step.

Public API: ``get_debug_benchmark()``, ``make_debug_agent(task_id)``.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from importlib.resources import files
from typing import Any

from cube.core import Action, ActionSchema, Observation

from bfcl_cube._vendor.schema_convert import normalize_function_name
from bfcl_cube.benchmark import BfclBenchmarkConfig, iter_data_records
from bfcl_cube.task import _IRRELEVANCE_CATEGORIES, _RELEVANCE_CATEGORIES

logger = logging.getLogger(__name__)

_FINAL_STEP = Action(name="final_step", arguments={})


def _load_records() -> dict[str, dict]:
    return {r["id"]: r for r in iter_data_records()}


def _load_categories() -> dict[str, str]:
    meta = json.loads((files("bfcl_cube") / "task_metadata.json").read_text())
    return {m["id"]: m["category"] for m in meta}


_MISSING = object()


def _concretize(acceptable: list) -> Any:
    """Reduce a BFCL acceptable-value list to one concrete value.

    Picks the first non-empty-string entry and concretizes it; returns
    ``_MISSING`` when every entry is '' (an optional parameter to omit).
    """
    for v in acceptable:
        if v != "":
            return _concretize_value(v)
    return _MISSING


def _concretize_value(v: Any) -> Any:
    """Concretize one acceptable value, recursing into nested ground-truth shapes.

    BFCL ground truth nests acceptable-lists: a dict parameter is
    ``{subkey: [acceptable, ...]}`` and a list-of-dict parameter is
    ``[{subkey: [acceptable, ...]}, ...]``. Plain scalar/list values are concrete
    and returned as-is.
    """
    if isinstance(v, dict):
        out = {}
        for k, sub in v.items():
            cv = _concretize(sub) if isinstance(sub, list) else sub
            if cv is not _MISSING:
                out[k] = cv
        return out
    if isinstance(v, list) and v and all(isinstance(e, dict) for e in v):
        return [_concretize_value(e) for e in v]
    return v


def _gold_actions(category: str, record: dict) -> list[Action]:
    """Build the gold call sequence (excluding the trailing final_step)."""
    if category in _IRRELEVANCE_CATEGORIES:
        return []
    if category in _RELEVANCE_CATEGORIES:
        name = normalize_function_name(record["functions"][0]["name"])
        return [Action(name=name, arguments={})]

    actions: list[Action] = []
    for entry in record["ground_truth"]:
        for fname, params in entry.items():
            args = {}
            for param, acceptable in params.items():
                value = _concretize(acceptable)
                if value is not _MISSING:
                    args[param] = value
            actions.append(Action(name=normalize_function_name(fname), arguments=args))
    return actions


@lru_cache(maxsize=1)
def debug_task_actions() -> dict[str, list[Action]]:
    """One representative task per category → its gold action sequence + final_step.

    Cached and computed lazily so a plain ``import bfcl_cube`` does not pay the
    cost of reading/decompressing the bundled data (only debug paths need it).
    """
    records = _load_records()
    categories = _load_categories()
    by_category: dict[str, str] = {}
    for tid, cat in categories.items():
        by_category.setdefault(cat, tid)  # first task id seen per category

    plan: dict[str, list[Action]] = {}
    for cat, tid in by_category.items():
        plan[tid] = _gold_actions(cat, records[tid]) + [_FINAL_STEP]
    return plan


class DebugAgent:
    """Replays a fixed action sequence for one task."""

    def __init__(self, task_id: str) -> None:
        actions = debug_task_actions()
        if task_id not in actions:
            raise ValueError(f"No debug actions for {task_id!r}. Known: {list(actions)}")
        self._task_id = task_id
        self._step = 0
        self._actions = list(actions[task_id])

    def get_action(self, obs: Observation) -> Action:
        if self._step >= len(self._actions):
            raise StopIteration(f"All actions exhausted for task {self._task_id!r}")
        action = self._actions[self._step]
        self._step += 1
        return action

    def __call__(self, obs: Observation, action_set: list[ActionSchema]) -> Action:
        return self.get_action(obs)


def get_debug_benchmark() -> BfclBenchmarkConfig:
    return BfclBenchmarkConfig().subset_from_list(list(debug_task_actions().keys()))


def make_debug_agent(task_id: str) -> DebugAgent:
    return DebugAgent(task_id)


if __name__ == "__main__":
    import sys

    import bfcl_cube.debug as _mod
    from cube.testing import run_debug_suite

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
    results = run_debug_suite("bfcl-cube", _mod)
    failed = [r for r in results if r["error"] or not r["done"] or r["reward"] < 1.0]
    sys.exit(1 if failed else 0)
