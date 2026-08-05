"""Generate bfcl-cube's committed data artifacts from the BFCL v4 dataset.

Author-time only (not shipped in the wheel). Reads the BFCL data files and
emits two artifacts next to the package:

- ``src/bfcl_cube/task_metadata.json`` — lightweight per-task metadata
  (auto-loaded by the framework at import time).
- ``src/bfcl_cube/data/bfcl_single_turn.jsonl.gz`` — heavy per-task data
  (question, function schemas, ground truth), unpacked into the per-task cache
  by ``BfclBenchmarkConfig.install()``.

Scope: the **Python single-turn** categories (NON_LIVE + LIVE). Java/JavaScript
(need tree-sitter) and multi_turn/agentic (separate effort) are excluded.

Source data: pass ``--source-dir`` pointing at a directory containing the
``BFCL_v4_*.json`` files and a ``possible_answer/`` subdirectory. With no
argument the script locates an installed ``bfcl_eval`` package
(``pip install bfcl-eval``) and uses its bundled data. BFCL is Apache-2.0; the
emitted data is redistributed under that license (see the cube README).
"""

from __future__ import annotations

import gzip
import json
from importlib.util import find_spec
from pathlib import Path

import typer

# category -> BFCL "collection" (drives the cube's named subsets)
NON_LIVE = ["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance"]
LIVE = [
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
]
COLLECTIONS = {c: "non_live" for c in NON_LIVE} | {c: "live" for c in LIVE}
CATEGORIES = NON_LIVE + LIVE

_PKG_ROOT = Path(__file__).resolve().parents[1] / "src" / "bfcl_cube"


def _default_source_dir() -> Path:
    spec = find_spec("bfcl_eval")
    if spec is None or not spec.origin:
        raise typer.BadParameter(
            "bfcl_eval is not installed and --source-dir was not given. "
            "Run `pip install bfcl-eval` or pass --source-dir <dir with BFCL_v4_*.json>."
        )
    return Path(spec.origin).parent / "data"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _abstract(question: list[list[dict[str, str]]], limit: int = 200) -> str:
    """First user message, truncated — used as the task's abstract_description."""
    for turn in question:
        for msg in turn:
            if msg.get("role") == "user":
                text = msg.get("content", "").strip().replace("\n", " ")
                return text[:limit]
    return "BFCL function-calling task"


def main(
    source_dir: Path = typer.Option(  # noqa: B008
        None, help="Dir with BFCL_v4_*.json + possible_answer/. Defaults to installed bfcl_eval."
    ),
) -> None:
    src = source_dir or _default_source_dir()
    pa_dir = src / "possible_answer"
    typer.echo(f"Reading BFCL data from {src}")

    metadata: list[dict] = []
    heavy_lines: list[str] = []
    seen_ids: set[str] = set()

    for category in CATEGORIES:
        data_file = src / f"BFCL_v4_{category}.json"
        if not data_file.exists():
            raise typer.BadParameter(f"Missing data file: {data_file}")
        rows = _read_jsonl(data_file)

        pa_file = pa_dir / f"BFCL_v4_{category}.json"
        ground_truth_by_id: dict[str, list] = {}
        if pa_file.exists():
            ground_truth_by_id = {r["id"]: r["ground_truth"] for r in _read_jsonl(pa_file)}

        for row in rows:
            tid = row["id"]
            if tid in seen_ids:
                raise ValueError(f"Duplicate task id across categories: {tid}")
            seen_ids.add(tid)

            # One step per gold call (the agent may emit them across steps) + one
            # for final_step; abstention categories (no ground truth) need just 1.
            n_calls = len(ground_truth_by_id.get(tid) or [])
            metadata.append(
                {
                    # Polymorphic discriminator so the framework's metadata loader
                    # (``TaskMetadata.model_validate``) dispatches to our subclass.
                    "_type": "bfcl_cube.task.BfclTaskMetadata",
                    "id": tid,
                    "abstract_description": _abstract(row["question"]),
                    "recommended_max_steps": max(n_calls, 1) + 1,
                    "category": category,
                    "language": "python",
                    "collection": COLLECTIONS[category],
                }
            )
            heavy_lines.append(
                json.dumps(
                    {
                        "id": tid,
                        "question": row["question"],
                        "functions": row["function"],
                        "ground_truth": ground_truth_by_id.get(tid),
                    }
                )
            )
        typer.echo(f"  {category}: {len(rows)} tasks")

    meta_path = _PKG_ROOT / "task_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    typer.echo(f"Wrote {len(metadata)} task metadata entries -> {meta_path}")

    data_path = _PKG_ROOT / "data" / "bfcl_single_turn.jsonl.gz"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(data_path, "wt", encoding="utf-8") as f:
        f.write("\n".join(heavy_lines))
    typer.echo(f"Wrote heavy data ({data_path.stat().st_size // 1024} KiB) -> {data_path}")


if __name__ == "__main__":
    typer.run(main)
