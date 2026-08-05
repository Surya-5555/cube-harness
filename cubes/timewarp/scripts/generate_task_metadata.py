#!/usr/bin/env python3
"""Generate src/timewarp_cube/task_metadata.json from the browsergym-timewarp library.

This is a developer tool. Run it when the TimeWarp task list changes to regenerate
the shipped package resource. The output file is committed to the repository — end
users never need to run this script.

Only lightweight public fields are written (sites, intent_template_id, eval_types).
TimeWarp has no heavy execution data — all task logic is available from the
browsergym-timewarp library at runtime via the numeric task id.

Usage:
    python scripts/generate_task_metadata.py [--output PATH] [--force]

Options:
    --output    Destination file (default: task_metadata.json inside the timewarp_cube package).
    --force     Overwrite task_metadata.json even if it already exists.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

import timewarp_cube
from timewarp_cube._data import load_raw_tasks
from timewarp_cube.task import TimeWarpTaskMetadata

logger = logging.getLogger(__name__)

assert timewarp_cube.__file__ is not None
_DEFAULT_OUTPUT = Path(timewarp_cube.__file__).parent / "task_metadata.json"

_RECOMMENDED_MAX_STEPS = 30


def generate_task_metadata(
    output_path: Annotated[
        Path,
        typer.Option(
            "--output", help="Destination file (default: task_metadata.json inside the timewarp_cube package)"
        ),
    ] = _DEFAULT_OUTPUT,
    *,
    force: Annotated[bool, typer.Option(help="Regenerate even if file already exists")] = False,
) -> int:
    """Load tasks from the browsergym-timewarp data file and write task_metadata.json.

    Args:
        output_path: Destination path. Defaults to src/timewarp_cube/task_metadata.json.
        force:       Overwrite even if output_path already exists.

    Returns:
        Number of tasks written (0 if skipped due to idempotency).
    """
    if output_path.exists() and not force:
        logger.info(
            "task_metadata.json already exists at %s — skipping. Pass --force to regenerate.",
            output_path,
        )
        return 0

    logger.info("Loading tasks from browsergym-timewarp data...")
    raw_tasks = load_raw_tasks()
    logger.info("  %d tasks loaded", len(raw_tasks))

    metadata: dict[str, TimeWarpTaskMetadata] = {
        str(t["task_id"]): TimeWarpTaskMetadata(
            id=str(t["task_id"]),
            abstract_description=t.get("intent", ""),
            recommended_max_steps=_RECOMMENDED_MAX_STEPS,
            sites=list(t.get("sites", [])),
            intent_template_id=(None if t.get("intent_template_id") is None else int(t["intent_template_id"])),
            eval_types=list(t.get("eval", {}).get("eval_types", [])),
        )
        for t in raw_tasks
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([tm.model_dump() for tm in metadata.values()], indent=2))
    logger.info("Saved %d tasks to %s", len(metadata), output_path)
    return len(metadata)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    typer.run(generate_task_metadata)
