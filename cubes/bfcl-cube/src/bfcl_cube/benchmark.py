"""Benchmark layer for bfcl-cube — Berkeley Function Calling Leaderboard (single-turn).

No infrastructure: tasks are scored in-process by AST-matching recorded calls.
Heavy per-task data ships gzipped in the package and is unpacked into the
per-task execution cache by :meth:`BfclBenchmarkConfig.install`.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Iterator
from importlib.resources import files
from typing import Any, ClassVar

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.task import TaskConfig

from bfcl_cube.task import INSTALL_SENTINEL, BfclTaskConfig, BfclTaskMetadata

logger = logging.getLogger(__name__)

_DATA_RESOURCE = "data/bfcl_single_turn.jsonl.gz"


def iter_data_records() -> Iterator[dict[str, Any]]:
    """Yield each per-task record (id, question, functions, ground_truth) from the
    bundled gzipped data. Shared by ``install()`` and the debug module."""
    raw = (files("bfcl_cube") / _DATA_RESOURCE).read_bytes()
    for line in gzip.decompress(raw).decode("utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


class BfclBenchmark(Benchmark["BfclBenchmarkConfig"]):
    """Runtime pair — BFCL single-turn needs no shared infrastructure."""

    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass


class BfclBenchmarkConfig(BenchmarkConfig[BfclTaskMetadata]):
    """Berkeley Function Calling Leaderboard — Python single-turn AST categories."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="bfcl-cube",
        version="0.1.0",
        description=(
            "Berkeley Function Calling Leaderboard (v4) — Python single-turn function "
            "calling, AST-scored. NON_LIVE + LIVE categories."
        ),
        num_tasks=3491,
        tags=["tool-use", "function-calling", "bfcl"],
        named_subsets={
            "non_live": ("collection", "non_live"),
            "live": ("collection", "live"),
        },
    )
    task_config_class: ClassVar[type[TaskConfig]] = BfclTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = BfclBenchmark

    @classmethod
    def install(cls) -> None:
        """Unpack the bundled gzipped data into the per-task execution cache.

        Idempotent: skips when the completion sentinel is present. Re-runs (e.g.
        after a crash mid-write) safely overwrite per-task files and only then
        write the sentinel. Reads only the package's own committed data — no network.
        """
        exec_cache_dir = cls.task_config_class.task_execution_cache_dir()
        sentinel = exec_cache_dir / INSTALL_SENTINEL
        if sentinel.exists():
            logger.info("bfcl-cube execution cache already populated, skipping installation")
            return
        exec_cache_dir.mkdir(parents=True, exist_ok=True)

        n = 0
        for record in iter_data_records():
            (exec_cache_dir / f"{record['id']}.json").write_text(json.dumps(record))
            n += 1
        sentinel.write_text(str(n))
        logger.info("bfcl-cube installed %d per-task execution files -> %s", n, exec_cache_dir)
