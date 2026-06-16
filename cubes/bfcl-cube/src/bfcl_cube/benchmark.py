"""Benchmark layer for bfcl-cube — Berkeley Function Calling Leaderboard (single-turn).

No infrastructure: tasks are scored in-process by AST-matching recorded calls.
Heavy per-task data ships gzipped in the package and is unpacked into the
per-task execution cache by :meth:`BfclBenchmarkConfig.install`.
"""

from __future__ import annotations

import gzip
import json
import logging
from importlib.resources import files
from typing import ClassVar

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.task import TaskConfig

from bfcl_cube.task import BfclTaskConfig, BfclTaskMetadata

logger = logging.getLogger(__name__)

_DATA_RESOURCE = "data/bfcl_single_turn.jsonl.gz"


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

        Idempotent: skips when the cache directory already exists and is
        non-empty. Reads only the package's own committed data — no network.
        """
        exec_cache_dir = cls.task_config_class.task_execution_cache_dir()
        if exec_cache_dir.exists() and any(exec_cache_dir.iterdir()):
            logger.info("bfcl-cube execution cache already populated, skipping installation")
            return
        exec_cache_dir.mkdir(parents=True, exist_ok=True)

        raw = (files("bfcl_cube") / _DATA_RESOURCE).read_bytes()
        n = 0
        for line in gzip.decompress(raw).decode("utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            (exec_cache_dir / f"{record['id']}.json").write_text(json.dumps(record))
            n += 1
        logger.info("bfcl-cube installed %d per-task execution files -> %s", n, exec_cache_dir)
