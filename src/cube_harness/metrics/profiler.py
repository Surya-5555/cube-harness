"""Always-on, low-overhead episode profiling.

This module is the **foundation** for Auto-CUBE's profile use-case: turn a
real (or RL-rollout) episode into a phase × resource breakdown cheap enough
to leave on for every task, so the orchestrator can aggregate across a run
and Pareto-rank the bottlenecks that actually matter.

Design constraints (see the proposal in the Auto-CUBE README):

- **Default off.** `ProfileConfig` is opt-in; when an episode has no config
  the sampler never starts. The coarse phase timers (4 `perf_counter` pairs
  per episode) run unconditionally — negligible, never on the inner loop.
- **Transport-agnostic.** The same primitives serve the standard runner and
  the RL rollout path — both build an :class:`~cube_harness.episode.Episode`,
  so wiring lives at the episode level and both inherit it.
- **Cheap tier first.** A background ``psutil`` sampler (~1-2 Hz) plus coarse
  phase timers. Deep profilers (cProfile/tracemalloc/py-spy) and remote
  sandbox / GPU sampling are deliberately *not* here — they are opt-in,
  hotspot-targeted follow-ups that build on this schema.

The unit that lands on disk is one ``profile.json`` per episode output dir,
deserialisable as :class:`EpisodeProfile`. ``ch-profile`` aggregates them.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psutil
from pydantic import BaseModel, Field

try:  # optional: only used when ``ProfileConfig.gpu`` and an NVIDIA GPU is present
    import pynvml
except ImportError:  # pragma: no cover - exercised only on GPU hosts
    pynvml = None

logger = logging.getLogger(__name__)

PROFILE_FILENAME = "profile.json"

# Canonical phase taxonomy. Coarse on purpose — these map 1:1 onto the
# episode lifecycle (see `Episode._run_episode`) and onto the agent-free
# `profile_rollout_phases.py` script, so the live and offline profilers
# agree on names. Finer LLM-wait vs tool-exec splits come from the OTel
# spans the harness already emits and are joined in by the rollup.
PHASE_SETUP = "setup"  # task.make() + task.reset() — provisioning + per-task prepare
PHASE_AGENT_LOOP = "agent_loop"  # agent.run() — generation + tool exec (the dominant, model-bound term)
PHASE_EVALUATE = "evaluate"  # task.evaluate() — re-runs the verifier; reuse-irreducible
PHASE_TEARDOWN = "teardown"  # task.close() — container/sandbox teardown

# Rough attribution of who owns a phase's cost — drives the rollup's
# "is this worth fixing" tag. Coarse heuristic, overridable by the rollup.
PHASE_OWNER = {
    PHASE_SETUP: "infra",
    PHASE_AGENT_LOOP: "model",
    PHASE_EVALUATE: "benchmark",
    PHASE_TEARDOWN: "infra",
}


class ProfileConfig(BaseModel):
    """Opt-in profiling knobs, threaded onto an episode.

    Carried on ``EpisodeConfig`` so it survives Ray serialisation and the
    resume/retry path. ``Experiment.profile`` and the RL ``RolloutConfig``
    both propagate into it. Absent (``None``) ⇒ no profiling, no overhead.
    """

    resource_sampling: bool = Field(
        default=True,
        description="Sample CPU/RSS/IO of the episode worker process tree in a background thread.",
    )
    sample_hz: float = Field(
        default=2.0,
        gt=0,
        le=50,
        description="Resource sampling frequency in Hz. 1-2 Hz is plenty for minute-scale episodes.",
    )
    gpu: bool = Field(
        default=False,
        description="Also sample local GPU (NVML) — only meaningful when a self-hosted inference "
        "server shares this host (RL rollout). No-op if pynvml/NVIDIA is unavailable.",
    )
    phases: bool = Field(
        default=True,
        description="Record coarse per-phase wall-clock (setup/agent_loop/evaluate/teardown).",
    )


class ResourceSummary(BaseModel):
    """Summary statistics for one resource series over an episode."""

    samples: int = 0
    mean: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    max: float = 0.0


class EpisodeProfile(BaseModel):
    """The per-episode profiling artifact written to ``profile.json``.

    Phase durations are seconds; resource summaries are per-series rollups.
    Everything here is cheap to compute and small to store.
    """

    task_id: str = ""
    trajectory_id: str = ""
    wall_time_s: float = 0.0
    phases: dict[str, float] = Field(default_factory=dict, description="Phase name → wall-clock seconds.")
    resources: dict[str, ResourceSummary] = Field(
        default_factory=dict,
        description="Resource series name (cpu_percent, rss_mb, io_read_mb, io_write_mb, "
        "num_fds, num_threads, gpu_util_percent, gpu_mem_mb) → summary.",
    )
    sample_count: int = 0

    def write(self, output_dir: Path) -> Path:
        path = Path(output_dir) / PROFILE_FILENAME
        try:
            path.write_text(self.model_dump_json(indent=2))
        except Exception:
            logger.warning("Failed to write %s", path, exc_info=True)
        return path

    @classmethod
    def load(cls, output_dir: Path) -> EpisodeProfile | None:
        path = Path(output_dir) / PROFILE_FILENAME
        if not path.exists():
            return None
        try:
            return cls.model_validate_json(path.read_text())
        except Exception:
            logger.warning("Failed to read %s", path, exc_info=True)
            return None


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 1]). Empty ⇒ 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def _summarize(values: list[float]) -> ResourceSummary:
    if not values:
        return ResourceSummary()
    return ResourceSummary(
        samples=len(values),
        mean=sum(values) / len(values),
        p50=_percentile(values, 0.50),
        p95=_percentile(values, 0.95),
        max=max(values),
    )


class PhaseAccumulator:
    """Accumulates named phase wall-clock. Re-entrant-safe via nesting guard.

    Phases are sequential in the episode body, so a simple start/stop is
    enough; we still tolerate repeated entries by summing.
    """

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._totals[name] = self._totals.get(name, 0.0) + (time.perf_counter() - start)

    def totals(self) -> dict[str, float]:
        return dict(self._totals)


class ResourceSampler:
    """Background-thread sampler of the current process tree.

    Samples CPU%, RSS, IO bytes, fd/thread counts at ``sample_hz`` using
    ``psutil``; optionally GPU via NVML. Best-effort: any missing backend
    (no psutil, no NVIDIA) degrades to fewer series, never an error. The
    sampler covers whatever process it runs in — the episode worker for the
    standard/RL paths — plus its children (so an in-process browser/container
    client is included; remote sandboxes are a separate, follow-up sampler).
    """

    def __init__(self, config: ProfileConfig) -> None:
        self._config = config
        self._series: dict[str, list[float]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: psutil.Process | None = None
        self._gpu_handles: list = []

        try:
            self._proc = psutil.Process(os.getpid())
            # Prime the cpu_percent counters so the first real sample is non-zero.
            self._proc.cpu_percent(None)
        except Exception:
            self._proc = None
            logger.debug("psutil process handle unavailable — resource sampling disabled", exc_info=True)

        if config.gpu:
            self._init_gpu()

    def _init_gpu(self) -> None:
        if pynvml is None:
            logger.debug("pynvml unavailable — GPU sampling disabled")
            return
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            self._gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
        except Exception:
            self._gpu_handles = []
            logger.debug("NVIDIA NVML init failed — GPU sampling disabled", exc_info=True)

    def _record(self, name: str, value: float) -> None:
        self._series.setdefault(name, []).append(value)

    def _sample_once(self) -> None:
        if self._config.resource_sampling and self._proc is not None:
            try:
                with self._proc.oneshot():
                    self._record("cpu_percent", self._proc.cpu_percent(None))
                    self._record("rss_mb", self._proc.memory_info().rss / 1e6)
                    self._record("num_threads", float(self._proc.num_threads()))
                    try:
                        self._record("num_fds", float(self._proc.num_fds()))
                    except Exception:
                        pass  # not available on all platforms
                    try:
                        io = self._proc.io_counters()
                        self._record("io_read_mb", io.read_bytes / 1e6)
                        self._record("io_write_mb", io.write_bytes / 1e6)
                    except Exception:
                        pass  # macOS denies io_counters without privileges
            except Exception:
                logger.debug("resource sample failed", exc_info=True)

        if self._gpu_handles and pynvml is not None:
            try:
                util = max(pynvml.nvmlDeviceGetUtilizationRates(h).gpu for h in self._gpu_handles)
                mem = max(pynvml.nvmlDeviceGetMemoryInfo(h).used / 1e6 for h in self._gpu_handles)
                self._record("gpu_util_percent", float(util))
                self._record("gpu_mem_mb", float(mem))
            except Exception:
                logger.debug("gpu sample failed", exc_info=True)

    def _loop(self) -> None:
        interval = 1.0 / self._config.sample_hz
        while not self._stop.wait(interval):
            self._sample_once()

    def start(self) -> ResourceSampler:
        """Start the background sampling thread (idempotent, best-effort).

        Starts if either source is active and available — host resource
        sampling (``resource_sampling`` + psutil) OR GPU sampling
        (``gpu`` + NVML). The two knobs are independent.
        """
        if self._thread is not None:
            return self
        host_active = self._config.resource_sampling and self._proc is not None
        if host_active or self._gpu_handles:
            self._thread = threading.Thread(target=self._loop, name="cube-profiler", daemon=True)
            self._thread.start()
        return self

    def __enter__(self) -> ResourceSampler:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def summaries(self) -> dict[str, ResourceSummary]:
        return {name: _summarize(values) for name, values in self._series.items()}

    @property
    def sample_count(self) -> int:
        return max((len(v) for v in self._series.values()), default=0)
