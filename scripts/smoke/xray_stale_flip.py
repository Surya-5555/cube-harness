#!/usr/bin/env python3
"""SMOKE: XRay live dashboard flips a dead experiment's episodes ▶️/🕐 → 👻 (BUG-1 regression).

End-to-end test that XRay's live refresh picks up a *status-only* transition — the bug fixed by
keying change-detection off the episode-directory mtime + re-injecting status onto stubs.

Flow:
  1. Launch a deterministic mock-cube experiment (sequential, no LLM/browser), slow enough to catch
     episodes mid-flight.
  2. Drive the real ``XRayState`` the way the viewer does: ``load_experiment`` then a 1s
     ``refresh_experiment`` poll loop (exactly what the dashboard's ``gr.Timer`` calls). Confirm the
     live view shows in-flight episodes (running/queued) and no stale.
  3. SIGINT the driver mid-run → experiment_status becomes INTERRUPTED, leaving orphaned episodes.
  4. Run the ghost-sweep (``_promote_ghost_episodes`` — the exact function XRay runs on Refresh),
     which writes STALE into the orphaned episodes' status.json *only* (no trajectory write).
  5. Assert the live ``refresh_experiment`` loop flips the view ▶️/🕐 → 👻. Before the BUG-1 fix the
     refresh keyed off trajectory-file mtimes and never noticed a status-only flip; after it, the
     per-dir-mtime detection (started episodes) and stub re-inject (queued episodes) propagate it.

Why ``XRayState`` and not Playwright: the bug lives in ``refresh_experiment`` (the timer's function),
and asserting there is deterministic. Driving the Gradio DataFrame's experiment-selection checkbox
headlessly is flaky and orthogonal to the fix; the rendered DOM has its own coverage in
``tests/test_xray_e2e.py``.

Run:
    .venv/bin/python scripts/smoke/xray_stale_flip.py
    .venv/bin/python scripts/smoke/xray_stale_flip.py --keep   # keep the temp results dir

Final line follows the cube-harness smoke contract:
    SMOKE OK: xray_stale_flip     (exit 0 — live refresh flipped to 👻)
    SMOKE FAIL: xray_stale_flip   (exit 1 — view never flipped, or setup failed)
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.core import Action, Observation
from cube.task import Task, TaskConfig, TaskMetadata
from cube.tool import Tool, ToolConfig, tool_action

from cube_harness.agent import Agent, AgentConfig
from cube_harness.analyze import xray_utils
from cube_harness.analyze.xray import XRayState
from cube_harness.analyze.xray_utils import _promote_ghost_episodes
from cube_harness.core import AgentOutput
from cube_harness.episode_status import STATUS_FILENAME, EpisodeStatus
from cube_harness.exp_runner import run_sequentially
from cube_harness.experiment import Experiment
from cube_harness.experiment_status import EXPERIMENT_STATUS_FILENAME, ExperimentStatus

NAME = "xray_stale_flip"
N_TASKS = int(os.environ.get("XRAY_SMOKE_N_TASKS", "5"))
STEP_SECS = float(os.environ.get("XRAY_SMOKE_STEP_SECS", "2.5"))


# ---------------------------------------------------------------------------
# Deterministic, LLM-free, browser-free mock benchmark + agent
# ---------------------------------------------------------------------------


class _NoopTool(Tool):
    @tool_action
    def click(self, element_id: str) -> str:
        """Click an element.

        Args:
            element_id: The element to click.
        """
        return f"clicked {element_id}"


class _NoopToolConfig(ToolConfig):
    def make(self, container: object = None) -> _NoopTool:
        _ = container
        return _NoopTool()


class _MockTask(Task):
    def reset(self) -> tuple[Observation, dict]:
        return Observation.from_text("smoke task goal"), {}

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict]:
        _ = obs
        return 1.0, {"success": True}


class _MockTaskConfig(TaskConfig):
    def make(self, runtime_context: object = None) -> _MockTask:
        _ = runtime_context
        return _MockTask(metadata=TaskMetadata(id=self.task_id), tool_config=self.tool_config or _NoopToolConfig())


class _MockBenchmark(Benchmark):
    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass


class _SmokeBenchmarkConfig(BenchmarkConfig):
    benchmark_metadata = BenchmarkMetadata(name="xray-smoke", version="0.1.0", description="XRay smoke benchmark")
    task_metadata = {f"smoke_task_{i}": TaskMetadata(id=f"smoke_task_{i}") for i in range(N_TASKS)}
    task_config_class = _MockTaskConfig
    benchmark_class = _MockBenchmark


class _SlowAgentConfig(AgentConfig):
    step_secs: float = STEP_SECS

    def make(self, action_set: object = None, **kwargs: object) -> "Agent":
        _ = action_set, kwargs
        return _SlowAgent(config=self)


class _SlowAgent(Agent):
    name = "XRaySmokeAgent"
    description = "Sleeps then submits final_step — deterministic, no LLM."
    input_content_types = ["text"]
    output_content_types = ["action"]

    def step(self, obs: Observation) -> AgentOutput:
        _ = obs
        time.sleep(self.config.step_secs)  # keep the episode RUNNING long enough to catch it mid-flight
        return AgentOutput(actions=[Action(name="final_step", arguments={})])


def _run_experiment(output_dir: Path) -> None:
    """Child entrypoint: run the mock experiment in-process (sequential, no Ray)."""
    exp = Experiment(
        name="xray_smoke",
        output_dir=output_dir,
        agent_config=_SlowAgentConfig(),
        benchmark_config=_SmokeBenchmarkConfig(),
        max_steps=2,
    )
    run_sequentially(exp)


# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------


def _experiment_dir(results_dir: Path) -> Path | None:
    for child in results_dir.iterdir():
        if child.is_dir() and (child / EXPERIMENT_STATUS_FILENAME).exists():
            return child
    return None


def _disk_statuses(exp_dir: Path) -> dict[str, str]:
    episodes = exp_dir / "episodes"
    if not episodes.exists():
        return {}
    out: dict[str, str] = {}
    for ep_dir in episodes.iterdir():
        if not ep_dir.is_dir() or ".archived_" in ep_dir.name:
            continue
        status = EpisodeStatus.read(ep_dir / STATUS_FILENAME)
        if status is not None:
            out[ep_dir.name] = status.status
    return out


def _view_statuses(state: XRayState) -> list[str]:
    """The display statuses XRay would render for the loaded trajectories."""
    return sorted(xray_utils.trajectory_status(t) for t in state.trajectories)


def _poll(predicate, timeout: float, interval: float = 0.3) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _fail(msg: str) -> int:
    print(f"  ✗ {msg}")
    print(f"SMOKE FAIL: {NAME}")
    return 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="Keep the temp results dir on exit.")
    args = parser.parse_args()

    tmp = tempfile.mkdtemp(prefix="xray_stale_flip_")
    results_dir = Path(tmp)
    exp_proc: subprocess.Popen | None = None
    try:
        # 1. Launch the experiment; wait for an episode to actually be RUNNING on disk.
        print(f"[1] launching mock-cube experiment ({N_TASKS} tasks, {STEP_SECS}s/episode)…")
        exp_proc = subprocess.Popen([sys.executable, __file__, "_run_experiment", str(results_dir / "xray_smoke")])
        if not _poll(
            lambda: (d := _experiment_dir(results_dir)) is not None and "RUNNING" in _disk_statuses(d).values(),
            timeout=30,
        ):
            return _fail("experiment never reached a RUNNING episode")
        exp_dir = _experiment_dir(results_dir)
        assert exp_dir is not None

        # 2. Load it into XRayState and poll refresh_experiment like the dashboard timer does.
        print("[2] loading into XRayState, polling refresh like the 1s dashboard timer…")
        state = XRayState(results_dir=results_dir)
        state.load_experiment(exp_dir)  # synchronous: stats come from the metadata stub, no bulk-loader
        for _ in range(3):
            time.sleep(0.8)
            state.refresh_experiment()
        before = _view_statuses(state)
        if not any(s in ("running", "queued") for s in before):
            return _fail(f"expected in-flight episodes (running/queued) in the view, got: {before}")
        if "stale" in before:
            return _fail(f"view already shows stale before the driver was killed: {before}")
        print(f"      view before kill: {before}")

        # 3. Kill the driver mid-run → INTERRUPTED experiment, orphaned episodes.
        print("[3] SIGINT the driver mid-run…")
        exp_proc.send_signal(signal.SIGINT)
        try:
            exp_proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            exp_proc.kill()
        if not _poll(
            lambda: (
                (s := ExperimentStatus.read(exp_dir / EXPERIMENT_STATUS_FILENAME)) is not None
                and s.status == "INTERRUPTED"
            ),
            timeout=15,
        ):
            return _fail("experiment_status never became INTERRUPTED after SIGINT")

        # 4. Run the ghost-sweep (the function XRay calls on Refresh) → writes STALE to status.json only.
        print("[4] running ghost-sweep (writes STALE into orphaned episodes)…")
        _promote_ghost_episodes(exp_dir)
        disk = _disk_statuses(exp_dir)
        if "STALE" not in disk.values():
            return _fail(f"ghost-sweep did not produce any STALE episode on disk: {disk}")

        # 5. THE REGRESSION: the live refresh loop must flip the view ▶️/🕐 → 👻.
        print("[5] polling refresh_experiment for the live ▶️/🕐 → 👻 flip…")
        flipped = _poll(lambda: (state.refresh_experiment() or True) and "stale" in _view_statuses(state), timeout=15)
        after = _view_statuses(state)
        if not flipped:
            return _fail(
                "live refresh never surfaced 👻 stale — BUG-1 regression "
                f"(status.json is STALE on disk but refresh_experiment missed it).\n"
                f"      disk: {disk}\n      view: {after}"
            )
        print(f"      view after sweep: {after}")
        print("  ✓ live refresh flipped the view to 👻 after a status-only transition")
        print(f"SMOKE OK: {NAME}")
        return 0
    finally:
        if exp_proc is not None and exp_proc.poll() is None:
            exp_proc.kill()
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_run_experiment":
        _run_experiment(Path(sys.argv[2]))
    else:
        sys.exit(main())
