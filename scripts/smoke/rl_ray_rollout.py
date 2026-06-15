# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cube-harness[rl]",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "../..", editable = true }
# ///
"""SMOKE: Ray-backed RL rollout execution.

This smoke keeps real Ray startup, scheduling, event publisher actor wiring, and
cancellation coverage out of the default pytest suite. It is intended for manual
or pre-merge verification on machines that can afford a small Ray cluster.

Example:

    uv run scripts/smoke/rl_ray_rollout.py

Prints SMOKE OK|FAIL: rl_ray_rollout  (exit 0|1).
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from typing import ClassVar

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.core import Action, EnvironmentOutput, Observation
from cube.task import Task, TaskConfig, TaskMetadata
from cube.tool import Tool, ToolConfig

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentOutput
from cube_harness.rl import CancelRequest, RayConfig, RolloutConfig, RolloutEngine, RolloutRequest
from cube_harness.rl.llm import RolloutLLMConfig

NAME = "rl_ray_rollout"


class SmokeTool(Tool):
    # Inherits the base `final_step` (@tool_action that raises AgentStop) — that's how the
    # agent terminates the rollout. Overriding it to *return* would never terminate (the
    # rollout would run to max_steps); final_step IS the termination action now.
    pass


class SmokeToolConfig(ToolConfig):
    def make(self, container=None) -> SmokeTool:
        _ = container
        return SmokeTool()


class SmokeAgentConfig(AgentConfig):
    name: str = "smoke_agent"

    def make(self, action_set=None, **kwargs) -> "SmokeAgent":
        _ = action_set, kwargs
        return SmokeAgent(config=self)


class SmokeAgent(Agent):
    name = "SmokeAgent"
    description = "Deterministic agent for Ray rollout smoke coverage."
    input_content_types = ["text"]
    output_content_types = ["action"]

    def step(self, obs: Observation) -> AgentOutput:
        _ = obs
        return AgentOutput(actions=[Action(name="final_step", arguments={})])


class SmokeTask(Task):
    def reset(self) -> tuple[Observation, dict]:
        return Observation.from_text("finish the smoke task"), {}

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict]:
        _ = obs
        return 1.0, {"success": True}


class SlowSmokeTask(SmokeTask):
    def step(self, action: Action | list[Action]) -> EnvironmentOutput:
        _ = action
        time.sleep(5.0)
        return EnvironmentOutput(obs=Observation.from_text("done"), reward=1.0, done=True, info={"success": True})


class SmokeTaskConfig(TaskConfig):
    def make(self, runtime_context=None) -> SmokeTask:
        _ = runtime_context
        return SmokeTask(metadata=TaskMetadata(id=self.task_id), tool_config=self.tool_config or SmokeToolConfig())


class SlowSmokeTaskConfig(TaskConfig):
    def make(self, runtime_context=None) -> SlowSmokeTask:
        _ = runtime_context
        return SlowSmokeTask(metadata=TaskMetadata(id=self.task_id), tool_config=self.tool_config or SmokeToolConfig())


class SmokeBenchmark(Benchmark):
    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass


class SmokeBenchmarkConfig(BenchmarkConfig):
    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="ray-smoke-cube",
        version="0.1.0",
        description="Deterministic benchmark for Ray rollout smoke coverage.",
    )
    task_metadata: ClassVar[dict[str, TaskMetadata]] = {
        "ray_smoke_task": TaskMetadata(id="ray_smoke_task"),
    }
    task_config_class: ClassVar[type[TaskConfig]] = SmokeTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = SmokeBenchmark

    @classmethod
    def install(cls) -> None:
        pass


class SlowSmokeBenchmarkConfig(SmokeBenchmarkConfig):
    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="slow-ray-smoke-cube",
        version="0.1.0",
        description="Slow deterministic benchmark for Ray cancellation smoke coverage.",
    )
    task_metadata: ClassVar[dict[str, TaskMetadata]] = {
        "slow_ray_smoke_task": TaskMetadata(id="slow_ray_smoke_task"),
    }
    task_config_class: ClassVar[type[TaskConfig]] = SlowSmokeTaskConfig


class SmokeError(Exception):
    pass


def _llm_config() -> RolloutLLMConfig:
    return RolloutLLMConfig(
        model_name="served-model",
        api_base="http://localhost:8000/v1",
        api_key="EMPTY",
        tokenizer_name="mock-tokenizer",
    )


def _config(output_dir: Path, benchmark_config: BenchmarkConfig, *, name: str) -> RolloutConfig:
    return RolloutConfig(
        name=name,
        output_dir=output_dir,
        benchmark_config=benchmark_config,
        agent_config=SmokeAgentConfig(),
        max_steps=2,
        execution_mode="ray",
        ray=RayConfig(num_workers=1),
    )


async def _events_until_terminal(rollout: RolloutEngine, request_id: str, *, timeout_s: float) -> list[dict]:
    events: list[dict] = []
    async for event in rollout.events(
        from_offset=0, stop_request_id=request_id, timeout_s=timeout_s, poll_timeout_s=0.1
    ):
        events.append(event)
    return events


async def _check_completed_rollout(root: Path) -> None:
    rollout = RolloutEngine(config=_config(root / "complete", SmokeBenchmarkConfig(), name="ray_complete_smoke"))
    try:
        health = rollout.stats()
        if not health["ray"]["initialized"]:
            raise SmokeError("Ray was not initialized")
        if health["ray"]["cluster_cpus"] < 1:
            raise SmokeError(f"expected at least one Ray CPU, got {health['ray']['cluster_cpus']}")
        if health["ray"]["estimated_rollout_slots"] < 1:
            raise SmokeError("expected at least one estimated rollout slot")

        request = RolloutRequest(
            request_id="ray-complete-request",
            task_id="ray_smoke_task",
            llm_config=_llm_config(),
            rollout_index=1,
        )
        result = await rollout.submit(request)
        if result != {"accepted": True, "request_id": request.request_id}:
            raise SmokeError(f"unexpected submit result: {result}")

        events = await _events_until_terminal(rollout, request.request_id, timeout_s=30.0)
        terminals = [event for event in events if event["type"] == "terminal"]
        if len(terminals) != 1:
            raise SmokeError(f"expected one completed terminal, got {len(terminals)}")
        terminal = terminals[0]
        if terminal["rollout_status"] != "completed" or terminal["trajectory_id"] != request.request_id:
            raise SmokeError(f"unexpected completed terminal: {terminal}")
    finally:
        rollout.close()


async def _check_cancelled_rollout(root: Path) -> None:
    rollout = RolloutEngine(config=_config(root / "cancel", SlowSmokeBenchmarkConfig(), name="ray_cancel_smoke"))
    try:
        request = RolloutRequest(
            request_id="ray-cancel-request",
            task_id="slow_ray_smoke_task",
            llm_config=_llm_config(),
        )
        await rollout.submit(request)
        result = await rollout.cancel(CancelRequest(request_id=request.request_id))
        if result != {"cancelled": 1}:
            raise SmokeError(f"unexpected cancel result: {result}")

        events = await _events_until_terminal(rollout, request.request_id, timeout_s=30.0)
        terminals = [event for event in events if event["type"] == "terminal"]
        if len(terminals) != 1:
            raise SmokeError(f"expected one cancelled terminal, got {len(terminals)}")
        terminal = terminals[0]
        if terminal["rollout_status"] != "cancelled" or terminal["trainable"] is not False:
            raise SmokeError(f"unexpected cancelled terminal: {terminal}")

        stats = rollout.stats()
        if stats["executor"]["cancelled_rollouts"] != 1 or stats["executor"]["pending_cancel_request_ids"]:
            raise SmokeError(f"unexpected cancellation stats: {stats['executor']}")
    finally:
        rollout.close()


async def _run() -> None:
    with tempfile.TemporaryDirectory(prefix="rl-ray-smoke-") as tmp:
        root = Path(tmp)
        await _check_completed_rollout(root)
        await _check_cancelled_rollout(root)


def main() -> int:
    try:
        asyncio.run(_run())
    except Exception as exc:
        print(f"SMOKE FAIL: {NAME} {type(exc).__name__}: {exc}")
        return 1
    print(f"SMOKE OK: {NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
