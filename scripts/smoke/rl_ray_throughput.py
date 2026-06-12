# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cube-harness[rl]",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "../..", editable = true }
# ///
"""SMOKE: Ray rollout throughput scales with worker count.

This is the smoke version of the former pytest throughput check. It starts real
Ray runtimes and compares a slow deterministic rollout batch with one worker vs
four workers, so it intentionally lives outside the default PR test suite.

Example:

    uv run scripts/smoke/rl_ray_throughput.py

Prints SMOKE OK|FAIL: rl_ray_throughput  (exit 0|1).
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
from cube_harness.rl import RayConfig, RolloutConfig, RolloutEngine, RolloutRequest
from cube_harness.rl.llm import RolloutLLMConfig

NAME = "rl_ray_throughput"
BATCH_SIZE = 16
SPEEDUP_RATIO = 0.85


class SmokeTool(Tool):
    # Inherits the base `final_step` (@tool_action raising AgentStop) — the termination action.
    pass


class SmokeToolConfig(ToolConfig):
    def make(self, container=None) -> SmokeTool:
        _ = container
        return SmokeTool()


class SmokeAgentConfig(AgentConfig):
    name: str = "throughput_smoke_agent"

    def make(self, action_set=None, **kwargs) -> "SmokeAgent":
        _ = action_set, kwargs
        return SmokeAgent(config=self)


class SmokeAgent(Agent):
    name = "ThroughputSmokeAgent"
    description = "Deterministic agent for Ray throughput smoke coverage."
    input_content_types = ["text"]
    output_content_types = ["action"]

    def step(self, obs: Observation) -> AgentOutput:
        _ = obs
        time.sleep(0.25)
        return AgentOutput(actions=[Action(name="final_step", arguments={})])


class SlowSmokeTask(Task):
    def reset(self) -> tuple[Observation, dict]:
        return Observation.from_text("finish the slow smoke task"), {}

    def step(self, action: Action | list[Action]) -> EnvironmentOutput:
        _ = action
        return EnvironmentOutput(obs=Observation.from_text("done"), reward=1.0, done=True, info={"success": True})

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict]:
        _ = obs
        return 1.0, {"success": True}


class SlowSmokeTaskConfig(TaskConfig):
    def make(self, runtime_context=None) -> SlowSmokeTask:
        _ = runtime_context
        return SlowSmokeTask(metadata=TaskMetadata(id=self.task_id), tool_config=self.tool_config or SmokeToolConfig())


class SlowSmokeBenchmark(Benchmark):
    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass


class SlowSmokeBenchmarkConfig(BenchmarkConfig):
    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="slow-rollout-smoke-cube",
        version="0.1.0",
        description="Slow deterministic benchmark for Ray throughput smoke coverage.",
    )
    task_metadata: ClassVar[dict[str, TaskMetadata]] = {
        "slow_rollout_smoke_task": TaskMetadata(id="slow_rollout_smoke_task"),
    }
    task_config_class: ClassVar[type[TaskConfig]] = SlowSmokeTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = SlowSmokeBenchmark

    @classmethod
    def install(cls) -> None:
        pass


class SmokeError(Exception):
    pass


def _llm_config() -> RolloutLLMConfig:
    return RolloutLLMConfig(
        model_name="served-model",
        api_base="http://localhost:8000/v1",
        api_key="EMPTY",
        tokenizer_name="mock-tokenizer",
    )


def _config(root: Path, num_workers: int) -> RolloutConfig:
    return RolloutConfig(
        name=f"throughput_{num_workers}",
        output_dir=root / f"throughput_{num_workers}",
        benchmark_config=SlowSmokeBenchmarkConfig(),
        agent_config=SmokeAgentConfig(),
        max_steps=1,
        execution_mode="ray",
        ray=RayConfig(num_workers=num_workers),
    )


async def _submit_and_wait(rollout: RolloutEngine, *, prefix: str, batch_size: int) -> float:
    requests = [
        RolloutRequest(
            request_id=f"{prefix}-{idx}",
            task_id="slow_rollout_smoke_task",
            llm_config=_llm_config(),
            rollout_index=idx,
        )
        for idx in range(batch_size)
    ]
    start_offset = rollout.event_publisher.health()["next_offset"]
    start = time.perf_counter()
    await asyncio.gather(*(rollout.submit(request) for request in requests))
    terminal_ids: set[str] = set()
    async for event in rollout.events(from_offset=start_offset, timeout_s=60.0, poll_timeout_s=0.1):
        if event["type"] == "terminal" and event["request_id"].startswith(prefix):
            terminal_ids.add(event["request_id"])
            if len(terminal_ids) == batch_size:
                break
    if len(terminal_ids) != batch_size:
        raise SmokeError(f"expected {batch_size} terminals for {prefix}, got {len(terminal_ids)}")
    return time.perf_counter() - start


async def _run_batch(root: Path, *, num_workers: int, batch_size: int) -> float:
    rollout = RolloutEngine(config=_config(root, num_workers))
    try:
        await _submit_and_wait(rollout, prefix=f"warmup-{num_workers}", batch_size=batch_size)
        return await _submit_and_wait(rollout, prefix=f"throughput-{num_workers}", batch_size=batch_size)
    finally:
        rollout.close()


async def _run() -> tuple[float, float]:
    with tempfile.TemporaryDirectory(prefix="rl-ray-throughput-") as tmp:
        root = Path(tmp)
        single_worker_s = await _run_batch(root, num_workers=1, batch_size=BATCH_SIZE)
        four_workers_s = await _run_batch(root, num_workers=4, batch_size=BATCH_SIZE)
    if four_workers_s >= single_worker_s * SPEEDUP_RATIO:
        raise SmokeError(
            f"expected higher rollout throughput with more Ray workers; "
            f"1 worker took {single_worker_s:.3f}s, 4 workers took {four_workers_s:.3f}s"
        )
    return single_worker_s, four_workers_s


def main() -> int:
    try:
        single_worker_s, four_workers_s = asyncio.run(_run())
    except Exception as exc:
        print(f"SMOKE FAIL: {NAME} {type(exc).__name__}: {exc}")
        return 1
    print(f"SMOKE OK: {NAME} one_worker_s={single_worker_s:.3f} four_workers_s={four_workers_s:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
