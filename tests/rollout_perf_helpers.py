from __future__ import annotations

import time

from cube.benchmark import BenchmarkMetadata
from cube.core import EnvironmentOutput, Observation
from cube.task import TaskMetadata

from tests.conftest import (
    MockCubeBenchmark,
    MockCubeBenchmarkConfig,
    MockCubeTask,
    MockCubeTaskConfig,
    MockToolConfig,
)


class SlowRolloutTask(MockCubeTask):
    def step(self, actions) -> EnvironmentOutput:
        _ = actions
        time.sleep(0.25)
        return EnvironmentOutput(obs=Observation.from_text("done"), reward=1.0, done=True, info={"success": True})


class SlowRolloutTaskConfig(MockCubeTaskConfig):
    def make(self, runtime_context=None) -> SlowRolloutTask:
        _ = runtime_context
        return SlowRolloutTask(
            metadata=TaskMetadata(id=self.task_id),
            tool_config=self.tool_config or MockToolConfig(),
        )


class SlowRolloutBenchmarkConfig(MockCubeBenchmarkConfig):
    benchmark_metadata = BenchmarkMetadata(
        name="slow-rollout-cube",
        version="0.1.0",
        description="Slow mock cube benchmark for rollout throughput tests",
    )
    task_metadata = {
        "slow_rollout_task": TaskMetadata(id="slow_rollout_task"),
    }
    task_config_class = SlowRolloutTaskConfig
    benchmark_class = MockCubeBenchmark
