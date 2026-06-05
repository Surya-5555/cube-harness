from __future__ import annotations

# Run explicitly with:
# uv run pytest --run-perf -m perf tests/perf/test_rollout_throughput.py
import asyncio
import time

import pytest

from cube_harness.rl import RayConfig, RolloutConfig, RolloutEngine, RolloutRequest
from cube_harness.rl.llm import RolloutLLMConfig
from tests.conftest import MockAgentConfig
from tests.rollout_perf_helpers import SlowRolloutBenchmarkConfig


@pytest.mark.perf
def test_rollout_throughput_increases_with_more_ray_workers(tmp_dir) -> None:
    async def run_batch(num_workers: int, batch_size: int) -> float:
        rollout = RolloutEngine(
            config=RolloutConfig(
                name=f"throughput_{num_workers}",
                output_dir=tmp_dir / f"throughput_{num_workers}",
                benchmark_config=SlowRolloutBenchmarkConfig(),
                agent_config=MockAgentConfig(),
                max_steps=1,
                ray=RayConfig(num_workers=num_workers),
            )
        )

        async def submit_and_wait(prefix: str) -> float:
            requests = [
                RolloutRequest(
                    request_id=f"{prefix}-{idx}",
                    client_id=f"throughput-{num_workers}",
                    task_id="slow_rollout_task",
                    llm_config=RolloutLLMConfig(
                        model_name="served-model",
                        api_base="http://localhost:8000/v1",
                        api_key="EMPTY",
                        tokenizer_name="mock-tokenizer",
                    ),
                    rollout_index=idx,
                )
                for idx in range(batch_size)
            ]
            start_offset = rollout.sink.health()["next_offset"]
            start = time.perf_counter()
            await asyncio.gather(*(rollout.submit(request) for request in requests))
            terminal_ids: set[str] = set()
            async for event in rollout.events(from_offset=start_offset, timeout_s=60.0, poll_timeout_s=0.1):
                if event["type"] == "terminal" and event["request_id"].startswith(prefix):
                    terminal_ids.add(event["request_id"])
                    if len(terminal_ids) == batch_size:
                        break
            assert len(terminal_ids) == batch_size
            return time.perf_counter() - start

        try:
            await submit_and_wait(prefix=f"warmup-{num_workers}")
            return await submit_and_wait(prefix=f"throughput-{num_workers}")
        finally:
            rollout.close()

    batch_size = 16
    single_worker_s = asyncio.run(run_batch(num_workers=1, batch_size=batch_size))
    four_workers_s = asyncio.run(run_batch(num_workers=4, batch_size=batch_size))

    assert four_workers_s < single_worker_s * 0.85, (
        f"expected higher rollout throughput with more Ray workers; "
        f"1 worker took {single_worker_s:.3f}s, 4 workers took {four_workers_s:.3f}s"
    )
