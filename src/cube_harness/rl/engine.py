from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, AsyncIterator

from cube.benchmark import Benchmark
from cube.task import TaskConfig

from cube_harness.rl.event_publisher import EventPublisher, EventPublisherConfig
from cube_harness.rl.events import AcceptedEvent, EventContext, TerminalEvent
from cube_harness.rl.rollout import AckRequest, CancelRequest, RolloutConfig, RolloutRequest

logger = logging.getLogger(__name__)


class RolloutEngine:
    """Benchmark-scoped rollout domain engine, independent of HTTP/SSE."""

    def __init__(
        self,
        *,
        event_publisher: Any | None = None,
        event_publisher_config: EventPublisherConfig | None = None,
        config: RolloutConfig | dict[str, Any],
        owns_event_publisher: bool = False,
    ) -> None:
        if isinstance(config, RolloutConfig):
            self.config = config
        else:
            self.config = RolloutConfig.model_validate(config)

        self._owns_ray = False
        self.event_publisher = event_publisher or self._make_event_publisher(event_publisher_config)
        self._owns_event_publisher = owns_event_publisher or event_publisher is None
        self._closed = False
        self._benchmark: Benchmark | None = None
        self._task_configs: dict[str, TaskConfig] = {}
        self._accepted_request_ids: set[str] = set()
        self._submit_lock = asyncio.Lock()
        self._setup_runtime()
        self.executor = self._make_executor()

    def _make_event_publisher(self, event_publisher_config: EventPublisherConfig | None) -> Any:
        if self.config.execution_mode == "local":
            return EventPublisher(event_publisher_config)
        from cube_harness.rl.ray_runtime import RayEventPublisher, ensure_ray_initialized

        self._owns_ray = ensure_ray_initialized(self._ray_init_kwargs())
        return RayEventPublisher.create(event_publisher_config, ray_options=self.config.ray.event_publisher_options)

    def _make_executor(self) -> Any:
        if self.config.execution_mode == "local":
            from cube_harness.rl.executor import LocalRolloutExecutor

            return LocalRolloutExecutor(
                payload_builder=self._rollout_payload,
                publisher_handle=self.event_publisher,
                has_terminal=self.event_publisher.has_terminal,
                publish_terminal=self.publish_terminal,
            )

        from cube_harness.rl.executor import RayRolloutExecutor

        return RayRolloutExecutor(
            ray_config=self.config.ray,
            payload_builder=self._rollout_payload,
            publisher_handle=self.event_publisher.publisher_handle,
            has_terminal=self.event_publisher.has_terminal,
            publish_terminal=self.publish_terminal,
        )

    def _ray_stats(self) -> dict[str, Any]:
        if self.config.execution_mode == "local":
            return {
                "initialized": False,
                "configured_num_workers": self.config.ray.num_workers,
                "task_num_cpus": self.config.ray.task_num_cpus,
                "poll_interval_s": self.config.ray.poll_interval_s,
                "cluster_resources": {},
                "available_resources": {},
                "cluster_cpus": 0.0,
                "available_cpus": 0.0,
                "estimated_rollout_slots": 0,
                "estimated_available_rollout_slots": 0,
            }

        from cube_harness.rl.ray_runtime import ray

        ray_initialized = ray.is_initialized()
        resources = ray.cluster_resources() if ray_initialized else {}
        available = ray.available_resources() if ray_initialized else {}
        cluster_cpus = float(resources.get("CPU", 0.0) or 0.0)
        available_cpus = float(available.get("CPU", 0.0) or 0.0)
        task_num_cpus = self.config.ray.task_num_cpus
        estimated_slots = int(cluster_cpus / task_num_cpus) if task_num_cpus > 0 else 0
        estimated_available_slots = int(available_cpus / task_num_cpus) if task_num_cpus > 0 else 0
        return {
            "initialized": ray_initialized,
            "configured_num_workers": self.config.ray.num_workers,
            "task_num_cpus": task_num_cpus,
            "poll_interval_s": self.config.ray.poll_interval_s,
            "cluster_resources": resources,
            "available_resources": available,
            "cluster_cpus": cluster_cpus,
            "available_cpus": available_cpus,
            "estimated_rollout_slots": estimated_slots,
            "estimated_available_rollout_slots": estimated_available_slots,
        }

    @property
    def ready(self) -> bool:
        return self._benchmark is not None and not self._closed

    @property
    def benchmark_name(self) -> str:
        return self.config.benchmark_config.benchmark_metadata.name

    async def submit(self, request: RolloutRequest) -> dict[str, Any]:
        self.validate_request(request)
        async with self._submit_lock:
            if request.request_id in self._accepted_request_ids:
                raise ValueError(f"request_id already accepted: {request.request_id}")
            self._accepted_request_ids.add(request.request_id)
            await asyncio.to_thread(self.publish_accepted, request)
            try:
                return await self.executor.submit(request)
            except Exception as exc:
                await asyncio.to_thread(
                    self.publish_terminal,
                    request,
                    "ray_error",
                    None,
                    False,
                    False,
                    {"type": type(exc).__name__, "message": str(exc)[:500]},
                )
                raise

    async def cancel(self, request: CancelRequest) -> dict[str, Any]:
        return await self.executor.cancel(request)

    async def events(
        self,
        *,
        from_offset: int = 0,
        stop_request_id: str | None = None,
        timeout_s: float | None = None,
        poll_timeout_s: float = 15.0,
    ) -> AsyncIterator[dict]:
        """Yield rollout events incrementally from the shared event publisher."""
        next_offset = from_offset
        deadline = None if timeout_s is None else asyncio.get_running_loop().time() + timeout_s
        while True:
            if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("timed out waiting for rollout events")
            wait_s = poll_timeout_s
            if deadline is not None:
                wait_s = max(0.0, min(wait_s, deadline - asyncio.get_running_loop().time()))
            batch = await self.wait_for_events(next_offset, timeout_s=wait_s)
            if not batch:
                continue
            for event in batch:
                next_offset = int(event["offset"]) + 1
                yield event
                if (
                    stop_request_id is not None
                    and event.get("type") == "terminal"
                    and event.get("request_id") == stop_request_id
                ):
                    return

    async def wait_for_events(self, from_offset: int, *, timeout_s: float = 15.0) -> list[dict]:
        return await asyncio.to_thread(self.event_publisher.wait_for_events, from_offset, timeout_s)

    async def wait_terminal(self, request_id: str, *, timeout_s: float | None = None) -> None:
        deadline = None if timeout_s is None else asyncio.get_running_loop().time() + timeout_s
        while not await asyncio.to_thread(self.event_publisher.has_terminal, request_id):
            if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"rollout did not emit terminal event: {request_id}")
            await asyncio.sleep(self.config.ray.poll_interval_s)

    async def ack(self, request: AckRequest) -> None:
        await asyncio.to_thread(self.event_publisher.ack, request.offset)

    def events_from(self, from_offset: int) -> list[dict]:
        return self.event_publisher.events_from(from_offset)

    def validate_request(self, request: RolloutRequest) -> None:
        if not self.ready:
            raise RuntimeError("rollout engine is not ready")
        if request.task_id not in self._task_configs:
            raise KeyError(f"unknown task_id {request.task_id!r}; available tasks: {sorted(self._task_configs)[:20]}")

    def task_configs(self) -> dict[str, Any]:
        return {
            "benchmark": {
                "name": self.benchmark_name,
                "task_count": len(self._task_configs),
            },
            "task_configs": [
                self._task_descriptor(task_id, task_config)
                for task_id, task_config in sorted(self._task_configs.items())
            ],
        }

    def _task_descriptor(self, task_id: str, task_config: TaskConfig) -> dict[str, Any]:
        metadata = getattr(task_config, "metadata", None)
        metadata_payload = (
            metadata.model_dump(mode="json", serialize_as_any=True) if hasattr(metadata, "model_dump") else None
        )
        return {
            "task_id": task_id,
            "metadata": metadata_payload,
        }

    def stats(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "name": self.config.name,
            "execution_mode": self.config.execution_mode,
            "persist_rollout": self.config.persist_rollout,
            "benchmark": {
                "name": self.benchmark_name,
                "task_count": len(self._task_configs),
                "task_ids": sorted(self._task_configs),
            },
            "ray": self._ray_stats(),
            "executor": self.executor.stats(),
            "event_publisher": self.event_publisher.health(),
        }

    def publish_accepted(self, request: RolloutRequest) -> None:
        self.event_publisher.publish(AcceptedEvent(event_index=-1, **self.event_context(request).model_dump()))

    def publish_terminal(
        self,
        request: RolloutRequest,
        status: str,
        final_reward: float | None,
        rollout_valid: bool,
        trainable: bool,
        error: dict[str, Any] | None,
    ) -> None:
        if self.event_publisher.has_terminal(request.request_id):
            return
        self.event_publisher.publish(
            TerminalEvent(
                event_index=-1,
                **self.event_context(request).model_dump(),
                rollout_status=status,  # type: ignore[arg-type]
                outcome_success=final_reward is not None and final_reward > 0,
                final_reward=final_reward,
                rollout_valid=rollout_valid,
                trainable=trainable,
                error=error,
            )
        )

    def event_context(self, request: RolloutRequest) -> EventContext:
        return EventContext(
            request_id=request.request_id,
            trajectory_id=request.request_id,
            env_name=self.benchmark_name,
            task_id=request.task_id,
            group_id=request.group_id,
            rollout_index=request.rollout_index,
            model_version=request.model_version,
        )

    def _ray_init_kwargs(self) -> dict[str, Any]:
        kwargs = dict(self.config.ray.init_kwargs)
        kwargs.setdefault("num_cpus", self.config.ray.num_workers)
        return kwargs

    def _setup_runtime(self) -> None:
        if self.config.persist_rollout:
            self._save_config()
        self.config.benchmark_config.install()
        self._benchmark = self.config.benchmark_config.make(self.config.infra)
        self._task_configs = {
            str(task_config.task_id): task_config for task_config in self.config.benchmark_config.get_task_configs()
        }
        logger.info(
            "Rollout ready for benchmark=%s tasks=%d output_dir=%s ray_capacity=%d",
            self.benchmark_name,
            len(self._task_configs),
            self.config.output_dir,
            self.config.ray.num_workers if self.config.execution_mode == "ray" else 0,
        )

    def _save_config(self) -> None:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "rollout_config.json"
        config_path.write_text(self.config.model_dump_json(indent=2, serialize_as_any=True))
        logger.info("Saved rollout config to %s", config_path)

    def _rollout_payload(self, request: RolloutRequest) -> dict[str, Any]:
        if self._benchmark is None:
            raise RuntimeError("rollout engine is closed")
        output_dir = Path(self.config.output_dir) / request.request_id
        request_payload = request.model_dump(mode="python")
        llm_payload = request.llm_config.model_dump(mode="python")
        llm_payload["api_key"] = request.llm_config.api_key.get_secret_value()
        request_payload["llm_config"] = llm_payload
        return {
            "request": request_payload,
            "task_config": self._task_configs[request.task_id],
            "agent_config": self.config.agent_config.model_copy(deep=True),
            "runtime_context": getattr(self._benchmark, "_runtime_context", None),
            "output_dir": str(output_dir),
            "persist_rollout": self.config.persist_rollout,
            "service_name": self.config.name,
            "benchmark_name": self.benchmark_name,
            "max_steps": self.config.max_steps,
            "event_context": self.event_context(request),
        }

    def close(self) -> None:
        self._closed = True
        self.executor.close()
        if self._benchmark is not None:
            try:
                self._benchmark.close()
            except Exception:
                logger.warning("benchmark close failed", exc_info=True)
            self._benchmark = None
        if self._owns_event_publisher:
            try:
                self.event_publisher.close()
            except Exception:
                logger.debug("Ray event publisher already stopped", exc_info=True)
        if self._owns_ray:
            from cube_harness.rl.ray_runtime import ray

            ray.shutdown()
