from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from cube_harness.rl.ray_runtime import ray, run_rollout_task
from cube_harness.rl.rollout import CancelRequest, RayConfig, RolloutRequest
from cube_harness.rl.task_runner import RolloutTaskRunner

logger = logging.getLogger(__name__)


class RayRolloutExecutor:
    """Ray execution manager for accepted rollout payloads."""

    def __init__(
        self,
        *,
        ray_config: RayConfig,
        payload_builder: Callable[[RolloutRequest], dict[str, Any]],
        publisher_handle: Any,
        has_terminal: Callable[[str], bool],
        publish_terminal: Callable[[RolloutRequest, str, float | None, bool, bool, dict[str, Any] | None], None],
    ) -> None:
        self.ray_config = ray_config
        self.payload_builder = payload_builder
        self.publisher_handle = publisher_handle
        self.has_terminal = has_terminal
        self.publish_terminal = publish_terminal
        self._closed = False
        self._accepted_requests: dict[str, RolloutRequest] = {}
        self._terminal: set[str] = set()
        self._cancelled: set[str] = set()
        self._ref_to_request: dict[Any, RolloutRequest] = {}
        self._request_to_ref: dict[str, Any] = {}
        self._monitor_task: asyncio.Task | None = None
        self._completed_ray_tasks = 0
        self._failed_ray_tasks = 0
        self._synthesized_terminal_events = 0
        self._terminal_publish_failures = 0
        # Protects only in-memory request/ref bookkeeping. Awaiting it yields;
        # blocking Ray and sink calls happen outside the lock via to_thread.
        self._lock = asyncio.Lock()

    async def submit(self, request: RolloutRequest) -> dict[str, Any]:
        async with self._lock:
            if request.request_id in self._accepted_requests:
                raise ValueError(f"request_id already accepted: {request.request_id}")
            self._accepted_requests[request.request_id] = request

        try:
            ref = await asyncio.to_thread(self._start_request, request)
        except Exception:
            async with self._lock:
                self._accepted_requests.pop(request.request_id, None)
            raise

        async with self._lock:
            if request.request_id in self._cancelled:
                should_cancel = True
            else:
                should_cancel = False
                self._ref_to_request[ref] = request
                self._request_to_ref[request.request_id] = ref
                self._ensure_monitor_locked()

        if should_cancel:
            await self._cancel_ref(ref, request, "rollout cancelled before Ray dispatch completed")

        return {"accepted": True, "request_id": request.request_id}

    async def cancel(self, request: CancelRequest) -> dict[str, Any]:
        matched: list[tuple[Any | None, RolloutRequest]] = []
        async with self._lock:
            for request_id, active in list(self._accepted_requests.items()):
                if request_id in self._terminal or request_id in self._cancelled:
                    continue
                if not self._matches_cancel(request, active):
                    continue
                self._cancelled.add(request_id)
                ref = self._request_to_ref.get(request_id)
                if ref is not None:
                    self._drop_ref_locked(ref)
                matched.append((ref, active))

        for ref, active in matched:
            if ref is None:
                await self._publish_terminal_status(
                    active,
                    "cancelled",
                    None,
                    False,
                    False,
                    {"type": "Cancelled", "message": "rollout cancelled before Ray dispatch"},
                )
            else:
                await self._cancel_ref(ref, active, "rollout cancelled while running")
        return {"cancelled": len(matched)}

    def stats(self) -> dict[str, Any]:
        active_request_ids = sorted(self._request_to_ref)
        estimated_slots = max(int(self.ray_config.num_workers / self.ray_config.task_num_cpus), 1)
        if ray.is_initialized():
            cpus = float(ray.cluster_resources().get("CPU", 0.0) or 0.0)
            estimated_slots = max(int(cpus / self.ray_config.task_num_cpus), 1)
        estimated_running = min(len(self._ref_to_request), estimated_slots)
        estimated_queued = max(len(self._ref_to_request) - estimated_running, 0)
        return {
            "inflight_rollouts": len(self._ref_to_request),
            "estimated_running_rollouts": estimated_running,
            "estimated_queued_rollouts": estimated_queued,
            "accepted_rollouts": len(self._accepted_requests),
            "terminal_rollouts": len(self._terminal),
            "cancelled_rollouts": len(self._cancelled),
            "completed_ray_tasks": self._completed_ray_tasks,
            "failed_ray_tasks": self._failed_ray_tasks,
            "synthesized_terminal_events": self._synthesized_terminal_events,
            "terminal_publish_failures": self._terminal_publish_failures,
            "running_request_ids": active_request_ids,
            "pending_cancel_request_ids": sorted(self._cancelled - self._terminal),
        }

    def close(self) -> None:
        self._closed = True
        if self._monitor_task is not None:
            self._monitor_task.cancel()
        for ref in list(self._ref_to_request):
            try:
                ray.cancel(ref, force=True)
            except Exception:
                logger.debug("Ray rollout ref already complete during executor close", exc_info=True)
        self._ref_to_request.clear()
        self._request_to_ref.clear()

    async def _cancel_ref(self, ref: Any, request: RolloutRequest, message: str) -> None:
        try:
            await asyncio.to_thread(ray.cancel, ref, force=True)
        except Exception:
            logger.debug("Ray rollout ref already complete during cancellation", exc_info=True)
        await self._publish_terminal_status(
            request,
            "cancelled",
            None,
            False,
            False,
            {"type": "Cancelled", "message": message},
        )

    async def _publish_terminal_status(
        self,
        request: RolloutRequest,
        status: str,
        final_reward: float | None,
        rollout_valid: bool,
        trainable: bool,
        error: dict[str, Any] | None,
    ) -> None:
        try:
            if not await asyncio.to_thread(self.has_terminal, request.request_id):
                await asyncio.to_thread(
                    self.publish_terminal,
                    request,
                    status,
                    final_reward,
                    rollout_valid,
                    trainable,
                    error,
                )
        except Exception:
            self._terminal_publish_failures += 1
            logger.exception("Failed to publish terminal event for request %s", request.request_id)
            raise
        async with self._lock:
            self._terminal.add(request.request_id)

    def _start_request(self, request: RolloutRequest) -> Any:
        options = {"max_retries": 0, "num_cpus": self.ray_config.task_num_cpus}
        options.update(self.ray_config.task_options)
        ref = run_rollout_task.options(**options).remote(self.payload_builder(request), self.publisher_handle)
        logger.info("Submitted rollout %s to Ray", request.request_id)
        return ref

    async def _finish_request(self, request: RolloutRequest, ref: Any) -> None:
        try:
            await asyncio.to_thread(ray.get, ref)
        except Exception as exc:
            self._failed_ray_tasks += 1
            logger.exception("Ray rollout task failed for request %s", request.request_id)
            if not await asyncio.to_thread(self.has_terminal, request.request_id):
                self._synthesized_terminal_events += 1
                await self._publish_terminal_status(
                    request,
                    "ray_error",
                    None,
                    False,
                    False,
                    {"type": type(exc).__name__, "message": str(exc)[:500]},
                )
            return

        self._completed_ray_tasks += 1
        if not await asyncio.to_thread(self.has_terminal, request.request_id):
            self._synthesized_terminal_events += 1
            await self._publish_terminal_status(
                request,
                "event_error",
                None,
                False,
                False,
                {"type": "MissingTerminal", "message": "worker completed without terminal event"},
            )

    def _ensure_monitor_locked(self) -> None:
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_workers(), name="ray-rollout-executor-monitor")

    async def _monitor_workers(self) -> None:
        while not self._closed:
            async with self._lock:
                refs = list(self._ref_to_request)
                if not refs:
                    return
            done_refs, _ = await asyncio.to_thread(
                ray.wait,
                refs,
                num_returns=1,
                timeout=self.ray_config.poll_interval_s,
            )
            if not done_refs:
                continue
            for ref in done_refs:
                await self._handle_completed_ref(ref)

    async def _handle_completed_ref(self, ref: Any) -> None:
        async with self._lock:
            request = self._ref_to_request.get(ref)
            if request is None:
                return
        await self._finish_request(request, ref)
        has_terminal = await asyncio.to_thread(self.has_terminal, request.request_id)
        async with self._lock:
            self._drop_ref_locked(ref)
            if has_terminal:
                self._terminal.add(request.request_id)

    def _drop_ref_locked(self, ref: Any) -> None:
        request = self._ref_to_request.pop(ref, None)
        if request is not None:
            self._request_to_ref.pop(request.request_id, None)

    def _matches_cancel(self, cancel: CancelRequest, request: RolloutRequest) -> bool:
        if cancel.request_id is not None and request.request_id != cancel.request_id:
            return False
        if cancel.group_id is not None and request.group_id != cancel.group_id:
            return False
        return True


class LocalRolloutExecutor:
    """Sequential in-process executor for local rollout/debug runs."""

    def __init__(
        self,
        *,
        payload_builder: Callable[[RolloutRequest], dict[str, Any]],
        publisher_handle: Any,
        has_terminal: Callable[[str], bool],
        publish_terminal: Callable[[RolloutRequest, str, float | None, bool, bool, dict[str, Any] | None], None],
    ) -> None:
        self.payload_builder = payload_builder
        self.publisher_handle = publisher_handle
        self.has_terminal = has_terminal
        self.publish_terminal = publish_terminal
        self._accepted_requests: dict[str, RolloutRequest] = {}
        self._terminal: set[str] = set()
        self._cancelled: set[str] = set()
        self._running_request_id: str | None = None
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._synthesized_terminal_events = 0
        self._terminal_publish_failures = 0
        self._lock = asyncio.Lock()

    async def submit(self, request: RolloutRequest) -> dict[str, Any]:
        async with self._lock:
            if request.request_id in self._accepted_requests:
                raise ValueError(f"request_id already accepted: {request.request_id}")
            self._accepted_requests[request.request_id] = request
            if request.request_id in self._cancelled:
                await asyncio.to_thread(
                    self._publish_terminal_status,
                    request,
                    "cancelled",
                    None,
                    False,
                    False,
                    {"type": "Cancelled", "message": "rollout cancelled before local execution"},
                )
                return {"accepted": True, "request_id": request.request_id}

            self._running_request_id = request.request_id
            try:
                await asyncio.to_thread(self._run_request, request)
            finally:
                self._running_request_id = None
        return {"accepted": True, "request_id": request.request_id}

    async def cancel(self, request: CancelRequest) -> dict[str, Any]:
        matched: list[RolloutRequest] = []
        async with self._lock:
            for request_id, active in list(self._accepted_requests.items()):
                if request_id in self._terminal or request_id in self._cancelled:
                    continue
                if not self._matches_cancel(request, active):
                    continue
                self._cancelled.add(request_id)
                matched.append(active)

            for active in matched:
                if active.request_id == self._running_request_id:
                    # In-process local execution cannot preempt arbitrary task code.
                    continue
                await asyncio.to_thread(
                    self._publish_terminal_status,
                    active,
                    "cancelled",
                    None,
                    False,
                    False,
                    {"type": "Cancelled", "message": "rollout cancelled before local execution"},
                )
        return {"cancelled": len(matched)}

    def stats(self) -> dict[str, Any]:
        return {
            "inflight_rollouts": 1 if self._running_request_id is not None else 0,
            "accepted_rollouts": len(self._accepted_requests),
            "terminal_rollouts": len(self._terminal),
            "cancelled_rollouts": len(self._cancelled),
            "completed_local_tasks": self._completed_tasks,
            "failed_local_tasks": self._failed_tasks,
            "synthesized_terminal_events": self._synthesized_terminal_events,
            "terminal_publish_failures": self._terminal_publish_failures,
            "running_request_ids": [self._running_request_id] if self._running_request_id is not None else [],
            "pending_cancel_request_ids": sorted(self._cancelled - self._terminal),
        }

    def close(self) -> None:
        return None

    def _run_request(self, request: RolloutRequest) -> None:
        try:
            RolloutTaskRunner(self.payload_builder(request), self.publisher_handle).run()
        except Exception as exc:
            self._failed_tasks += 1
            logger.exception("Local rollout task failed for request %s", request.request_id)
            if not self.has_terminal(request.request_id):
                self._synthesized_terminal_events += 1
                self._publish_terminal_status(
                    request,
                    "agent_error",
                    None,
                    False,
                    False,
                    {"type": type(exc).__name__, "message": str(exc)[:500]},
                )
            return

        self._completed_tasks += 1
        if not self.has_terminal(request.request_id):
            self._synthesized_terminal_events += 1
            self._publish_terminal_status(
                request,
                "event_error",
                None,
                False,
                False,
                {"type": "MissingTerminal", "message": "local worker completed without terminal event"},
            )
        else:
            self._terminal.add(request.request_id)

    def _publish_terminal_status(
        self,
        request: RolloutRequest,
        status: str,
        final_reward: float | None,
        rollout_valid: bool,
        trainable: bool,
        error: dict[str, Any] | None,
    ) -> None:
        try:
            if not self.has_terminal(request.request_id):
                self.publish_terminal(request, status, final_reward, rollout_valid, trainable, error)
        except Exception:
            self._terminal_publish_failures += 1
            logger.exception("Failed to publish terminal event for request %s", request.request_id)
            raise
        self._terminal.add(request.request_id)

    def _matches_cancel(self, cancel: CancelRequest, request: RolloutRequest) -> bool:
        if cancel.request_id is not None and request.request_id != cancel.request_id:
            return False
        if cancel.group_id is not None and request.group_id != cancel.group_id:
            return False
        return True
