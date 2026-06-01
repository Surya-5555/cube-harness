from __future__ import annotations

import os
from typing import Any

_RAY_ENABLE_UV_RUN_RUNTIME_ENV = "RAY_ENABLE_UV_RUN_RUNTIME_ENV"


def _configure_ray_environment() -> None:
    # Ray reads this during startup/import under uv; keep it at the Ray boundary.
    os.environ.setdefault(_RAY_ENABLE_UV_RUN_RUNTIME_ENV, "0")


_configure_ray_environment()

import ray  # noqa: E402

from cube_harness.rl.sink import EventSink, EventSinkConfig  # noqa: E402
from cube_harness.rl.task_runner import RolloutTaskRunner  # noqa: E402


class _RayEventSinkActor:
    """Private Ray actor that owns the local EventSink state and offset counter."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._sink = EventSink(EventSinkConfig.model_validate(config or {}))

    def publish_payload(self, payload: dict[str, Any]) -> dict:
        return self._sink.publish_payload(payload)

    def events_from(self, from_offset: int) -> list[dict]:
        return self._sink.events_from(from_offset)

    def wait_for_events(self, from_offset: int, timeout: float = 15.0) -> list[dict]:
        return self._sink.wait_for_events(from_offset, timeout)

    def ack(self, client_id: str, offset: int) -> None:
        self._sink.ack(client_id, offset)

    def has_terminal(self, request_id: str) -> bool:
        return self._sink.has_terminal(request_id)

    def health(self) -> dict:
        return self._sink.health()

    def close(self) -> None:
        return None


class RayEventSink:
    """Cross-process event sink implementation used by Ray rollout tasks."""

    def __init__(self, actor: Any) -> None:
        self._actor = actor

    @property
    def publisher_handle(self) -> Any:
        return self._actor

    @classmethod
    def create(
        cls, config: EventSinkConfig | None = None, *, ray_options: dict[str, Any] | None = None
    ) -> "RayEventSink":
        options = {"num_cpus": 0, "max_restarts": 0, "max_task_retries": 0, "max_concurrency": 64}
        options.update(ray_options or {})
        config_payload = (config or EventSinkConfig()).model_dump(mode="json")
        actor = ray.remote(_RayEventSinkActor).options(**options).remote(config_payload)
        return cls(actor)

    def publish(self, event: Any) -> dict:
        payload = event.model_dump(mode="json")
        published = self.publish_payload(payload)
        event.offset = published["offset"]
        return published

    def publish_payload(self, payload: dict[str, Any]) -> dict:
        return ray.get(self._actor.publish_payload.remote(payload))

    def events_from(self, from_offset: int) -> list[dict]:
        return ray.get(self._actor.events_from.remote(from_offset))

    def wait_for_events(self, from_offset: int, timeout: float = 15.0) -> list[dict]:
        return ray.get(self._actor.wait_for_events.remote(from_offset, timeout))

    def ack(self, client_id: str, offset: int) -> None:
        ray.get(self._actor.ack.remote(client_id, offset))

    def has_terminal(self, request_id: str) -> bool:
        return bool(ray.get(self._actor.has_terminal.remote(request_id)))

    def health(self) -> dict:
        return ray.get(self._actor.health.remote())

    def close(self) -> None:
        try:
            ray.get(self._actor.close.remote(), timeout=2)
        finally:
            ray.kill(self._actor, no_restart=True)


def ensure_ray_initialized(init_kwargs: dict[str, Any] | None = None) -> bool:
    if ray.is_initialized():
        return False
    kwargs = {"include_dashboard": False, "log_to_driver": False, "ignore_reinit_error": True}
    kwargs.update(init_kwargs or {})
    ray.init(**kwargs)
    return True


@ray.remote(max_retries=0, num_cpus=0.25)
def run_rollout_task(payload: dict[str, Any], publisher_handle: Any) -> dict[str, Any]:
    """Ray entry point for one stateless rollout, matching exp_runner's episode task shape."""
    return RolloutTaskRunner(payload, publisher_handle).run()
