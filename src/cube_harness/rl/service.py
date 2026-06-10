from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from cube_harness.episode_logs import LOG_FORMAT
from cube_harness.rl.engine import RolloutEngine
from cube_harness.rl.event_publisher import EventPublisherConfig
from cube_harness.rl.rollout import AckRequest, CancelRequest, RolloutConfig, RolloutRequest


def configure_terminal_logging(level: str | int = logging.INFO, *, force: bool = False) -> None:
    """Configure terminal logging for rollout service entrypoints before app creation."""
    if isinstance(level, str):
        level_value = getattr(logging, level.upper(), logging.INFO)
    else:
        level_value = level
    logging.basicConfig(level=level_value, format=LOG_FORMAT, force=force)
    logging.getLogger().setLevel(level_value)


class RolloutService:
    """HTTP/SSE control plane adapter around a benchmark-scoped RolloutEngine."""

    def __init__(self, rollout: RolloutEngine) -> None:
        self.rollout = rollout

    async def submit(self, request: RolloutRequest) -> dict[str, Any]:
        return await self.rollout.submit(request)

    async def cancel(self, request: CancelRequest) -> dict[str, Any]:
        return await self.rollout.cancel(request)

    async def ack(self, request: AckRequest) -> None:
        await self.rollout.ack(request)

    def events_from(self, from_offset: int) -> list[dict]:
        return self.rollout.events_from(from_offset)

    def health(self) -> dict[str, Any]:
        return self.rollout.stats()

    def task_configs(self) -> dict[str, Any]:
        return self.rollout.task_configs()

    def close(self) -> None:
        self.rollout.close()


def _sse_frame(event: dict) -> str:
    return f"id: {event['offset']}\nevent: {event['type']}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n"


def serve(
    *,
    config: RolloutConfig | dict[str, Any],
    event_publisher: Any | None = None,
    event_publisher_config: EventPublisherConfig | None = None,
) -> FastAPI:
    service_config = config if isinstance(config, RolloutConfig) else RolloutConfig.model_validate(config)
    rollout = RolloutEngine(
        event_publisher=event_publisher,
        event_publisher_config=event_publisher_config,
        config=service_config,
        owns_event_publisher=event_publisher is None,
    )
    service = RolloutService(rollout)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        service.close()

    # TODO(auth): add optional bearer-token auth plus endpoint/tokenizer allowlists
    # before supporting non-local or untrusted rollout clients.
    app = FastAPI(title="cube-harness rollouts", lifespan=lifespan)
    app.state.service = service

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return await asyncio.to_thread(service.health)

    @app.get("/task-configs")
    async def task_configs() -> dict[str, Any]:
        return await asyncio.to_thread(service.task_configs)

    @app.post("/rollouts")
    async def rollouts(request: RolloutRequest) -> dict[str, Any]:
        try:
            return await service.submit(request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/acks")
    async def acks(request: AckRequest) -> dict[str, Any]:
        await service.ack(request)
        return {"ok": True}

    @app.post("/cancel")
    async def cancel(request: CancelRequest) -> dict[str, Any]:
        return await service.cancel(request)

    @app.get("/events")
    async def events(
        from_offset: int = Query(0),
    ) -> StreamingResponse:
        async def stream_events():
            next_offset = from_offset
            while True:
                events_batch = await service.rollout.wait_for_events(next_offset, timeout_s=15.0)
                if not events_batch:
                    yield ": keepalive\n\n"
                    continue
                for event in events_batch:
                    next_offset = int(event["offset"]) + 1
                    yield _sse_frame(event)

        return StreamingResponse(stream_events(), media_type="text/event-stream")

    return app
