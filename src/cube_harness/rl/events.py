from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

EventType = Literal["accepted", "llm_call", "tool_call", "evaluation", "agent_error", "terminal"]
RolloutStatus = Literal[
    "accepted",
    "completed",
    "max_steps",
    "agent_error",
    "tool_error",
    "llm_error",
    "event_error",
    "ray_error",
    "training_data_error",
    "cancelled",
]


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(exclude_none=True))
    if hasattr(value, "dict"):
        return _jsonable(value.dict(exclude_none=True))
    return str(value)


class EventContext(BaseModel):
    request_id: str
    trajectory_id: str
    env_name: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    group_id: str | None = None
    rollout_index: int = 0
    model_version: int | None = None


class RolloutControlEvent(BaseModel):
    type: EventType
    offset: int | None = None
    event_index: int
    request_id: str
    trajectory_id: str
    env_name: str | None = None
    task_id: str | None = None
    group_id: str | None = None
    rollout_index: int = 0
    model_version: int | None = None
    timestamp: float = Field(default_factory=time.time)


class AcceptedEvent(RolloutControlEvent):
    type: Literal["accepted"] = "accepted"


class TerminalEvent(RolloutControlEvent):
    type: Literal["terminal"] = "terminal"
    rollout_status: RolloutStatus
    outcome_success: bool = False
    final_reward: float | None = None
    rollout_valid: bool = False
    trainable: bool = False
    error: dict[str, Any] | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


AnyRolloutEvent = AcceptedEvent | TerminalEvent


def event_context_payload(ctx: EventContext) -> dict[str, Any]:
    return ctx.model_dump()


def rollout_event_payload(ctx: EventContext, event_type: EventType, event_index: int) -> dict[str, Any]:
    return {
        "type": event_type,
        "event_index": event_index,
        **ctx.model_dump(),
        "timestamp": time.time(),
    }


def dump_for_event(value: Any) -> Any:
    return _jsonable(value)
