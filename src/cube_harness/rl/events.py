from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

EventType = Literal["accepted", "llm_call", "agent_step", "env_step", "terminal"]
RolloutStatus = Literal[
    "accepted",
    "completed",
    "max_steps",
    "agent_error",
    "env_error",
    "llm_error",
    "event_error",
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
    group_id: str | None = None
    rollout_index: int = 0
    model_version: int | None = None


class RolloutEvent(BaseModel):
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



class AcceptedEvent(RolloutEvent):
    type: Literal["accepted"] = "accepted"


class LLMCallEvent(RolloutEvent):
    type: Literal["llm_call"] = "llm_call"
    agent_step_id: str | None = None
    agent_step_index: int | None = None
    llm_call_index: int
    trainable_call_index: int | None = None
    llm_call_id: str
    llm_call_tag: str = ""
    trainable: bool = False
    emitted_at: Literal["agent_step_immediate", "agent_step_fallback"] = "agent_step_fallback"
    prompt: dict[str, Any]
    output: dict[str, Any]
    usage: dict[str, Any] = Field(default_factory=dict)
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    finish_reason: str | None = None
    state_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentStepEvent(RolloutEvent):
    type: Literal["agent_step"] = "agent_step"
    agent_step_id: str
    agent_step_index: int
    actions: list[dict[str, Any]] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    profiling: dict[str, Any] = Field(default_factory=dict)
    thoughts: str | None = None
    llm_call_ids: list[str] = Field(default_factory=list)
    started_at: float | None = None
    ended_at: float | None = None


class EnvStepEvent(RolloutEvent):
    type: Literal["env_step"] = "env_step"
    agent_step_id: str | None = None
    agent_step_index: int | None = None
    env_step_index: int
    reward: float | None = None
    done: bool = False
    error: dict[str, Any] | None = None
    info: dict[str, Any] = Field(default_factory=dict)
    state_ref: str | None = None
    next_state_ref: str | None = None
    step_reward: float | None = None
    discount: float | None = None
    episode_done: bool = False
    bootstrap_required: bool = False
    started_at: float | None = None
    ended_at: float | None = None


class TerminalEvent(RolloutEvent):
    type: Literal["terminal"] = "terminal"
    rollout_status: RolloutStatus
    outcome_success: bool = False
    final_reward: float | None = None
    rollout_valid: bool = False
    trainable: bool = False
    error: dict[str, Any] | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


AnyRolloutEvent = AcceptedEvent | LLMCallEvent | AgentStepEvent | EnvStepEvent | TerminalEvent


def event_context_payload(ctx: EventContext) -> dict[str, Any]:
    return ctx.model_dump()


def dump_for_event(value: Any) -> Any:
    return _jsonable(value)
