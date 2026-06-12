from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from cube.benchmark import BenchmarkConfig
from cube.resource import InfraConfig
from pydantic import BaseModel, Field, SerializeAsAny

from cube_harness.agent import AgentConfig
from cube_harness.episode import MAX_STEPS
from cube_harness.rl.llm import RolloutLLMConfig


class RolloutRequest(BaseModel):
    request_id: str
    task_id: str
    llm_config: RolloutLLMConfig
    model_version: int | None = None
    group_id: str | None = None
    rollout_index: int = 0
    max_steps: int | None = None
    extras: dict[str, Any] = Field(default_factory=dict)


class AckRequest(BaseModel):
    offset: int


class CancelRequest(BaseModel):
    request_id: str | None = None
    group_id: str | None = None


class RayConfig(BaseModel):
    """Pydantic Ray settings for stateless rollout tasks."""

    num_workers: int = Field(default=1, ge=1, description="Default Ray CPU capacity when this process initializes Ray.")
    task_num_cpus: float = Field(default=0.25, gt=0, description="Ray CPU reservation for each rollout task.")
    init_kwargs: dict[str, Any] = Field(default_factory=dict)
    task_options: dict[str, Any] = Field(default_factory=dict)
    event_publisher_options: dict[str, Any] = Field(default_factory=dict)
    poll_interval_s: float = Field(default=0.05, gt=0)


class RolloutConfig(BaseModel):
    """Pydantic config for one benchmark-scoped rollout engine."""

    name: str = "rollout"
    output_dir: Path = Field(description="Root directory for optional rollout debug artifacts.")
    persist_rollout: bool = Field(
        default=False,
        description="When true, write rollout config and per-episode logs to output_dir.",
    )
    benchmark_config: SerializeAsAny[BenchmarkConfig]
    agent_config: SerializeAsAny[AgentConfig]
    infra: SerializeAsAny[InfraConfig] | None = None
    max_steps: int = MAX_STEPS
    execution_mode: Literal["ray", "local"] = "ray"
    ray: RayConfig = Field(default_factory=RayConfig)
