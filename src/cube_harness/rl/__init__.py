from __future__ import annotations

from typing import Any

__all__ = [
    "AckRequest",
    "CancelRequest",
    "RolloutLLMConfig",
    "RayConfig",
    "RolloutEngine",
    "RolloutTaskRunner",
    "RolloutRequest",
    "RolloutService",
    "RolloutConfig",
    "configure_terminal_logging",
    "serve",
]


def __getattr__(name: str) -> Any:
    if name in {
        "AckRequest",
        "CancelRequest",
        "RayConfig",
        "RolloutRequest",
        "RolloutConfig",
    }:
        from cube_harness.rl import rollout

        return getattr(rollout, name)
    if name == "RolloutLLMConfig":
        from cube_harness.rl.llm import RolloutLLMConfig

        return RolloutLLMConfig
    if name == "RolloutEngine":
        from cube_harness.rl import engine

        return getattr(engine, name)
    if name == "RolloutTaskRunner":
        from cube_harness.rl.task_runner import RolloutTaskRunner

        return RolloutTaskRunner
    if name in {"RolloutService", "configure_terminal_logging", "serve"}:
        from cube_harness.rl import service

        return getattr(service, name)
    raise AttributeError(name)
