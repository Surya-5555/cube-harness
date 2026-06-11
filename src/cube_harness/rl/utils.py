from __future__ import annotations

from typing import Any

from cube_harness.rl.llm import RolloutLLMConfig


def _openai_model_name(model_name: str) -> str:
    return model_name if model_name.startswith("openai/") else f"openai/{model_name}"


def _normalized_api_base(api_base: str) -> str:
    value = str(api_base).rstrip("/")
    if not value.endswith("/v1"):
        value += "/v1"
    return value


def _rollout_llm_override(rollout_llm: RolloutLLMConfig) -> RolloutLLMConfig:
    override = rollout_llm.model_copy(
        deep=True,
        update={
            "api_base": _normalized_api_base(rollout_llm.api_base),
            "model_name": _openai_model_name(str(rollout_llm.model_name)),
        },
    )
    for name, value in rollout_llm.overrides.items():
        if hasattr(override, name):
            setattr(override, name, value)
    return override


def override_rollout_llm_config(agent_config: Any, rollout_llm: RolloutLLMConfig) -> None:
    """Apply trainer-supplied rollout LLM overrides to an agent config in-place."""
    override = _rollout_llm_override(rollout_llm)
    if hasattr(agent_config, "llm_config"):
        agent_config.llm_config = override
    if getattr(agent_config, "summarize_llm_config", None) is not None:
        agent_config.summarize_llm_config = override.model_copy(deep=True)
