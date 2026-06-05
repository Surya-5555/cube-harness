from __future__ import annotations

from typing import Any

from cube_harness.rl.llm import RolloutLLMConfig


def _openai_model_name(model_name: str) -> str:
    return model_name if model_name.startswith("openai/") else f"openai/{model_name}"


def apply_rollout_llm_config(agent_config: Any, rollout_llm: RolloutLLMConfig) -> None:
    """Apply trainer-supplied rollout LLM overrides to an agent config in-place."""
    llm_config = getattr(agent_config, "llm_config", None)
    if llm_config is None:
        return

    if rollout_llm.api_base and hasattr(llm_config, "api_base"):
        api_base = str(rollout_llm.api_base).rstrip("/")
        if not api_base.endswith("/v1"):
            api_base += "/v1"
        llm_config.api_base = api_base
    if rollout_llm.api_key is not None and hasattr(llm_config, "api_key"):
        llm_config.api_key = rollout_llm.api_key.get_secret_value()
    if rollout_llm.model_name:
        llm_config.model_name = _openai_model_name(str(rollout_llm.model_name))
    if hasattr(llm_config, "tokenizer_name") and rollout_llm.tokenizer_name:
        llm_config.tokenizer_name = rollout_llm.tokenizer_name

    for field_name in (
        "temperature",
        "top_p",
        "top_k",
        "max_completion_tokens",
        "max_tokens",
        "timeout",
    ):
        value = getattr(rollout_llm, field_name, None)
        if value is not None and hasattr(llm_config, field_name):
            setattr(llm_config, field_name, value)

    if rollout_llm.extra_body and hasattr(llm_config, "extra_body"):
        llm_config.extra_body = {**getattr(llm_config, "extra_body", {}), **rollout_llm.extra_body}

    for name, value in rollout_llm.overrides.items():
        if hasattr(llm_config, name):
            setattr(llm_config, name, value)
