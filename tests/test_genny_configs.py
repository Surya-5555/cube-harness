"""Pin the public shape of GENNY_CONFIGS entries.

This is the only file in the suite that pins the *defaults* of canonical
Genny configs — if these change, downstream recipes silently get a
different agent. Surface the change here so reviewers see it.
"""

from __future__ import annotations

from cube_harness.agents.genny_configs import DEFAULT_MODEL, GENNY_CONFIGS


def test_swe_uses_once_thinking_cadence() -> None:
    """``GENNY_CONFIGS["swe"]`` runs in the "once" thinking cadence:
    ``reasoning_effort="medium"`` + ``interleaved_thinking=False``.

    This is the cadence the Auto-CUBE matrix (2026-05-20 swebench session)
    converged on for swe-bench-style tasks. Per-step thinking via the
    Anthropic interleaved-thinking beta gave no measurable accuracy gain
    on the 4-step Reproduce → Explore → Fix → Verify workflow and roughly
    doubled wall-clock per episode.
    """
    agent = GENNY_CONFIGS["swe"]
    assert agent.llm_config.reasoning_effort == "medium"
    assert agent.llm_config.interleaved_thinking is False
    assert agent.llm_config.model_name == DEFAULT_MODEL


def test_default_keeps_helper_built_llm_config() -> None:
    """``GENNY_CONFIGS["default"]`` keeps ``make_agent_config``'s
    auto-built ``LLMConfig`` — both thinking knobs at their defaults
    (``interleaved_thinking=False``, ``reasoning_effort=None``). Recipes
    that want extended thinking opt in explicitly.

    Pinned here so a future change to ``make_agent_config`` defaults
    surfaces as a test break, not a silent agent-behavior shift.
    """
    agent = GENNY_CONFIGS["default"]
    assert agent.llm_config.interleaved_thinking is False
    assert agent.llm_config.reasoning_effort is None
    assert agent.llm_config.model_name == DEFAULT_MODEL


def test_each_lookup_returns_a_fresh_copy() -> None:
    """``ConfigRegistry`` returns deep copies — mutating one shouldn't
    affect the registry or a second lookup. This is the property recipes
    rely on when they reassign ``agent.llm_config = ...``.
    """
    a = GENNY_CONFIGS["swe"]
    a.llm_config.reasoning_effort = "high"
    b = GENNY_CONFIGS["swe"]
    assert b.llm_config.reasoning_effort == "medium"
