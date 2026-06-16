"""Task layer for bfcl-cube (single-turn AST categories).

One episode = the agent reads a user query plus a per-task set of function
schemas, then emits the function call(s) it would make (in a single turn). The
calls are recorded, not executed; ``evaluate()`` AST-matches them against BFCL's
ground truth (or, for (ir)relevance categories, checks call presence/absence).
"""

from __future__ import annotations

from typing import Any

from cube.benchmark import RuntimeContext
from cube.core import Observation
from cube.task import Task, TaskConfig, TaskExecutionInfo, TaskMetadata

from bfcl_cube._vendor.ast_checker import ast_checker
from bfcl_cube.tool import BfclTool, BfclToolConfig

# Categories scored purely by whether the model (correctly) abstained / called,
# rather than by argument matching. They ship no ground truth.
_IRRELEVANCE_CATEGORIES = frozenset({"irrelevance", "live_irrelevance"})
_RELEVANCE_CATEGORIES = frozenset({"live_relevance"})

# Marker file written last by BfclBenchmarkConfig.install(); its presence means
# the per-task cache is complete. Defined here (not in benchmark.py) so
# verify_installed() can reference it without a circular import.
INSTALL_SENTINEL = ".installed"


class BfclTaskMetadata(TaskMetadata):
    """Lightweight per-task fields (shipped in ``task_metadata.json``)."""

    category: str
    """BFCL category, e.g. ``simple_python`` / ``multiple`` / ``live_parallel``."""

    collection: str
    """BFCL collection: ``non_live`` or ``live`` (drives the named subsets)."""

    language: str = "python"
    """Function-definition language. This cube ships the Python subset only."""


class BfclExecutionInfo(TaskExecutionInfo):
    """Heavy per-task data, loaded on the worker from the install cache."""

    question: list[list[dict[str, str]]]
    """BFCL ``question``: a list of turns, each a list of role/content messages.
    Single-turn tasks have exactly one turn."""

    functions: list[dict[str, Any]]
    """Raw BFCL function schemas offered to the agent for this task."""

    ground_truth: list[dict[str, Any]] | None = None
    """BFCL possible-answer ``ground_truth`` (None for (ir)relevance categories)."""


class BfclTask(Task[BfclTaskMetadata, BfclTool]):
    """A single BFCL single-turn function-calling task."""

    validate_per_step: bool = False

    @property
    def _exec(self) -> BfclExecutionInfo:
        if not isinstance(self.execution_info, BfclExecutionInfo):
            raise RuntimeError(
                f"BfclTask {self.metadata.id!r}: execution_info is "
                f"{type(self.execution_info).__name__}, expected BfclExecutionInfo. "
                f"Construct via BfclTaskConfig.make() so it is populated."
            )
        return self.execution_info

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self.tool.reset()
        prompt = _render_question(self._exec.question)
        return Observation.from_text(prompt), {
            "id": self.metadata.id,
            "category": self.metadata.category,
        }

    def finished(self, obs: Observation | None = None) -> bool:
        # Single-turn BFCL gives the model one response. The episode ends when the
        # agent calls ``final_step`` (AgentStop) — including the abstain case for
        # (ir)relevance — or hits max_steps; the harness evaluates either way.
        # We do not auto-finish on a recorded call so that parallel calls emitted
        # across separate agent steps all accumulate before scoring.
        return False

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict[str, Any]]:
        calls = self.tool.recorded_calls
        category = self.metadata.category

        if category in _IRRELEVANCE_CATEGORIES:
            success = len(calls) == 0
            return float(success), {"category": category, "n_calls": len(calls), "valid": success}
        if category in _RELEVANCE_CATEGORIES:
            success = len(calls) > 0
            return float(success), {"category": category, "n_calls": len(calls), "valid": success}

        model_output = [{c.name: c.arguments} for c in calls]
        result = ast_checker(
            func_description=self._exec.functions,
            model_output=model_output,
            possible_answer=self._exec.ground_truth or [],
            test_category=category,
        )
        reward = 1.0 if result.get("valid") else 0.0
        return reward, {
            "category": category,
            "valid": result.get("valid", False),
            "error": result.get("error"),
            "error_type": result.get("error_type"),
            "model_output": model_output,
        }


class BfclTaskConfig(TaskConfig[BfclTaskMetadata]):
    """Serializable factory producing a :class:`BfclTask`.

    Loads heavy data (question, functions, ground truth) from the per-task
    execution cache written by ``BfclBenchmarkConfig.install()``.
    """

    def verify_installed(self) -> None:
        cache_dir = type(self).task_execution_cache_dir()
        if not (cache_dir / INSTALL_SENTINEL).exists():
            raise RuntimeError(
                f"bfcl-cube per-task execution cache is not installed at {cache_dir}. "
                f"Run `cube install bfcl-cube` (or `BfclBenchmarkConfig.install()`) first."
            )

    def make(self, runtime_context: RuntimeContext | None = None) -> BfclTask:
        self.verify_installed()
        execution_info = BfclExecutionInfo.model_validate(self.load_task_execution_info())
        return BfclTask(
            metadata=self.metadata,
            execution_info=execution_info,
            tool_config=self.tool_config or BfclToolConfig(functions=execution_info.functions),
            runtime_context=runtime_context,
        )


def _render_question(question: list[list[dict[str, str]]]) -> str:
    """Flatten BFCL's ``question`` (turns → messages) into a prompt string.

    Single-turn tasks have one turn. A leading ``system`` message is prefixed;
    user/other messages follow. Role labels are kept only for non-user roles so
    the common single-user-message case reads as a plain query.
    """
    parts: list[str] = []
    for turn in question:
        for msg in turn:
            content = msg.get("content", "")
            role = msg.get("role", "user")
            parts.append(content if role == "user" else f"[{role}] {content}")
    return "\n\n".join(parts)
