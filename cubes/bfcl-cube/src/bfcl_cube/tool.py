"""BfclTool — a per-task, data-driven function-calling surface.

Unlike a normal cube tool (a fixed set of ``@tool_action`` methods), BFCL gives
every task its **own** list of functions. So this tool's ``action_set`` is built
at construction time from the task's function schemas; there are no static
action methods. ``execute_action`` simply **records** each call — single-turn
BFCL is AST-scored (the functions are never executed), so the task's
``evaluate()`` reads the recorded calls back and matches them against the ground
truth. ``final_step`` (inherited from ``Tool``) remains the STOP action.
"""

from __future__ import annotations

from typing import Any

from cube.container import Container
from cube.core import Action, ActionSchema, Content, Observation
from cube.tool import Tool, ToolConfig, tool_action

from bfcl_cube._vendor.schema_convert import bfcl_parameters_to_openai, normalize_function_name


class BfclToolConfig(ToolConfig):
    """Serializable factory for :class:`BfclTool`.

    Carries the per-task function schemas (raw BFCL format). ``make()`` builds
    the live tool; the ``container`` argument is unused (single-turn BFCL needs
    no infrastructure).
    """

    functions: list[dict[str, Any]] = []

    def make(self, container: Container | None = None) -> "BfclTool":
        return BfclTool(functions=self.functions)


class BfclTool(Tool):
    """Records the agent's function calls for AST scoring.

    Args:
        functions: the task's BFCL function schemas (raw ``{name, description,
            parameters}`` dicts, BFCL type vocabulary).
    """

    def __init__(self, functions: list[dict[str, Any]]) -> None:
        self._functions = functions
        # Agent-facing schemas: BFCL types → OpenAI/JSON-Schema, names normalised.
        self._schemas = [
            ActionSchema(
                name=normalize_function_name(fn["name"]),
                description=fn.get("description", "") or fn["name"],
                parameters=bfcl_parameters_to_openai(fn["parameters"]),
            )
            for fn in functions
        ]
        self._known_names = {s.name for s in self._schemas}
        self._calls: list[Action] = []

    def reset(self) -> None:
        self._calls = []

    @property
    def action_set(self) -> list[ActionSchema]:
        # Per-task function schemas + inherited STOP action (``final_step``).
        return self._schemas + super().action_set

    def execute_action(self, action: Action) -> Observation:
        """Record a BFCL function call; delegate everything else (``final_step``,
        unknown actions) to the base dispatch."""
        if action.name not in self._known_names:
            return super().execute_action(action)
        self._calls.append(action)
        msg = (
            f"Recorded call to {action.name}({action.arguments}). "
            "Make any other function calls this request requires, then call "
            "final_step to submit. Do not repeat a call you have already made."
        )
        return Observation(contents=[Content.from_data(msg, tool_call_id=action.id)])

    @property
    def recorded_calls(self) -> list[Action]:
        """Function calls recorded this episode, in order (read by ``evaluate``)."""
        return self._calls

    @tool_action
    def final_step(self) -> str:
        """Indicate the task is complete (no further function call is needed)."""
        # Override only to give BFCL-appropriate help text; behaviour (raise
        # AgentStop) is inherited from Tool.final_step.
        return super().final_step()
