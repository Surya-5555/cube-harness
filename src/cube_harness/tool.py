"""MonitoredTool — the runtime's instrumented view over a cube-standard `AgentView`.

The agent drives an `AgentView` (cube-standard's agent-facing view of a `Task`:
dynamic `action_set` + `execute_action(action) -> Observation`). All the *task
semantics* — STOP (`final_step` -> `AgentStop`), tool dispatch, `obs_postprocess`,
and tool-error-becomes-observation — live in `AgentView` / `Task`. This module adds
only the *runtime* concerns the harness owns:

  * budget enforcement (`BudgetExceeded`),
  * `ToolCallEvent` emission (trajectory recording),
  * the per-action `finished()` / `evaluate()` cadence (emit `EvaluationEvent`,
    raise `AgentStop` when the task signals done).

`MonitoredTool` exposes the same dual call surface the agent loop expects:

  * `tool.execute_action(action)` — sync.
  * `await tool.async_execute_action(action)` — async (parallel tool calls).
"""

import logging
import time
from typing import Any, Callable

from cube.core import Action, ActionSchema, AgentStop, Observation
from cube.task import AgentView, Task

from cube_harness.budget import Budget, BudgetExceeded
from cube_harness.core import EvaluationEvent, ToolCallEvent, TrajectoryEvent

logger = logging.getLogger(__name__)

# Re-export Budget + BudgetExceeded so existing call sites
# `from cube_harness.tool import Budget` keep working. Canonical home
# is `cube_harness.budget`. `AgentStop` (the clean end-of-episode signal,
# raised by cube-standard's AgentView) is re-exported for the same reason.
__all__ = ["Budget", "BudgetExceeded", "AgentStop", "MonitoredTool", "build_agent_tools"]


class MonitoredTool:
    """Wraps a cube-standard `AgentView` to add budget enforcement, `ToolCallEvent`
    emission, and the per-action `finished()` / `evaluate()` cadence.

    Holds the `AgentView` (the agent surface) and the live `Task` (for `finished` /
    `evaluate` / `validate_per_step`). One instance per agent seat — `agent_id`
    tags its events. Construction is per-episode; re-using across episodes is a bug
    (budget + trajectory are episode-scoped).
    """

    def __init__(
        self,
        agent_view: AgentView,
        task: Task,
        emit: "Callable[[TrajectoryEvent], str]",
        budget: Budget,
        parent_event_id_getter: Callable[[], str] | None = None,
        agent_id: str = "agent",
        role: str | None = None,
    ) -> None:
        self._agent_view = agent_view
        self._task = task
        self._emit = emit
        self._budget = budget
        self._parent_event_id_getter = parent_event_id_getter
        self.agent_id = agent_id
        self.role = role
        self._last_tool_event_id = "no-parent"
        # Per-step eval fires inside AgentView.execute_action and is surfaced out-of-band
        # via this callback; we stash it and emit the EvaluationEvent in `_post_action`,
        # AFTER the ToolCallEvent is recorded, so its `parent_event_id` is the action it
        # scores (and the events stay in causal order in the stream).
        self._pending_eval: tuple[float, dict] | None = None
        self._agent_view.set_eval_callback(self._on_eval)

    def _on_eval(self, reward: float, info: dict) -> None:
        self._pending_eval = (reward, info)

    # --- delegation ---

    @property
    def action_set(self) -> list[ActionSchema]:
        """The actions legal right now — delegates to the `AgentView` (dynamic;
        already includes STOP when the task accepts it)."""
        return self._agent_view.action_set

    # --- monitored execution ---

    def execute_action(self, action: Action) -> Observation:
        """Sync dispatch. Budget -> `AgentView.execute_action` (may raise `AgentStop` on
        STOP) -> record ToolCallEvent -> per-step evaluate/emit + finished check
        (`_post_action`). Returns the observation only."""
        if self._budget.exhausted:
            raise BudgetExceeded(action=action)
        self._pending_eval = None
        start = time.time()
        obs = self._agent_view.execute_action(action)
        end = time.time()
        self._record_tool_call(action, obs, start, end)
        self._post_action(obs)
        return obs

    async def async_execute_action(self, action: Action) -> Observation:
        """Async (parallel-safe) twin of `execute_action`."""
        if self._budget.exhausted:
            raise BudgetExceeded(action=action)
        self._pending_eval = None
        start = time.time()
        obs = await self._agent_view.async_execute_action(action)
        end = time.time()
        self._record_tool_call(action, obs, start, end)
        self._post_action(obs)
        return obs

    # --- recording / cadence ---

    def _parent_event_id(self) -> str:
        if self._parent_event_id_getter is not None:
            value = self._parent_event_id_getter()
            if value is not None:
                return value
        return "no-parent"

    def _record_tool_call(self, action: Action, obs: Observation, start: float, end: float) -> str:
        """Emit one `ToolCallEvent` and bump the budget. The error text is folded into
        `obs` (errors are observations, non-terminal); the *structured* error is also
        recorded on the event for telemetry/stats — `obs.error` carries the StepError
        for the action just dispatched (None when it succeeded)."""
        event = ToolCallEvent(
            parent_event_id=self._parent_event_id(),
            action_id=action.id,
            action=action,
            obs=obs,
            error=obs.error,
            agent_id=self.agent_id,
        )
        self._emit(TrajectoryEvent(output=event, start_time=start, end_time=end))
        self._budget.bump_tool_calls()
        self._last_tool_event_id = event.id
        return event.id

    def _post_action(self, obs: Observation) -> None:
        """The runtime cadence after each action: emit the step-wise `EvaluationEvent` the
        AgentView's eval callback just handed us (when `validate_per_step`), parented to the
        ToolCallEvent just recorded (reward is the runtime's concern, never the agent's — it
        only sees `obs`); then the `finished()` check that ends the episode via `AgentStop`."""
        if self._pending_eval is not None:
            reward, info = self._pending_eval
            self._pending_eval = None
            now = time.time()
            self._emit(
                TrajectoryEvent(
                    output=EvaluationEvent(
                        reward=reward,
                        info=info,
                        is_terminal=False,
                        parent_event_id=self._last_tool_event_id,
                        agent_id=self.agent_id,
                    ),
                    start_time=now,
                    end_time=now,
                )
            )
        if self._task.finished(obs):
            raise AgentStop(obs)


def build_agent_tools(task: Task, streamer: Any) -> list[MonitoredTool]:
    """Build one `MonitoredTool` per agent seat from `task.agent_roles()`.

    Walks the roster (`agent_roles()` -> {role: count}) and grabs one `AgentView` per
    seat via `task.get_agent_view(role)` — called `count` times per role. The seat index
    is the task's internal concern: a multi-agent task overrides `get_agent_view` and
    hands out a distinct view (stable `agent_id`) on each call. Single-agent tasks have
    the default roster `{None: 1}` -> one view (`agent_id` "agent"). All views share the
    one `Task`. `streamer` is an `EventStreamer`; we read `.emit`, `.budget`, and
    `.current_parent_event_id`.
    """
    emit = streamer.emit
    budget = streamer.budget
    parent_event_id_getter = streamer.current_parent_event_id
    tools: list[MonitoredTool] = []
    for role, count in task.agent_roles().items():
        for _ in range(count):
            agent_view = task.get_agent_view(role)
            tools.append(
                MonitoredTool(
                    agent_view,
                    task,
                    emit,
                    budget,
                    parent_event_id_getter,
                    agent_id=agent_view.agent_id,
                    role=agent_view.role,
                )
            )
    return tools
