"""Tests for MonitoredTool — the runtime view over cube-standard's AgentView.

MonitoredTool adds only runtime concerns over `AgentView`: budget enforcement,
`ToolCallEvent` emission, and the per-action `finished()`/`evaluate()` cadence.
The task semantics (STOP, dispatch, obs_postprocess, tool-error-as-observation)
live in `AgentView`/`Task` and are exercised here through the real path.
"""

import asyncio

from cube.container import Container
from cube.core import Action, Observation
from cube.task import STOP_ACTION, Task, TaskMetadata
from cube.tool import Tool, ToolConfig, tool_action

from cube_harness.core import EvaluationEvent, ToolCallEvent, TrajectoryEvent
from cube_harness.streamer import EventStreamer
from cube_harness.tool import AgentStop, Budget, BudgetExceeded, MonitoredTool, build_agent_tools


class _EchoTool(Tool):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @tool_action
    def echo(self, text: str) -> str:
        """Echo the text back."""
        self.calls += 1
        return text

    @tool_action
    def boom(self) -> str:
        """Always raises."""
        raise ValueError("boom")


class _EchoToolConfig(ToolConfig):
    def make(self, container: Container | None = None) -> _EchoTool:
        return _EchoTool()


class _EchoTask(Task):
    """finished() after `done_after_n` echo calls; evaluate() scores on it."""

    done_after_n: int = 99

    def reset(self) -> tuple[Observation, dict]:
        return Observation.from_text("ready"), {}

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict]:
        return (1.0 if self._tool.calls >= self.done_after_n else 0.0), {"calls": self._tool.calls}

    def finished(self, obs: Observation | None = None) -> bool:
        return self._tool.calls >= self.done_after_n


def _make_task(done_after_n: int = 99, validate_per_step: bool = False) -> _EchoTask:
    return _EchoTask(
        metadata=TaskMetadata(id="echo-task"),
        tool_config=_EchoToolConfig(),
        done_after_n=done_after_n,
        validate_per_step=validate_per_step,
    )


class _FakeStorage:
    def __init__(self) -> None:
        self.events: list[TrajectoryEvent] = []
        self._n = 0

    def save_event(self, te: TrajectoryEvent, trajectory_id: str) -> int:
        n = self._n
        self._n += 1
        self.events.append(te)
        return n

    def outputs(self) -> list:
        return [te.output for te in self.events]


def _setup(task: _EchoTask, budget: Budget) -> tuple[MonitoredTool, _FakeStorage]:
    storage = _FakeStorage()
    streamer = EventStreamer(trajectory_id="t", storage=storage, budget=budget)
    env_tool = build_agent_tools(task, streamer)[0]
    return env_tool, storage


def _echo(text: str = "hi") -> Action:
    return Action(id=f"id-{text}", name="echo", arguments={"text": text})


# --- build_agent_tools / delegation -----------------------------------------


def test_build_agent_tools_one_seat_by_default() -> None:
    env_tool, _ = _setup(_make_task(), Budget(max_agent_steps=5))
    assert isinstance(env_tool, MonitoredTool)
    assert env_tool.agent_id == "agent"


def test_action_set_delegates_and_includes_stop() -> None:
    env_tool, _ = _setup(_make_task(), Budget(max_agent_steps=5))
    names = {a.name for a in env_tool.action_set}
    assert {"echo", "boom", STOP_ACTION.name} <= names


# --- execution / recording ---------------------------------------------------


def test_execute_action_dispatches_and_records() -> None:
    task = _make_task()
    env_tool, storage = _setup(task, Budget(max_agent_steps=5))
    obs = env_tool.execute_action(_echo("hello"))
    assert isinstance(obs, Observation)
    assert "hello" in obs.to_markdown()
    assert task._tool.calls == 1
    assert sum(1 for e in storage.outputs() if isinstance(e, ToolCallEvent)) == 1


def test_tool_error_becomes_observation_non_terminal() -> None:
    """A failing action returns an observation (not StepError), and does NOT end
    the episode — only finished()/evaluate() do."""
    env_tool, storage = _setup(_make_task(), Budget(max_agent_steps=5))
    obs = env_tool.execute_action(Action(id="b", name="boom", arguments={}))
    assert isinstance(obs, Observation)
    assert "ValueError" in obs.to_markdown()
    # Recorded a ToolCallEvent; the structured error is preserved for telemetry (the
    # error stays visible in stats) even though it is non-terminal and folded into obs.
    tool_calls = [e for e in storage.outputs() if isinstance(e, ToolCallEvent)]
    assert len(tool_calls) == 1
    assert tool_calls[0].error is not None and tool_calls[0].error.error_type == "ValueError"
    assert tool_calls[0].agent_id == "agent"


def test_stop_action_raises_agent_stop() -> None:
    env_tool, _ = _setup(_make_task(), Budget(max_agent_steps=5))
    try:
        env_tool.execute_action(Action(id="s", name=STOP_ACTION.name, arguments={}))
        raised = False
    except AgentStop:
        raised = True
    assert raised


def test_finished_raises_agent_stop_after_record() -> None:
    task = _make_task(done_after_n=1)
    env_tool, storage = _setup(task, Budget(max_agent_steps=5))
    try:
        env_tool.execute_action(_echo())
        raised = False
    except AgentStop:
        raised = True
    assert raised
    # The action was recorded BEFORE finished() raised.
    assert sum(1 for e in storage.outputs() if isinstance(e, ToolCallEvent)) == 1


def test_validate_per_step_emits_evaluation_event() -> None:
    task = _make_task(validate_per_step=True)
    env_tool, storage = _setup(task, Budget(max_agent_steps=5))
    env_tool.execute_action(_echo())
    evals = [e for e in storage.outputs() if isinstance(e, EvaluationEvent)]
    assert len(evals) == 1
    assert evals[0].is_terminal is False


def test_budget_exhausted_raises_before_dispatch() -> None:
    task = _make_task()
    env_tool, _ = _setup(task, Budget(max_agent_steps=100, max_tool_calls=1))
    env_tool.execute_action(_echo())  # 1st ok
    try:
        env_tool.execute_action(_echo())  # 2nd exceeds
        raised = False
    except BudgetExceeded:
        raised = True
    assert raised
    assert task._tool.calls == 1  # 2nd never dispatched


def test_async_execute_action_dispatches_and_records() -> None:
    task = _make_task()
    env_tool, storage = _setup(task, Budget(max_agent_steps=5))
    obs = asyncio.run(env_tool.async_execute_action(_echo("async")))
    assert isinstance(obs, Observation)
    assert "async" in obs.to_markdown()
    assert task._tool.calls == 1
    assert sum(1 for e in storage.outputs() if isinstance(e, ToolCallEvent)) == 1


def test_concurrent_eval_events_parented_to_distinct_tool_calls() -> None:
    # Parallel tool calls share ONE MonitoredTool (one `_pending_eval` / `_last_tool_event_id`).
    # The per-step EvaluationEvent of each action must be parented to ITS OWN ToolCallEvent —
    # i.e. no cross-action leak from the shared stash. Guards the gather/validate_per_step
    # interleaving (a future `await` inside the stash->record->emit region would break this).
    task = _make_task(validate_per_step=True)
    env_tool, storage = _setup(task, Budget(max_agent_steps=99))

    async def run() -> None:
        await asyncio.gather(*(env_tool.async_execute_action(_echo(f"a{i}")) for i in range(3)))

    asyncio.run(run())
    outs = storage.outputs()
    tool_ids = sorted(e.id for e in outs if isinstance(e, ToolCallEvent))
    eval_parents = sorted(e.parent_event_id for e in outs if isinstance(e, EvaluationEvent))
    assert len(tool_ids) == 3 and len(eval_parents) == 3
    assert eval_parents == tool_ids  # each eval parented to a distinct, correct tool call
