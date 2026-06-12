"""Default Agent.run tests — verify the base class implementation
reproduces today's gym-style loop and integrates with EventStreamer +
MonitoredTool (the runtime view over cube-standard's AgentView).

Uses a real-but-minimal cube `Task` (a counter tool, no LLM, no infra), so
the test drives the actual `task.agent_roles()` -> `AgentView` ->
`MonitoredTool` path and stays fast + deterministic. The structural-parity
check for real cubes lives in `cube test <name>`.
"""

from cube.container import Container
from cube.core import Action, ActionSchema, Observation
from cube.task import Task, TaskMetadata
from cube.tool import Tool, ToolConfig, tool_action

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentOutput, LLMCallEvent, ToolCallEvent, TrajectoryEvent
from cube_harness.streamer import EventStreamer
from cube_harness.tool import AgentStop, Budget, BudgetExceeded, build_agent_tools

# ---------------------------------------------------------------------------
# Mock pieces: a real cube Task whose single action bumps a counter, and that
# reports finished() after N increments.
# ---------------------------------------------------------------------------


class _CounterTool(Tool):
    """Single-action cube tool. `inc` bumps an internal counter."""

    def __init__(self) -> None:
        super().__init__()
        self.counter = 0

    @tool_action
    def inc(self) -> str:
        """Bump the counter."""
        self.counter += 1
        return f"counter={self.counter}"


class _CounterToolConfig(ToolConfig):
    def make(self, container: Container | None = None) -> _CounterTool:
        return _CounterTool()


class _MockTask(Task):
    """Minimal real Task: `finished()` returns True after `done_after_n`
    increments. The agent's `MonitoredTool` polls `finished()` after each
    action and raises `AgentStop` — that's how the loop terminates."""

    done_after_n: int = 3

    def reset(self) -> tuple[Observation, dict]:
        return Observation.from_text("ready"), {}

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict]:
        done = self._tool.counter >= self.done_after_n
        return (1.0 if done else 0.0), {"counter": self._tool.counter}

    def finished(self, obs: Observation | None = None) -> bool:
        return self._tool.counter >= self.done_after_n


def _make_task(done_after_n: int = 3) -> _MockTask:
    return _MockTask(
        metadata=TaskMetadata(id="counter-task"),
        tool_config=_CounterToolConfig(),
        done_after_n=done_after_n,
    )


class _CounterAgentConfig(AgentConfig):
    def make(self, action_set: list[ActionSchema] | None = None, **kwargs) -> "Agent":
        return _CounterAgent(self)


class _CounterAgent(Agent):
    """Scripted "agent" — every step emits one inc action. Mirrors what
    a deterministic debug agent does in a cube's debug.py."""

    name = "counter-agent"
    description = "increments a counter every step"
    input_content_types = ["text/plain"]
    output_content_types = ["application/json"]

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        self.steps_taken = 0

    def step(self, obs: Observation) -> AgentOutput:
        self.steps_taken += 1
        return AgentOutput(
            actions=[Action(id=f"a-{self.steps_taken}", name="inc", arguments={})],
        )


class _FakeStorage:
    """Captures every save_event call + assigns event_nums itself
    (matches the Storage.save_event(event, id) -> int contract)."""

    def __init__(self) -> None:
        self.events: list[tuple[str, int, TrajectoryEvent]] = []
        self._next_num = 0

    def save_event(self, te: TrajectoryEvent, trajectory_id: str) -> int:
        n = self._next_num
        self._next_num += 1
        self.events.append((trajectory_id, n, te))
        return n

    def outputs(self) -> list:
        return [te.output for _, _, te in self.events]


def _setup(task: _MockTask, budget: Budget) -> tuple[EventStreamer, _FakeStorage, object]:
    """Build EventStreamer + storage + the agent-facing env_tool — the way
    Episode does it. Storage owns event numbering. The returned env_tool is a
    `MonitoredTool` over the task's single agent seat; the task keeps its
    concrete tool."""
    storage = _FakeStorage()
    streamer = EventStreamer(trajectory_id="t", storage=storage, budget=budget)
    env_tool = build_agent_tools(task, streamer)[0]
    return streamer, storage, env_tool


# ---------------------------------------------------------------------------
# Default Agent.run shape
# ---------------------------------------------------------------------------


def test_default_run_completes_when_task_signals_done() -> None:
    """task.finished() returning True triggers AgentStop from MonitoredTool;
    Episode catches it normally. The unit test catches here since there's no
    Episode to drive."""

    task = _make_task(done_after_n=3)
    budget = Budget(max_agent_steps=100)
    recorder, storage, env_tool = _setup(task, budget)

    agent = _CounterAgent(_CounterAgentConfig())
    agent.attach_recorder(recorder)
    try:
        agent.run(initial_obs=Observation(), env_tool=env_tool)
    except AgentStop:
        pass  # expected: task.finished() returned True after 3 counter increments

    outputs = storage.outputs()
    n_tool = sum(1 for e in outputs if isinstance(e, ToolCallEvent))
    # MockAgent has no LLM → no LLMCallEvent. Three rounds emit three
    # ToolCallEvents; the 3rd raises AgentStop AFTER recording.
    assert n_tool == 3
    assert task._tool.counter == 3


def test_summary_stats_counts_agent_steps_without_llm_calls() -> None:
    task = _make_task(done_after_n=3)
    budget = Budget(max_agent_steps=10)
    recorder, _storage, env_tool = _setup(task, budget)
    agent = _CounterAgent(_CounterAgentConfig())
    agent.attach_recorder(recorder)

    try:
        agent.run(initial_obs=Observation(), env_tool=env_tool)
    except AgentStop:
        pass

    stats = recorder.summary_stats(duration=1.0, final_reward=0.0)
    assert stats["n_agent_steps"] == 3
    assert stats["total_llm_calls"] == 0


def test_default_run_terminates_on_empty_actions() -> None:
    """An agent that returns empty actions and no error signals 'done'
    — the loop must return without an env step."""

    class _NoopAgent(Agent):
        name = "noop"
        description = ""
        input_content_types = []
        output_content_types = []

        def step(self, obs: Observation) -> AgentOutput:
            return AgentOutput(actions=[])

    task = _make_task(done_after_n=100)
    budget = Budget(max_agent_steps=10)
    recorder, storage, env_tool = _setup(task, budget)
    agent = _NoopAgent(_CounterAgentConfig())
    agent.attach_recorder(recorder)
    agent.run(initial_obs=Observation(), env_tool=env_tool)
    outputs = storage.outputs()
    # No LLM call + empty actions => nothing was emitted by the agent loop.
    assert sum(1 for e in outputs if isinstance(e, LLMCallEvent)) == 0
    assert sum(1 for e in outputs if isinstance(e, ToolCallEvent)) == 0
    assert task._tool.counter == 0


def test_default_run_records_parent_event_id_on_tool_calls() -> None:
    """Each ToolCallEvent must reference either the RESET sentinel
    (LLM-less paths) or a preceding LLMCallEvent."""

    task = _make_task(done_after_n=2)
    budget = Budget(max_agent_steps=10)
    recorder, storage, env_tool = _setup(task, budget)
    agent = _CounterAgent(_CounterAgentConfig())
    agent.attach_recorder(recorder)
    try:
        agent.run(Observation(), env_tool)
    except AgentStop:
        pass

    valid_parents: set[str] = {"reset"}
    for ev in storage.outputs():
        if isinstance(ev, LLMCallEvent):
            valid_parents.add(ev.id)
        elif isinstance(ev, ToolCallEvent):
            assert ev.parent_event_id in valid_parents, (
                "ToolCallEvent.parent_event_id must reference RESET or a preceding LLMCallEvent.id"
            )


def test_default_run_propagates_budget_exceeded() -> None:
    """BudgetExceeded from the MonitoredTool must surface through
    agent.run for Episode to capture."""

    task = _make_task(done_after_n=100)
    budget = Budget(max_agent_steps=100, max_tool_calls=1)
    recorder, storage, env_tool = _setup(task, budget)

    agent = _CounterAgent(_CounterAgentConfig())
    agent.attach_recorder(recorder)
    # The second tool call (turn 2) raises.
    raised: list[BaseException] = []
    try:
        agent.run(initial_obs=Observation(), env_tool=env_tool)
    except BaseException as e:  # noqa: BLE001
        raised.append(e)
    assert any(isinstance(e, BudgetExceeded) for e in raised)
    # At least one full round completed before the budget kicked in.
    assert sum(1 for e in storage.outputs() if isinstance(e, ToolCallEvent)) >= 1
