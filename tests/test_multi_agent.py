"""Tests for the turn-based multi-agent arena (cube_harness.multi_agent).

Drives a real 2-seat cube `Task` (one shared counter, two `AgentView` seats) through
the scheduler + `MultiAgentEpisode`, validating: round-robin, joint budget, the
all-retired-or-exhausted termination, and per-agent reward from the global evaluate().
"""

from cube.container import Container
from cube.core import Action, ActionSchema, Observation
from cube.task import AgentView, Task, TaskConfig, TaskMetadata
from cube.tool import Tool, ToolConfig, tool_action
from pydantic import Field, PrivateAttr

from cube_harness.agent import Agent, AgentConfig
from cube_harness.budget import Budget
from cube_harness.core import AgentOutput, TrajectoryEvent
from cube_harness.multi_agent import MultiAgentEpisode, Seat, run_turn_based
from cube_harness.streamer import EventStreamer
from cube_harness.tool import build_agent_tools


class _SharedCounterTool(Tool):
    def __init__(self) -> None:
        super().__init__()
        self.counter = 0

    @tool_action
    def inc(self) -> str:
        """Bump the shared counter."""
        self.counter += 1
        return f"counter={self.counter}"


class _SharedCounterToolConfig(ToolConfig):
    def make(self, container: Container | None = None) -> _SharedCounterTool:
        return _SharedCounterTool()


class _TwoSeatTask(Task):
    """One shared world (a counter), `n_seats` agent seats over it. finished() when
    the counter reaches `done_at`; evaluate() returns the same scalar to every seat."""

    n_seats: int = 2
    done_at: int = 4
    _next_seat: int = PrivateAttr(default=0)  # internal seat counter (the task owns seat)

    def agent_roles(self) -> dict[str | None, int]:
        return {"player": self.n_seats}

    def get_agent_view(self, role=None) -> AgentView:
        # Shared-world topology: every seat acts through the ONE shared counter tool
        # (self._tool). The task owns the seat index, handed out per get_agent_view call.
        if role is None:
            return super().get_agent_view(None)
        seat = self._next_seat
        self._next_seat += 1
        return AgentView(self, role=role, tool=self._tool, seat=seat)

    def reset(self) -> tuple[Observation, dict]:
        return Observation.from_text("go"), {}

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict]:
        c = self._tool.counter
        return float(c), {"counter": c, "per_agent": {f"player-{i}": float(c) for i in range(self.n_seats)}}

    def finished(self, obs: Observation | None = None) -> bool:
        return self._tool.counter >= self.done_at


class _TwoSeatTaskConfig(TaskConfig):
    metadata: TaskMetadata = Field(default_factory=lambda: TaskMetadata(id="two-seat"))
    tool_config: ToolConfig = Field(default_factory=_SharedCounterToolConfig)
    n_seats: int = 2
    done_at: int = 4

    def make(self, runtime_context: object = None) -> _TwoSeatTask:
        return _TwoSeatTask(
            metadata=TaskMetadata(id="two-seat"),
            tool_config=_SharedCounterToolConfig(),
            n_seats=self.n_seats,
            done_at=self.done_at,
        )


class _ScriptedAgentConfig(AgentConfig):
    n_inc: int = 100

    def make(self, action_set: list[ActionSchema] | None = None, **kwargs: object) -> "Agent":
        return _ScriptedAgent(self, agent_id=str(kwargs.get("agent_id", "agent")))


class _ScriptedAgent(Agent):
    """Emits `inc` until it has done `config.n_inc`, then `final_step`."""

    name = "scripted"
    description = ""
    input_content_types: list[str] = []
    output_content_types: list[str] = []

    def __init__(self, config: _ScriptedAgentConfig, agent_id: str = "agent") -> None:
        super().__init__(config)
        self.agent_id = agent_id
        self._n = 0

    def step(self, obs: Observation) -> AgentOutput:
        self._n += 1
        if self._n > self.config.n_inc:
            return AgentOutput(actions=[Action(id=f"{self.agent_id}-stop", name="final_step", arguments={})])
        return AgentOutput(actions=[Action(id=f"{self.agent_id}-{self._n}", name="inc", arguments={})])


class _FakeStorage:
    def __init__(self) -> None:
        self.events: list[TrajectoryEvent] = []
        self._n = 0

    def save_event(self, te: TrajectoryEvent, trajectory_id: str) -> int:
        n = self._n
        self._n += 1
        self.events.append(te)
        return n


def _streamer(budget: Budget) -> EventStreamer:
    return EventStreamer(trajectory_id="t", storage=_FakeStorage(), budget=budget)


# --- run_turn_based scheduler ------------------------------------------------


def _seats(task, budget: Budget, agents: dict[str, Agent]) -> list[Seat]:
    streamer = _streamer(budget)
    env_tools = build_agent_tools(task, streamer)
    obs, _ = task.reset()
    seats = []
    for et in env_tools:
        agents[et.agent_id].attach_recorder(streamer)
        seats.append(Seat(agent_id=et.agent_id, agent=agents[et.agent_id], env_tool=et, obs=obs))
    return seats


def test_round_robin_retires_one_seat_independently() -> None:
    """p0 stops immediately; p1 keeps going. The arena must retire p0 without
    ending p1's participation."""
    task = _TwoSeatTaskConfig(n_seats=2, done_at=100).make()
    budget = Budget(max_agent_steps=1000)
    p0 = _ScriptedAgent(_ScriptedAgentConfig(n_inc=0), agent_id="player-0")  # stops on first step
    p1 = _ScriptedAgent(_ScriptedAgentConfig(n_inc=3), agent_id="player-1")
    seats = _seats(task, budget, {"player-0": p0, "player-1": p1})
    rounds, exhausted = run_turn_based(seats, budget)
    assert not exhausted
    assert {s.agent_id for s in seats if not s.active} == {"player-0", "player-1"}  # both eventually retired
    assert task._tool.counter == 3  # only player-1's three incs landed


def test_joint_budget_ends_episode_for_everyone() -> None:
    """A shared budget cap stops ALL seats, not just the one that tripped it."""
    task = _TwoSeatTaskConfig(n_seats=2, done_at=1000).make()
    budget = Budget(max_agent_steps=1000, max_tool_calls=2)
    agents = {f"player-{i}": _ScriptedAgent(_ScriptedAgentConfig(n_inc=1000), agent_id=f"player-{i}") for i in range(2)}
    seats = _seats(task, budget, agents)
    _rounds, exhausted = run_turn_based(seats, budget)
    assert exhausted
    assert all(not s.active for s in seats)
    assert task._tool.counter == 2  # exactly the budgeted number of tool calls landed


# --- MultiAgentEpisode end-to-end -------------------------------------------


def test_multi_agent_episode_per_agent_reward() -> None:
    """Homogeneous 2-seat episode runs to finished() and returns a per-agent reward
    for each seat from the global evaluate()."""
    budget = Budget(max_agent_steps=1000)
    ep = MultiAgentEpisode(
        task_config=_TwoSeatTaskConfig(n_seats=2, done_at=4),
        agent_config=_ScriptedAgentConfig(n_inc=1000),
        streamer=_streamer(budget),
    )
    result = ep.run()
    assert set(result.rewards) == {"player-0", "player-1"}
    assert all(r >= 4.0 for r in result.rewards.values())  # counter reached done_at
    assert not result.budget_exhausted
    assert result.rounds >= 2


def test_multi_agent_episode_three_seats() -> None:
    """N>2 works the same — one seat per agent_roles() count."""
    budget = Budget(max_agent_steps=1000)
    ep = MultiAgentEpisode(
        task_config=_TwoSeatTaskConfig(n_seats=3, done_at=6),
        agent_config=_ScriptedAgentConfig(n_inc=1000),
        streamer=_streamer(budget),
    )
    result = ep.run()
    assert set(result.rewards) == {"player-0", "player-1", "player-2"}
