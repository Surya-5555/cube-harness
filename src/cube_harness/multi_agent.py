"""Multi-agent arena — a turn-based scheduler over one shared cube-standard `Task`.

`task.agent_roles()` gives the roster ({role: count}); each (role, seat) yields an
`AgentView` over a single shared `Task` (single-agent = the default `{None: 1}`).
The arena builds one agent per seat (homogeneous v1: one `AgentConfig` parameterized
by `agent_id`), drives them **turn-based** (round-robin, one agent-step per seat per
round), and finalizes per-agent reward from the task's **global** `evaluate()`.

v1 scope (the decisions settled in the RFC):
  * fixed N · turn-based · homogeneous · sync;
  * **joint budget** — one `Budget` shared by every seat's `MonitoredTool`;
  * the episode ends when **all seats are retired OR the budget is exhausted**;
  * per-agent reward comes from `evaluate()` over the joint state — a task maps the
    joint outcome to per-agent reward via `info` (e.g. `info["per_agent"]`).

`async`/`batch` schedulers, heterogeneous per-role configs, and per-agent event
`agent_id` tagging in storage are deferred (tracked in the RFC). The single-agent
`Episode` remains the N=1 fast path and is unchanged.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from cube.core import EnvironmentOutput, Observation
from cube.task import TaskConfig

from cube_harness.agent import Agent, AgentConfig
from cube_harness.budget import Budget, BudgetExceeded
from cube_harness.streamer import EventStreamer
from cube_harness.tool import AgentStop, MonitoredTool, build_agent_tools

logger = logging.getLogger(__name__)


@dataclass
class Seat:
    """One agent occupying one `AgentView` seat over the shared task."""

    agent_id: str
    agent: Agent
    env_tool: MonitoredTool
    obs: Observation
    active: bool = True


@dataclass
class ArenaResult:
    """Outcome of a turn-based episode."""

    rewards: dict[str, float]
    info: dict[str, Any]
    rounds: int
    budget_exhausted: bool = field(default=False)


def run_turn_based(seats: list[Seat], budget: Budget, max_rounds: int = 1000) -> tuple[int, bool]:
    """Round-robin scheduler: each active seat takes one agent-step per round.

    A seat retires on empty actions, a step error, or `AgentStop` (its agent emitted
    STOP / `final_step`, or the task `finished()` after its action). `BudgetExceeded`
    is the **joint** stop — it ends the episode for every seat. Returns
    `(rounds_run, budget_exhausted)`. Stops when all seats are retired, the budget is
    exhausted, or `max_rounds` is hit (runaway guard).
    """
    rounds = 0
    while any(s.active for s in seats) and not budget.exhausted and rounds < max_rounds:
        rounds += 1
        for seat in seats:
            if not seat.active:
                continue
            # The whole per-seat body is budget-guarded: BudgetExceeded can come from
            # the tool dispatch, from on_step() (max_agent_steps), or from inside
            # step() (an LLM call tripping a token/cost cap) — any of them ends the
            # episode for EVERY seat (joint budget).
            try:
                agent_output = seat.agent.step(seat.obs)
                if seat.agent._recorder is not None:
                    seat.agent._recorder.on_step()
                if agent_output.error is not None or not agent_output.actions:
                    seat.active = False
                    continue
                last_obs: Observation | None = None
                for action in agent_output.actions:
                    last_obs = seat.env_tool.execute_action(action)
                if last_obs is not None:
                    seat.obs = last_obs
            except AgentStop:
                seat.active = False
            except BudgetExceeded:
                for other in seats:
                    other.active = False
                return rounds, True
    # Joint budget can also exhaust at the top-of-loop guard (a seat's bump hit the
    # cap); retire everyone so "episode over => all seats retired" holds either way.
    if budget.exhausted:
        for seat in seats:
            seat.active = False
    return rounds, budget.exhausted


class MultiAgentEpisode:
    """Drives N agents over one shared task, turn-based, under a joint budget.

    Thin orchestrator: builds the task, one `MonitoredTool` seat per
    `AgentView` in `task.agent_roles()`, one agent per seat (from a single
    `AgentConfig`), runs the turn-based scheduler, then scores per-agent reward
    from the task's global `evaluate()`.
    """

    def __init__(
        self, task_config: TaskConfig, agent_config: AgentConfig, streamer: EventStreamer, max_rounds: int = 1000
    ) -> None:
        self.task_config = task_config
        self.agent_config = agent_config
        self.streamer = streamer
        self.max_rounds = max_rounds

    def run(self) -> ArenaResult:
        task = self.task_config.make()
        env_tools = build_agent_tools(task, self.streamer)
        obs, _info = task.reset()
        self.streamer.record_reset(EnvironmentOutput(obs=obs))

        seats: list[Seat] = []
        for env_tool in env_tools:
            # role is forwarded for heterogeneous (per-role) agent configs later; today's
            # homogeneous agents ignore it.
            agent = self.agent_config.make(env_tool.action_set, agent_id=env_tool.agent_id, role=env_tool.role)
            agent.attach_recorder(self.streamer)
            seats.append(Seat(agent_id=env_tool.agent_id, agent=agent, env_tool=env_tool, obs=obs))

        rounds, exhausted = run_turn_based(seats, self.streamer.budget, self.max_rounds)

        # Per-agent reward from the task's GLOBAL evaluate(): the task maps the joint
        # outcome to per-agent reward via info["per_agent"]; fall back to the scalar
        # reward for every seat when the task is single-reward.
        reward, info = task.evaluate()
        # Record one terminal EvaluationEvent (global reward + per_agent in info) so the
        # stream carries the outcome. NB v1: all seats share one streamer and only
        # ToolCall/Evaluation events are agent_id-tagged — full per-agent trajectory
        # persistence (per-seat metadata, LLM-event tagging, demux) is deferred.
        self.streamer.record_evaluation(reward, info, is_terminal=True)
        per_agent = info.get("per_agent") if isinstance(info, dict) else None
        rewards = {
            seat.agent_id: float(per_agent[seat.agent_id])
            if per_agent and seat.agent_id in per_agent
            else float(reward)
            for seat in seats
        }
        try:
            task.close()
        except Exception:
            logger.exception("Failed to close task")
        return ArenaResult(
            rewards=rewards, info=info if isinstance(info, dict) else {}, rounds=rounds, budget_exhausted=exhausted
        )
