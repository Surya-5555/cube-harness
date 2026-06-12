# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "aiohttp",
#     "cube-harness[rl]",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "../..", editable = true }
# ///
"""SMOKE: deterministic multi-turn rollout-service metadata path.

This smoke does not call a live LLM. It uses a mock agent and mock task
so the rollout always produces multiple trainable LLM calls. Use it when you want
to verify event ordering, trajectory reconstruction, and token-ID JSONL export
without depending on model capability.

Example:

    uv run scripts/smoke/rl_mock_multiturn_service.py --turns 4

Expected shape for --turns 4:

    4 llm_call events
    5 tool_call events, including the initial reset observation
    1 terminal evaluation event
    4 JSONL SFT records

Prints SMOKE OK|FAIL: rl_mock_multiturn_service  (exit 0|1).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, ClassVar
from uuid import uuid4

import aiohttp
import typer
import uvicorn
from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.core import Action, EnvironmentOutput, Observation
from cube.task import Task, TaskConfig, TaskMetadata
from cube.tool import Tool, ToolConfig, tool_action
from litellm import Message
from pydantic import PrivateAttr

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentOutput
from cube_harness.llm import LLMCall, Prompt, Usage
from cube_harness.rl import AckRequest, RolloutConfig, RolloutRequest, configure_terminal_logging, serve
from cube_harness.rl.llm import RolloutLLMConfig

HOST = os.getenv("CUBE_HARNESS_ROLLOUT_HOST", "127.0.0.1")
PORT = int(os.getenv("CUBE_HARNESS_ROLLOUT_PORT", "8776"))
BASE_URL = f"http://{HOST}:{PORT}"
OUTPUT_DIR = Path(os.getenv("CUBE_HARNESS_ROLLOUT_OUTPUT_DIR", "tmp/cube_harness_results/mock_multiturn"))
LOG_LEVEL = os.getenv("CUBE_HARNESS_LOG_LEVEL", "INFO")


def event_body(event: dict[str, Any]) -> dict[str, Any]:
    body = event.get("event")
    return body if isinstance(body, dict) else {}


def rl_body(event: dict[str, Any]) -> dict[str, Any]:
    body = event.get("rl")
    return body if isinstance(body, dict) else {}


def llm_call_body(event: dict[str, Any]) -> dict[str, Any]:
    call = event_body(event).get("call")
    return call if isinstance(call, dict) else {}


def _rollout_request_payload(request: RolloutRequest) -> dict[str, Any]:
    payload = request.model_dump(mode="json")
    payload["llm_config"] = request.llm_config.model_dump(mode="json")
    payload["llm_config"]["api_key"] = request.llm_config.api_key.get_secret_value()
    return payload


class MockTool(Tool):
    def __init__(self, target_turns: int = 4) -> None:
        self.target_turns = target_turns
        self.turn = 0

    @tool_action
    def advance(self, label: str) -> str:
        """Advance the mock environment by one turn."""
        self.turn += 1
        done = self.turn >= self.target_turns
        return f"turn {self.turn}/{self.target_turns}; done={done}; label={label}"


class MockToolConfig(ToolConfig):
    target_turns: int = 4

    def make(self, container=None) -> MockTool:
        _ = container
        return MockTool(target_turns=self.target_turns)


class MockTask(Task):
    target_turns: int = 4
    _turn: int = PrivateAttr(default=0)

    def reset(self) -> tuple[Observation, dict]:
        self._turn = 0
        tool = getattr(self, "tool", None)
        if hasattr(tool, "turn"):
            tool.turn = 0
        return Observation.from_text("mock multi-turn goal: call advance once per turn"), {"turn": self._turn}

    def step(self, action: Action | list[Action]) -> EnvironmentOutput:
        actions = [action] if isinstance(action, Action) else action
        if len(actions) != 1 or actions[0].name != "advance":
            return EnvironmentOutput(
                obs=Observation.from_text(f"invalid action at turn {self._turn}"),
                reward=0.0,
                done=True,
                info={"turn": self._turn, "expected_action": "advance"},
            )

        self._turn += 1
        tool = getattr(self, "tool", None)
        if hasattr(tool, "turn"):
            tool.turn = self._turn
        done = self._turn >= self.target_turns
        reward = 1.0 if done else 0.0
        return EnvironmentOutput(
            obs=Observation.from_text(f"turn {self._turn}/{self.target_turns}; done={done}"),
            reward=reward,
            done=done,
            info={"turn": self._turn, "action_label": actions[0].arguments.get("label")},
        )

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict]:
        _ = obs
        tool = getattr(self, "tool", None)
        turn = int(getattr(tool, "turn", self._turn))
        return (1.0 if turn >= self.target_turns else 0.0), {"turn": turn}


class MockTaskConfig(TaskConfig):
    target_turns: int = 4

    def make(self, runtime_context=None) -> MockTask:
        return MockTask(
            metadata=self.metadata,
            tool_config=self.tool_config or MockToolConfig(target_turns=self.target_turns),
            runtime_context=runtime_context,
            target_turns=self.target_turns,
        )


class MockBenchmark(Benchmark):
    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass


class MockBenchmarkConfig(BenchmarkConfig):
    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="mock-multiturn",
        version="0.1.0",
        description="Deterministic multi-turn rollout metadata smoke benchmark.",
    )
    task_metadata: ClassVar[dict[str, TaskMetadata]] = {
        "mock_multiturn": TaskMetadata(id="mock_multiturn"),
    }
    task_config_class: ClassVar[type[TaskConfig]] = MockTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = MockBenchmark
    turns: int = 4

    @classmethod
    def install(cls) -> None:
        pass

    def get_task_configs(self):
        yield MockTaskConfig(
            metadata=TaskMetadata(id="mock_multiturn"),
            tool_config=MockToolConfig(target_turns=self.turns),
            target_turns=self.turns,
        )


class MockAgentConfig(AgentConfig):
    turns: int = 4

    def make(self, action_set=None, **kwargs) -> "MockAgent":
        _ = action_set, kwargs
        return MockAgent(config=self)


class MockAgent(Agent):
    name = "MockAgent"
    description = "Deterministic agent that emits one trainable LLM call and one action per turn."
    input_content_types = ["text"]
    output_content_types = ["action"]

    def __init__(self, config: MockAgentConfig):
        super().__init__(config)
        self.turn = 0

    def step(self, obs: Observation) -> AgentOutput:
        _ = obs
        if self.turn >= self.config.turns:
            return AgentOutput(actions=[])
        step_index = self.turn
        self.turn += 1
        prompt_token_ids = [1000 + step_index * 10 + i for i in range(4 + step_index)]
        completion_token_ids = [2000 + step_index * 10 + i for i in range(3)]
        llm_call = LLMCall(
            tag="act",
            llm_config=RolloutLLMConfig(
                model_name="mock", api_base="http://localhost:8000/v1", api_key="EMPTY", tokenizer_name="mock-tokenizer"
            ),
            prompt=Prompt(messages=[{"role": "user", "content": f"turn {step_index}"}], tools=[]),
            output=Message(role="assistant", content=f"advance turn {step_index}"),
            usage=Usage(
                prompt_tokens=len(prompt_token_ids),
                completion_tokens=len(completion_token_ids),
                total_tokens=len(prompt_token_ids) + len(completion_token_ids),
            ),
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=completion_token_ids,
            logprobs=[-0.1, -0.2, -0.3],
            finish_reason="tool_calls",
        )
        if self._recorder is not None:
            self._recorder.on_llm_call(llm_call)
        action = Action(id=f"advance-{step_index}", name="advance", arguments={"label": f"turn-{step_index}"})
        return AgentOutput(actions=[action])


@dataclass
class PartialTrajectory:
    request_id: str
    trajectory_id: str
    events: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    agent_errors: list[dict[str, Any]] = field(default_factory=list)
    terminal: dict[str, Any] | None = None

    def add(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        event_type = event.get("type")
        if event_type == "llm_call":
            self.llm_calls.append(event)
        elif event_type == "tool_call":
            self.tool_calls.append(event)
        elif event_type == "evaluation":
            self.evaluations.append(event)
        elif event_type == "agent_error":
            self.agent_errors.append(event)
        elif event_type == "terminal":
            self.terminal = event

    def complete(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "trajectory_id": self.trajectory_id,
            "events": sorted(self.events, key=lambda event: int(event["event_index"])),
            "llm_calls": sorted(self.llm_calls, key=lambda event: int(event["event_index"])),
            "tool_calls": sorted(self.tool_calls, key=lambda event: int(rl_body(event)["tool_call_index"])),
            "evaluations": sorted(self.evaluations, key=lambda event: int(event["event_index"])),
            "agent_errors": sorted(self.agent_errors, key=lambda event: int(event["event_index"])),
            "terminal": self.terminal,
        }


class TrajectoryReconstructor:
    def __init__(self) -> None:
        self._partials: dict[str, PartialTrajectory] = {}
        self.completed: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def add_event(self, event: dict[str, Any]) -> None:
        request_id = str(event["request_id"])
        trajectory_id = str(event["trajectory_id"])
        partial = self._partials.setdefault(
            trajectory_id,
            PartialTrajectory(request_id=request_id, trajectory_id=trajectory_id),
        )
        partial.add(event)
        if event.get("type") == "terminal":
            await self.completed.put(partial.complete())
            self._partials.pop(trajectory_id, None)


class RolloutEventConsumer:
    def __init__(self, *, turns: int) -> None:
        self.turns = turns
        self.reconstructor = TrajectoryReconstructor()
        self.server: uvicorn.Server | None = None
        self.session: aiohttp.ClientSession | None = None
        self.task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "RolloutEventConsumer":
        self.server = start_service(self.turns)
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        await wait_for_health(self.session)
        self.task = asyncio.create_task(self._consume_events(), name="mock-rollout-event-consumer")
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        if self.session is not None:
            await self.session.close()
        if self.server is not None:
            self.server.should_exit = True

    async def submit(self) -> str:
        if self.session is None:
            raise RuntimeError("consumer has not started")
        request_id = f"mock-{uuid4().hex}"
        request = RolloutRequest(
            request_id=request_id,
            task_id="mock_multiturn",
            llm_config=RolloutLLMConfig(
                model_name="mock", api_base="http://localhost:8000/v1", api_key="EMPTY", tokenizer_name="mock-tokenizer"
            ),
            group_id="mock-group-0",
            rollout_index=0,
            max_steps=self.turns + 2,
        )
        payload = _rollout_request_payload(request)
        async with self.session.post(f"{BASE_URL}/rollouts", json=payload) as response:
            response.raise_for_status()
            await response.json()
        return request_id

    async def wait_trajectory(self, request_id: str, *, timeout_s: float = 10.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            trajectory = await asyncio.wait_for(
                self.reconstructor.completed.get(),
                timeout=max(0.1, min(1.0, deadline - time.monotonic())),
            )
            if trajectory["request_id"] == request_id:
                return trajectory
        raise TimeoutError(f"timed out waiting for trajectory {request_id}")

    async def _consume_events(self) -> None:
        if self.session is None:
            raise RuntimeError("consumer has not started")
        params = {"from_offset": "0"}
        async with self.session.get(
            f"{BASE_URL}/events", params=params, headers={"Accept": "text/event-stream"}
        ) as response:
            response.raise_for_status()
            data_lines: list[str] = []
            async for raw in response.content:
                line = raw.decode("utf-8").rstrip("\r\n")
                if not line:
                    if data_lines:
                        event = json.loads("\n".join(data_lines))
                        await self.reconstructor.add_event(event)
                        await self.session.post(
                            f"{BASE_URL}/acks",
                            json=AckRequest(offset=event["offset"]).model_dump(),
                        )
                        if event.get("type") == "terminal":
                            return
                    data_lines = []
                elif line.startswith("data:"):
                    data_lines.append(line.split(":", 1)[1].strip())


async def wait_for_health(session: aiohttp.ClientSession, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            async with session.get(f"{BASE_URL}/health") as response:
                if response.status == 200 and (await response.json()).get("ready"):
                    return
        except Exception:
            pass
        await asyncio.sleep(0.1)
    raise RuntimeError(f"rollout service did not become healthy at {BASE_URL}")


def rollout_config(turns: int) -> RolloutConfig:
    return RolloutConfig(
        name="mock_multiturn_rollout_smoke",
        output_dir=OUTPUT_DIR / "episodes",
        persist_rollout=False,
        benchmark_config=MockBenchmarkConfig(turns=turns),
        agent_config=MockAgentConfig(turns=turns),
        max_steps=turns + 2,
        execution_mode="local",
    )


def start_service(turns: int) -> uvicorn.Server:
    app = serve(config=rollout_config(turns))
    config = uvicorn.Config(app, host=HOST, port=PORT, log_level=LOG_LEVEL.lower())
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="mock-rollout-service", daemon=True)
    thread.start()
    return server


def training_examples(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    reward = float((trajectory["terminal"] or {}).get("final_reward") or 0.0)
    records: list[dict[str, Any]] = []
    for event in trajectory["llm_calls"]:
        call = llm_call_body(event)
        prompt_token_ids = call["prompt_token_ids"]
        completion_token_ids = call["completion_token_ids"]
        records.append(
            {
                "trajectory_id": trajectory["trajectory_id"],
                "request_id": trajectory["request_id"],
                "task_config_id": event["task_id"],
                "group_id": event["group_id"],
                "rollout_index": event["rollout_index"],
                "step_index": rl_body(event)["llm_call_index"],
                "llm_call_id": call.get("id", event_body(event).get("id")),
                "input_ids": prompt_token_ids + completion_token_ids,
                "labels": [-100] * len(prompt_token_ids) + completion_token_ids,
                "reward": reward,
                "trajectory_reward": reward,
                "raw_trajectory_reward": reward,
                "step_reward": reward,
            }
        )
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def assert_multiturn_metadata(trajectory: dict[str, Any], *, turns: int, jsonl_records: list[dict[str, Any]]) -> None:
    assert len(trajectory["llm_calls"]) == turns, len(trajectory["llm_calls"])
    assert len(trajectory["tool_calls"]) == turns + 1, len(trajectory["tool_calls"])
    assert len(trajectory["evaluations"]) == 1, len(trajectory["evaluations"])
    assert not trajectory["agent_errors"], trajectory["agent_errors"]
    assert len(jsonl_records) == turns, len(jsonl_records)
    assert (trajectory["terminal"] or {}).get("rollout_status") == "completed"
    assert (trajectory["terminal"] or {}).get("trainable") is True

    request_events = trajectory["events"]
    assert [event["event_index"] for event in request_events] == list(range(len(request_events)))
    assert [rl_body(event)["llm_call_index"] for event in trajectory["llm_calls"]] == list(range(turns))
    assert [rl_body(event)["tool_call_index"] for event in trajectory["tool_calls"]] == list(range(turns + 1))
    assert event_body(trajectory["tool_calls"][0])["parent_event_id"] == "reset"
    assert event_body(trajectory["tool_calls"][0])["obs"] is not None
    assert all(event_body(event)["parent_event_id"] for event in trajectory["tool_calls"][1:])
    assert all(event_body(event)["action"] is not None for event in trajectory["tool_calls"][1:])
    assert event_body(trajectory["evaluations"][0])["is_terminal"] is True

    for record in jsonl_records:
        labels = record["labels"]
        input_ids = record["input_ids"]
        assert len(labels) == len(input_ids)
        first_target = next(index for index, value in enumerate(labels) if value != -100)
        assert labels[:first_target] == [-100] * first_target
        assert labels[first_target:] == input_ids[first_target:]


async def run(turns: int, jsonl_path: Path) -> None:
    async with RolloutEventConsumer(turns=turns) as consumer:
        submit_task = asyncio.create_task(consumer.submit(), name="mock-rollout-submit")
        request_id = await submit_task
        trajectory = await consumer.wait_trajectory(request_id)

    records = training_examples(trajectory)
    write_jsonl(jsonl_path, records)
    assert_multiturn_metadata(trajectory, turns=turns, jsonl_records=records)
    print(
        f"SMOKE OK: rl_mock_multiturn_service trajectory={trajectory['trajectory_id']} "
        f"llm_calls={len(trajectory['llm_calls'])} "
        f"tool_calls={len(trajectory['tool_calls'])} evaluations={len(trajectory['evaluations'])} jsonl={jsonl_path}",
        flush=True,
    )


def main(
    turns: Annotated[int, typer.Option(help="Number of turns (>= 2) for the multi-turn metadata check.")] = 4,
) -> None:
    """Run a deterministic multi-turn rollout-service metadata smoke."""
    configure_terminal_logging(LOG_LEVEL, force=True)
    jsonl_path = OUTPUT_DIR / "training_examples.jsonl"
    try:
        if turns < 2:
            raise ValueError("--turns must be >= 2 to test multi-turn metadata")
        asyncio.run(run(turns, jsonl_path))
    except Exception as exc:
        print(f"SMOKE FAIL: rl_mock_multiturn_service {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    typer.run(main)
