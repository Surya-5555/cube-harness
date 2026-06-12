# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "aiohttp",
#     "cube-harness[rl]",
#     "miniwob-cube",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "../..", editable = true }
# miniwob-cube = { path = "../../cubes/miniwob", editable = true }
# ///
"""Async mock RL trainer consuming cube-harness rollout events over HTTP/SSE.

Example usage:

    uv run recipes/rl/hello_miniwob_service.py --num-groups 2 --task-ids form-sequence,click-button-sequence,click-checkboxes-large,email-inbox-star-reply

The generated JSONL contains one token-ID SFT record per trainable LLM call.
Inspect it with, for example:

    uv run python -c 'import json,sys; print(json.dumps(json.loads(open(sys.argv[1]).readline()), indent=2)[:2000])' \
        tmp/cube_harness_results/mock_rl_trainer/training_examples.jsonl

"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import aiohttp
import typer
import uvicorn

from cube_harness.agents.react_configs import REACT_CONFIGS
from cube_harness.rl import RayConfig, RolloutConfig, RolloutRequest, configure_terminal_logging, serve
from cube_harness.rl.llm import RolloutLLMConfig

HOST = os.getenv("CUBE_HARNESS_ROLLOUT_HOST", "127.0.0.1")
PORT = int(os.getenv("CUBE_HARNESS_ROLLOUT_PORT", "8765"))
BASE_URL = f"http://{HOST}:{PORT}"
MODEL = os.getenv("CUBE_HARNESS_MODEL", "qwen36_27b")
TOKENIZER_NAME = os.getenv("CUBE_HARNESS_TOKENIZER_NAME", "/home/toolkit/huggingface/base_models/Qwen3.6-27B")
LLM_BASE_URL = os.getenv("CUBE_HARNESS_LLM_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("CUBE_HARNESS_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
OUTPUT_DIR = Path(os.getenv("CUBE_HARNESS_ROLLOUT_OUTPUT_DIR", "tmp/cube_harness_results/mock_rl_trainer"))
MAX_STEPS = int(os.getenv("CUBE_HARNESS_MAX_STEPS", "20"))
MINIWOB_PORT = int(os.getenv("CUBE_HARNESS_MINIWOB_PORT", "8011"))
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

    def complete_payload(self) -> dict[str, Any]:
        ordered = sorted(self.events, key=lambda item: int(item["event_index"]))
        return {
            "request_id": self.request_id,
            "trajectory_id": self.trajectory_id,
            "events": ordered,
            "llm_calls": sorted(self.llm_calls, key=lambda item: int(item["event_index"])),
            "tool_calls": sorted(self.tool_calls, key=lambda item: int(rl_body(item)["tool_call_index"])),
            "evaluations": sorted(self.evaluations, key=lambda item: int(item["event_index"])),
            "agent_errors": sorted(self.agent_errors, key=lambda item: int(item["event_index"])),
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
            await self.completed.put(partial.complete_payload())
            self._partials.pop(trajectory_id, None)


@dataclass(frozen=True)
class TrajectoryDiscard:
    trajectory_id: str
    reason: str
    component: str
    step_index: int | None = None


@dataclass
class TrainingStats:
    kept_trajectories: int = 0
    discarded_trajectories: int = 0
    emitted_examples: int = 0


@dataclass(frozen=True)
class TrainingGroup:
    group_key: tuple[str, str]
    trajectories: list[dict[str, Any]]


def _token_ids(value: Any) -> list[int] | None:
    if not isinstance(value, list) or not value:
        return None
    result: list[int] = []
    for item in value:
        if not isinstance(item, int):
            return None
        result.append(item)
    return result


def _float_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _event_step_index(event: dict[str, Any]) -> int | None:
    value = rl_body(event).get("llm_call_index")
    return int(value) if isinstance(value, int) else None


def _raw_trajectory_reward(trajectory: dict[str, Any]) -> float | None:
    terminal = trajectory.get("terminal") or {}
    reward = _float_value(terminal.get("final_reward"))
    if reward is not None:
        return reward
    summary = terminal.get("summary") or {}
    return _float_value(summary.get("final_reward")) if isinstance(summary, dict) else None


def _step_reward_by_llm_call(trajectory: dict[str, Any]) -> dict[int, float]:
    del trajectory
    return {}


def _validate_trajectory(trajectory: dict[str, Any]) -> TrajectoryDiscard | None:
    trajectory_id = str(trajectory.get("trajectory_id", "<unknown>"))
    terminal = trajectory.get("terminal") or {}
    status = terminal.get("rollout_status")
    if status not in {"completed", "max_steps"}:
        return TrajectoryDiscard(trajectory_id, f"terminal status is {status}", "terminal")
    if not terminal.get("trainable"):
        return TrajectoryDiscard(trajectory_id, "terminal event marked trajectory non-trainable", "terminal")
    if _raw_trajectory_reward(trajectory) is None:
        return TrajectoryDiscard(trajectory_id, "missing numeric final reward", "terminal")

    for event in trajectory.get("agent_errors") or []:
        return TrajectoryDiscard(trajectory_id, "agent error event present", "agent_error")
    for event in trajectory.get("tool_calls") or []:
        if event_body(event).get("error") is not None:
            return TrajectoryDiscard(trajectory_id, "tool call has an error", "tool_call")

    llm_calls = trajectory.get("llm_calls") or []
    if not llm_calls:
        return TrajectoryDiscard(trajectory_id, "trajectory has no LLM calls", "llm_call")

    for event in llm_calls:
        step_index = _event_step_index(event)
        if not rl_body(event).get("trainable"):
            return TrajectoryDiscard(trajectory_id, "LLM call is marked non-trainable", "llm_call", step_index)
        call = llm_call_body(event)
        prompt_token_ids = _token_ids(call.get("prompt_token_ids"))
        completion_token_ids = _token_ids(call.get("completion_token_ids"))
        if prompt_token_ids is None:
            return TrajectoryDiscard(trajectory_id, "LLM call is missing prompt_token_ids", "llm_call", step_index)
        if completion_token_ids is None:
            return TrajectoryDiscard(trajectory_id, "LLM call is missing completion_token_ids", "llm_call", step_index)
        logprobs = call.get("logprobs")
        if not isinstance(logprobs, list) or len(logprobs) != len(completion_token_ids):
            return TrajectoryDiscard(
                trajectory_id,
                "LLM call has missing or misaligned completion logprobs",
                "llm_call",
                step_index,
            )
    return None


def _training_examples_for_trajectory(
    trajectory: dict[str, Any],
    *,
    trajectory_reward: float,
) -> list[dict[str, Any]]:
    raw_reward = _raw_trajectory_reward(trajectory)
    if raw_reward is None:
        raise ValueError(f"trajectory {trajectory.get('trajectory_id')} is missing final reward")

    step_rewards = _step_reward_by_llm_call(trajectory)
    examples: list[dict[str, Any]] = []
    for event in trajectory["llm_calls"]:
        call = llm_call_body(event)
        prompt_token_ids = _token_ids(call.get("prompt_token_ids"))
        completion_token_ids = _token_ids(call.get("completion_token_ids"))
        if prompt_token_ids is None or completion_token_ids is None:
            raise ValueError(f"trajectory {trajectory.get('trajectory_id')} has incomplete token ids")

        step_index = _event_step_index(event)
        step_reward = step_rewards.get(step_index, trajectory_reward) if step_index is not None else trajectory_reward
        input_ids = prompt_token_ids + completion_token_ids
        labels = [-100] * len(prompt_token_ids) + completion_token_ids
        examples.append(
            {
                "trajectory_id": trajectory["trajectory_id"],
                "request_id": trajectory["request_id"],
                "task_config_id": event.get("task_id"),
                "group_id": event.get("group_id"),
                "rollout_index": event.get("rollout_index", 0),
                "step_index": step_index,
                "llm_call_id": call.get("id", event_body(event).get("id")),
                "llm_call_index": rl_body(event).get("llm_call_index"),
                "trainable_call_index": rl_body(event).get("trainable_call_index"),
                "input_ids": input_ids,
                "labels": labels,
                "reward": trajectory_reward,
                "trajectory_reward": trajectory_reward,
                "raw_trajectory_reward": raw_reward,
                "step_reward": step_reward,
            }
        )
    return examples


async def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return

    def write_records() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    await asyncio.to_thread(write_records)


async def discover_task_configs(session: aiohttp.ClientSession) -> dict[str, Any]:
    async with session.get(f"{BASE_URL}/task-configs") as response:
        response.raise_for_status()
        payload = await response.json()
    task_ids = [task["task_id"] for task in payload["task_configs"]]
    print(
        f"trainer: discovered cube={payload['benchmark']['name']} with {len(task_ids)} tasks={task_ids[:5]}{'...' if len(task_ids) > 5 else ''}",
        flush=True,
    )
    return payload


def _rollout_llm_config(*, model_name: str) -> RolloutLLMConfig:
    return RolloutLLMConfig(
        api_base=LLM_BASE_URL,
        api_key=API_KEY,
        model_name=model_name,
        temperature=1.0,
        timeout=3600.0,
        num_retries=1,
        tokenizer_name=TOKENIZER_NAME,
        max_completion_tokens=int(os.getenv("CUBE_HARNESS_MAX_COMPLETION_TOKENS", "2048")),
    )


def _rollout_request_payload(request: RolloutRequest) -> dict[str, Any]:
    payload = request.model_dump(mode="json")
    payload["llm_config"] = request.llm_config.model_dump(mode="json")
    payload["llm_config"]["api_key"] = request.llm_config.api_key.get_secret_value()
    return payload


async def submit_rollout(session: aiohttp.ClientSession, rollout_index: int, group_id: str, task_id: str) -> str:
    request_id = f"mock-trainer-{uuid4().hex}"
    request = RolloutRequest(
        request_id=request_id,
        task_id=task_id,
        llm_config=_rollout_llm_config(model_name=os.getenv("CUBE_HARNESS_SERVED_MODEL_NAME") or MODEL),
        model_version=0,
        group_id=group_id,
        rollout_index=rollout_index,
        max_steps=MAX_STEPS,
    )
    payload = _rollout_request_payload(request)

    async with session.post(f"{BASE_URL}/rollouts", json=payload) as response:
        response.raise_for_status()
        await response.json()
    return request_id


async def mock_training_loop(
    consumer: "RolloutEventConsumer",
    *,
    expected: int,
    num_groups: int,
    jsonl_path: Path,
) -> None:
    stats = TrainingStats()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    processed_groups: set[tuple[str, str]] = set()
    received = 0
    group_queue: asyncio.Queue[TrainingGroup | None] = asyncio.Queue(maxsize=8)
    writer = asyncio.create_task(
        _training_example_writer(group_queue, jsonl_path=jsonl_path, stats=stats),
        name="training-example-writer",
    )

    try:
        if jsonl_path.exists():
            await asyncio.to_thread(jsonl_path.unlink)

        while received < expected:
            if writer.done():
                writer.result()

            try:
                trajectory = await asyncio.wait_for(consumer.completed.get(), timeout=0.25)
            except asyncio.TimeoutError:
                print("trainer: doing optimizer/accounting work while rollouts run", flush=True)
                continue

            received += 1
            terminal = trajectory.get("terminal") or {}
            task_config_id = str(terminal.get("task_id") or "<unknown-task>")
            group_id = str(terminal.get("group_id") or trajectory.get("request_id") or "<unknown-group>")
            group_key = (task_config_id, group_id)
            grouped.setdefault(group_key, []).append(trajectory)

            print(
                f"trainer: received trajectory={trajectory['trajectory_id']} task_config_id={task_config_id} "
                f"group_id={group_id} rollout_index={terminal.get('rollout_index')} "
                f"llm_calls={len(trajectory.get('llm_calls') or [])} raw_reward={_raw_trajectory_reward(trajectory)}",
                flush=True,
            )

            if len(grouped[group_key]) < num_groups or group_key in processed_groups:
                continue

            processed_groups.add(group_key)
            await group_queue.put(TrainingGroup(group_key=group_key, trajectories=list(grouped[group_key])))
            print(
                f"trainer: queued training group task_config_id={task_config_id} group_id={group_id} "
                f"queue_depth={group_queue.qsize()}",
                flush=True,
            )

        await group_queue.join()
        await group_queue.put(None)
        await writer
    finally:
        if not writer.done():
            writer.cancel()
            try:
                await writer
            except asyncio.CancelledError:
                pass

    print(
        f"trainer: JSONL written to {jsonl_path} examples={stats.emitted_examples} "
        f"kept_trajectories={stats.kept_trajectories} discarded_trajectories={stats.discarded_trajectories}",
        flush=True,
    )


async def _training_example_writer(
    queue: asyncio.Queue[TrainingGroup | None],
    *,
    jsonl_path: Path,
    stats: TrainingStats,
) -> None:
    while True:
        group = await queue.get()
        try:
            if group is None:
                return
            await _emit_training_examples_for_group(
                group_key=group.group_key,
                trajectories=group.trajectories,
                jsonl_path=jsonl_path,
                stats=stats,
            )
        finally:
            queue.task_done()


async def _emit_training_examples_for_group(
    *,
    group_key: tuple[str, str],
    trajectories: list[dict[str, Any]],
    jsonl_path: Path,
    stats: TrainingStats,
) -> None:
    task_config_id, group_id = group_key
    raw_rewards = [_raw_trajectory_reward(trajectory) for trajectory in trajectories]
    numeric_rewards = [reward for reward in raw_rewards if reward is not None]
    if not numeric_rewards:
        group_reward = 0.0
        print(
            f"trainer: group task_config_id={task_config_id} group_id={group_id} has no numeric rewards; using 0.0",
            flush=True,
        )
    else:
        group_reward = sum(numeric_rewards) / len(numeric_rewards)

    members = [
        {
            "trajectory_id": trajectory["trajectory_id"],
            "rollout_index": (trajectory.get("terminal") or {}).get("rollout_index"),
            "raw_reward": reward,
        }
        for trajectory, reward in zip(trajectories, raw_rewards, strict=True)
    ]
    print(
        f"trainer: group task_config_id={task_config_id} group_id={group_id} members={members} "
        f"group_reward={group_reward}",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    for trajectory in trajectories:
        discard = _validate_trajectory(trajectory)
        if discard is not None:
            stats.discarded_trajectories += 1
            print(
                f"trainer: discarded trajectory={discard.trajectory_id} reason={discard.reason} "
                f"component={discard.component} step_index={discard.step_index}",
                flush=True,
            )
            continue

        examples = _training_examples_for_trajectory(trajectory, trajectory_reward=group_reward)
        for example in examples:
            print(
                f"trainer: emitting example trajectory={example['trajectory_id']} "
                f"step_index={example['step_index']} group_id={example['group_id']} "
                f"raw_reward={example['raw_trajectory_reward']} trajectory_reward={example['trajectory_reward']} "
                f"step_reward={example['step_reward']}",
                flush=True,
            )
        records.extend(examples)
        stats.kept_trajectories += 1
        stats.emitted_examples += len(examples)

    await _append_jsonl(jsonl_path, records)


async def wait_for_health(session: aiohttp.ClientSession, timeout_s: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            async with session.get(f"{BASE_URL}/health") as response:
                if response.status == 200 and (await response.json()).get("ready"):
                    return
        except Exception as exc:
            last_error = exc
        await asyncio.sleep(0.1)
    raise RuntimeError(f"rollout service did not become healthy at {BASE_URL}") from last_error


def rollout_config() -> RolloutConfig:
    agent = REACT_CONFIGS["default"]
    agent.llm_config = _rollout_llm_config(model_name=MODEL)
    agent.max_actions = MAX_STEPS
    ray_config = RayConfig(
        num_workers=4,
        task_num_cpus=0.25,
    )

    return RolloutConfig(
        name="mock_rl_trainer_miniwob",
        output_dir=OUTPUT_DIR / "episodes",
        persist_rollout=True,
        benchmark_config={
            "_type": "miniwob_cube.benchmark.MiniWobBenchmarkConfig",
            "port": MINIWOB_PORT,
            "tool_config": {
                "_type": "cube_browser_tool.bgym_tool.BgymToolConfig",
                "use_html": True,
                "use_axtree": False,
                "use_screenshot": False,
            },
        },
        agent_config=agent,
        max_steps=MAX_STEPS,
        execution_mode="ray",
        ray=ray_config,
    )


class RolloutEventConsumer:
    """Owns the rollout service and reconstructs trajectories from its SSE stream."""

    def __init__(self, *, expected_terminals: int) -> None:
        self.expected_terminals = expected_terminals
        self.reconstructor = TrajectoryReconstructor()
        self.completed = self.reconstructor.completed
        self.server: uvicorn.Server | None = None
        self.session: aiohttp.ClientSession | None = None
        self.task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "RolloutEventConsumer":
        self.server = start_service()
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        await wait_for_health(self.session)
        self.task = asyncio.create_task(self._consume_events(), name="rollout-event-consumer")
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

    async def discover_task_configs(self) -> dict[str, Any]:
        if self.session is None:
            raise RuntimeError("consumer has not started")
        return await discover_task_configs(self.session)

    async def submit_rollout(self, rollout_index: int, group_id: str, task_id: str) -> str:
        if self.session is None:
            raise RuntimeError("consumer has not started")
        return await submit_rollout(self.session, rollout_index, group_id, task_id)

    async def wait_finished(self) -> None:
        if self.task is not None:
            await self.task

    async def _consume_events(self) -> None:
        if self.session is None:
            raise RuntimeError("consumer has not started")
        completed = 0
        params = {"from_offset": "0"}
        async with self.session.get(
            f"{BASE_URL}/events",
            params=params,
            headers={"Accept": "text/event-stream"},
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
                            json={"offset": event["offset"]},
                        )
                        if event.get("type") == "terminal":
                            completed += 1
                            if completed >= self.expected_terminals:
                                return
                    data_lines = []
                elif line.startswith("data:"):
                    data_lines.append(line.split(":", 1)[1].strip())


def start_service() -> uvicorn.Server:
    app = serve(config=rollout_config())
    config = uvicorn.Config(app, host=HOST, port=PORT, log_level=LOG_LEVEL.lower())
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="cube-harness-rollout-service", daemon=True)
    thread.start()
    return server


async def run(num_rollouts: int, num_groups: int, task_ids: list[str] | None, jsonl_path: Path) -> None:
    if num_rollouts < 1:
        raise ValueError("--num-rollouts must be >= 1")
    if num_groups < 1:
        raise ValueError("--num-groups must be >= 1")

    total_rollouts = num_rollouts * num_groups

    async with RolloutEventConsumer(expected_terminals=total_rollouts) as consumer:
        task_configs = await consumer.discover_task_configs()
        discovered_task_ids = [task["task_id"] for task in task_configs["task_configs"]]

        if task_ids is not None:
            selected_task_ids = [task_id for task_id in task_ids if task_id in discovered_task_ids]
        else:
            if not discovered_task_ids:
                raise RuntimeError("No task configs discovered")

            selected_task_ids = [discovered_task_ids[i % len(discovered_task_ids)] for i in range(num_rollouts)]

        if not selected_task_ids:
            raise RuntimeError("No valid task IDs selected")

        print(
            f"trainer: selected {len(selected_task_ids)} task config(s); "
            f"submitting {num_groups} rollout(s) per task config",
            flush=True,
        )

        trainer = asyncio.create_task(
            mock_training_loop(
                consumer,
                expected=total_rollouts,
                num_groups=num_groups,
                jsonl_path=jsonl_path,
            ),
            name="mock-training-loop",
        )

        submissions = []
        for task_slot, task_id in enumerate(selected_task_ids):
            group_id = f"mock-trainer-task-{task_slot}-{task_id}"
            for rollout_index in range(num_groups):
                submissions.append(consumer.submit_rollout(rollout_index, group_id, task_id))

        request_ids = await asyncio.gather(*submissions)

        print(
            f"trainer: submitted {len(request_ids)} rollout request(s); training JSONL target={jsonl_path}",
            flush=True,
        )

        await asyncio.gather(consumer.wait_finished(), trainer)


def main(
    num_rollouts: Annotated[int, typer.Option(help="Number of task configs to sample.")] = 1,
    num_groups: Annotated[int, typer.Option(help="Number of rollouts per selected task config.")] = 1,
    task_ids: Annotated[
        str | None, typer.Option(help="Optional comma-separated task IDs (overrides random selection).")
    ] = None,
) -> None:
    """Run an async mock RL trainer against cube-harness rollouts."""
    configure_terminal_logging(LOG_LEVEL, force=True)
    jsonl_path = OUTPUT_DIR / "training_examples.jsonl"
    ids = task_ids.split(",") if task_ids else None
    if ids:
        num_rollouts = len(ids)
    asyncio.run(run(num_rollouts, num_groups, ids, jsonl_path))


if __name__ == "__main__":
    typer.run(main)
