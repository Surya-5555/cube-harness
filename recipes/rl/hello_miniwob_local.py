# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cube-harness[rl]",
#     "miniwob-cube",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "../..", editable = true }
# miniwob-cube = { path = "../../cubes/miniwob", editable = true }
# ///
"""Minimal local MiniWoB rollout smoke: no HTTP server, prints events as they arrive.

Example usage:

    uv run recipes/rl/hello_miniwob_local.py --task-ids form-sequence,click-button-sequence,click-checkboxes-large,email-inbox-star-reply

"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import typer

from cube_harness.agents.react_configs import REACT_CONFIGS
from cube_harness.rl import AckRequest, RolloutConfig, RolloutEngine, RolloutRequest, configure_terminal_logging
from cube_harness.rl.llm import RolloutLLMConfig

MODEL = os.getenv("CUBE_HARNESS_MODEL", "qwen36_27b")
TOKENIZER_NAME = os.getenv("CUBE_HARNESS_TOKENIZER_NAME", "/home/toolkit/huggingface/base_models/Qwen3.6-27B")
LLM_BASE_URL = os.getenv("CUBE_HARNESS_LLM_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("CUBE_HARNESS_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
OUTPUT_DIR = Path(os.getenv("CUBE_HARNESS_ROLLOUT_OUTPUT_DIR", "tmp/cube_harness_results/local_miniwob"))
MAX_STEPS = int(os.getenv("CUBE_HARNESS_MAX_STEPS", "10"))
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


def rollout_config() -> RolloutConfig:
    agent = REACT_CONFIGS["default"]
    agent.llm_config = _rollout_llm_config(model_name=MODEL)
    agent.max_actions = MAX_STEPS
    return RolloutConfig(
        name="local_miniwob_rollout",
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
        execution_mode="local",
    )


def rollout_request(task_id, rollout_index) -> RolloutRequest:
    return RolloutRequest(
        request_id=f"local-miniwob-{uuid4().hex}",
        task_id=task_id,
        llm_config=_rollout_llm_config(model_name=os.getenv("CUBE_HARNESS_SERVED_MODEL_NAME") or MODEL),
        model_version=0,
        group_id="local-miniwob-smoke",
        rollout_index=rollout_index,
        max_steps=MAX_STEPS,
    )


def print_event(event: dict[str, Any]) -> None:
    event_type = event.get("type")
    offset = event.get("offset")
    event_index = event.get("event_index")
    if event_type == "accepted":
        print(
            f"[{offset}/{event_index}] accepted request={event.get('request_id')} task={event.get('task_id')}",
            flush=True,
        )
    elif event_type == "llm_call":
        call = llm_call_body(event)
        print(
            f"[{offset}/{event_index}] llm_call tag={call.get('tag')!r} "
            f"trainable={rl_body(event).get('trainable')} tokens={len(call.get('completion_token_ids') or [])}",
            flush=True,
        )
    elif event_type == "tool_call":
        body = event_body(event)
        action = body.get("action") or {}
        print(
            f"[{offset}/{event_index}] tool_call index={rl_body(event).get('tool_call_index')} "
            f"action={action.get('name')} parent={body.get('parent_event_id')}",
            flush=True,
        )
    elif event_type == "evaluation":
        body = event_body(event)
        print(
            f"[{offset}/{event_index}] evaluation terminal={body.get('is_terminal')} reward={body.get('reward')}",
            flush=True,
        )
    elif event_type == "agent_error":
        error = event_body(event).get("error") or {}
        print(
            f"[{offset}/{event_index}] agent_error type={error.get('error_type') or error.get('type')}",
            flush=True,
        )
    elif event_type == "terminal":
        print(
            f"[{offset}/{event_index}] terminal status={event.get('rollout_status')} "
            f"valid={event.get('rollout_valid')} trainable={event.get('trainable')} reward={event.get('final_reward')}",
            flush=True,
        )
    else:
        print(f"[{offset}/{event_index}] {event_type}: {event}", flush=True)


async def run(num_rollouts: int, task_ids: list[str] | None) -> None:
    rollout = RolloutEngine(config=rollout_config())
    task_configs = rollout.task_configs()
    discovered_task_ids = [task["task_id"] for task in task_configs["task_configs"]]
    if task_ids is not None:
        selected_task_ids = [task_id for task_id in task_ids if task_id in discovered_task_ids]
    else:
        selected_task_ids = discovered_task_ids[:num_rollouts]

    if not selected_task_ids:
        raise RuntimeError("No valid task IDs selected")

    try:
        for i, task_id in enumerate(selected_task_ids):
            print(f"Submitting rollout for task_id={task_id}")
            request = rollout_request(task_id, i)
            submit_task = asyncio.create_task(rollout.submit(request))
            async for event in rollout.events(
                from_offset=0,
                stop_request_id=request.request_id,
                timeout_s=3600.0,
                poll_timeout_s=0.1,
            ):
                print_event(event)
                if event.get("type") == "terminal":
                    await rollout.ack(AckRequest(offset=event["offset"]))
                await submit_task
    finally:
        rollout.close()


def main(
    num_rollouts: Annotated[int, typer.Option(help="Number of task configs to sample.")] = 1,
    task_ids: Annotated[
        str | None, typer.Option(help="Optional comma-separated task IDs (overrides random selection).")
    ] = None,
) -> None:
    """Run one MiniWoB rollout in local mode and print events."""
    configure_terminal_logging(LOG_LEVEL, force=True)
    ids = task_ids.split(",") if task_ids else None
    if ids:
        print(f"Using specified task IDs: {ids}")
        num_rollouts = len(ids)

    asyncio.run(run(num_rollouts, ids))


if __name__ == "__main__":
    typer.run(main)
