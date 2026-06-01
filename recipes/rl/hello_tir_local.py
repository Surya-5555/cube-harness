# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cube-harness[rl]",
#     "tir-cube",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "../..", editable = true }
# tir-cube = { path = "../../cubes/tir", editable = true }
# cube-standard = { path = "../../../cube-standard" }
# ///
"""Minimal local TIR rollout smoke: no HTTP server, prints events as they arrive.

Example usage:

    uv run recipes/rl/hello_tir_local.py --num-rollouts 5

"""

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from tir_cube.agent import TirAgentConfig

from cube_harness.rl import AckRequest, RolloutConfig, RolloutEngine, RolloutRequest, configure_terminal_logging
from cube_harness.rl.llm import RolloutLLMConfig

CLIENT_ID = os.getenv("CUBE_HARNESS_CLIENT_ID", "local-rollout-smoke")
MODEL = os.getenv("CUBE_HARNESS_MODEL", "qwen3_4b")
TOKENIZER_NAME = os.getenv(
    "CUBE_HARNESS_TOKENIZER_NAME", "/home/toolkit/huggingface/base_models/Qwen3-4B-Instruct-2507"
)
LLM_BASE_URL = os.getenv("CUBE_HARNESS_LLM_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("CUBE_HARNESS_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
OUTPUT_DIR = Path(os.getenv("CUBE_HARNESS_ROLLOUT_OUTPUT_DIR", "tmp/cube_harness_results/local_tir"))
MAX_STEPS = int(os.getenv("CUBE_HARNESS_MAX_STEPS", "10"))
MAX_COMPLETION_TOKENS = int(os.getenv("CUBE_HARNESS_MAX_COMPLETION_TOKENS", "16000"))
MAX_TOKENS = int(os.getenv("CUBE_HARNESS_MAX_TOKENS", "32000"))
LOG_LEVEL = os.getenv("CUBE_HARNESS_LOG_LEVEL", "INFO")
SANDBOX_ENDPOINT = os.getenv(
    "CUBE_HARNESS_SANDBOX_ENDPOINT", "http://dns-24e3447c-506e-4b21-92df-156e18db5087-sandboxfusion"
)

_SYSTEM_PROMPT = """You are a math-focused AI Agent. Solve problems by combining clear symbolic reasoning
with short, deterministic Python code.
Keep your replies concise and direct. Prioritize clarity and avoid over-elaboration.
Always present the final answer in LaTeX \\boxed{}.
Do not express emotions or opinions about user questions.

Workflow:
1. Draft a brief plan in plain text.
2. Execute one run_python_code call to compute or verify the result.
3. Finalize by calling MathAnswer with the LaTeX-formatted answer.

Python execution policy (run_python_code):
- Use Python strictly for pure computation to verify and validate the final answer.
- No network, file system, OS or environment access.
- Keep snippets minimal and self-contained; print only the final result.

Validation:
- Cross-check results (alternative derivation, invariants, higher precision) before finalizing.
- If execution fails, propose the minimal fix and retry.
Always verify with run_python_code before invoking MathAnswer."""


def rollout_config() -> RolloutConfig:
    llm_config = RolloutLLMConfig(
        model_name=MODEL,
        temperature=1.0,
        timeout=3600.0,
        num_retries=1,
        tokenizer_name=TOKENIZER_NAME,
    )
    agent = TirAgentConfig(
        llm_config=llm_config,
        system_prompt=_SYSTEM_PROMPT,
        max_actions=3,
    )
    return RolloutConfig(
        name="local_tir_rollout",
        output_dir=OUTPUT_DIR / "episodes",
        persist_rollout=True,
        benchmark_config={
            "_type": "tir_cube.benchmark.TIRBenchmarkConfig",
            "tool_config": {
                "_type": "tir_cube.tool.TIRToolConfig",
                "sandbox_endpoint": SANDBOX_ENDPOINT,
            },
        },
        agent_config=agent.model_dump(mode="json", serialize_as_any=True),
        max_steps=MAX_STEPS,
        execution_mode="local",
    )


def rollout_request(task_id, rollout_index) -> RolloutRequest:
    return RolloutRequest(
        request_id=f"local-tir-{uuid4().hex}",
        client_id=CLIENT_ID,
        task_id=task_id,
        llm_config=RolloutLLMConfig(
            api_base=LLM_BASE_URL,
            model_name=os.getenv("CUBE_HARNESS_SERVED_MODEL_NAME") or MODEL,
            api_key=API_KEY,
            temperature=1.0,
            tokenizer_name=TOKENIZER_NAME,
            max_tokens=MAX_TOKENS,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        ),
        model_version=0,
        group_id="local-tir-smoke",
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
        print(
            f"[{offset}/{event_index}] llm_call tag={event.get('llm_call_tag')!r} "
            f"trainable={event.get('trainable')} tokens={len(event.get('completion_token_ids') or [])}",
            flush=True,
        )
    elif event_type == "tool_call":
        action = event.get("action") or {}
        print(
            f"[{offset}/{event_index}] tool_call index={event.get('tool_call_index')} "
            f"action={action.get('name')} parent={event.get('parent_event_id')}",
            flush=True,
        )
    elif event_type == "evaluation":
        print(
            f"[{offset}/{event_index}] evaluation terminal={event.get('is_terminal')} reward={event.get('reward')}",
            flush=True,
        )
    elif event_type == "agent_error":
        error = event.get("error") or {}
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
                client_id=CLIENT_ID,
                from_offset=0,
                stop_request_id=request.request_id,
                timeout_s=3600.0,
                poll_timeout_s=0.1,
            ):
                print_event(event)
                if event.get("type") == "terminal":
                    await rollout.ack(AckRequest(client_id=CLIENT_ID, offset=event["offset"]))
                await submit_task
    finally:
        rollout.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one TIR rollout in local mode and print events.")
    parser.add_argument("--num-rollouts", type=int, default=1, help="Number of task configs to sample.")
    parser.add_argument(
        "--task-ids", type=lambda s: s.split(","), help="Optional list of task IDs to use (overrides random selection)."
    )
    return parser.parse_args()


def main() -> None:
    configure_terminal_logging(LOG_LEVEL, force=True)
    args = parse_args()
    if args.task_ids:
        print(f"Using specified task IDs: {args.task_ids}")
        args.num_rollouts = len(args.task_ids)

    asyncio.run(run(args.num_rollouts, args.task_ids))


if __name__ == "__main__":
    main()
