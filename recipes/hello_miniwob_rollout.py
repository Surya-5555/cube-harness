# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cube-harness",
#     "miniwob-cube",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "..", editable = true }
# miniwob-cube = { path = "../cubes/miniwob", editable = true }
# cube-standard = { path = "../../cube-standard" }
# cube-browser-tool = { path = "../../cube-standard/cube-tools/cube-browser-tool" }
# ///
"""Reference recipe: MiniWoB through Cube-harness rollout APIs.

Modes:

- HTTP service mode starts FastAPI/Uvicorn, submits POST /rollouts, and reads GET /events.
- Local mode instantiates RolloutEngine directly, submits a rollout, and consumes events without starting HTTP.

For a local vLLM/OpenAI-compatible server, set:

    CUBE_HARNESS_LLM_BASE_URL=http://127.0.0.1:8000 \
    CUBE_HARNESS_MODEL=Qwen3-4B-Instruct-2507 \
    uv run recipes/hello_miniwob_rollout.py --mode http

Use local rollout mode with:

    uv run recipes/hello_miniwob_rollout.py --mode local

CUBE_HARNESS_ROLLOUT_MODE can also set the default mode.

Without CUBE_HARNESS_LLM_BASE_URL, LiteLLM uses the configured model name and
provider credentials from the environment (for example OPENAI_API_KEY).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from uuid import uuid4

import uvicorn

from cube_harness.agents.react_configs import REACT_CONFIGS
from cube_harness.llm import LLMConfig
from cube_harness.rl import AckRequest, RolloutConfig, RolloutEngine, RolloutRequest, configure_terminal_logging, serve, RolloutTaskRunner
from cube_harness.rl.sink import EventSinkConfig

DEFAULT_MODE = os.getenv("CUBE_HARNESS_ROLLOUT_MODE", "local").strip().lower()
HOST = os.getenv("CUBE_HARNESS_ROLLOUT_HOST", "127.0.0.1")
PORT = int(os.getenv("CUBE_HARNESS_ROLLOUT_PORT", "8765"))
BASE_URL = f"http://{HOST}:{PORT}"
CLIENT_ID = os.getenv("CUBE_HARNESS_CLIENT_ID", "hello-miniwob")
TASK_ID = os.getenv("CUBE_HARNESS_MINIWOB_TASK", "click-button")
MODEL = os.getenv("CUBE_HARNESS_MODEL", "gpt-5.4-mini")
LLM_BASE_URL = os.getenv("CUBE_HARNESS_LLM_BASE_URL")
API_KEY = os.getenv("CUBE_HARNESS_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
OUTPUT_DIR = Path(os.getenv("CUBE_HARNESS_ROLLOUT_OUTPUT_DIR", "tmp/cube_harness_results/service_miniwob"))
MAX_STEPS = int(os.getenv("CUBE_HARNESS_MAX_STEPS", "10"))
MINIWOB_PORT = int(os.getenv("CUBE_HARNESS_MINIWOB_PORT", "8011"))
LOG_LEVEL = os.getenv("CUBE_HARNESS_LOG_LEVEL", "INFO")


def _json_request(method: str, url: str, payload: dict | None = None, timeout: float = 10.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return json.loads(body) if body else {}


def _wait_for_health(deadline_s: float = 15.0) -> None:
    deadline = time.monotonic() + deadline_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            health = _json_request("GET", f"{BASE_URL}/health", timeout=1.0)
            if health.get("ready"):
                return
        except Exception as exc:
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"rollout service did not become healthy at {BASE_URL}") from last_error


def _rollout_config(name: str) -> RolloutConfig:
    return RolloutConfig(
        name=name,
        output_dir=OUTPUT_DIR / "episodes",
        benchmark_config=_miniwob_benchmark_cfg(),
        agent_config=_agent_cfg(),
        max_steps=MAX_STEPS,
        execution_mode='local' if DEFAULT_MODE == "local" else "ray",
    )


def _start_service() -> uvicorn.Server:
    app = serve(
        sink_config=EventSinkConfig(
            persist_events_dir=OUTPUT_DIR / "events",
        ),
        config=_rollout_config("service_miniwob"),
    )
    config = uvicorn.Config(app, host=HOST, port=PORT, log_level=LOG_LEVEL.lower())
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="cube-harness-rollout-service", daemon=True)
    thread.start()
    _wait_for_health()
    return server


def _miniwob_benchmark_cfg() -> dict:
    return {
        "_type": "miniwob_cube.benchmark.MiniWobBenchmarkConfig",
        "task_ids": [TASK_ID],
        "port": MINIWOB_PORT,
        "tool_config": {
            "_type": "cube_browser_tool.bgym_tool.BgymToolConfig",
            "use_html": True,
            "use_axtree": False,
            "use_screenshot": False,
        },
    }


def _agent_cfg() -> dict:
    agent = REACT_CONFIGS["default"]
    agent.llm_config = LLMConfig(
        model_name=MODEL,
        temperature=1.0,
        timeout=3600.0,
        num_retries=1,
    )
    agent.max_actions = MAX_STEPS
    return agent.model_dump(mode="json", serialize_as_any=True)


def _llm_payload() -> dict:
    extra_body = {"return_token_ids": True} if LLM_BASE_URL else {}
    return {
        "api_base": LLM_BASE_URL,
        "model_name": os.getenv("CUBE_HARNESS_SERVED_MODEL_NAME") or MODEL,
        "logprobs": os.getenv("CUBE_HARNESS_COLLECT_LOGPROBS", "1") != "0",
        "api_key": API_KEY or "EMPTY",
        "extra_body": extra_body,
        "temperature": 1.0,
        "max_completion_tokens": int(os.getenv("CUBE_HARNESS_MAX_COMPLETION_TOKENS", "2048")),
    }


def _request_payload(mode: str) -> dict:
    request_id = f"hello-miniwob-{uuid4().hex}"
    return {
        "request_id": request_id,
        "client_id": CLIENT_ID,
        "task_id": TASK_ID,
        "llm_config": _llm_payload(),
        "model_version": 0,
        "group_id": f"hello-miniwob-{mode}",
        "rollout_index": 0,
        "max_steps": MAX_STEPS,
    }


def _stream_events(from_offset: int = 0) -> None:
    query = urllib.parse.urlencode({"client_id": CLIENT_ID, "from_offset": from_offset})
    request = urllib.request.Request(f"{BASE_URL}/events?{query}", headers={"Accept": "text/event-stream"})
    event_id: int | None = None
    data_lines: list[str] = []
    last_wait_log = time.monotonic()
    with urllib.request.urlopen(request, timeout=3600.0) as response:
        for raw in response:
            line = raw.decode("utf-8").rstrip("\r\n")
            if not line:
                if data_lines:
                    event = json.loads("\n".join(data_lines))
                    if event_id is not None:
                        event.setdefault("offset", event_id)
                    _print_event(event)
                    if event.get("type") == "terminal":
                        _json_request("POST", f"{BASE_URL}/acks", {"client_id": CLIENT_ID, "offset": event["offset"]})
                        return
                event_id = None
                data_lines = []
                continue
            if line.startswith(":"):
                now = time.monotonic()
                if now - last_wait_log >= 15.0:
                    print("Waiting for rollout events...", flush=True)
                    last_wait_log = now
                continue
            if line.startswith("id:"):
                event_id = int(line.split(":", 1)[1].strip())
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())


def _print_event(event: dict) -> None:
    event_type = event.get("type")
    offset = event.get("offset")
    if event_type == "accepted":
        print(f"[{offset}] Accepted request={event.get('request_id')} task={event.get('task_id')}", flush=True)
    elif event_type == "llm_call":
        print(
            f"[{offset}] LLM call tag={event.get('llm_call_tag')!r} "
            f"trainable={event.get('trainable')} tokens={len(event.get('completion_token_ids') or [])}",
            flush=True,
        )
    elif event_type == "agent_step":
        actions = [a.get("name") for a in event.get("actions") or []]
        print(f"[{offset}] Agent step {event.get('agent_step_index')}: actions={actions}", flush=True)
    elif event_type == "env_step":
        print(f"[{offset}] Env step {event.get('env_step_index')}: reward={event.get('reward')} done={event.get('done')}", flush=True)
    elif event_type == "terminal":
        print(
            f"[{offset}] Terminal status={event.get('rollout_status')} "
            f"valid={event.get('rollout_valid')} trainable={event.get('trainable')} "
            f"reward={event.get('final_reward')} success={event.get('outcome_success')}",
            flush=True,
        )
    else:
        print(f"[{offset}] {event_type}: {event}", flush=True)


def _run_http_mode(payload: dict) -> None:
    server = _start_service()
    try:
        print(f"Cube-harness rollout service: {BASE_URL}", flush=True)
        print(f"Submitting MiniWoB task {TASK_ID!r} with model {MODEL!r} via HTTP", flush=True)
        response = _json_request("POST", f"{BASE_URL}/rollouts", payload)
        print(f"Submit response: {response}", flush=True)
        _stream_events()
    finally:
        server.should_exit = True


async def _run_local_mode_async(payload: dict) -> None:
    rollout = RolloutEngine(
        config=_rollout_config("local_miniwob"),
        sink_config=EventSinkConfig(persist_events_dir=OUTPUT_DIR / "events"),
    )
    try:
        print("Cube-harness local rollout", flush=True)
        print(f"Submitting MiniWoB task {TASK_ID!r} with model {MODEL!r} locally", flush=True)
        request = RolloutRequest.model_validate(payload)
        response = await rollout.submit(request)
        print(f"Submit response: {response}", flush=True)
        async for event in rollout.events(
            client_id=CLIENT_ID,
            from_offset=0,
            stop_request_id=request.request_id,
            timeout_s=3600.0,
        ):
            _print_event(event)
            if event.get("type") == "terminal":
                await rollout.ack(AckRequest(client_id=CLIENT_ID, offset=event["offset"]))
    finally:
        rollout.close()


def _run_local_mode(payload: dict) -> None:
    asyncio.run(_run_local_mode_async(payload))

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a MiniWoB rollout through Cube-harness.")
    parser.add_argument(
        "--mode",
        choices=("http", "local"),
        default=DEFAULT_MODE,
        help="Rollout mode. 'http' starts the service and streams SSE; 'local' streams directly from RolloutEngine.",
    )
    return parser.parse_args()


def main() -> None:
    configure_terminal_logging(LOG_LEVEL, force=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args = _parse_args()
    payload = _request_payload(args.mode)
    if args.mode == "http":
        _run_http_mode(payload)
    elif args.mode == "local":
        _run_local_mode(payload)
    else:
        raise ValueError("mode must be 'http' or 'local'")


if __name__ == "__main__":
    main()
