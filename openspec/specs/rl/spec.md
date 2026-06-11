# RL Rollout System

**Module:** `cube_harness.rl`

## Purpose

Run high-throughput rollout collection for RL trainers while reusing the
cube-harness runtime. This spec covers the whole PR 478 RL surface:

- rollout service, engine, executor, Ray runtime, local mode, and CLI;
- realtime rollout event publishing from the canonical event stream;
- rollout LLM endpoint/configuration and trainable token/logprob metadata;
- optional in-memory vs file-backed storage/debug behavior;
- RL recipes, deterministic smoke, and tests.

RL uses the agent-owned `Episode` runtime and consumes the canonical
`TrajectoryEvent` stream; it does not own a second episode loop or recorder
stack.

```text
RolloutRequest
    |
RolloutEngine / executor (Ray or local)
    |
Episode + EventStreamer
    |
RLEventSink / publisher
    |
trainer
```

Disk persistence is optional debug/replay support, not part of the default RL
hot path.

## Public API

### Service and Engine

`cube_harness.rl.service.serve(config)` exposes the rollout HTTP service.
`RolloutEngine` is the benchmark-scoped runtime behind the service. It loads the
benchmark once, accepts rollout requests, publishes realtime events, and supports
ack/cancel control.

The HTTP service is intended for one trusted trainer client per benchmark-scoped
rollout server. The event stream and acknowledgement cursor are server-global;
run a dedicated service per cube/trainer pair rather than multiplexing trainers
through one process. The service accepts trainer-supplied LLM endpoint and
tokenizer configuration, so deployments must keep it on a trusted network
boundary (for example localhost, a private job network, or an authenticated
control plane). TODO(auth): do not expose it directly to untrusted clients
without adding authentication plus allowlists for endpoint/tokenizer choices.

### `RolloutConfig`

```python
class RolloutConfig(BaseModel):
    name: str = "rollout"
    output_dir: Path
    persist_rollout: bool = False
    benchmark_config: BenchmarkConfig
    agent_config: AgentConfig
    infra: InfraConfig | None = None
    max_steps: int = MAX_STEPS
    execution_mode: Literal["ray", "local"] = "ray"
    ray: RayConfig = RayConfig()
```

`RayConfig` controls Ray execution: `num_workers` (default `1`), `init_kwargs`
(forwarded to `ray.init()`), `task_num_cpus` (default `0.25`), `task_options`,
`event_publisher_options`, and `poll_interval_s` (default `0.05`).

- `execution_mode="ray"` runs rollout tasks as Ray work.
- `execution_mode="local"` is for debugging/tests without Ray scheduling.
- `persist_rollout=False` is the throughput default.

### Request / Control Models

```python
class RolloutRequest(BaseModel):
    request_id: str
    task_id: str
    llm_config: RolloutLLMConfig
    model_version: int | None = None
    group_id: str | None = None
    rollout_index: int = 0
    max_steps: int | None = None
    extras: dict[str, Any] = {}

class AckRequest(BaseModel):
    offset: int

class CancelRequest(BaseModel):
    request_id: str | None = None
    group_id: str | None = None
```

### Event Publisher / Sink

`EventPublisher` (`rl/event_publisher.py`) stores an ordered in-memory event stream for clients
and trainer consumers — offset assignment, ack cursor, keepalives, optional
spill. (Distinct from the structural `EventSink` Protocol in
`cube_harness.streamer`.) `RLEventSink` (`rl/trajectory_sink.py`) is an
`EventStreamer` sink that transforms canonical trajectory events into rollout
payloads, then publishes them to the `EventPublisher`:

- `LLMCallEvent` → `llm_call`
- `ToolCallEvent` → `tool_call`
- `EvaluationEvent` → `evaluation` and terminal summary when terminal
- `AgentErrorEvent` → `agent_error` and terminal failure when needed

Trajectory-derived rollout payloads use a stable envelope plus canonical event
body:

```python
{
    "type": "llm_call" | "tool_call" | "evaluation" | "agent_error",
    "offset": int,                 # assigned by EventPublisher
    "event_index": int,            # per-rollout event order
    "request_id": str,
    "trajectory_id": str,          # equals request_id; unique rollout identity
    "env_name": str | None,
    "task_id": str | None,
    "group_id": str | None,
    "rollout_index": int,
    "model_version": int | None,
    "timestamp": float,
    "event": dict,                # dump_for_event(TrajectoryEvent.output)
    "rl": dict,                   # RL-only annotations, when applicable
    "trajectory_event": dict,     # start_time/end_time when needed
}
```

`event` is the canonical `LLMCallEvent` / `ToolCallEvent` / `EvaluationEvent` /
`AgentErrorEvent` dump and should evolve with `cube_harness.core`. RL-only fields
such as `llm_call_index`, `tool_call_index`, `trainable_call_index`, `trainable`,
and `state_ref` live under `rl`; trainers must not treat them as canonical
trajectory fields. Tool-call timing from the outer `TrajectoryEvent` is carried
under `trajectory_event` when present.

Accepted and terminal events are rollout control events, so they remain flat
`AcceptedEvent` / `TerminalEvent` payloads rather than canonical trajectory-event
wrappers. `request_id` is the unique rollout key. RL sets `trajectory_id` equal
to `request_id` so trainer-side reconstruction and optional persisted episode
artifacts share one collision-free identity; `task_id`, `group_id`,
`rollout_index`, and `model_version` remain metadata fields, not identity.

Trainable LLM calls are selected by tag (`""` and `"act"` by default) and must
carry aligned `event.call.prompt_token_ids`, `event.call.completion_token_ids`,
and `event.call.logprobs`.

### Rollout LLM

Rollout LLM configuration lives in `cube_harness.rl.llm`. `RolloutLLMConfig`
inherits from `cube_harness.llm.LLMConfig`, requires trainer-facing
OpenAI/vLLM endpoint controls, and sets `capture_training_metadata=True`.
`api_base`, `api_key`, and `tokenizer_name` are required because the trainer is
selecting the served policy endpoint and tokenizer for data capture. `api_key`
is secret/redacted at serialization boundaries and must not be written to rollout
configs, episode configs, trajectory events, or logs. The runtime remains the
single `cube_harness.llm.LLM`, which requests and validates token ids/logprobs
when training capture is enabled.

### Task Runner

`RolloutTaskRunner` deep-copies the configured agent, applies the request LLM
override, and runs one normal `Episode`. It passes `trajectory_id=request_id`:

```python
rl_sink = RLEventSink(...)
recorder_config = EventStreamerConfig(extra_sinks=[rl_sink])
Episode(..., recorder_config=recorder_config, write_eval_log=persist_rollout)
```

When `persist_rollout=False`, `InMemoryStorage` satisfies the episode contract
without writing trajectory/debug artifacts. When `persist_rollout=True`,
`FileStorage`, logs, and eval-log artifacts are enabled for debugging/replay
under `output_dir/<request_id>/episodes/<request_id>/`. Debug/replay tooling
that needs rollout grouping must retain or join against the rollout event
envelope, because `group_id`, `rollout_index`, and `model_version` are RL event
fields rather than persisted `TrajectoryMetadata` fields.

## Recipes, Smoke, and Tests

RL examples live under `recipes/rl/`:

- `hello_miniwob_local.py`
- `hello_miniwob_service.py`

Deterministic system smokes live under `scripts/smoke/`:

```bash
uv run scripts/smoke/rl_mock_multiturn_service.py --turns 2
uv run scripts/smoke/rl_ray_rollout.py
uv run scripts/smoke/rl_ray_throughput.py
```

`rl_mock_multiturn_service.py` runs the rollout service in local mode with a
mock benchmark/agent, reconstructs partial trajectory events, validates
trainable metadata, writes JSONL training examples, and prints
`SMOKE OK: rl_mock_multiturn_service` on success. `rl_ray_rollout.py` is the
Ray-backed smoke for real Ray startup, scheduling, event publisher actor wiring, and
cancellation. `rl_ray_throughput.py` preserves the throughput scaling check as
a smoke. Ray coverage is intentionally smoke-only because GitHub-hosted
runners are resource constrained and can make Ray scheduling tests flaky.

Focused unit tests for the PR live in:

- `tests/test_rollout_service.py`

The PR also extends `tests/test_llm.py` (retry strategy + cost fallback for the
shared LLM refactor) and `tests/test_default_agent_run.py` (agent-step counting
in `EventStreamer.summary_stats`).


## Invariants

1. RL rollouts use `Episode` + `EventStreamer`; no RL-owned episode loop.
2. RL consumes `cube_harness.core.TrajectoryEvent`; it must not define duplicate
   LLM/tool/evaluation trajectory event models.
3. Disk persistence is optional. The default rollout hot path must not require
   `FileStorage`, per-episode logs, or eval-log writes.
4. Terminal rollout events are emitted exactly once per request.
5. Publisher failures surface as rollout terminal errors without corrupting the
   canonical episode/event path.
6. Ray rollout tasks must be cancellable by request or group. Stale rollout work
   should not require process restart.
7. Rollout LLM config remains under `cube_harness.rl.llm`; the runtime LLM
   implementation remains unified in `cube_harness.llm`.

## Gotchas

- `RLEventSink` publishes synchronously today so trainers can see partial
  trajectories in real time. Rollout workers configure the publisher as required, so
  publisher failures fail the worker and the executor emits an error terminal
  instead of silently producing a partial trajectory. If publisher latency becomes
  a bottleneck, add an ordered realtime async/actor-backed publisher; do not route
  through disk.
- `rl/events.py` contains rollout control/publisher payloads, not a competing
  trajectory event model.
- TODO(replay-gap): when `EventPublisher` drops hot events without a spill directory,
  clients resuming from older offsets need an explicit gap signal.
- Keep unreleased RL compatibility shims out of the core runtime. Non-RL
  compatibility belongs in the existing `agent`, `episode`, `storage`, and
  `llm` specs.
