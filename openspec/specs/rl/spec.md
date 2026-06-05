# RL Rollout System

**Module:** `cube_harness.rl`

## Purpose

Run high-throughput rollout collection for RL trainers while reusing the
cube-harness runtime. This spec covers the whole PR 487 RL surface:

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
    name: str
    output_dir: Path
    persist_rollout: bool = False
    benchmark_config: BenchmarkConfig
    agent_config: AgentConfig
    infra: InfraConfig | None = None
    max_steps: int = MAX_STEPS
    execution_mode: Literal["ray", "local"] = "ray"
    ray: RayConfig
```

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
    extras: dict = {}

class AckRequest(BaseModel):
    offset: int

class CancelRequest(BaseModel):
    request_id: str | None = None
    group_id: str | None = None
```

### Event Publisher / Sink

`EventPublisher` stores an ordered in-memory event stream for clients and trainer
consumers. `RLEventSink` is an `EventStreamer` sink that transforms canonical
trajectory events into rollout payloads:

- `LLMCallEvent` → `llm_call`
- `ToolCallEvent` → `tool_call`
- `EvaluationEvent` → `evaluation` and terminal summary when terminal
- `AgentErrorEvent` → `agent_error` and terminal failure when needed

Trainable LLM calls are selected by tag (`""` and `"act"` by default) and must
carry aligned `prompt_token_ids`, `completion_token_ids`, and `logprobs`.

### Rollout LLM

Rollout LLM code lives in `cube_harness.rl.llm`. `RolloutLLMConfig` inherits
shared fields from `BaseLLMConfig` and adds trainer-facing OpenAI/vLLM endpoint
controls. `api_base`, `api_key`, and `tokenizer_name` are required because the
trainer is selecting the served policy endpoint and tokenizer for data capture.
`api_key` is secret/redacted at serialization boundaries and must not be written
to rollout configs, episode configs, trajectory events, or logs. `RolloutLLM`
requests and validates token ids/logprobs needed for policy-gradient style
training data.

### Task Runner

`RolloutTaskRunner` deep-copies the configured agent, applies the request LLM
override, and runs one normal `Episode`:

```python
rl_sink = RLEventSink(...)
recorder_config = EventStreamerConfig(
    event_sinks=[rl_sink],
    include_storage_sink=persist_rollout,
)
Episode(..., recorder_config=recorder_config, write_eval_log=persist_rollout)
```

When `persist_rollout=False`, `InMemoryStorage` satisfies the episode contract
without writing trajectory/debug artifacts. When `persist_rollout=True`,
`FileStorage`, logs, and eval-log artifacts are enabled for debugging/replay.

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
Ray-backed smoke for real Ray startup, scheduling, event-sink actor wiring, and
cancellation. `rl_ray_throughput.py` preserves the throughput scaling check as
a smoke. Ray coverage is intentionally smoke-only because GitHub-hosted
runners are resource constrained and can make Ray scheduling tests flaky.

Focused unit tests for the PR live in:

- `tests/test_rollout_service.py`


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
7. Rollout LLM code remains under `cube_harness.rl.llm`; `cube_harness.llm`
   contains shared and benchmark LLM primitives only.

## Gotchas

- `RLEventSink` publishes synchronously today so trainers can see partial
  trajectories in real time. Rollout workers configure the sink as required, so
  publisher failures fail the worker and the executor emits an error terminal
  instead of silently producing a partial trajectory. If publisher latency becomes
  a bottleneck, add an ordered realtime async/actor-backed publisher; do not route
  through disk.
- `rl/events.py` contains rollout control/publisher payloads, not a competing
  trajectory event model.
- TODO(replay-gap): when `EventSink` drops hot events without a spill directory,
  clients resuming from older offsets need an explicit gap signal.
- Keep unreleased RL compatibility shims out of the core runtime. Non-RL
  compatibility belongs in the existing `agent`, `episode`, `storage`, and
  `llm` specs.
