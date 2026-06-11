# Feature: PR 478 RL Rollout System

**Status:** DRAFT
**Date:** 2026-06-04
**PR:** `pull/478/head:feat/rl`

## Problem

PR 478 adds the first RL rollout collection path to cube-harness. The feature is
broader than event streaming alone: it introduces a trainer-facing rollout
service, local and Ray execution, rollout-specific LLM metadata capture,
examples, and a deterministic smoke.

cube-harness can already run benchmark episodes, but RL training needs a system
that can:

- accept trainer requests for specific tasks and model endpoints;
- run many rollouts through Ray or a local debug mode;
- stream partial trajectory events in realtime;
- capture token ids/logprobs for trainable LLM calls;
- avoid mandatory disk writes on the rollout hot path;
- cancel stale rollout work.

## Intended Trainer Workflow

An RL trainer uses cube-harness as a rollout producer, not as the optimizer. The
trainer owns sampling policy, grouping, reward shaping, filtering, and gradient
updates. cube-harness owns benchmark setup, task execution, agent/tool/LLM
runtime, and streaming rollout observations back to the trainer.

From the trainer's point of view, the system boundary is:

```text
RL trainer / optimizer process
|-- starts one rollout server per cube/benchmark
|-- serves or selects inference endpoint(s) for RolloutLLMConfig
|-- GET /task-configs from each rollout server
|-- chooses task_ids, group_ids, rollout_index values, model_version
|-- keeps rollout requests in flight per cube
|-- reconstructs trajectories, filters, rewards, writes train data
|
|   cube A rollout server                          cube B rollout server
|   (one cube / benchmark config)                  (one cube / benchmark config)
|   |                                              |
|   |-- GET /events (SSE) -----------------------> |-- GET /events (SSE)
|   |-- POST /rollouts --------------------------> |-- POST /rollouts
|   |-- POST /acks / POST /cancel ---------------> |-- POST /acks / POST /cancel
|   |                                              |
|   |-- FastAPI HTTP/SSE control plane             |-- FastAPI HTTP/SSE control plane
|   |-- RolloutEngine                              |-- RolloutEngine
|   |-- executor: local debug mode or Ray mode      |-- executor: local debug mode or Ray mode
|   |-- RolloutTaskRunner                          |-- RolloutTaskRunner
|   |-- Episode + EventStreamer                     |-- Episode + EventStreamer
|   |-- RLEventSink -> rollout event payloads       |-- RLEventSink -> rollout event payloads
|   `-- optional FileStorage/debug sinks            `-- optional FileStorage/debug sinks
|       when persist_rollout=True                       when persist_rollout=True
|
`-- benchmark/task/tools + inference endpoint(s)
```

The service-oriented flow is:

1. Start one rollout server per cube with `serve(config=RolloutConfig(...))`.
   Each server is benchmark-scoped: it owns one cube/benchmark config, one agent
   config, and one rollout engine. A trainer that samples across multiple cubes
   should run multiple rollout servers and route requests to the appropriate
   server.
2. Query available task configs from each server through `GET /task-configs`.
   The trainer uses this response to choose task ids, build groups, and decide
   how many rollouts to request per task.
3. Open `GET /events?from_offset=...` as an SSE stream for each server
   the trainer is consuming.
4. Submit rollout work with `POST /rollouts`, including `task_id`,
   `llm_config`, `group_id`, `rollout_index`, and optional `max_steps`.
5. Reconstruct partial trajectories from realtime `accepted`, `llm_call`,
   `tool_call`, `evaluation`, `agent_error`, and `terminal` events. Trajectory-derived
   events carry canonical event data under `event` and rollout-only annotations
   such as trainable/index metadata under `rl`.
6. Ack consumed offsets with `POST /acks` so a trainer can resume from the next
   offset it has processed.
7. Convert trainable LLM events into training examples once enough rollout
   members for a task/group are available.
8. Cancel stale work with `POST /cancel` by request or group.

`recipes/rl/hello_miniwob_service.py` is the reference mock-trainer shape. It
starts the service, discovers tasks, submits multiple rollout groups, consumes
SSE events, reconstructs partial trajectories, validates trainable metadata, and
writes one JSONL SFT-style record per trainable LLM call. Token IDs and
logprobs come from the canonical `event.call` dump, while call indices and
`trainable` live under the `rl` annotation:

```text
input_ids = event.call.prompt_token_ids + event.call.completion_token_ids
labels    = [-100] * len(event.call.prompt_token_ids) + event.call.completion_token_ids
reward    = group/trajectory reward
```

`scripts/smoke/rl_mock_multiturn_service.py` is the deterministic local-mode
integration check for this contract. It uses a mock benchmark and mock agent,
requires no live LLM, validates event ordering and token-id metadata,
reconstructs a multi-turn trajectory, and writes JSONL training examples.
`scripts/smoke/rl_ray_rollout.py` is the Ray-backed smoke for real Ray startup,
scheduling, event publisher actor wiring, and cancellation; this coverage is kept out
of default pytest because GitHub-hosted runners are resource constrained.

For tighter local debugging, `recipes/rl/hello_miniwob_local.py` uses
`RolloutEngine` directly in process, without an HTTP server, and prints rollout
events as they arrive.

## Proposal

Add `cube_harness.rl` as a benchmark-scoped rollout service and executor layer,
with companion recipes and smoke coverage.
RL rollouts run normal `Episode`s and publish from the canonical event stream:

```text
RolloutRequest
    |
RolloutEngine / executor
    |
Episode + EventStreamer
    |
RLEventSink (convert) -> EventPublisher (publish)
    |
trainer
```

The whole PR contract includes:

- `cube_harness.rl`: service, engine, executor, Ray runtime, task runner, publisher,
  publisher payloads, utilities, and CLI entrypoint;
- `cube_harness.rl.llm`: rollout LLM config and tokenizer counting for
  OpenAI-compatible trainer endpoints;
- `cube_harness.llm`: unified LLM runtime plus `LLMCall` metadata fields used
  by both benchmark and rollout calls;
- `cube_harness.episode`, `streamer`, and `storage`: extension points that let
  RL attach sinks and avoid disk writes without forking the episode loop;
- `recipes/rl`: local and service examples for MiniWoB;
- `scripts/smoke/rl_mock_multiturn_service.py`: deterministic local-mode end-to-end smoke;
- `scripts/smoke/rl_ray_rollout.py`: Ray-backed scheduling and cancellation smoke;
- `scripts/smoke/rl_ray_throughput.py`: Ray-backed throughput scaling smoke;
- tests covering rollout service behavior, optional storage, and event publishing
  without starting a real Ray cluster.

## HTTP Service

`cube_harness.rl.service.serve(config=...)` returns a FastAPI app around one
`RolloutEngine`. The service is intentionally small: HTTP is only the trainer
control plane and event transport; the actual runtime remains `Episode` plus the
canonical `EventStreamer`.

Endpoints:

- `GET /health`: returns readiness and rollout engine stats.
- `GET /task-configs`: returns benchmark metadata and available task configs.
- `POST /rollouts`: submits one `RolloutRequest`; returns immediately after the
  request is accepted/scheduled.
- `GET /events?from_offset=...`: streams ordered events as Server-Sent
  Events. Empty periods emit keepalives.
- `POST /acks`: records the latest consumed offset for the server-global trainer stream.
- `POST /cancel`: cancels work by `request_id` or `group_id`.

The SSE event `id` is the publisher offset. Event payloads also include rollout
identity fields such as `request_id`, `trajectory_id`, `task_id`, `group_id`,
`rollout_index`, and per-request `event_index`, so trainer clients can both
resume by offset and reconstruct each trajectory independently.

## Throughput Controls

The hot path is designed to stream events without mandatory trajectory files.
The main throughput controls are:

- `RolloutConfig.persist_rollout`: default `False`. Keep this off for training
  collection; enable it only for debugging/replay artifacts.
- `RolloutConfig.execution_mode`: `ray` for concurrent rollout collection,
  `local` for single-process debugging.
- `RolloutConfig.max_steps` and `RolloutRequest.max_steps`: bound rollout length
  globally or per request.
- `RayConfig.num_workers`: default Ray CPU capacity when this process initializes
  Ray.
- `RayConfig.init_kwargs`: extra keyword arguments forwarded to `ray.init()` when
  this process initializes Ray.
- `RayConfig.task_num_cpus`: CPU reservation per rollout task. Lower values allow
  more concurrent rollout tasks when the bottleneck is remote inference or tool
  I/O rather than local CPU.
- `RayConfig.task_options`: extra Ray remote task options for placement,
  resources, runtime env, or scheduling.
- `RayConfig.event_publisher_options`: Ray options for the event publisher actor path.
- `RayConfig.poll_interval_s`: executor polling interval for Ray task
  completion/cancellation bookkeeping.
- `RolloutLLMConfig.timeout`, `num_retries`, `max_tokens`, and
  `max_completion_tokens`: affect inference latency, tail behavior, and memory
  pressure on the inference server.
- Trainer-side submission fanout: the service accepts requests independently;
  the trainer controls how many task/group/rollout requests it keeps in flight.

For high-throughput training, use Ray mode, keep `persist_rollout=False`, stream
from `/events`, ack offsets, and write trainer-side artifacts only after the
trainer has decided which events/trajectories to keep. File storage, per-episode
logs, and eval logs are debug aids, not required training transport.

## Debugging Modes

Use local mode when validating event shape, LLM metadata, prompts, tools, or a
single benchmark integration. Local mode runs `RolloutEngine` and `Episode` in
the same process, can be used without HTTP, and is the easiest place to print or
step through events. The local MiniWoB recipe is the current example.

Use service mode when validating the trainer-facing contract. It exercises the
FastAPI app, SSE framing, ack offsets, cancellation endpoint, and client-side
trajectory reconstruction. `scripts/smoke/rl_mock_multiturn_service.py` is the
fast deterministic smoke for this mode.

Use Ray mode when validating realistic rollout concurrency. Ray mode schedules
stateless rollout tasks and supports cancellation of stale work by request,
group, or client. Debugging Ray mode should usually start with low worker counts
and `persist_rollout=True` for a small run, then switch persistence off once the
runtime behavior is understood.

## Subsystems and Maintenance Boundaries

The RL implementation should stay split along these boundaries:

- **Rollout models** (`rollout.py`): request/control/config schemas. These are
  the trainer contract and should stay small and serializable.
- **Service** (`service.py`): FastAPI HTTP/SSE adapter. It should not contain
  episode, benchmark, or training-example logic.
- **Engine/executor/Ray runtime** (`engine.py`, `executor.py`, `ray_runtime.py`):
  scheduling, lifecycle, cancellation, and result bookkeeping.
- **Task runner** (`task_runner.py`): per-rollout bridge from a request to a
  normal `Episode` run.
- **Event conversion** (`trajectory_sink.py`): `RLEventSink` converts canonical
  `TrajectoryEvent` objects into trainer-facing rollout payloads with a stable
  context envelope, canonical event dump under `event`, and RL-only annotations
  under `rl`. This is not a second trajectory model.
- **Event publisher / payloads** (`event_publisher.py`, `events.py`): `EventPublisher`
  (`rl/event_publisher.py`) holds the ordered in-memory event stream — offset assignment,
  ack cursor, keepalives, and optional spill; `events.py` holds the rollout
  control/publisher payload models. (Note: distinct from the structural
  `EventSink` Protocol in `cube_harness.streamer`.)
- **LLM configuration** (`rl/llm.py`): rollout endpoint fields and tokenizer
  counting. Token id/logprob request and validation live in the unified
  `cube_harness.llm.LLM` training-capture path.
- **Storage/debug** (`storage.py`): optional `InMemoryStorage` / `FileStorage`
  replay and inspection support. Must not be required for rollout publishing.
- **Recipes/smokes** (`recipes/rl`, `scripts/smoke`): executable examples that
  document real trainer usage and catch integration regressions.

Keep new behavior in the narrow subsystem that owns it. Avoid adding trainer
logic to the service, service logic to the publisher, or rollout-specific branching to
the core episode loop.

## Inference Engine Caveat

`RolloutLLMConfig` is exposed through LiteLLM/OpenAI-compatible configuration, but the
current design implicitly assumes vLLM-like behavior for trainable rollout
metadata: token ids, completion logprobs, finish reasons, long timeouts, and
OpenAI-compatible response shapes. Other inference engines may differ in token-id
availability, logprob alignment, tokenizer naming, streaming behavior, or finish
reason fields.

Before claiming backend-agnostic support, test this path against each intended
inference engine and add compatibility checks at the unified `LLM` training-capture boundary. The
core RL event/runtime architecture should stay backend-neutral, but trainable
metadata extraction is currently the highest-risk provider-specific area.

## Non-Goals

- Do not add a second episode loop or recorder stack.
- Do not preserve unreleased old RL API shapes for compatibility.
- Do not require disk-backed trajectory storage for training rollout publishing.
- Do not move cube-standard protocol changes into cube-harness.
- Do not put optimizer/trainer policy logic inside cube-harness.

## Review Focus

- `persist_rollout=False` should avoid `FileStorage`, logs, and eval-log writes.
- RL should consume `TrajectoryEvent` through `RLEventSink`.
- Ray cancellation should kill stale rollout work by request/group/client.
- The unified `LLM` should validate token ids/logprobs needed for training when `capture_training_metadata=True`.
- Recipes and `scripts/smoke/rl_mock_multiturn_service.py` should remain runnable.
