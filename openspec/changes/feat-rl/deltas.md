# Deltas: PR 487 RL Rollout System

These deltas describe the full `pull/487/head:feat/rl` PR, not only the event
streaming part of the rollout path.

## ADDED — `openspec/specs/rl/spec.md`

Defines the new RL rollout subsystem:

- `RolloutConfig`, `RolloutRequest`, `AckRequest`, `CancelRequest`, `RayConfig`
- `RolloutEngine`, service, executor, Ray runtime
- `RolloutLLMConfig` / `RolloutLLM`
- `EventPublisher` and `RLEventSink`
- optional storage/debug behavior
- recipes and smoke expectations

## MODIFIED — `openspec/specs/episode/spec.md`

`EpisodeConfig` gains:

- `recorder_config: EventStreamerConfig`
- `write_eval_log: bool = True`

`Episode` passes `recorder_config` to `EventStreamer`. Rollout workers set
`write_eval_log=False` when `persist_rollout=False` to keep debug artifacts off
the hot path.

## MODIFIED — `openspec/specs/storage/spec.md`

`InMemoryStorage` is a valid storage implementation for rollout collection. It
satisfies the episode runtime contract without requiring disk-backed trajectory
storage.

## MODIFIED — `openspec/specs/llm/spec.md`

`cube_harness.llm` owns shared LLM primitives:

- `BaseLLMConfig`
- `BaseLLM`
- `LLMCall` trainable metadata fields:
  `prompt_token_ids`, `completion_token_ids`, `logprobs`, `finish_reason`,
  `metadata`

Rollout-specific endpoint behavior and token/logprob validation live in
`cube_harness.rl.llm`.


## ADDED — `recipes/rl/` and RL smoke

Adds rollout recipes for MiniWoB plus the deterministic smoke:

```bash
uv run scripts/smoke/rl_mock_multiturn_service.py --turns 2
```

The smoke is the PR-level integration check for the rollout service when a
coding agent needs a fast end-to-end validation signal.

## ADDED — rollout service tests and perf helpers

Adds rollout tests and opt-in throughput coverage:

- `tests/test_rollout_service.py`
- `tests/perf/test_rollout_throughput.py`
- `tests/rollout_perf_helpers.py`

## REMOVED / NOT CARRIED FORWARD — old unreleased RL compatibility layer

Do not restore:

- `episode_loop.py`
- `episode_recorders.py`
- `RolloutEventRecorder`
- RL-specific LLM/tool/evaluation trajectory event models
- compatibility shims that only preserve old `feat/rl` API shapes
