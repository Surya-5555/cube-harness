# Tasks: PR 478 RL Rollout System

- [x] Rebase `feat/rl` onto `dev`.
- [x] Remove old RL loop/recorder stack.
- [x] Add rollout service, engine, executor, Ray runtime, and local execution mode.
- [x] Add request, ack, cancel, and publisher event APIs.
- [x] Add `RLEventSink` as an `EventStreamer` sink.
- [x] Make rollout storage optional with `persist_rollout`.
- [x] Keep the LLM runtime unified in `cube_harness.llm`.
- [x] Keep rollout-only LLM configuration in `cube_harness.rl.llm`.
- [x] Add RL recipes.
- [x] Add rollout service tests and throughput opt-in tests.
- [x] Move deterministic RL smoke to `scripts/smoke/rl_mock_multiturn_service.py`.

Validation:

- `uv run pytest tests/test_rollout_service.py tests/test_default_agent_run.py tests/test_monitored_tool_compat.py tests/test_storage_event_layout.py tests/test_cube_episode.py tests/test_core.py tests/test_config_registry.py`
- `uv run scripts/smoke/rl_mock_multiturn_service.py --turns 2`
- `make lint`
