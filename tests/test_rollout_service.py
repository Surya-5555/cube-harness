from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest
from cube.core import Action, Observation, StepError
from fastapi.testclient import TestClient
from litellm import Message

from cube_harness.agents.react_configs import REACT_CONFIGS
from cube_harness.core import AgentErrorEvent, EvaluationEvent, LLMCallEvent, ToolCallEvent, TrajectoryEvent
from cube_harness.llm import LLM, LLMCall, LLMConfig, Prompt, Usage
from cube_harness.rl import RayConfig, RolloutConfig, RolloutEngine, RolloutRequest, serve
from cube_harness.rl.events import EventContext
from cube_harness.rl.llm import RolloutLLMConfig
from cube_harness.rl.trajectory_sink import RLEventSink
from cube_harness.streamer import EventStreamer, EventStreamerConfig
from tests.conftest import MockAgentConfig, MockCubeBenchmarkConfig


def _rollout_llm_request_config() -> RolloutLLMConfig:
    return RolloutLLMConfig(
        model_name="served-model",
        api_base="http://localhost:8000/v1",
        api_key="EMPTY",
        tokenizer_name="mock-tokenizer",
    )


def _event_body(event: dict) -> dict:
    return event["event"]


def _rl_body(event: dict) -> dict:
    return event["rl"]


def _llm_call_body(event: dict) -> dict:
    call = _event_body(event).get("call")
    assert isinstance(call, dict)
    return call


def test_ray_config_defaults_to_fractional_rollout_cpu() -> None:
    assert RayConfig().task_num_cpus == 0.25


def test_rollout_config_preserves_pydantic_config_types(tmp_dir) -> None:
    config = RolloutConfig(
        name="rollout_test",
        output_dir=tmp_dir,
        benchmark_config=MockCubeBenchmarkConfig(),
        agent_config=MockAgentConfig(name="typed_agent"),
        max_steps=1,
    )

    payload = json.loads(config.model_dump_json(serialize_as_any=True))
    restored = RolloutConfig.model_validate(payload)

    assert payload["benchmark_config"]["_type"].endswith("MockCubeBenchmarkConfig")
    assert payload["agent_config"]["_type"].endswith("MockAgentConfig")
    assert isinstance(restored.benchmark_config, MockCubeBenchmarkConfig)
    assert isinstance(restored.agent_config, MockAgentConfig)
    assert restored.agent_config.name == "typed_agent"


def test_rollout_request_redacts_api_key_but_worker_payload_keeps_transient_secret(tmp_dir) -> None:
    request = RolloutRequest(
        request_id="secret-request",
        task_id="mock_cube_task_1",
        llm_config=RolloutLLMConfig(
            model_name="served-model",
            api_base="http://localhost:8000/v1",
            api_key="super-secret",
            tokenizer_name="mock-tokenizer",
        ),
    )

    dumped = request.model_dump(mode="json")
    assert "super-secret" not in json.dumps(dumped)
    assert "api_key" not in dumped["llm_config"]

    rollout = RolloutEngine(
        config=RolloutConfig(
            name="secret_payload_test",
            output_dir=tmp_dir,
            benchmark_config=MockCubeBenchmarkConfig(),
            agent_config=MockAgentConfig(),
            max_steps=1,
            execution_mode="local",
        )
    )
    try:
        payload = rollout._rollout_payload(request)
        assert payload["request"]["llm_config"]["api_key"] == "super-secret"
    finally:
        rollout.close()


def test_rollout_service_runs_native_episode_from_service_benchmark(tmp_dir) -> None:
    config = RolloutConfig(
        name="rollout_test",
        output_dir=tmp_dir,
        benchmark_config=MockCubeBenchmarkConfig(),
        agent_config=MockAgentConfig(),
        max_steps=2,
        execution_mode="local",
    )
    app = serve(config=config)
    service = app.state.service

    try:
        health = service.health()
        assert health["ready"] is True
        assert health["benchmark"]["name"] == "mock-cube"
        assert health["ray"]["initialized"] is False
        assert health["executor"]["inflight_rollouts"] == 0
        assert health["persist_rollout"] is False
        assert not (tmp_dir / "rollout_config.json").exists()

        request = RolloutRequest(
            request_id="request-1",
            task_id="mock_cube_task_1",
            llm_config=_rollout_llm_request_config(),
            rollout_index=2,
        )

        async def run_rollout() -> None:
            await service.submit(request)
            for _ in range(100):
                if any(event["type"] == "terminal" for event in service.events_from(0)):
                    return
                await asyncio.sleep(0.1)
            raise AssertionError("rollout did not emit a terminal event")

        asyncio.run(run_rollout())

        events = service.events_from(0)
        accepted = [event for event in events if event["type"] == "accepted"]
        terminals = [event for event in events if event["type"] == "terminal"]

        assert len(accepted) == 1
        assert accepted[0]["request_id"] == "request-1"
        assert accepted[0]["trajectory_id"] == "request-1"
        assert len(terminals) == 1
        assert terminals[0]["rollout_status"] == "completed"
        assert terminals[0]["env_name"] == "mock-cube"
        assert terminals[0]["trajectory_id"] == "request-1"
        episode_dir = tmp_dir / "request-1" / "episodes" / "request-1"
        assert not (episode_dir / "episode.log").exists()
        assert not (episode_dir / "episode.metadata.json").exists()
        assert not (episode_dir / "events").exists()
        assert not (episode_dir / "steps").exists()
    finally:
        service.close()


def test_rollout_debug_persistence_is_opt_in(tmp_dir) -> None:
    config = RolloutConfig(
        name="rollout_debug_test",
        output_dir=tmp_dir,
        benchmark_config=MockCubeBenchmarkConfig(),
        agent_config=MockAgentConfig(),
        max_steps=2,
        persist_rollout=True,
        execution_mode="local",
    )
    app = serve(config=config)
    service = app.state.service

    try:
        assert service.health()["persist_rollout"] is True
        assert (tmp_dir / "rollout_config.json").exists()

        request = RolloutRequest(
            request_id="debug-request",
            task_id="mock_cube_task_1",
            llm_config=_rollout_llm_request_config(),
        )

        async def run_rollout() -> None:
            await service.submit(request)
            for _ in range(100):
                if any(event["type"] == "terminal" for event in service.events_from(0)):
                    return
                await asyncio.sleep(0.1)
            raise AssertionError("rollout did not emit a terminal event")

        asyncio.run(run_rollout())
        episode_dir = tmp_dir / "debug-request" / "episodes" / "debug-request"
        assert (episode_dir / "episode.log").exists()
        assert (episode_dir / "episode.metadata.json").exists()
        assert (episode_dir / "events").exists()
    finally:
        service.close()


def test_rollout_debug_persistence_allows_same_task_index_across_groups(tmp_dir) -> None:
    config = RolloutConfig(
        name="rollout_collision_test",
        output_dir=tmp_dir,
        benchmark_config=MockCubeBenchmarkConfig(),
        agent_config=MockAgentConfig(),
        max_steps=2,
        persist_rollout=True,
        execution_mode="local",
    )
    rollout = RolloutEngine(config=config)

    try:
        requests = [
            RolloutRequest(
                request_id="group-a-request",
                task_id="mock_cube_task_1",
                llm_config=_rollout_llm_request_config(),
                group_id="group-a",
                rollout_index=0,
            ),
            RolloutRequest(
                request_id="group-b-request",
                task_id="mock_cube_task_1",
                llm_config=_rollout_llm_request_config(),
                group_id="group-b",
                rollout_index=0,
            ),
        ]

        async def run_rollouts() -> None:
            for request in requests:
                await rollout.submit(request)

        asyncio.run(run_rollouts())

        terminals = [event for event in rollout.events_from(0) if event["type"] == "terminal"]
        assert {event["trajectory_id"] for event in terminals} == {"group-a-request", "group-b-request"}
        assert {event["group_id"] for event in terminals} == {"group-a", "group-b"}
        assert all(event["task_id"] == "mock_cube_task_1" for event in terminals)
        assert all(event["rollout_index"] == 0 for event in terminals)
        for request in requests:
            episode_dir = tmp_dir / request.request_id / "episodes" / request.request_id
            assert (episode_dir / "episode.metadata.json").exists()
            assert (episode_dir / "episode_config.json").exists()
            with open(episode_dir / "episode.metadata.json") as f:
                metadata = json.load(f)["metadata"]
            assert metadata["task_id"] == request.task_id
    finally:
        rollout.close()


def test_rollout_streaming_mode_does_not_construct_file_storage(tmp_dir) -> None:
    config = RolloutConfig(
        name="rollout_streaming_storage_test",
        output_dir=tmp_dir,
        benchmark_config=MockCubeBenchmarkConfig(),
        agent_config=MockAgentConfig(),
        max_steps=2,
        execution_mode="local",
    )
    rollout = RolloutEngine(config=config)

    try:
        request = RolloutRequest(
            request_id="streaming-no-file-storage",
            task_id="mock_cube_task_1",
            llm_config=_rollout_llm_request_config(),
        )

        async def run_rollout() -> list[dict]:
            with patch("cube_harness.rl.task_runner.FileStorage", side_effect=AssertionError("FileStorage hot path")):
                await rollout.submit(request)
                async for _event in rollout.events(
                    from_offset=0,
                    stop_request_id=request.request_id,
                    timeout_s=10.0,
                ):
                    pass
            return rollout.events_from(0)

        events = asyncio.run(run_rollout())
        terminal = [event for event in events if event["type"] == "terminal"][-1]
        assert terminal["rollout_status"] == "completed"
        assert not (tmp_dir / "streaming-no-file-storage" / "episodes").exists()
    finally:
        rollout.close()


def test_event_streamer_publishes_rl_events_without_storage_sink() -> None:
    class BombStorage:
        def save_event(self, event: TrajectoryEvent, trajectory_id: str) -> None:
            raise AssertionError("storage sink should be disabled")

    published: list[dict] = []
    rl_sink = _rl_sink(published)
    event_streamer_config = EventStreamerConfig(extra_sinks=[rl_sink])
    streamer = EventStreamer(
        trajectory_id="task_ep0",
        storage=BombStorage(),
    )
    streamer._sinks.extend(event_streamer_config.extra_sinks)

    streamer.emit(TrajectoryEvent(output=LLMCallEvent(call=_llm_call(tag="summary"))))
    streamer.emit(TrajectoryEvent(output=EvaluationEvent(reward=1.0, is_terminal=True)))

    assert [event["type"] for event in published] == ["llm_call", "evaluation", "terminal"]
    assert published[-1]["rollout_status"] == "completed"


def test_unreleased_rl_streamer_config_compat_fields_are_removed() -> None:
    assert "rl_event_sink" not in EventStreamerConfig.model_fields
    assert "rl_event_publisher" not in EventStreamerConfig.model_fields
    assert "rl_event_context" not in EventStreamerConfig.model_fields
    assert "trainable_call_tags" not in EventStreamerConfig.model_fields
    assert "extra_sinks" in EventStreamerConfig.model_fields


def test_rollout_service_exposes_task_configs(tmp_dir) -> None:
    config = RolloutConfig(
        name="rollout_test",
        output_dir=tmp_dir,
        benchmark_config=MockCubeBenchmarkConfig(),
        agent_config=MockAgentConfig(),
        max_steps=1,
        execution_mode="local",
    )
    app = serve(config=config)
    service = app.state.service

    try:
        payload = service.task_configs()
        assert payload["benchmark"] == {"name": "mock-cube", "task_count": 2}
        assert [task["task_id"] for task in payload["task_configs"]] == ["mock_cube_task_1", "mock_cube_task_2"]
        assert payload["task_configs"][0]["metadata"]["id"] == "mock_cube_task_1"
        assert "config" not in payload["task_configs"][0]

        with TestClient(app) as client:
            response = client.get("/task-configs")
        assert response.status_code == 200
        assert response.json() == payload
    finally:
        service.close()


def test_rollout_health_reports_local_capacity(tmp_dir) -> None:
    config = RolloutConfig(
        name="rollout_test",
        output_dir=tmp_dir,
        benchmark_config=MockCubeBenchmarkConfig(),
        agent_config=MockAgentConfig(),
        max_steps=2,
        execution_mode="local",
    )
    app = serve(config=config)
    service = app.state.service

    try:
        health = service.health()
        assert health["ready"] is True
        assert health["ray"]["initialized"] is False
        assert health["ray"]["configured_num_workers"] == 1
        assert health["ray"]["task_num_cpus"] == 0.25
        assert health["ray"]["cluster_cpus"] == 0.0
        assert health["ray"]["estimated_rollout_slots"] == 0
        assert health["executor"]["inflight_rollouts"] == 0
        assert health["executor"]["cancelled_rollouts"] == 0
        assert health["event_publisher"]["next_offset"] == 0
        assert health["event_publisher"]["oldest_available_offset"] == 0
        assert health["event_publisher"]["max_hot_events"] > 0
    finally:
        service.close()


def test_rollout_streams_events_without_http_service(tmp_dir) -> None:
    config = RolloutConfig(
        name="rollout_test",
        output_dir=tmp_dir,
        benchmark_config=MockCubeBenchmarkConfig(),
        agent_config=MockAgentConfig(),
        max_steps=2,
        execution_mode="local",
    )
    rollout = RolloutEngine(config=config)

    try:
        request = RolloutRequest(
            request_id="request-direct",
            task_id="mock_cube_task_1",
            llm_config=_rollout_llm_request_config(),
            rollout_index=3,
        )

        async def run_rollout() -> list[dict]:
            events: list[dict] = []
            await rollout.submit(request)
            async for event in rollout.events(
                from_offset=0,
                stop_request_id="request-direct",
                timeout_s=10.0,
            ):
                events.append(event)
            return events

        events = asyncio.run(run_rollout())
        terminals = [event for event in events if event["type"] == "terminal"]
        assert len(terminals) == 1
        assert terminals[0]["request_id"] == "request-direct"
        assert terminals[0]["rollout_status"] == "completed"
        assert terminals[0]["trajectory_id"] == "request-direct"
    finally:
        rollout.close()


def test_duplicate_rollout_request_does_not_emit_second_accepted(tmp_dir) -> None:
    config = RolloutConfig(
        name="rollout_test",
        output_dir=tmp_dir,
        benchmark_config=MockCubeBenchmarkConfig(),
        agent_config=MockAgentConfig(),
        max_steps=2,
        execution_mode="local",
    )
    rollout = RolloutEngine(config=config)

    try:
        request = RolloutRequest(
            request_id="duplicate-request",
            task_id="mock_cube_task_1",
            llm_config=_rollout_llm_request_config(),
        )

        async def run_duplicate() -> None:
            await rollout.submit(request)
            try:
                await rollout.submit(request)
            except ValueError:
                return
            raise AssertionError("duplicate request was accepted")

        asyncio.run(run_duplicate())
        accepted = [event for event in rollout.events_from(0) if event["type"] == "accepted"]
        assert len(accepted) == 1
    finally:
        rollout.close()


def test_rollout_events_are_reconstructable_from_stream(tmp_dir) -> None:
    config = RolloutConfig(
        name="rollout_test",
        output_dir=tmp_dir,
        benchmark_config=MockCubeBenchmarkConfig(),
        agent_config=MockAgentConfig(),
        max_steps=2,
        execution_mode="local",
    )
    rollout = RolloutEngine(config=config)

    try:
        request = RolloutRequest(
            request_id="request-reconstruct",
            task_id="mock_cube_task_1",
            llm_config=_rollout_llm_request_config(),
        )

        async def run_rollout() -> list[dict]:
            await rollout.submit(request)
            async for _event in rollout.events(
                from_offset=0,
                stop_request_id=request.request_id,
                timeout_s=10.0,
            ):
                pass
            return rollout.events_from(0)

        events = asyncio.run(run_rollout())
        request_events = [event for event in events if event["request_id"] == request.request_id]
        assert [event["event_index"] for event in request_events] == list(range(len(request_events)))

        tool_calls = [event for event in request_events if event["type"] == "tool_call"]
        assert tool_calls
        assert all(_event_body(event)["obs"] is not None for event in tool_calls)
        assert _event_body(tool_calls[0])["parent_event_id"] == "reset"
        assert any(_event_body(event).get("action") for event in tool_calls)
    finally:
        rollout.close()


def _rl_sink(published: list[dict], *, fail: bool = False) -> RLEventSink:
    def publish(payload: dict) -> None:
        if fail:
            raise RuntimeError("publisher down")
        published.append(dict(payload))

    return RLEventSink(
        event_context=EventContext(
            request_id="request-token-data",
            trajectory_id="task_ep0",
            env_name="mock",
            task_id="task",
        ),
        event_publisher=publish,
    )


def _llm_call(**kwargs) -> LLMCall:
    return LLMCall(
        tag=kwargs.pop("tag", "act"),
        llm_config=LLMConfig(model_name="gpt-5-nano"),
        prompt=Prompt(messages=[{"role": "user", "content": "hi"}]),
        output=Message(role="assistant", content="hello"),
        usage=Usage(),
        **kwargs,
    )


def test_trainable_llm_call_missing_token_data_marks_rollout_non_trainable() -> None:
    published: list[dict] = []
    sink = _rl_sink(published)

    sink.save_event(TrajectoryEvent(output=LLMCallEvent(call=_llm_call())), "task_ep0")
    sink.save_event(TrajectoryEvent(output=EvaluationEvent(reward=1.0, is_terminal=True)), "task_ep0")

    terminal = published[-1]
    assert terminal["type"] == "terminal"
    assert terminal["rollout_status"] == "training_data_error"
    assert terminal["rollout_valid"] is False
    assert terminal["trainable"] is False
    assert terminal["error"]["type"] == "MissingTrainingData"


def test_trainable_llm_call_emits_prompt_and_completion_token_ids() -> None:
    published: list[dict] = []
    sink = _rl_sink(published)
    llm_call = _llm_call(
        prompt_token_ids=[1, 2, 3],
        completion_token_ids=[4, 5],
        logprobs=[-0.1, -0.2],
    )

    sink.save_event(TrajectoryEvent(output=LLMCallEvent(call=llm_call)), "task_ep0")

    llm_event = published[0]
    assert llm_event["type"] == "llm_call"
    assert _rl_body(llm_event)["trainable"] is True
    assert _llm_call_body(llm_event)["prompt_token_ids"] == [1, 2, 3]
    assert _llm_call_body(llm_event)["completion_token_ids"] == [4, 5]
    assert _llm_call_body(llm_event)["logprobs"] == [-0.1, -0.2]


def test_rl_sink_publishes_tool_call_event() -> None:
    published: list[dict] = []
    sink = _rl_sink(published)
    action = Action(id="a1", name="click", arguments={"target": "ok"})

    sink.save_event(
        TrajectoryEvent(
            output=ToolCallEvent(
                parent_event_id="llm1",
                action_id=action.id,
                action=action,
                obs=Observation.from_text("clicked"),
                turn_id="llm1",
            ),
            start_time=1.0,
            end_time=2.0,
        ),
        "task_ep0",
    )

    event = published[0]
    assert event["type"] == "tool_call"
    assert _event_body(event)["parent_event_id"] == "llm1"
    assert _event_body(event)["action"]["name"] == "click"
    assert _event_body(event)["obs"] is not None


def test_rl_sink_publishes_terminal_evaluation_event() -> None:
    published: list[dict] = []
    sink = _rl_sink(published)

    sink.save_event(
        TrajectoryEvent(output=EvaluationEvent(reward=1.0, info={"success": True}, is_terminal=True)),
        "task_ep0",
    )

    evaluation = published[0]
    terminal = published[1]
    assert evaluation["type"] == "evaluation"
    assert _event_body(evaluation)["reward"] == 1.0
    assert _event_body(evaluation)["is_terminal"] is True
    assert terminal["type"] == "terminal"
    assert terminal["rollout_status"] == "completed"
    assert terminal["final_reward"] == 1.0
    assert terminal["rollout_valid"] is True


def test_rl_sink_budget_failure_keeps_terminal_evaluation_reward() -> None:
    # Mirrors the real episode sequence: Episode catches BudgetExceeded without
    # re-raising, records the failure, then still runs task.evaluate() and emits
    # a terminal EvaluationEvent. The terminal must carry that reward.
    published: list[dict] = []
    sink = _rl_sink(published)

    sink.save_event(
        TrajectoryEvent(
            output=AgentErrorEvent(
                error=StepError(error_type="BudgetExceeded", exception_str="budget exceeded", stack_trace="")
            )
        ),
        "task_ep0",
    )

    agent_error = published[0]
    assert agent_error["type"] == "agent_error"
    assert _event_body(agent_error)["error"]["error_type"] == "BudgetExceeded"
    assert [e["type"] for e in published] == ["agent_error"], "no terminal before the evaluation arrives"

    sink.save_event(
        TrajectoryEvent(output=EvaluationEvent(reward=0.5, info={"partial": True}, is_terminal=True)),
        "task_ep0",
    )

    terminal = published[-1]
    assert terminal["type"] == "terminal"
    assert terminal["rollout_status"] == "max_steps"
    assert terminal["final_reward"] == 0.5
    assert terminal["rollout_valid"] is True


def test_rl_sink_budget_failure_then_eval_crash_emits_error_terminal() -> None:
    # If task.evaluate() raises after BudgetExceeded, the episode records a
    # second AgentErrorEvent (and re-raises) — no evaluation ever arrives, so
    # the sink must emit the error terminal itself.
    published: list[dict] = []
    sink = _rl_sink(published)

    sink.save_event(
        TrajectoryEvent(
            output=AgentErrorEvent(
                error=StepError(error_type="BudgetExceeded", exception_str="budget exceeded", stack_trace="")
            )
        ),
        "task_ep0",
    )
    sink.save_event(
        TrajectoryEvent(
            output=AgentErrorEvent(error=StepError(error_type="ValueError", exception_str="eval boom", stack_trace=""))
        ),
        "task_ep0",
    )

    terminal = published[-1]
    assert terminal["type"] == "terminal"
    assert terminal["final_reward"] is None
    assert terminal["rollout_valid"] is False
    assert terminal["trainable"] is False
    assert terminal["error"]["error_type"] == "ValueError"


def test_rl_sink_publisher_failure_is_recorded() -> None:
    sink = _rl_sink([], fail=True)

    try:
        sink.save_event(TrajectoryEvent(output=EvaluationEvent(reward=1.0, is_terminal=True)), "task_ep0")
    except RuntimeError:
        pass
    else:
        raise AssertionError("publisher failure did not propagate to EventStreamer")

    assert sink.publisher_error == {"type": "RuntimeError", "message": "publisher down"}


def test_rollout_llm_config_rejects_disabled_training_capture() -> None:
    with pytest.raises(ValueError):
        RolloutLLMConfig(
            model_name="served-model",
            api_base="http://localhost:8000/v1",
            api_key="EMPTY",
            tokenizer_name="mock-tokenizer",
            capture_training_metadata=False,
        )


def test_rollout_llm_config_rejects_benchmark_only_options() -> None:
    with pytest.raises(ValueError):
        RolloutLLMConfig(
            model_name="served-model",
            api_base="http://localhost:8000/v1",
            api_key="EMPTY",
            tokenizer_name="mock-tokenizer",
            reasoning_effort="low",
        )


def test_rollout_llm_always_requests_training_capture_fields() -> None:
    response = MagicMock()
    response.prompt_token_ids = [1, 2, 3]
    response.usage = None
    response.choices = [
        MagicMock(
            message=Message(role="assistant", content="ok"),
            logprobs=MagicMock(content=[MagicMock(token_id=4, logprob=-0.1), MagicMock(token_id=5, logprob=-0.2)]),
            finish_reason="stop",
        )
    ]

    config = RolloutLLMConfig(
        model_name="served-model",
        api_base="http://localhost:8000/v1",
        api_key="EMPTY",
        tokenizer_name="mock-tokenizer",
    )

    with patch.object(LLM, "_completion_with_retry", return_value=response) as completion:
        llm = LLM(config=config)
        result = llm(Prompt(messages=[{"role": "user", "content": "hi"}]))

    kwargs = completion.call_args.kwargs
    assert kwargs["api_key"] == "EMPTY"
    assert kwargs["logprobs"] == 1
    assert kwargs["skip_special_tokens"] is False
    assert kwargs["include_stop_str_in_output"] is True
    assert kwargs["extra_body"]["return_token_ids"] is True
    assert kwargs["extra_body"]["return_tokens_as_token_ids"] is True
    assert kwargs["max_completion_tokens"] == config.max_completion_tokens
    # max_tokens is the deprecated alias of max_completion_tokens; sending both
    # is redundant and rejected by some OpenAI-compatible servers.
    assert "max_tokens" not in kwargs
    assert result.prompt_token_ids == [1, 2, 3]
    assert result.completion_token_ids == [4, 5]
    assert result.logprobs == [-0.1, -0.2]


def test_rollout_llm_config_replaces_plain_agent_llm_config() -> None:
    from cube_harness.rl import RolloutLLMConfig
    from cube_harness.rl.utils import override_rollout_llm_config

    agent_config = REACT_CONFIGS["default"]
    override_rollout_llm_config(
        agent_config,
        RolloutLLMConfig(
            api_base="http://127.0.0.1:8000",
            model_name="served-model",
            api_key="EMPTY",
            tokenizer_name="mock-tokenizer",
            temperature=0.7,
            max_completion_tokens=128,
            extra_body={"custom": True},
            overrides={"num_retries": 3, "does_not_exist": "ignored"},
        ),
    )

    assert isinstance(agent_config.llm_config, RolloutLLMConfig)
    assert agent_config.llm_config.model_name == "openai/served-model"
    assert agent_config.llm_config.api_base == "http://127.0.0.1:8000/v1"
    assert agent_config.llm_config.temperature == 0.7
    assert agent_config.llm_config.max_completion_tokens == 128
    assert agent_config.llm_config.num_retries == 3
    assert agent_config.llm_config.extra_body == {"custom": True}
    assert not hasattr(agent_config.llm_config, "does_not_exist")


def test_event_streamer_can_raise_required_sink_failure() -> None:
    class FailingSink:
        raise_on_emit_error = True

        def save_event(self, event: TrajectoryEvent, trajectory_id: str) -> None:
            raise RuntimeError("required sink down")

    streamer_config = EventStreamerConfig(extra_sinks=[FailingSink()])
    streamer = EventStreamer(trajectory_id="task_ep0")
    streamer._sinks.extend(streamer_config.extra_sinks)

    with pytest.raises(RuntimeError, match="required sink down"):
        streamer.emit(TrajectoryEvent(output=EvaluationEvent(reward=1.0, is_terminal=True)))
