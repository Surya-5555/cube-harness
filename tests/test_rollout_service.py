from __future__ import annotations

import asyncio
import json

from cube_harness.rl import CancelRequest, RayConfig, RolloutConfig, RolloutEngine, RolloutRequest, serve
from tests.conftest import MockAgentConfig, MockCubeBenchmarkConfig
from tests.rollout_perf_helpers import SlowRolloutBenchmarkConfig


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


def test_rollout_service_runs_native_episode_from_service_benchmark(tmp_dir) -> None:
    config = RolloutConfig(
        name="rollout_test",
        output_dir=tmp_dir,
        benchmark_config=MockCubeBenchmarkConfig(),
        agent_config=MockAgentConfig(),
        max_steps=1,
    )
    app = serve(config=config)
    service = app.state.service

    try:
        health = service.health()
        assert health["ready"] is True
        assert health["benchmark"]["name"] == "mock-cube"
        assert health["ray"]["initialized"] is True
        assert health["executor"]["inflight_rollouts"] == 0
        assert (tmp_dir / "rollout_config.json").exists()

        request = RolloutRequest(
            request_id="request-1",
            client_id="client-a",
            task_id="mock_cube_task_1",
            llm_config={},
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
        assert accepted[0]["trajectory_id"] == "mock_cube_task_1_ep2"
        assert len(terminals) == 1
        assert terminals[0]["rollout_status"] == "completed"
        assert terminals[0]["env_name"] == "mock-cube"
        assert terminals[0]["trajectory_id"] == "mock_cube_task_1_ep2"
        episode_dir = tmp_dir / "client-a" / "request-1" / "episodes" / "mock_cube_task_1_ep2"
        assert (episode_dir / "episode.log").exists()
        assert not (episode_dir / "episode.metadata.json").exists()
        assert not (episode_dir / "steps").exists()
    finally:
        service.close()


def test_rollout_health_reports_ray_capacity(tmp_dir) -> None:
    config = RolloutConfig(
        name="rollout_test",
        output_dir=tmp_dir,
        benchmark_config=MockCubeBenchmarkConfig(),
        agent_config=MockAgentConfig(),
        max_steps=1,
        ray=RayConfig(num_workers=2),
    )
    app = serve(config=config)
    service = app.state.service

    try:
        health = service.health()
        assert health["ready"] is True
        assert health["ray"]["initialized"] is True
        assert health["ray"]["configured_num_workers"] == 2
        assert health["ray"]["task_num_cpus"] == 0.25
        assert health["ray"]["cluster_cpus"] >= 1
        assert health["ray"]["estimated_rollout_slots"] >= 8
        assert health["executor"]["inflight_rollouts"] == 0
        assert health["executor"]["cancelled_rollouts"] == 0
        assert health["sink"]["next_offset"] == 0
        assert health["sink"]["oldest_available_offset"] == 0
        assert health["sink"]["max_hot_events"] > 0
    finally:
        service.close()


def test_rollout_cancel_emits_single_cancelled_terminal(tmp_dir) -> None:
    config = RolloutConfig(
        name="rollout_cancel_test",
        output_dir=tmp_dir,
        benchmark_config=SlowRolloutBenchmarkConfig(),
        agent_config=MockAgentConfig(),
        max_steps=1,
        ray=RayConfig(num_workers=1),
    )
    rollout = RolloutEngine(config=config)

    try:
        request = RolloutRequest(
            request_id="cancel-request-1",
            client_id="client-a",
            task_id="slow_rollout_task",
            llm_config={},
        )

        async def run_cancel() -> list[dict]:
            await rollout.submit(request)
            result = await rollout.cancel(CancelRequest(request_id=request.request_id))
            assert result == {"cancelled": 1}
            async for _event in rollout.events(
                from_offset=0,
                stop_request_id=request.request_id,
                timeout_s=10.0,
                poll_timeout_s=0.1,
            ):
                pass
            return rollout.events_from(0)

        events = asyncio.run(run_cancel())
        terminals = [event for event in events if event["type"] == "terminal"]
        assert len(terminals) == 1
        assert terminals[0]["request_id"] == request.request_id
        assert terminals[0]["rollout_status"] == "cancelled"
        assert terminals[0]["rollout_valid"] is False
        assert terminals[0]["trainable"] is False

        health = rollout.stats()
        assert health["executor"]["cancelled_rollouts"] == 1
        assert health["executor"]["terminal_rollouts"] == 1
        assert health["executor"]["pending_cancel_request_ids"] == []
    finally:
        rollout.close()


def test_rollout_streams_events_without_http_service(tmp_dir) -> None:
    config = RolloutConfig(
        name="rollout_test",
        output_dir=tmp_dir,
        benchmark_config=MockCubeBenchmarkConfig(),
        agent_config=MockAgentConfig(),
        max_steps=1,
    )
    rollout = RolloutEngine(config=config)

    try:
        request = RolloutRequest(
            request_id="request-direct",
            client_id="client-a",
            task_id="mock_cube_task_1",
            llm_config={},
            rollout_index=3,
        )

        async def run_rollout() -> list[dict]:
            events: list[dict] = []
            await rollout.submit(request)
            async for event in rollout.events(
                client_id="client-a",
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
        assert terminals[0]["trajectory_id"] == "mock_cube_task_1_ep3"
    finally:
        rollout.close()


def test_rollout_llm_config_applies_supported_overrides() -> None:
    from cube_harness.llm import LLMConfig
    from cube_harness.rl import RolloutLLMConfig
    from cube_harness.rl.llm import apply_rollout_llm_config

    class AgentConfigWithLLM:
        llm_config = LLMConfig(model_name="openai/original", api_base="http://old/v1", api_key="old")

    agent_config = AgentConfigWithLLM()
    apply_rollout_llm_config(
        agent_config,
        RolloutLLMConfig(
            api_base="http://127.0.0.1:8000",
            model_name="served-model",
            api_key="EMPTY",
            temperature=0.7,
            max_completion_tokens=128,
            logprobs=True,
            extra_body={"return_token_ids": True},
            overrides={"num_retries": 1, "does_not_exist": "ignored"},
        ),
    )

    assert agent_config.llm_config.api_base == "http://127.0.0.1:8000/v1"
    assert agent_config.llm_config.model_name == "openai/served-model"
    assert agent_config.llm_config.api_key == "EMPTY"
    assert agent_config.llm_config.temperature == 0.7
    assert agent_config.llm_config.max_completion_tokens == 128
    assert agent_config.llm_config.logprobs is True
    assert agent_config.llm_config.extra_body == {"return_token_ids": True}
    assert agent_config.llm_config.num_retries == 1
    assert not hasattr(agent_config.llm_config, "does_not_exist")
