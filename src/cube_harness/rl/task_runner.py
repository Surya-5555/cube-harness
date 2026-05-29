from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from cube_harness.episode_logs import LOG_FORMAT, get_log_path, redirect_output_to_log, trajectory_log_id
from cube_harness.episode_loop import EpisodeLoop
from cube_harness.episode_recorders import RolloutEventRecorder
from cube_harness.rl.llm import RolloutLLMConfig, apply_rollout_llm_config


class RolloutTaskRunner:
    """Runs exactly one cube-harness episode rollout inside a worker process."""

    def __init__(self, payload: dict[str, Any], publisher_handle: Any) -> None:
        self.payload = payload
        self.publisher_handle = publisher_handle
        self.request = dict(payload["request"])
        self.request_id = str(self.request["request_id"])
        self.task_config = payload["task_config"]
        self.agent_config = copy.deepcopy(payload["agent_config"])
        self.output_dir = Path(payload["output_dir"])
        self.episode_id = int(self.request.get("rollout_index") or 0)
        self.task_id = str(self.request["task_id"])
        self.trajectory_id = trajectory_log_id(self.task_id, self.episode_id)

    def run(self) -> dict[str, Any]:
        apply_rollout_llm_config(self.agent_config, RolloutLLMConfig.model_validate(self.request["llm_config"]))
        recorder = RolloutEventRecorder(
            event_context=self.payload["event_context"],
            event_publisher=self.publish_event,
        )
        loop = EpisodeLoop(
            id=self.episode_id,
            output_dir=self.output_dir,
            agent_config=self.agent_config,
            task_config=self.task_config,
            exp_name=str(self.payload["service_name"]),
            max_steps=int(self.request.get("max_steps") or self.payload["max_steps"]),
            runtime_context=self.payload.get("runtime_context"),
            recorder=recorder,
        )
        log_file = get_log_path(self.output_dir, self.trajectory_id)
        with redirect_output_to_log(log_file, append=True, tee=False, log_format=LOG_FORMAT):
            loop.run()
        return {"ok": True, "request_id": self.request_id}

    def publish_event(self, event: Any) -> dict:
        payload = event.model_dump(mode="json")
        publish_payload = self.publisher_handle.publish_payload
        if callable(publish_payload):
            return publish_payload(payload)

        from cube_harness.rl.ray_runtime import ray

        return ray.get(publish_payload.remote(payload))
