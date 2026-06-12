from __future__ import annotations

import copy
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from cube_harness.episode import Episode
from cube_harness.episode_logs import LOG_FORMAT, get_log_path, redirect_output_to_log
from cube_harness.rl.llm import RolloutLLMConfig
from cube_harness.rl.trajectory_sink import RLEventSink
from cube_harness.rl.utils import override_rollout_llm_config
from cube_harness.storage import FileStorage, InMemoryStorage
from cube_harness.streamer import EventStreamerConfig


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
        self.trajectory_id = self.request_id

    def run(self) -> dict[str, Any]:
        override_rollout_llm_config(self.agent_config, RolloutLLMConfig.model_validate(self.request["llm_config"]))
        persist_rollout = bool(self.payload.get("persist_rollout"))
        run_output_dir = self.output_dir
        storage = FileStorage(run_output_dir) if persist_rollout else InMemoryStorage(run_output_dir)
        rl_sink = RLEventSink(
            event_context=self.payload["event_context"],
            event_publisher=self.publisher_handle,
        )
        recorder_config = EventStreamerConfig(
            extra_sinks=[rl_sink],
        )
        episode = Episode(
            id=self.episode_id,
            output_dir=run_output_dir,
            agent_config=self.agent_config,
            task_config=self.task_config,
            exp_name=str(self.payload["service_name"]),
            max_steps=int(self.request.get("max_steps") or self.payload["max_steps"]),
            storage=storage,
            runtime_context=self.payload.get("runtime_context"),
            recorder_config=recorder_config,
            write_eval_log=persist_rollout,
            trajectory_id=self.trajectory_id,
        )
        if persist_rollout:
            log_file = get_log_path(self.output_dir, self.trajectory_id)
            context = redirect_output_to_log(log_file, append=True, tee=True, log_format=LOG_FORMAT)
        else:
            context = nullcontext()
        with context:
            episode.run()
        return {"ok": True, "request_id": self.request_id}
