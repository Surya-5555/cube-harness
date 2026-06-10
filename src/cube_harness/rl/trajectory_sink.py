from __future__ import annotations

import logging
from typing import Any

from cube_harness.core import (
    AgentErrorEvent as TrajectoryAgentErrorEvent,
)
from cube_harness.core import (
    EvaluationEvent as TrajectoryEvaluationEvent,
)
from cube_harness.core import (
    LLMCallEvent as TrajectoryLLMCallEvent,
)
from cube_harness.core import (
    ToolCallEvent as TrajectoryToolCallEvent,
)
from cube_harness.core import (
    TrajectoryEvent,
)
from cube_harness.rl.events import EventContext, dump_for_event, rollout_event_payload

logger = logging.getLogger(__name__)


class RLEventSink:
    """EventStreamer sink that publishes RL rollout payloads from trajectory events."""

    raise_on_emit_error: bool = True

    def __init__(
        self,
        *,
        event_context: EventContext,
        event_publisher: Any,
        trainable_call_tags: set[str] | None = None,
    ) -> None:
        self.ctx = event_context
        self.event_publisher = event_publisher
        self.trainable_call_tags = trainable_call_tags or {"", "act"}
        self.event_idx = 0
        self.llm_call_idx = 0
        self.tool_call_idx = 0
        self.trainable_call_idx = 0
        self.rollout_trainable = False
        self.training_data_error: dict[str, Any] | None = None
        self.publisher_error: dict[str, Any] | None = None
        self.error: dict[str, Any] | None = None
        self.status = "completed"
        self.terminal_emitted = False

    def save_event(self, te: TrajectoryEvent, trajectory_id: str) -> None:
        output = te.output
        if isinstance(output, TrajectoryLLMCallEvent):
            self._publish_llm_call(output, trajectory_id)
        elif isinstance(output, TrajectoryToolCallEvent):
            self._publish_tool_call(output, te, trajectory_id)
        elif isinstance(output, TrajectoryAgentErrorEvent):
            self._publish_agent_error(output)
        elif isinstance(output, TrajectoryEvaluationEvent):
            self._publish_evaluation(output)

    def _publish_llm_call(self, event: TrajectoryLLMCallEvent, trajectory_id: str) -> None:
        call = event.call
        trainable_tag = call is not None and call.tag in self.trainable_call_tags
        has_training_data = (
            call is not None
            and bool(call.prompt_token_ids)
            and bool(call.completion_token_ids)
            and bool(call.logprobs)
            and len(call.completion_token_ids or []) == len(call.logprobs or [])
        )
        trainable = bool(trainable_tag and has_training_data and event.error is None)
        if trainable_tag and not has_training_data and self.training_data_error is None:
            self.training_data_error = {
                "type": "MissingTrainingData",
                "message": "trainable LLM call is missing prompt_token_ids or aligned completion_token_ids/logprobs",
                "llm_call_id": call.id if call is not None else event.id,
                "llm_call_tag": call.tag if call is not None else "",
            }
        if event.error is not None:
            self.error = dump_for_event(event.error)
            self.status = "llm_error"
        if trainable:
            self.rollout_trainable = True

        payload = rollout_event_payload(self.ctx, "llm_call", self.event_idx)
        payload["event"] = dump_for_event(event)
        payload["rl"] = {
            "llm_call_index": self.llm_call_idx,
            "trainable_call_index": self.trainable_call_idx if trainable else None,
            "trainable": trainable,
            "state_ref": f"{trajectory_id}:event:{self.event_idx}",
        }
        self._publish(payload)
        self.event_idx += 1
        self.llm_call_idx += 1
        if trainable:
            self.trainable_call_idx += 1

    def _publish_tool_call(self, event: TrajectoryToolCallEvent, te: TrajectoryEvent, trajectory_id: str) -> None:
        if event.error is not None:
            self.error = dump_for_event(event.error)
            self.status = "tool_error"

        payload = rollout_event_payload(self.ctx, "tool_call", self.event_idx)
        payload["event"] = dump_for_event(event)
        payload["rl"] = {
            "tool_call_index": self.tool_call_idx,
            "state_ref": f"{trajectory_id}:tool:{self.tool_call_idx}",
        }
        payload["trajectory_event"] = {
            "start_time": te.start_time,
            "end_time": te.end_time,
        }
        self._publish(payload)
        self.event_idx += 1
        self.tool_call_idx += 1

    def _publish_agent_error(self, event: TrajectoryAgentErrorEvent) -> None:
        payload = rollout_event_payload(self.ctx, "agent_error", self.event_idx)
        payload["event"] = dump_for_event(event)
        self._publish(payload)
        self.event_idx += 1
        if event.error.error_type == "BudgetExceeded":
            # Episode catches BudgetExceeded WITHOUT re-raising and still runs
            # task.evaluate(); the terminal EvaluationEvent that follows carries
            # the real final reward. Emitting a terminal (or latching self.error)
            # here would drop that reward via the terminal_emitted guard and mark
            # every max_steps rollout invalid. If evaluate() itself raises, the
            # episode records a second AgentErrorEvent (different error_type)
            # which takes the branch below and emits the error terminal.
            self.status = "max_steps"
            return
        self.error = dump_for_event(event.error)
        if self.status not in {"llm_error", "tool_error", "training_data_error"}:
            self.status = "agent_error"
        if not self.terminal_emitted:
            self._publish_terminal_payload(final_reward=None)

    def _publish_evaluation(self, event: TrajectoryEvaluationEvent) -> None:
        payload = rollout_event_payload(self.ctx, "evaluation", self.event_idx)
        payload["event"] = dump_for_event(event)
        self._publish(payload)
        self.event_idx += 1
        if event.is_terminal:
            self._publish_terminal_payload(final_reward=event.reward, summary={"info": dump_for_event(event.info)})

    def _publish_terminal_payload(self, *, final_reward: float | None, summary: dict[str, Any] | None = None) -> None:
        if self.terminal_emitted:
            return
        terminal_error = self.publisher_error or self.training_data_error or self.error
        status = "training_data_error" if self.training_data_error is not None else self.status
        valid = terminal_error is None and final_reward is not None
        payload = rollout_event_payload(self.ctx, "terminal", self.event_idx)
        payload.update(
            {
                "rollout_status": status,
                "outcome_success": final_reward is not None and final_reward > 0,
                "final_reward": final_reward,
                "rollout_valid": valid,
                "trainable": valid and self.rollout_trainable,
                "error": terminal_error,
                "summary": summary or {},
            }
        )
        self._publish(payload)
        self.event_idx += 1
        self.terminal_emitted = True

    def _publish(self, payload: dict[str, Any]) -> None:
        try:
            if hasattr(self.event_publisher, "publish_payload"):
                publish_payload = self.event_publisher.publish_payload
                if hasattr(publish_payload, "remote"):
                    from cube_harness.rl.ray_runtime import ray

                    ray.get(publish_payload.remote(payload))
                else:
                    publish_payload(payload)
            else:
                self.event_publisher(payload)
        except Exception as exc:
            self.publisher_error = {"type": type(exc).__name__, "message": str(exc)[:500]}
            logger.exception("Failed to publish RL rollout event")
            raise
