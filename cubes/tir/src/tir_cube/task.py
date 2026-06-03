from typing import Any, Tuple

from cube.benchmark import RuntimeContext
from cube.core import Observation
from cube.task import Task, TaskConfig, TaskMetadata  # noqa: F401 — TaskMetadata kept for typing
from tir_cube.tool import TIRTool
import math_verify
from pydantic import BaseModel


class RewardTable(BaseModel):
    wrong_answer_not_finished: float
    wrong_answer_finished: float
    no_answer_not_finished: float
    no_answer_finished: float
    unparsable_not_finished: float
    unparsable_finished: float
    correct_answer_not_finished: float
    correct_answer_finished: float

    def get_reward_range(self) -> tuple[float, float]:
        values = [
            self.wrong_answer_not_finished,
            self.wrong_answer_finished,
            self.no_answer_not_finished,
            self.no_answer_finished,
            self.unparsable_not_finished,
            self.unparsable_finished,
            self.correct_answer_not_finished,
            self.correct_answer_finished,
        ]
        return min(values), max(values)


def get_reward(answer_status: str, finished: bool, reward_table: RewardTable) -> float:
    match (answer_status, finished):
        case ("wrong", False):
            return reward_table.wrong_answer_not_finished
        case ("wrong", True):
            return reward_table.wrong_answer_finished
        case ("no_answer", False):
            return reward_table.no_answer_not_finished
        case ("no_answer", True):
            return reward_table.no_answer_finished
        case ("unparsable", False):
            return reward_table.unparsable_not_finished
        case ("unparsable", True):
            return reward_table.unparsable_finished
        case ("correct", False):
            return reward_table.correct_answer_not_finished
        case ("correct", True):
            return reward_table.correct_answer_finished
        case _:
            raise ValueError(f"Invalid answer_status/finished combination: {answer_status}/{finished}")


def _verify_answer_status(
    prediction: str,
    gold: str,
    *,
    strict: bool = True,
    max_prediction_length: int = 1000,
) -> str:
    boxed_start = prediction.rfind("\\boxed{")
    if boxed_start < 0:
        return "no_answer"

    boxed_prediction = prediction[boxed_start:]
    if "\\boxed{}" in boxed_prediction:
        return "unparsable"
    if len(boxed_prediction) > max_prediction_length:
        return "unparsable"

    try:
        gold_parsed = math_verify.parse(gold)
        boxed_prediction_parsed = math_verify.parse(boxed_prediction)
        if not boxed_prediction_parsed:
            return "unparsable"
        equivalent = math_verify.verify(gold_parsed, boxed_prediction_parsed, strict=strict, timeout_seconds=1)
        return "correct" if equivalent else "wrong"
    except Exception:
        return "unparsable"


class TIRTaskMetadata(TaskMetadata):
    """TaskMetadata subclass for TIR tasks.
    Adds cube-specific public fields that are safe to ship in task_metadata.json.
    """

    dataset: str
    question: str
    expected: str
    rewards: dict[str, float]


class TIRTask(Task):
    """Solve a math problem via Python and submit final LaTeX answer with MathAnswer."""

    accept_agent_stop: bool = False  # agent should not be allowed to stop before submitting an answer
    validate_per_step: bool = False  # only validate at the end based on submitted answer, not per step

    @property
    def _expected(self) -> str:
        return self.metadata.expected

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self.tool.reset()
        question = self.metadata.question
        return Observation.from_text(question), {"question": question, "expected": self._expected}

    def evaluate(self, obs: Observation | None = None) -> Tuple[float, dict]:
        if self.tool.final_answer is not None:
            submitted = self.tool.final_answer
        else:
            submitted = self.tool._last_python_output or ""

        answer_status = _verify_answer_status(submitted, self._expected, strict=True)
        reward_table = RewardTable(**self.metadata.rewards)
        reward = get_reward(answer_status, self.finished(obs), reward_table)

        return (
            reward,
            {
                "success": answer_status == "correct",
                "no_answer": answer_status == "no_answer",
                "no_error": answer_status != "unparsable",
                "num_python_calls": self.tool.python_call_count,
                "overflow": not self.tool.submitted_final_answer,
            },
        )

    def finished(self, obs: Observation | None = None) -> bool:
        assert isinstance(self.tool, TIRTool)
        return self.tool.submitted_final_answer


class TIRTaskConfig(TaskConfig[TIRTaskMetadata]):
    """Serializable configuration that produces TIRTask."""

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
    ) -> TIRTask:
        _ = runtime_context
        assert self.tool_config is not None, "tool_config must be set"

        # from tir_cube.benchmark import TIRBenchmark

        # task_metadata = TIRBenchmark.task_metadata[self.task_id]
        # tool_cfg = self.tool_config or TIRToolConfig()

        return TIRTask(
            metadata=self.metadata,
            tool_config=self.tool_config,
        )
