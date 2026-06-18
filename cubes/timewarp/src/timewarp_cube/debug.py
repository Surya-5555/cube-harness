"""Smoke-test script for timewarp-cube — validates infrastructure end-to-end.

Verifies that tasks load, the BrowserGym task setup runs against the live TimeWarp
servers, the browser + chat tools initialise, and the agent can submit an answer
through the chat tool that ``GenericTimeWarpTask.validate`` consumes.

Requires the TimeWarp environments running with TW_WIKI / TW_NEWS / TW_WEBSHOP set
(see timewarp/scripts/environment/run_all_env.sh) and, for a non-zero reward,
OPENAI_API_KEY (every TimeWarp task is scored by the llm_judge). Like workarena,
the suite only fails on errors (Python exceptions), not on reward.

Public API (cube.testing protocol)
-----------------------------------
get_debug_benchmark()              -> TimeWarpBenchmarkConfig
make_debug_agent(task_id: str)     -> ReferenceAnswerAgent

Usage:
    uv run python -m timewarp_cube.debug
"""

from __future__ import annotations

import logging
import sys

from cube.core import Action, ActionSchema, Observation
from cube.testing import run_debug_suite

from timewarp_cube._data import load_raw_tasks
from timewarp_cube.benchmark import TimeWarpBenchmarkConfig
from timewarp_cube.configs import _browser_with_chat

logger = logging.getLogger(__name__)

# Two wiki tasks — only the wiki server is needed to exercise the setup/validate path.
_DEBUG_TASK_IDS = ["1", "2"]


def _reference_answer(task_id: str) -> str:
    """Read the task's reference answer from the shipped browsergym-timewarp data."""
    config = next((c for c in load_raw_tasks() if str(c.get("task_id")) == str(task_id)), None)
    if config is None:
        raise ValueError(f"No TimeWarp task with task_id={task_id}")
    ref = config["eval"].get("reference_answers", {})
    answer = ref.get("fuzzy_match", ref.get("exact_match", ""))
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    return str(answer)


class ReferenceAnswerAgent:
    """Submits the task's reference answer via the chat tool, then stops."""

    def __init__(self, answer: str) -> None:
        self._answer = answer
        self._answered = False

    def __call__(self, obs: Observation, action_set: list[ActionSchema]) -> Action:
        if not self._answered:
            self._answered = True
            return Action(name="send_message", arguments={"text": self._answer})
        return Action(name="final_step", arguments={})


def make_debug_agent(task_id: str) -> ReferenceAnswerAgent:
    return ReferenceAnswerAgent(_reference_answer(task_id))


def get_debug_benchmark() -> TimeWarpBenchmarkConfig:
    tool_config = _browser_with_chat(use_screenshot=False, headless=True)
    return TimeWarpBenchmarkConfig(tool_config=tool_config).subset_from_list(
        _DEBUG_TASK_IDS, benchmark_name_suffix="debug"
    )


if __name__ == "__main__":
    import timewarp_cube.debug as _this_module

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    results = run_debug_suite("timewarp-cube", _this_module)
    failed = [r for r in results if r["error"]]
    sys.exit(1 if failed else 0)
