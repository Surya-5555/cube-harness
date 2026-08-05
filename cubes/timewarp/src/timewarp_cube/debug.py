"""Smoke-test script for timewarp-cube — validates infrastructure end-to-end.

Verifies that tasks load, the BrowserGym task setup runs against the live TimeWarp
servers, the browser + chat tools initialise, and the agent can submit an answer
through the chat tool that ``GenericTimeWarpTask.validate`` consumes.

Requires the TimeWarp environments running with TW_WIKI / TW_NEWS / TW_WEBSHOP set
(start them from the upstream TimeWarp repo, https://github.com/sparklabutah/timewarp,
via scripts/environment/run_all_env.sh — see the cube README). No API key: the debug
tasks are scored by deterministic verifiers, so the suite asserts ``reward == 1.0`` —
submitting a task's own gold answer must score it. (Unlike workarena, which is
reward-blind because its cheat agent is nondeterministic.) A silent zero — a renamed
chat role, say — would otherwise take the no-answer branch and still report success.

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
    """Read the task's reference answer from the shipped browsergym-timewarp data.

    ``fuzzy_match`` is the human-readable gold answer and is present on every task,
    including the deterministically-scored ones whose verifier reads other keys
    (``must_include``, ``number_match``, …) — so it stays the right thing to submit.
    """
    config = next((c for c in load_raw_tasks() if str(c.get("task_id")) == str(task_id)), None)
    if config is None:
        raise ValueError(f"No TimeWarp task with task_id={task_id}")
    ref = config["eval"].get("reference_answers", {})
    answer = ref.get("fuzzy_match", "")
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
    # Manual mode: the debug suite requires the three TW_* servers already running (see the
    # module docstring). The test harness calls `config.install()` before `make()`, but
    # `install()` is now lightweight and does no provisioning, so this suite never triggers the
    # auto-mode clone + conda build + multi-GB download — it fails fast with an actionable
    # message (from `_setup_manual`) if the servers aren't up.
    tool_config = _browser_with_chat(use_screenshot=False, headless=True)
    return TimeWarpBenchmarkConfig(tool_config=tool_config, provision_mode="manual").subset_from_list(
        _DEBUG_TASK_IDS, benchmark_name_suffix="debug"
    )


if __name__ == "__main__":
    import timewarp_cube.debug as _this_module

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    results = run_debug_suite("timewarp-cube", _this_module)
    failed = [r for r in results if r["error"] or not r["done"] or r["reward"] != 1.0]
    sys.exit(1 if failed else 0)
