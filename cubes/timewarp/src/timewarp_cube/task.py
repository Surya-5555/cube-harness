"""TimeWarp task implementation for the CUBE framework.

Wraps BrowserGym's ``GenericTimeWarpTask`` (an ``AbstractBrowserTask``): the cube
reuses its ``setup(page)`` / ``validate(page, chat_messages)`` directly, driving
them through a cube ``BrowserTool`` (the live Playwright page) and a ``ChatTool``
(the agent's final text answer). TimeWarp scores the agent's last chat message,
so a ``ChatTool`` is required in the toolbox.
"""

import logging
from typing import Any, Protocol, runtime_checkable

from browsergym.timewarp.task import GenericTimeWarpTask
from cube.benchmark import RuntimeContext
from cube.core import Observation
from cube.task import Task, TaskConfig, TaskMetadata
from cube.tool import Toolbox
from cube.tools.browser import BrowserTool
from cube_chat_tool import ChatTool
from playwright.sync_api import Page
from pydantic import PrivateAttr

from timewarp_cube import provisioning
from timewarp_cube._data import verify_upstream_data

logger = logging.getLogger(__name__)

#: Default seed for a TimeWarp task when none is supplied. TimeWarp is deterministic
#: per task_id, so the exact value only matters for reproducibility of any randomness.
_DEFAULT_SEED = 42


class TimeWarpTaskMetadata(TaskMetadata):
    """TaskMetadata subclass for TimeWarp tasks.

    Public fields shipped in task_metadata.json (available at import time).
    TimeWarp has no heavy execution data — all task logic is available from the
    browsergym-timewarp library at runtime via the numeric task_id (``id``).
    """

    sites: list[str]
    """TimeWarp sites required for this task, e.g. ['wiki'], ['news'], ['webshop']."""

    intent_template_id: int | None
    """Intent template identifier for grouping tasks with the same underlying intent.
    None when the source task carries no template id (true for most TimeWarp tasks)."""

    eval_types: list[str]
    """Evaluator types for this task, e.g. ['string_match'], ['number_match'], ['list_match'].
    ['llm_judge'] on the two tasks upstream still scores with a model."""


@runtime_checkable
class TimeWarpBrowserTool(Protocol):
    """Browser tools usable by TimeWarp — must expose the live Playwright page
    and a cube observation. Both ``BgymTool`` and ``SyncPlaywrightTool`` satisfy this.
    """

    @property
    def page(self) -> Page: ...

    def page_obs(self) -> Observation: ...


class TimeWarpTask(Task[TimeWarpTaskMetadata]):
    """CUBE Task wrapper for a single BrowserGym TimeWarp task."""

    metadata: TimeWarpTaskMetadata  # type: ignore[assignment]
    seed: int = _DEFAULT_SEED
    tw_urls: dict[str, str] | None = None
    """site → URL for the running servers, read from the benchmark's ``runtime_context`` in
    auto mode (see ``TimeWarpTaskConfig.make``). Applied to ``os.environ`` in ``reset()`` (the
    worker process) because BrowserGym's TimeWarpInstance reads TW_WIKI/TW_NEWS/TW_WEBSHOP from
    the environment, and driver env does not reach Ray workers. None in manual mode → BrowserGym
    reads the ambient (shell-exported) env vars."""
    tw_ui_version: int | None = None
    """Temporal UI era the servers were actually launched at, from the benchmark's
    ``runtime_context``. None whenever the cube did not launch them (manual mode, or auto mode
    reusing external servers): the era is then genuinely unknowable, and recording that honestly
    beats recording a requested-but-unapplied value. Surfaced in ``reset()``'s info so every
    trajectory carries the UI it ran against — the axis TimeWarp exists to measure (PS-001)."""
    # validate_per_step stays at the default (False): TimeWarp only scores the agent's
    # terminal chat answer, so evaluate() need only run once the task is done. Per-step
    # validation added no reward signal (non-answer steps always score 0) and cost a
    # page scan every step.

    _bgym_task: GenericTimeWarpTask | None = PrivateAttr(default=None)

    @property
    def _browser_tool(self) -> TimeWarpBrowserTool:
        """Resolve the browser tool from the Toolbox (or a bare tool)."""
        if isinstance(self.tool, Toolbox):
            tool = self.tool.find_tool(BrowserTool)
            if tool is None:
                raise RuntimeError("No BrowserTool found in the Toolbox.")
        else:
            tool = self.tool
        if not isinstance(tool, TimeWarpBrowserTool):
            raise RuntimeError(
                f"The browser tool must expose .page and .page_obs() (e.g. BgymTool), got {type(tool).__name__}."
            )
        return tool

    @property
    def _chat_tool(self) -> ChatTool:
        """Resolve the required ChatTool — TimeWarp scores the agent's final chat message."""
        if isinstance(self.tool, Toolbox):
            tool = self.tool.find_tool(ChatTool)
            if isinstance(tool, ChatTool):
                return tool
        raise RuntimeError(
            "TimeWarp requires a ChatTool in the Toolbox so the agent can submit its answer via send_message()."
        )

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        """Instantiate the BrowserGym task, run its setup(), and return the initial observation.

        Combines the task intent text with the initial page state. The intent is also
        posted into the chat session as a 'user' message so it stays visible in chat_obs().
        """
        # Auto mode: inject the resolved server URLs into this worker's env before BrowserGym
        # reads them. No-op in manual mode (tw_urls is None → ambient TW_* env vars are used).
        if self.tw_urls is not None:
            provisioning.apply_to_env(self.tw_urls)
        self._bgym_task = GenericTimeWarpTask(seed=self.seed, task_id=int(self.metadata.id))
        self.tool.reset()
        goal, task_info = self._bgym_task.setup(self._browser_tool.page)
        self._chat_tool.add_message("user", goal)
        obs = Observation.from_text(goal) + self._browser_tool.page_obs()
        # Spread BrowserGym's task_info first, then let the cube's canonical values win:
        # task_info also carries task_id/sites/goal, and its task_id is BrowserGym's int,
        # not our string self.id.
        info = {
            **task_info,
            "task_id": self.id,
            "sites": self.metadata.sites,
            "goal": goal,
            "ui_version": self.tw_ui_version,
        }
        return obs, info

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict[str, Any]]:
        """Score the agent's answer via BrowserGym's validate().

        ``GenericTimeWarpTask.validate`` reads the last chat message (role 'assistant'
        from send_message(), or 'infeasible' from report_infeasible()) and runs the
        task's evaluator — deterministic string/number/list matching for all but two
        tasks, which use an LLM judge. Returns 0.0 when no answer has been submitted
        yet, without invoking the evaluator at all.

        Evaluator failures propagate (upstream ≥ 0.2.0 no longer swallows them into a
        0.0 score): the harness records the episode as FAILED with the real error, which
        is what we want — a judge outage is not the same as an agent getting it wrong.

        BrowserGym's own ``done`` is reported as ``bgym_done``, not ``done``: the harness owns
        ``reward_info["done"]`` ("this episode finalized", which XRay and inspect_results read),
        while BrowserGym's means "solved or stopped" and is False for a correct-but-unscored
        answer. They are different questions and the names must not collide.
        """
        if self._bgym_task is None:
            raise RuntimeError("TimeWarp task is not initialized. Call reset() first.")
        score, done, _user_message, task_info = self._bgym_task.validate(
            self._browser_tool.page, self._chat_tool.messages
        )
        # A hard zero from validate()'s own pre-checks reports itself only through `error`, and
        # nothing downstream turns that into an error_type — so without this line an episode
        # poisoned by an off-domain tab is indistinguishable from an agent that answered wrongly.
        if task_info.get("error"):
            logger.warning(
                "TimeWarp task %s scored %.1f with an evaluation error: %s", self.id, score, task_info["error"]
            )
        return score, {**task_info, "bgym_done": done}

    def finished(self, obs: Observation | None = None) -> bool:
        """Done once the agent has submitted a final answer (assistant) or reported infeasible.

        Computed cheaply from chat state to avoid triggering the (LLM) evaluator on
        every step. BrowserGym's ``validate()`` additionally reports done on an
        off-domain navigation; we intentionally do not replicate that here (it would
        require a per-step ``validate()`` call), so such an episode terminates at
        max_steps rather than early. The score is unaffected — it is already 0.
        """
        messages = self._chat_tool.messages
        return bool(messages) and messages[-1]["role"] in ("assistant", "infeasible")

    def close(self) -> None:
        """Tear down the BrowserGym task, then close the tool."""
        if self._bgym_task is not None:
            try:
                self._bgym_task.teardown()
            except Exception as e:
                logger.warning(f"Error during TimeWarp task teardown: {e}")
            finally:
                self._bgym_task = None
        super().close()


class TimeWarpTaskConfig(TaskConfig[TimeWarpTaskMetadata]):
    """Serializable configuration for a single TimeWarp task."""

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
    ) -> TimeWarpTask:
        assert self.tool_config is not None, "TimeWarpTaskConfig requires a tool_config."
        # Also checked in TimeWarpBenchmark._setup, but a Ray worker builds tasks straight from a
        # pickled TaskConfig and never runs _setup — so without this the process that actually
        # scores would be the one process not verifying what it is scoring with. Cached: free.
        verify_upstream_data()
        # Auto mode publishes the resolved server URLs and UI era into runtime_context
        # (re-derived each run); manual mode leaves it unset → both None → BrowserGym reads the
        # ambient env and the episode records the era as unknown.
        tw_urls = runtime_context.get("tw_urls") if runtime_context else None
        tw_ui_version = runtime_context.get("tw_ui_version") if runtime_context else None
        return TimeWarpTask(
            metadata=self.metadata,
            tool_config=self.tool_config,
            seed=self.seed if self.seed is not None else _DEFAULT_SEED,
            tw_urls=tw_urls,
            tw_ui_version=tw_ui_version,
        )
