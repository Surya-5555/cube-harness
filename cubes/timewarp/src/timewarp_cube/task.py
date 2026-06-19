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
    """Evaluator types for this task, e.g. ['llm_judge'] or ['exact_match']."""


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
        }
        return obs, info

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict[str, Any]]:
        """Score the agent's answer via BrowserGym's validate().

        ``GenericTimeWarpTask.validate`` reads the last chat message (role 'assistant'
        from send_message(), or 'infeasible' from report_infeasible()) and runs the
        task's evaluator (exact_match or OpenAI llm_judge). Returns 0.0 when no answer
        has been submitted yet.
        """
        if self._bgym_task is None:
            raise RuntimeError("TimeWarp task is not initialized. Call reset() first.")
        score, done, _user_message, task_info = self._bgym_task.validate(
            self._browser_tool.page, self._chat_tool.messages
        )
        return score, {"done": done, **task_info}

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
        _ = runtime_context
        assert self.tool_config is not None, "TimeWarpTaskConfig requires a tool_config."
        return TimeWarpTask(
            metadata=self.metadata,
            tool_config=self.tool_config,
            seed=self.seed if self.seed is not None else _DEFAULT_SEED,
        )
