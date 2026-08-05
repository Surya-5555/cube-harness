"""Canonical TimeWarp benchmark configs.

    from timewarp_cube import TIMEWARP_CONFIGS
    benchmark = TIMEWARP_CONFIGS["default"]

Runs on BrowserGym with axtree + screenshot (the canonical web observation),
bundled with a ChatTool so the agent can submit its final answer via
send_message() — TimeWarp scores that answer. "default" is the full benchmark;
"wiki"/"news"/"webshop" are the per-site splits.
"""

from cube.core import ConfigRegistry
from cube.tool import ToolboxConfig
from cube_browser_playwright.playwright_session import PlaywrightSessionConfig
from cube_browser_tool.bgym_tool import BgymToolConfig
from cube_chat_tool import ChatToolConfig

from timewarp_cube.benchmark import TimeWarpBenchmarkConfig


def _browser_with_chat(*, use_screenshot: bool = True, headless: bool | None = None) -> ToolboxConfig:
    """TimeWarp toolbox: a BrowserGym browser tool plus the ChatTool the agent uses to
    submit its final answer (TimeWarp scores that message). ``headless=None`` keeps the
    BgymToolConfig browser default; the debug suite overrides it for headless CI runs."""
    browser = PlaywrightSessionConfig() if headless is None else PlaywrightSessionConfig(headless=headless)
    return ToolboxConfig(
        tool_configs=[
            BgymToolConfig(browser=browser, use_html=False, use_axtree=True, use_screenshot=use_screenshot),
            ChatToolConfig(),
        ]
    )


#: Opt-in action-description overrides that spell out TimeWarp's answer protocol.
#:
#: Set them on the *agent* config, not the benchmark::
#:
#:     agent.description_overrides = dict(ANSWER_PROTOCOL_OVERRIDES)
#:
#: Two measured traps motivate this, both invisible to an agent reading the default
#: descriptions, and both costing accuracy for reasons unrelated to browsing:
#:
#: * ``send_message`` is **terminal and one-shot**. ``TimeWarpTask.finished()`` returns True on
#:   the first assistant message, so a model that narrates before answering has its narration
#:   scored. Submitting each task's own gold answer scores 40/40 on a stratified slice; prefixing
#:   it with a single scratchpad line scores **0/40**.
#: * 60 of the 229 deterministically-scored tasks match only against the answer's **first
#:   sentence** (``scope: first_sentence``). Prefixing each task's own gold answer with one
#:   lead-in sentence drops it from 229/229 to 168/229 — **-26.6 points** for phrasing alone.
#:
#: Kept out of ``TIMEWARP_CONFIGS`` deliberately. This is a knob a recipe opts into, not a change
#: to the benchmark: it alters what the agent sees, so numbers with and without it are not
#: comparable. Per ``AgentConfig.description_overrides``, a proven override graduates into the
#: tool's own docstring via a PR — here that would be upstream in cube-standard, since
#: ``cube.tool.tool_action`` has no way for a task to declare an action terminal.
ANSWER_PROTOCOL_OVERRIDES: dict[str, str] = {
    "send_message": (
        "Submit your final answer to the task. This ENDS the episode immediately — it is your "
        "one and only message, and whatever you send is what gets scored, so do not use it to "
        "think out loud or to report progress. Start with the answer itself, in the first "
        "sentence (some tasks only read that sentence); any explanation goes after it."
    ),
    "report_infeasible": (
        "Report that the task cannot be completed. This ENDS the episode and is scored as the "
        "literal answer 'N/A', which is only ever correct on tasks whose true answer is that "
        "nothing was found — so prefer answering with send_message whenever you have an answer."
    ),
}

TIMEWARP_CONFIGS: ConfigRegistry[TimeWarpBenchmarkConfig] = ConfigRegistry(
    {
        "default": TimeWarpBenchmarkConfig(tool_config=_browser_with_chat()),
        "wiki": TimeWarpBenchmarkConfig(tool_config=_browser_with_chat()).named_subset("wiki"),
        "news": TimeWarpBenchmarkConfig(tool_config=_browser_with_chat()).named_subset("news"),
        "webshop": TimeWarpBenchmarkConfig(tool_config=_browser_with_chat()).named_subset("webshop"),
    }
)
