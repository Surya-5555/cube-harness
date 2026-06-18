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


TIMEWARP_CONFIGS: ConfigRegistry[TimeWarpBenchmarkConfig] = ConfigRegistry(
    {
        "default": TimeWarpBenchmarkConfig(tool_config=_browser_with_chat()),
        "wiki": TimeWarpBenchmarkConfig(tool_config=_browser_with_chat()).named_subset("wiki"),
        "news": TimeWarpBenchmarkConfig(tool_config=_browser_with_chat()).named_subset("news"),
        "webshop": TimeWarpBenchmarkConfig(tool_config=_browser_with_chat()).named_subset("webshop"),
    }
)
