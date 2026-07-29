"""Unit tests for timewarp_cube — metadata, named subsets, config wiring.

All tests here are server-free and browser-free: they exercise only the shipped
``task_metadata.json`` and the Pydantic config layer, so they run in CI without
the external TimeWarp wiki/news/webshop servers or a live browser. Anything that
calls ``make()`` on a task/benchmark (which builds a real BrowserGym browser and
probes the servers) belongs in the debug suite (``python -m timewarp_cube.debug``),
not here.
"""

from __future__ import annotations

import pytest
from cube.tool import ToolboxConfig
from cube_browser_tool.bgym_tool import BgymToolConfig
from cube_chat_tool import ChatToolConfig

from timewarp_cube import TIMEWARP_CONFIGS, TimeWarpBenchmarkConfig, provisioning
from timewarp_cube.configs import _browser_with_chat
from timewarp_cube.debug import get_debug_benchmark
from timewarp_cube.task import TimeWarpTaskConfig, TimeWarpTaskMetadata

# Exact per-site task counts from the shipped task_metadata.json. Tasks may require
# several sites, so the subsets overlap (their counts sum to >231); see
# test_subsets_cover_all_tasks for the coverage invariant.
_SUBSET_COUNTS = {"wiki": 111, "news": 89, "webshop": 95}


def _config() -> TimeWarpBenchmarkConfig:
    return TimeWarpBenchmarkConfig(tool_config=_browser_with_chat())


def test_benchmark_config_loads_231_tasks() -> None:
    cfg = _config()
    assert cfg.name == "timewarp-cube"
    assert cfg.benchmark_metadata.num_tasks == 231
    assert len(cfg.task_metadata) == 231
    # num_tasks is derived from the shipped task_metadata.json (benchmark._NUM_TASKS), so the
    # metadata count and the registry must always agree — this guards against drift.
    assert cfg.benchmark_metadata.num_tasks == len(cfg.task_metadata)

    tm = next(iter(cfg.tasks().values()))
    assert isinstance(tm, TimeWarpTaskMetadata)
    assert isinstance(tm.sites, list) and tm.sites
    assert isinstance(tm.eval_types, list)
    assert tm.intent_template_id is None or isinstance(tm.intent_template_id, int)


def test_metadata_scoring_is_mostly_deterministic() -> None:
    """Upstream v0.2.0 moved all but two tasks onto deterministic verifiers. Guards the
    regeneration of task_metadata.json: a stale snapshot would show 231 llm_judge tasks and
    silently reintroduce the API-key requirement."""
    judged = [tm.id for tm in _config().tasks().values() if tm.eval_types == ["llm_judge"]]
    assert judged == ["32", "143"]


def test_named_subsets_registered() -> None:
    assert set(TimeWarpBenchmarkConfig.benchmark_metadata.named_subsets) == set(_SUBSET_COUNTS)


@pytest.mark.parametrize(("site", "expected_count"), list(_SUBSET_COUNTS.items()))
def test_named_subset_filters_by_site(site: str, expected_count: int) -> None:
    sub = _config().named_subset(site)
    tasks = list(sub.tasks().values())
    assert len(tasks) == expected_count
    # named_subset(site) keeps any task whose `sites` contains a matching entry.
    assert all(any(site in s for s in tm.sites) for tm in tasks)


def test_subsets_cover_all_tasks() -> None:
    """The three site subsets overlap but together cover every task."""
    cfg = _config()
    covered = set().union(*(cfg.named_subset(s).tasks().keys() for s in _SUBSET_COUNTS))
    assert covered == set(cfg.tasks())


def test_benchmark_config_round_trip_preserves_tasks() -> None:
    # Full-config equality is intentionally not asserted: the nested browser
    # ToolboxConfig does not compare equal after a serialize/validate cycle.
    # The task selection (what actually defines the benchmark) does survive.
    cfg = _config().named_subset("wiki")
    rehydrated = TimeWarpBenchmarkConfig.model_validate_json(cfg.model_dump_json())
    assert rehydrated.name == cfg.name
    assert set(rehydrated.tasks()) == set(cfg.tasks())


def test_task_config_round_trips() -> None:
    task_cfg = next(iter(_config().get_task_configs()))
    assert isinstance(task_cfg, TimeWarpTaskConfig)
    rehydrated = TimeWarpTaskConfig.model_validate_json(task_cfg.model_dump_json())
    assert isinstance(rehydrated.metadata, TimeWarpTaskMetadata)
    assert rehydrated.metadata == task_cfg.metadata


def test_toolbox_pairs_browser_with_chat() -> None:
    """TimeWarp scores the agent's chat answer, so the toolbox must carry a
    BgymTool (browser) and a ChatTool. ``headless`` only tweaks the browser config."""
    toolbox = _browser_with_chat(use_screenshot=False, headless=True)
    assert isinstance(toolbox, ToolboxConfig)
    kinds = [type(t).__name__ for t in toolbox.tool_configs]
    assert any(isinstance(t, BgymToolConfig) for t in toolbox.tool_configs), kinds
    assert any(isinstance(t, ChatToolConfig) for t in toolbox.tool_configs), kinds


def test_timewarp_configs_registry() -> None:
    assert set(TIMEWARP_CONFIGS.keys()) == {"default", "wiki", "news", "webshop"}
    assert isinstance(TIMEWARP_CONFIGS["default"], TimeWarpBenchmarkConfig)
    assert len(TIMEWARP_CONFIGS["default"].tasks()) == 231
    for site, expected_count in _SUBSET_COUNTS.items():
        assert len(TIMEWARP_CONFIGS[site].tasks()) == expected_count


def test_debug_benchmark_constructs() -> None:
    cfg = get_debug_benchmark()
    assert isinstance(cfg, TimeWarpBenchmarkConfig)
    assert set(cfg.tasks()) == {"1", "2"}


def test_install_does_not_provision(monkeypatch: pytest.MonkeyPatch) -> None:
    """install() is a lightweight L1 hook: auto mode provisions lazily at make()-time, so
    install() must never trigger a clone/setup.sh — the harness calls it before every debug run.

    Regression for the reviewed footgun: the dangerous state is conda-present + servers-down,
    which used to kick off a multi-GB download and then fail the manual-mode debug suite. Force
    that exact state so any re-introduction of gated provisioning in install() trips this test.
    """
    for var in provisioning.SITE_ENV_VARS.values():
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(provisioning, "has_conda", lambda: True)

    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("install() must not provision")

    monkeypatch.setattr(provisioning, "ensure_provisioned", _fail)
    TimeWarpBenchmarkConfig.install()  # no exception → did not provision
