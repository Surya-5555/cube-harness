"""Infra-free unit tests for bfcl-cube.

Covers the BenchmarkConfig contract (metadata auto-load, named subsets,
subsetting, serialization, debug factory), the vendored AST checker, the BFCL→
OpenAI schema conversion, and the BfclTool record/action-set behaviour.
"""

from __future__ import annotations

from cube.benchmark import BenchmarkConfig
from cube.core import Action, ActionSchema
from cube.task import TaskExecutionInfo

from bfcl_cube._vendor.ast_checker import ast_checker
from bfcl_cube._vendor.schema_convert import bfcl_parameters_to_openai, normalize_function_name
from bfcl_cube.benchmark import BfclBenchmarkConfig
from bfcl_cube.debug import debug_task_actions, get_debug_benchmark
from bfcl_cube.task import BfclExecutionInfo, BfclTaskConfig, BfclTaskMetadata
from bfcl_cube.tool import BfclTool, BfclToolConfig

_DEBUG_TASK_IDS = list(debug_task_actions())


# ── BenchmarkConfig contract ────────────────────────────────────────────────
def test_task_metadata_loaded() -> None:
    cfg = BfclBenchmarkConfig()
    assert cfg.benchmark_metadata.num_tasks == 3491
    assert len(cfg.task_metadata) == 3491
    sample = next(iter(cfg.task_metadata.values()))
    assert isinstance(sample, BfclTaskMetadata)
    assert sample.category
    assert sample.collection in {"non_live", "live"}


def test_named_subsets_partition() -> None:
    cfg = BfclBenchmarkConfig()
    assert set(cfg.named_subsets()) == {"non_live", "live"}
    non_live = cfg.named_subset("non_live")
    live = cfg.named_subset("live")
    assert non_live.subset_name == "non_live"
    # The two named subsets partition the full benchmark.
    assert non_live.num_tasks + live.num_tasks == cfg.num_tasks
    assert all(cfg.task_metadata[t].collection == "non_live" for t in non_live.task_ids)


def test_config_roundtrip() -> None:
    cfg = BfclBenchmarkConfig().subset_from_list(_DEBUG_TASK_IDS)
    restored = BfclBenchmarkConfig.model_validate_json(cfg.model_dump_json())
    assert restored.task_ids == _DEBUG_TASK_IDS
    assert restored.num_tasks == len(_DEBUG_TASK_IDS)
    assert restored.benchmark_metadata.name == "bfcl-cube"


def test_get_task_configs_stamps_metadata() -> None:
    cfg = BfclBenchmarkConfig().subset_from_list(_DEBUG_TASK_IDS)
    configs = list(cfg.get_task_configs())
    assert len(configs) == len(_DEBUG_TASK_IDS)
    for tc in configs:
        assert isinstance(tc, BfclTaskConfig)
        assert isinstance(tc.metadata, BfclTaskMetadata)
        assert tc.metadata.id == tc.task_id
        assert tc.metadata.category


def test_debug_benchmark_type() -> None:
    cfg = get_debug_benchmark()
    assert isinstance(cfg, BfclBenchmarkConfig)
    assert isinstance(cfg, BenchmarkConfig)
    assert set(cfg.task_ids) == set(_DEBUG_TASK_IDS)


def test_execution_info_roundtrip() -> None:
    ei = BfclExecutionInfo(
        question=[[{"role": "user", "content": "hi"}]],
        functions=[{"name": "f", "description": "d", "parameters": {"type": "dict", "properties": {}, "required": []}}],
        ground_truth=[{"f": {}}],
    )
    assert isinstance(ei, TaskExecutionInfo)
    restored = BfclExecutionInfo.model_validate_json(ei.model_dump_json())
    assert restored.functions[0]["name"] == "f"
    assert restored.ground_truth == [{"f": {}}]


# ── Vendored AST checker ────────────────────────────────────────────────────
_TRIANGLE_FN = {
    "name": "calculate_triangle_area",
    "description": "Area of a triangle.",
    "parameters": {
        "type": "dict",
        "properties": {
            "base": {"type": "integer", "description": "base"},
            "height": {"type": "integer", "description": "height"},
            "unit": {"type": "string", "description": "unit"},
        },
        "required": ["base", "height"],
    },
}
_TRIANGLE_GT = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]


def test_ast_checker_simple_pass() -> None:
    out = [{"calculate_triangle_area": {"base": 10, "height": 5}}]
    assert ast_checker([_TRIANGLE_FN], out, _TRIANGLE_GT, "simple_python")["valid"]


def test_ast_checker_simple_wrong_value() -> None:
    out = [{"calculate_triangle_area": {"base": 11, "height": 5}}]
    assert not ast_checker([_TRIANGLE_FN], out, _TRIANGLE_GT, "simple_python")["valid"]


def test_ast_checker_missing_required() -> None:
    out = [{"calculate_triangle_area": {"base": 10}}]
    assert not ast_checker([_TRIANGLE_FN], out, _TRIANGLE_GT, "simple_python")["valid"]


def test_ast_checker_parallel_no_order() -> None:
    fn2 = {**_TRIANGLE_FN, "name": "g"}
    gt = [
        {"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}},
        {"g": {"base": [2], "height": [3], "unit": [""]}},
    ]
    # Order-independent: provide the second call first.
    out = [{"g": {"base": 2, "height": 3}}, {"calculate_triangle_area": {"base": 10, "height": 5}}]
    assert ast_checker([_TRIANGLE_FN, fn2], out, gt, "parallel")["valid"]


# ── Schema conversion ───────────────────────────────────────────────────────
def test_schema_convert_types() -> None:
    params = bfcl_parameters_to_openai(
        {
            "type": "dict",
            "properties": {
                "x": {"type": "integer", "description": "d"},
                "y": {"type": "float", "description": "d"},
                "z": {"type": "array", "items": {"type": "string"}, "description": "d"},
                "w": {"type": "dict", "description": "d"},
            },
            "required": ["x"],
        }
    )
    assert params["type"] == "object"
    assert params["properties"]["x"]["type"] == "integer"
    assert params["properties"]["y"]["type"] == "number"
    assert params["properties"]["z"]["type"] == "array"
    assert params["properties"]["z"]["items"]["type"] == "string"
    assert params["properties"]["w"]["type"] == "object"


def test_normalize_function_name() -> None:
    assert normalize_function_name("math.factorial") == "math_factorial"


# ── BfclTool ────────────────────────────────────────────────────────────────
def test_tool_action_set_and_record() -> None:
    tool = BfclToolConfig(functions=[_TRIANGLE_FN]).make()
    assert isinstance(tool, BfclTool)
    names = {s.name for s in tool.action_set}
    assert "calculate_triangle_area" in names
    assert "final_step" in names  # inherited STOP action
    for s in tool.action_set:
        assert isinstance(s, ActionSchema)

    tool.reset()
    obs = tool.execute_action(Action(name="calculate_triangle_area", arguments={"base": 10, "height": 5}))
    assert obs.error is None
    assert len(tool.recorded_calls) == 1
    assert tool.recorded_calls[0].arguments == {"base": 10, "height": 5}


def test_tool_normalizes_dotted_names() -> None:
    fn = {**_TRIANGLE_FN, "name": "geo.area"}
    tool = BfclToolConfig(functions=[fn]).make()
    assert "geo_area" in {s.name for s in tool.action_set}
