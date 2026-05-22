#!/usr/bin/env python
"""SMOKE: render the exact prompt the Auto-CUBE debug investigator receives.

Runs the real `investigate_experiment` pipeline (transcript extraction →
context-file resolution → prompt assembly → extra-fragment append) against a
self-contained episode fixture, using a **capture-only driver** that records the
system + user prompt instead of calling an LLM. Then prints both, so a human can
read precisely what the Sonnet investigator would see — including the debug
use-case's `investigator_extra.md` fragment and the injected codebase map.

No API key, no network. Offline and repeatable.

Usage:
    .venv/bin/python scripts/smoke/investigator_prompt_preview.py
    .venv/bin/python scripts/smoke/investigator_prompt_preview.py --experiment-dir <dir> --episode <traj_id>

The --experiment-dir form previews against a real experiment; it requires an
existing `investigation_context.md` there (we won't spend an Opus call in a
smoke). Prints `SMOKE OK/FAIL/SKIP: investigator_prompt_preview`.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Any

import msgpack
import typer
import zstandard

from cube_harness.analyze.investigator.agent_driver import DriverResult
from cube_harness.analyze.investigator.context import INVESTIGATION_CONTEXT_FILENAME
from cube_harness.analyze.investigator.core import InvestigationConfig, investigate_experiment
from cube_harness.eval_log import EPISODE_RECORD_FILENAME, EpisodeRecord, UsageSummary

# A valid findings JSON so the pipeline completes after we've captured the prompt.
_CANNED_FINDINGS = json.dumps(
    {
        "analysis": "(smoke — not a real analysis)",
        "evidence": [{"step": 1, "quote": "cat operations.py"}],
        "summary": "Smoke placeholder.",
        "outcome": "failure",
        "primary_blame": "model_capability",
        "primary_blame_confidence": 2,
        "other_blames": [],
        "hypothesis": "(smoke)",
        "hypothesis_confidence": 1,
    }
)

# A realistic codebase map (what the upgraded Opus context agent would produce),
# so the preview shows the full orientation + tree, not just dir paths. Uses `##`
# headings (no H1) so it nests under the prompt's `# Codebase map`.
_RICH_CONTEXT_MD = """\
## Orientation

**cube-standard** is the protocol layer every benchmark and tool implements. A
`Task` exposes `reset()` (initial observation), `step()` (apply an action), and
`evaluate()` (reward on termination). A `Tool` exposes actions via `@tool_action`.
A `Benchmark` is the task collection; `Resource` / `Container` provision infra.
Generalist tools are shared here — notably the terminal tool in
`cube/tools/terminal.py`, which most code/agent benchmarks reuse rather than
re-implement.

**cube-harness** is the runtime that drives cube-standard. The agent
(`agents/genny.py`) is the LLM loop: it reads the observation, emits actions,
parses tool results. `Episode` drives the agent ⇄ env exchange and enforces the
budget; the recorded result is a `Trajectory` (what you are reading now).
`exp_runner` runs many episodes; the investigator (this sub-system) lives under
`analyze/investigator/`.

**This benchmark (SWE-bench-verified)** hands the agent a repository with a
failing test and asks for a patch. The action surface is the shared terminal
tool (shell). An episode starts with the issue text + repo checkout. The cube
is a thin **adapter over the upstream `swebench` package** — the real patch
application and test harness live there, and the cube delegates to them. The
**verifier** is `evaluate()`, which calls into upstream `swebench` to run the
tests. The **ground truth** is the `FAIL_TO_PASS` / `PASS_TO_PASS` test lists
carried in the task metadata (`TaskMetadata.extra`) — a solution is correct iff
those tests flip to passing without regressing the rest.

## Key locations

    /Users/alex/dev/cube/cube-standard/src/cube/            (cube-standard root)
    ├── task.py                   — Task ABC: reset() / step() / evaluate()
    ├── tool.py                   — Tool ABC, @tool_action
    └── tools/terminal.py         — TerminalTool: shared shell action surface

    /Users/alex/dev/cube/cube-harness/src/cube_harness/     (cube-harness root)
    ├── agents/genny.py:Genny.run — the agent loop (LLM call, action parse/exec)
    └── episode.py:Episode        — drives agent ⇄ env, enforces budget

    /Users/alex/dev/cube/cube-harness/cubes/swebench-verified-cube/src/swebench_verified_cube/
    ├── task.py:evaluate          — VERIFIER (delegates to upstream swebench)
    ├── task.py (TaskMetadata.extra) — GROUND TRUTH: FAIL_TO_PASS / PASS_TO_PASS test ids
    ├── task.py:reset             — initial observation (issue text + repo)
    └── benchmark.py              — task collection + shared setup

    /Users/alex/.venv/lib/python3.12/site-packages/swebench/   (upstream — the wrapped benchmark)
    ├── harness/run_evaluation.py — real test harness the cube's evaluate() calls
    └── harness/grading.py        — the actual pass/fail grading logic

## Paths

```paths
cube_package: {cube_package}
agent_package: {agent_package}
upstream_package: {upstream_package}
```
"""


class _CaptureDriver:
    """AgentDriver that records each run()'s system+user prompt and returns canned findings."""

    name = "capture-driver"
    max_parallelism = 1

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def run(self, **kwargs: Any) -> DriverResult:
        self.calls.append(kwargs)
        return DriverResult(output_text=f"```json\n{_CANNED_FINDINGS}\n```")

    async def continue_session(self, **kwargs: Any) -> DriverResult:  # pragma: no cover - unused
        raise NotImplementedError


def _write_step(steps_dir: Path, name: str, payload: dict) -> None:
    raw = msgpack.packb(payload, use_bin_type=True)
    (steps_dir / name).write_bytes(zstandard.ZstdCompressor().compress(raw))


def _build_fixture(root: Path) -> tuple[Path, str]:
    """Create a minimal-but-realistic experiment dir. Returns (experiment_dir, trajectory_id)."""
    traj_id = "swebench-django-12345_ep0"
    exp = root / "exp"
    ep = exp / "episodes" / traj_id
    (ep / "steps").mkdir(parents=True)

    _write_step(
        ep / "steps",
        "000_obs.msgpack.zst",
        {
            "output": {
                "obs": {
                    "contents": [{"data": "Fix the JSON serializer bug.", "tool_call_id": None}],
                    "reward": None,
                    "done": False,
                }
            }
        },
    )
    _write_step(
        ep / "steps",
        "001_act.msgpack.zst",
        {
            "output": {
                "actions": [{"name": "Bash", "arguments": {"command": "cat operations.py"}}],
                "llm_calls": [],
                "error": None,
            }
        },
    )

    record = EpisodeRecord(
        evaluation_id="smoke-eval",
        sample_id="django__django-12345",
        is_correct=False,
        score=0.0,
        num_turns=2,
        n_agent_steps=1,
        n_env_steps=1,
        usage=UsageSummary(),
        trajectory_id=traj_id,
        timestamp=0.0,
    )
    (ep / EPISODE_RECORD_FILENAME).write_text(record.model_dump_json(indent=2))

    # Minimal episode.metadata.json (a Trajectory) so the Episode header shows
    # real task_id / reward / steps / description instead of "unknown".
    (ep / "episode.metadata.json").write_text(
        json.dumps(
            {
                "id": traj_id,
                "metadata": {
                    "task_id": "django__django-12345",
                    "task_description": (
                        "Fix the JSON serializer in django/db/backends/sqlite3/operations.py "
                        "so it handles datetime objects."
                    ),
                },
                "reward_info": {"reward": 0.0},
                "summary_stats": {"n_agent_steps": 12},
            }
        )
    )

    # Minimal experiment_config.json so agent/benchmark resolve (dict-view _type
    # fallback) instead of rendering "unknown" in the Episode header.
    (exp / "experiment_config.json").write_text(
        json.dumps(
            {
                "agent_config": {"_type": "cube_harness.agents.genny.Genny"},
                "benchmark_config": {"_type": "swebench_verified_cube.SWEBenchVerified"},
            }
        )
    )

    # Seed a rich investigation_context.md so the pipeline reuses it (no Opus call).
    # Its listed paths must exist locally — point them at real package roots.
    cube_pkg = root / "fake_cube_package"
    agent_pkg = root / "fake_agent_package"
    upstream_pkg = root / "fake_upstream_package"
    for d in (cube_pkg, agent_pkg, upstream_pkg):
        d.mkdir()
    (exp / INVESTIGATION_CONTEXT_FILENAME).write_text(
        _RICH_CONTEXT_MD.format(cube_package=cube_pkg, agent_package=agent_pkg, upstream_package=upstream_pkg)
    )
    return exp, traj_id


def _capture_prompt(exp: Path, traj_id: str, extra_prompt_fragment: str | None) -> dict[str, Any]:
    driver = _CaptureDriver()
    investigate_experiment(
        exp,
        InvestigationConfig(
            driver=driver,
            ids=[traj_id],
            synthesis_model="",  # skip meta-analysis — we only want the per-episode prompt
            extra_prompt_fragment=extra_prompt_fragment,
        ),
    )
    if not driver.calls:
        raise RuntimeError("driver was never called — pipeline did not reach the investigator")
    # With a seeded context file, the only run() is the investigator dispatch.
    return driver.calls[-1]


def main(
    experiment_dir: Annotated[Path | None, typer.Option(help="Preview against a real experiment dir.")] = None,
    episode: Annotated[str | None, typer.Option(help="Trajectory id within --experiment-dir.")] = None,
) -> None:
    """Render and print the investigator's system + user prompt for the debug use-case."""
    repo_root = Path(__file__).resolve().parents[2]
    extra_path = repo_root / "src/cube_harness/auto_cube/use_cases/debug/investigator_extra.md"
    extra_fragment = extra_path.read_text() if extra_path.exists() else None

    tmp: tempfile.TemporaryDirectory[str] | None = None
    try:
        if experiment_dir is not None:
            if episode is None:
                print("SMOKE FAIL: investigator_prompt_preview — --episode required with --experiment-dir")
                raise typer.Exit(1)
            if not (experiment_dir / INVESTIGATION_CONTEXT_FILENAME).exists():
                print(
                    f"SMOKE SKIP: investigator_prompt_preview — no {INVESTIGATION_CONTEXT_FILENAME} in {experiment_dir}"
                )
                raise typer.Exit(2)
            exp, traj_id = experiment_dir, episode
        else:
            tmp = tempfile.TemporaryDirectory()
            exp, traj_id = _build_fixture(Path(tmp.name))

        call = _capture_prompt(exp, traj_id, extra_fragment)

        print("=" * 72)
        print("SYSTEM PROMPT")
        print("=" * 72)
        print(call["system_prompt"])
        print()
        print("=" * 72)
        print("USER PROMPT")
        print("=" * 72)
        print(call["user_prompt"])
        print()
        print("SMOKE OK: investigator_prompt_preview")
    except typer.Exit:
        raise
    except Exception as e:  # noqa: BLE001 - smoke wants the failure visible
        print(f"SMOKE FAIL: investigator_prompt_preview — {type(e).__name__}: {e}")
        sys.exit(1)
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    typer.run(main)
