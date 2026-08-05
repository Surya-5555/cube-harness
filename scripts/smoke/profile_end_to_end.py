#!/usr/bin/env python3
"""SMOKE: episode profiling end-to-end through the real sequential runner.

Runs a tiny **arithmetic-cube** experiment (pure-python, no infra, no LLM) with
`Experiment(profile=ProfileConfig(...))` and verifies the full profiling
pipeline the Auto-CUBE profile use-case relies on:

  - every solved episode writes a `profile.json` beside its trajectory;
  - it carries all four phases (setup / agent_loop / evaluate / teardown) with
    non-negative wall-clock, plus host resource series (cpu_percent, rss_mb);
  - `ch-profile`'s `build_rollup` aggregates them into a phase Pareto table whose
    shares are sane and whose owner tags resolve;
  - a run WITHOUT profiling writes no profile.json (zero footprint when off).

The scripted agent replays arithmetic-cube's own deterministic debug actions, so
rewards are real (1.0) and the agent_loop phase contains genuine tool dispatch —
this exercises the real Episode body, not a stub.

Run:
    .venv/bin/python scripts/smoke/profile_end_to_end.py

Prints SMOKE OK|FAIL|SKIP: profile_end_to_end  (exit 0|1|2).
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

from cube.core import Action, ActionSchema, Observation

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentOutput
from cube_harness.exp_runner import run_sequentially
from cube_harness.experiment import Experiment
from cube_harness.metrics.profile_rollup import build_rollup
from cube_harness.metrics.profiler import (
    PHASE_AGENT_LOOP,
    PHASE_EVALUATE,
    PHASE_SETUP,
    PHASE_TEARDOWN,
    EpisodeProfile,
    ProfileConfig,
)

NAME = "profile_end_to_end"


class _ScriptedArithmeticAgentConfig(AgentConfig):
    """Drives arithmetic-cube via its own deterministic debug actions (no LLM)."""

    name: str = "scripted_arithmetic"

    def make(self, action_set: list[ActionSchema] | None = None, **kwargs) -> "Agent":
        _ = action_set
        task_id = str(kwargs.get("task_id", ""))
        return _ScriptedAgent(config=self, task_id=task_id)


class _ScriptedAgent(Agent):
    name = "ScriptedArithmeticAgent"
    description = "Replays arithmetic-cube debug actions, then stops."
    input_content_types = ["text"]
    output_content_types = ["action"]

    def __init__(self, config: _ScriptedArithmeticAgentConfig, task_id: str) -> None:
        super().__init__(config)
        from arithmetic_cube.debug import make_debug_agent  # noqa: PLC0415 - cube only needed for this smoke

        self._debug = make_debug_agent(task_id)
        self._first = True

    def step(self, obs: Observation) -> AgentOutput:
        # Spend a beat on the first step so the agent_loop phase lasts long
        # enough for the background resource sampler to capture a few ticks
        # (arithmetic tasks otherwise finish in ~1ms — faster than one sample).
        if self._first:
            self._first = False
            time.sleep(0.2)
        try:
            return AgentOutput(actions=[self._debug.get_action(obs)])
        except StopIteration:
            return AgentOutput(actions=[Action(name="final_step", arguments={})])


def _run(out_dir: Path, *, profile: ProfileConfig | None) -> Experiment:
    from arithmetic_cube import ARITHMETIC_CONFIGS  # noqa: PLC0415 - smoke-only cube dependency

    exp = Experiment(
        name=f"{NAME}_{'on' if profile else 'off'}",
        output_dir=out_dir,
        agent_config=_ScriptedArithmeticAgentConfig(),
        benchmark_config=ARITHMETIC_CONFIGS["default"],
        infra=None,  # arithmetic-cube needs no shared infrastructure
        max_steps=4,
        is_official=False,
        profile=profile,
    )
    run_sequentially(exp)
    return exp


def main() -> int:
    try:
        import arithmetic_cube  # noqa: F401, PLC0415
    except ImportError:
        print(f"SMOKE SKIP: {NAME} (arithmetic-cube not installed — `uv pip install -e cubes/arithmetic-cube`)")
        return 2

    workdir = Path(tempfile.mkdtemp(prefix="profile-smoke-"))
    try:
        # --- 1. profiling ON ---
        on_dir = workdir / "on"
        exp_on = _run(on_dir, profile=ProfileConfig(sample_hz=20))
        profiles = list((exp_on.output_dir / "episodes").glob("*/profile.json"))
        if not profiles:
            print(f"SMOKE FAIL: {NAME} — no profile.json written under {exp_on.output_dir}/episodes/")
            return 1

        prof = EpisodeProfile.load(profiles[0].parent)
        assert prof is not None, "profile.json failed to load"
        for phase in (PHASE_SETUP, PHASE_AGENT_LOOP, PHASE_EVALUATE, PHASE_TEARDOWN):
            if phase not in prof.phases:
                print(f"SMOKE FAIL: {NAME} — missing phase {phase!r} in {prof.phases}")
                return 1
            if prof.phases[phase] < 0:
                print(f"SMOKE FAIL: {NAME} — negative duration for {phase}: {prof.phases[phase]}")
                return 1
        if "cpu_percent" not in prof.resources or "rss_mb" not in prof.resources:
            print(f"SMOKE FAIL: {NAME} — resource series missing: {list(prof.resources)}")
            return 1

        # --- 2. rollup aggregates ---
        rollup = build_rollup(exp_on.output_dir)
        if rollup["episodes"] != len(profiles):
            print(f"SMOKE FAIL: {NAME} — rollup saw {rollup['episodes']} != {len(profiles)} profiles")
            return 1
        share_sum = sum(p["share"] for p in rollup["phases"])
        if not (0.0 < share_sum <= 1.01):
            print(f"SMOKE FAIL: {NAME} — phase shares out of range: {share_sum}")
            return 1
        owners = {p["owner"] for p in rollup["phases"]}
        if "?" in owners:
            print(f"SMOKE FAIL: {NAME} — unresolved owner tag in {rollup['phases']}")
            return 1

        # --- 3. profiling OFF leaves no footprint ---
        off_dir = workdir / "off"
        exp_off = _run(off_dir, profile=None)
        if list((exp_off.output_dir / "episodes").glob("*/profile.json")):
            print(f"SMOKE FAIL: {NAME} — profile.json written when profile is None")
            return 1

        top = rollup["phases"][0]
        print(
            f"SMOKE OK: {NAME} — {rollup['episodes']} episodes profiled; "
            f"top phase {top['phase']}={top['mean_s']:.3f}s ({top['share'] * 100:.0f}%, {top['owner']}); "
            f"resources={list(prof.resources)}; off-run wrote no profile.json"
        )
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
