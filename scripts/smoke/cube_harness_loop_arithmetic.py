# /// script
# requires-python = ">=3.12"
# dependencies = ["cube-harness", "arithmetic-cube"]
#
# [tool.uv.sources]
# cube-harness = { path = "../..", editable = true }
# arithmetic-cube = { path = "../../cubes/arithmetic-cube", editable = true }
# ///
"""SMOKE: the full cube-harness agent loop over a REAL cube, with NO LLM and NO infra.

Drives the arithmetic cube (4 tasks) through `Episode` -> `build_agent_tools` ->
`MonitoredTool` -> `AgentView.execute_action` -> per-step eval callback -> `_post_action`
-> `finished()`/`AgentStop`, with a deterministic solver agent (parses the question, submits
the answer). Exercises the streamable-task contract end-to-end against a real Task without
a model or container — the cheap guard that would have caught a cube construction/contract
regression (e.g. a recursive `tool` property) that unit tests over a mock Task miss.

Prints `SMOKE OK/FAIL: cube_harness_loop_arithmetic` and exits 0/1.
"""

import glob
import json
import re
import tempfile

from arithmetic_cube import ARITHMETIC_CONFIGS
from cube.core import Action, ActionSchema, Observation

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentOutput
from cube_harness.exp_runner import run_sequentially
from cube_harness.experiment import Experiment

NAME = "cube_harness_loop_arithmetic"
_Q = re.compile(r"(-?\d+)\s*([+\-*])\s*(-?\d+)")


class SolverAgent(Agent):
    """Deterministic: read 'What is A op B?' from the obs, submit the answer. No LLM."""

    name = "arithmetic-solver"
    description = "solves arithmetic deterministically"
    input_content_types: list[str] = ["text"]
    output_content_types: list[str] = ["action"]

    def step(self, obs: Observation) -> AgentOutput:
        m = _Q.search(obs.to_markdown())
        if not m:
            return AgentOutput(actions=[Action(name="final_step", arguments={})])
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        ans = {"+": a + b, "-": a - b, "*": a * b}[op]
        return AgentOutput(actions=[Action(name="submit_answer", arguments={"answer": ans})])


class SolverAgentConfig(AgentConfig):
    def make(self, action_set: "list[ActionSchema] | None" = None, **kwargs: object) -> Agent:
        return SolverAgent(self)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        exp = Experiment(
            name="smoke-arith",
            agent_config=SolverAgentConfig(),
            benchmark_config=ARITHMETIC_CONFIGS["default"],
            output_dir=f"{tmp}/run",  # resolved to a unique sibling `<...>/run_<hash>` inside tmp
            max_steps=4,
            is_official=False,
        )
        run_sequentially(exp)
        # `exp.output_dir` is the resolved unique dir; episodes live under it.
        metas = glob.glob(f"{exp.output_dir}/**/episode.metadata.json", recursive=True)
        rewards = []
        for mfile in metas:
            ri = json.load(open(mfile)).get("reward_info") or {}
            rewards.append(ri.get("reward"))

    n_ok = sum(1 for r in rewards if r == 1.0)
    if metas and n_ok == len(metas) == 4:
        print(f"SMOKE OK: {NAME} solved {n_ok}/{len(metas)} arithmetic tasks through the harness loop (no LLM)")
        return 0
    print(f"SMOKE FAIL: {NAME} rewards={rewards} (expected four 1.0); episodes={len(metas)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
