"""Sub-agent that auto-generates the investigation_context.md codebase map.

When `_investigate_episode_impl` runs and no cached context file exists at the
path `resolve_context_path` picks (per-experiment, or per-session when Auto-CUBE
sets `context_dir`), this agent walks `experiment_config.json`, identifies the
cube package, agent package, and `cube_harness` source, explores them, and emits
an architecture orientation + key-location pointers + a ```paths fenced block in
the format `validate_context_file` already parses.

A driver is required — the previous "no driver, use a venv-walk heuristic"
fallback was speculative and never used in practice. Callers without a
driver have no business invoking this function.

A thin CLI wrapper (`ch-investigate init-context`) calls the same function for
ad-hoc bootstrap.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from cube_harness.analyze.investigator.agent_driver import AgentDriver
from cube_harness.analyze.investigator.context import _PATHS_FENCE_RE, INVESTIGATION_CONTEXT_FILENAME

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_MODEL = "claude-opus-4-7"

BENCHMARK_CONTEXT_SYSTEM_PROMPT = """You are a setup agent for the cube-harness trajectory investigator.

A Sonnet-class investigator will read agent episodes and attribute failures to a
fixed blame taxonomy, grounding every claim in source code. You run **once** (you
are Opus-class) to give it a strong head-start: a map of the codebase it will
navigate, so it lands on the right files instead of grepping blind. Your output
is read directly into the investigator's prompt — make it a genuine orientation,
not just a list of directories.

You have read-only tools (Read / Glob / Grep / Bash). Do not write files via
Bash — your only output is the assistant message containing the markdown.

## Procedure

1. Read `experiment_config.json` in the working directory. It is JSON with
   `_type` strings naming the agent class and benchmark class (full dotted paths),
   and possibly an `infra._type`.
2. For each `_type`, resolve the on-disk package directory:
   `python -c "import importlib.util as u; print(u.find_spec('PKG').origin)"`
   returns a file path; the parent directory is what you want.
3. Always resolve the `cube_harness` source root and the `cube` (cube-standard)
   source root. Include the infra package root if `infra._type` is present.
4. **Explore** (this is the value you add): open the resolved packages and find
   the entry points the investigator will most likely need —
   - the agent loop (where the LLM is called, where actions are parsed/executed),
   - the tool wrapper(s) for this benchmark (the action surface),
   - the task's reward / evaluate function,
   - task setup / reset (the initial observation),
   - infra entrypoint (how the container/VM is provisioned),
   - the submission protocol (how the agent signals "done").
   Note the `path:symbol` (file + function/class) for each, verified by actually
   reading enough to be sure.
5. Verify every path you cite exists. Skip anything missing — better to omit than
   to hallucinate.

## Output format

A markdown document with three parts, in this order:

1. **Architecture orientation** (≈ 5–12 sentences): how cube-standard defines the
   contract (Task / Tool / Benchmark / Resource), how cube-harness runs it (agent
   loop → episode → trajectory), then the specifics of *this* benchmark — what the
   task is, what the action surface looks like, how reward is computed. Orient,
   don't transcribe; the investigator will drill in itself.
2. **Key locations**: a short bullet list of `path:symbol — what it is` for the
   entry points from step 4. These are the head-start pointers.
3. A fenced ```paths block of the package roots (this part is machine-parsed —
   keep the exact format). Each line is `name: /absolute/path`:

```paths
cube_package: /abs/path/to/cubes/swebench_verified
agent_package: /abs/path/to/cube_harness/agents
cube_harness: /abs/path/to/src/cube_harness
cube_standard: /abs/path/to/cube
```

Keep the whole document tight — a head-start, not a manual. The investigator can
open any file it needs; your job is to point it at the right ones fast.

Reply with the markdown content only — no preamble, no closing chatter."""


def _user_prompt_for(experiment_dir: Path) -> str:
    """Build the per-experiment user prompt for the context sub-agent."""
    return f"""Experiment directory: {experiment_dir}

Read `experiment_config.json` from this directory and produce `investigation_context.md`
contents per the procedure in the system prompt. Reply with the markdown only."""


def _extract_markdown(output_text: str) -> str:
    """Extract the markdown body from the agent's response.

    The agent is instructed to reply with markdown only, but in practice it may
    wrap its answer in a ```markdown fence or add preamble. We try to find a
    fenced markdown block first; failing that, we look for a `paths` fence and
    keep everything from the start of the message up through it.
    """
    fence_md = re.search(r"```(?:markdown|md)\s*\n(.*?)```", output_text, re.DOTALL | re.IGNORECASE)
    if fence_md:
        return fence_md.group(1).strip() + "\n"
    if _PATHS_FENCE_RE.search(output_text):
        return output_text.strip() + "\n"
    raise ValueError("benchmark-context-agent did not emit a ```paths block")


async def generate_context_file(
    experiment_dir: Path,
    *,
    driver: AgentDriver,
    model: str = DEFAULT_CONTEXT_MODEL,
    verbose: bool = False,
    out_path: Path | None = None,
) -> Path:
    """Invoke the sub-agent and write the `investigation_context.md`.

    Writes to `out_path` if given (Auto-CUBE points this at a per-session cache),
    else `<experiment_dir>/investigation_context.md`. The driver is required —
    there is no offline / no-driver fallback.
    """
    experiment_dir = Path(experiment_dir).resolve()
    out = Path(out_path) if out_path is not None else experiment_dir / INVESTIGATION_CONTEXT_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)

    user_prompt = _user_prompt_for(experiment_dir)
    result = await driver.run(
        system_prompt=BENCHMARK_CONTEXT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        cwd=experiment_dir,
        additional_dirs=[],
        model=model,
        verbose=verbose,
    )
    markdown = _extract_markdown(result.output_text)
    out.write_text(markdown)
    logger.info("benchmark-context-agent wrote %s", out)
    return out


__all__ = [
    "BENCHMARK_CONTEXT_SYSTEM_PROMPT",
    "DEFAULT_CONTEXT_MODEL",
    "generate_context_file",
]
