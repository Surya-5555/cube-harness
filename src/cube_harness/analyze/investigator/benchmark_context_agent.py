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

The investigator will read agent episodes and attribute failures to a fixed
blame taxonomy, grounding every claim in source code. You run **once per
session** to give it a strong head-start: a map of the codebase it will
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
   the entry points the investigator will most likely need. **Cubes lay these
   out very differently — search, do not assume a layout, and stay
   cube-agnostic: let *this* benchmark's source drive the map, never a template
   carried over from another cube.** Find:
   - the agent loop (where the LLM is called, where actions are parsed/executed),
   - the tool wrapper(s) for this benchmark (the action surface),
   - **the upstream benchmark, if this cube wraps one.** Many cubes are thin
     adapters over an installed third-party benchmark package — the real task
     definitions, environment, and scoring live *upstream* and the cube
     delegates to them (e.g. a `workarena`/`browsergym`-style package). Resolve
     that upstream package on disk and map it too: pointers should follow the
     delegation all the way to where the real task/eval logic lives, not stop
     at the cube wrapper. Find it by grepping the cube's imports.
   - **the verifier — the `evaluate` / reward function: where and exactly how a
     solution is scored. Pin the `path:symbol`** (this may be in the cube *or*
     delegated to the upstream package — follow it).
   - **where the ground truth / expected answer lives — this varies per cube: it
     may be a field in the task metadata, on the task config, a file staged in
     the container, or computed inline in `evaluate`. Locate it concretely so
     the investigator can check the agent's answer against it — this is what
     distinguishes a real failure from `eval_brittle` / `should_have_been_rewarded`.**
   - task setup / reset (the initial observation),
   - infra entrypoint (how the container/VM is provisioned),
   - the submission protocol (how the agent signals "done").
   Note the `path:symbol` (file + function/class, or the metadata key/path) for
   each, verified by actually reading enough to be sure.
5. Verify every path you cite exists. Skip anything missing — better to omit than
   to hallucinate.

## Output format

The investigator is likely **unfamiliar with the cube codebase** — your
output is its orientation. The whole document is embedded directly into the
investigator's prompt, so do not refer to it as a file ("see investigation_context.md");
just write the content. Use `##` headings (not `#`) so it nests cleanly when
embedded.

Three parts, in this order:

### `## Orientation`

Teach the investigator the codebase. Be generous — several paragraphs is the
right length; a strong orientation saves the investigator many grep/read
round-trips.

- **cube-standard** — the protocol/contract layer. Explain the pieces it will
  meet: `Task` (reset / step / evaluate), `Tool` (`@tool_action`), `Benchmark`,
  `Resource` (provisioned infra), `Container`, `Server`, `CLI`. Benchmarks and
  tools *implement* these ABCs; generalist tools (e.g. the terminal tool) live
  in cube-standard and are shared.
- **cube-harness** — the runtime. The agent (the LLM loop that emits actions),
  `Episode` (drives agent ⇄ env, enforces budget), `Trajectory` (the recorded
  result the investigator reads), `exp_runner`, storage. Note that the
  investigator itself lives here.
- **this benchmark** — what the task is, the action surface (which tools the
  agent had), how an episode starts (initial observation / reset), **how a
  solution is verified (the evaluate function) and where the ground truth /
  expected answer lives** (so the investigator can check the agent's answer
  itself), and any infra specifics (container/VM). These vary per cube — say
  what *this* one does. **If the cube wraps an upstream benchmark package, say
  so and note that the real task/eval logic lives upstream** (the cube is the
  adapter).

### `## Key locations`

A **tree** rooted at each package's on-disk directory, annotating the files and
symbols the investigator is most likely to need. Show **at least the absolute
root path** of each package; entries beneath may be relative. Cover: the agent
loop, the tool wrappers (benchmark + shared), the **verifier / `evaluate`
function**, **where the ground truth lives**, task `reset`, the infra
entrypoint, and the submission protocol. The example below is *illustrative
shape only* — the real files differ per benchmark; map what you actually find:

    /abs/path/to/cube-standard/src/cube/          (cube-standard root)
    ├── task.py                  — Task ABC: reset() / step() / evaluate()
    ├── tool.py                  — Tool ABC, @tool_action
    └── tools/terminal.py        — TerminalTool: shared shell action surface

    /abs/path/to/cube-harness/src/cube_harness/   (cube-harness root)
    ├── agents/<agent>.py        — <Agent>.run: the agent loop
    └── episode.py               — Episode: drives agent ⇄ env, budget

    /abs/path/to/cubes/<this-cube>/src/<pkg>/     (benchmark root — the adapter)
    ├── task.py:evaluate         — VERIFIER (or where it delegates upstream)
    ├── task.py / metadata       — GROUND TRUTH: where the expected answer lives
    └── tools.py                 — benchmark-specific actions (if any)

    /abs/path/to/<upstream-pkg>/                  (upstream benchmark, if wrapped)
    ├── ...:<eval fn>            — the real scoring the cube delegates to
    └── ...                      — real task definitions / environment

### `## Paths`

A fenced ```paths block of the package roots — **machine-parsed, keep this exact
format**. One `name: /absolute/path` per line. Verify each exists. Include an
`upstream_package` entry when the cube wraps one:

```paths
cube_standard: /abs/path/to/cube-standard/src/cube
cube_harness: /abs/path/to/cube-harness/src/cube_harness
cube_package: /abs/path/to/cubes/<this-cube>/src/<pkg>
agent_package: /abs/path/to/cube_harness/agents
upstream_package: /abs/path/to/<wrapped-benchmark>
```

The orientation may be long; the tree must be precise. The investigator opens
any file it needs — your job is the head-start.

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
