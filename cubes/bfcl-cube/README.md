# bfcl-cube

A CUBE wrapper for the [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
(BFCL v4) — the recognized function-calling benchmark. This cube covers the
**Python single-turn, AST-scored** categories: an agent reads a user query plus a
per-task set of function schemas and emits the function call(s) it would make;
the calls are matched against BFCL's ground truth.

No infrastructure: tasks score in-process (no Docker, no network), which makes
the cube fast, CI-friendly, and RL-friendly.

## Categories (3491 tasks)

| Collection (`named_subset`) | Categories | Scoring |
|---|---|---|
| `non_live` (1240) | `simple_python`, `multiple`, `parallel`, `parallel_multiple`, `irrelevance` | AST match / abstention |
| `live` (2251) | `live_simple`, `live_multiple`, `live_parallel`, `live_parallel_multiple`, `live_irrelevance`, `live_relevance` | AST match / (ir)relevance |

`live_*` here means *real-world-sourced* function schemas — these are **offline**,
AST-scored tasks (not network calls).

## How it maps onto CUBE

- **Tool** (`BfclTool`) — a *data-driven* function-calling surface. Each task has
  its own functions, so `action_set` is built per task from the task's schemas
  (BFCL types → OpenAI/JSON-Schema), rather than from fixed `@tool_action`
  methods. `execute_action` **records** each call; single-turn BFCL never
  executes the functions. `final_step` is the STOP / abstain action.
- **Task** (`BfclTask`) — `reset()` presents the query; `evaluate()` AST-matches
  the recorded calls against the ground truth (or checks call presence/absence for
  the (ir)relevance categories). **Termination is category-aware** (single-turn
  BFCL scores one model response): exactly-one-call and abstain categories end
  after the agent's first call, so a model that re-issues a correct call across
  steps is not over-counted; the **parallel** categories accumulate calls across
  steps (harness agents emit one call per step) and end on `final_step`.
- **Benchmark** (`BfclBenchmarkConfig`) — lightweight metadata ships in
  `task_metadata.json`; heavy per-task data (question, function schemas, ground
  truth) ships gzipped and is unpacked into the per-task cache by `install()`.

## Scoring fidelity

The AST checker and the BFCL→OpenAI schema converter are **vendored** (trimmed to
Python) from `bfcl_eval` under `src/bfcl_cube/_vendor/` — see that package's
docstring. We vendor rather than depend on `bfcl-eval` because its import graph
pulls in every model-provider SDK plus vllm/torch/faiss, none of which a scoring
cube needs. The checker logic is a faithful port; re-diff it against the matching
BFCL release when bumping the data.

## Usage

```python
from bfcl_cube import BFCL_CONFIGS
benchmark = BFCL_CONFIGS["default"]    # all 3491
benchmark = BFCL_CONFIGS["non_live"]   # 1240
benchmark = BFCL_CONFIGS["live"]       # 2251
```

```bash
make install        # venv + editable install
make test           # cube test bfcl-cube (debug suite, reward==1.0)
uv run pytest       # unit tests (checker, schema convert, config contract)
make gen            # regenerate committed data from BFCL v4 (author-time)
```

## Scope & follow-ups

This cube is the offline single-turn core. Deliberately **out of scope** here,
tracked as follow-ups:

- **Java / JavaScript** `simple_*` categories — need the tree-sitter type
  converters; the vendored checker is Python-only.
- **`multi_turn_*`** — stateful, multi-step; needs BFCL's backend simulators +
  state checker. The tool/task design accommodates it (no schema change), but it
  is a separate effort.
- **`bfcl-agentic`** (`web_search_*`, `memory_*`) — the genuinely network/creds
  -dependent categories. These belong in a separate cube so this one stays fully
  offline; they can't reach `reward==1.0` in a debug suite without live services.

## License & attribution

Cube wrapper: see repository license. The BFCL dataset and the vendored checker
are from [gorilla](https://github.com/ShishirPatil/gorilla) (BFCL), Apache-2.0;
redistributed under that license. Please cite the BFCL authors when reporting
results.
