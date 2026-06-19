# timewarp-cube

[TimeWarp](https://arxiv.org/abs/2603.04949) ported to the [CUBE](../../) protocol — 231 web tasks across **wiki**, **news**, and **webshop**, designed to test agent robustness to *temporal* changes in web UI.

## Overview

TimeWarp recreates three web environments in multiple historical UI versions (different eras of design and layout) and asks the agent to complete realistic navigation/information tasks against them. The agent submits its final answer through a `ChatTool`; every task is scored by an LLM judge (`llm_judge`, OpenAI), so a chat answer and `OPENAI_API_KEY` are both required for a non-zero reward.

The three environments are **external Flask servers** that you start yourself — this cube runs in **manual mode**. There is no Docker provisioning here: `TimeWarpBenchmark._setup()` only verifies that `TW_WIKI` / `TW_NEWS` / `TW_WEBSHOP` are set and reachable. (Same pattern as [`webarena-verified`](../webarena-verified).)

## Prerequisites

- The upstream **TimeWarp** repo, which ships the environment servers and start scripts: <https://github.com/sparklabutah/timewarp>. The servers are **not** part of the `browsergym-timewarp` Python package or this cube.
- An **`OPENAI_API_KEY`** for the `llm_judge` evaluator (create one at <https://platform.openai.com/api-keys>).

## Installation

```bash
uv pip install timewarp-cube   # not yet on PyPI — see pyproject for the git source
```

## Setup → run

### 1. Start the servers (from the upstream TimeWarp checkout)

```bash
git clone https://github.com/sparklabutah/timewarp && cd timewarp
bash setup.sh                              # one-time: conda env + dependencies
bash scripts/environment/run_all_env.sh 1  # start wiki + news + webshop at UI version 1
# ... run experiments ...
bash scripts/environment/stop_all_ports.sh # stop all servers when done
```

`run_all_env.sh [1-6]` selects the UI era (default `1`). See the upstream README for exact ports and per-environment scripts (`env/wiki/start_wiki.sh`, …).

### 2. Point the cube at the servers and the judge

```bash
export TW_WIKI=http://localhost:<wiki-port>
export TW_NEWS=http://localhost:<news-port>
export TW_WEBSHOP=http://localhost:<webshop-port>/abc
export OPENAI_API_KEY=sk-...
```

### 3. Run

```bash
make debug                  # python -m timewarp_cube.debug — scripted end-to-end smoke
make test                   # cube test timewarp-cube
```

Programmatically:

```python
from timewarp_cube import TIMEWARP_CONFIGS

cfg = TIMEWARP_CONFIGS["wiki"]   # "default" | "wiki" | "news" | "webshop"
bench = cfg.make()               # verifies the servers are reachable (manual mode)
for task_cfg in cfg.get_task_configs():
    task = bench.spawn(task_cfg)
    obs, _ = task.reset()
    # ... agent loop: task.step(action) until env_out.done ...
    task.close()
bench.close()
```

## Named subsets

Filter by environment via named subsets (or `subset_from_glob("sites", "*wiki*")`):

```python
TimeWarpBenchmarkConfig(tool_config=...).named_subset("wiki")   # or "news" / "webshop"
```

| Subset    | Tasks |
| --------- | ----- |
| `default` | 231   |
| `wiki`    | 111   |
| `news`    | 89    |
| `webshop` | 95    |

Subsets **overlap** — a task may require more than one environment (e.g. `sites=['wiki', 'news']`), so the per-site counts sum to more than 231 while their union covers all 231 tasks.

## Environment variables

| Variable         | Required | Purpose                                                                     |
| ---------------- | -------- | --------------------------------------------------------------------------- |
| `TW_WIKI`        | yes      | URL of the running wiki server                                              |
| `TW_NEWS`        | yes      | URL of the running news server                                              |
| `TW_WEBSHOP`     | yes      | URL of the running webshop server (note the `/abc` path in upstream)        |
| `OPENAI_API_KEY` | for reward | Consumed by the `llm_judge` evaluator that scores every task              |

## Debug / Testing

```bash
make debug                  # end-to-end smoke against the live servers (needs the env above)
uv run pytest tests/        # fast unit tests — no servers, no browser, no API key
```

The unit tests in [`tests/`](tests/) cover the parts that don't need infrastructure: metadata loads 231 tasks, the named subsets filter/cover correctly, the configs round-trip, and the toolbox pairs a browser tool with a `ChatTool`. The [`debug.py`](src/timewarp_cube/debug.py) suite exercises the full setup→validate path against the live servers with a scripted reference-answer agent.

`task_metadata.json` is a shipped package resource holding only public fields (`sites`, `intent_template_id`, `eval_types`). TimeWarp has no heavy execution data — all task logic loads from `browsergym-timewarp` at runtime via the numeric task id. To regenerate it after a task-list change (developer use only):

```bash
uv run scripts/generate_task_metadata.py --force
```

## Attribution & License

This cube wraps two upstream projects; please cite/observe their terms:

- **TimeWarp** — the benchmark, environments, and tasks (MIT). Upstream: <https://github.com/sparklabutah/timewarp>.
- **BrowserGym** (`browsergym-timewarp`) — the `GenericTimeWarpTask` implementation (Apache-2.0, ServiceNow). Upstream: <https://github.com/ServiceNow/BrowserGym>.

```bibtex
@misc{timewarp2026,
      title={TimeWarp: Evaluating Web Agents by Revisiting the Past},
      author={Md Farhan Ishmam and Kenneth Marino},
      year={2026},
      eprint={2603.04949},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.04949}
}
```

The cube code itself is distributed under the cube-harness repository license. (No standalone `LICENSE` ships in this directory — consistent with the other cubes; confirm upstream-license attribution when filing the cube-registry entry.)

## Package structure

```
src/timewarp_cube/
├── __init__.py          # Public exports
├── benchmark.py         # TimeWarpBenchmark / TimeWarpBenchmarkConfig (manual-mode setup)
├── task.py              # TimeWarpTask, TimeWarpTaskConfig, TimeWarpTaskMetadata
├── configs.py           # TIMEWARP_CONFIGS registry (default / wiki / news / webshop)
├── debug.py             # Scripted reference-answer debug suite
├── _data.py             # Cached accessor for the browsergym-timewarp raw task data
└── task_metadata.json   # Shipped public-field metadata for all 231 tasks
```
