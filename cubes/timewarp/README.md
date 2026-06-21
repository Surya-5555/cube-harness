# timewarp-cube

[TimeWarp](https://arxiv.org/abs/2603.04949) ported to the [CUBE](../../) protocol — 231 web tasks across **wiki**, **news**, and **webshop**, designed to test agent robustness to *temporal* changes in web UI.

## Overview

TimeWarp recreates three web environments in multiple historical UI versions (different eras of design and layout) and asks the agent to complete realistic navigation/information tasks against them. The agent submits its final answer through a `ChatTool`; every task is scored by an LLM judge (`llm_judge`, OpenAI), so a chat answer and `OPENAI_API_KEY` are both required for a non-zero reward.

The three environments are **Flask servers** — no Docker. The cube provisions and runs them
for you (`provision_mode="auto"`, the default); set `provision_mode="manual"` if you'd rather
start them yourself.

- **auto** (default) — on `_setup()` the cube checks whether the upstream environment is already
  set up (repo cloned, `timewarp` conda env built, the Google-Drive + HuggingFace data present).
  If not, it runs the upstream **idempotent** `setup.sh` to download/build everything, then
  launches the three Flask apps (at `ui_version`, 1–6) and waits until they're healthy. Resolved
  URLs are published into the benchmark's `runtime_context` (re-derived every run, so a resumed
  run picks up the freshly-launched ports), so this works under Ray on a single host. If
  `TW_WIKI` / `TW_NEWS` / `TW_WEBSHOP` already point at reachable servers, those are reused and
  nothing is launched. **Requires `conda`** (the servers run in the `timewarp` env the upstream
  `setup.sh` creates).
- **manual** — start the servers yourself and set the three env vars; `_setup()` only verifies
  reachability. (Same pattern as [`webarena-verified`](../webarena-verified).)

## Prerequisites

- **`conda`** (auto mode only) — the servers run in the `timewarp` conda env that upstream
  `setup.sh` builds. The upstream **TimeWarp** repo (<https://github.com/sparklabutah/timewarp>)
  is cloned automatically; the servers are **not** part of the `browsergym-timewarp` package or
  this cube.
- An **`OPENAI_API_KEY`** for the `llm_judge` evaluator (create one at <https://platform.openai.com/api-keys>).

## Installation

```bash
uv pip install timewarp-cube   # not yet on PyPI — see pyproject for the git source
cube install timewarp-cube     # auto mode: one-time clone + setup.sh (conda env + ~GBs of data)
```

`cube install` does the slow once-per-machine L1 prep ahead of time; it's optional (auto mode
runs the same idempotent step on first use), but doing it upfront makes the first run fast. The
upstream repo is cloned to `~/.cache/timewarp` — override with the `TIMEWARP_HOME` env var.

## Setup → run

### Auto (default)

```bash
export OPENAI_API_KEY=sk-...        # for the llm_judge reward
make debug                          # provisions if needed, launches servers, runs the smoke
```

```python
from timewarp_cube import TIMEWARP_CONFIGS

cfg = TIMEWARP_CONFIGS["wiki"]      # "default" | "wiki" | "news" | "webshop"
bench = cfg.make()                  # auto-provisions + launches the Flask servers, waits healthy
for task_cfg in bench.config.get_task_configs():
    task = bench.spawn(task_cfg)
    obs, _ = task.reset()
    # ... agent loop: task.step(action) until env_out.done ...
    task.close()
bench.close()                       # stops the servers it launched
```

Pick the UI era and checkout dir on the config:

```python
from timewarp_cube import TimeWarpBenchmarkConfig
cfg = TimeWarpBenchmarkConfig(tool_config=..., ui_version=3)   # temporal UI era 1–6
```

### Manual

Start the servers from the upstream checkout and point the cube at them:

```bash
git clone https://github.com/sparklabutah/timewarp && cd timewarp
bash setup.sh                              # one-time: conda env + dependencies
bash scripts/environment/run_all_env.sh 1  # start wiki + news + webshop at UI version 1
export TW_WIKI=http://localhost:<wiki-port>
export TW_NEWS=http://localhost:<news-port>
export TW_WEBSHOP=http://localhost:<webshop-port>/abc
export OPENAI_API_KEY=sk-...
bash scripts/environment/stop_all_ports.sh # stop all servers when done
```

```python
TimeWarpBenchmarkConfig(tool_config=..., provision_mode="manual").make()  # verifies reachability
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
| `TW_WIKI`        | manual mode | URL of the wiki server. Auto mode sets it for you; if already set+reachable, auto reuses it. |
| `TW_NEWS`        | manual mode | URL of the news server (same auto behaviour).                            |
| `TW_WEBSHOP`     | manual mode | URL of the webshop server, with the `/abc` path (same auto behaviour).   |
| `TIMEWARP_HOME`  | no       | Auto mode: where the upstream repo is cloned (default `~/.cache/timewarp`).  |
| `OPENAI_API_KEY` | for reward | Consumed by the `llm_judge` evaluator that scores every task              |

## Debug / Testing

```bash
make debug                  # end-to-end smoke; auto mode provisions + launches the servers
uv run pytest tests/        # fast unit tests — no servers, no browser, no conda, no API key
```

The unit tests in [`tests/`](tests/) cover the parts that don't need infrastructure: metadata loads 231 tasks, the named subsets filter/cover correctly, the configs round-trip, the toolbox pairs a browser tool with a `ChatTool`, and the provisioning helpers (mode toggle, `is_provisioned` completeness checks, the auto-launch path, `install()` short-circuits, server teardown, and `runtime_context` URL threading) with subprocess/socket calls mocked. The [`debug.py`](src/timewarp_cube/debug.py) suite exercises the full setup→validate path against live servers (auto mode launches them) with a scripted reference-answer agent.

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
├── benchmark.py         # TimeWarpBenchmark / TimeWarpBenchmarkConfig (auto + manual modes)
├── provisioning.py      # Non-Docker auto-provisioning: clone + setup.sh (L1), launch servers (L2)
├── task.py              # TimeWarpTask, TimeWarpTaskConfig, TimeWarpTaskMetadata
├── configs.py           # TIMEWARP_CONFIGS registry (default / wiki / news / webshop)
├── debug.py             # Scripted reference-answer debug suite
├── _data.py             # Cached accessor for the browsergym-timewarp raw task data
└── task_metadata.json   # Shipped public-field metadata for all 231 tasks
```
