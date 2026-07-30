# timewarp-cube

[TimeWarp](https://arxiv.org/abs/2603.04949) ported to the [CUBE](../../) protocol — 231 web tasks across **wiki**, **news**, and **webshop**, designed to test agent robustness to *temporal* changes in web UI.

## Overview

TimeWarp recreates three web environments in multiple historical UI versions (different eras of design and layout) and asks the agent to complete realistic navigation/information tasks against them. The agent submits its final answer through a `ChatTool`, so a chat answer is required for a non-zero reward. Since upstream v0.2.0, 229 of the 231 tasks are scored by **deterministic verifiers** (`string_match` / `number_match` / `list_match`); only tasks 32 and 143 still use an LLM judge, so `OPENAI_API_KEY` is **optional**.

> **Upstream version.** This cube pins TimeWarp [`v0.2.0`](https://github.com/sparklabutah/timewarp/releases/tag/v0.2.0) (`312ad52`) and `browsergym-timewarp==0.2.0`. Scores are **not comparable** with runs made against 0.1.0: the gold answers were regenerated for the deterministic verifiers, and a judge verdict-parsing bug that inflated 0.1.0 scores was fixed upstream.

The three environments are **Flask servers** — no Docker. The cube provisions and runs them
for you (`provision_mode="auto"`, the default); set `provision_mode="manual"` if you'd rather
start them yourself.

- **auto** (default) — on `_setup()` the cube checks whether the upstream environment is already
  set up (repo cloned, `timewarp` conda env built, the HuggingFace data present).
  If not, it runs the upstream **idempotent** `setup.sh` to download/build everything, then
  launches the three Flask apps (at `ui_version`, 1–6) and waits until they're healthy. Resolved
  URLs are published into the benchmark's `runtime_context` (re-derived every run, so a resumed
  run picks up the freshly-launched ports), so this works under Ray on a single host. If
  `TW_WIKI` / `TW_NEWS` / `TW_WEBSHOP` already point at reachable servers, those are reused and
  nothing is launched — see the `ui_version` warning below. **Requires `conda`** (the servers run
  in the `timewarp` env the upstream `setup.sh` creates). Concurrent runs on one host are safe:
  provisioning is serialized by a lockfile beside the checkout, and each run gets its own ports
  and its own private log directory.
- **manual** — start the servers yourself and set the three env vars; `_setup()` only verifies
  reachability. (Same pattern as [`webarena-verified`](../webarena-verified).)

> **`ui_version` only applies to servers the cube launches.** It is TimeWarp's independent
> variable, so this matters: in manual mode, or in auto mode when reachable `TW_*` servers are
> reused, the sites render whatever era they were *started* with and the cube cannot tell which
> that is. It logs a warning and records `ui_version: None` on every episode rather than
> labelling them with an era that never took effect. **To sweep `ui_version`, keep `TW_WIKI` /
> `TW_NEWS` / `TW_WEBSHOP` unset** so the cube owns the servers. Episodes then carry the real
> era in their trajectory metadata.

## Prerequisites

- **`conda`** (auto mode only) — the servers run in the `timewarp` conda env that upstream
  `setup.sh` builds. The upstream **TimeWarp** repo (<https://github.com/sparklabutah/timewarp>)
  is cloned automatically; the servers are **not** part of the `browsergym-timewarp` package or
  this cube.
- An **`OPENAI_API_KEY`** — *optional*, only for the two `llm_judge` tasks (create one at
  <https://platform.openai.com/api-keys>), or point `TW_JUDGE*` at another judge. Everything
  else scores offline.

## Installation

```bash
uv pip install timewarp-cube   # not yet on PyPI — see pyproject for the git source
```

There is no separate heavy install step. In auto mode the cube clones the upstream repo and runs
its **idempotent** `setup.sh` (conda env + ~GBs of data) lazily on the **first** benchmark run,
cloning to `~/.cache/timewarp` — override with the `TIMEWARP_HOME` env var. (`cube install
timewarp-cube` is a lightweight no-op hook; it does not pre-provision.)

## Setup → run

### Auto (default)

```bash
export OPENAI_API_KEY=sk-...        # optional — only the two llm_judge tasks need it
```

Calling `make()` provisions and launches the servers on first run (`make debug` does **not** —
it is a manual-mode smoke; see [Debug / Testing](#debug--testing)):

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

Pick the UI era and checkout dir on the config (`ui_version` needs `TW_*` unset — see the
warning above):

```python
from timewarp_cube import TimeWarpBenchmarkConfig
cfg = TimeWarpBenchmarkConfig(tool_config=..., ui_version=3)   # UI theme index 1–6
```

`ui_version` is a **theme index, not a year**, and two things about it routinely surprise people:

| `ui_version` | wiki | news | webshop |
| --- | --- | --- | --- |
| 1 | 2001 | 2000s | 2000 |
| 2 | 2002 | 2004s | 2005 |
| 3 | 2003-4 | 2008s | 2010 |
| 4 | 2005-2022 | 2016s | 2015 |
| 5 | 2023-2025 | 2024s | 2025 |
| 6 | minimal | base-minimal | classic |

- **The eras are not aligned across sites.** A multi-site run at `ui_version=4` renders 2005-2022
  wiki alongside 2016 news and 2015 webshop — so "era 4" is not a point in time.
- **6 is a neutral control theme, not a sixth era**, so a 1→6 sweep is not monotonic in time at
  its endpoint. Use 1–5 for the temporal axis and 6 as the control.

To run this under the harness (agent, trajectory storage, Ray) rather than by hand, copy
[`recipes/timewarp.py`](../../recipes/timewarp.py) and edit the values — it is the config, not a CLI.

### Manual

Start the servers from the upstream checkout and point the cube at them:

```bash
git clone https://github.com/sparklabutah/timewarp && cd timewarp
bash setup.sh                              # one-time: conda env + dependencies
bash scripts/environment/run_all_env.sh 1  # start wiki + news + webshop at UI version 1
export TW_WIKI=http://localhost:<wiki-port>
export TW_NEWS=http://localhost:<news-port>
export TW_WEBSHOP=http://localhost:<webshop-port>/abc
export OPENAI_API_KEY=sk-...               # optional — only the two llm_judge tasks
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
| `TW_WIKI`        | manual mode | URL of the wiki server. Auto mode sets it for you; if already set+reachable, auto reuses it **and `ui_version` stops applying** — leave unset to control the era. |
| `TW_NEWS`        | manual mode | URL of the news server (same auto behaviour).                            |
| `TW_WEBSHOP`     | manual mode | URL of the webshop server, with the `/abc` path (same auto behaviour).   |
| `TIMEWARP_HOME`  | no       | Auto mode: where the upstream repo is cloned (default `~/.cache/timewarp`).  |
| `OPENAI_API_KEY` | no       | Only the two `llm_judge` tasks (32, 143); without it those episodes error rather than score 0. |
| `TW_JUDGE`, `TW_JUDGE_MODEL`, `TW_JUDGE_BASE_URL`, `TW_JUDGE_API_KEY` | no | Select/point the judge for those two tasks (e.g. `TW_JUDGE=gemma` against a local OpenAI-compatible endpoint). Read in-process by `browsergym-timewarp` at scoring time. |

## Debug / Testing

```bash
make debug                  # manual-mode smoke: needs the 3 servers already running (no API key)
uv run pytest tests/        # fast unit tests — no servers, no browser, no conda, no API key
```

The unit tests run on every PR via the `TimeWarp unit tests` job in
[`cube-ci-fast.yml`](../../.github/workflows/cube-ci-fast.yml). The debug suite does not: it needs
three live servers behind a conda env, which a hosted runner can't stand up.

## Known upstream behaviours that cost accuracy

Measured against a reference agent that submits each task's own gold answer (baseline: 40/40 on a
stratified 40-task slice). None of these are port bugs; all three are worth knowing before reading
a TimeWarp score.

- **`send_message` is terminal *and* one-shot.** The episode ends on the agent's first chat
  message, so an agent that narrates before answering has its narration scored — the same agent
  scores **0/40** with one scratchpad line in front of the answer. Opt into
  `ANSWER_PROTOCOL_OVERRIDES` (see below) to tell it so.
- **60 of the 229 deterministic tasks match only the answer's first sentence**
  (`scope: first_sentence`). Prefixing each gold answer with one lead-in sentence drops the oracle
  from 229/229 to 168/229 — **−26.6 points for phrasing alone**.
- **`report_infeasible` scores as the literal string `"N/A"`**, which only the LLM judge
  special-cases. On the 229 deterministic tasks it is effectively always wrong, even when the task
  really is infeasible.

```python
from timewarp_cube import ANSWER_PROTOCOL_OVERRIDES
agent.description_overrides = dict(ANSWER_PROTOCOL_OVERRIDES)   # opt-in, on the AGENT config
```

These overrides change what the agent sees, so scores with and without them are **not
comparable** — pick one and stay consistent within an experiment.

Two more things the benchmark does not isolate per episode:

- **Webshop keeps one server-side session** (hardcoded id `abc`) shared by every episode on a
  server, and upstream exposes no reset hook — once any episode reaches its done route, that
  session's `done`/`reward` persist for the life of the server.
- **A page on any unauthorized host hard-zeros the episode** before the answer is even read, and
  that includes *any other local port* (the check compares `netloc`, which carries the port, but
  only authorizes bare `localhost` / `127.0.0.1`). The cube logs a warning when this happens, since
  otherwise it is indistinguishable from a wrong answer.

`make debug` is a **manual-mode** smoke: it only *verifies* that the three servers are reachable —
it does not provision or launch anything. Before running it, have the servers up (either left over
from a prior auto-mode benchmark run, or started manually from the upstream repo — see
[Manual](#manual)) with `TW_WIKI` / `TW_NEWS` / `TW_WEBSHOP` exported. Both debug tasks are
scored by deterministic verifiers, so no API key is needed.

The unit tests in [`tests/`](tests/) cover the parts that don't need infrastructure: metadata loads 231 tasks, the named subsets filter/cover correctly, the configs round-trip, the toolbox pairs a browser tool with a `ChatTool`, that `install()` stays lightweight (no provisioning), and the provisioning helpers (mode toggle, `is_provisioned` completeness checks, the auto-launch path, server teardown, and `runtime_context` URL threading) with subprocess/socket calls mocked. The [`debug.py`](src/timewarp_cube/debug.py) suite exercises the full setup→validate path against the live servers you started with a scripted reference-answer agent.

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
