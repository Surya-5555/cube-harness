# Episode

**Module:** `cube_harness.episode`

## Purpose

An `Episode` runs one agent against one task and produces a trajectory view. It owns
task setup/reset/finalization, event streaming, OpenTelemetry tracing, and error
recovery. The agent owns the per-turn loop and emits through `EventStreamer`-attached
LLMs/tools. Workers receive an `EpisodeConfig` (serializable) and materialize the
`Episode` locally.

## Public API

### `MAX_STEPS = 1000`
Module-level upper limit. An episode also accepts a lower `max_steps` argument;
whichever is smaller wins.

### `EpisodeConfig` (serializable)
```python
class EpisodeConfig(TypedBaseModel):
    id: int                          # per-experiment episode number
    agent_config: AgentConfig
    exp_name: str
    output_dir: Path
    max_steps: int
    task_config: TaskConfig          # cube.task.TaskConfig
    trajectory_id: str | None = None # optional storage/event id override
```

Saved to disk at `{output_dir}/episodes/{trajectory_id}/episode_config.json` before
the episode runs, so experiments can resume after crashes.

RL rollout hooks:

- `recorder_config: EventStreamerConfig` lets callers attach additional event
  sinks, including `RLEventSink`, without changing the episode loop.
- `write_eval_log: bool = True` may be set false by high-throughput rollout
  workers to skip debug/eval-log artifacts.
- `trajectory_id` lets specialized callers such as RL use a caller-owned unique
  identity while ordinary experiments keep `{task_id}_ep{episode_id}`.

### `Episode`
```python
class Episode:
    def __init__(
        self, id: int, output_dir: Path, agent_config: AgentConfig, task_config: TaskConfig,
        exp_name: str = "default", max_steps: int = MAX_STEPS,
        storage: Storage | None = None,        # defaults to FileStorage(output_dir)
        runtime_context: RuntimeContext | None = None,    # from Benchmark._setup()
        container_backend: ContainerBackend | None = None,
    )

    @classmethod
    def load_episode_from_config(cls, config_path: Path, benchmark: Benchmark | None = None) -> "Episode"
    # Accepts both V2 (episodes/<id>/episode_config.json) and V1 layouts.
    # If benchmark provided, forwards runtime_context and container_backend.

    def run(self) -> Trajectory
    # Main loop. Creates the task via task_config.make(...), runs reset → step*, streams
    # every step to disk (never retained in memory), closes the task in finally. The
    # returned Trajectory carries metadata + summary_stats + reward_info with steps == [];
    # load step content lazily via storage.load_trajectory(id).

    allow_overwrite: bool = False   # when True, archives existing trajectory before saving
```

### `summary_stats`
`Trajectory.summary_stats` is accumulated incrementally by `EventStreamer` as events
stream in, then written at end-of-episode. Includes
`n_env_steps`, `n_agent_steps`, `total_actions`, `total_llm_calls`, token counts,
`cost`, `duration`, `final_reward`, `error_type`. It is the only per-episode aggregate the
runner needs — no end-of-run walk over the steps.

## Main loop semantics

1. Enter `tracer.episode(task_id, experiment=exp_name)` span.
2. `_open_status` writes RUNNING `status.json` with `current_step=0`.
3. `task_config.make(runtime_context=...)` → live Task, then `task.reset()` → initial observation.
4. Save start-of-episode metadata and `episode_config`.
5. Build `Budget` + `EventStreamer`, install monitored tools, and record the reset event.
6. Attach the streamer to the agent and run `await agent.run(initial.obs, env_tool)`.
   The agent owns the turn loop; LLM calls, tool calls, failures, and evaluations stream
   as canonical `TrajectoryEvent`s.
7. Run terminal `task.evaluate()`, record a terminal `EvaluationEvent`, persist final
   metadata/summary/eval-log, and update status.
8. `finally`: call `task.close()` and `tracer.shutdown()`.

Final episode status is `OK` if `final_reward > 0`, else `ERROR` (sets OTel span status).

## Invariants

1. Every step is persisted incrementally **and never accumulated in memory** — the returned
   Trajectory (including the one a Ray worker returns) carries no steps, only metadata +
   summary_stats + reward_info. Load step content via `storage.load_trajectory(id)`.
2. `task.close()` is always called (finally block), even on exceptions.
3. Agent and env exceptions are caught, written as a step with `error` populated, then
   re-raised. Callers see the exception; the trajectory remains on disk.
4. Empty actions + no error → graceful break (agent says "done").
5. By default `trajectory.id = f"{task_id}_ep{episode_id}"`; callers may pass
   `trajectory_id` only when they own a stronger unique identity.

## Storage layout (V2)

```
<output_dir>/episodes/<trajectory_id>/
├── episode_config.json
├── episode.metadata.json      # Trajectory minus steps
├── steps/
│   ├── 000_obs.msgpack.zst
│   ├── 001_act.msgpack.zst
│   └── ...
├── episode_summary.jsonl      # written by SummaryProcessor
├── failure.txt                # stack trace if run crashed before completion
└── logs/...                   # redirected stdout/stderr
```

Steps are msgpack + zstd compressed. V1 JSONL layout under `trajectories/` is still
loadable but no longer written.

## Contracts for implementers

- Agents that need to recover from partial episodes can reload via
  `Episode.load_episode_from_config()`. Pass the `Benchmark` if the task needs
  `runtime_context` or `container_backend`.
- Storage backends must implement the `Storage` protocol (`save_trajectory`,
  `save_step`, `save_episode_config`, `update_experiment_summary`). See
  [storage spec](../storage/spec.md).

## Gotchas

- `Episode.__init__` does NOT call `task_config.make()` — the task is created inside
  `run()` so long-lived resources are owned by the worker, not the scheduler.
- Ray workers share `benchmark._runtime_context` by reference — treat it as
  read-only after `setup()` returns (see cube-standard benchmark spec).
- `summary_stats` is accumulated incrementally by `SummaryProcessor` as steps stream in, so
  it stays O(1) per step regardless of trajectory length — no end-of-run walk.
- Episode timeouts are enforced by `run_with_ray` at the scheduler level, not inside
  the episode. Sequential runs have no timeout.
