# Deltas — multi-agent-episode (cube-harness)

> Thin until v1 firms up. Upstream contract: `cube-standard/openspec/changes/streamable-task`
> (`Task` + `TaskTool` + `Streamer`). Targets: `episode`, `agent`, `experiment`, `core` specs.

## ADDED (provisional)

- **`MultiAgentEpisode`** (`episode` spec) — runtime that builds the task, reads
  `task.agent_tools()`, builds one agent per `TaskTool`, drives them under a scheduler,
  finalizes per-agent + episode. Single-agent `Episode` = the N=1 fast path.
- **Scheduler** (`episode` spec) — v1 `turn-based` (round-robin, sequential, sync).
  `async` / `batch` deferred.
- **`agent_id` on trajectory events** (`core`/`eval_log`) — capture is harness-side (no
  standard `Streamer`): each agent loop self-emits its tool + LLM events; the arena recovers
  reward via `task.evaluate()`. `ToolCallEvent` / `LLMCallEvent` / eval carry `agent_id`.

## MODIFIED (provisional)

- **`AgentConfig.make()`** (`agent` spec) — takes per-seat identity from the `TaskTool`
  (`action_set` + `agent_id` + `role`), so one `AgentConfig` yields N correctly-shaped agents.
  `role`/`agent_id` come from cube-standard's `agent_roles()` seam (`role=None`→`"agent"`, else
  `"{role}-{seat}"`). v1 agents ignore `role` (homogeneous); per-role heterogeneous configs
  (a different `AgentConfig` per role) are a forward extension.
- **`EpisodeConfig` / experiment recipe** (`episode`/`experiment` spec) — carries the
  (single, v1) `AgentConfig` consumed once per `TaskTool`. Fixed N agents in v1.
- **(Forward extension, NOT in v1) Agent loop re-reads `action_set` per turn.**
  `TaskTool.action_set` is dynamic upstream, so an agent *could* rebuild its tool schema
  each turn (legal-action masking / phase gating / real-time). Today agents **snapshot the
  set at `make()`** — fine because no current cube varies it. Wire the per-turn re-read only
  when a cube actually needs a changing action set.

## OPEN (block firming up)

1. `make(action_set, agent_id)` vs `make(task_tool)`.
2. Termination policy; 3. per-agent vs shared budget; 4. heterogeneous per-role configs;
5. `async`/`batch` schedulers.
