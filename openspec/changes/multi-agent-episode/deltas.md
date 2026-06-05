# Deltas — multi-agent-episode (cube-harness)

> Thin until v1 firms up. Upstream contract: `cube-standard/openspec/changes/streamable-task`
> (`Task` + `TaskTool` + `Streamer`). Targets: `episode`, `agent`, `experiment`, `core` specs.

## ADDED (provisional)

- **`MultiAgentEpisode`** (`episode` spec) — runtime that builds the task, reads
  `task.agent_tools()`, builds one agent per `TaskTool`, drives them under a scheduler,
  finalizes per-agent + episode. Single-agent `Episode` = the N=1 fast path.
- **Scheduler** (`episode` spec) — v1 `turn-based` (round-robin, sequential, sync).
  `async` / `batch` deferred.
- **`agent_id` on trajectory events** (`core`/`eval_log`) — `ToolCallEvent` /
  `LLMCallEvent` carry `agent_id`; the `EventStreamer` tags env + LLM events per agent.

## MODIFIED (provisional)

- **`AgentConfig.make()`** (`agent` spec) — takes per-agent identity from the `TaskTool`
  (`agent_id` + that seat's `action_set`), so one `AgentConfig` yields N correctly-shaped
  agents. (Signature: decision (1).)
- **`EpisodeConfig` / experiment recipe** (`episode`/`experiment` spec) — carries the
  (single, v1) `AgentConfig` consumed once per `TaskTool`. Fixed N agents in v1.

## OPEN (block firming up)

1. `make(action_set, agent_id)` vs `make(task_tool)`.
2. Termination policy; 3. per-agent vs shared budget; 4. heterogeneous per-role configs;
5. `async`/`batch` schedulers.
