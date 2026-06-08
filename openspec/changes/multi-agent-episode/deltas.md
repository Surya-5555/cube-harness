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

- **`AgentConfig.make()`** (`agent` spec) — takes per-agent identity from the `TaskTool`
  (`agent_id` + that seat's `action_set`), so one `AgentConfig` yields N correctly-shaped
  agents. (Signature: decision (1).)
- **`EpisodeConfig` / experiment recipe** (`episode`/`experiment` spec) — carries the
  (single, v1) `AgentConfig` consumed once per `TaskTool`. Fixed N agents in v1.
- **Agent loop re-reads `action_set` per turn** (`agent`/`episode` spec) — `TaskTool.action_set`
  is dynamic upstream, so the agent rebuilds its tool schema each turn instead of caching it
  at `make()`. Enables legal-action masking / phase gating / real-time; single-agent inherits it.

## OPEN (block firming up)

1. `make(action_set, agent_id)` vs `make(task_tool)`.
2. Termination policy; 3. per-agent vs shared budget; 4. heterogeneous per-role configs;
5. `async`/`batch` schedulers.
