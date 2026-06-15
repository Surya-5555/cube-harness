# Deltas — multi-agent-episode (cube-harness)

> Thin until v1 firms up. Upstream contract: `cube-standard/openspec/changes/streamable-task`
> (`Task` + `AgentView`). Targets: `episode`, `agent`, `experiment`, `core` specs.

## ADDED (provisional)

- **`MultiAgentEpisode`** (`episode` spec) — runtime that builds the task, calls
  `build_agent_tools(task, streamer)` (walks `task.agent_roles()`, calls
  `task.get_agent_view(role)` once per seat — the task owns the seat index), builds one
  agent per seat, drives them under a scheduler, finalizes per-agent + episode. Single-agent
  `Episode` = the N=1 fast path (default roster `{None: 1}` → one seat).
- **`MonitoredTool`** (`tool` spec) — wraps a cube-standard `AgentView` (one per seat)
  to add budget enforcement, `ToolCallEvent` emission, and the per-action
  `finished()`/`evaluate()` cadence; carries `agent_id` + `role`.
- **Scheduler** (`episode` spec) — v1 `turn-based` (round-robin, sequential, sync, joint
  budget). `async` / `batch` deferred.
- **`agent_id` on trajectory events** (`core`) — capture is harness-side (no standard
  `Streamer`): seats share one `EventStreamer`; the arena recovers reward via
  `task.evaluate()`. `ToolCallEvent` + `EvaluationEvent` carry a nullable `agent_id`
  (`None` / `"agent"` single-agent, `"{role}-{seat}"` multi-agent). Per-seat tagging of
  `LLMCallEvent` (and full per-agent trajectory demux) is **deferred**.

## MODIFIED (provisional)

- **`AgentConfig.make()`** (`agent` spec) — takes per-seat identity from the seat's
  `MonitoredTool` (`make(action_set, agent_id=..., role=...)`), so one `AgentConfig`
  yields N correctly-shaped agents. `role`/`agent_id` come from the `AgentView` behind
  cube-standard's `agent_roles()` seam (`role=None`→`"agent"`, else `"{role}-{seat}"`).
  v1 agents ignore `role` (homogeneous); per-role heterogeneous configs (a different
  `AgentConfig` per role) are a forward extension.
- **`EpisodeConfig` / experiment recipe** (`episode`/`experiment` spec) — carries the
  (single, v1) `AgentConfig` consumed once per seat. Fixed N agents in v1.
- **(Forward extension, NOT in v1) Agent loop re-reads `action_set` per turn.**
  `AgentView.action_set` is dynamic upstream, so an agent *could* rebuild its tool schema
  each turn (legal-action masking / phase gating / real-time). Today agents **snapshot the
  set at `make()`** — fine because no current cube varies it. Wire the per-turn re-read only
  when a cube actually needs a changing action set.

## DEFERRED (past v1)

1. Heterogeneous per-role configs (`{role: AgentConfig}` map).
2. `async` / `batch` schedulers — after turn-based lands.
3. Per-agent (vs the v1 joint) budget.
4. Per-seat `LLMCallEvent` tagging + per-agent trajectory demux / XRay lanes.
