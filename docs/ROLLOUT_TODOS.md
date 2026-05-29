# Rollout TODOs

## Event Sink Durability and Backpressure

The current rollout event sink is a functional first-pass implementation for short/local rollouts, but it is not yet production-ready for very long RL trajectories.

Known gaps:

- Spilled events are not replayed by `events_from()` or `wait_for_events()`; those APIs only read the hot in-memory deque.
- If `persist_events_dir` is unset and `max_hot_events` is exceeded, old events are dropped from hot memory.
- `ack()` compacts hot memory, but compacted/spilled events are not currently available through SSE replay.
- Spill writes happen synchronously while the sink condition lock is held.
- `event_publish_timeout_s` retries publish exceptions, but it does not bound a slow/blocking filesystem write.
- Offset assignment and spill are not fully atomic for non-terminal events if persistence fails after mutation.

Desired design:

- Make retention semantics explicit in health via `oldest_available_offset` and spill/drop counters.
- Add durable replay from spill files for reconnects from older offsets.
- Use a global or indexed event log keyed by offset, not only per-request/per-trajectory JSONL, because SSE offsets are global.
- Never silently drop accepted rollout events when durable replay is required.
- Apply backpressure, or emit `event_error`, when memory and spill capacity are saturated.
- Move slow spill IO out of the sink lock or make the blocking behavior intentional and visible in metrics.
- Commit offsets atomically only after the event is safely retained in memory or durable storage.
