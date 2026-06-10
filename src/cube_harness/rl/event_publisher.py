from __future__ import annotations

import json
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from cube_harness.rl.events import AnyRolloutEvent

_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9_-]")


def _safe_path_component(value: str) -> str:
    """Sanitize a client-supplied id into one safe path segment.

    request_id/trajectory_id come from rollout clients; used raw in a spill path
    a value containing `/` or `..` could escape persist_events_dir.
    """
    return _UNSAFE_PATH_CHARS.sub("_", value) or "unknown"


class EventPublisherConfig(BaseModel):
    max_hot_events: int = 10000
    persist_events_dir: Path | None = None
    event_publish_timeout_s: float = 30.0


class EventPublisher:
    """Bounded hot event log with optional JSONL spill and SSE replay by offset."""

    def __init__(self, config: EventPublisherConfig | None = None) -> None:
        self.config = config or EventPublisherConfig()
        self._events: deque[dict] = deque()
        self._next_offset = 0
        self._ack_offset = -1
        self._terminal_events: dict[str, dict] = {}
        self._request_next_event_index: dict[str, int] = defaultdict(int)
        self._spilled_event_count = 0
        self._dropped_event_count = 0
        self._condition = threading.Condition()
        if self.config.persist_events_dir is not None:
            self.config.persist_events_dir.mkdir(parents=True, exist_ok=True)

    @property
    def next_offset(self) -> int:
        with self._condition:
            return self._next_offset

    @property
    def hot_event_count(self) -> int:
        with self._condition:
            return len(self._events)

    def publish(self, event: AnyRolloutEvent) -> dict:
        payload = event.model_dump(mode="json")
        published = self.publish_payload(payload)
        event.offset = published["offset"]
        return published

    def publish_payload(self, payload: dict[str, Any]) -> dict:
        deadline = time.monotonic() + self.config.event_publish_timeout_s
        while True:
            try:
                return self._publish_payload_once(payload)
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)

    def _publish_payload_once(self, payload: dict[str, Any]) -> dict:
        with self._condition:
            if payload.get("type") == "terminal":
                request_id = str(payload.get("request_id") or "")
                if request_id and request_id in self._terminal_events:
                    return self._terminal_events[request_id]
            payload = dict(payload)
            request_id = str(payload.get("request_id") or "")
            if request_id:
                proposed_index = int(payload.get("event_index", -1))
                next_index = self._request_next_event_index[request_id]
                event_index = proposed_index if proposed_index >= next_index else next_index
                payload["event_index"] = event_index
                self._request_next_event_index[request_id] = event_index + 1
            payload["offset"] = self._next_offset
            self._next_offset += 1
            self._events.append(payload)
            if payload.get("type") == "terminal":
                request_id = str(payload.get("request_id") or "")
                if request_id:
                    self._terminal_events[request_id] = payload
            if len(self._events) > self.config.max_hot_events:
                spilled = self._events.popleft()
                self._spill(spilled)
            self._condition.notify_all()
            return payload

    def _spill(self, event: dict) -> None:
        # TODO(replay-gap): signal an explicit gap to clients that resume from
        # an offset older than the hot buffer when spill persistence is disabled.
        if self.config.persist_events_dir is None:
            self._dropped_event_count += 1
            return
        client = _safe_path_component(str(event.get("request_id") or "unknown"))
        trajectory = _safe_path_component(str(event.get("trajectory_id") or "unknown"))
        path = self.config.persist_events_dir / client / f"{trajectory}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(event, separators=(",", ":")) + "\n")
        self._spilled_event_count += 1

    def events_from(self, from_offset: int) -> list[dict]:
        with self._condition:
            return [e for e in self._events if int(e["offset"]) >= from_offset]

    def has_terminal(self, request_id: str) -> bool:
        with self._condition:
            return request_id in self._terminal_events

    def wait_for_events(self, from_offset: int, timeout: float = 15.0) -> list[dict]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                events = [e for e in self._events if int(e["offset"]) >= from_offset]
                if events:
                    return events
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._condition.wait(timeout=min(remaining, 1.0))

    def ack(self, offset: int) -> None:
        with self._condition:
            self._ack_offset = max(self._ack_offset, int(offset))
            self._compact_locked()

    def _compact_locked(self) -> None:
        while self._events and int(self._events[0]["offset"]) <= self._ack_offset:
            self._spill(self._events.popleft())

    def health(self) -> dict:
        with self._condition:
            oldest_hot_offset = int(self._events[0]["offset"]) if self._events else None
            newest_hot_offset = int(self._events[-1]["offset"]) if self._events else None
            ack_offset = self._ack_offset if self._ack_offset >= 0 else None
            return {
                "next_offset": self._next_offset,
                "oldest_hot_offset": oldest_hot_offset,
                "newest_hot_offset": newest_hot_offset,
                "oldest_available_offset": oldest_hot_offset if oldest_hot_offset is not None else self._next_offset,
                "hot_event_count": len(self._events),
                "max_hot_events": self.config.max_hot_events,
                "hot_capacity_remaining": max(self.config.max_hot_events - len(self._events), 0),
                "terminal_count": len(self._terminal_events),
                "tracked_request_count": len(self._request_next_event_index),
                "ack_offset": ack_offset,
                "spill_enabled": self.config.persist_events_dir is not None,
                "persist_events_dir": str(self.config.persist_events_dir) if self.config.persist_events_dir else None,
                "spilled_event_count": self._spilled_event_count,
                "dropped_event_count": self._dropped_event_count,
                "event_publish_timeout_s": self.config.event_publish_timeout_s,
            }

    def close(self) -> None:
        return None
