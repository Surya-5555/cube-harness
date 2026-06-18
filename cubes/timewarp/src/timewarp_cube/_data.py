"""Shared accessor for the raw browsergym-timewarp task data (``data/test.raw.json``).

Both the debug suite (reference answers) and the task_metadata.json generator read
this resource; keeping the path and parse in one place means a BrowserGym data move
touches a single call site. Cached so repeated lookups don't re-read and re-parse.
"""

from __future__ import annotations

import functools
import importlib.resources
import json


@functools.cache
def load_raw_tasks() -> list[dict]:
    """Load and parse the browsergym-timewarp raw task configs. Treat the result as read-only."""
    raw = importlib.resources.files("browsergym.timewarp").joinpath("data/test.raw.json").read_text()
    return json.loads(raw)
