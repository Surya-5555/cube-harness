"""Tests for XRayState live-refresh status reconciliation (BUG-1).

A status transition that rewrites only ``status.json`` — STALE via the ghost-sweep,
CANCELLED via the stall-killer, or a never-started QUEUED episode going STALE — must be
picked up by ``refresh_experiment`` during live polling. Previously the change-detection
keyed off trajectory-file mtimes and missed these; now it keys off the episode-directory
mtime (bumped by every atomic ``status.json`` write).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from cube_harness.analyze import xray_utils
from cube_harness.analyze.xray import XRayState
from cube_harness.core import Trajectory
from cube_harness.episode_status import EpisodeStatus
from cube_harness.storage import FileStorage


def _bump_dir_mtime(ep_dir: Path) -> None:
    """Force the episode dir's mtime forward so refresh treats it as changed regardless
    of filesystem mtime granularity (writes within the same tick can otherwise tie)."""
    future = time.time() + 100
    os.utime(ep_dir, (future, future))


def _status_of(state: XRayState, traj_id: str) -> str:
    traj = next(t for t in state.trajectories if t.id == traj_id)
    return xray_utils.trajectory_status(traj)


def _write_status(storage: FileStorage, traj_id: str, task_id: str, status: str, *, heartbeat: bool) -> None:
    storage.write_episode_status(
        traj_id,
        EpisodeStatus(
            status=status,
            task_id=task_id,
            episode_id=0,
            started_at=1.0,
            last_heartbeat_at=time.time() if heartbeat else None,
        ),
    )


def test_running_episode_to_stale_is_picked_up_live(tmp_path: Path) -> None:
    """A started (metadata-having) RUNNING episode flipped to STALE in status.json only
    must refresh to 'stale' — exercises the full-reload path off the dir mtime."""
    exp_dir = tmp_path / "exp_run"
    storage = FileStorage(exp_dir)
    traj_id, task_id = "task_1_ep0", "task_1"
    # A started episode: trajectory file exists, no end_time, status RUNNING.
    storage.save_trajectory(Trajectory(id=traj_id, metadata={"task_id": task_id, "agent_name": "a"}, start_time=1.0))
    _write_status(storage, traj_id, task_id, "RUNNING", heartbeat=True)

    state = XRayState(results_dir=tmp_path)
    assert state.load_experiment(exp_dir)  # synchronous: no bulk-loader to wait for
    assert _status_of(state, traj_id) == "running"

    # Worker dies → ghost-sweep rewrites only status.json (trajectory file untouched).
    _write_status(storage, traj_id, task_id, "STALE", heartbeat=True)
    _bump_dir_mtime(storage._episode_dir(traj_id))

    assert state.refresh_experiment() is True
    assert _status_of(state, traj_id) == "stale"


def test_queued_stub_to_stale_is_picked_up_live(tmp_path: Path) -> None:
    """A never-started QUEUED episode (config + status.json, no trajectory file) flipped to
    STALE must refresh to 'stale' — exercises the stub re-inject fallback."""
    exp_dir = tmp_path / "exp_queue"
    storage = FileStorage(exp_dir)
    traj_id, task_id = "task_2_ep0", "task_2"
    ep_dir = storage._episode_dir(traj_id)
    ep_dir.mkdir(parents=True)
    (ep_dir / "episode_config.json").write_text(json.dumps({"task_id": task_id}))
    _write_status(storage, traj_id, task_id, "QUEUED", heartbeat=False)

    state = XRayState(results_dir=tmp_path)
    assert state.load_experiment(exp_dir)  # synchronous: no bulk-loader to wait for
    assert _status_of(state, traj_id) == "queued"

    # Driver dies → ghost-sweep promotes the orphaned QUEUED episode to STALE.
    _write_status(storage, traj_id, task_id, "STALE", heartbeat=False)
    _bump_dir_mtime(ep_dir)

    assert state.refresh_experiment() is True
    assert _status_of(state, traj_id) == "stale"


def test_no_status_change_is_not_reported_as_changed(tmp_path: Path) -> None:
    """A refresh with no on-disk change must report nothing changed (no spurious reloads)."""
    exp_dir = tmp_path / "exp_idle"
    storage = FileStorage(exp_dir)
    traj_id, task_id = "task_3_ep0", "task_3"
    storage.save_trajectory(Trajectory(id=traj_id, metadata={"task_id": task_id, "agent_name": "a"}, start_time=1.0))
    _write_status(storage, traj_id, task_id, "RUNNING", heartbeat=True)

    state = XRayState(results_dir=tmp_path)
    assert state.load_experiment(exp_dir)  # synchronous: no bulk-loader to wait for

    assert state.refresh_experiment() is False
