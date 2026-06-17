"""Unit tests for the episode profiling foundation (infra-free, fast)."""

from __future__ import annotations

import time
from pathlib import Path

from cube_harness.episode import Episode, EpisodeConfig
from cube_harness.metrics.profile_rollup import build_rollup
from cube_harness.metrics.profiler import (
    PHASE_AGENT_LOOP,
    PHASE_SETUP,
    EpisodeProfile,
    PhaseAccumulator,
    ProfileConfig,
    ResourceSampler,
    ResourceSummary,
    _percentile,
    _summarize,
)


def test_percentile_and_summarize() -> None:
    assert _percentile([], 0.5) == 0.0
    assert _percentile([5.0], 0.95) == 5.0
    assert _percentile([1.0, 2.0, 3.0], 0.5) == 2.0  # nearest-rank
    assert _percentile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
    summ = _summarize([0.0, 10.0, 20.0])
    assert summ.samples == 3
    assert summ.mean == 10.0
    assert summ.max == 20.0


def test_phase_accumulator_sums_durations() -> None:
    acc = PhaseAccumulator()
    with acc.phase(PHASE_SETUP):
        time.sleep(0.02)
    with acc.phase(PHASE_AGENT_LOOP):
        time.sleep(0.01)
    with acc.phase(PHASE_SETUP):  # re-entry accumulates
        time.sleep(0.01)
    totals = acc.totals()
    assert totals[PHASE_SETUP] > totals[PHASE_AGENT_LOOP]
    assert totals[PHASE_AGENT_LOOP] >= 0.01


def test_resource_sampler_captures_series_under_load() -> None:
    with ResourceSampler(ProfileConfig(sample_hz=20)) as sampler:
        # burn CPU so the sampler records a non-trivial process tree
        _ = sum(i * i for i in range(2_000_000))
        time.sleep(0.15)
    summaries = sampler.summaries()
    assert sampler.sample_count >= 1
    assert "cpu_percent" in summaries
    assert "rss_mb" in summaries
    assert summaries["rss_mb"].max > 0.0


def test_resource_sampler_disabled_records_nothing() -> None:
    with ResourceSampler(ProfileConfig(resource_sampling=False)) as sampler:
        time.sleep(0.05)
    assert sampler.sample_count == 0
    assert sampler.summaries() == {}


def test_gpu_knob_independent_of_resource_sampling_no_hardware() -> None:
    # gpu=True with resource_sampling=False must not sample host CPU/RSS, and on
    # a GPU-less host (no NVML handles) it degrades to recording nothing — never
    # an error. (The fix: the two knobs are independent; the old code gated GPU
    # behind resource_sampling.)
    with ResourceSampler(ProfileConfig(resource_sampling=False, gpu=True)) as sampler:
        _ = sum(i * i for i in range(500_000))
        time.sleep(0.05)
    assert "cpu_percent" not in sampler.summaries()  # host sampling stays off
    assert "rss_mb" not in sampler.summaries()


def test_episode_profile_roundtrip(tmp_path: Path) -> None:
    prof = EpisodeProfile(
        task_id="task-1",
        trajectory_id="traj-1",
        wall_time_s=12.5,
        phases={PHASE_SETUP: 2.0, PHASE_AGENT_LOOP: 9.0},
        resources={"cpu_percent": ResourceSummary(samples=3, mean=40.0, p50=35.0, p95=90.0, max=95.0)},
        sample_count=3,
    )
    prof.write(tmp_path)
    loaded = EpisodeProfile.load(tmp_path)
    assert loaded is not None
    assert loaded.task_id == "task-1"
    assert loaded.phases[PHASE_AGENT_LOOP] == 9.0
    assert loaded.resources["cpu_percent"].p95 == 90.0


def test_episode_profile_load_missing_returns_none(tmp_path: Path) -> None:
    assert EpisodeProfile.load(tmp_path) is None


def test_episode_config_profile_defaults_off_and_serializes() -> None:
    from cube.task import TaskConfig  # noqa: PLC0415 - test-local to avoid heavy import at module load

    # default: no profiling
    assert EpisodeConfig.model_fields["profile"].default is None
    # round-trips through JSON (Ray pickling / resume path uses model_validate_json)
    cfg = ProfileConfig(sample_hz=4.0, gpu=True)
    dumped = cfg.model_dump_json()
    assert ProfileConfig.model_validate_json(dumped).sample_hz == 4.0
    assert isinstance(TaskConfig, type)  # import sanity


def test_breakdown_agent_loop_splits_llm_tool_and_per_tool() -> None:
    from cube.core import Action  # noqa: PLC0415 - test-local

    from cube_harness.core import LLMCallEvent, ToolCallEvent, TrajectoryEvent  # noqa: PLC0415
    from cube_harness.metrics.profiler import breakdown_agent_loop

    def tool(name: str, action_id: str, start: float, end: float) -> TrajectoryEvent:
        return TrajectoryEvent(
            output=ToolCallEvent(parent_event_id="p", action_id=action_id, action=Action(name=name, arguments={})),
            start_time=start,
            end_time=end,
        )

    events = [
        TrajectoryEvent(output=LLMCallEvent(), start_time=0.0, end_time=2.0),  # llm 2s
        tool("reset", "reset", 2.0, 2.1),  # excluded — synthetic initial obs
        tool("bash", "a1", 2.1, 3.1),  # bash 1.0s
        tool("bash", "a2", 3.1, 3.6),  # bash 0.5s
        tool("read_file", "a3", 3.6, 3.7),  # read_file 0.1s
        TrajectoryEvent(output=LLMCallEvent(), start_time=3.7, end_time=4.7),  # llm 1s
    ]
    bd = breakdown_agent_loop(events)
    assert bd.n_llm_calls == 2
    assert abs(bd.llm_wait_s - 3.0) < 1e-6
    assert bd.n_tool_calls == 3  # reset excluded
    assert abs(bd.tool_exec_s - 1.6) < 1e-6
    assert bd.tools["bash"].count == 2
    assert abs(bd.tools["bash"].total_s - 1.5) < 1e-6
    # per-call distribution: floor (min) vs max distinguishes transport- from work-bound
    assert abs(bd.tools["bash"].min_s - 0.5) < 1e-6
    assert abs(bd.tools["bash"].max_s - 1.0) < 1e-6
    assert abs(bd.tools["bash"].mean_s - 0.75) < 1e-6
    assert "reset" not in bd.tools


def test_live_episode_writes_profile(tmp_dir, mock_agent_config, mock_cube_task_config) -> None:  # noqa: ANN001 - fixtures
    """End-to-end: an Episode run with profile set drops profile.json with phases."""
    episode = Episode(
        id=0,
        output_dir=tmp_dir,
        agent_config=mock_agent_config,
        task_config=mock_cube_task_config,
        exp_name="profile_test",
        max_steps=5,
        storage=None,
        runtime_context=None,
        profile=ProfileConfig(sample_hz=20),
    )
    view = episode.run()
    ep_dir = Path(tmp_dir) / "episodes" / view.id
    profile = EpisodeProfile.load(ep_dir)
    assert profile is not None
    assert profile.task_id == mock_cube_task_config.task_id
    # setup + evaluate phases always run; agent_loop runs for the mock agent too.
    assert profile.phases[PHASE_SETUP] >= 0.0
    assert "evaluate" in profile.phases
    assert profile.wall_time_s >= 0.0


def test_live_episode_without_profile_writes_nothing(tmp_dir, mock_agent_config, mock_cube_task_config) -> None:  # noqa: ANN001
    """Default (profile=None) must leave no profile.json — zero footprint."""
    episode = Episode(
        id=0,
        output_dir=tmp_dir,
        agent_config=mock_agent_config,
        task_config=mock_cube_task_config,
        exp_name="profile_test",
        max_steps=5,
        storage=None,
        runtime_context=None,
    )
    view = episode.run()
    assert EpisodeProfile.load(Path(tmp_dir) / "episodes" / view.id) is None


def test_build_rollup_agent_loop_transport_floor_split(tmp_path: Path) -> None:
    """End-to-end rollup: persist events for 2 episodes with differing per-call
    tool durations and assert the transport-floor split uses PER-EPISODE floors
    (not global-min × global-count) and divides by covered episodes."""
    from cube.core import Action  # noqa: PLC0415

    from cube_harness.core import (  # noqa: PLC0415
        ToolCallEvent,
        TrajectoryEvent,
        TrajectoryMetadata,
    )
    from cube_harness.metrics.profile_rollup import build_rollup  # noqa: PLC0415
    from cube_harness.storage import FileStorage  # noqa: PLC0415

    storage = FileStorage(tmp_path)

    def write_ep(tid: str, durations: list[float]) -> None:
        agent_loop = sum(durations) + 1.0  # +1s of non-tool agent-loop time (overhead+llm)
        storage.save_metadata(TrajectoryMetadata(id=tid, start_time=0.0))
        t = 0.0
        for d in durations:
            storage.save_event(
                TrajectoryEvent(
                    output=ToolCallEvent(parent_event_id="p", action_id="x", action=Action(name="bash", arguments={})),
                    start_time=t,
                    end_time=t + d,
                ),
                tid,
            )
            t += d
        EpisodeProfile(
            task_id=tid,
            trajectory_id=tid,
            wall_time_s=agent_loop,
            phases={"agent_loop": agent_loop},
            sample_count=0,
        ).write(tmp_path / "episodes" / tid)

    # ep0: bash calls 1.0 + 3.0  (floor=1.0 each → 2×1.0=2.0 floor, work=2.0)
    # ep1: bash calls 2.0 + 2.0  (floor=2.0 each → 2×2.0=4.0 floor, work=0.0)
    write_ep("ep0", [1.0, 3.0])
    write_ep("ep1", [2.0, 2.0])

    rollup = build_rollup(tmp_path)
    al = rollup["agent_loop"]
    assert al["episodes_covered"] == 2
    tes = al["tool_exec_split"]
    # tool_exec total = 8.0 over 2 episodes → mean 4.0/ep
    tool_split = next(s for s in al["split"] if s["part"] == "tool_exec")
    assert abs(tool_split["mean_s"] - 4.0) < 1e-6
    # PER-EPISODE floor: ep0 2×1.0=2.0, ep1 2×2.0=4.0 → 6.0 total → 3.0/ep.
    # (A global-min×global-count bug would give 4×1.0=4.0 → 2.0/ep — wrong.)
    assert abs(tes["transport_floor_s"] - 3.0) < 1e-6
    assert abs(tes["work_s"] - 1.0) < 1e-6  # (8.0 - 6.0)/2
    # reclaimable: ep0 (2-1)×1.0=1.0, ep1 (2-1)×2.0=2.0 → 3.0 total → 1.5/ep
    assert abs(tes["reclaimable_by_batching_s"] - 1.5) < 1e-6


def test_build_rollup_pareto_ranks_phases(tmp_path: Path) -> None:
    # two episodes, each with profile.json under episodes/<id>/
    for i, (setup, loop, evaluate) in enumerate([(1.0, 8.0, 3.0), (2.0, 10.0, 4.0)]):
        ep_dir = tmp_path / "episodes" / f"traj-{i}"
        ep_dir.mkdir(parents=True)
        EpisodeProfile(
            task_id=f"t{i}",
            trajectory_id=f"traj-{i}",
            wall_time_s=setup + loop + evaluate,
            phases={"setup": setup, "agent_loop": loop, "evaluate": evaluate},
            resources={"rss_mb": ResourceSummary(samples=2, mean=100.0, p50=100.0, p95=180.0, max=200.0)},
            sample_count=2,
        ).write(ep_dir)

    rollup = build_rollup(tmp_path)
    assert rollup["episodes"] == 2
    phases = rollup["phases"]
    # agent_loop dominates → ranked first; shares sum to ~1.0
    assert phases[0]["phase"] == "agent_loop"
    assert phases[0]["owner"] == "model"
    assert abs(sum(p["share"] for p in phases) - 1.0) < 1e-6
    # resource peaks aggregated (max-of-max)
    rss = next(r for r in rollup["resources"] if r["resource"] == "rss_mb")
    assert rss["max"] == 200.0
