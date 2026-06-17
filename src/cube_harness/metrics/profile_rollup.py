"""``ch-profile`` — aggregate per-episode ``profile.json`` into a bottleneck map.

This is the Tier-0 rollup of the Auto-CUBE profile use-case: instead of
dispatching a per-trajectory Investigator, read the cheap always-on profiling
artifacts an experiment already produced and Pareto-rank where wall-clock and
resources actually go. The orchestrator reads the ranked table and only spends
deeper profiling (py-spy/cProfile) on the top, *actionable* bars.

Usage::

    ch-profile ~/auto_cube/<session>/experiments/<exp_dir>
    ch-profile <exp_dir> --json out.json

It walks ``<exp_dir>/episodes/*/profile.json`` (the layout
:class:`~cube_harness.metrics.profiler.EpisodeProfile` writes) and prints:

- **Phase Pareto** — mean seconds per phase, share of episode wall-clock, and a
  coarse owner tag (infra / model / benchmark) so non-actionable bars (e.g.
  model-bound generation) are visually separated from addressable ones.
- **Resource peaks** — p95/max of CPU/RSS/IO/GPU across episodes, to catch a
  memory or fd leak the phase view won't show.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from cube_harness.metrics.profiler import PHASE_AGENT_LOOP, PHASE_OWNER, EpisodeProfile, breakdown_agent_loop
from cube_harness.storage import FileStorage

logger = logging.getLogger(__name__)


def _load_profiles(exp_dir: Path) -> list[EpisodeProfile]:
    """Load every episode ``profile.json`` under an experiment dir.

    Tolerant of either ``<exp_dir>/episodes/<id>/`` (canonical) or a flat dir
    of profile files, and of being pointed straight at an episode dir.
    """
    profiles: list[EpisodeProfile] = []
    for path in sorted(exp_dir.rglob("profile.json")):
        try:
            profiles.append(EpisodeProfile.model_validate_json(path.read_text()))
        except Exception:
            logger.warning("Skipping unreadable %s", path, exc_info=True)
    return profiles


def _aggregate_phases(profiles: list[EpisodeProfile]) -> list[dict]:
    """Mean seconds and wall-clock share per phase, ranked descending."""
    total_wall = sum(p.wall_time_s for p in profiles) or 1.0
    sums: dict[str, float] = {}
    for prof in profiles:
        for name, secs in prof.phases.items():
            sums[name] = sums.get(name, 0.0) + secs
    n = len(profiles) or 1
    rows = [
        {
            "phase": name,
            "mean_s": total / n,
            "share": total / total_wall,
            "owner": PHASE_OWNER.get(name, "?"),
        }
        for name, total in sums.items()
    ]
    rows.sort(key=lambda r: r["mean_s"], reverse=True)
    return rows


def _aggregate_resources(profiles: list[EpisodeProfile]) -> list[dict]:
    """Across-episode peak (max of per-episode p95, and absolute max) per series."""
    series: dict[str, dict[str, float]] = {}
    for prof in profiles:
        for name, summ in prof.resources.items():
            agg = series.setdefault(name, {"p95": 0.0, "max": 0.0, "mean_sum": 0.0, "n": 0.0})
            agg["p95"] = max(agg["p95"], summ.p95)
            agg["max"] = max(agg["max"], summ.max)
            agg["mean_sum"] += summ.mean
            agg["n"] += 1
    rows = [
        {"resource": name, "mean": agg["mean_sum"] / (agg["n"] or 1), "p95": agg["p95"], "max": agg["max"]}
        for name, agg in series.items()
    ]
    rows.sort(key=lambda r: r["resource"])
    return rows


def _aggregate_agent_loop(exp_dir: Path, profiles: list[EpisodeProfile]) -> dict:
    """Split the aggregate ``agent_loop`` wall-clock into llm / tool / overhead
    plus a per-tool table, computed from each episode's persisted events.

    Event-derived and post-hoc: works on any past run, no hot-path cost. The
    ``overhead`` slice is the residual (agent_loop − llm − tool): parsing,
    prompt build, obs formatting, serialization.
    """
    storage = FileStorage(exp_dir)
    n = len(profiles) or 1
    agent_loop_total = sum(p.phases.get(PHASE_AGENT_LOOP, 0.0) for p in profiles)
    llm_total = tool_total = 0.0
    n_llm = n_tool = 0
    tools: dict[str, dict] = {}
    covered = 0
    for prof in profiles:
        if not prof.trajectory_id:
            continue
        try:
            view = storage.load_episode(prof.trajectory_id)
        except Exception:
            logger.warning("Could not load events for %s (skipping breakdown)", prof.trajectory_id, exc_info=True)
            continue
        bd = breakdown_agent_loop(view)
        covered += 1
        llm_total += bd.llm_wait_s
        tool_total += bd.tool_exec_s
        n_llm += bd.n_llm_calls
        n_tool += bd.n_tool_calls
        for name, stat in bd.tools.items():
            agg = tools.setdefault(name, {"count": 0, "total_s": 0.0})
            agg["count"] += stat.count
            agg["total_s"] += stat.total_s
    overhead_total = max(0.0, agent_loop_total - llm_total - tool_total)
    base = agent_loop_total or 1.0
    tool_rows = sorted(
        (
            {"tool": k, "mean_s": v["total_s"] / n, "calls": v["count"], "share_of_loop": v["total_s"] / base}
            for k, v in tools.items()
        ),
        key=lambda r: r["mean_s"],
        reverse=True,
    )
    return {
        "episodes_covered": covered,
        "split": [
            {"part": "llm_wait", "mean_s": llm_total / n, "share": llm_total / base, "owner": "model"},
            {"part": "tool_exec", "mean_s": tool_total / n, "share": tool_total / base, "owner": "infra/transport"},
            {"part": "overhead", "mean_s": overhead_total / n, "share": overhead_total / base, "owner": "harness"},
        ],
        "mean_llm_calls": n_llm / n,
        "mean_tool_calls": n_tool / n,
        "tools": tool_rows,
    }


def build_rollup(exp_dir: Path) -> dict:
    """Assemble the full rollup dict (also the ``--json`` payload)."""
    profiles = _load_profiles(exp_dir)
    return {
        "exp_dir": str(exp_dir),
        "episodes": len(profiles),
        "mean_wall_s": (sum(p.wall_time_s for p in profiles) / len(profiles)) if profiles else 0.0,
        "phases": _aggregate_phases(profiles),
        "agent_loop": _aggregate_agent_loop(exp_dir, profiles) if profiles else {},
        "resources": _aggregate_resources(profiles),
    }


def _render(rollup: dict) -> str:
    lines = [
        f"Profile rollup — {rollup['exp_dir']}",
        f"  episodes: {rollup['episodes']}   mean wall-clock: {rollup['mean_wall_s']:.1f}s",
        "",
        "Phase Pareto (mean seconds / share of wall-clock / owner):",
        f"  {'phase':<14}{'mean_s':>10}{'share':>9}  owner",
    ]
    for r in rollup["phases"]:
        lines.append(f"  {r['phase']:<14}{r['mean_s']:>10.2f}{r['share'] * 100:>8.1f}%  {r['owner']}")

    al = rollup.get("agent_loop") or {}
    if al.get("split"):
        lines += [
            "",
            f"agent_loop breakdown (mean seconds / share of agent_loop / owner)"
            f"  [{al.get('mean_llm_calls', 0):.1f} llm calls, {al.get('mean_tool_calls', 0):.1f} tool calls/ep]:",
            f"  {'part':<14}{'mean_s':>10}{'share':>9}  owner",
        ]
        for r in al["split"]:
            lines.append(f"  {r['part']:<14}{r['mean_s']:>10.2f}{r['share'] * 100:>8.1f}%  {r['owner']}")
        if al.get("tools"):
            lines += [
                "",
                "  per-tool (mean seconds / calls-per-ep / share of agent_loop):",
                f"    {'tool':<16}{'mean_s':>10}{'calls':>8}{'share':>9}",
            ]
            for r in al["tools"]:
                lines.append(f"    {r['tool']:<16}{r['mean_s']:>10.2f}{r['calls']:>8}{r['share_of_loop'] * 100:>8.1f}%")

    lines += [
        "",
        "Resource peaks (mean / p95 / max across episodes):",
        f"  {'resource':<18}{'mean':>10}{'p95':>10}{'max':>10}",
    ]
    for r in rollup["resources"]:
        lines.append(f"  {r['resource']:<18}{r['mean']:>10.1f}{r['p95']:>10.1f}{r['max']:>10.1f}")
    return "\n".join(lines)


def main(
    exp_dir: Annotated[Path, typer.Argument(help="Experiment dir holding episodes/*/profile.json.")],
    json_out: Annotated[Path | None, typer.Option("--json", help="Also write the rollup as JSON to this path.")] = None,
) -> None:
    """Aggregate per-episode profiling into a Pareto bottleneck table."""
    rollup = build_rollup(exp_dir)
    if rollup["episodes"] == 0:
        typer.echo(f"No profile.json found under {exp_dir} — run with Experiment(profile=ProfileConfig()).")
        raise typer.Exit(code=1)
    typer.echo(_render(rollup))
    if json_out is not None:
        json_out.write_text(json.dumps(rollup, indent=2))
        typer.echo(f"\nWrote {json_out}")


def cli() -> None:
    typer.run(main)


if __name__ == "__main__":
    cli()
