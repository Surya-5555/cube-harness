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
    # Aggregate only over episodes whose events we can load (covered); means
    # divide by `covered`, not len(profiles), so RL in-memory rollouts / unreadable
    # episodes don't dilute the per-episode numbers (or dump their agent_loop into
    # the overhead bucket).
    agent_loop_total = llm_total = tool_total = 0.0
    floor_total = reclaimable_total = 0.0  # accumulated PER EPISODE (honest floors)
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
        agent_loop_total += prof.phases.get(PHASE_AGENT_LOOP, 0.0)
        llm_total += bd.llm_wait_s
        tool_total += bd.tool_exec_s
        n_llm += bd.n_llm_calls
        n_tool += bd.n_tool_calls
        # Per-episode transport floor: each call pays at least this episode's
        # cheapest call of that tool; what's reclaimable by batching is the
        # floor on every call past the first (you can't collapse a tool to zero).
        for stat in bd.tools.values():
            floor_total += stat.count * stat.min_s
            reclaimable_total += (stat.count - 1) * stat.min_s
        for name, stat in bd.tools.items():
            agg = tools.setdefault(
                name, {"count": 0, "total_s": 0.0, "min_s": stat.min_s, "max_s": 0.0, "floor_s": 0.0}
            )
            agg["count"] += stat.count
            agg["total_s"] += stat.total_s
            agg["floor_s"] += stat.count * stat.min_s  # per-episode floor contribution (honest across episodes)
            agg["min_s"] = min(agg["min_s"], stat.min_s)  # display only
            agg["max_s"] = max(agg["max_s"], stat.max_s)  # display only

    cov = covered or 1
    overhead_total = max(0.0, agent_loop_total - llm_total - tool_total)
    base = agent_loop_total or 1.0
    work_total = max(0.0, tool_total - floor_total)
    tool_base = tool_total or 1.0

    tool_rows = sorted(
        (
            {
                "tool": k,
                "per_call_s": v["total_s"] / (v["count"] or 1),  # per-call mean (comparable to min/max)
                "calls": v["count"],  # total across episodes
                "min_s": v["min_s"],
                "max_s": v["max_s"],
                "share_of_loop": v["total_s"] / base,
                "floor_frac": v["floor_s"] / (v["total_s"] or 1.0),  # how transport-bound this tool is
            }
            for k, v in tools.items()
        ),
        key=lambda r: r["share_of_loop"],
        reverse=True,
    )
    return {
        "episodes_covered": covered,
        "split": [
            {"part": "llm_wait", "mean_s": llm_total / cov, "share": llm_total / base, "owner": "model"},
            {"part": "tool_exec", "mean_s": tool_total / cov, "share": tool_total / base, "owner": "infra/transport"},
            {"part": "overhead", "mean_s": overhead_total / cov, "share": overhead_total / base, "owner": "harness"},
        ],
        "tool_exec_split": {
            "transport_floor_s": floor_total / cov,
            "transport_floor_frac": floor_total / tool_base,
            "work_s": work_total / cov,
            "work_frac": work_total / tool_base,
            "reclaimable_by_batching_s": reclaimable_total / cov,
        },
        "mean_llm_calls": n_llm / cov,
        "mean_tool_calls": n_tool / cov,
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
            f"  [{al.get('mean_llm_calls', 0):.1f} llm calls, {al.get('mean_tool_calls', 0):.1f} tool calls/ep"
            f"; {al.get('episodes_covered', 0)}/{rollup['episodes']} episodes have events]:",
            f"  {'part':<14}{'mean_s':>10}{'share':>9}  owner",
        ]
        for r in al["split"]:
            lines.append(f"  {r['part']:<14}{r['mean_s']:>10.2f}{r['share'] * 100:>8.1f}%  {r['owner']}")

        tes = al.get("tool_exec_split") or {}
        if tes:
            lines += [
                "",
                "  why tool_exec dominates — transport-floor vs in-container work:",
                f"    transport-floor : {tes['transport_floor_s']:8.2f}s/ep "
                f"({tes['transport_floor_frac'] * 100:.0f}% of tool_exec)  ← per-call RPC; batch/co-locate",
                f"    in-container work: {tes['work_s']:8.2f}s/ep "
                f"({tes['work_frac'] * 100:.0f}% of tool_exec)  ← real command runtime; ≈irreducible",
                f"    reclaimable by batching/co-location ≈ {tes['reclaimable_by_batching_s']:.2f}s/ep",
            ]
        if al.get("tools"):
            lines += [
                "",
                "  per-tool, per-call seconds (mean / min=floor / max) · total calls · share of loop · floor%=transport-bound:",
                f"    {'tool':<14}{'per-call':>9}{'min':>8}{'max':>8}{'calls':>7}{'share':>8}{'floor%':>8}",
            ]
            for r in al["tools"]:
                lines.append(
                    f"    {r['tool']:<14}{r['per_call_s']:>9.2f}{r['min_s']:>8.2f}{r['max_s']:>8.2f}"
                    f"{r['calls']:>7}{r['share_of_loop'] * 100:>7.1f}%{r['floor_frac'] * 100:>7.0f}%"
                )

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
