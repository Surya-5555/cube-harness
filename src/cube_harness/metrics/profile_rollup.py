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

from cube_harness.metrics.profiler import PHASE_OWNER, EpisodeProfile

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


def build_rollup(exp_dir: Path) -> dict:
    """Assemble the full rollup dict (also the ``--json`` payload)."""
    profiles = _load_profiles(exp_dir)
    return {
        "exp_dir": str(exp_dir),
        "episodes": len(profiles),
        "mean_wall_s": (sum(p.wall_time_s for p in profiles) / len(profiles)) if profiles else 0.0,
        "phases": _aggregate_phases(profiles),
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
