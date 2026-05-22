"""Symlink each Auto-CUBE use-case's `SKILL.md` into `.claude/skills/auto-cube-<name>`.

Mirrors `scripts/sync_investigator_skills.py`. Run from the repo root.
Idempotent: existing symlinks pointing inside `use_cases/` are
refreshed; correct ones are skipped.

Uses *relative* symlink targets so the `.claude/skills/auto-cube-*`
entries travel through git unchanged and resolve in any checkout
(worktrees included).

Also creates a `.claude/skills/auto-cube/SKILL.md → use_cases/debug/SKILL.md`
alias so `/auto-cube` (no use-case suffix) keeps working — it dispatches
the debug use-case by default.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

USE_CASES_DIR = Path("src/cube_harness/auto_cube/use_cases")
SKILLS_DIR = Path(".claude/skills")
DEFAULT_USE_CASE = "debug"  # what `/auto-cube` (no suffix) aliases to


def _link(skill_md: Path, link_path: Path, *, dry_run: bool) -> bool:
    """Create or refresh a single symlink. Returns True if it changed."""
    relative_target = Path(os.path.relpath(skill_md.resolve(), start=link_path.parent.resolve()))
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink() and link_path.readlink() == relative_target:
            return False  # already correct
        if not dry_run:
            link_path.unlink()
    if not dry_run:
        link_path.parent.mkdir(parents=True, exist_ok=True)
        link_path.symlink_to(relative_target)
    return True


def _sync(*, repo_root: Path, dry_run: bool) -> list[Path]:
    """Create / refresh the symlinks. Returns the list of links created or updated."""
    use_cases_dir = (repo_root / USE_CASES_DIR).resolve()
    if not use_cases_dir.exists():
        raise typer.BadParameter(f"use_cases directory not found: {use_cases_dir}")

    skills_dir = repo_root / SKILLS_DIR
    skills_dir.mkdir(parents=True, exist_ok=True)

    changed: list[Path] = []
    default_skill_md: Path | None = None
    for sub in sorted(use_cases_dir.iterdir()):
        if not sub.is_dir():
            continue
        skill_md = sub / "SKILL.md"
        if not skill_md.exists():
            continue
        link_path = skills_dir / f"auto-cube-{sub.name}" / "SKILL.md"
        if _link(skill_md, link_path, dry_run=dry_run):
            changed.append(link_path)
        if sub.name == DEFAULT_USE_CASE:
            default_skill_md = skill_md

    # The default alias: .claude/skills/auto-cube/SKILL.md → use_cases/<default>/SKILL.md.
    if default_skill_md is not None:
        alias_link = skills_dir / "auto-cube" / "SKILL.md"
        if _link(default_skill_md, alias_link, dry_run=dry_run):
            changed.append(alias_link)

    return changed


def main(
    repo_root: Annotated[Path, typer.Option(help="Repo root.")] = Path("."),
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would change, do nothing.")] = False,
) -> None:
    """Sync auto-cube-skill symlinks. Idempotent."""
    changed = _sync(repo_root=repo_root, dry_run=dry_run)
    if not changed:
        typer.echo("All auto-cube-skill symlinks are up to date.")
        return
    verb = "Would create/update" if dry_run else "Created/updated"
    typer.echo(f"{verb} {len(changed)} symlink(s):")
    for link in changed:
        typer.echo(f"  {link}")


if __name__ == "__main__":
    typer.run(main)
