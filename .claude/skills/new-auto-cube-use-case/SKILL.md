# new-auto-cube-use-case

Scaffold a new Auto-CUBE use-case end-to-end. An Auto-CUBE use-case
plugs a goal function and methodology into the same outer-loop
skeleton; you'll find the debug use-case already in
[`src/cube_harness/auto_cube/use_cases/debug/`](../../../src/cube_harness/auto_cube/use_cases/debug/) —
read it as a canonical example before scaffolding a new one.

Invoke as `/new-auto-cube-use-case`. You interview the user, propose a
shape, scaffold the directory + files, register the symlink, commit.

## When to create a new use-case

When you can answer **yes** to:

- Does my goal differ from the debug use-case (which classifies into
  dispositions and ships Fix Report PRs)? Examples: capability
  profiling (no PRs, just a report of what model can do), optimization
  (sweep configs, pick the best one), coverage-only (fill the matrix
  without shipping fixes).
- Will the same agent prompt + Investigator dispatch pattern repeat
  across multiple sessions? (One-off investigations don't justify a
  use-case.)
- Is it qualitatively different enough that adding a flag to debug
  would muddy its prompt? (Soft test; lean toward "yes, new use-case"
  when in doubt — the cost is low.)

## Interview

Walk the user through these questions and capture the answers verbatim
in a scratch buffer:

1. **Name** — short, hyphenated, identifier-like (`profile`,
   `coverage`, `optimize`, …). Becomes the slash command
   `/auto-cube-<name>` and the directory
   `src/cube_harness/auto_cube/use_cases/<name>/`.
2. **Downstream goal** — 1–2 sentences. What does this use-case
   produce across sessions?
3. **When to use it** — 2–4 bullets. The conditions under which a
   user would invoke this rather than debug.
4. **Methodology / posture** — how does the agent approach a session?
   (Curious scientist? Sweeping engineer? Read-only profiler?) Free
   prose, leave the agent room to reason about numbers; avoid baking
   in rigid constants.
5. **Sampling & rounds** — broad-cheap zoom-out then zoom-in (like
   debug)? Pure sweep? Single-pass profile? Adapt to the goal.
6. **Investigator dispatch** — which Investigator use_case(s) does this
   dispatch by default (`general_blame`, `profiling`,
   `agent_scaffolding`, `hinter`, `fix_audit`, ...)? Will it bias the
   Investigator with an `investigator_extra.md` fragment?
7. **Dispositions** — what classifications should the agent produce
   per task / cell? (Debug uses PASS / model-ceiling / infra-suspect /
   scaffold-suspect / benchmark-suspect / interesting.) Other use-cases
   may want different ontologies — e.g. profile uses pass / fail /
   timeout / refused.
8. **Stopping criterion** — when is a session done?
9. **Outputs** — does the use-case file Fix Report PRs?
   `design-debt` issues? Just produce a REPORT.md? Update the
   coverage ledger?
10. **Cross-session state** — does it read/write the shared
    `coverage.json` ledger? Does it have its own ledger format?

If the user can't answer #2 or #8 crisply, the use-case isn't ready
yet — push back rather than scaffold a vague skill.

## Scaffold

Once the interview is settled:

1. **Create the use-case directory**:
   `src/cube_harness/auto_cube/use_cases/<name>/`

2. **Write `SKILL.md`** by copying
   [`templates/use_case_skill.md`](templates/use_case_skill.md) and
   filling every `<TODO: ...>` placeholder with the interview answers.
   Section headers are part of the schema — keep them as-is so future
   contributors can navigate any use-case the same way. Add or remove
   subsection prose freely.

3. **Write `investigator_extra.md`** if the use-case biases the
   Investigator. One short Markdown file telling the Investigator
   which dispositions to attribute toward / which axes to focus on.
   Skip this file entirely if the default Investigator prompts are
   sufficient — its presence is optional.

4. **(Optional) `scripts/`** — drop in any helpers the use-case needs
   (custom ledger updater, report aggregator, etc.).

5. **Register the symlink**:
   ```bash
   .venv/bin/python scripts/sync_auto_cube_skills.py
   ```
   This creates `.claude/skills/auto-cube-<name>/SKILL.md` as a
   symlink to the source. Commit both the source files and the
   symlink.

6. **Update `src/cube_harness/auto_cube/README.md`** — add a row in
   the "Use-cases" table near the top describing the new use-case.

## Commit message style

```
feat(auto-cube): add <name> use-case for <one-line goal>

<2-3 sentences describing the methodology, what Investigator(s) it
dispatches, and what artefacts it produces. Reference the debug
use-case if this one is derived from / contrasted with it.>
```

## Validation

Soft checks before committing — grep the new `SKILL.md` for these
section headers (the template's schema):

- `## Posture`
- `## Goal`
- `## When to use this use-case`
- `## Methodology`
- `## Sampling & rounds`
- `## Investigator dispatch`
- `## Dispositions`
- `## Stopping criterion`
- `## Outputs`
- `## Cross-session state`

Missing headers don't fail CI — they're hygiene. If the user has a
genuine reason to omit one (e.g. no cross-session state), keep the
header with a one-line "n/a — this use-case is single-session".

Run `make lint` to ensure the sync script's output is clean.
