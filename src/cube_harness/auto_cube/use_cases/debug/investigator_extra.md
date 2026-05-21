## Auto-CUBE debug use-case — Investigator biasing

You are being dispatched by Auto-CUBE running its **debug** use-case.
The orchestrator is building a sparse coverage map across
task × infra × tool × model × agent-config; your output feeds straight
back into its disposition decisions.

When you write your report, **bias your attribution toward one of these
dispositions** (use whichever fits; multiple are allowed if the evidence
supports it):

- **PASS** — the agent solved the task. No blame.
- **model-ceiling** — the task is plausibly solvable but the model
  lacks the capability (reasoning depth, knowledge, tool use skill).
  Don't blame infra/scaffold here. Future-model retry is the action.
- **infra-suspect** — failure looks driven by the infrastructure
  layer (container, network, resource provisioning, lifecycle,
  timeouts, missing CLIs, broken volumes). Name the suspected
  component if you can.
- **scaffold-suspect** — failure looks driven by the agent loop or
  tool surface (parser bug, missing action, prompt issue, premature
  termination, broken tool output rendering). Be specific about which
  scaffold layer.
- **benchmark-suspect** — the cube itself looks wrong (ambiguous task
  prompt, broken ground truth, contaminated training data leaking into
  the task, impossible-but-marked-possible scoring, evaluation logic
  bug). This is a high-value finding — flag it explicitly.

If the cause is genuinely ambiguous, say so — "pending, needs
zoom-in with these axes varied: ..." is a valid output and tells
the orchestrator what to try next.

**Be conservative.** A wrong attribution wastes a zoom-in round.
When two dispositions are plausible, name both and rank them; the
orchestrator will design the next experiment to disambiguate.
