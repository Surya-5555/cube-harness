# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cube-harness",
#     "timewarp-cube",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "..", editable = true }
# timewarp-cube = { path = "../cubes/timewarp", editable = true }
# ///
"""Reference recipe: Genny on TimeWarp, servers auto-provisioned (no Docker).

This file IS the config — copy it and edit the values. It is not a CLI.

No `infra` entry, unlike recipes/webarena.py: TimeWarp's three sites are plain
Flask processes the cube stands up itself, so there is nothing for an
InfraConfig to provision. The first run clones the upstream repo, runs its
setup.sh (conda env + several GB from HuggingFace), then launches the servers
and waits until they answer; later runs reuse that checkout. Needs `conda` on
PATH — see cubes/timewarp/README.md.

`ui_version` is the experiment: it picks which historical UI era the sites
render (1-6), which is the axis TimeWarp exists to measure. To sweep it, copy
this file per era — and keep TW_WIKI / TW_NEWS / TW_WEBSHOP *unset*, because
auto mode reuses reachable servers found in those vars and then cannot apply
the era you asked for (it warns, and records `ui_version: None`).
"""

from timewarp_cube import ANSWER_PROTOCOL_OVERRIDES, TIMEWARP_CONFIGS

from cube_harness.agents.genny_configs import GENNY_CONFIGS
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig
from cube_harness.recipe import run

agent = GENNY_CONFIGS["default"]
agent.llm_config = LLMConfig(model_name="gpt-5.4-mini", temperature=1.0)

# TimeWarp's answer protocol is not discoverable from the default action descriptions, and two
# measured traps follow from that: `send_message` is terminal and one-shot (submitting the gold
# answer scores 40/40 on a stratified slice, but 0/40 if one scratchpad line precedes it), and 60
# of the 229 deterministic tasks match only the answer's first sentence (-26.6 points for a lead-in
# phrase). These overrides say both out loud. Drop this line to measure the un-hinted baseline —
# it changes what the agent sees, so the two are not comparable.
agent.description_overrides = dict(ANSWER_PROTOCOL_OVERRIDES)

# ConfigRegistry hands out a deep copy, so mutating this is local to the recipe.
benchmark = TIMEWARP_CONFIGS["wiki"]  # "default" | "wiki" | "news" | "webshop"
benchmark.ui_version = 1

exp = Experiment(
    name="timewarp",
    agent_config=agent,
    benchmark_config=benchmark,
    max_steps=30,  # matches recommended_max_steps in task_metadata.json
)

if __name__ == "__main__":
    run(exp)
