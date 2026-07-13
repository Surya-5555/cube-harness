"""TimeWarp benchmark implementation for the CUBE framework.

TimeWarp's three web environments (wiki / news / webshop) are external Flask servers,
addressed via the ``TW_WIKI`` / ``TW_NEWS`` / ``TW_WEBSHOP`` environment variables. The
servers and their start scripts live in the upstream TimeWarp project
(https://github.com/sparklabutah/timewarp), not in this package. There is no Docker.

Two provisioning modes (``provision_mode`` on the config):

* ``"auto"`` (default) — the cube stands the servers up itself, no Docker. On setup it
  checks whether the environment is already provisioned (repo cloned, conda env built,
  Google-Drive + HuggingFace data present); if not, it runs the upstream (idempotent)
  ``setup.sh`` to download/build, then launches the three Flask apps and waits until they
  are healthy. If ``TW_WIKI`` / ``TW_NEWS`` / ``TW_WEBSHOP`` already point at reachable
  servers, those are used as-is (nothing is launched). The whole flow is single-host —
  the servers bind ``127.0.0.1`` — so resolved URLs are published into the benchmark's
  ``runtime_context`` (re-derived on every run, so a resumed run picks up the freshly-launched
  ports) rather than relying on env vars reaching Ray workers. See ``provisioning.py``.

* ``"manual"`` — the historical behaviour: start the servers yourself (set the three env
  vars) and ``_setup()`` only verifies they are reachable.
"""

import importlib.resources
import json
import logging
import os
from pathlib import Path
from typing import ClassVar, Literal

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.resource import InfraConfig
from cube.task import TaskConfig

from timewarp_cube import provisioning
from timewarp_cube.task import TimeWarpTaskConfig, TimeWarpTaskMetadata

logger = logging.getLogger(__name__)

#: Environment variables that must point at reachable TimeWarp servers (manual mode).
_REQUIRED_ENV_VARS = tuple(provisioning.SITE_ENV_VARS.values())  # TW_WIKI, TW_NEWS, TW_WEBSHOP

_START_HINT = (
    "Either run in the default auto mode (provision_mode='auto') with conda available, or\n"
    f"start the servers manually from the upstream TimeWarp repo ({provisioning.UPSTREAM_REPO}):\n"
    "  bash setup.sh                              # one-time: conda env + deps + data\n"
    "  bash scripts/environment/run_all_env.sh 1  # start all three (UI version 1)\n"
    "Then point the cube at the running servers and the judge:\n"
    "  export TW_WIKI=... TW_NEWS=... TW_WEBSHOP=...   # URLs printed by the script\n"
    "  export OPENAI_API_KEY=...                       # llm_judge scores every task"
)

#: Task count derived from the shipped task_metadata.json so ``num_tasks`` can't drift from it.
_NUM_TASKS = len(json.loads(importlib.resources.files("timewarp_cube").joinpath("task_metadata.json").read_text()))


class TimeWarpBenchmark(Benchmark["TimeWarpBenchmarkConfig"]):
    """Runtime pair. In ``auto`` mode it owns the launched Flask servers and resolves the
    site URLs; in ``manual`` mode it only verifies the externally-started servers are up.
    """

    def __init__(self, config: "TimeWarpBenchmarkConfig", infra: InfraConfig | None = None) -> None:
        super().__init__(config, infra=infra)
        self._servers: provisioning.TimeWarpServers | None = None

    def _setup(self) -> None:
        if self.config.provision_mode == "manual":
            self._setup_manual()
        else:
            self._setup_auto()
        logger.info("TimeWarp benchmark ready with %d tasks", self.config.num_tasks)

    def _setup_manual(self) -> None:
        """Verify the three externally-started servers are set and reachable."""
        missing = [var for var in _REQUIRED_ENV_VARS if not os.environ.get(var)]
        if missing:
            raise RuntimeError(f"Missing TimeWarp environment variable(s): {', '.join(missing)}.\n{_START_HINT}")
        for var in _REQUIRED_ENV_VARS:
            url = os.environ[var]
            if not provisioning.is_reachable(url):
                raise RuntimeError(f"Cannot reach TimeWarp server {var}={url}.\n{_START_HINT}")

    def _setup_auto(self) -> None:
        """Reuse already-running servers if present, else provision + launch them."""
        set_vars = [var for var in _REQUIRED_ENV_VARS if os.environ.get(var)]
        if set_vars and len(set_vars) != len(_REQUIRED_ENV_VARS):
            raise RuntimeError(
                f"Auto mode found only some TimeWarp server env vars set ({', '.join(set_vars)}). "
                f"Set all of {', '.join(_REQUIRED_ENV_VARS)} to reuse running servers, or none "
                f"to let the cube launch them.\n{_START_HINT}"
            )
        existing = provisioning.urls_from_env()
        if existing is not None and all(provisioning.is_reachable(url) for url in existing.values()):
            logger.info("Using already-running TimeWarp servers: %s", existing)
            urls = existing
        else:
            if existing is not None:
                # All three env vars are set but at least one server isn't answering — don't
                # silently run against them. Launch cube-managed servers instead and say so.
                logger.warning(
                    "TimeWarp server env vars (%s) are set but not all reachable — launching "
                    "cube-managed servers and overriding them. Unset them, or fix/restart the "
                    "servers, to reuse your own.",
                    ", ".join(_REQUIRED_ENV_VARS),
                )
            checkout = self.config.checkout_dir
            provisioning.ensure_provisioned(checkout)
            self._servers = provisioning.start_servers(checkout, self.config.ui_version)
            urls = self._servers.urls
        # Resolved URLs reach Ray workers via runtime_context (re-derived per run, so a resume
        # uses these ports, not a stale persisted set); also export them here so driver-side /
        # sequential (debug) runs see them directly.
        self._runtime_context["tw_urls"] = urls
        provisioning.apply_to_env(urls)

    def close(self) -> None:
        if self._servers is not None:
            logger.info("Stopping TimeWarp servers.")
            self._servers.stop()
            self._servers = None
        logger.info("TimeWarp benchmark closed.")


class TimeWarpBenchmarkConfig(BenchmarkConfig[TimeWarpTaskMetadata]):
    """CUBE BenchmarkConfig for TimeWarp — 231 temporal-UI web tasks.

    ``provision_mode="auto"`` (default) stands the wiki/news/webshop Flask servers up with
    no Docker: it checks whether the upstream environment is already set up and, if not,
    runs the upstream ``setup.sh`` (conda env + Google-Drive/HuggingFace data) before
    launching the servers at ``ui_version``. ``provision_mode="manual"`` instead expects
    you to start the servers and set ``TW_WIKI`` / ``TW_NEWS`` / ``TW_WEBSHOP`` yourself.
    Either way, set ``OPENAI_API_KEY`` for the ``llm_judge`` evaluator.

    Filter by site in user-land via named subsets or glob:
        cfg.named_subset("wiki")
        cfg.subset_from_glob("sites", "*news*")

    task_metadata.json is a shipped package resource with lightweight public fields
    (sites, intent_template_id, eval_types). No heavy execution data exists — all
    task logic is available from the browsergym-timewarp library at runtime.

    To regenerate task_metadata.json (developer use only), run:
        scripts/generate_task_metadata.py
    """

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="timewarp-cube",
        version="0.1.0",
        description=(
            "TimeWarp benchmark — 231 web tasks across wiki, news, and webshop, "
            "designed to test agent robustness to temporal changes in web UI. "
            "Use named_subset('wiki'/'news'/'webshop') to filter by site."
        ),
        num_tasks=_NUM_TASKS,
        tags=["browser", "web", "ui", "timewarp"],
        named_subsets={
            "wiki": ("sites", "*wiki*"),
            "news": ("sites", "*news*"),
            "webshop": ("sites", "*webshop*"),
        },
    )
    task_config_class: ClassVar[type[TaskConfig]] = TimeWarpTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = TimeWarpBenchmark

    provision_mode: Literal["auto", "manual"] = "auto"
    """``auto``: clone-check + run upstream setup.sh if needed + launch the Flask servers.
    ``manual``: start the servers yourself and set TW_WIKI/TW_NEWS/TW_WEBSHOP."""
    ui_version: int = 1
    """Temporal UI era (1-6) the auto-launched servers render. Ignored in manual mode."""
    checkout_dir: Path | None = None
    """Where ``_setup_auto`` (via ``provisioning.ensure_provisioned``) clones the upstream
    TimeWarp repo in auto mode. None → ``TIMEWARP_HOME`` env var, else ``~/.cache/timewarp``
    (see provisioning.default_checkout_dir)."""

    @classmethod
    def install(cls) -> None:
        """L1 hook — intentionally lightweight.

        Auto mode provisions the upstream repo + conda env + data lazily on first run
        (``_setup_auto`` -> ``provisioning.ensure_provisioned``, idempotent), so there is nothing
        slow to do here. This matters because the test harness calls ``install()`` before every
        ``cube test`` — including the manual-mode debug suite, which only needs the servers
        reachable — so heavy provisioning here would run (and usually fail) on the common case of a
        machine that has conda but no running servers.
        """
        super().install()
        logger.info(
            "timewarp-cube install(): auto mode provisions the environment lazily on first run; nothing to do here."
        )
