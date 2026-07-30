"""TimeWarp benchmark implementation for the CUBE framework.

TimeWarp's three web environments (wiki / news / webshop) are external Flask servers,
addressed via the ``TW_WIKI`` / ``TW_NEWS`` / ``TW_WEBSHOP`` environment variables. The
servers and their start scripts live in the upstream TimeWarp project
(https://github.com/sparklabutah/timewarp), not in this package. There is no Docker.

Two provisioning modes (``provision_mode`` on the config):

* ``"auto"`` (default) — the cube stands the servers up itself, no Docker. On setup it
  checks whether the environment is already provisioned (repo cloned, conda env built,
  HuggingFace data present); if not, it runs the upstream (idempotent)
  ``setup.sh`` to download/build, then launches the three Flask apps and waits until they
  are healthy. If ``TW_WIKI`` / ``TW_NEWS`` / ``TW_WEBSHOP`` already point at reachable
  servers, those are used as-is (nothing is launched). The whole flow is single-host —
  the servers bind ``127.0.0.1`` — so resolved URLs are published into the benchmark's
  ``runtime_context`` (re-derived on every run, so a resumed run picks up the freshly-launched
  ports) rather than relying on env vars reaching Ray workers. See ``provisioning.py``.

* ``"manual"`` — the historical behaviour: start the servers yourself (set the three env
  vars) and ``_setup()`` only verifies they are reachable.
"""

import logging
import os
from pathlib import Path
from typing import Annotated, ClassVar, Literal

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.resource import InfraConfig
from cube.task import TaskConfig
from pydantic import Field

from timewarp_cube import provisioning
from timewarp_cube._data import load_task_metadata, verify_upstream_data
from timewarp_cube.task import TimeWarpTaskConfig, TimeWarpTaskMetadata

logger = logging.getLogger(__name__)

#: Environment variables that must point at reachable TimeWarp servers (manual mode).
_REQUIRED_ENV_VARS = tuple(provisioning.SITE_ENV_VARS.values())  # TW_WIKI, TW_NEWS, TW_WEBSHOP

_START_HINT = (
    "Either run in the default auto mode (provision_mode='auto') with conda available, or\n"
    f"start the servers manually from the upstream TimeWarp repo ({provisioning.UPSTREAM_REPO}):\n"
    "  bash setup.sh                              # one-time: conda env + deps + data\n"
    "  bash scripts/environment/run_all_env.sh 1  # start all three (UI version 1)\n"
    "Then point the cube at the running servers:\n"
    "  export TW_WIKI=... TW_NEWS=... TW_WEBSHOP=...   # URLs printed by the script\n"
    "  export OPENAI_API_KEY=...                       # optional: only the two llm_judge tasks"
)

#: Task count derived from the shipped task_metadata.json so ``num_tasks`` can't drift from it.
_NUM_TASKS = len(load_task_metadata())


class TimeWarpBenchmark(Benchmark["TimeWarpBenchmarkConfig"]):
    """Runtime pair. In ``auto`` mode it owns the launched Flask servers and resolves the
    site URLs; in ``manual`` mode it only verifies the externally-started servers are up.
    """

    def __init__(self, config: "TimeWarpBenchmarkConfig", infra: InfraConfig | None = None) -> None:
        super().__init__(config, infra=infra)
        self._servers: provisioning.TimeWarpServers | None = None

    def _setup(self) -> None:
        verify_upstream_data()  # a wrong upstream release scores every task wrong, silently
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
            # Servers we did not launch: we cannot know — or change — which UI era they render,
            # so ui_version does not apply and the era is recorded as unknown rather than as the
            # value that was asked for but never took effect. Warn, because ui_version *is* the
            # experiment in TimeWarp: silently running era 1 under an "era 3" label is a wrong
            # result, not a slow one.
            logger.warning(
                "Using already-running TimeWarp servers (%s). ui_version=%d is NOT applied — the UI "
                "era these servers render is whatever they were started with, and cannot be verified "
                "from here; episodes will record ui_version=None. Unset %s to have the cube launch "
                "its own servers at ui_version=%d.",
                existing,
                self.config.ui_version,
                ", ".join(_REQUIRED_ENV_VARS),
                self.config.ui_version,
            )
            urls, ui_version = existing, None
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
            # Take ensure_provisioned's resolved path rather than passing the possibly-None config
            # value on to start_servers: otherwise each resolves default_checkout_dir()
            # independently, and a TIMEWARP_HOME changed in between would provision one tree and
            # launch from another.
            checkout = provisioning.ensure_provisioned(self.config.checkout_dir)
            self._servers = provisioning.start_servers(checkout, self.config.ui_version)
            urls, ui_version = self._servers.urls, self.config.ui_version
        # Resolved URLs reach Ray workers via runtime_context (re-derived per run, so a resume
        # uses these ports, not a stale persisted set); also export them here so driver-side /
        # sequential (debug) runs see them directly. The resolved era rides along so every
        # trajectory records which UI it actually ran against (PS-001) — see TimeWarpTask.reset.
        self._runtime_context["tw_urls"] = urls
        self._runtime_context["tw_ui_version"] = ui_version
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
    runs the upstream ``setup.sh`` (conda env + HuggingFace data) before launching the
    servers at ``ui_version``. ``provision_mode="manual"`` instead expects you to start
    the servers and set ``TW_WIKI`` / ``TW_NEWS`` / ``TW_WEBSHOP`` yourself.

    Scoring is deterministic for all but two tasks, so ``OPENAI_API_KEY`` (or the
    ``TW_JUDGE*`` vars, to point at another judge) is only needed for those two — without
    it they error rather than score 0.

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
        version="0.2.0",
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
    ui_version: Annotated[int, Field(ge=1, le=6)] = 1
    """Which UI theme (1-6) the cube's own servers render — TimeWarp's independent variable.

    **1-5 are eras; 6 is a neutral control theme, not a sixth era.** And the era indices are *not
    time-aligned across sites* — read from upstream's ``num_to_theme`` maps:

    ==========  ==============  ================  ============
    ui_version  wiki            news              webshop
    ==========  ==============  ================  ============
    1           2001            2000s             2000
    2           2002            2004s             2005
    3           2003-4          2008s             2010
    4           2005-2022       2016s             2015
    5           2023-2025       2024s             2025
    6           minimal         base-minimal      classic
    ==========  ==============  ================  ============

    So a multi-site run at ``ui_version=4`` is 2005-2022 wiki + 2016 news + 2015 webshop, and a
    1→6 sweep is not a monotonic walk through time at its endpoint. Treat the number as a theme
    index; do not read a year off it. (Upstream wiki also defines ``7: modern`` but excludes it
    from its own sweeps, and news/webshop have no 7 — hence the 1-6 bound.)

    Applies only when the cube launches the servers. Not applied in manual mode, nor when auto
    mode reuses reachable ``TW_*`` servers (logged as a warning; episodes then record
    ``ui_version: None``, since the theme of a server we did not start is unknowable)."""
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
