"""Non-Docker auto-provisioning for the TimeWarp environment servers.

TimeWarp's three sites (wiki / news / webshop) are plain Flask apps that live in the
upstream project (https://github.com/sparklabutah/timewarp), not in this package.
This module reproduces the *logic* of that repo's ``setup.sh`` / ``run_all_env.sh``
so the cube can stand the servers up on its own — no Docker, just local processes:

    L1 (slow, shared, once per machine) — ``ensure_provisioned()``
        Clone the upstream repo into a cache dir and run its (idempotent) ``setup.sh``:
        create the ``timewarp`` conda env, install Playwright, run the webshop setup
        (gdown data from Google Drive + faiss index), and fetch the wiki/news index
        pickles from HuggingFace. Skipped when ``is_provisioned()`` is already True.

    L2 (per benchmark run) — ``start_servers()`` / ``TimeWarpServers.stop()``
        Pick free ports via the shared ``cube.infra_utils.free_port`` helper (a
        process-wide lock + PID-derived offset keep parallel runs off the same port),
        launch the three apps under the ``timewarp`` conda env (one process group each,
        for clean teardown), and health-check them.

The servers bind ``127.0.0.1`` only, so the whole flow is intrinsically single-host; the
benchmark threads the resolved URLs through its ``runtime_context`` — re-derived on every
run, so a resumed run always uses the freshly-launched ports — rather than env vars, which
do not reach Ray workers. See ``benchmark.py`` for the orchestration.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from cube.infra_utils import free_port

logger = logging.getLogger(__name__)

#: Upstream TimeWarp project that ships the environment servers and start scripts.
UPSTREAM_REPO = "https://github.com/sparklabutah/timewarp"
UPSTREAM_BRANCH = "master"

#: Optional pinned commit for reproducibility (PS-001). None tracks the branch tip, which is
#: NOT reproducible; set to a full SHA to lock the provisioned upstream code + data layout.
#: The resolved HEAD is always logged after a clone so a run records exactly what it used.
UPSTREAM_COMMIT: str | None = None

#: Conda environment that upstream ``setup.sh`` creates; the servers run inside it.
CONDA_ENV = "timewarp"

#: site -> environment variable browsergym-timewarp reads (see instance.py).
SITE_ENV_VARS: dict[str, str] = {"wiki": "TW_WIKI", "news": "TW_NEWS", "webshop": "TW_WEBSHOP"}

#: First port we ask ``free_port`` to probe from; mirrors upstream ``run_all_env.sh`` (5000+).
_DEFAULT_START_PORT = 5000

#: Subprocess timeouts (s): quick conda/git queries, the shallow clone, and the one-time
#: setup.sh (multi-GB downloads — large but bounded so a stalled fetch can't hang forever).
_QUICK_CMD_TIMEOUT_S = 60
_CLONE_TIMEOUT_S = 600
_SETUP_TIMEOUT_S = 3600


def default_checkout_dir() -> Path:
    """Where the upstream repo is cloned. Override with ``TIMEWARP_HOME``."""
    base = os.environ.get("TIMEWARP_HOME")
    return Path(base).expanduser() if base else Path.home() / ".cache" / "timewarp"


# ── Reachability helpers ─────────────────────────────────────────────────────


def is_reachable(url: str, timeout: float = 5.0) -> bool:
    """True if *url* answers at all — any HTTP status (even 4xx/5xx) counts as up."""
    try:
        with contextlib.closing(urllib.request.urlopen(url, timeout=timeout)):  # noqa: S310 (localhost only)
            return True
    except urllib.error.HTTPError:
        return True  # server answered (login redirect, 404 at /, …) — it's up
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def urls_from_env() -> dict[str, str] | None:
    """The three site URLs from ``TW_WIKI`` / ``TW_NEWS`` / ``TW_WEBSHOP``, or None if any is unset."""
    urls: dict[str, str] = {}
    for site, var in SITE_ENV_VARS.items():
        val = os.environ.get(var)
        if not val:
            return None  # all-or-nothing: a partial set is treated as "not configured"
        urls[site] = val
    return urls


def apply_to_env(urls: dict[str, str]) -> None:
    """Export the resolved URLs so ``browsergym.timewarp.TimeWarpInstance`` (which reads
    ``os.environ`` directly) picks them up. Called both on the driver and, crucially, in
    the worker process via ``TimeWarpTask.reset()`` — driver env does not reach Ray workers.
    """
    for site, var in SITE_ENV_VARS.items():
        os.environ[var] = urls[site]


# ── conda helpers ─────────────────────────────────────────────────────────────


def has_conda() -> bool:
    """True if a ``conda`` executable is on PATH (auto mode needs it for the server env)."""
    return shutil.which("conda") is not None


def _conda_bin() -> str:
    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError(
            "conda not found on PATH. TimeWarp's servers run in a conda env created by the "
            "upstream setup.sh. Install conda/miniconda, or run the cube in manual mode "
            "(start the servers yourself and set TW_WIKI/TW_NEWS/TW_WEBSHOP)."
        )
    return conda


def _conda_env_exists(env: str = CONDA_ENV) -> bool:
    """True iff *env* is a known conda env. Returns False (never raises) when conda is absent
    or unresponsive, so ``is_provisioned`` stays a clean predicate and the actionable
    "install conda" error is raised by the explicit ``_conda_bin()`` gate in the slow path."""
    conda = shutil.which("conda")
    if conda is None:
        return False
    try:
        result = subprocess.run([conda, "env", "list"], capture_output=True, text=True, timeout=_QUICK_CMD_TIMEOUT_S)
    except (subprocess.SubprocessError, OSError):
        return False
    if result.returncode != 0:
        return False
    # `conda env list` rows are "<name>  [*]  <path>"; the leading token is the env name.
    return any(line.split() and line.split()[0] == env for line in result.stdout.splitlines())


# ── L1: provisioning (clone + setup.sh) ──────────────────────────────────────


def _nonempty_file(path: Path) -> bool:
    """True iff *path* is a regular file with content — a truncated download has size 0."""
    return path.is_file() and path.stat().st_size > 0


def _nonempty_dir(path: Path) -> bool:
    """True iff *path* is a directory with at least one entry — an interrupted setup can
    leave the data dir created but empty."""
    return path.is_dir() and any(path.iterdir())


def is_provisioned(checkout_dir: Path | None = None) -> bool:
    """True when the repo, the conda env, and the downloaded data/index files are all present
    *and non-empty*.

    Mirrors what upstream ``setup.sh`` produces, so ``ensure_provisioned`` can skip the slow
    path on a machine that has already been set up. The data/index checks look at file size /
    directory contents (not mere existence) so an interrupted ``setup.sh`` — which can leave a
    zero-byte index pickle or an empty data dir behind — is treated as *not* provisioned and
    re-run rather than launching servers against corrupt data.
    """
    checkout_dir = checkout_dir or default_checkout_dir()
    return (
        (checkout_dir / "setup.sh").is_file()
        and _nonempty_file(checkout_dir / "env" / "wiki" / "wiki_index.pkl")
        and _nonempty_file(checkout_dir / "env" / "news" / "news_index.pkl")
        and _nonempty_dir(checkout_dir / "env" / "webshop" / "data")
        and _conda_env_exists()
    )


def _clone_repo(checkout_dir: Path) -> None:
    if (checkout_dir / "setup.sh").is_file():
        return
    checkout_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning %s (%s) into %s …", UPSTREAM_REPO, UPSTREAM_COMMIT or UPSTREAM_BRANCH, checkout_dir)
    if UPSTREAM_COMMIT:
        # A specific SHA may not be a branch tip, so a shallow --branch clone can't reach it.
        subprocess.run(["git", "clone", UPSTREAM_REPO, str(checkout_dir)], check=True, timeout=_CLONE_TIMEOUT_S)
        subprocess.run(
            ["git", "checkout", UPSTREAM_COMMIT], cwd=str(checkout_dir), check=True, timeout=_QUICK_CMD_TIMEOUT_S
        )
    else:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", UPSTREAM_BRANCH, UPSTREAM_REPO, str(checkout_dir)],
            check=True,
            timeout=_CLONE_TIMEOUT_S,
        )
    _log_resolved_commit(checkout_dir)


def _log_resolved_commit(checkout_dir: Path) -> None:
    """Record the exact upstream commit the checkout resolved to (PS-001 reproducibility)."""
    with contextlib.suppress(subprocess.SubprocessError, OSError):
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(checkout_dir),
            capture_output=True,
            text=True,
            timeout=_QUICK_CMD_TIMEOUT_S,
        )
        if result.returncode == 0:
            logger.info("TimeWarp upstream checked out at commit %s", result.stdout.strip())


def ensure_provisioned(checkout_dir: Path | None = None, *, force: bool = False) -> Path:
    """L1 — clone the upstream repo and run its idempotent ``setup.sh`` if not already set up.

    Cheap when ``is_provisioned()`` is already True (the common case). ``setup.sh`` itself
    skips re-creating the conda env and re-downloading existing index/data files, so a
    forced re-run only fetches what is genuinely missing.
    """
    checkout_dir = checkout_dir or default_checkout_dir()
    _clone_repo(checkout_dir)
    if not force and is_provisioned(checkout_dir):
        logger.info("TimeWarp already provisioned at %s — skipping setup.sh", checkout_dir)
        return checkout_dir
    _conda_bin()  # fail fast with an actionable message before the long setup
    logger.info("Running upstream setup.sh in %s (one-time; downloads conda env + data) …", checkout_dir)
    subprocess.run(["bash", "setup.sh"], cwd=str(checkout_dir), check=True, timeout=_SETUP_TIMEOUT_S)
    return checkout_dir


# ── L2: launching the servers ────────────────────────────────────────────────


def _env_python(env: str = CONDA_ENV) -> str:
    """Absolute path to the conda env's interpreter — launching it directly avoids
    `conda run` output-buffering/signal quirks with long-lived background servers.

    Resolved fresh on each launch (no process-level cache) so a rebuilt/relocated conda env
    is always picked up — start_servers calls this once per benchmark setup, so it is cheap.
    """
    result = subprocess.run(
        [_conda_bin(), "run", "-n", env, "python", "-c", "import sys; print(sys.executable)"],
        capture_output=True,
        text=True,
        check=True,
        timeout=_QUICK_CMD_TIMEOUT_S,
    )
    # `conda run` can prepend activation notices; the interpreter path is the last line.
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Could not resolve the timewarp conda env's python interpreter.")
    return lines[-1].strip()


def _site_command(site: str, app_python: str, port: int, ui_version: int) -> list[str]:
    """The exact app invocation upstream ``run_all_env.sh`` uses, per site."""
    if site == "wiki":
        return [app_python, "wiki_app.py", f"-{ui_version}", f"--port={port}"]
    if site == "news":
        return [app_python, "news_app.py", f"-{ui_version}", f"--port={port}"]
    if site == "webshop":
        return [app_python, "-m", "web_agent_site.app", str(ui_version), f"--port={port}", "--log", "--attrs"]
    raise ValueError(f"Unknown TimeWarp site: {site!r}")


@dataclass
class TimeWarpServers:
    """Live handle to the three locally-launched TimeWarp Flask servers.

    ``urls`` maps site -> base URL (webshop carries the upstream ``/abc`` suffix).
    ``stop()`` kills each server's process group, reaps it, and closes its log files; safe to
    call more than once. ``_stderr_paths`` (site order) lets the healthcheck surface a crashed
    server's stderr instead of just an exit code.
    """

    urls: dict[str, str]
    checkout_dir: Path
    _procs: list[subprocess.Popen] = field(default_factory=list, repr=False)
    _logs: list[TextIO] = field(default_factory=list, repr=False)
    _stderr_paths: list[Path] = field(default_factory=list, repr=False)

    def stop(self) -> None:
        for proc in self._procs:
            if proc.poll() is not None:
                continue
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                continue
        for proc in self._procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                # Reap the killed child so it does not linger as a zombie for the driver's life.
                with contextlib.suppress(subprocess.SubprocessError, OSError):
                    proc.wait(timeout=10)
        self._procs.clear()
        for log in self._logs:
            with contextlib.suppress(OSError):
                log.close()
        self._logs.clear()

    def __enter__(self) -> TimeWarpServers:
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def _build_url(site: str, port: int, host: str) -> str:
    base = f"http://{host}:{port}"
    return f"{base}/abc" if site == "webshop" else base  # webshop is mounted under /abc upstream


def _open_logs(site: str) -> tuple[TextIO, TextIO, Path]:
    """Open per-site stdout/stderr log files so a crashed server's output is inspectable
    (returns the open handles plus the stderr path for the healthcheck error message)."""
    tmp = Path(tempfile.gettempdir())
    out_path = tmp / f"timewarp_{site}_stdout.log"
    err_path = tmp / f"timewarp_{site}_stderr.log"
    return out_path.open("w"), err_path.open("w"), err_path


def _tail(path: Path, max_lines: int = 20) -> str:
    """Last few lines of a server log, formatted for an error message."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return "(no server log captured)"
    tail = "\n".join(lines[-max_lines:])
    return f"--- last lines of {path} ---\n{tail}" if tail else f"(server log {path} is empty)"


def start_servers(
    checkout_dir: Path | None = None,
    ui_version: int = 1,
    *,
    host: str = "127.0.0.1",
    start_port: int = _DEFAULT_START_PORT,
    healthcheck_timeout_s: float = 180.0,
) -> TimeWarpServers:
    """L2 — launch wiki/news/webshop under the ``timewarp`` conda env and wait until healthy.

    Each app runs in its own process group (``start_new_session=True``) so ``stop()`` can
    tear it down precisely without the host-wide ``pkill`` sweep upstream's stop script uses.
    Ports come from the shared ``cube.infra_utils.free_port`` helper — its process-wide lock
    plus PID-derived offset stop two concurrent benchmark setups racing onto the same port.
    Bind-probe, URL, and health-check all use ``127.0.0.1`` so there is no IPv4/IPv6 ambiguity
    about which interface is actually being served.
    """
    checkout_dir = checkout_dir or default_checkout_dir()
    if not (1 <= ui_version <= 6):
        raise ValueError(f"ui_version must be 1-6, got {ui_version}")

    app_python = _env_python()
    sites = list(SITE_ENV_VARS)  # wiki, news, webshop
    ports = [free_port(start=start_port) for _ in sites]
    urls = {site: _build_url(site, port, host) for site, port in zip(sites, ports)}

    servers = TimeWarpServers(urls=urls, checkout_dir=checkout_dir)
    try:
        for site, port in zip(sites, ports):
            site_dir = checkout_dir / "env" / site
            cmd = _site_command(site, app_python, port, ui_version)
            logger.info("Starting %s (theme %d) on port %d: %s", site, ui_version, port, " ".join(cmd))
            out_f, err_f, err_path = _open_logs(site)
            servers._logs += [out_f, err_f]
            servers._stderr_paths.append(err_path)
            servers._procs.append(
                subprocess.Popen(cmd, cwd=str(site_dir), stdout=out_f, stderr=err_f, start_new_session=True)
            )
        _wait_until_healthy(servers, timeout_s=healthcheck_timeout_s)
    except Exception:
        servers.stop()
        raise
    logger.info("TimeWarp servers ready: %s", urls)
    return servers


def _wait_until_healthy(servers: TimeWarpServers, timeout_s: float, interval_s: float = 2.0) -> None:
    """Block until every site answers, or raise if any process dies / the timeout elapses."""
    deadline = time.monotonic() + timeout_s
    sites = list(servers.urls)
    pending = dict(servers.urls)
    while pending and time.monotonic() < deadline:
        # Any early exit is a failure — a server that bound nothing but exited 0 would
        # otherwise never become reachable and silently burn the whole timeout.
        for site, proc, stderr_path in zip(sites, servers._procs, servers._stderr_paths):
            if proc.poll() is not None:
                raise RuntimeError(
                    f"TimeWarp '{site}' server exited early (code {proc.returncode}). "
                    f"Check the conda env '{CONDA_ENV}' and that {servers.checkout_dir}/env/{site} is set up.\n"
                    f"{_tail(stderr_path)}"
                )
        for site, url in list(pending.items()):
            if is_reachable(url, timeout=5.0):
                logger.info("  %s healthy at %s", site, url)
                del pending[site]
        if pending:
            time.sleep(interval_s)
    if pending:
        raise TimeoutError(f"TimeWarp servers not healthy after {timeout_s:.0f}s: {sorted(pending)}")
