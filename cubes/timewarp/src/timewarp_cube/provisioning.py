"""Non-Docker auto-provisioning for the TimeWarp environment servers.

TimeWarp's three sites (wiki / news / webshop) are plain Flask apps that live in the
upstream project (https://github.com/sparklabutah/timewarp), not in this package.
This module reproduces the *logic* of that repo's ``setup.sh`` / ``run_all_env.sh``
so the cube can stand the servers up on its own — no Docker, just local processes:

    L1 (slow, shared, once per machine) — ``ensure_provisioned()``
        Clone the upstream repo into a cache dir and run its (idempotent) ``setup.sh``:
        create the ``timewarp`` conda env, install Playwright, run the webshop setup
        (data + search index), and fetch the wiki/news index pickles. All downloads come
        from HuggingFace. Skipped when ``is_provisioned()`` is already True.

    L2 (per benchmark run) — ``start_servers()`` / ``TimeWarpServers.stop()``
        Pick free ports via the shared ``cube.infra_utils.free_port`` helper (a
        process-wide lock + PID-derived offset keep parallel runs off the same port),
        launch the three apps under the ``timewarp`` conda env (one process group each,
        for clean teardown), and health-check them.

The servers bind ``127.0.0.1`` only, so the whole flow is intrinsically single-host; the
benchmark threads the resolved URLs through its ``runtime_context`` — re-derived on every
run, so a resumed run always uses the freshly-launched ports — rather than env vars, which
do not reach Ray workers. See ``benchmark.py`` for the orchestration.

Scope note (for whoever needs this next): what follows provisions long-lived *local
processes*, which cube-standard's resource layer deliberately does not model — ``cube.resource``
covers Docker services (``DockerServiceConfig``) and VMs (``VMResourceConfig``), and TimeWarp is
neither. A one-cube exception is the right size for a one-cube need. If a **second** cube ever
needs bare-process provisioning, propose a process-service ``ResourceConfig`` upstream in
cube-standard rather than copying this module — see AGENTS.md, "External contracts".
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from cube.infra_utils import free_port

logger = logging.getLogger(__name__)

#: Upstream TimeWarp project that ships the environment servers and start scripts.
UPSTREAM_REPO = "https://github.com/sparklabutah/timewarp"

#: Pinned commit for reproducibility (PS-001). Locked to upstream tag ``v0.2.0`` (also master
#: HEAD as of 2026-07-29) so the provisioned upstream code + data layout are deterministic;
#: bump deliberately, together with the ``browsergym-timewarp`` pin in pyproject.toml — the
#: servers here and the task data there come from the same upstream release.
#: The resolved HEAD is always logged so a run records exactly what it used.
UPSTREAM_COMMIT = "312ad5287499eef2e4dfbd3614f3e1d2f0776d10"

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

#: Remediation appended to every checkout-reconciliation error — the checkout is a cache, so
#: deleting it is always a safe (if slow) way out.
_CHECKOUT_HINT = (
    "Reset/stash the changes, delete the directory to force a fresh clone, or point TIMEWARP_HOME elsewhere."
)


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


def _run_setup_sh(checkout_dir: Path) -> None:
    """Run upstream ``setup.sh``, tearing down its whole process group if it overruns.

    ``subprocess.run(timeout=…)`` kills only bash. The conda build and the multi-GB HuggingFace
    fetches it spawned would keep writing into the checkout after ``_provision_lock`` releases —
    letting the next caller run a *second* ``setup.sh`` alongside them, which is exactly the
    concurrent corruption the lock exists to prevent. ``start_new_session`` puts the descendants
    in their own group so one ``killpg`` reaches all of them.

    ``BaseException``, not ``Exception``: this call can block for an hour, and a Ctrl-C inside
    that window orphans the same downloads as a timeout does.
    """
    cmd = ["bash", "setup.sh"]
    proc = subprocess.Popen(cmd, cwd=str(checkout_dir), start_new_session=True)
    try:
        returncode = proc.wait(timeout=_SETUP_TIMEOUT_S)
    except BaseException:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        with contextlib.suppress(subprocess.SubprocessError, OSError):
            proc.wait(timeout=10)  # reap, so it does not linger as a zombie
        raise
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


def _nonempty_file(path: Path) -> bool:
    """True iff *path* is a regular file with content — a truncated download has size 0."""
    return path.is_file() and path.stat().st_size > 0


def _nonempty_dir(path: Path) -> bool:
    """True iff *path* is a directory with at least one entry — an interrupted setup can
    leave the data dir created but empty."""
    return path.is_dir() and any(path.iterdir())


def _missing_components(checkout_dir: Path) -> list[str]:
    """Names of the provisioning pieces upstream ``setup.sh`` should have produced but didn't
    (absent or empty). Empty list == fully provisioned."""
    checks = {
        "upstream checkout (setup.sh)": (checkout_dir / "setup.sh").is_file(),
        "wiki index (env/wiki/wiki_index.pkl)": _nonempty_file(checkout_dir / "env" / "wiki" / "wiki_index.pkl"),
        "news index (env/news/news_index.pkl)": _nonempty_file(checkout_dir / "env" / "news" / "news_index.pkl"),
        "webshop data (env/webshop/data)": _nonempty_dir(checkout_dir / "env" / "webshop" / "data"),
        f"conda env '{CONDA_ENV}'": _conda_env_exists(),
    }
    return [name for name, ok in checks.items() if not ok]


def is_provisioned(checkout_dir: Path | None = None) -> bool:
    """True when the repo, the conda env, and the downloaded data/index files are all present
    *and non-empty*.

    Mirrors what upstream ``setup.sh`` produces, so ``ensure_provisioned`` can skip the slow
    path on a machine that has already been set up. The data/index checks look at file size /
    directory contents (not mere existence) so an interrupted ``setup.sh`` — which can leave a
    zero-byte index pickle or an empty data dir behind — is treated as *not* provisioned and
    re-run rather than launching servers against corrupt data.
    """
    checkout_dir = checkout_dir.expanduser() if checkout_dir else default_checkout_dir()
    return not _missing_components(checkout_dir)


def _clone_repo(checkout_dir: Path) -> None:
    """Fresh clone of the upstream repo — callers go through ``_sync_checkout``."""
    checkout_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning %s (%s) into %s …", UPSTREAM_REPO, UPSTREAM_COMMIT, checkout_dir)
    # A specific SHA may not be a branch tip, so a shallow --branch clone can't reach it.
    subprocess.run(["git", "clone", UPSTREAM_REPO, str(checkout_dir)], check=True, timeout=_CLONE_TIMEOUT_S)
    subprocess.run(
        ["git", "checkout", UPSTREAM_COMMIT], cwd=str(checkout_dir), check=True, timeout=_QUICK_CMD_TIMEOUT_S
    )


def _head_commit(checkout_dir: Path) -> str:
    """HEAD SHA of an existing checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(checkout_dir),
            capture_output=True,
            text=True,
            timeout=_QUICK_CMD_TIMEOUT_S,
        )
    except (subprocess.SubprocessError, OSError) as e:
        raise RuntimeError(f"Could not read the git HEAD of {checkout_dir}: {e}. {_CHECKOUT_HINT}") from e
    if result.returncode != 0:
        raise RuntimeError(
            f"{checkout_dir} exists but is not a git checkout ({result.stderr.strip()}). {_CHECKOUT_HINT}"
        )
    return result.stdout.strip()


def _require_clean_worktree(checkout_dir: Path) -> None:
    """Refuse to move a checkout whose *tracked* files are locally modified.

    ``--untracked-files=no`` is load-bearing: the multi-GB index/data files ``setup.sh``
    downloads live untracked inside the worktree. They are not local edits, and ``git
    checkout`` leaves them in place — so they must not block (or be destroyed by) a pin bump.
    """
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=str(checkout_dir),
        capture_output=True,
        text=True,
        timeout=_QUICK_CMD_TIMEOUT_S,
    )
    if result.returncode != 0:
        raise RuntimeError(f"`git status` failed in {checkout_dir}: {result.stderr.strip()}. {_CHECKOUT_HINT}")
    if result.stdout.strip():
        raise RuntimeError(
            f"The TimeWarp checkout at {checkout_dir} has locally modified tracked files:\n"
            f"{result.stdout.strip()}\n{_CHECKOUT_HINT}"
        )


def _sync_checkout(checkout_dir: Path) -> bool:
    """Clone the upstream repo, or advance an existing checkout to ``UPSTREAM_COMMIT``.

    Returns True when HEAD moved (fresh clone or pin advance) so the caller re-runs
    ``setup.sh``: a new pin can need data or conda-env packages the old one didn't.

    Reconciling matters because the pin is a *constant in this file* — without it, a machine
    that cloned once would keep running the old upstream forever while the pin says otherwise.
    """
    changed = False
    if not (checkout_dir / "setup.sh").is_file():
        _clone_repo(checkout_dir)
        changed = True
    elif _head_commit(checkout_dir) != UPSTREAM_COMMIT:
        _require_clean_worktree(checkout_dir)
        logger.info("Advancing TimeWarp checkout %s to pinned commit %s …", checkout_dir, UPSTREAM_COMMIT)
        try:
            # Fetch the SHA explicitly so this works whatever the existing clone looks like —
            # a shallow or single-branch clone can't reach an arbitrary commit otherwise.
            subprocess.run(
                ["git", "fetch", "--depth", "1", "origin", UPSTREAM_COMMIT],
                cwd=str(checkout_dir),
                check=True,
                timeout=_CLONE_TIMEOUT_S,
            )
            subprocess.run(
                ["git", "checkout", UPSTREAM_COMMIT],
                cwd=str(checkout_dir),
                check=True,
                timeout=_QUICK_CMD_TIMEOUT_S,
            )
        except (subprocess.SubprocessError, OSError) as e:
            raise RuntimeError(f"Could not advance {checkout_dir} to {UPSTREAM_COMMIT}: {e}. {_CHECKOUT_HINT}") from e
        changed = True
    _log_resolved_commit(checkout_dir)
    return changed


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


@contextlib.contextmanager
def _provision_lock(checkout_dir: Path) -> Iterator[None]:
    """Serialize L1 across every process sharing *checkout_dir*.

    L2 is already safe for concurrent runs on one host (``free_port`` hands out distinct ports;
    each run logs into its own private directory), but L1 was not: two experiments starting
    together on a fresh machine both see ``is_provisioned() == False`` and both run ``setup.sh``
    into the same tree, racing on the same multi-GB downloads and the same search index. The
    loser leaves a *corrupt* tree — which ``_missing_components`` cannot detect, because every
    piece is present, just malformed.

    Holding the lock across the check *and* the setup makes the second caller wait, then find
    the tree already complete and skip. The lockfile lives beside the checkout, not inside it,
    so it works before the clone exists and can never be mistaken for repo content.

    Two honest limits. The lock does **not** survive holder death: flock is released by the
    kernel, so a process killed mid-``setup.sh`` lets the next caller in to find a tree that
    ``_missing_components`` calls complete but that is actually malformed. And on a shared NFS
    ``TIMEWARP_HOME`` flock is advisory-at-best across hosts — though the servers bind
    ``127.0.0.1``, so this module is single-host by construction anyway.

    One deliberate degradation: a checkout whose parent directory is **not writable** cannot be
    provisioned into by anyone, so there is nothing to race on — a read-only or admin-provisioned
    tree (``TIMEWARP_HOME=/opt/shared/timewarp``, a baked container image) previously took the
    "already provisioned" fast path and must keep working. That case warns and runs unlocked.

    A failure to take the lock in a *writable* directory is a different thing and stays fatal:
    the tree can still be written, so the race the lock exists to prevent is live. The common
    shape is a lockfile left at 0644 by another user on a shared ``TIMEWARP_HOME`` — silently
    continuing there is exactly how two concurrent ``setup.sh`` runs corrupt the tree.
    """
    lock_path = checkout_dir.parent / f"{checkout_dir.name}.provision.lock"
    handle: TextIO | None = None
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("w")
    except OSError as exc:
        if os.access(lock_path.parent, os.W_OK):
            raise
        unlockable = exc
    if handle is None:
        logger.warning(
            "TimeWarp checkout parent %s is not writable (%s) — provisioning unlocked. Safe for a "
            "pre-provisioned or read-only tree, which is the only way to reach this state.",
            lock_path.parent,
            unlockable,
        )
        yield  # outside the except block, so a failure in the body carries no misleading context
        return
    with handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.ENOLCK, errno.EOPNOTSUPP):
                # The filesystem has no locking at all (NFS without lockd, some FUSE/overlay
                # mounts). That is not contention, and retrying blocks forever or raises again.
                logger.warning("%s does not support flock (%s) — provisioning unlocked.", lock_path, exc.strerror)
                yield
                return
            logger.info("Another process is provisioning TimeWarp at %s — waiting for it …", checkout_dir)
            fcntl.flock(handle, fcntl.LOCK_EX)
        yield  # closing the handle on the way out releases the lock


def ensure_provisioned(checkout_dir: Path | None = None, *, force: bool = False) -> Path:
    """L1 — clone the upstream repo and run its idempotent ``setup.sh`` if not already set up.

    Cheap when ``is_provisioned()`` is already True (the common case) — the steady-state cost
    is one local ``git rev-parse``. ``setup.sh`` itself skips re-creating the conda env and
    re-downloading existing index/data files, so a forced re-run only fetches what is
    genuinely missing.

    A pin advance (``_sync_checkout`` moved HEAD) always re-runs ``setup.sh``, even when the
    old tree looks complete: newer upstream can need extra data or newer packages inside the
    existing conda env, and only ``setup.sh`` knows what those are.

    Serialized per checkout dir by ``_provision_lock`` so concurrent runs on one host cannot
    race on the same downloads.
    """
    checkout_dir = checkout_dir.expanduser() if checkout_dir else default_checkout_dir()
    with _provision_lock(checkout_dir):
        updated = _sync_checkout(checkout_dir)
        if not (force or updated) and is_provisioned(checkout_dir):
            logger.info("TimeWarp already provisioned at %s — skipping setup.sh", checkout_dir)
            return checkout_dir
        _conda_bin()  # fail fast with an actionable message before the long setup
        logger.info("Running upstream setup.sh in %s (one-time; downloads conda env + data) …", checkout_dir)
        _run_setup_sh(checkout_dir)
        # Belt and braces: upstream setup.sh is fail-fast as of v0.2.0, but a pre-existing broken
        # conda env or an interrupted run can still leave gaps. Verify the postcondition here so a
        # partial provision fails with an actionable message instead of crashing at server launch.
        missing = _missing_components(checkout_dir)
        if missing:
            raise RuntimeError(
                f"Upstream setup.sh finished but TimeWarp is still not fully provisioned at {checkout_dir} — "
                f"missing: {', '.join(missing)}. Inspect the setup.sh output above for the step that failed."
            )
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


def _server_env(app_python: str) -> dict[str, str]:
    """Process environment for the launched servers.

    The servers run the conda env's interpreter directly (see ``_env_python``), so the env's
    ``activate.d`` exports never run. Replicate the one the servers depend on: openjdk's
    ``JAVA_HOME`` (webshop's pyserini/jnius refuses to start without a JVM, even when openjdk
    is installed in the env). Also prepend the env's ``bin`` so server subprocesses resolve
    tools (``java``, …) from the same env, exactly as an activated shell would.
    """
    env = os.environ.copy()
    prefix = Path(app_python).parent.parent
    jvm = prefix / "lib" / "jvm"
    if jvm.is_dir():
        env["JAVA_HOME"] = str(jvm)  # override, as conda activation does
    env["PATH"] = f"{prefix / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    return env


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
    ``stop()`` kills each server's process group, reaps it and closes its log files; safe to call
    more than once. The log *directory* is deliberately left behind (see ``stop()``) and reclaimed
    by a later launch. ``_stderr_paths`` (site order) lets the healthcheck surface a crashed
    server's stderr instead of just an exit code.
    """

    urls: dict[str, str]
    checkout_dir: Path
    _procs: list[subprocess.Popen] = field(default_factory=list, repr=False)
    _logs: list[TextIO] = field(default_factory=list, repr=False)
    _stderr_paths: list[Path] = field(default_factory=list, repr=False)
    _log_dir: Path | None = field(default=None, repr=False)

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
        # The logs are deliberately NOT deleted here. Deciding at teardown whether a run "went
        # fine" cannot be done reliably: `proc.poll()` reports a killed server as still running
        # for tens to hundreds of milliseconds after the signal, and a Flask app that raises on
        # every request never dies at all — `is_reachable` accepts any HTTP status, so a server
        # 500-ing its way through an entire experiment looks perfectly healthy at stop() time.
        # Both cases would delete the only record of what went wrong. `_prune_log_dirs` bounds
        # the accumulation instead, on the next launch, which needs no such judgement.
        if self._log_dir is not None:
            logger.info("TimeWarp server logs: %s", self._log_dir)
            self._log_dir = None

    def __enter__(self) -> TimeWarpServers:
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def _build_url(site: str, port: int, host: str) -> str:
    base = f"http://{host}:{port}"
    return f"{base}/abc" if site == "webshop" else base  # webshop is mounted under /abc upstream


def _open_logs(log_dir: Path, site: str, port: int) -> tuple[TextIO, TextIO, Path]:
    """Open per-site stdout/stderr log files so a crashed server's output is inspectable
    (returns the open handles plus the stderr path for the healthcheck error message).

    *log_dir* is the caller's private per-run directory (``start_servers`` creates it with
    ``tempfile.mkdtemp``, mode 0700), so these paths are neither shared nor guessable: two
    concurrent auto-mode runs cannot truncate each other's logs, and no other local user can
    pre-create — or symlink — a path we are about to open for writing. That integrity is
    load-bearing, because the healthcheck's ``_tail`` reads these files back to diagnose a
    crash. The port stays in the filename purely so the files are self-describing.
    """
    out_path = log_dir / f"timewarp_{site}_{port}_stdout.log"
    err_path = log_dir / f"timewarp_{site}_{port}_stderr.log"
    out_f = out_path.open("w")
    try:
        err_f = err_path.open("w")
    except OSError:
        out_f.close()  # don't leak the stdout handle if opening stderr fails (disk full / perms)
        raise
    return out_f, err_f, err_path


#: How long a run's server logs survive before a later launch reclaims them. Well beyond any
#: experiment, so a concurrent run's live directory is never a candidate.
_LOG_RETENTION_S = 7 * 24 * 3600

#: Shared prefix so a later run can find (and reclaim) an earlier one's log directory.
_LOG_DIR_PREFIX = "timewarp-logs-"


def _prune_log_dirs(parent: Path, prefix: str) -> None:
    """Reclaim server-log directories from runs that finished over ``_LOG_RETENTION_S`` ago.

    ``stop()`` never deletes logs, because at teardown there is no reliable way to tell a run
    that went fine from one that did not. Pruning on the *next* launch needs no such judgement:
    by then the logs are old enough that nobody is coming back for them, and anything still
    being written to is far too recent to match.
    """
    cutoff = time.time() - _LOG_RETENTION_S
    for stale in parent.glob(f"{prefix}*"):
        try:
            if stale.is_dir() and stale.stat().st_mtime < cutoff:
                shutil.rmtree(stale)
                logger.debug("Reclaimed stale TimeWarp log dir %s", stale)
        except OSError as exc:  # another run's dir, a permissions quirk — never fatal
            logger.debug("Could not reclaim %s: %s", stale, exc)


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
    checkout_dir = checkout_dir.expanduser() if checkout_dir else default_checkout_dir()
    if not (1 <= ui_version <= 6):
        raise ValueError(f"ui_version must be 1-6, got {ui_version}")

    app_python = _env_python()
    server_env = _server_env(app_python)
    sites = list(SITE_ENV_VARS)  # wiki, news, webshop
    ports = [free_port(start=start_port) for _ in sites]
    urls = {site: _build_url(site, port, host) for site, port in zip(sites, ports)}
    _prune_log_dirs(Path(tempfile.gettempdir()), _LOG_DIR_PREFIX)
    log_dir = Path(tempfile.mkdtemp(prefix=_LOG_DIR_PREFIX))  # mkdtemp is 0700 — private to this run
    logger.info("TimeWarp server logs: %s", log_dir)

    servers = TimeWarpServers(urls=urls, checkout_dir=checkout_dir, _log_dir=log_dir)
    try:
        for site, port in zip(sites, ports):
            site_dir = checkout_dir / "env" / site
            cmd = _site_command(site, app_python, port, ui_version)
            logger.info("Starting %s (theme %d) on port %d: %s", site, ui_version, port, " ".join(cmd))
            out_f, err_f, err_path = _open_logs(log_dir, site, port)
            servers._logs += [out_f, err_f]
            servers._stderr_paths.append(err_path)
            servers._procs.append(
                subprocess.Popen(
                    cmd, cwd=str(site_dir), env=server_env, stdout=out_f, stderr=err_f, start_new_session=True
                )
            )
        _wait_until_healthy(servers, timeout_s=healthcheck_timeout_s)
    except BaseException:
        # BaseException, not Exception: the healthcheck can block for up to three minutes, and a
        # Ctrl-C or a SIGTERM-turned-SystemExit inside that window would otherwise skip stop()
        # entirely. `servers` is a local and `benchmark._servers` has not been assigned yet, so
        # nothing downstream could ever reach these processes — they would hold their ports for
        # the life of the machine.
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
