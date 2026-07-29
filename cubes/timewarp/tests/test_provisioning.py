"""Unit tests for timewarp_cube.provisioning and the auto/manual benchmark wiring.

Server-free, conda-free, network-free: subprocess / Popen / urlopen / free_port are all
mocked. Anything that actually clones, runs setup.sh, or launches servers belongs in the
debug suite (``python -m timewarp_cube.debug``) or the smoke script, not here.
"""

from __future__ import annotations

import io
import subprocess
import urllib.error
from pathlib import Path

import pytest

from timewarp_cube import TimeWarpBenchmarkConfig, TimeWarpTaskConfig, provisioning
from timewarp_cube.configs import _browser_with_chat

# User-supplied / ambient URLs (host-agnostic): used for env-var and runtime_context tests.
_URLS = {
    "wiki": "http://localhost:5000",
    "news": "http://localhost:5001",
    "webshop": "http://localhost:5002/abc",
}
# What start_servers builds: bound to 127.0.0.1 so probe/URL/healthcheck agree on one host.
_LOCAL_URLS = {
    "wiki": "http://127.0.0.1:5000",
    "news": "http://127.0.0.1:5001",
    "webshop": "http://127.0.0.1:5002/abc",
}


# ── checkout dir / env helpers ────────────────────────────────────────────────


def test_default_checkout_dir_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIMEWARP_HOME", "/custom/tw")
    assert provisioning.default_checkout_dir() == Path("/custom/tw")
    monkeypatch.delenv("TIMEWARP_HOME", raising=False)
    assert provisioning.default_checkout_dir() == Path.home() / ".cache" / "timewarp"


def test_urls_from_env_all_or_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in provisioning.SITE_ENV_VARS.values():
        monkeypatch.delenv(var, raising=False)
    assert provisioning.urls_from_env() is None
    for site, var in provisioning.SITE_ENV_VARS.items():
        monkeypatch.setenv(var, _URLS[site])
    assert provisioning.urls_from_env() == _URLS


def test_apply_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # setenv (not delenv) so monkeypatch records the keys and restores them on teardown —
    # apply_to_env writes os.environ directly, so without this the values would leak.
    for var in provisioning.SITE_ENV_VARS.values():
        monkeypatch.setenv(var, "placeholder")
    provisioning.apply_to_env(_URLS)
    assert provisioning.urls_from_env() == _URLS


# ── reachability ──────────────────────────────────────────────────────────────


def test_is_reachable_treats_http_error_as_up(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_http(*_a: object, **_k: object) -> None:
        raise urllib.error.HTTPError("http://x", 404, "Not Found", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(provisioning.urllib.request, "urlopen", _raise_http)
    assert provisioning.is_reachable("http://x") is True


def test_is_reachable_false_on_conn_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_url(*_a: object, **_k: object) -> None:
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(provisioning.urllib.request, "urlopen", _raise_url)
    assert provisioning.is_reachable("http://x") is False


# ── is_provisioned (completeness, not mere existence) ──────────────────────────


def _make_provisioned_tree(root: Path) -> None:
    (root / "setup.sh").write_text("#!/bin/bash\n")
    (root / "env" / "wiki").mkdir(parents=True)
    (root / "env" / "news").mkdir(parents=True)
    (root / "env" / "webshop" / "data").mkdir(parents=True)
    (root / "env" / "wiki" / "wiki_index.pkl").write_text("x")
    (root / "env" / "news" / "news_index.pkl").write_text("x")
    (root / "env" / "webshop" / "data" / "items.json").write_text("[]")  # non-empty data dir


def test_is_provisioned_true_when_complete(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _make_provisioned_tree(tmp_path)
    monkeypatch.setattr(provisioning, "_conda_env_exists", lambda env=provisioning.CONDA_ENV: True)
    assert provisioning.is_provisioned(tmp_path) is True


def test_is_provisioned_false_when_index_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _make_provisioned_tree(tmp_path)
    (tmp_path / "env" / "wiki" / "wiki_index.pkl").unlink()
    monkeypatch.setattr(provisioning, "_conda_env_exists", lambda env=provisioning.CONDA_ENV: True)
    assert provisioning.is_provisioned(tmp_path) is False


def test_is_provisioned_false_when_index_truncated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An interrupted setup.sh can leave a zero-byte index pickle — treat it as not provisioned."""
    _make_provisioned_tree(tmp_path)
    (tmp_path / "env" / "wiki" / "wiki_index.pkl").write_text("")  # truncated
    monkeypatch.setattr(provisioning, "_conda_env_exists", lambda env=provisioning.CONDA_ENV: True)
    assert provisioning.is_provisioned(tmp_path) is False


def test_is_provisioned_false_when_data_dir_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _make_provisioned_tree(tmp_path)
    (tmp_path / "env" / "webshop" / "data" / "items.json").unlink()  # dir exists but empty
    monkeypatch.setattr(provisioning, "_conda_env_exists", lambda env=provisioning.CONDA_ENV: True)
    assert provisioning.is_provisioned(tmp_path) is False


def test_is_provisioned_false_when_conda_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _make_provisioned_tree(tmp_path)
    monkeypatch.setattr(provisioning, "_conda_env_exists", lambda env=provisioning.CONDA_ENV: False)
    assert provisioning.is_provisioned(tmp_path) is False


# ── checkout reconciliation (_sync_checkout) ──────────────────────────────────


class _FakeGit:
    """Stands in for subprocess.run over `git`, recording the argv of every call.

    ``head`` is what `git rev-parse HEAD` reports; ``status`` is `git status --porcelain`
    output ("" == clean). Any other git subcommand succeeds silently.
    """

    def __init__(self, head: str = "old" * 13 + "0", status: str = "") -> None:
        self.head = head
        self.status = status
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], **_kw: object) -> subprocess.CompletedProcess[str]:
        self.calls.append(cmd)
        stdout = ""
        if cmd[:2] == ["git", "rev-parse"]:
            stdout = self.head
        elif cmd[:2] == ["git", "status"]:
            stdout = self.status
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=stdout, stderr="")

    def ran(self, *prefix: str) -> bool:
        return any(cmd[: len(prefix)] == list(prefix) for cmd in self.calls)


def test_sync_checkout_advances_stale_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The pin is a constant in this module — an existing checkout must be moved to it, or the
    machine silently keeps running the upstream it first cloned."""
    _make_provisioned_tree(tmp_path)
    git = _FakeGit(head="stale-sha")
    monkeypatch.setattr(provisioning.subprocess, "run", git)

    assert provisioning._sync_checkout(tmp_path) is True
    assert git.ran("git", "fetch", "--depth", "1", "origin", provisioning.UPSTREAM_COMMIT)
    assert git.ran("git", "checkout", provisioning.UPSTREAM_COMMIT)


def test_sync_checkout_noop_when_already_pinned(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _make_provisioned_tree(tmp_path)
    git = _FakeGit(head=provisioning.UPSTREAM_COMMIT or "")
    monkeypatch.setattr(provisioning.subprocess, "run", git)

    assert provisioning._sync_checkout(tmp_path) is False
    assert not git.ran("git", "fetch")
    assert not git.ran("git", "checkout")


def test_sync_checkout_refuses_dirty_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Locally modified *tracked* files would be clobbered by the checkout — refuse instead."""
    _make_provisioned_tree(tmp_path)
    git = _FakeGit(head="stale-sha", status=" M setup.sh")
    monkeypatch.setattr(provisioning.subprocess, "run", git)

    with pytest.raises(RuntimeError, match="locally modified tracked files"):
        provisioning._sync_checkout(tmp_path)
    assert not git.ran("git", "fetch")


def test_sync_checkout_ignores_untracked_downloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The multi-GB index/data files setup.sh downloads are untracked and must not read as dirt —
    the status probe has to pass --untracked-files=no."""
    _make_provisioned_tree(tmp_path)
    git = _FakeGit(head="stale-sha")
    monkeypatch.setattr(provisioning.subprocess, "run", git)

    assert provisioning._sync_checkout(tmp_path) is True
    assert git.ran("git", "status", "--porcelain", "--untracked-files=no")


def test_sync_checkout_clones_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cloned: list[Path] = []
    monkeypatch.setattr(provisioning, "_clone_repo", lambda d: cloned.append(d))
    monkeypatch.setattr(provisioning.subprocess, "run", _FakeGit())

    assert provisioning._sync_checkout(tmp_path) is True  # tmp_path has no setup.sh
    assert cloned == [tmp_path]


# ── ensure_provisioned postcondition ──────────────────────────────────────────


def test_ensure_provisioned_skips_setup_when_complete(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _make_provisioned_tree(tmp_path)
    monkeypatch.setattr(provisioning, "_conda_env_exists", lambda env=provisioning.CONDA_ENV: True)
    monkeypatch.setattr(provisioning, "_sync_checkout", lambda d: False)

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("setup.sh must not run when already provisioned")

    monkeypatch.setattr(provisioning.subprocess, "run", _boom)
    assert provisioning.ensure_provisioned(tmp_path) == tmp_path


def test_ensure_provisioned_reruns_setup_after_pin_advance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A pin bump can need data/packages the old tree lacks, so a moved HEAD re-runs setup.sh
    even though the tree still looks fully provisioned."""
    _make_provisioned_tree(tmp_path)
    monkeypatch.setattr(provisioning, "_conda_env_exists", lambda env=provisioning.CONDA_ENV: True)
    monkeypatch.setattr(provisioning, "_conda_bin", lambda: "/usr/bin/conda")
    monkeypatch.setattr(provisioning, "_sync_checkout", lambda d: True)
    ran: list[list[str]] = []
    monkeypatch.setattr(provisioning.subprocess, "run", lambda cmd, **k: ran.append(cmd))

    assert provisioning.ensure_provisioned(tmp_path) == tmp_path
    assert ran == [["bash", "setup.sh"]]


def test_ensure_provisioned_raises_when_setup_leaves_gaps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A setup.sh run that leaves the tree incomplete must raise with the missing pieces named,
    not return normally and crash later at server launch."""
    _make_provisioned_tree(tmp_path)
    (tmp_path / "env" / "webshop" / "data" / "items.json").unlink()  # data dir left empty
    monkeypatch.setattr(provisioning, "_conda_env_exists", lambda env=provisioning.CONDA_ENV: True)
    monkeypatch.setattr(provisioning, "_conda_bin", lambda: "/usr/bin/conda")
    monkeypatch.setattr(provisioning, "_sync_checkout", lambda d: False)
    ran: list[object] = []
    monkeypatch.setattr(provisioning.subprocess, "run", lambda *a, **k: ran.append(a))
    with pytest.raises(RuntimeError, match="webshop data"):
        provisioning.ensure_provisioned(tmp_path)
    assert ran  # setup.sh was attempted before the postcondition check


# ── commands / urls ─────────────────────────────────────────────────────────


def test_site_command_matches_upstream() -> None:
    assert provisioning._site_command("wiki", "/py", 5000, 2) == ["/py", "wiki_app.py", "-2", "--port=5000"]
    assert provisioning._site_command("news", "/py", 5001, 2) == ["/py", "news_app.py", "-2", "--port=5001"]
    assert provisioning._site_command("webshop", "/py", 5002, 2) == [
        "/py",
        "-m",
        "web_agent_site.app",
        "2",
        "--port=5002",
        "--log",
        "--attrs",
    ]


def test_build_url_webshop_has_abc_suffix() -> None:
    assert provisioning._build_url("wiki", 5000, "127.0.0.1") == "http://127.0.0.1:5000"
    assert provisioning._build_url("webshop", 5002, "127.0.0.1") == "http://127.0.0.1:5002/abc"


# ── log files (_open_logs) ─────────────────────────────────────────────────────


def test_open_logs_paths_are_port_qualified(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Two concurrent auto-mode runs share a site name but get distinct ports — the log paths
    must carry the port so they don't collide and truncate each other's logs (Fix 1)."""
    monkeypatch.setattr(provisioning.tempfile, "gettempdir", lambda: str(tmp_path))
    out_a, err_a, err_path_a = provisioning._open_logs("wiki", 5000)
    out_b, err_b, err_path_b = provisioning._open_logs("wiki", 6000)
    try:
        assert err_path_a != err_path_b  # same site, different port → distinct paths
        assert "5000" in err_path_a.name and "6000" in err_path_b.name
    finally:
        for handle in (out_a, err_a, out_b, err_b):
            handle.close()


class _TrackingHandle:
    """Minimal file-handle stand-in that records whether it was closed."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_open_logs_closes_stdout_when_stderr_open_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """If opening the stderr log raises (disk full / perms), the already-open stdout handle
    must be closed rather than leaked, and the error re-raised (Fix 2)."""
    handles: list[_TrackingHandle] = []

    def _fake_open(self: Path, *_a: object, **_k: object) -> _TrackingHandle:
        if not handles:  # first call (stdout) succeeds
            handle = _TrackingHandle()
            handles.append(handle)
            return handle
        raise OSError("disk full")  # second call (stderr) fails

    monkeypatch.setattr(provisioning.Path, "open", _fake_open)
    with pytest.raises(OSError, match="disk full"):
        provisioning._open_logs("wiki", 5000)
    assert len(handles) == 1 and handles[0].closed  # stdout handle opened then closed, not leaked


# ── start_servers / stop (free_port + Popen + healthcheck mocked) ──────────────


class _FakeProc:
    def __init__(self, cmd: list[str], cwd: str | None = None, **_kw: object) -> None:
        self.cmd = cmd
        self.cwd = cwd
        self.pid = 4242
        self.returncode = 0

    def poll(self) -> int | None:
        return None  # always "running"

    def wait(self, timeout: float | None = None) -> int:
        return 0


def _fake_logs(site: str, port: int) -> tuple[io.StringIO, io.StringIO, Path]:
    """Stand-in for _open_logs that touches no filesystem (Popen is mocked, so handles are unused)."""
    return io.StringIO(), io.StringIO(), Path(f"/tmp/{site}_{port}.err")


def _seq_free_port(start: int = 5000, count: int = 1000) -> object:
    """Return distinct ports 5000, 5001, 5002 on successive calls (free_port is process-stateful)."""
    seq = iter([5000, 5001, 5002])

    def _next(start: int = 5000, count: int = 1000) -> int:
        return next(seq)

    return _next


def test_start_servers_builds_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provisioning, "_env_python", lambda env=provisioning.CONDA_ENV: "/py")
    monkeypatch.setattr(provisioning, "free_port", _seq_free_port())
    monkeypatch.setattr(provisioning, "_wait_until_healthy", lambda *a, **k: None)
    monkeypatch.setattr(provisioning.subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(provisioning, "_open_logs", _fake_logs)

    servers = provisioning.start_servers(Path("/repo"), ui_version=2)
    assert servers.urls == _LOCAL_URLS
    assert [p.cmd for p in servers._procs] == [  # type: ignore[attr-defined]
        ["/py", "wiki_app.py", "-2", "--port=5000"],
        ["/py", "news_app.py", "-2", "--port=5001"],
        ["/py", "-m", "web_agent_site.app", "2", "--port=5002", "--log", "--attrs"],
    ]
    assert servers._procs[0].cwd.endswith("env/wiki")  # type: ignore[attr-defined,union-attr]


def test_server_env_threads_conda_java_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Servers are launched via the env's python directly, skipping conda activation — the
    launch env must replicate openjdk's JAVA_HOME export (webshop's jnius needs a JVM) and
    put the env's bin first on PATH."""
    prefix = tmp_path / "envs" / "timewarp"
    (prefix / "lib" / "jvm").mkdir(parents=True)
    (prefix / "bin").mkdir(parents=True)
    monkeypatch.setenv("JAVA_HOME", "/somewhere/else")  # conda activation overrides — so do we
    env = provisioning._server_env(str(prefix / "bin" / "python"))
    assert env["JAVA_HOME"] == str(prefix / "lib" / "jvm")
    assert env["PATH"].startswith(str(prefix / "bin"))


def test_server_env_without_jvm_leaves_java_home_alone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("JAVA_HOME", raising=False)
    (tmp_path / "bin").mkdir(parents=True)
    env = provisioning._server_env(str(tmp_path / "bin" / "python"))
    assert "JAVA_HOME" not in env  # no env-provided JVM → nothing to point at
    assert env["PATH"].startswith(str(tmp_path / "bin"))


def test_start_servers_passes_server_env_to_popen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prefix = tmp_path / "envs" / "timewarp"
    (prefix / "lib" / "jvm").mkdir(parents=True)
    (prefix / "bin").mkdir(parents=True)
    popen_envs: list[dict[str, str] | None] = []

    class _RecordingProc(_FakeProc):
        def __init__(
            self, cmd: list[str], cwd: str | None = None, env: dict[str, str] | None = None, **kw: object
        ) -> None:
            super().__init__(cmd, cwd=cwd, **kw)
            popen_envs.append(env)

    monkeypatch.setattr(provisioning, "_env_python", lambda env=provisioning.CONDA_ENV: str(prefix / "bin" / "python"))
    monkeypatch.setattr(provisioning, "free_port", _seq_free_port())
    monkeypatch.setattr(provisioning, "_wait_until_healthy", lambda *a, **k: None)
    monkeypatch.setattr(provisioning.subprocess, "Popen", _RecordingProc)
    monkeypatch.setattr(provisioning, "_open_logs", _fake_logs)

    provisioning.start_servers(Path("/repo"), ui_version=1)
    assert len(popen_envs) == 3
    assert all(e is not None and e["JAVA_HOME"] == str(prefix / "lib" / "jvm") for e in popen_envs)


def test_start_servers_rejects_bad_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provisioning, "_env_python", lambda env=provisioning.CONDA_ENV: "/py")
    with pytest.raises(ValueError, match="ui_version"):
        provisioning.start_servers(Path("/repo"), ui_version=9)


def test_stop_kills_process_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[int] = []
    monkeypatch.setattr(provisioning.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(provisioning.os, "killpg", lambda pgid, sig: killed.append(pgid))
    servers = provisioning.TimeWarpServers(
        urls=_LOCAL_URLS, checkout_dir=Path("/repo"), _procs=[_FakeProc(["a"]), _FakeProc(["b"])]
    )
    servers.stop()
    assert killed == [4242, 4242]
    assert servers._procs == []


class _StubbornProc:
    """Ignores SIGTERM (wait() times out once) then is reaped after SIGKILL."""

    def __init__(self) -> None:
        self.pid = 5555
        self.returncode = -9
        self._waits = 0

    def poll(self) -> int | None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        self._waits += 1
        if self._waits == 1:
            raise provisioning.subprocess.TimeoutExpired(cmd="x", timeout=timeout or 0)
        return -9  # reaped after SIGKILL


def test_stop_escalates_to_sigkill_and_closes_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    signals: list[int] = []
    monkeypatch.setattr(provisioning.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(provisioning.os, "killpg", lambda pgid, sig: signals.append(sig))
    proc = _StubbornProc()
    log = io.StringIO()
    servers = provisioning.TimeWarpServers(
        urls={"wiki": "http://127.0.0.1:5000"},
        checkout_dir=Path("/repo"),
        _procs=[proc],  # type: ignore[list-item]
        _logs=[log],
    )
    servers.stop()
    assert provisioning.signal.SIGTERM in signals and provisioning.signal.SIGKILL in signals
    assert proc._waits == 2  # waited again after SIGKILL to reap the zombie
    assert log.closed and servers._logs == []


# ── benchmark-level wiring ────────────────────────────────────────────────────


def _config(**kw: object) -> TimeWarpBenchmarkConfig:
    return TimeWarpBenchmarkConfig(tool_config=_browser_with_chat(), **kw)  # type: ignore[arg-type]


def test_provision_mode_defaults_to_auto() -> None:
    assert _config().provision_mode == "auto"
    assert _config().ui_version == 1


def test_task_config_make_reads_tw_urls_from_runtime_context() -> None:
    """Auto mode publishes URLs via runtime_context; TaskConfig.make threads them onto the task."""
    cfg = _config().named_subset("wiki")
    task_cfg = next(iter(cfg.get_task_configs()))
    task = task_cfg.make(runtime_context={"tw_urls": _LOCAL_URLS})
    assert task.tw_urls == _LOCAL_URLS


def test_task_config_make_no_urls_without_runtime_context() -> None:
    cfg = _config(provision_mode="manual").named_subset("wiki")
    task_cfg = next(iter(cfg.get_task_configs()))
    assert task_cfg.make(runtime_context=None).tw_urls is None  # manual mode → ambient env vars


def test_setup_manual_raises_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in provisioning.SITE_ENV_VARS.values():
        monkeypatch.delenv(var, raising=False)
    bench = TimeWarpBenchmarkConfig.benchmark_class(_config(provision_mode="manual"))
    with pytest.raises(RuntimeError, match="Missing TimeWarp environment"):
        bench._setup()


def test_setup_auto_reuses_running_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    """If TW_* already point at reachable servers, auto mode reuses them — no launch."""
    for site, var in provisioning.SITE_ENV_VARS.items():
        monkeypatch.setenv(var, _URLS[site])
    monkeypatch.setattr(provisioning, "is_reachable", lambda url, timeout=5.0: True)

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("should not provision/launch when servers already run")

    monkeypatch.setattr(provisioning, "ensure_provisioned", _boom)
    monkeypatch.setattr(provisioning, "start_servers", _boom)

    bench = TimeWarpBenchmarkConfig.benchmark_class(_config())
    bench._setup()
    assert bench._runtime_context["tw_urls"] == _URLS
    assert bench._servers is None


def test_setup_auto_launches_when_no_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    """No TW_* set → auto mode provisions + launches, publishes URLs into runtime_context + env."""
    for var in provisioning.SITE_ENV_VARS.values():
        monkeypatch.delenv(var, raising=False)
    provisioned: list[Path | None] = []
    monkeypatch.setattr(provisioning, "ensure_provisioned", lambda checkout=None: provisioned.append(checkout))

    fake_servers = provisioning.TimeWarpServers(urls=_LOCAL_URLS, checkout_dir=Path("/repo"))
    monkeypatch.setattr(provisioning, "start_servers", lambda checkout, ui_version: fake_servers)

    bench = TimeWarpBenchmarkConfig.benchmark_class(_config())
    bench._setup()
    assert provisioned == [None]  # ensure_provisioned was called (checkout_dir defaults to None)
    assert bench._servers is fake_servers
    assert bench._runtime_context["tw_urls"] == _LOCAL_URLS
    assert provisioning.urls_from_env() == _LOCAL_URLS  # also exported to driver env
    for var in provisioning.SITE_ENV_VARS.values():
        monkeypatch.delenv(var, raising=False)  # apply_to_env wrote os.environ directly — clean up


def test_setup_auto_raises_on_partial_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto mode refuses a half-configured environment instead of silently relaunching all three."""
    for var in provisioning.SITE_ENV_VARS.values():
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TW_WIKI", _URLS["wiki"])  # only one of three
    bench = TimeWarpBenchmarkConfig.benchmark_class(_config())
    with pytest.raises(RuntimeError, match="only some TimeWarp server env vars"):
        bench._setup()


def test_setup_auto_warns_and_launches_when_set_but_unreachable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """All TW_* set but a server is down → warn and launch cube-managed servers, not silently reuse."""
    for site, var in provisioning.SITE_ENV_VARS.items():
        monkeypatch.setenv(var, _URLS[site])
    monkeypatch.setattr(provisioning, "is_reachable", lambda url, timeout=5.0: False)
    monkeypatch.setattr(provisioning, "ensure_provisioned", lambda checkout=None: None)
    fake_servers = provisioning.TimeWarpServers(urls=_LOCAL_URLS, checkout_dir=Path("/repo"))
    monkeypatch.setattr(provisioning, "start_servers", lambda checkout, ui_version: fake_servers)

    bench = TimeWarpBenchmarkConfig.benchmark_class(_config())
    with caplog.at_level("WARNING"):
        bench._setup()
    assert any("not all reachable" in r.message for r in caplog.records)
    assert bench._runtime_context["tw_urls"] == _LOCAL_URLS
    for var in provisioning.SITE_ENV_VARS.values():
        monkeypatch.delenv(var, raising=False)


def test_urls_from_env_partial_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in provisioning.SITE_ENV_VARS.values():
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TW_NEWS", _URLS["news"])  # partial → not configured
    assert provisioning.urls_from_env() is None


# ── healthcheck fails fast and surfaces server stderr ──────────────────────────


class _DeadProc:
    """A process that has already exited (with code 0) and never bound a port."""

    returncode = 0

    def poll(self) -> int:
        return 0


def test_wait_until_healthy_raises_on_zero_code_early_exit(tmp_path: Path) -> None:
    err = tmp_path / "wiki.err"
    err.write_text("Traceback ...\nOSError: address already in use\n")
    servers = provisioning.TimeWarpServers(
        urls={"wiki": "http://127.0.0.1:5000"},
        checkout_dir=tmp_path,
        _procs=[_DeadProc()],  # type: ignore[list-item]
        _stderr_paths=[err],
    )
    with pytest.raises(RuntimeError, match="exited early") as exc:
        provisioning._wait_until_healthy(servers, timeout_s=1.0)
    assert "address already in use" in str(exc.value)  # crash stderr is surfaced


def test_wait_until_healthy_times_out_when_never_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Complement to the early-exit test: the process stays alive but never binds a port, so the
    site is never reachable — _wait_until_healthy must give up at the deadline with TimeoutError."""
    monkeypatch.setattr(provisioning, "is_reachable", lambda url, timeout=5.0: False)
    servers = provisioning.TimeWarpServers(
        urls={"wiki": "http://127.0.0.1:5000"},
        checkout_dir=Path("/repo"),
        _procs=[_FakeProc(["wiki"])],  # poll() → None, i.e. still running  # type: ignore[list-item]
        _stderr_paths=[Path("/tmp/wiki.err")],
    )
    with pytest.raises(TimeoutError, match="not healthy"):
        provisioning._wait_until_healthy(servers, timeout_s=0.05, interval_s=0.01)


def test_get_task_configs_yields_timewarp_configs() -> None:
    """The benchmark no longer overrides get_task_configs — the base yields TimeWarpTaskConfig."""
    configs = list(_config().named_subset("wiki").get_task_configs())
    assert configs and all(isinstance(tc, TimeWarpTaskConfig) for tc in configs)
