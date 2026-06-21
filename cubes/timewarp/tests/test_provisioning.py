"""Unit tests for timewarp_cube.provisioning and the auto/manual benchmark wiring.

Server-free, conda-free, network-free: subprocess / Popen / urlopen / free_port are all
mocked. Anything that actually clones, runs setup.sh, or launches servers belongs in the
debug suite (``python -m timewarp_cube.debug``) or the smoke script, not here.
"""

from __future__ import annotations

import io
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


def _fake_logs(site: str) -> tuple[io.StringIO, io.StringIO, Path]:
    """Stand-in for _open_logs that touches no filesystem (Popen is mocked, so handles are unused)."""
    return io.StringIO(), io.StringIO(), Path(f"/tmp/{site}.err")


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


# ── install() short-circuits (runs before every cube test / debug run) ─────────


def test_install_skips_provisioning_when_servers_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Manual-mode debug suite: servers already up → install() must not clone/setup."""
    for site, var in provisioning.SITE_ENV_VARS.items():
        monkeypatch.setenv(var, _URLS[site])
    monkeypatch.setattr(provisioning, "is_reachable", lambda url, timeout=5.0: True)

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("install() must not provision when servers are reachable")

    monkeypatch.setattr(provisioning, "ensure_provisioned", _boom)
    TimeWarpBenchmarkConfig.install()  # no raise


def test_install_skips_provisioning_when_no_conda(monkeypatch: pytest.MonkeyPatch) -> None:
    """No conda → auto mode can't run anyway; install() skips instead of cloning + crashing."""
    for var in provisioning.SITE_ENV_VARS.values():
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(provisioning, "has_conda", lambda: False)

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("install() must not provision when conda is absent")

    monkeypatch.setattr(provisioning, "ensure_provisioned", _boom)
    TimeWarpBenchmarkConfig.install()  # no raise


def test_install_provisions_when_conda_and_no_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in provisioning.SITE_ENV_VARS.values():
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(provisioning, "has_conda", lambda: True)
    called: list[bool] = []
    monkeypatch.setattr(provisioning, "ensure_provisioned", lambda: called.append(True))
    TimeWarpBenchmarkConfig.install()
    assert called == [True]


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


def test_get_task_configs_yields_timewarp_configs() -> None:
    """The benchmark no longer overrides get_task_configs — the base yields TimeWarpTaskConfig."""
    configs = list(_config().named_subset("wiki").get_task_configs())
    assert configs and all(isinstance(tc, TimeWarpTaskConfig) for tc in configs)
