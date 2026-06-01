"""artoo/local.py tests — safe-root enforcement is load-bearing here.

dev/build are tested via mock (those modules pull in OpenRouter +
runtime state and we don't want network in CI). read/write/run are
exercised end-to-end against a tmp safe-root.
"""
import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from artoo import local as local_tool


@pytest.fixture
def safe_root(tmp_path: Path, monkeypatch):
    """Redirect SAFE_ROOT to a tmpdir per test."""
    monkeypatch.setenv("ARTOO_LOCAL_SAFE_ROOT", str(tmp_path))
    return tmp_path


def _call(op: str, **kwargs):
    """Sync wrapper around the async dispatch — fresh loop per call so
    we don't fight pytest-asyncio's autouse-style event-loop scoping."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(local_tool.call(op, **kwargs))
    finally:
        loop.close()


# ── unknown op ─────────────────────────────────────────────────────

def test_unknown_op_returns_error(safe_root):
    out = _call("teleport")
    assert out["ok"] is False
    assert "unknown op" in out["error"]


# ── write + read round-trip ────────────────────────────────────────

def test_write_then_read_round_trip(safe_root):
    w = _call("write", path="hello.txt", content="hi there")
    assert w["ok"] is True
    assert w["bytes_written"] == 8

    r = _call("read", path="hello.txt")
    assert r["ok"] is True
    assert r["text"] == "hi there"
    assert r["bytes"] == 8


def test_write_creates_intermediate_dirs(safe_root):
    out = _call("write", path="deep/nested/dir/file.txt", content="x")
    assert out["ok"] is True
    assert (safe_root / "deep/nested/dir/file.txt").is_file()


def test_read_missing_file_returns_error(safe_root):
    out = _call("read", path="does/not/exist.txt")
    assert out["ok"] is False
    assert "not a file" in out["error"]


# ── path traversal ─────────────────────────────────────────────────

def test_read_rejects_dot_dot_traversal(safe_root):
    out = _call("read", path="../../../etc/passwd")
    assert out["ok"] is False
    assert "outside safe root" in out["error"]


def test_write_rejects_absolute_path_outside_root(safe_root):
    out = _call("write", path="/etc/evil.conf", content="x")
    assert out["ok"] is False
    assert "outside safe root" in out["error"]


def test_read_rejects_symlink_escape(safe_root, tmp_path):
    """A symlink under SAFE_ROOT pointing outside must be refused on
    read — resolve() follows it and the post-resolve check catches it."""
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")
    (safe_root / "escape.txt").symlink_to(outside)
    out = _call("read", path="escape.txt")
    assert out["ok"] is False
    assert "outside safe root" in out["error"]


# ── write: required args + size cap ────────────────────────────────

def test_write_requires_path(safe_root):
    out = _call("write", content="x")
    assert out["ok"] is False
    assert "path is required" in out["error"]


def test_write_requires_content(safe_root):
    out = _call("write", path="x.txt", content=None)
    assert out["ok"] is False
    assert "content is required" in out["error"]


def test_write_rejects_oversize_content(safe_root):
    huge = "x" * (local_tool.MAX_WRITE_BYTES + 1)
    out = _call("write", path="big.txt", content=huge)
    assert out["ok"] is False
    assert "exceeds" in out["error"]


# ── run / shell ────────────────────────────────────────────────────

def test_run_captures_stdout_and_exit_code(safe_root):
    out = _call("run", cmd="echo hello")
    assert out["ok"] is True
    assert "hello" in out["stdout"]
    assert out["exit_code"] == 0


def test_run_captures_stderr_and_nonzero(safe_root):
    out = _call("run", cmd="echo oops >&2 ; exit 3")
    assert out["ok"] is False  # non-zero exit
    assert "oops" in out["stderr"]
    assert out["exit_code"] == 3


def test_run_times_out(safe_root):
    out = _call("run", cmd="sleep 10", timeout_s=1)
    assert out["ok"] is False
    assert "timeout" in out["error"]


def test_run_rejects_cwd_outside_safe_root(safe_root):
    out = _call("run", cmd="pwd", cwd="../outside")
    assert out["ok"] is False
    assert "outside safe root" in out["error"]


def test_run_accepts_cwd_inside_safe_root(safe_root):
    (safe_root / "sub").mkdir()
    out = _call("run", cmd="pwd", cwd="sub")
    assert out["ok"] is True
    assert "sub" in out["stdout"]


def test_shell_alias_dispatches_to_run(safe_root):
    out = _call("shell", cmd="echo via-shell")
    assert out["ok"] is True
    assert "via-shell" in out["stdout"]


# ── dev ────────────────────────────────────────────────────────────

def test_dev_requires_task(safe_root):
    out = _call("dev")
    assert out["ok"] is False
    assert "task is required" in out["error"]


def test_dev_dispatches_to_pipeline_run(safe_root, monkeypatch):
    """dev wraps dev_pipeline.run — verify call shape + return mapping
    without hitting OpenRouter."""
    from artoo import dev_pipeline

    class FakeResult:
        ok = True
        task = "build x"
        approved = True
        rounds = 2
        cost_usd = 0.05
        code = "print('hi')"
        error = None

    called = {}

    def fake_run(*, task, max_rounds, context, security_review):
        called["task"] = task
        called["max_rounds"] = max_rounds
        called["context"] = context
        called["security_review"] = security_review
        return FakeResult()

    monkeypatch.setattr(dev_pipeline, "run", fake_run)
    out = _call("dev", task="build x", context="some ctx", max_rounds=2, security_review=True)
    assert out["ok"] is True
    assert out["approved"] is True
    assert out["code"] == "print('hi')"
    assert called == {"task": "build x", "max_rounds": 2, "context": "some ctx", "security_review": True}


# ── build ──────────────────────────────────────────────────────────

def test_build_requires_project_dir(safe_root):
    out = _call("build")
    assert out["ok"] is False
    assert "project_dir is required" in out["error"]


def test_build_requires_goal_for_new_project(tmp_path: Path):
    fresh = tmp_path / "fresh-project"
    out = _call("build", project_dir=str(fresh))
    assert out["ok"] is False
    assert "goal is required" in out["error"]


# ── kwarg validation ───────────────────────────────────────────────

def test_bad_kwargs_surface_cleanly(safe_root):
    out = _call("read", path="x.txt", unexpected_kwarg=True)
    assert out["ok"] is False
    assert "bad args" in out["error"]


# ── restart ────────────────────────────────────────────────────────

def test_restart_test_first_skips_pytest_when_disabled(safe_root, monkeypatch):
    """test_first=False MUST NOT shell out to pytest. the operator only sets
    this when he's already verified the change manually."""
    spawned = []

    async def fake_subprocess_shell(cmd, **kwargs):
        spawned.append(cmd)
        class _P:
            returncode = 0
            async def communicate(self):
                return (b"", b"")
            async def wait(self):
                return 0
            def kill(self):
                return None
        return _P()

    monkeypatch.setattr(local_tool.asyncio, "create_subprocess_shell", fake_subprocess_shell)
    out = _call("restart", reason="manual", test_first=False)
    assert out["ok"] is True
    assert out["tests_passed"] is None  # didn't run
    # Only one subprocess: the systemctl restart scheduler. No pytest.
    assert any("systemctl --user restart artoo.service" in c for c in spawned)
    assert not any("pytest" in c for c in spawned)


def test_restart_runs_pytest_when_test_first_true(safe_root, monkeypatch):
    spawned = []

    async def fake_subprocess_shell(cmd, **kwargs):
        spawned.append(cmd)
        class _P:
            returncode = 0
            async def communicate(self):
                return (b"all tests passed\n", b"")
            async def wait(self):
                return 0
            def kill(self):
                return None
        return _P()

    monkeypatch.setattr(local_tool.asyncio, "create_subprocess_shell", fake_subprocess_shell)
    out = _call("restart", reason="default", test_first=True)
    assert out["ok"] is True
    assert out["tests_passed"] is True
    # Both pytest AND the restart scheduler ran.
    assert any("pytest" in c for c in spawned)
    assert any("systemctl --user restart" in c for c in spawned)


def test_restart_refuses_on_failing_tests(safe_root, monkeypatch):
    """Restarting onto broken code is how you crash-loop the service.
    The default test_first=True must hard-stop on red."""
    spawned = []

    async def fake_subprocess_shell(cmd, **kwargs):
        spawned.append(cmd)
        # First call (pytest) fails; second (systemctl) should never happen.
        exit_code = 1 if "pytest" in cmd else 0
        class _P:
            returncode = exit_code
            async def communicate(self):
                if "pytest" in cmd:
                    return (b"FAILED tests/x.py::test_foo\n", b"AssertionError\n")
                return (b"", b"")
            async def wait(self):
                return 0
            def kill(self):
                return None
        return _P()

    monkeypatch.setattr(local_tool.asyncio, "create_subprocess_shell", fake_subprocess_shell)
    out = _call("restart", reason="should refuse")
    assert out["ok"] is False
    assert out["tests_passed"] is False
    assert "tests failed" in out["error"]
    # Critical: NO systemctl restart was spawned.
    assert not any("systemctl --user restart" in c for c in spawned)
    # The boss gets a tail of the test output so it can tell the operator
    # what broke without re-running.
    assert "FAILED" in out["tests"]["stdout_tail"]


def test_restart_clamps_delay_to_max(safe_root, monkeypatch):
    """A boss with a confused max_tokens can't ask for a 5-minute delay
    to ship a 'final message' — clamp to 30s."""
    async def fake_subprocess_shell(cmd, **kwargs):
        class _P:
            returncode = 0
            async def communicate(self):
                return (b"", b"")
            async def wait(self):
                return 0
            def kill(self):
                return None
        return _P()

    monkeypatch.setattr(local_tool.asyncio, "create_subprocess_shell", fake_subprocess_shell)
    out = _call("restart", reason="x", test_first=False, delay_s=600)
    assert out["delay_s"] == local_tool._RESTART_DELAY_MAX
