"""Ground-truth self-inspection.

Artoo's boss model has a confabulation failure mode: claiming it lacks tools it
has, that pytest isn't installed, that committed code is uncommitted, that it
"can't run shell" (2026-05-26 incident — all four happened in one conversation,
despite the system prompt explicitly forbidding the first three). This module
returns LIVE facts about Artoo's own repo, toolchain, and service so the boss
can verify reality before claiming a limitation. Exposed via the `self_check`
tool; cheap enough to call whenever the boss is about to say "I can't ...".
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# The repo root is the parent of the artoo package — /home/youruser/artoo. Derived
# from __file__ so it stays correct regardless of cwd or config.
REPO = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 15) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, cwd=str(cwd) if cwd else None,
            capture_output=True, text=True, timeout=timeout,
        )
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return -1, str(e)


def git_state() -> dict:
    """Current branch, HEAD, and whether the working tree is clean."""
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO)[1]
    head = _run(["git", "log", "-1", "--pretty=%h %s"], cwd=REPO)[1]
    _, status = _run(["git", "status", "--porcelain"], cwd=REPO)
    dirty = [ln for ln in status.splitlines() if ln.strip()]
    return {
        "branch": branch,
        "head": head,
        "clean": not dirty,
        "uncommitted_files": dirty[:20],
    }


def pytest_available() -> bool:
    """Whether pytest is importable from the venv (or the running interpreter)."""
    venv_py = REPO / ".venv" / "bin" / "python"
    py = str(venv_py) if venv_py.exists() else sys.executable
    code, _ = _run([py, "-c", "import pytest"])
    return code == 0


def service_state() -> dict:
    """artoo.service active state + when the running process started."""
    active = _run(["systemctl", "--user", "is-active", "artoo.service"])[1]
    started = _run(
        ["systemctl", "--user", "show", "artoo.service",
         "-p", "ExecMainStartTimestamp", "--value"]
    )[1]
    return {"active": active or "unknown", "started": started}


def snapshot() -> dict:
    """One ground-truth bundle for the self_check tool. The tool handler adds
    the live tool catalog (which it has directly) before returning."""
    return {
        "repo": str(REPO),
        "git": git_state(),
        "pytest_installed": pytest_available(),
        "service": service_state(),
    }
