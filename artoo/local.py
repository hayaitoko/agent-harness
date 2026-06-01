"""Unified local tool — gives the boss read/write/run/dev/build access
to its own homelab VM.

The boss previously had no way to modify its own source, run shell
commands, or invoke its own pipelines (`dev_pipeline.run`,
`round_pipeline.RoundEngine`) — it could only spawn workers and call
github(). This module closes that gap with one tool: `local`, dispatched
by an `op` field, returning a JSON-serializable dict.

Operations:
    read    — read a file under SAFE_ROOT, return its text
    write   — write a file under SAFE_ROOT (creates intermediate dirs)
    run     — execute a shell command inside SAFE_ROOT, capture output
    shell   — alias for run (mental ergonomics for the boss)
    dev     — invoke `artoo.dev_pipeline.run(task, context=...)`
              and return the structured DevResult
    build   — kick off a round_pipeline run for a project directory
              (or a single step against an existing project), return
              the round status + project state

Safety:
- SAFE_ROOT defaults to /home/youruser/artoo (the agent's own source tree).
  Override via ARTOO_LOCAL_SAFE_ROOT env var if you want a tighter
  sandbox.
- read/write reject any path that resolves outside SAFE_ROOT after
  pathlib.Path.resolve() (catches ../ traversal AND symlinks pointing
  outside).
- run/shell default cwd to SAFE_ROOT. The caller can pass a `cwd` that
  must also be under SAFE_ROOT.
- build's `project_dir` is allowed outside SAFE_ROOT because that's
  literally what /build does — scratch projects live in the build
  workshop (config.BUILDS_DIR, default ~/builds/<slug>/).
  The boss is already trusted to dispatch builds via Telegram.

The boss is async, so dev/run hand off to threads via asyncio.to_thread
(dev_pipeline.run is sync; create_subprocess_shell is async). Errors
surface as strings inside the dict — never raise — so the boss sees
them and can self-correct.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from . import config

_log = logging.getLogger("artoo.local")


def _safe_root() -> Path:
    """Read SAFE_ROOT lazily so tests can monkeypatch ARTOO_LOCAL_SAFE_ROOT.
    Default is the repo root (config.SAFE_ROOT) — the agent's own source tree."""
    override = os.environ.get("ARTOO_LOCAL_SAFE_ROOT", "").strip()
    if override:
        return Path(override).resolve()
    return config.SAFE_ROOT


# Per-call timeout for run/shell. Boss can override per-call via the
# `timeout_s` param up to MAX_RUN_TIMEOUT_S. Default is conservative —
# long-running stuff should go through /build or /dev instead.
DEFAULT_RUN_TIMEOUT_S = 60
MAX_RUN_TIMEOUT_S = 600

# read/write caps. Single-file ops above this size suggest you want
# binary handling or chunked reads, which this tool isn't built for.
MAX_READ_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_WRITE_BYTES = 5 * 1024 * 1024


# ── Public entrypoint (orchestrator hook) ──────────────────────────

async def call(op: str, **kwargs: Any) -> dict:
    """Dispatch entrypoint used by the boss's tool handler.

    Async because run/shell + dev/build need to not block the event
    loop. Returns a JSON-serializable dict — never raises; errors
    appear under the `error` key so the boss can recover.
    """
    handlers = {
        "read":    _op_read,
        "write":   _op_write,
        "run":     _op_run,
        "shell":   _op_run,   # alias
        "dev":     _op_dev,
        "build":   _op_build,
        "restart": _op_restart,
    }
    handler = handlers.get(op)
    if handler is None:
        return {"ok": False, "error": f"unknown op: {op!r}; "
                f"valid: {sorted(handlers.keys())}"}
    try:
        return await handler(**kwargs)
    except TypeError as e:
        # Bad kwargs — usually a missing required param. Surface
        # cleanly so the boss sees what's wrong.
        return {"ok": False, "op": op, "error": f"bad args: {e}"}
    except Exception as e:  # noqa: BLE001 — boundary; we never raise to caller
        _log.exception("local op %s raised", op)
        return {"ok": False, "op": op, "error": f"{type(e).__name__}: {e}"}


# ── read / write ───────────────────────────────────────────────────

async def _op_read(*, path: str = "", max_bytes: int = MAX_READ_BYTES) -> dict:
    if not path:
        return {"ok": False, "op": "read", "error": "path is required"}
    # allow_missing=True so a non-existent file inside the safe root
    # surfaces as "not a file" (clearer than "outside safe root").
    target = _resolve_under_safe(path, allow_missing=True)
    if target is None:
        return {"ok": False, "op": "read",
                "error": f"path outside safe root: {path}"}
    if not target.is_file():
        return {"ok": False, "op": "read", "error": f"not a file: {path}"}
    cap = min(int(max_bytes or MAX_READ_BYTES), MAX_READ_BYTES)
    try:
        data = await asyncio.to_thread(target.read_bytes)
    except (OSError, PermissionError) as e:
        return {"ok": False, "op": "read", "error": f"read failed: {e}"}
    truncated = len(data) > cap
    payload = data[:cap]
    try:
        text = payload.decode("utf-8")
        binary = False
    except UnicodeDecodeError:
        text = payload.decode("utf-8", errors="replace")
        binary = True
    return {
        "ok": True,
        "op": "read",
        "path": str(target.relative_to(_safe_root())),
        "bytes": len(data),
        "truncated": truncated,
        "binary_decoded_with_replacement": binary,
        "text": text,
    }


async def _op_write(*, path: str = "", content: Optional[str] = None) -> dict:
    if not path:
        return {"ok": False, "op": "write", "error": "path is required"}
    if content is None:
        return {"ok": False, "op": "write", "error": "content is required"}
    if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
        return {"ok": False, "op": "write",
                "error": f"content exceeds {MAX_WRITE_BYTES} bytes; split the write"}
    target = _resolve_under_safe(path, allow_missing=True)
    if target is None:
        return {"ok": False, "op": "write",
                "error": f"path outside safe root: {path}"}
    try:
        await asyncio.to_thread(_mkdir_and_write, target, content)
    except (OSError, PermissionError) as e:
        return {"ok": False, "op": "write", "error": f"write failed: {e}"}
    return {
        "ok": True,
        "op": "write",
        "path": str(target.relative_to(_safe_root())),
        "bytes_written": len(content.encode("utf-8")),
    }


def _mkdir_and_write(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


# ── run / shell ────────────────────────────────────────────────────

async def _op_run(
    *,
    cmd: str = "",
    cwd: Optional[str] = None,
    timeout_s: int = DEFAULT_RUN_TIMEOUT_S,
) -> dict:
    if not cmd:
        return {"ok": False, "op": "run", "error": "cmd is required"}
    timeout = max(1, min(int(timeout_s or DEFAULT_RUN_TIMEOUT_S), MAX_RUN_TIMEOUT_S))
    workdir = _safe_root()
    if cwd:
        resolved = _resolve_under_safe(cwd)
        if resolved is None:
            return {"ok": False, "op": "run",
                    "error": f"cwd outside safe root: {cwd}"}
        if not resolved.is_dir():
            return {"ok": False, "op": "run", "error": f"cwd not a directory: {cwd}"}
        workdir = resolved

    started = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=str(workdir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as e:
        return {"ok": False, "op": "run", "error": f"could not start: {e}"}
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        try:
            await proc.wait()
        except Exception:  # noqa: BLE001
            pass
        return {
            "ok": False, "op": "run", "error": f"timeout after {timeout}s",
            "duration_ms": int((time.monotonic() - started) * 1000),
        }
    return {
        "ok": (proc.returncode or 0) == 0,
        "op": "run",
        "cmd": cmd,
        "cwd": str(workdir),
        "exit_code": proc.returncode or 0,
        "stdout": stdout_b.decode("utf-8", errors="replace"),
        "stderr": stderr_b.decode("utf-8", errors="replace"),
        "duration_ms": int((time.monotonic() - started) * 1000),
    }


# ── dev (artoo.dev_pipeline.run) ───────────────────────────────────

async def _op_dev(
    *,
    task: str = "",
    context: str = "",
    max_rounds: int = 3,
    security_review: bool = False,
) -> dict:
    """Dispatch a dev task via dev_pipeline. Sync internally — runs on
    a thread so the orchestrator event loop stays responsive."""
    if not task:
        return {"ok": False, "op": "dev", "error": "task is required"}
    from . import dev_pipeline  # local import — pipeline imports runtime
    try:
        result = await asyncio.to_thread(
            dev_pipeline.run,
            task=task,
            max_rounds=max(1, min(int(max_rounds or 3), 5)),
            context=context or "",
            security_review=bool(security_review),
        )
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "op": "dev", "error": f"{type(e).__name__}: {e}"}
    return {
        "ok": result.ok,
        "op": "dev",
        "task": result.task,
        "approved": result.approved,
        "rounds": result.rounds,
        "cost_usd": round(result.cost_usd, 4),
        "code": result.code,
        "error": result.error,
    }


# ── build (artoo.conductor) ────────────────────────────────────────

async def _op_build(
    *,
    project_dir: str = "",
    goal: str = "",
    autonomous: bool = False,
    max_rounds: int = 1,
) -> dict:
    """Drive a build/dev pass on a project via the agentic CONDUCTOR.

    A Sonnet-driven tool loop with a code-enforced verify gate (artoo.conductor).
    Works for greenfield builds AND developing an existing project — point it at
    a dir + a goal; it reads, edits, delegates, and drives to verify-green.
    `autonomous`/`max_rounds` are accepted for boss-tool back-compat but unused
    (the conductor self-bounds via cycles + a daily budget cap).
    """
    if not project_dir:
        return {"ok": False, "op": "build", "error": "project_dir is required"}

    # A brand-new project with no goal and no SPEC/PROGRESS on disk has nothing
    # to build toward — refuse rather than start a conductor that would flail.
    # (Existing PROGRESS.md/SPEC.md count as a goal.)
    pdir_check = Path(project_dir).expanduser()
    if not pdir_check.is_absolute():
        pdir_check = config.BUILDS_DIR / project_dir
    if (not goal
            and not (pdir_check / "PROGRESS.md").exists()
            and not (pdir_check / "SPEC.md").exists()):
        return {"ok": False, "op": "build",
                "error": "project_dir is new — goal is required to seed it"}

    from .conductor import run_build
    try:
        res = await asyncio.to_thread(run_build, project_dir, goal)
    except Exception as e:  # noqa: BLE001 — boundary; never raise to the boss
        return {"ok": False, "op": "build", "error": f"{type(e).__name__}: {e}"}
    return {
        "ok": res.ok,
        "op": "build",
        "engine": "conductor",
        "project_dir": res.project_dir,
        "cycles": res.cycles,
        "cost_usd": round(res.cost_usd, 4),
        "halt_reason": res.halt_reason,
        "verify_summary": res.verify_summary[:1200],
        "files_written": res.files_written,
        "error": res.error,
    }


# ── restart (self-restart with safety) ─────────────────────────────

# Service unit name. Hardcoded — the boss should not be picking
# arbitrary systemd units to restart.
_ARTOO_UNIT = "artoo.service"

# Delay between command spawn and the actual restart. Gives the Telegram
# adapter time to flush its final response message before the process
# dies. 3s is enough for one outbound HTTPS call; longer is fine too.
_RESTART_DELAY_DEFAULT = 3
_RESTART_DELAY_MAX = 30


async def _op_restart(
    *,
    reason: str = "",
    delay_s: int = _RESTART_DELAY_DEFAULT,
    test_first: bool = True,
    tests_cmd: str = "pytest tests/ -q",
    tests_timeout_s: int = 120,
) -> dict:
    """Trigger a systemctl-user restart of artoo.service.

    Safety:
    - `test_first=True` (default) runs `pytest tests/ -q` first and
      refuses to restart on non-zero exit. The boss should NEVER
      override this without explicit human confirmation — restarting
      onto broken code can crash-loop the service.
    - The actual restart is scheduled with a small delay (default 3s)
      via `sleep <delay> && systemctl --user restart artoo.service`
      run in the background. This lets the boss's final response
      message land in Telegram before the process dies.

    Returns immediately. Caller (the boss) is expected to send a
    final "back in ~Ns" message to the operator and stop emitting tool
    calls — the next message will hit the freshly-restarted process.

    A `reason` string is logged + included in the return for the
    artoo.log audit trail.
    """
    delay = max(0, min(int(delay_s or _RESTART_DELAY_DEFAULT), _RESTART_DELAY_MAX))

    payload: dict = {
        "ok": True,
        "op": "restart",
        "delay_s": delay,
        "reason": str(reason or "")[:300],
        "test_first": bool(test_first),
        "tests_passed": None,
    }

    if test_first:
        # Run the test command inside the safe root so it picks up the
        # repo's pyproject + .venv. We use _op_run to inherit its
        # timeout + cwd guards; that also surfaces stdout/stderr the
        # boss can show the operator if tests fail.
        test_result = await _op_run(
            cmd=tests_cmd,
            timeout_s=max(10, min(int(tests_timeout_s or 120), 600)),
        )
        payload["tests_passed"] = bool(test_result.get("ok"))
        if not test_result.get("ok"):
            return {
                **payload,
                "ok": False,
                "error": "tests failed — refusing to restart onto broken code",
                "tests": {
                    "exit_code": test_result.get("exit_code"),
                    "stdout_tail": (test_result.get("stdout") or "")[-1500:],
                    "stderr_tail": (test_result.get("stderr") or "")[-1500:],
                    "duration_ms": test_result.get("duration_ms"),
                    "error": test_result.get("error"),
                },
            }

    _log.warning(
        "local.restart firing in %ds (reason=%r, tests=%s)",
        delay, reason, payload["tests_passed"],
    )

    # Fire-and-forget the actual restart. The shell handles the sleep
    # so we return BEFORE the systemctl call fires. By the time
    # systemd kills this process, the tool's return value has already
    # been serialized and sent back through the agent loop, and the
    # boss has emitted its final message.
    cmd = f"(sleep {delay} && systemctl --user restart {_ARTOO_UNIT}) &"
    try:
        await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError as e:
        return {**payload, "ok": False, "error": f"could not schedule restart: {e}"}

    payload["note"] = (
        f"Restart scheduled in {delay}s. Tell the operator to wait then send another "
        f"message — the new process will pick up there."
    )
    return payload


# ── Path safety ────────────────────────────────────────────────────

def _resolve_under_safe(path: str, *, allow_missing: bool = False) -> Optional[Path]:
    """Resolve `path` and return it if and only if it's under SAFE_ROOT
    after symlink resolution. Returns None for any traversal attempt
    (absolute outside root, ../ segments, symlinks pointing out).

    For writes (`allow_missing=True`), the target file doesn't need to
    exist yet but its resolved parent must be under SAFE_ROOT.
    """
    if not path:
        return None
    root = _safe_root()
    candidate = (root / path).expanduser()
    # Absolute paths handed in are resolved as-is (allows the boss to
    # pass /home/youruser/artoo/foo.py if it wants); they still must end
    # up inside root.
    if Path(path).is_absolute():
        candidate = Path(path).expanduser()
    try:
        # For missing targets, resolve the parent and append the file
        # name — Path.resolve() on a missing file works in 3.6+ via
        # strict=False, but we want the parent to actually exist or be
        # reachable. Resolve in non-strict mode then check.
        if allow_missing:
            resolved = candidate.resolve()
        else:
            resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved
