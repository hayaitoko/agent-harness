"""Recurring agentic cron jobs.

Unlike one-shot reminders (artoo/reminders.py), a cron job fires on a repeating
cron schedule and, on each fire, runs a PROMPT through the full boss agent loop
(orchestrator.respond) — Artoo reasons with all its tools and the result is sent
to the operator's Telegram. This is how the operator schedules autonomous recurring work, e.g.
"every weekday at 8am, check my GitHub notifications and summarize them".

Jobs persist as JSON under DATA_DIR/cron_jobs/<uuid>.json so they survive
restarts; load_all() re-registers them into the scheduler at startup. Creating
or deleting a job mutates both disk and the live scheduler, so changes take
effect immediately — the scheduler re-reads its job dict on every tick.

Public surface:
  - create(schedule, prompt, chat_id=None) -> uuid   (raises ValueError on bad input)
  - list_jobs() -> list[dict]
  - delete(jid) -> bool
  - load_all() -> int                                 (startup: register persisted jobs)

The MCP `cron` tool wraps create/list/delete; load_all() runs once from the
telegram adapter's post_init, right after the code-defined crons register.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid as _uuid
from pathlib import Path

from croniter import croniter

from . import config, notify
from .scheduler import Job, scheduler

log = logging.getLogger("artoo.jobs")

JOBS_DIR = config.DATA_DIR / "cron_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Scheduler job names for dynamic crons are namespaced so they can't collide
# with code-defined crons (spend_check, reminders, ...) and so delete() can't
# unregister one by accident.
_JOB_PREFIX = "userjob:"

# jids are uuid4().hex — 32 hex chars. Validate before any path construction so
# a caller can't traverse out of JOBS_DIR.
_JID_RE = re.compile(r"^[0-9a-f]{32}$")

# A fired job runs an agentic boss turn (LLM + tools), which can take a while.
# Give it well more headroom than a normal cron's 300s default.
_JOB_TIMEOUT_S = 900


def _valid_schedule(schedule: str) -> bool:
    """True if `schedule` is a parseable cron expression."""
    try:
        croniter(schedule)
        return True
    except (ValueError, KeyError, TypeError):
        return False


def _job_name(jid: str) -> str:
    return f"{_JOB_PREFIX}{jid}"


def _path_for(jid: str) -> Path:
    return JOBS_DIR / f"{jid}.json"


def _touch_last_run(jid: str) -> None:
    """Record the last fire time on disk. Best-effort — a failure here must not
    break delivery, so swallow and log."""
    path = _path_for(jid)
    try:
        payload = json.loads(path.read_text())
        payload["last_run"] = time.time()
        path.write_text(json.dumps(payload, indent=2))
    except Exception as e:  # noqa: BLE001
        log.warning("could not record last_run for %s: %s", jid[:8], e)


def _make_fire_fn(jid: str, prompt: str, chat_id: str | None):
    """Build the async fire body for a job: run the boss agent loop on `prompt`
    and deliver the result to Telegram."""

    async def _fire() -> None:
        # Lazy import: orchestrator imports jobs (the cron tool handler), so a
        # module-level import here would be circular.
        from . import orchestrator

        try:
            result = await asyncio.to_thread(
                orchestrator.respond, prompt, chat_id=chat_id or "cron"
            )
        except Exception as e:  # noqa: BLE001
            log.exception("cron job %s agent turn failed: %s", jid[:8], e)
            await notify.to_telegram(
                f"⚠️ scheduled job failed: {e}", chat_id=chat_id
            )
            return

        text = (result.text or "").strip()
        if not text:
            detail = f"; error: {result.error}" if result.error else ""
            text = f"(scheduled job ran but produced no text{detail})"
        await notify.to_telegram(text, chat_id=chat_id)
        _touch_last_run(jid)

    return _fire


def _register(payload: dict) -> None:
    """Register one persisted payload into the live scheduler."""
    jid = payload["uuid"]
    scheduler.register(
        Job(
            name=_job_name(jid),
            schedule=payload["schedule"],
            fn=_make_fire_fn(jid, payload["prompt"], payload.get("chat_id")),
            timeout=_JOB_TIMEOUT_S,
            enabled=payload.get("enabled", True),
        )
    )


def create(schedule: str, prompt: str, chat_id: str | int | None = None) -> str:
    """Create a recurring agentic job. `schedule` is a 5-field cron expression
    (e.g. '0 8 * * 1-5' = 8am on weekdays). Persists, then registers into the
    live scheduler. Returns the job uuid. Raises ValueError on bad input."""
    schedule = schedule.strip()
    if not _valid_schedule(schedule):
        raise ValueError(f"invalid cron expression: {schedule!r}")
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")

    jid = _uuid.uuid4().hex
    payload = {
        "uuid": jid,
        "schedule": schedule,
        "prompt": prompt.strip(),
        "chat_id": str(chat_id) if chat_id is not None else None,
        "enabled": True,
        "created_at": time.time(),
        "last_run": None,
    }
    _path_for(jid).write_text(json.dumps(payload, indent=2))
    _register(payload)
    log.info("cron job created %s schedule=%r: %r", jid[:8], schedule, prompt[:80])
    return jid


def list_jobs() -> list[dict]:
    """Return all persisted job payloads, newest first."""
    out: list[dict] = []
    for p in JOBS_DIR.glob("*.json"):
        try:
            out.append(json.loads(p.read_text()))
        except Exception as e:  # noqa: BLE001
            log.error("could not read cron job %s: %s", p.name, e)
    out.sort(key=lambda j: j.get("created_at", 0), reverse=True)
    return out


def delete(jid: str) -> bool:
    """Delete a job from disk and unregister it from the scheduler. Returns
    True if it existed."""
    if not _JID_RE.match(jid):
        return False
    path = _path_for(jid)
    existed = path.exists()
    if existed:
        path.unlink()
    # Unregister regardless, in case the file was already gone but the live job
    # lingered. unregister() returns False harmlessly if absent.
    scheduler.unregister(_job_name(jid))
    if existed:
        log.info("cron job deleted %s", jid[:8])
    return existed


def load_all() -> int:
    """Register every persisted job into the scheduler. Called once at startup.
    Returns the number of jobs loaded."""
    count = 0
    for p in sorted(JOBS_DIR.glob("*.json")):
        try:
            payload = json.loads(p.read_text())
        except Exception as e:  # noqa: BLE001
            log.error("could not read cron job %s: %s", p.name, e)
            continue
        if not _valid_schedule(payload.get("schedule", "")):
            log.error("cron job %s has an invalid schedule; skipping", p.name)
            continue
        _register(payload)
        count += 1
    if count:
        log.info("loaded %d dynamic cron job(s)", count)
    return count
