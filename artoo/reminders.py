"""One-shot reminder scheduling.

Reminders are persisted as JSON files under DATA_DIR/reminders/<uuid>.json so
they survive restarts. The scheduler's `reminders` cron ticks every minute,
fires anything whose `fire_at` has passed, then deletes the file (self-clean
— no leftover crons after the reminder fires).

Public surface:
  - parse_delta("30 minutes") -> 1800   (raises ValueError on unparseable)
  - schedule(delta_s, text, chat_id=None) -> uuid
  - list_pending() -> list[dict]
  - cancel(uuid) -> bool

The MCP `remind` tool wraps schedule(); the in-process cron in
artoo/crons/reminders.py wraps the fire+cleanup loop.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid as _uuid
from pathlib import Path

from . import config

log = logging.getLogger("artoo.reminders")

REMINDERS_DIR = config.DATA_DIR / "reminders"
REMINDERS_DIR.mkdir(parents=True, exist_ok=True)

# Unit -> seconds. Accepts singular + plural + common shorthand.
_UNITS: dict[str, int] = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
}

# Matches one or more "<number><optional space><unit>" chunks, e.g.
# "30m", "1 hour", "2h 30m", "1d 4h 15m".
_CHUNK_RE = re.compile(r"(\d+)\s*([a-zA-Z]+)")

# rids are uuid4().hex — 32 hex chars, no dashes. Validate before any
# path construction so a caller can't traverse out of REMINDERS_DIR.
_RID_RE = re.compile(r"^[0-9a-f]{32}$")


def parse_delta(text: str) -> int:
    """Parse a human time delta into seconds. Raises ValueError if nothing matched."""
    total = 0
    for n, unit in _CHUNK_RE.findall(text.strip().lower()):
        secs = _UNITS.get(unit)
        if secs is None:
            raise ValueError(f"unknown time unit: {unit!r}")
        total += int(n) * secs
    if total <= 0:
        raise ValueError(f"could not parse delta from {text!r}")
    return total


def schedule(delta_s: int, text: str, chat_id: str | int | None = None) -> str:
    """Persist a reminder firing `delta_s` seconds from now. Returns its uuid."""
    if delta_s <= 0:
        raise ValueError("delta_s must be > 0")
    rid = _uuid.uuid4().hex
    payload = {
        "uuid": rid,
        "fire_at": time.time() + delta_s,
        "text": text,
        "chat_id": str(chat_id) if chat_id is not None else None,
        "created_at": time.time(),
    }
    path = REMINDERS_DIR / f"{rid}.json"
    path.write_text(json.dumps(payload, indent=2))
    log.info("reminder scheduled %s in %ds: %r", rid[:8], delta_s, text[:80])
    return rid


def list_pending() -> list[dict]:
    """Return all pending reminder payloads, ordered by fire_at ascending."""
    out: list[dict] = []
    for p in REMINDERS_DIR.glob("*.json"):
        try:
            out.append(json.loads(p.read_text()))
        except Exception as e:  # noqa: BLE001
            log.error("could not read reminder %s: %s", p.name, e)
    out.sort(key=lambda r: r.get("fire_at", 0))
    return out


def _path_for(rid: str) -> Path:
    return REMINDERS_DIR / f"{rid}.json"


def cancel(rid: str) -> bool:
    """Delete a pending reminder. Returns True if it existed."""
    if not _RID_RE.match(rid):
        return False
    path = _path_for(rid)
    if not path.exists():
        return False
    path.unlink()
    log.info("reminder cancelled %s", rid[:8])
    return True


def claim_due(now: float | None = None) -> list[dict]:
    """Atomically claim every reminder whose fire_at has passed.

    Deletes the on-disk file *before* returning, so concurrent ticks can't
    double-fire. Caller is responsible for delivery; if delivery fails the
    caller should log — re-queueing would risk retry storms.
    """
    now = now or time.time()
    due: list[dict] = []
    for p in sorted(REMINDERS_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except Exception as e:  # noqa: BLE001
            log.error("could not read reminder %s: %s", p.name, e)
            continue
        if data.get("fire_at", 0) <= now:
            try:
                p.unlink()
            except FileNotFoundError:
                continue  # raced with another tick; skip
            due.append(data)
    return due
