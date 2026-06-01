"""Daily spend budget for the conductor.

A single rolling daily cap (default $5, resets on the operator's Pacific day),
persisted to data/conductor_budget.json. Operator controls: /budget set N,
/budget +N (bump), and a tap-to-add-$5 button on a budget halt.

The cap is a wallet stop-loss, NOT a quality gate — it trips only on cumulative
daily spend, never on "the build is struggling."
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import threading
from zoneinfo import ZoneInfo

from .. import config

_log = logging.getLogger("artoo.conductor.budget")

# Serializes read-modify-write of the ledger. run_build executes in
# asyncio.to_thread (telegram.py / local.py), so concurrent builds in one
# process would otherwise race add_spend and undercount the day's spend,
# letting the cap overshoot. Same-process only; the atomic _save below is what
# protects against a torn read across processes/crashes.
_LOCK = threading.Lock()

# the operator's day — resets at local midnight, not UTC (his personal budget, not a
# customer-facing surface, so local TZ is correct here).
_TZ = ZoneInfo("America/Los_Angeles")
DEFAULT_DAILY_CAP_USD = 5.0


def _ledger_path():
    return config.DATA_DIR / "conductor_budget.json"


def _today() -> str:
    return datetime.datetime.now(_TZ).date().isoformat()


def _default_cap() -> float:
    try:
        return float(config.optional("ARTOO_CONDUCTOR_DAILY_BUDGET_USD") or DEFAULT_DAILY_CAP_USD)
    except (TypeError, ValueError):
        return DEFAULT_DAILY_CAP_USD


def _load() -> dict:
    p = _ledger_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save(d: dict) -> None:
    # Atomic write: a plain write_text truncates-then-writes, so a concurrent
    # _load could read a partial file, hit JSONDecodeError, treat the day as
    # un-started, and RESET spent to 0.0 — silently reopening the stop-loss.
    # Write to a temp file and os.replace (atomic rename) so readers only ever
    # see a complete ledger.
    try:
        p = _ledger_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(d, indent=2))
        os.replace(tmp, p)
    except OSError:
        _log.exception("could not write budget ledger")


def _current() -> dict:
    """Today's ledger row, resetting spend + cap when the day rolls over."""
    d = _load()
    today = _today()
    if d.get("date") != today:
        d = {"date": today, "spent": 0.0, "cap": _default_cap()}
        _save(d)
    return d


def spent_today() -> float:
    return float(_current().get("spent", 0.0))


def cap_today() -> float:
    return float(_current().get("cap", _default_cap()))


def remaining_today() -> float:
    c = _current()
    return max(0.0, float(c.get("cap", 0.0)) - float(c.get("spent", 0.0)))


def add_spend(amount: float) -> None:
    if not amount:
        return
    with _LOCK:
        d = _current()
        d["spent"] = round(float(d.get("spent", 0.0)) + float(amount), 6)
        _save(d)


def bump_cap(amount: float) -> float:
    """Raise today's cap by `amount` (the +$5 button / `/budget +N`)."""
    with _LOCK:
        d = _current()
        d["cap"] = round(float(d.get("cap", 0.0)) + float(amount), 4)
        _save(d)
        return d["cap"]


def set_cap(amount: float) -> float:
    with _LOCK:
        d = _current()
        d["cap"] = round(max(0.0, float(amount)), 4)
        _save(d)
        return d["cap"]


def status_line() -> str:
    c = _current()
    spent = float(c.get("spent", 0.0))
    cap = float(c.get("cap", 0.0))
    return (f"💰 daily budget ({c.get('date')}): ${spent:.3f} spent / ${cap:.2f} cap "
            f"— ${max(0.0, cap - spent):.2f} left")
