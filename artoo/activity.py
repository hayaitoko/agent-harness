"""Append-only activity log — JSONL per day at data/activity/YYYY-MM-DD.jsonl.

Captures the events the nightly `reflection` cron will eventually mine for
self-improvement suggestions: every chat turn, every worker dispatch, every
cron fire. Cheap to write, cheap to read sequentially.

Schema is intentionally loose — events are dicts with `type` + `ts` + payload.
Reflection can grep/filter as it needs.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date
from typing import Any

from . import config

log = logging.getLogger("artoo.activity")

_DIR = config.DATA_DIR / "activity"
_DIR.mkdir(parents=True, exist_ok=True)


def _path_for_today() -> str:
    return str(_DIR / f"{date.today().isoformat()}.jsonl")


def log_event(event_type: str, **payload: Any) -> None:
    """Append one event line. Never raises — logs but doesn't propagate errors."""
    event = {"type": event_type, "ts": time.time(), **payload}
    try:
        with open(_path_for_today(), "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception as e:  # noqa: BLE001
        log.warning("activity log write failed: %s", e)


def log_turn(
    *,
    chat_id: str,
    user_text: str,
    response_text: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    duration_s: float,
    error: str | None = None,
) -> None:
    log_event(
        "turn",
        chat_id=chat_id,
        user_text=user_text[:1000],
        response_text=response_text[:1000],
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        duration_s=round(duration_s, 2),
        error=error,
    )


def log_worker(
    *,
    name: str,
    model: str,
    prompt: str,
    result: str,
    tokens_in: int,
    tokens_out: int,
    duration_s: float,
    error: str | None = None,
) -> None:
    log_event(
        "worker",
        name=name,
        model=model,
        prompt=prompt[:500],
        result=result[:500],
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        duration_s=round(duration_s, 2),
        error=error,
    )


def log_cron(*, name: str, duration_s: float, error: str | None = None) -> None:
    log_event("cron", name=name, duration_s=round(duration_s, 2), error=error)
