"""Reminder fire loop.

Ticks every minute. Any reminder whose `fire_at` has passed gets sent to its
chat (or TELEGRAM_HOME_CHANNEL if none set) and removed from disk in the
same atomic step (see reminders.claim_due). One-shot semantics — nothing
leftover after firing.
"""
from __future__ import annotations

import logging

from .. import notify, reminders
from ..scheduler import Job

log = logging.getLogger("artoo.cron.reminders")


async def run() -> None:
    due = reminders.claim_due()
    if not due:
        return
    for r in due:
        text = r.get("text") or "(reminder)"
        chat_id = r.get("chat_id")
        try:
            await notify.to_telegram(f"⏰ {text}", chat_id=chat_id)
            log.info("reminder fired %s: %r", str(r.get("uuid", ""))[:8], text[:80])
        except Exception as e:  # noqa: BLE001
            # We've already deleted the file — log loudly and move on. Retry
            # storms are worse than the occasional missed reminder.
            log.exception("reminder delivery failed for %s: %s", r.get("uuid"), e)


JOB = Job(
    name="reminders",
    schedule="* * * * *",  # every minute — sub-minute precision isn't worth it
    fn=run,
    timeout=60,
)
