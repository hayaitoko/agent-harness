"""Daily OpenRouter spend check.

Fires once a day at 9:00 AM local, queries OpenRouter's /credits endpoint,
posts the current balance + total usage to Telegram. v0.0.2 will track
deltas in sqlite and only message on anomaly.
"""
from __future__ import annotations

import logging

import httpx

from .. import config, notify
from ..scheduler import Job

log = logging.getLogger("artoo.cron.spend_check")


async def run() -> None:
    api_key = config.optional("OPENROUTER_API_KEY")
    if not api_key:
        log.warning("OPENROUTER_API_KEY not set; skipping spend_check")
        return

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(
                "https://openrouter.ai/api/v1/credits",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            r.raise_for_status()
            data = r.json().get("data", {})
    except Exception as e:  # noqa: BLE001
        await notify.to_telegram(f"💸 spend_check error: {e}")
        return

    total = float(data.get("total_credits", 0) or 0)
    usage = float(data.get("total_usage", 0) or 0)
    remaining = total - usage
    text = (
        f"💰 OpenRouter today\n"
        f"   used: ${usage:.4f}\n"
        f"   total: ${total:.4f}\n"
        f"   remaining: ${remaining:.4f}"
    )
    log.info("spend_check: %s", text.replace("\n", " | "))
    await notify.to_telegram(text)


JOB = Job(
    name="spend_check",
    schedule="0 9 * * *",  # 9:00 AM daily
    fn=run,
    timeout=60,
)
