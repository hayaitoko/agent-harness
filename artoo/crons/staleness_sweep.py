"""Monthly staleness sweep.

Soft-archives active memories that haven't been retrieved in 180+ days.
Search filters archived memories out by default; explicit recall still
works via ``include_archived=True``. Nothing is hard-deleted — only the
status flag flips, so a bad threshold or surprise sweep is easily
reversible.

Grandfathering: memories without ``last_retrieved_at`` (legacy points
written before the field shipped on 2026-05-20) fall back to
``created_at``; the first time they're returned by search_memory the
field gets populated and they enter the regular lifecycle. Points with
NEITHER field are treated as fresh (never archived) — better to leak a
stale memory than to surprise-archive something we have no telemetry on.

Output: Telegram digest with count + a sample of titles archived so
the operator knows what just got tucked away. Restore via the boss's
``restore_memory`` tool if anything looks wrong.
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

from .. import config, memory, notify
from ..scheduler import Job

log = logging.getLogger("artoo.cron.staleness_sweep")

# Soft-archive points untouched for this many days. Conservative —
# quarterly-relevant facts survive. Tunable; revisit after a few months
# of real data.
STALENESS_DAYS = 180

# Cap per-run archives so a misconfigured threshold can't sweep
# everything overnight. If the cap fires, the rest survive until next
# month with a Telegram notice.
MAX_ARCHIVES_PER_RUN = 100

# Sample of titles to include in the Telegram digest. Beyond this we
# just give the count.
DIGEST_TITLE_SAMPLE = 8


async def run() -> None:
    cutoff = _now() - datetime.timedelta(days=STALENESS_DAYS)
    cutoff_iso = cutoff.isoformat() + "Z"

    client = memory.client()
    archived: list[dict] = []
    skipped_grandfathered = 0
    capped = False

    offset = None
    while True:
        if len(archived) >= MAX_ARCHIVES_PER_RUN:
            capped = True
            break
        points, offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
            scroll_filter=memory._active_filter(),
        )
        for pt in points:
            if len(archived) >= MAX_ARCHIVES_PER_RUN:
                capped = True
                break
            payload = pt.payload or {}
            uid = str(pt.id)
            anchor = _retrieval_anchor(payload)
            if anchor is None:
                skipped_grandfathered += 1
                continue
            if anchor >= cutoff_iso:
                continue
            try:
                outcome = memory.delete_memory(uid, hard=False)
            except Exception as e:  # noqa: BLE001
                log.warning("soft-archive failed for %s: %s", uid[:8], e)
                continue
            if outcome.get("result") == "archived":
                archived.append({
                    "uuid": uid,
                    "title": _payload_title(payload),
                    "anchor": anchor,
                })
        if offset is None:
            break

    summary = (
        f"🛏 staleness_sweep: archived {len(archived)} memory(ies) older than "
        f"{STALENESS_DAYS} days"
    )
    if skipped_grandfathered:
        summary += f"; skipped {skipped_grandfathered} grandfathered (no retrieval anchor)"
    if capped:
        summary += f" — cap {MAX_ARCHIVES_PER_RUN} reached, more stale memories remain"
    if archived:
        sample = archived[:DIGEST_TITLE_SAMPLE]
        summary += "\n\nSample:\n" + "\n".join(
            f"• `{a['uuid'][:8]}` — {a['title']}" for a in sample
        )
        if len(archived) > DIGEST_TITLE_SAMPLE:
            summary += f"\n…and {len(archived) - DIGEST_TITLE_SAMPLE} more."
    summary += (
        "\n\nRestore any with `restore_memory(<uuid>)` if I judged wrong."
    )
    log.info(summary.replace("\n", " | "))
    if archived or capped:
        await notify.to_telegram(summary)


def _retrieval_anchor(payload: dict) -> Optional[str]:
    """Return the ISO timestamp we'll compare against the staleness
    cutoff. Prefer last_retrieved_at; fall back to created_at; if
    neither exists, return None so the caller grandfathers the point."""
    anchor = payload.get("last_retrieved_at") or payload.get("created_at")
    if not isinstance(anchor, str) or not anchor:
        return None
    return anchor


def _payload_title(payload: dict) -> str:
    text = payload.get("text", "") if isinstance(payload, dict) else ""
    if not isinstance(text, str):
        return "(no text)"
    first = text.strip().split("\n", 1)[0]
    return (first[:80] + ("…" if len(first) > 80 else "")) or "(empty)"


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


JOB = Job(
    name="staleness_sweep",
    schedule="0 5 1 * *",  # 05:00 on the 1st of every month
    fn=run,
    timeout=900,
)
