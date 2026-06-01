"""Nightly resync — pull any new points from claude_memories (v1) into
claude_memories_v2 with mxbai embeddings.

Why this exists: the Anthropic-managed Qdrant MCP connector
(mcp__claude_ai_Qdrant__save_memory, used by Claude sessions including this
one for cross-session memory) writes to the v1 collection. Artoo reads from
v2. Without this cron, memories saved via that path are invisible to artoo
until manually migrated.

Strategy: find UUIDs in v1 missing from v2, embed text with mxbai, upsert
to v2 with same UUID + payload. Idempotent + cheap on GPU (~7ms/point).
"""
from __future__ import annotations

import logging

from qdrant_client.http.models import PointStruct

from .. import config, embed, memory, notify
from ..scheduler import Job

log = logging.getLogger("artoo.cron.memory_resync")

SRC = "claude_memories"
DST = "claude_memories_v2"


async def run() -> None:
    client = memory.client()

    # Collect all UUIDs already in v2 so we know what to skip
    v2_ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=DST,
            limit=500,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        for pt in points:
            v2_ids.add(str(pt.id))
        if offset is None:
            break

    # Scan v1 for UUIDs not yet in v2
    new_points: list[PointStruct] = []
    skipped_empty = 0
    failed = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=SRC,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for pt in points:
            if str(pt.id) in v2_ids:
                continue
            text = (pt.payload or {}).get("text", "")
            if not text:
                skipped_empty += 1
                continue
            try:
                vec = embed.embed(text)
            except Exception as e:  # noqa: BLE001
                log.error("embed failed for %s: %s", str(pt.id)[:8], e)
                failed += 1
                continue
            new_points.append(
                PointStruct(id=pt.id, vector=vec, payload=pt.payload or {})
            )
        if offset is None:
            break

    if new_points:
        # Chunk upserts so we don't push a huge request
        for i in range(0, len(new_points), 50):
            client.upsert(collection_name=DST, points=new_points[i : i + 50])

    summary = (
        f"🧠 memory_resync: {len(new_points)} new point(s) migrated v1→v2"
        + (f", {skipped_empty} empty skipped" if skipped_empty else "")
        + (f", {failed} embed failures" if failed else "")
    )
    log.info(summary)
    if new_points or failed:
        await notify.to_telegram(summary)


JOB = Job(
    name="memory_resync",
    schedule="0 4 * * *",  # 4:00 AM — after memory_inventory at 3:30
    fn=run,
    timeout=600,
)
