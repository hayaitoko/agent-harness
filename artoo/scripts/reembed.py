"""One-shot re-embed: migrate claude_memories (legacy 384-dim) →
claude_memories_v2 (mxbai-embed-large 1024-dim) preserving point UUIDs.

Idempotent: if v2 exists, upserts overwrite same-UUID points. Safe to re-run
when new memories land in the source collection.

Run with: .venv/bin/python -m artoo.scripts.reembed
"""
from __future__ import annotations

import time

from qdrant_client.http.models import Distance, PointStruct, VectorParams

from .. import config, embed, memory

SRC = "claude_memories"
DST = "claude_memories_v2"
BATCH = 25  # ollama can do bigger batches but 25 keeps GPU memory low


def main() -> None:
    client = memory.client()

    existing = {c.name for c in client.get_collections().collections}
    if DST not in existing:
        print(f"creating {DST} ({config.EMBED_DIM}-dim, Cosine)...")
        client.create_collection(
            collection_name=DST,
            vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE),
        )
    else:
        print(f"{DST} exists; upserting into it")

    offset = None
    total = 0
    embedded = 0
    skipped = 0
    failed = 0
    pending: list[PointStruct] = []

    while True:
        points, offset = client.scroll(
            collection_name=SRC,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for pt in points:
            total += 1
            text = (pt.payload or {}).get("text", "")
            if not text:
                skipped += 1
                continue
            try:
                # embed() auto-chunks long texts internally
                vec = embed.embed(text)
            except Exception as e:
                print(f"  embed fail {str(pt.id)[:8]}: {e}")
                failed += 1
                continue
            pending.append(PointStruct(id=pt.id, vector=vec, payload=pt.payload or {}))
            embedded += 1

            if len(pending) >= BATCH:
                client.upsert(collection_name=DST, points=pending)
                print(f"  upserted {embedded}/{total} ({skipped} empty, {failed} failed)")
                pending = []

        if offset is None:
            break

    if pending:
        client.upsert(collection_name=DST, points=pending)

    print(f"done: total={total}, embedded={embedded}, skipped_empty={skipped}, failed={failed}")


if __name__ == "__main__":
    t0 = time.monotonic()
    main()
    print(f"total time: {time.monotonic() - t0:.1f}s")
