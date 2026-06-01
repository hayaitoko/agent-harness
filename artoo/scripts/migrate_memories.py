"""One-shot migration: copy points from `claude_memories_v2` →
`memories`, preserving UUIDs and payloads.

Idempotent: if target point count matches source, exits cleanly. Safe to
re-run after new writes land in the source (overwriting same-UUID points
in the target).

Usage:
    .venv/bin/python -m artoo.scripts.migrate_memories
    .venv/bin/python -m artoo.scripts.migrate_memories --dry-run
    .venv/bin/python -m artoo.scripts.migrate_memories --source claude_memories_v2 --target memories
"""
from __future__ import annotations

import argparse
import sys
import time

from qdrant_client.http.models import Distance, PointStruct, VectorParams

from .. import config, memory

DEFAULT_SOURCE = "claude_memories_v2"
DEFAULT_TARGET = "memories"
BATCH = 128


def _count(client, name: str) -> int:
    return client.count(name).count


def _ensure_target(client, source: str, target: str) -> None:
    names = {c.name for c in client.get_collections().collections}
    if target in names:
        print(f"{target}: exists (skipping create)")
        return
    src_info = client.get_collection(source)
    vp = src_info.config.params.vectors
    print(
        f"creating {target} (size={vp.size}, distance={vp.distance.value}, "
        f"on_disk_payload={src_info.config.params.on_disk_payload})"
    )
    client.create_collection(
        collection_name=target,
        vectors_config=VectorParams(size=vp.size, distance=vp.distance),
        on_disk_payload=src_info.config.params.on_disk_payload,
    )


def _copy(client, source: str, target: str) -> int:
    offset = None
    copied = 0
    pending: list[PointStruct] = []
    while True:
        points, offset = client.scroll(
            collection_name=source,
            limit=BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for pt in points:
            pending.append(PointStruct(id=pt.id, vector=pt.vector, payload=pt.payload or {}))
        if pending:
            client.upsert(collection_name=target, points=pending)
            copied += len(pending)
            pending = []
            if copied % 100 < BATCH:
                print(f"  copied {copied} points")
        if offset is None:
            break
    return copied


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    client = memory.client()

    names = {c.name for c in client.get_collections().collections}
    if args.source not in names:
        print(f"source collection {args.source!r} does not exist", file=sys.stderr)
        return 2

    src_n = _count(client, args.source)
    tgt_n = _count(client, args.target) if args.target in names else 0
    print(f"{args.source}: {src_n} points")
    print(f"{args.target}: {tgt_n} points{' (does not exist)' if args.target not in names else ''}")

    if args.target in names and tgt_n == src_n:
        print(f"all {src_n} points already migrated")
        return 0

    if args.dry_run:
        print(f"dry-run: would copy {src_n - tgt_n} new/changed point(s)")
        return 0

    _ensure_target(client, args.source, args.target)
    t0 = time.monotonic()
    copied = _copy(client, args.source, args.target)
    dt = time.monotonic() - t0

    final_src = _count(client, args.source)
    final_tgt = _count(client, args.target)
    print(f"done in {dt:.1f}s — copied {copied} point(s)")
    print(f"final: {args.source}={final_src}, {args.target}={final_tgt}")
    if final_tgt != final_src:
        print("WARNING: counts differ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
