"""Nightly memory maintenance: three-way reconcile across Qdrant + vault.

Replaces the older `memory_inventory` cron, which only filled missing
link-nodes. Hygiene does everything `memory_inventory` did plus:

  - **Orphan page sweep.** Pages whose UUID is no longer in Qdrant get
    deleted. Stale residue from hard-deletes or migrations.
  - **Orphan link-node sweep.** Link-nodes whose UUID is no longer in
    Qdrant get deleted. The vault's link graph stays in lockstep with
    the source of truth.
  - **Backlink scrub.** Every link-node body is filtered for lines that
    name UUIDs no longer in Qdrant — removes dead references the graph
    walker would otherwise chase to nothing.
  - **uuid-map rebuild.** `_index/uuid-map.md` is rewritten from current
    Qdrant state instead of being append-only. Stays accurate forever.

All destructive operations target *derived* state (vault files) not
the source of truth (Qdrant points). The cron will never silently
delete a Qdrant point — that path goes through `memory.delete_memory`
with explicit caller intent. Pages without Qdrant points are derived
state from a prior hard-delete that didn't cascade fully; sweeping
them now closes the loop.

If something pathological is happening upstream, the per-run delete
cap (`_MAX_DELETIONS_PER_RUN`) prevents a runaway sweep. Anything over
the cap survives until next night, with a Telegram notice so the operator
can investigate.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .. import config, memory, notify
from ..scheduler import Job

log = logging.getLogger("artoo.cron.memory_hygiene")

_MAX_DELETIONS_PER_RUN = 200  # belt-and-suspenders against runaway sweeps


async def run() -> None:
    vault = config.OBSIDIAN_VAULT
    link_dir = vault / "link-nodes"
    index_dir = vault / "_index"
    vault.mkdir(parents=True, exist_ok=True)
    link_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: enumerate Qdrant ground truth — UUID set + obsidian paths.
    client = memory.client()
    qdrant_uuids: set[str] = set()
    uuid_to_path: dict[str, str] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for pt in points:
            uid = str(pt.id)
            qdrant_uuids.add(uid)
            payload = pt.payload or {}
            path = payload.get("obsidian_path", "")
            if path:
                uuid_to_path[uid] = path
        if offset is None:
            break

    # Phase 2: fill missing link-nodes (old memory_inventory behavior).
    nodes_created = 0
    for uid in qdrant_uuids:
        link_file = link_dir / f"{uid}.md"
        if link_file.exists():
            continue
        if nodes_created >= _MAX_DELETIONS_PER_RUN:
            break
        try:
            link_file.touch()
            nodes_created += 1
        except OSError as e:
            log.error("link-node touch failed for %s: %s", uid, e)

    # Phase 3: sweep orphan pages (page exists but Qdrant point gone).
    pages_deleted, pages_capped = _sweep_orphan_pages(vault, qdrant_uuids)

    # Phase 4: sweep orphan link-nodes.
    nodes_deleted, nodes_capped = _sweep_orphan_link_nodes(link_dir, qdrant_uuids)

    # Phase 5: scrub dead UUIDs from every link-node body.
    backlinks_scrubbed = _scrub_dead_backlinks(link_dir, qdrant_uuids)

    # Phase 6: rebuild uuid-map from current state.
    _rebuild_uuid_map(index_dir, uuid_to_path)

    summary = (
        f"🧠 memory_hygiene: {len(qdrant_uuids)} Qdrant points · "
        f"+{nodes_created} new link-node(s) · "
        f"−{pages_deleted} orphan page(s) · "
        f"−{nodes_deleted} orphan link-node(s) · "
        f"{backlinks_scrubbed} link-node(s) had dead UUIDs scrubbed"
    )
    if pages_capped or nodes_capped:
        summary += f" · ⚠ deletion cap {_MAX_DELETIONS_PER_RUN} reached, more remain"
    log.info(summary)
    if nodes_created or pages_deleted or nodes_deleted or backlinks_scrubbed or pages_capped or nodes_capped:
        await notify.to_telegram(summary)


def _sweep_orphan_pages(vault: Path, qdrant_uuids: set[str]) -> tuple[int, bool]:
    """Delete page files whose UUID isn't in Qdrant. Page filename
    convention: `<date>.<slug>.<uuid>.md`. The UUID is the last
    dot-separated segment before .md."""
    deleted = 0
    capped = False
    for page in vault.glob("*.md"):
        # uuid-map and other index files are top-level *.md too — but
        # they live in _index/ and don't end with a UUID. Pages match
        # the date.slug.uuid.md shape.
        parts = page.stem.split(".")
        if len(parts) < 3:
            continue
        candidate = parts[-1]
        if not memory._UUID_RE.fullmatch(candidate):
            continue
        if candidate in qdrant_uuids:
            continue
        if deleted >= _MAX_DELETIONS_PER_RUN:
            capped = True
            break
        try:
            page.unlink()
            deleted += 1
        except OSError as e:
            log.warning("orphan page delete failed: %s: %s", page.name, e)
    return deleted, capped


def _sweep_orphan_link_nodes(link_dir: Path, qdrant_uuids: set[str]) -> tuple[int, bool]:
    """Delete link-node files whose UUID isn't in Qdrant."""
    deleted = 0
    capped = False
    for node in link_dir.glob("*.md"):
        uid = node.stem
        if not memory._UUID_RE.fullmatch(uid):
            continue
        if uid in qdrant_uuids:
            continue
        if deleted >= _MAX_DELETIONS_PER_RUN:
            capped = True
            break
        try:
            node.unlink()
            deleted += 1
        except OSError as e:
            log.warning("orphan link-node delete failed: %s: %s", node.name, e)
    return deleted, capped


def _scrub_dead_backlinks(link_dir: Path, qdrant_uuids: set[str]) -> int:
    """Walk every link-node body and drop lines referencing UUIDs that
    no longer exist in Qdrant. Returns the count of files modified.

    Lines with [[wikilinks]] but no UUID (external Notion refs etc.)
    are left untouched — the regex match anchors on UUIDs.
    """
    modified = 0
    for node in link_dir.glob("*.md"):
        try:
            body = node.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        if not body.strip():
            continue
        kept_lines: list[str] = []
        had_dead = False
        for line in body.splitlines():
            referenced = memory._UUID_RE.findall(line)
            if referenced and any(u not in qdrant_uuids for u in referenced):
                # If every UUID in the line is dead, drop the line.
                if all(u not in qdrant_uuids for u in referenced):
                    had_dead = True
                    continue
                # Mixed live + dead on one line — rare; conservatively
                # keep the line so we don't break wikilink context.
            kept_lines.append(line)
        if not had_dead:
            continue
        new_body = "\n".join(kept_lines)
        if kept_lines and not new_body.endswith("\n"):
            new_body += "\n"
        try:
            node.write_text(new_body)
            modified += 1
        except OSError as e:
            log.warning("backlink scrub write failed: %s: %s", node.name, e)
    return modified


def _rebuild_uuid_map(index_dir: Path, uuid_to_path: dict[str, str]) -> None:
    """Rewrite _index/uuid-map.md from current Qdrant state. The old
    file was append-only and accumulated stale entries forever."""
    map_path = index_dir / "uuid-map.md"
    lines = ["# UUID Map"]
    for uid in sorted(uuid_to_path):
        lines.append(f"{uid} | {uuid_to_path[uid]}")
    try:
        map_path.write_text("\n".join(lines) + "\n")
    except OSError as e:
        log.warning("uuid-map rebuild failed: %s", e)


JOB = Job(
    name="memory_hygiene",
    schedule="30 3 * * *",  # 03:30 nightly — same slot as the old memory_inventory
    fn=run,
    timeout=900,
)
