"""Memory tool: qdrant vector search + obsidian link graph traversal.

As of v1.5 (re-embed cutover, 2026-05-14): search uses real semantic vectors
against `claude_memories_v2` (mxbai-embed-large, 1024-dim, Cosine). Keyword
scroll+filter survives as a fallback for when the embed service is down.

The obsidian vault at OBSIDIAN_VAULT is a link graph over Qdrant content,
not the data store. Per-memory layout:
  - qdrant point: the content + 1024-dim vector
  - obsidian page at VAULT/<date>.<slug>.<uuid>.md (human-readable)
  - link-node at VAULT/link-nodes/<uuid>.md (other UUIDs = related memories)

The filename of a link-node IS that memory's UUID; the file's contents are
UUIDs of related memories (with optional [[wikilinks]] to external refs).
search_memory follows the link graph one hop.

Hygiene (2026-05-20): memories carry a `status` field on the payload —
"active" by default, "archived" for soft-deleted. Search filters
archived points out by default; pass include_archived=True to recall
them. Hard delete cascades across Qdrant + page + link-node + scrubs
inbound backlinks + rebuilds the uuid-map index. Last-retrieval is
tracked in `last_retrieved_at` for the staleness_sweep cron.
"""
import datetime
import logging
import re
import uuid as _uuid
from typing import Optional
from urllib.parse import urlparse

from qdrant_client import QdrantClient
from qdrant_client.http.models import (FieldCondition, Filter, MatchValue,
                                       PointStruct)

from . import config, embed

_log = logging.getLogger("artoo.memory")

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_LINKED_CAP = 10  # max linked memories to surface — boss has limited context

STATUS_ACTIVE = "active"
STATUS_ARCHIVED = "archived"

_client: Optional[QdrantClient] = None


def client() -> QdrantClient:
    """QdrantClient with port resolved from URL scheme (https→443).

    qdrant-client defaults `port` to 6333 even when the URL specifies https,
    so we infer it explicitly to talk to TLS-fronted (Cloudflare-tunneled)
    Qdrant endpoints.
    """
    global _client
    if _client is None:
        parsed = urlparse(config.QDRANT_URL)
        port = parsed.port or (443 if parsed.scheme == "https" else 6333)
        _client = QdrantClient(
            url=config.QDRANT_URL,
            port=port,
            api_key=config.QDRANT_API_KEY or None,
            check_compatibility=False,
        )
    return _client


def search_memory(query: str, limit: int = 5, *, include_archived: bool = False) -> dict:
    """Semantic search via mxbai-embed-large + walk obsidian link graph.

    Returns:
        {
          "primary": [Memory, ...],  # vector-search hits, with `score`
          "linked":  [Memory, ...],  # related via obsidian link-nodes
        }
    where Memory = {uuid, text, tags, created_at, obsidian_path, score?, status?}

    Archived memories (status == STATUS_ARCHIVED) are filtered out by
    default — pass include_archived=True for explicit recall. Falls back
    to keyword scroll+filter if the embed service is unavailable so a
    downed Ollama doesn't take the agent offline.

    Side-effect: bumps `last_retrieved_at` on every primary hit so the
    staleness_sweep cron can identify cold memories.
    """
    # Vector recall is strong but top-1 ranking is weak (recall@5=0.90 vs
    # recall@1=0.35, 2026-05-31), so over-fetch and rerank to surface the best
    # at the top. ARTOO_MEMORY_RERANK=0 disables (falls back to plain top-`limit`).
    rerank_on = (config.optional("ARTOO_MEMORY_RERANK") or "1").strip().lower() \
        not in ("0", "false", "no", "off")
    fetch_n = max(limit * 4, 20) if rerank_on else limit
    try:
        primary = _vector_search(query, fetch_n, include_archived=include_archived)
    except Exception as e:
        _log.warning("vector search failed (%s) — falling back to keyword scroll", e)
        primary = _keyword_scroll(query, fetch_n, include_archived=include_archived)
    if rerank_on and len(primary) > limit:
        from . import memory_rerank
        primary = memory_rerank.rerank(query, primary, limit)
    else:
        primary = primary[:limit]

    linked: list[dict] = []
    seen = {p["uuid"] for p in primary}
    for hit in primary:
        if len(linked) >= _LINKED_CAP:
            break
        for linked_uuid in _follow_links(hit["uuid"]):
            if linked_uuid in seen:
                continue
            seen.add(linked_uuid)
            point = _get_point(linked_uuid)
            if not point:
                continue
            # Surface linked memories only if active — same default as
            # primary search. Callers wanting archived can re-fetch with
            # include_archived directly.
            if not include_archived and point.get("status") == STATUS_ARCHIVED:
                continue
            linked.append(point)
            if len(linked) >= _LINKED_CAP:
                break

    if primary:
        try:
            mark_retrieved([p["uuid"] for p in primary])
        except Exception as e:  # noqa: BLE001 — telemetry must not break search
            _log.warning("mark_retrieved failed: %s", e)

    return {"primary": primary, "linked": linked}


def _active_filter() -> Filter:
    """Qdrant filter excluding ONLY archived points; everything else passes,
    INCLUDING legacy points with no `status` field.

    BUG FIX (2026-05-31, caught by the retrieval eval): the old impl used
    `must=[MatchExcept(status != archived)]`, whose comment CLAIMED field-less
    points still match — they don't. In practice Qdrant's MatchExcept-in-`must`
    requires the field to exist, so every legacy memory without a `status`
    field (≈391 of 392) was silently filtered out of every search. `must_not`
    is the correct semantics: a missing field doesn't satisfy the
    `status == archived` condition, so the point is NOT excluded.
    """
    return Filter(must_not=[
        FieldCondition(key="status", match=MatchValue(value=STATUS_ARCHIVED)),
    ])


def _vector_search(query: str, limit: int, *, include_archived: bool = False) -> list[dict]:
    vec = embed.embed(query)
    response = client().query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=vec,
        limit=limit,
        with_payload=True,
        query_filter=None if include_archived else _active_filter(),
    )
    return [_format_point(pt, score=pt.score) for pt in response.points]


def _keyword_scroll(query: str, limit: int, *, include_archived: bool = False) -> list[dict]:
    terms = [t.lower() for t in query.split() if len(t) > 2]
    if not terms:
        return []

    matches: list[dict] = []
    offset = None
    pages = 0
    scroll_filter = None if include_archived else _active_filter()
    while pages < 20:  # cap scroll depth (≈2000 points)
        points, offset = client().scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
            scroll_filter=scroll_filter,
        )
        pages += 1
        for pt in points:
            text = (pt.payload or {}).get("text", "").lower()
            if any(term in text for term in terms):
                matches.append(_format_point(pt))
        if offset is None:
            break

    matches.sort(key=lambda m: _score(m, terms, query), reverse=True)
    return matches[:limit]


def _score(m: dict, terms: list[str], query: str) -> int:
    t = m["text"].lower()
    n = sum(1 for term in terms if term in t)
    if query.lower() in t:
        n += 10
    return n


def _follow_links(uuid: str) -> list[str]:
    """Return UUIDs linked to `uuid` via the obsidian link-node file.

    The link-node filename IS the memory's UUID. Its content is UUIDs of
    related memories (and optional [[wikilinks]] to external refs, which we
    ignore here). Returns linked UUIDs, excluding self.
    """
    link_file = config.OBSIDIAN_VAULT / "link-nodes" / f"{uuid}.md"
    if not link_file.exists():
        return []
    linked = set(_UUID_RE.findall(link_file.read_text()))
    linked.discard(uuid)
    return list(linked)


def _get_point(uuid: str) -> Optional[dict]:
    try:
        points = client().retrieve(
            collection_name=config.QDRANT_COLLECTION,
            ids=[uuid],
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        return None
    if not points:
        return None
    return _format_point(points[0])


def save_memory(
    text: str,
    *,
    title: Optional[str] = None,
    tags: Optional[list[str]] = None,
    linked_uuids: Optional[list[str]] = None,
) -> dict:
    """Save a new memory.

    Writes are: embed -> qdrant upsert -> obsidian page -> empty link-node
    file -> uuid-map index. Linked UUIDs get the new memory's UUID appended
    to their link-nodes (backlink-style, matching the legacy proxy pattern).

    The caller (usually the boss via MCP) is expected to provide title/tags/
    linked_uuids based on its judgment + a prior search_memory call. If omitted,
    sensible defaults are used (first line as title, no tags, no links).
    """
    mem_uuid = str(_uuid.uuid4())
    date_str = datetime.date.today().isoformat()
    title = (title or _default_title(text)).strip()
    tags = tags or []
    # Drop anything that isn't a real UUID — the boss occasionally hands us
    # garbage strings and we don't want them on the filesystem path.
    linked_uuids = [u for u in (linked_uuids or []) if _UUID_RE.fullmatch(u)]

    slug = _slugify(title) or "memory"
    obsidian_path = f"{date_str}.{slug}.{mem_uuid}.md"
    created_at = (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(tzinfo=None)
        .isoformat()
        + "Z"
    )

    vec = embed.embed(text)
    client().upsert(
        collection_name=config.QDRANT_COLLECTION,
        points=[
            PointStruct(
                id=mem_uuid,
                vector=vec,
                payload={
                    "text": text,
                    "tags": tags,
                    "created_at": created_at,
                    "last_retrieved_at": created_at,
                    "obsidian_path": obsidian_path,
                    "status": STATUS_ACTIVE,
                },
            )
        ],
    )

    vault = config.OBSIDIAN_VAULT
    vault.mkdir(parents=True, exist_ok=True)
    (vault / "link-nodes").mkdir(exist_ok=True)
    (vault / "_index").mkdir(exist_ok=True)

    # Obsidian page (human-readable, mirrors qdrant content)
    tag_line = " ".join(f"#{t}" for t in tags) if tags else ""
    page_body = f"<!-- uuid: {mem_uuid} -->\n\n# {title}\n\n{tag_line}\n\n{text}\n".strip() + "\n"
    (vault / obsidian_path).write_text(page_body)

    # Empty link-node — future writes that link TO this memory append here
    (vault / "link-nodes" / f"{mem_uuid}.md").touch()

    # Append to uuid-map index
    with open(vault / "_index" / "uuid-map.md", "a") as f:
        f.write(f"{mem_uuid} | {obsidian_path}\n")

    # Backlinks: each linked memory's link-node gets our UUID appended
    for linked in linked_uuids:
        target = vault / "link-nodes" / f"{linked}.md"
        if target.exists():
            with open(target, "a") as f:
                f.write(f"\n{mem_uuid}")

    _log.info("save_memory: %s '%s' tags=%s links=%d", mem_uuid[:8], title[:50], tags, len(linked_uuids))
    return {
        "uuid": mem_uuid,
        "obsidian_path": obsidian_path,
        "title": title,
        "tags": tags,
        "linked_to": linked_uuids,
    }


def delete_memory(uuid: str, *, hard: bool = False) -> dict:
    """Delete a memory.

    Soft delete (default): sets payload.status="archived" so search
    filters it out but the content survives for explicit recall.
    Reversible via restore_memory.

    Hard delete: cascades across all four surfaces — Qdrant point
    removed, page file unlinked, link-node unlinked, this UUID scrubbed
    from every other link-node's body. The uuid-map index is rebuilt
    on the next memory_hygiene run; we don't touch it here to keep this
    function cheap.

    Returns a small status dict so the caller can log what happened.
    """
    if not _UUID_RE.fullmatch(uuid):
        raise ValueError(f"not a valid UUID: {uuid!r}")

    if not hard:
        return _soft_archive(uuid)

    return _hard_delete(uuid)


def _soft_archive(uuid: str) -> dict:
    """Mark archived in Qdrant payload. Files on disk are left alone —
    the vault still surfaces them for the human, but search hides them
    unless include_archived is set."""
    point = _get_point(uuid)
    if not point:
        return {"uuid": uuid, "result": "not_found"}
    if point.get("status") == STATUS_ARCHIVED:
        return {"uuid": uuid, "result": "already_archived"}
    client().set_payload(
        collection_name=config.QDRANT_COLLECTION,
        points=[uuid],
        payload={
            "status": STATUS_ARCHIVED,
            "archived_at": _now_iso(),
        },
    )
    _log.info("delete_memory: soft-archived %s", uuid[:8])
    return {"uuid": uuid, "result": "archived"}


def _hard_delete(uuid: str) -> dict:
    """Cascading hard delete. Survives partial failure — each surface
    is wiped independently so a missing file doesn't block the Qdrant
    deletion (or vice versa). Returns a dict naming what was removed."""
    removed: dict[str, bool] = {
        "qdrant": False, "page": False, "link_node": False, "backlinks": 0,
    }

    # Qdrant point. Look up the page path before deleting so we can
    # remove the page file even if the payload has it.
    point = _get_point(uuid)
    obsidian_path = (point or {}).get("obsidian_path", "")
    try:
        client().delete(
            collection_name=config.QDRANT_COLLECTION,
            points_selector=[uuid],
        )
        removed["qdrant"] = True
    except Exception as e:  # noqa: BLE001
        _log.warning("hard_delete %s: qdrant delete failed: %s", uuid[:8], e)

    vault = config.OBSIDIAN_VAULT
    if obsidian_path:
        page = vault / obsidian_path
        if page.is_file():
            try:
                page.unlink()
                removed["page"] = True
            except OSError as e:
                _log.warning("hard_delete %s: page unlink failed: %s", uuid[:8], e)

    link_node = vault / "link-nodes" / f"{uuid}.md"
    if link_node.is_file():
        try:
            link_node.unlink()
            removed["link_node"] = True
        except OSError as e:
            _log.warning("hard_delete %s: link-node unlink failed: %s", uuid[:8], e)

    removed["backlinks"] = _scrub_backlinks(uuid)

    _log.info(
        "delete_memory: hard-deleted %s — qdrant=%s page=%s link_node=%s backlinks=%d",
        uuid[:8], removed["qdrant"], removed["page"], removed["link_node"],
        removed["backlinks"],
    )
    return {"uuid": uuid, "result": "hard_deleted", **removed}


def restore_memory(uuid: str) -> dict:
    """Reverse a soft-archive. Hard-deleted memories cannot be restored
    through this path — they'd need to be re-saved from scratch."""
    if not _UUID_RE.fullmatch(uuid):
        raise ValueError(f"not a valid UUID: {uuid!r}")
    point = _get_point(uuid)
    if not point:
        return {"uuid": uuid, "result": "not_found"}
    if point.get("status") != STATUS_ARCHIVED:
        return {"uuid": uuid, "result": "not_archived"}
    client().set_payload(
        collection_name=config.QDRANT_COLLECTION,
        points=[uuid],
        payload={"status": STATUS_ACTIVE, "restored_at": _now_iso()},
    )
    _log.info("restore_memory: restored %s", uuid[:8])
    return {"uuid": uuid, "result": "restored"}


def mark_retrieved(uuids: list[str]) -> None:
    """Bump last_retrieved_at on the named UUIDs. Called by search_memory
    for every primary hit. Failures are swallowed — this is telemetry
    for the staleness_sweep cron, never load-bearing."""
    if not uuids:
        return
    now = _now_iso()
    for uuid in uuids:
        if not _UUID_RE.fullmatch(uuid):
            continue
        try:
            client().set_payload(
                collection_name=config.QDRANT_COLLECTION,
                points=[uuid],
                payload={"last_retrieved_at": now},
            )
        except Exception as e:  # noqa: BLE001
            _log.debug("mark_retrieved %s failed: %s", uuid[:8], e)


def _scrub_backlinks(deleted_uuid: str) -> int:
    """Walk every link-node and drop lines referencing `deleted_uuid`.
    Returns the number of link-node files modified. Cheap on a few-
    hundred-file vault; if this ever gets slow at scale we can index
    backlinks separately, but for now linear is fine.
    """
    vault = config.OBSIDIAN_VAULT
    link_dir = vault / "link-nodes"
    if not link_dir.is_dir():
        return 0
    modified = 0
    for path in link_dir.glob("*.md"):
        try:
            body = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        if deleted_uuid not in body:
            continue
        # Drop lines that mention the deleted UUID. Conservatively keep
        # the line if it contains another UUID we shouldn't lose — but
        # given the file format (one UUID per line, optionally followed
        # by a wikilink), that's rare. Filter line-by-line for safety.
        kept = [
            line for line in body.splitlines()
            if deleted_uuid not in line
        ]
        new_body = "\n".join(kept)
        if kept and not new_body.endswith("\n"):
            new_body += "\n"
        if new_body != body:
            try:
                path.write_text(new_body)
                modified += 1
            except OSError as e:
                _log.warning("backlink scrub failed for %s: %s", path.name, e)
    return modified


def _now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(tzinfo=None)
        .isoformat()
        + "Z"
    )


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60]


def _default_title(text: str) -> str:
    first_line = text.strip().split("\n", 1)[0]
    return first_line[:80].strip()


def _format_point(pt, score: Optional[float] = None) -> dict:
    p = pt.payload or {}
    out = {
        "uuid": str(pt.id),
        "text": p.get("text", ""),
        "tags": p.get("tags", []),
        "created_at": p.get("created_at", ""),
        "last_retrieved_at": p.get("last_retrieved_at", ""),
        "obsidian_path": p.get("obsidian_path", ""),
        "status": p.get("status", STATUS_ACTIVE),
    }
    if score is not None:
        out["score"] = round(float(score), 4)
    return out
