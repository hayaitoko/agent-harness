"""Weekly near-duplicate detection.

Walks every active Qdrant point; for each, vector-searches for
neighbors above DUPLICATE_THRESHOLD cosine similarity. Pairs with one
side dominantly newer / more useful surface to the operator's Telegram via an
inline keyboard (Forget A / Forget B / Merge / Ignore).

The 0.95 default is conservative — mxbai-1024 cosine pairs at that
level are essentially "the same memory phrased two ways". Tunable via
config.DUPLICATE_THRESHOLD if it proves too tight/loose in practice.

State is persisted in DATA_DIR/duplicate_review.json so callbacks can
resolve after a restart and "Ignore" entries don't re-surface for
IGNORE_TTL_DAYS.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from .. import config, memory, notify
from ..scheduler import Job

log = logging.getLogger("artoo.cron.duplicate_digest")

# Cosine similarity threshold above which two memories are flagged as
# near-duplicates. Conservative — see module docstring.
DUPLICATE_THRESHOLD = 0.95

# Cap the digest at a sane size — beyond ~12 pairs the Telegram message
# becomes unscannable. Remaining pairs roll over to the next week.
MAX_PAIRS_PER_DIGEST = 12

# How many neighbors to fetch per anchor. With 387 points and a 0.95
# threshold the typical pair count is small; 3 leaves headroom without
# burning Qdrant cycles.
NEIGHBORS_PER_ANCHOR = 3

# Ignored pairs stay out of the digest for this many days. Long enough
# that the user isn't pestered, short enough that genuine borderline
# pairs eventually re-surface for another look.
IGNORE_TTL_DAYS = 30

REVIEW_STATE_FILE = "duplicate_review.json"


def _state_path() -> Path:
    """Path used by both the cron writer and the telegram callback
    reader. Lives under DATA_DIR so it persists across restarts."""
    return config.DATA_DIR / REVIEW_STATE_FILE


async def run() -> None:
    state = _load_state()
    _expire_old_ignores(state)

    client = memory.client()
    # Pull every active point with its vector — we need the embedding
    # to find neighbors. Archived points are deliberately excluded;
    # they're already in the "out of search" state.
    active_points: list[tuple[str, list[float], dict]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=True,
            scroll_filter=memory._active_filter(),
        )
        for pt in points:
            vec = pt.vector
            # Some collections return dict-of-vectors when there are
            # multiple named vector spaces. claude_memories_v2 is single-
            # vector so this is a flat list — guard just in case.
            if isinstance(vec, dict):
                vec = next(iter(vec.values()), None)
            if vec is None:
                continue
            active_points.append((str(pt.id), list(vec), pt.payload or {}))
        if offset is None:
            break

    pairs = _find_duplicate_pairs(client, active_points, state)
    if not pairs:
        log.info("duplicate_digest: scanned %d points, no new pairs above %.2f",
                 len(active_points), DUPLICATE_THRESHOLD)
        return

    state["pending"].update({p["pair_id"]: p for p in pairs})
    _save_state(state)

    digest = _format_digest(pairs)
    keyboards = _format_keyboards(pairs)
    await notify.to_telegram_with_keyboards(digest, keyboards)
    log.info("duplicate_digest: surfaced %d pair(s) (over threshold %.2f)",
             len(pairs), DUPLICATE_THRESHOLD)


def _find_duplicate_pairs(
    client,
    active_points: list[tuple[str, list[float], dict]],
    state: dict,
) -> list[dict]:
    """For each active point, query for its top-N neighbors and emit
    pairs above DUPLICATE_THRESHOLD. Deduplicates so each pair appears
    only once. Skips pairs already in pending or recently ignored.
    """
    seen: set[tuple[str, str]] = set()
    ignored = set(state.get("ignored", {}).keys())
    pending_ids = set(state.get("pending", {}).keys())
    pairs: list[dict] = []

    for anchor_id, vec, anchor_payload in active_points:
        if len(pairs) >= MAX_PAIRS_PER_DIGEST:
            break
        try:
            response = client.query_points(
                collection_name=config.QDRANT_COLLECTION,
                query=vec,
                limit=NEIGHBORS_PER_ANCHOR + 1,  # +1 because self matches
                with_payload=True,
                with_vectors=False,
                query_filter=memory._active_filter(),
            )
        except Exception as e:  # noqa: BLE001
            log.warning("neighbor search failed for %s: %s", anchor_id[:8], e)
            continue
        for hit in response.points:
            other_id = str(hit.id)
            if other_id == anchor_id:
                continue
            if hit.score < DUPLICATE_THRESHOLD:
                continue
            pair_key = tuple(sorted([anchor_id, other_id]))
            if pair_key in seen:
                continue
            pair_id = f"{pair_key[0][:8]}-{pair_key[1][:8]}"
            if pair_id in pending_ids or pair_id in ignored:
                seen.add(pair_key)
                continue
            seen.add(pair_key)
            pairs.append({
                "pair_id": pair_id,
                "a": _summarize_for_digest(pair_key[0], _find_payload(active_points, pair_key[0]) or anchor_payload),
                "b": _summarize_for_digest(pair_key[1], _find_payload(active_points, pair_key[1]) or hit.payload or {}),
                "cosine": round(float(hit.score), 4),
                "queued_at": _now_iso(),
            })
            if len(pairs) >= MAX_PAIRS_PER_DIGEST:
                break
    return pairs


def _find_payload(active_points: list[tuple[str, list[float], dict]], uuid: str) -> Optional[dict]:
    for uid, _, payload in active_points:
        if uid == uuid:
            return payload
    return None


def _summarize_for_digest(uuid: str, payload: dict) -> dict:
    text = (payload or {}).get("text", "")
    return {
        "uuid": uuid,
        "title": _derive_title(payload),
        "snippet": (text[:140] + "…") if len(text) > 140 else text,
        "created_at": (payload or {}).get("created_at", ""),
    }


def _derive_title(payload: dict) -> str:
    """Prefer payload.title, fall back to first line of text."""
    if not payload:
        return "(no title)"
    title = payload.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()[:80]
    text = payload.get("text", "")
    if not isinstance(text, str):
        return "(no text)"
    first = text.strip().split("\n", 1)[0]
    return (first[:80] + ("…" if len(first) > 80 else "")) or "(empty)"


def _format_digest(pairs: list[dict]) -> str:
    lines = [
        f"🧹 *Memory hygiene — {len(pairs)} near-duplicate pair(s)*",
        f"_Threshold: cosine ≥ {DUPLICATE_THRESHOLD}_",
        "",
    ]
    for i, p in enumerate(pairs, 1):
        lines.append(f"*{i}. {p['cosine']:.4f}* — `{p['pair_id']}`")
        lines.append(f"  A `{p['a']['uuid'][:8]}` ({p['a']['created_at'][:10]}): {p['a']['title']}")
        lines.append(f"  B `{p['b']['uuid'][:8]}` ({p['b']['created_at'][:10]}): {p['b']['title']}")
        lines.append("")
    return "\n".join(lines)


def _format_keyboards(pairs: list[dict]) -> list[list[list[tuple[str, str]]]]:
    """Return a list of inline-keyboard specifications, one per pair.
    Each spec is a list of rows; each row is a list of (label, callback_data).
    The telegram layer translates these into InlineKeyboardMarkup objects.
    """
    keyboards: list[list[list[tuple[str, str]]]] = []
    for p in pairs:
        keyboards.append([
            [
                (f"Forget A ({p['a']['uuid'][:8]})", f"dup:forget_a:{p['pair_id']}"),
                (f"Forget B ({p['b']['uuid'][:8]})", f"dup:forget_b:{p['pair_id']}"),
            ],
            [
                ("Merge → A", f"dup:merge:{p['pair_id']}"),
                ("Ignore 30d", f"dup:ignore:{p['pair_id']}"),
            ],
        ])
    return keyboards


# ── State persistence ──────────────────────────────────────────────

def _load_state() -> dict:
    path = _state_path()
    if not path.exists():
        return {"pending": {}, "ignored": {}}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        log.warning("duplicate_review.json corrupt; resetting (%s)", e)
        return {"pending": {}, "ignored": {}}


def _save_state(state: dict) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(state, indent=2))
    except OSError as e:
        log.warning("duplicate_review.json write failed: %s", e)


def _expire_old_ignores(state: dict) -> None:
    """Drop ignore entries older than IGNORE_TTL_DAYS so borderline
    pairs eventually re-surface."""
    cutoff = time.time() - IGNORE_TTL_DAYS * 86400
    ignored = state.get("ignored", {})
    fresh = {k: v for k, v in ignored.items() if v.get("ts", 0) >= cutoff}
    if len(fresh) != len(ignored):
        state["ignored"] = fresh


def _now_iso() -> str:
    import datetime
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(tzinfo=None)
        .isoformat()
        + "Z"
    )


# ── Callback resolution (used by telegram channel) ─────────────────

def resolve_action(action: str, pair_id: str) -> dict:
    """Apply a user-chosen action against the pending pair. Called from
    the telegram CallbackQueryHandler. Returns a dict describing what
    happened so the channel can format the confirmation."""
    state = _load_state()
    pair = state.get("pending", {}).get(pair_id)
    if not pair:
        return {"ok": False, "error": f"pair {pair_id} not pending — maybe already resolved"}

    result: dict = {"ok": True, "pair_id": pair_id, "action": action}

    if action == "forget_a":
        outcome = memory.delete_memory(pair["a"]["uuid"], hard=True)
        result["deleted"] = outcome
        result["kept"] = pair["b"]["uuid"]
    elif action == "forget_b":
        outcome = memory.delete_memory(pair["b"]["uuid"], hard=True)
        result["deleted"] = outcome
        result["kept"] = pair["a"]["uuid"]
    elif action == "merge":
        # Merge policy: keep A (the older / first-discovered side by
        # sort order in the pair), hard-delete B. The boss can do a
        # smarter merge through save_memory if it wants the content
        # consolidated — this is a fast "B was a near-dup of A" path.
        outcome = memory.delete_memory(pair["b"]["uuid"], hard=True)
        result["deleted"] = outcome
        result["kept"] = pair["a"]["uuid"]
        result["merged_into"] = pair["a"]["uuid"]
    elif action == "ignore":
        state.setdefault("ignored", {})[pair_id] = {
            "ts": time.time(),
            "a": pair["a"]["uuid"],
            "b": pair["b"]["uuid"],
        }
    else:
        return {"ok": False, "error": f"unknown action: {action}"}

    state.get("pending", {}).pop(pair_id, None)
    _save_state(state)
    return result


JOB = Job(
    name="duplicate_digest",
    schedule="0 6 * * 0",  # 06:00 every Sunday
    fn=run,
    timeout=900,
)
