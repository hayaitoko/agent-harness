"""SQLite-backed conversation history.

One row per turn (user or assistant). Channel adapters key on a stable
conversation_id (telegram chat_id, discord channel_id, etc.).

Session partitioning
--------------------
Turns belong to a `session` — a row in the `sessions` table. /new closes
the current session (optionally tagging it with an AI-generated title) and
opens a fresh one. History queries filter to the currently-open session, so
archived sessions stay in the DB for review but don't pollute new context.

Concurrency
-----------
WAL mode + busy_timeout=5000ms is set per connection so concurrent crons,
schedulers, and Telegram handlers don't deadlock on writes. Schema +
migration run exactly once at module import; subsequent `_conn()` calls
are cheap.
"""
from __future__ import annotations

import sqlite3
import threading
import time

from . import config

DB_PATH = config.DATA_DIR / "artoo.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_conv_idx ON turns(conversation_id, id);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    title TEXT,
    started_at REAL NOT NULL,
    ended_at REAL
);
CREATE INDEX IF NOT EXISTS sessions_conv_idx ON sessions(conversation_id, id);
"""

# Composite indexes that match the real read predicates. `turns` reads
# always filter by (conversation_id, session_id) and order by id;
# `sessions` reads find the open session via (conversation_id) where
# ended_at IS NULL. Without these the hot path is a full scan once the
# tables grow.
_POST_MIGRATION_INDEXES = """
CREATE INDEX IF NOT EXISTS turns_conv_session_idx
    ON turns(conversation_id, session_id, id);
CREATE INDEX IF NOT EXISTS sessions_open_idx
    ON sessions(conversation_id, ended_at);
"""

_init_lock = threading.Lock()
_initialized = False


def _initialize() -> None:
    """Run schema + migration exactly once. Subsequent _conn() calls skip this."""
    global _initialized
    with _init_lock:
        if _initialized:
            return
        c = sqlite3.connect(DB_PATH)
        try:
            c.executescript(_SCHEMA)
            # Migration: add session_id column to turns if missing. SQLite has no
            # idempotent "ADD COLUMN IF NOT EXISTS" — try the ALTER and catch
            # the error when it's already there. On a fresh ALTER, backfill any
            # orphan rows (NULL session_id) into a per-conversation legacy
            # session so pre-migration history isn't blacked out from the
            # next query.
            try:
                c.execute(
                    "ALTER TABLE turns ADD COLUMN session_id INTEGER REFERENCES sessions(id)"
                )
                now = time.time()
                orphans = c.execute(
                    "SELECT DISTINCT conversation_id FROM turns WHERE session_id IS NULL"
                ).fetchall()
                for (conv,) in orphans:
                    cursor = c.execute(
                        "INSERT INTO sessions (conversation_id, title, started_at, ended_at) "
                        "VALUES (?, ?, ?, ?)",
                        (conv, "(legacy — pre-session)", now, now),
                    )
                    legacy_sid = cursor.lastrowid
                    c.execute(
                        "UPDATE turns SET session_id = ? "
                        "WHERE conversation_id = ? AND session_id IS NULL",
                        (legacy_sid, conv),
                    )
                c.commit()
            except sqlite3.OperationalError:
                # Column already exists — migration already ran on a prior boot.
                pass
            c.executescript(_POST_MIGRATION_INDEXES)
            c.commit()
        finally:
            c.close()
        _initialized = True


def _conn() -> sqlite3.Connection:
    """Open a connection with pragmas set for safe concurrent use.

    WAL lets readers and writers run concurrently. busy_timeout makes
    competing writes wait up to 5s rather than instant-fail with
    `database is locked`. foreign_keys actually enforces the FK on
    `turns.session_id → sessions.id`.

    Callers are responsible for closing the connection (use `try/finally`
    or `contextlib.closing`); the `with _conn() as c:` form does NOT
    close — it only commits/rolls back.
    """
    if not _initialized:
        _initialize()
    c = sqlite3.connect(DB_PATH)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=5000")
    c.execute("PRAGMA synchronous=NORMAL")
    c.execute("PRAGMA foreign_keys=ON")
    return c


# Run schema + migration on first import so the cost is paid once.
_initialize()


def _current_session_id(conversation_id: str, conn: sqlite3.Connection) -> int | None:
    """Return the open session id for a conversation, or None if there is none.

    Read-side helper — does NOT create a session. Read paths that find no
    open session return empty history; writes use `_ensure_session` instead.
    """
    row = conn.execute(
        "SELECT id FROM sessions "
        "WHERE conversation_id = ? AND ended_at IS NULL "
        "ORDER BY id DESC LIMIT 1",
        (conversation_id,),
    ).fetchone()
    return row[0] if row else None


def _ensure_session(conversation_id: str, conn: sqlite3.Connection) -> int:
    """Return the open session id, creating one if none is open.

    Write-side helper used by `append()` so the first turn of a new
    conversation (or the first turn after /new) automatically opens a
    session without the caller having to coordinate.
    """
    sid = _current_session_id(conversation_id, conn)
    if sid is not None:
        return sid
    cursor = conn.execute(
        "INSERT INTO sessions (conversation_id, started_at) VALUES (?, ?)",
        (conversation_id, time.time()),
    )
    return cursor.lastrowid


def new_session(conversation_id: str, title: str | None = None) -> int:
    """Close the current session (tagging it with `title`) and open a new one.

    The title is applied to the session being CLOSED — it describes what
    just happened, not what's about to start. Returns the id of the newly
    opened (untitled) session.
    """
    c = _conn()
    try:
        with c:
            c.execute(
                "UPDATE sessions SET ended_at = ?, title = ? "
                "WHERE conversation_id = ? AND ended_at IS NULL",
                (time.time(), title, conversation_id),
            )
            cursor = c.execute(
                "INSERT INTO sessions (conversation_id, started_at) VALUES (?, ?)",
                (conversation_id, time.time()),
            )
            return cursor.lastrowid
    finally:
        c.close()


def clear_session(conversation_id: str) -> int:
    """Delete the current session's turns and the session row itself.

    Unlike `new_session`, nothing is archived — the conversation is gone.
    Archived (already-ended) sessions are untouched. Returns the number of
    turns deleted; 0 if there was no open session.
    """
    c = _conn()
    try:
        with c:
            sid = _current_session_id(conversation_id, c)
            if sid is None:
                return 0
            deleted = c.execute(
                "DELETE FROM turns WHERE conversation_id = ? AND session_id = ?",
                (conversation_id, sid),
            ).rowcount
            c.execute("DELETE FROM sessions WHERE id = ?", (sid,))
            return deleted
    finally:
        c.close()


def session_stats(conversation_id: str) -> dict:
    """Return {current_index, total} for sessions of a conversation.

    current_index: 1-based ordinal of the currently-open session (0 if none).
    total:         total session rows (archived + current).
    """
    c = _conn()
    try:
        total = c.execute(
            "SELECT COUNT(*) FROM sessions WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()[0]
        sid = _current_session_id(conversation_id, c)
        if sid is None:
            return {"current_index": 0, "total": total}
        current_index = c.execute(
            "SELECT COUNT(*) FROM sessions "
            "WHERE conversation_id = ? AND id <= ?",
            (conversation_id, sid),
        ).fetchone()[0]
        return {"current_index": current_index, "total": total}
    finally:
        c.close()


def raw_history(conversation_id: str, limit: int = 10) -> list[tuple[str, str]]:
    """Return the current session's last `limit` turns as (role, content) tuples.

    Used by /new to feed recent turns into title generation before archiving
    the session. Returns chronological order (oldest first). Empty list if
    no session is open or the session has no turns.
    """
    c = _conn()
    try:
        sid = _current_session_id(conversation_id, c)
        if sid is None:
            return []
        rows = c.execute(
            "SELECT role, content FROM turns "
            "WHERE conversation_id = ? AND session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (conversation_id, sid, limit),
        ).fetchall()
        return list(reversed(rows))
    finally:
        c.close()


def history(conversation_id: str, limit: int = 30) -> list[dict]:
    """Return last `limit` turns of the CURRENT session, in chronological order.

    Pre-session history (archived sessions) is intentionally not returned —
    /new isolates conversations so old topics don't bleed into new ones.
    """
    c = _conn()
    try:
        sid = _current_session_id(conversation_id, c)
        if sid is None:
            return []
        rows = c.execute(
            "SELECT role, content FROM turns "
            "WHERE conversation_id = ? AND session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (conversation_id, sid, limit),
        ).fetchall()
        return [{"role": r, "content": ct} for r, ct in reversed(rows)]
    finally:
        c.close()


def history_tokens_with_stats(
    conversation_id: str, max_tokens: int = 180_000
) -> tuple[list[dict], dict]:
    """Return (turns, stats) where turns fit within max_tokens.

    Scoped to the CURRENT session — archived sessions are excluded. Uses a
    3 chars/token estimate (safer than 4 for code/logs/JSON where tokens
    pack more densely). Turns are returned in chronological order (oldest
    first).

    Stats dict:
        kept:       number of turns returned
        total:      total number of turns in the current session
        dropped:    True if any turns were trimmed off the oldest end
        est_tokens: estimated token count of the returned (active) window

    Edge case: if a single turn exceeds the entire budget, that one turn is
    still returned so callers don't get an empty context blackout from one
    huge message. `dropped` will still be True in that case.
    """
    max_chars = max_tokens * 3
    c = _conn()
    try:
        sid = _current_session_id(conversation_id, c)
        if sid is None:
            rows: list[tuple[str, str]] = []
        else:
            # Newest-first so we can trim from the oldest end.
            rows = c.execute(
                "SELECT role, content FROM turns "
                "WHERE conversation_id = ? AND session_id = ? "
                "ORDER BY id DESC",
                (conversation_id, sid),
            ).fetchall()
    finally:
        c.close()

    total = len(rows)

    # Walk newest → oldest, accumulating char budget.
    budget = 0
    kept: list[tuple[str, str]] = []
    for role, content in rows:
        chunk = len(content) + 20  # +20 for role label and formatting overhead
        if budget + chunk > max_chars:
            break
        budget += chunk
        kept.append((role, content))

    # Fallback: if the newest turn alone blew the budget, keep it anyway.
    # Without this, a single oversized message would erase the entire
    # active window.
    if not kept and rows:
        role, content = rows[0]
        kept.append((role, content))
        budget = len(content) + 20

    stats = {
        "kept": len(kept),
        "total": total,
        "dropped": len(kept) < total,
        "est_tokens": budget // 3,
    }
    return [{"role": r, "content": ct} for r, ct in reversed(kept)], stats


def history_tokens(conversation_id: str, max_tokens: int = 180_000) -> list[dict]:
    """Return turns that fit within max_tokens, newest-first truncation.

    Uses a 3 chars/token estimate — safer than 4 for code/logs/JSON where
    tokens pack more densely. Turns are returned in chronological order
    (oldest first). Thin wrapper around `history_tokens_with_stats` for
    callers that don't need overflow detection.
    """
    turns, _ = history_tokens_with_stats(conversation_id, max_tokens)
    return turns


def context_stats(conversation_id: str, max_tokens: int = 180_000) -> dict:
    """Return stats for the ACTIVE WINDOW (what Claude actually sees).

    Reports the trimmed window — not the raw row count — so the numbers
    surfaced to the user match what's sent to the model. Delegates to
    `history_tokens_with_stats` to keep the trimming logic single-sourced.
    """
    _, stats = history_tokens_with_stats(conversation_id, max_tokens)
    return stats


def append(conversation_id: str, role: str, content: str) -> None:
    c = _conn()
    try:
        with c:
            session_id = _ensure_session(conversation_id, c)
            c.execute(
                "INSERT INTO turns (conversation_id, role, content, created_at, session_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (conversation_id, role, content, time.time(), session_id),
            )
    finally:
        c.close()
