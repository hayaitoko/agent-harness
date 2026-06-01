"""Per-chat key/value preferences (boss model override and future settings).

Backed by SQLite in the same artoo.db that storage.py uses, so the
chat_prefs table travels with the rest of the agent state.

Initial use: boss model preference per chat. /model in Telegram writes
here, and the channel adapter reads here when no slash-override is
present on a turn. Slash-overrides (e.g. /opus) still win for one
turn at a time — they don't touch this store.

Schema is intentionally generic so future settings (per-chat persona,
per-chat reasoning effort, etc.) can live here too.

Concurrency
-----------
WAL mode + busy_timeout=5000ms is set per connection. Schema runs once
at import. The bot's event loop and crons both hit this; without these
pragmas, /model under load can throw `database is locked`.
"""
from __future__ import annotations

import sqlite3
import threading

from . import config

_DB_PATH = config.DATA_DIR / "artoo.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_prefs (
    chat_id TEXT NOT NULL,
    key     TEXT NOT NULL,
    value   TEXT,
    PRIMARY KEY (chat_id, key)
)
"""

_init_lock = threading.Lock()
_initialized = False


def _initialize() -> None:
    global _initialized
    with _init_lock:
        if _initialized:
            return
        con = sqlite3.connect(_DB_PATH)
        try:
            con.execute(_SCHEMA)
            con.commit()
        finally:
            con.close()
        _initialized = True


def _conn() -> sqlite3.Connection:
    if not _initialized:
        _initialize()
    con = sqlite3.connect(_DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=5000")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


_initialize()


def get(chat_id: str, key: str) -> str | None:
    con = _conn()
    try:
        row = con.execute(
            "SELECT value FROM chat_prefs WHERE chat_id = ? AND key = ?",
            (chat_id, key),
        ).fetchone()
        return row[0] if row else None
    finally:
        con.close()


def set(chat_id: str, key: str, value: str | None) -> None:  # noqa: A001 — shadowing builtin
    """Set or clear a preference. Pass value=None to delete."""
    con = _conn()
    try:
        if value is None:
            con.execute(
                "DELETE FROM chat_prefs WHERE chat_id = ? AND key = ?",
                (chat_id, key),
            )
        else:
            con.execute(
                "INSERT INTO chat_prefs (chat_id, key, value) VALUES (?, ?, ?) "
                "ON CONFLICT(chat_id, key) DO UPDATE SET value = excluded.value",
                (chat_id, key, value),
            )
        con.commit()
    finally:
        con.close()


# ──── Convenience for the boss-model preference ────────────────────────────

_MODEL_KEY = "model"


def get_model(chat_id: str) -> str | None:
    """Return the per-chat boss model preference, or None if unset."""
    return get(chat_id, _MODEL_KEY)


def set_model(chat_id: str, model: str | None) -> None:
    """Set the per-chat boss model preference. Pass None to clear (revert to global default)."""
    set(chat_id, _MODEL_KEY, model)
