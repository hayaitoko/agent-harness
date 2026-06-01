"""Tests for the monthly staleness sweep."""
import asyncio
import datetime
from pathlib import Path

import pytest

from artoo.crons import staleness_sweep

from .conftest import seed_memory


def _run_sweep(monkeypatch):
    async def _noop(text):  # noqa: ARG001
        return None
    from artoo import notify
    monkeypatch.setattr(notify, "to_telegram", _noop)
    asyncio.run(staleness_sweep.run())


def _iso(days_ago: int) -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        - datetime.timedelta(days=days_ago)
    ).isoformat() + "Z"


def test_archives_points_older_than_threshold(fake_qdrant, vault, monkeypatch):
    old_uid = seed_memory(
        fake_qdrant, vault,
        last_retrieved_at=_iso(days_ago=staleness_sweep.STALENESS_DAYS + 30),
    )
    fresh_uid = seed_memory(
        fake_qdrant, vault,
        last_retrieved_at=_iso(days_ago=10),
    )
    _run_sweep(monkeypatch)
    assert fake_qdrant.points[old_uid].payload["status"] == "archived"
    assert fake_qdrant.points[fresh_uid].payload["status"] == "active"


def test_falls_back_to_created_at_when_last_retrieved_missing(fake_qdrant, vault, monkeypatch):
    """A point with no last_retrieved_at but an old created_at gets archived."""
    uid = seed_memory(
        fake_qdrant, vault,
        last_retrieved_at="",  # explicit empty
        created_at=_iso(days_ago=staleness_sweep.STALENESS_DAYS + 30),
    )
    # Manually wipe last_retrieved_at so the anchor falls back to created_at.
    fake_qdrant.points[uid].payload.pop("last_retrieved_at", None)
    _run_sweep(monkeypatch)
    assert fake_qdrant.points[uid].payload["status"] == "archived"


def test_grandfathers_points_with_no_anchor(fake_qdrant, vault, monkeypatch):
    """A point with neither field set must survive — staleness has no
    signal to decide on, so do nothing."""
    uid = seed_memory(fake_qdrant, vault)
    fake_qdrant.points[uid].payload.pop("last_retrieved_at", None)
    fake_qdrant.points[uid].payload.pop("created_at", None)
    _run_sweep(monkeypatch)
    assert fake_qdrant.points[uid].payload.get("status", "active") == "active"


def test_skips_already_archived(fake_qdrant, vault, monkeypatch):
    """Archived points must not be touched by the sweep — they're
    filtered out by _active_filter at the query layer."""
    archived_uid = seed_memory(
        fake_qdrant, vault,
        status="archived",
        last_retrieved_at=_iso(days_ago=staleness_sweep.STALENESS_DAYS + 30),
    )
    _run_sweep(monkeypatch)
    # Status stays archived (no second archived_at overwrite or similar churn).
    assert fake_qdrant.points[archived_uid].payload["status"] == "archived"


def test_respects_max_archives_cap(fake_qdrant, vault, monkeypatch):
    monkeypatch.setattr(staleness_sweep, "MAX_ARCHIVES_PER_RUN", 2)
    for _ in range(5):
        seed_memory(
            fake_qdrant, vault,
            last_retrieved_at=_iso(days_ago=staleness_sweep.STALENESS_DAYS + 30),
        )
    _run_sweep(monkeypatch)
    archived = sum(
        1 for p in fake_qdrant.points.values()
        if p.payload.get("status") == "archived"
    )
    assert archived == 2
