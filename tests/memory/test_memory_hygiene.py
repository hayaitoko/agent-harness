"""Tests for the three-way reconcile cron."""
import asyncio
from pathlib import Path

import pytest

from artoo.crons import memory_hygiene

from .conftest import make_uuid, seed_memory


def _run_cron(monkeypatch):
    """Helper: skip the Telegram notify side-effect."""
    async def _stub_notify(text):  # noqa: ARG001
        return None
    from artoo import notify
    monkeypatch.setattr(notify, "to_telegram", _stub_notify)
    asyncio.run(memory_hygiene.run())


def test_creates_missing_link_node(fake_qdrant, vault, monkeypatch):
    uid = seed_memory(fake_qdrant, vault, write_link_node=False)
    assert not (vault / "link-nodes" / f"{uid}.md").exists()
    _run_cron(monkeypatch)
    assert (vault / "link-nodes" / f"{uid}.md").exists()


def test_sweeps_orphan_page(fake_qdrant, vault, monkeypatch):
    """Page on disk with no Qdrant point → deleted."""
    orphan_uid = make_uuid()
    orphan_page = vault / f"2026-01-01.orphan.{orphan_uid}.md"
    orphan_page.write_text("orphan content")
    seed_memory(fake_qdrant, vault)  # keep something active so cron has work
    _run_cron(monkeypatch)
    assert not orphan_page.exists()


def test_sweeps_orphan_link_node(fake_qdrant, vault, monkeypatch):
    """link-node on disk with no Qdrant point → deleted."""
    orphan_uid = make_uuid()
    orphan_ln = vault / "link-nodes" / f"{orphan_uid}.md"
    orphan_ln.write_text("")
    seed_memory(fake_qdrant, vault)
    _run_cron(monkeypatch)
    assert not orphan_ln.exists()


def test_scrubs_dead_uuid_from_link_node_body(fake_qdrant, vault, monkeypatch):
    keeper = seed_memory(fake_qdrant, vault, text="kept")
    dead_uuid = make_uuid()  # never seeded into Qdrant
    keeper_ln = vault / "link-nodes" / f"{keeper}.md"
    keeper_ln.write_text(f"{dead_uuid}\nlive-reference\n")

    _run_cron(monkeypatch)

    body = keeper_ln.read_text()
    assert dead_uuid not in body
    assert "live-reference" in body  # non-UUID content preserved


def test_rebuilds_uuid_map_from_current_state(fake_qdrant, vault, monkeypatch):
    a = seed_memory(fake_qdrant, vault, text="a")
    b = seed_memory(fake_qdrant, vault, text="b")
    # Seed the index with a stale extra line that should not survive.
    (vault / "_index" / "uuid-map.md").write_text(
        "# UUID Map\n00000000-0000-0000-0000-000000000000 | stale.md\n"
    )

    _run_cron(monkeypatch)

    map_text = (vault / "_index" / "uuid-map.md").read_text()
    assert a in map_text
    assert b in map_text
    assert "00000000-0000-0000-0000-000000000000" not in map_text


def test_keeps_pages_when_qdrant_point_exists(fake_qdrant, vault, monkeypatch):
    uid = seed_memory(fake_qdrant, vault)
    page = vault / fake_qdrant.points[uid].payload["obsidian_path"]
    assert page.exists()
    _run_cron(monkeypatch)
    assert page.exists()  # not orphaned, must survive
