"""Tests for the soft/hard delete cascade + restore + backlink scrub."""
from pathlib import Path

import pytest

from artoo import memory
from .conftest import seed_memory


def test_soft_delete_sets_archived_status(fake_qdrant, vault):
    uid = seed_memory(fake_qdrant, vault)
    result = memory.delete_memory(uid, hard=False)
    assert result["result"] == "archived"
    assert fake_qdrant.points[uid].payload["status"] == "archived"
    # Files still exist — soft archive doesn't touch the vault.
    assert (vault / fake_qdrant.points[uid].payload["obsidian_path"]).exists()
    assert (vault / "link-nodes" / f"{uid}.md").exists()


def test_soft_delete_idempotent(fake_qdrant, vault):
    uid = seed_memory(fake_qdrant, vault)
    memory.delete_memory(uid, hard=False)
    second = memory.delete_memory(uid, hard=False)
    assert second["result"] == "already_archived"


def test_hard_delete_cascades_to_all_surfaces(fake_qdrant, vault):
    uid = seed_memory(fake_qdrant, vault)
    obsidian_path = fake_qdrant.points[uid].payload["obsidian_path"]

    result = memory.delete_memory(uid, hard=True)
    assert result["result"] == "hard_deleted"
    assert result["qdrant"] is True
    assert result["page"] is True
    assert result["link_node"] is True
    assert uid not in fake_qdrant.points
    assert not (vault / obsidian_path).exists()
    assert not (vault / "link-nodes" / f"{uid}.md").exists()


def test_hard_delete_scrubs_backlinks(fake_qdrant, vault):
    target = seed_memory(fake_qdrant, vault, text="will be deleted")
    referrer_a = seed_memory(fake_qdrant, vault, text="references target")
    referrer_b = seed_memory(fake_qdrant, vault, text="also references target")
    # Wire the backlinks: referrers' link-nodes contain `target`.
    (vault / "link-nodes" / f"{referrer_a}.md").write_text(f"{target}\n")
    (vault / "link-nodes" / f"{referrer_b}.md").write_text(f"{target}\nsome-other-line\n")

    result = memory.delete_memory(target, hard=True)
    assert result["backlinks"] == 2
    assert target not in (vault / "link-nodes" / f"{referrer_a}.md").read_text()
    assert target not in (vault / "link-nodes" / f"{referrer_b}.md").read_text()
    # The unrelated line stays.
    assert "some-other-line" in (vault / "link-nodes" / f"{referrer_b}.md").read_text()


def test_delete_unknown_uuid_returns_not_found(fake_qdrant, vault):
    fake_uuid = "00000000-0000-0000-0000-000000000000"
    result = memory.delete_memory(fake_uuid, hard=False)
    assert result["result"] == "not_found"


def test_delete_rejects_invalid_uuid(fake_qdrant, vault):
    with pytest.raises(ValueError, match="not a valid UUID"):
        memory.delete_memory("not-a-uuid", hard=False)


def test_restore_flips_status_back(fake_qdrant, vault):
    uid = seed_memory(fake_qdrant, vault, status="archived")
    result = memory.restore_memory(uid)
    assert result["result"] == "restored"
    assert fake_qdrant.points[uid].payload["status"] == "active"


def test_restore_no_op_on_active(fake_qdrant, vault):
    uid = seed_memory(fake_qdrant, vault, status="active")
    result = memory.restore_memory(uid)
    assert result["result"] == "not_archived"


def test_restore_no_op_on_missing(fake_qdrant, vault):
    result = memory.restore_memory("11111111-1111-1111-1111-111111111111")
    assert result["result"] == "not_found"


def test_mark_retrieved_bumps_timestamp(fake_qdrant, vault):
    uid = seed_memory(fake_qdrant, vault, last_retrieved_at="2020-01-01T00:00:00Z")
    memory.mark_retrieved([uid])
    new_ts = fake_qdrant.points[uid].payload["last_retrieved_at"]
    assert new_ts > "2020-01-01T00:00:00Z"


def test_mark_retrieved_skips_invalid_uuid(fake_qdrant, vault):
    # Mixed valid + invalid — invalid is silently skipped.
    uid = seed_memory(fake_qdrant, vault, last_retrieved_at="2020-01-01T00:00:00Z")
    memory.mark_retrieved([uid, "garbage"])
    assert fake_qdrant.points[uid].payload["last_retrieved_at"] > "2020-01-01T00:00:00Z"


def test_search_memory_excludes_archived_by_default(fake_qdrant, vault, stub_embed):
    active = seed_memory(fake_qdrant, vault, text="active memory")
    seed_memory(fake_qdrant, vault, text="archived memory", status="archived")
    out = memory.search_memory("active memory", limit=5)
    uuids = {p["uuid"] for p in out["primary"]}
    assert active in uuids
    assert all(p["status"] == "active" for p in out["primary"])


def test_search_memory_include_archived_when_requested(fake_qdrant, vault, stub_embed):
    seed_memory(fake_qdrant, vault, text="active")
    archived_uid = seed_memory(fake_qdrant, vault, text="archived", status="archived")
    out = memory.search_memory("archived", limit=5, include_archived=True)
    uuids = {p["uuid"] for p in out["primary"]}
    assert archived_uid in uuids
