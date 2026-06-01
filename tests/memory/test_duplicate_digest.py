"""Tests for the duplicate-digest pair finder + action resolver."""
import json
from pathlib import Path

import pytest

from artoo.crons import duplicate_digest

from .conftest import make_uuid, seed_memory


def test_pair_finder_emits_each_pair_once(fake_qdrant, vault, monkeypatch):
    # Two near-identical vectors → one pair.
    v_a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    v_b = [0.99, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0]
    a = seed_memory(fake_qdrant, vault, text="A version", vector=v_a)
    b = seed_memory(fake_qdrant, vault, text="B version", vector=v_b)

    pairs = duplicate_digest._find_duplicate_pairs(
        fake_qdrant,
        [(uid, fake_qdrant.points[uid].vector, fake_qdrant.points[uid].payload)
         for uid in [a, b]],
        {"pending": {}, "ignored": {}},
    )
    pair_keys = {tuple(sorted([p["a"]["uuid"], p["b"]["uuid"]])) for p in pairs}
    assert len(pairs) == 1
    assert pair_keys == {tuple(sorted([a, b]))}


def test_pair_finder_skips_below_threshold(fake_qdrant, vault):
    v_a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    v_distant = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    a = seed_memory(fake_qdrant, vault, vector=v_a)
    b = seed_memory(fake_qdrant, vault, vector=v_distant)
    pairs = duplicate_digest._find_duplicate_pairs(
        fake_qdrant,
        [(uid, fake_qdrant.points[uid].vector, fake_qdrant.points[uid].payload)
         for uid in [a, b]],
        {"pending": {}, "ignored": {}},
    )
    assert pairs == []


def test_pair_finder_skips_pending_and_ignored(fake_qdrant, vault):
    v = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    a = seed_memory(fake_qdrant, vault, vector=v)
    b = seed_memory(fake_qdrant, vault, vector=v)
    pair_id = f"{sorted([a, b])[0][:8]}-{sorted([a, b])[1][:8]}"
    state = {"pending": {pair_id: {}}, "ignored": {}}
    pairs = duplicate_digest._find_duplicate_pairs(
        fake_qdrant,
        [(uid, fake_qdrant.points[uid].vector, fake_qdrant.points[uid].payload)
         for uid in [a, b]],
        state,
    )
    assert pairs == []


def test_resolve_action_forget_a_hard_deletes(fake_qdrant, vault, fixed_data_dir):
    a = seed_memory(fake_qdrant, vault, text="A")
    b = seed_memory(fake_qdrant, vault, text="B")
    pair_id = f"{sorted([a, b])[0][:8]}-{sorted([a, b])[1][:8]}"
    state_path = fixed_data_dir / "duplicate_review.json"
    state_path.write_text(json.dumps({
        "pending": {
            pair_id: {
                "pair_id": pair_id,
                "a": {"uuid": sorted([a, b])[0]},
                "b": {"uuid": sorted([a, b])[1]},
            },
        },
        "ignored": {},
    }))

    result = duplicate_digest.resolve_action("forget_a", pair_id)
    assert result["ok"]
    # Pair removed from pending after resolution.
    final_state = json.loads(state_path.read_text())
    assert pair_id not in final_state["pending"]
    # The "A" side (sorted-first uuid) is gone.
    assert sorted([a, b])[0] not in fake_qdrant.points
    # The "B" side survives.
    assert sorted([a, b])[1] in fake_qdrant.points


def test_resolve_action_ignore_records_ttl_entry(fake_qdrant, vault, fixed_data_dir):
    a = seed_memory(fake_qdrant, vault)
    b = seed_memory(fake_qdrant, vault)
    pair_id = f"{sorted([a, b])[0][:8]}-{sorted([a, b])[1][:8]}"
    state_path = fixed_data_dir / "duplicate_review.json"
    state_path.write_text(json.dumps({
        "pending": {
            pair_id: {
                "pair_id": pair_id,
                "a": {"uuid": sorted([a, b])[0]},
                "b": {"uuid": sorted([a, b])[1]},
            },
        },
        "ignored": {},
    }))

    result = duplicate_digest.resolve_action("ignore", pair_id)
    assert result["ok"]
    state = json.loads(state_path.read_text())
    assert pair_id in state["ignored"]
    assert "ts" in state["ignored"][pair_id]


def test_resolve_action_unknown_pair(fake_qdrant, vault, fixed_data_dir):
    result = duplicate_digest.resolve_action("forget_a", "deadbeef-cafebabe")
    assert not result["ok"]
    assert "not pending" in result["error"]


def test_resolve_action_unknown_action(fake_qdrant, vault, fixed_data_dir):
    a = seed_memory(fake_qdrant, vault)
    b = seed_memory(fake_qdrant, vault)
    pair_id = f"{sorted([a, b])[0][:8]}-{sorted([a, b])[1][:8]}"
    (fixed_data_dir / "duplicate_review.json").write_text(json.dumps({
        "pending": {pair_id: {"pair_id": pair_id, "a": {"uuid": a}, "b": {"uuid": b}}},
        "ignored": {},
    }))
    result = duplicate_digest.resolve_action("zap", pair_id)
    assert not result["ok"]


def test_format_keyboards_callback_shape(fake_qdrant, vault):
    a = seed_memory(fake_qdrant, vault)
    b = seed_memory(fake_qdrant, vault)
    pair = {
        "pair_id": "aaaaaaaa-bbbbbbbb",
        "a": {"uuid": a, "title": "A", "snippet": "", "created_at": ""},
        "b": {"uuid": b, "title": "B", "snippet": "", "created_at": ""},
        "cosine": 0.97,
        "queued_at": "now",
    }
    keyboards = duplicate_digest._format_keyboards([pair])
    flat = [cb for row in keyboards[0] for (_, cb) in row]
    assert "dup:forget_a:aaaaaaaa-bbbbbbbb" in flat
    assert "dup:forget_b:aaaaaaaa-bbbbbbbb" in flat
    assert "dup:merge:aaaaaaaa-bbbbbbbb" in flat
    assert "dup:ignore:aaaaaaaa-bbbbbbbb" in flat
