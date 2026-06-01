"""Rerank must stay degrade-safe AND never corrupt the result set: a model that
emits a duplicate index must not push a distinct memory out of the top_k window
(regression for the 2026-05-31 duplicate-index bug)."""
from __future__ import annotations

from artoo import memory_rerank as mr


def _cands(n: int) -> list[dict]:
    return [{"uuid": str(i), "text": f"t{i}"} for i in range(n)]


def test_rerank_dedupes_duplicate_indices(monkeypatch):
    monkeypatch.setattr(mr, "_local_order", lambda q, c: None)
    monkeypatch.setattr(mr, "_cloud_order", lambda q, c: [3, 0, 3, 1])  # dup 3
    out = mr.rerank("q", _cands(8), top_k=5)
    ids = [c["uuid"] for c in out]
    assert len(ids) == len(set(ids)), f"duplicate in result: {ids}"
    assert len(out) == 5  # still a full window
    assert ids[:3] == ["3", "0", "1"]  # dup collapsed, order preserved


def test_rerank_identity_on_no_order(monkeypatch):
    monkeypatch.setattr(mr, "_local_order", lambda q, c: None)
    monkeypatch.setattr(mr, "_cloud_order", lambda q, c: None)
    out = mr.rerank("q", _cands(6), top_k=3)
    assert [c["uuid"] for c in out] == ["0", "1", "2"]  # vector order kept


def test_rerank_never_fewer_than_min(monkeypatch):
    monkeypatch.setattr(mr, "_local_order", lambda q, c: None)
    monkeypatch.setattr(mr, "_cloud_order", lambda q, c: [1])  # partial order
    out = mr.rerank("q", _cands(4), top_k=4)
    assert len(out) == 4  # the dropped candidates are appended, not lost
    assert {c["uuid"] for c in out} == {"0", "1", "2", "3"}
