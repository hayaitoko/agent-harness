"""Shared fixtures for memory + cron tests.

We don't want real Qdrant calls in CI. FakeQdrant mimics the surface
artoo's memory module actually uses (scroll, query_points, retrieve,
upsert, delete, set_payload) with in-memory state. Tests can seed it
and assert side-effects without touching the network.
"""
from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import pytest


# ── FakeQdrant: minimal in-memory stand-in ──────────────────────────

@dataclass
class FakePoint:
    id: str
    payload: dict = field(default_factory=dict)
    vector: Optional[list[float]] = None
    score: Optional[float] = None


class FakeResponse:
    def __init__(self, points: list[FakePoint]) -> None:
        self.points = points


class FakeQdrant:
    """In-memory stand-in for qdrant_client.QdrantClient.

    Implements just the methods artoo's memory + crons actually call.
    Vector similarity is the dot-product of normalized vectors —
    sufficient to drive deterministic ordering in tests; we control the
    vectors directly so we don't need real cosine math.
    """
    def __init__(self) -> None:
        self.points: dict[str, FakePoint] = {}
        self.deletes: list[str] = []
        self.upserts: list[str] = []

    def add(self, uid: str, *, text: str = "", **payload_extra: Any) -> FakePoint:
        payload = {"text": text, **payload_extra}
        pt = FakePoint(id=uid, payload=payload, vector=payload_extra.get("vector"))
        self.points[uid] = pt
        return pt

    # ---- qdrant-client surface ----

    def scroll(self, collection_name=None, limit=200, offset=None,
               with_payload=True, with_vectors=False, scroll_filter=None):
        ids = sorted(self.points.keys())
        if offset is not None:
            start = ids.index(offset) if offset in ids else len(ids)
        else:
            start = 0
        end = min(start + limit, len(ids))
        page = [self.points[i] for i in ids[start:end]]
        if scroll_filter is not None:
            page = [p for p in page if _passes_active_filter(p, scroll_filter)]
        next_offset = ids[end] if end < len(ids) else None
        # Mirror what qdrant returns: points with payload as configured.
        prepared = []
        for p in page:
            np = FakePoint(id=p.id, payload=p.payload if with_payload else {},
                           vector=p.vector if with_vectors else None)
            prepared.append(np)
        return prepared, next_offset

    def query_points(self, collection_name=None, query=None, limit=5,
                     with_payload=True, with_vectors=False, query_filter=None):
        # Score = dot product of query vec and stored vec. Stable
        # deterministic ordering for tests. Any None vector ranks last.
        scored: list[FakePoint] = []
        for pt in self.points.values():
            if pt.vector is None or query is None:
                continue
            score = sum(a * b for a, b in zip(query, pt.vector))
            scored.append(FakePoint(id=pt.id, payload=pt.payload if with_payload else {},
                                    vector=None, score=score))
        scored.sort(key=lambda p: p.score, reverse=True)
        if query_filter is not None:
            scored = [p for p in scored if _passes_active_filter(p, query_filter)]
        return FakeResponse(points=scored[:limit])

    def retrieve(self, collection_name=None, ids=None, with_payload=True, with_vectors=False):
        out = []
        for uid in (ids or []):
            pt = self.points.get(str(uid))
            if not pt:
                continue
            out.append(FakePoint(id=pt.id, payload=pt.payload if with_payload else {},
                                 vector=pt.vector if with_vectors else None))
        return out

    def upsert(self, collection_name=None, points=None):
        for p in points or []:
            uid = str(p.id)
            self.points[uid] = FakePoint(
                id=uid,
                payload=dict(p.payload or {}),
                vector=list(p.vector) if p.vector is not None else None,
            )
            self.upserts.append(uid)

    def delete(self, collection_name=None, points_selector=None):
        for uid in points_selector or []:
            uid = str(uid)
            self.points.pop(uid, None)
            self.deletes.append(uid)

    def set_payload(self, collection_name=None, points=None, payload=None):
        for uid in points or []:
            uid = str(uid)
            if uid in self.points:
                merged = dict(self.points[uid].payload)
                merged.update(payload or {})
                self.points[uid].payload = merged


def _passes_active_filter(point: FakePoint, qfilter: Any) -> bool:
    """Mimic the _active_filter behavior — drop points whose status is
    explicitly archived. Anything else passes (including missing
    status, matching MatchExcept semantics)."""
    if qfilter is None:
        return True
    status = (point.payload or {}).get("status")
    return status != "archived"


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def fake_qdrant(monkeypatch) -> FakeQdrant:
    """Replace memory.client() with a FakeQdrant instance for the test."""
    fake = FakeQdrant()
    from artoo import memory
    monkeypatch.setattr(memory, "client", lambda: fake)
    monkeypatch.setattr(memory, "_client", fake, raising=False)
    return fake


@pytest.fixture
def vault(tmp_path: Path, monkeypatch) -> Path:
    """Redirect OBSIDIAN_VAULT to a tmpdir per test."""
    v = tmp_path / "vault"
    v.mkdir()
    (v / "link-nodes").mkdir()
    (v / "_index").mkdir()
    from artoo import config
    monkeypatch.setattr(config, "OBSIDIAN_VAULT", v)
    return v


@pytest.fixture
def fixed_data_dir(tmp_path: Path, monkeypatch) -> Path:
    """Redirect DATA_DIR (used by duplicate_digest review state)."""
    d = tmp_path / "data"
    d.mkdir()
    from artoo import config
    monkeypatch.setattr(config, "DATA_DIR", d)
    return d


@pytest.fixture
def stub_embed(monkeypatch):
    """Replace embed.embed with a deterministic stub so search_memory
    doesn't try to hit Ollama. Returns a unit-norm vector derived from
    a string hash."""
    def _stub(text: str) -> list[float]:
        h = hash(text) & 0xffff
        v = [0.0] * 8
        v[h % 8] = 1.0
        return v
    from artoo import embed
    monkeypatch.setattr(embed, "embed", _stub)
    return _stub


# ── Helpers tests can import ────────────────────────────────────────

def make_uuid() -> str:
    return str(_uuid.uuid4())


def seed_memory(fake: FakeQdrant, vault: Path, *, uid: Optional[str] = None,
                text: str = "test memory", status: str = "active",
                last_retrieved_at: Optional[str] = None,
                created_at: str = "2026-01-01T00:00:00Z",
                vector: Optional[list[float]] = None,
                write_page: bool = True, write_link_node: bool = True) -> str:
    """Add a memory to both Qdrant + vault. Returns the UUID."""
    uid = uid or make_uuid()
    obsidian_path = f"2026-01-01.test.{uid}.md"
    fake.add(
        uid,
        text=text,
        status=status,
        last_retrieved_at=last_retrieved_at or created_at,
        created_at=created_at,
        obsidian_path=obsidian_path,
        vector=vector or [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    if write_page:
        (vault / obsidian_path).write_text(f"# test\n{text}\n")
    if write_link_node:
        (vault / "link-nodes" / f"{uid}.md").write_text("")
    return uid
