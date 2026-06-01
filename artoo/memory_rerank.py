"""Rerank memory candidates: good recall but poor top-1 ranking (recall@5=0.90,
recall@1=0.35 baseline, 2026-05-31) — over-fetch by vector, then re-sort.

Local default → cloud failover → identity (the operator's call): the local llama3.1
on the embed host does the ranking for free/on-box; if it's down we fall over
to a cheap OpenRouter model; if BOTH fail we return the original vector order
(degrade-safe — never worse than no reranking).
"""
from __future__ import annotations

import json
import logging
import re

import httpx

from . import config, runtime

_log = logging.getLogger("artoo.memory_rerank")

# Local reranker = a small CROSS-ENCODER at config.RERANK_URL (TEI-style
# /rerank). bge-reranker-base (~280MB) fits the 3060 Ti alongside mxbai and
# reranks in ~10ms. We do NOT use an 8B LLM locally — it spilled to CPU on the
# 8GB card and timed out (bench 2026-05-31). If RERANK_URL is unset, the local
# path fast-fails and the cloud reranker carries it.
# Cloud reranker = Llama-3.3-70B (deepinfra, ZDR). Won a 5-model shootout
# (2026-05-31): recall@1=0.65, MRR=0.781, 0 fails, $0.20/1k — best quality AND
# 18x cheaper than GLM, which failed 7/20. (8B actively misranked below baseline;
# Kimi/405B worse or unavailable.) Effective default until a local cross-encoder
# exists; cloud is fast + ~18¢/mo so the local box is low priority.
_CLOUD_MODEL = "meta-llama/llama-3.3-70b-instruct"
_SNIPPET_CHARS = 300

_PROMPT = """You are ranking memory snippets by how well each ANSWERS the query.
Return ONLY a JSON array of snippet numbers, most-relevant first, e.g. [3,0,5,1].
Include every number exactly once. No prose.

QUERY: {query}

SNIPPETS:
{snippets}"""


def _build_prompt(query: str, cands: list[dict]) -> str:
    snippets = "\n".join(f"[{i}] {(c.get('text') or '')[:_SNIPPET_CHARS]}" for i, c in enumerate(cands))
    return _PROMPT.format(query=query, snippets=snippets)


def _parse_order(text: str, n: int) -> list[int] | None:
    m = re.search(r"\[[\d,\s]+\]", text or "")
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    order = [int(x) for x in arr if isinstance(x, int) and 0 <= x < n]
    return order or None


def _local_order(query: str, cands: list[dict]) -> list[int] | None:
    """Local cross-encoder rerank via a TEI-style /rerank endpoint. Fast-fails
    (returns None) if no endpoint is configured, so the cloud path carries it
    with no latency tax."""
    if not config.RERANK_URL:
        return None
    try:
        texts = [(c.get("text") or "")[:_SNIPPET_CHARS] for c in cands]
        r = httpx.post(
            f"{config.RERANK_URL.rstrip('/')}/rerank",
            json={"query": query, "texts": texts},
            timeout=10,
        )
        r.raise_for_status()
        # TEI returns [{"index": i, "score": s}, ...]; sort by score desc.
        scored = r.json()
        order = [int(x["index"]) for x in sorted(scored, key=lambda x: x["score"], reverse=True)
                 if isinstance(x.get("index"), int) and 0 <= x["index"] < len(cands)]
        return order or None
    except Exception as e:  # noqa: BLE001 — local down → cloud failover
        _log.warning("local cross-encoder rerank failed (%s) — cloud failover", e)
        return None


def _cloud_order(query: str, cands: list[dict]) -> list[int] | None:
    r = runtime.openrouter(_build_prompt(query, cands), model=_CLOUD_MODEL, max_tokens=512)
    if not r.ok or not r.text:
        _log.warning("cloud rerank failed (%s) — identity fallback", r.error)
        return None
    return _parse_order(r.text, len(cands))


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Re-sort `candidates` by relevance to `query`; return the best `top_k`.
    local → cloud → identity. Never raises; never returns fewer than min(top_k,
    len) items."""
    if len(candidates) <= 1:
        return candidates[:top_k]
    order = _local_order(query, candidates) or _cloud_order(query, candidates)
    if not order:
        return candidates[:top_k]  # identity: keep vector order
    # Dedupe, keeping first occurrence: a model can emit the same index twice
    # (observed). Without this the duplicate occupies a slot and silently pushes
    # a distinct, relevant memory out of the top_k window.
    seen: set[int] = set()
    order = [i for i in order if not (i in seen or seen.add(i))]
    ranked = [candidates[i] for i in order]
    ranked += [c for i, c in enumerate(candidates) if i not in seen]  # append any dropped
    return ranked[:top_k]
