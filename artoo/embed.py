"""Text embedding via Ollama on the GPU host.

Single endpoint, single model, single function. Returns a list of floats
matching EMBED_DIM. Raises on transport failure — callers handle retries
if they want them; for the steady state of "boss queries memory", just
propagate.
"""
from __future__ import annotations

import math

import httpx

from . import config

# mxbai-embed-large has a 512-token context window. Token density varies a
# lot (natural English ≈ 4 chars/token, but dense/structured text — hex UUIDs,
# MAC addresses, paths, punctuation — runs closer to 2 chars/token). At the old
# 1500-char cap a dense ~1300-char memory tokenizes past 512 and Ollama 400s
# ("the input length exceeds the context length"), killing the save. 1000 chars
# (~500 tokens worst case; empirically 1024 chars of dense text still embeds
# clean) keeps a single chunk inside the window. Long texts get chunked +
# mean-pooled.
_MAX_CHARS = 1000


def embed(text: str) -> list[float]:
    """Return the 1024-dim embedding for `text`.

    Chunks + mean-pools (then renormalizes for cosine distance) when text
    exceeds the model's context window. Chunks are embedded individually so
    a single oversized chunk doesn't kill the whole call.
    """
    if len(text) <= _MAX_CHARS:
        try:
            return _embed_one(text)
        except Exception:
            # Even under the char cap, exceptionally dense text can tokenize
            # past the 512-token window (Ollama 400s). Fall through to the
            # chunked path and embed smaller pieces rather than losing the
            # whole save by propagating the error.
            pass

    chunks = [text[i : i + _MAX_CHARS] for i in range(0, len(text), _MAX_CHARS)]
    vecs: list[list[float]] = []
    for chunk in chunks:
        try:
            vecs.append(_embed_one(chunk))
        except Exception:
            # Skip oversized/bad chunks rather than failing the whole text.
            continue
    if not vecs:
        # Last resort — truncate well under the window and try once. Halving
        # the cap guards against a chunk that's still too dense at full size.
        return _embed_one(text[: _MAX_CHARS // 2])
    return _mean_pool_normalize(vecs)


def _embed_one(text: str) -> list[float]:
    r = httpx.post(
        f"{config.OLLAMA_HOST}/api/embed",
        json={"model": config.EMBED_MODEL, "input": text},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    embeddings = data.get("embeddings") or []
    if not embeddings:
        raise RuntimeError(f"empty embedding response: {data}")
    return embeddings[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in one Ollama call. All texts must fit the model's
    window — caller is responsible for chunking long ones (use `embed()`)."""
    r = httpx.post(
        f"{config.OLLAMA_HOST}/api/embed",
        json={"model": config.EMBED_MODEL, "input": texts},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    embeddings = data.get("embeddings") or []
    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"embed_batch returned {len(embeddings)} vectors for {len(texts)} inputs"
        )
    return embeddings


def _mean_pool_normalize(vecs: list[list[float]]) -> list[float]:
    """Average a list of vectors and L2-normalize (for cosine distance)."""
    if not vecs:
        raise ValueError("cannot pool empty list")
    n = len(vecs)
    dim = len(vecs[0])
    mean = [sum(v[i] for v in vecs) / n for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in mean))
    if norm > 0:
        mean = [x / norm for x in mean]
    return mean

