# Memory layer

## What memory is

Artoo remembers things across conversations through a two-store system:

- **Qdrant** holds the structured data: text content, tags, timestamps,
  and a 1024-dim semantic vector per memory.
- **Obsidian vault** is a human-readable mirror plus a link graph. Each
  memory has a markdown page you can read in Obsidian, and a separate
  link-node file holding UUIDs of related memories.

The two stores are kept in sync by every write path. They serve
different roles: Qdrant powers retrieval; Obsidian powers human review
and graph traversal.

## The graph

Each memory has a UUID. The Obsidian vault is structured around that:

```
/mnt/artoo-vault/
  2026-05-14.urgent-list.<uuid>.md    # human-readable page
  link-nodes/<uuid>.md                # backlinks (UUIDs of related memories)
  _index/uuid-map.md                  # uuid → page-path manifest
```

A link-node file's content is a list of UUIDs (with optional
`[[wikilink]]` decorations for external resources like Notion pages).
Those UUIDs point at *other* memories that link TO this one. It's
a backward link graph — if memory A says "related to memory B", then
B's link-node file contains A's UUID, not the other way around.

`search_memory` follows the graph one hop: a query returns the
top-N vector matches (the "primary" results) plus any memories
backlinked from those (the "linked" results).

## Vector search

Embeddings come from Ollama running `mxbai-embed-large` on a GPU host.
- 1024-dim
- Cosine distance
- 512-token context window

For texts longer than the model's window we chunk into ~1500-char
slices, embed each chunk, mean-pool the vectors, then L2-normalize for
cosine. See `embed.py`. The chunked path skips bad chunks rather than
failing the whole text, so a single unicode hiccup doesn't kill the embedding.

## The Qdrant collection

`QDRANT_COLLECTION` (default `artoo_memories`) is the single collection
artoo uses. 1024-dim, mxbai-embed-large, Cosine. Both the boss and the
external Claude.ai MCP connector (via `mcp_server.py`) write here.

The two-collection migration (ADR-008) was retired 2026-05-15 once the
legacy v1 proxy had zero non-resync writers (see ADR-013). The
`memory_resync` cron is unregistered; `scripts/reembed.py` and
`scripts/migrate_memories.py` survive as historical one-shots.

## Read flow

```
search_memory(query, limit=5):
  vec = embed(query)
  primary = qdrant.query_points(collection, query=vec, limit=limit)
  for hit in primary:
    for linked_uuid in obsidian_link_nodes[hit.uuid]:
      linked.append(qdrant.retrieve(collection, linked_uuid))
      if len(linked) >= 10: break
  mark_retrieved(primary)   # bumps last_retrieved_at on each hit
  return { primary, linked }
```

Falls back to keyword scroll if Ollama is unreachable, so the agent
stays online even if the GPU host is down.

## Write flow

The boss calls `save_memory(text, title, tags, linked_uuids)`. Title /
tags / linked_uuids are LLM-judged values the boss decides AFTER a
`search_memory` to find what to link to. The save path:

1. `embed(text)` → vector
2. Upsert into Qdrant (UUID + vector + payload)
3. Write the Obsidian page at `<date>.<slug>.<uuid>.md`
4. Touch an empty link-node file for this UUID (so future links have an
   anchor)
5. Append a line to `_index/uuid-map.md`
6. For each `linked_uuids[i]`, append our UUID to that memory's
   link-node file (backlinking us into their graph)

The boss is responsible for the smart parts (title, tags, what to link
to). The Python code is responsible for the mechanical parts (embedding,
file I/O, atomicity).

## Maintenance crons

| Cron | Schedule | Purpose |
|---|---|---|
| `memory_hygiene`   | 03:30 nightly | Three-way reconcile of Qdrant ↔ Obsidian pages ↔ link-nodes. Creates missing link-nodes, deletes orphan pages + link-nodes, scrubs dead-UUID lines, rewrites `_index/uuid-map.md`. Never touches Qdrant. |
| `duplicate_digest` | 06:00 Sundays | Vector-search neighbor scan; pairs with cosine ≥ 0.95 posted to Telegram with Forget A / Forget B / Merge / Ignore inline buttons. Pending pairs persist in `data/duplicate_review.json` so callbacks survive restart. |
| `staleness_sweep`  | 05:00 monthly | Soft-archive memories not retrieved in 180 days. Cap 100/run. |

All three are silent when there's no work to do.

## Why this design

- **Search needs to be fast and semantic.** Vector search beats keyword
  scroll for "questions about things you've told me" (e.g. *"who is
  rylee?"* finds the right memory without matching the word "who").
- **The boss needs to write, not just read.** Without write capability,
  every conversation starts from the same baseline. The `save_memory`
  tool lets context accumulate over time.
- **Human review matters.** Obsidian's markdown pages are readable in
  any editor, searchable with `grep`, syncable with any sync tool, and
  survive Qdrant going away. The vault is the durable copy of truth.
- **Graph + vector together** catches things neither alone would. Vector
  finds semantic neighbors; graph finds explicitly-asserted relationships
  (made by either the boss or the user when curating).
