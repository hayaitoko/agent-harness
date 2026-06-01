# Architecture decisions

Captured decisions from the build night and beyond. These exist so
future-the operator (or future-Claude) doesn't re-litigate them without
context.

The early ADRs (002, 004, 007, 008) were reversed in the v1 → v2
overhaul. They're kept here annotated as **Superseded** with pointers
to the replacement ADR so the historical reasoning isn't lost.

---

## ADR-001 — Greenfield codebase over forking hermes-agent

**Context:** The starting plan was to gut `hermes-agent` in place,
keeping its Telegram adapter + gateway + DB and replacing only the
agent loop.

**Decision:** Build a fresh codebase from scratch.

**Why:** Hermes is a full framework (60+ modules in `agent/`, separate
gateway/CLI/TUI/web layers, multi-provider routing, training-data
generation, self-improving skills). 90% of it is dead weight for a
personal agent. Forking means fighting upstream changes for the 10%
that's useful, forever. A focused ~700-LoC codebase is more
understandable and longer-lived than a fork.

**Trade-off:** Lost the existing Telegram polling code and channel
infrastructure. Reimplementing was 50 lines with `python-telegram-bot`.

---

## ADR-002 — `claude -p` subprocess as the model interface

**Status:** **Superseded 2026-05-XX by ADR-011** — replaced by
OpenRouter-routed Python-native chat-completions in v2.0.

**Context:** Need a way to call Claude that doesn't burn $5-15/day on
Sonnet API.

**Decision:** Spawn `claude -p` (Claude Code CLI) per turn. It
authenticates via Max OAuth and counts against the Max subscription
quota, not per-token API billing.

**Why:** Personal agent, dozens of turns/day. API path = real money.
Max path = $0 within quota. Same models, same quality.

**Trade-off:** Subprocess startup overhead (~1s per turn) and a binary
dependency on a logged-in `claude` CLI. The `--bare` mode that would
strip overhead also disables Max OAuth, so we can't use it.

---

## ADR-003 — Stateless orchestrator, no `--resume`

**Context:** `claude -p --resume <session-id>` continues a previous
session. Tempting because it offloads conversation state to Claude
Code.

**Decision:** Don't use `--resume`. The orchestrator is a stateless
function. Conversation history lives in SQLite and gets prepended to
the prompt each turn.

**Why:**
- Session files grow until auto-compaction, then assistant memory
  shifts in subtle ways.
- Single-writer only — concurrent channels would corrupt session state.
- Recovery after a bad session is filesystem surgery.
- Prompt caching at the API layer makes the cost of re-sending the
  prefix small.

**Trade-off:** Each turn re-sends the conversation, which is more
tokens than `--resume`. Cache offsets this. Worth it for the
operational simplicity.

**Carries forward to v2:** the "stateless orchestrator that re-sends
prefix each turn" pattern survives the `claude -p` → OpenRouter
move (ADR-011). Same argument, different model interface.

---

## ADR-004 — MCP for tool exposure

**Status:** **Superseded 2026-05-XX by ADR-012** — orchestrator now
dispatches tools in-process; MCP server is a separate optional standalone.

**Context:** The boss needs to call Python functions (search_memory,
save_memory, spawn_worker). Options were:
- Custom text-based protocol (`<tool>name(args)</tool>` in responses)
- MCP server over stdio
- Anthropic API tool-use schema directly

**Decision:** Run an MCP server (`artoo/mcp_server.py`) that `claude -p`
loads via `--mcp-config`.

**Why:** MCP is the right abstraction. Claude Code's tool-use loop is
already excellent and built in — we get all of it (correct argument
parsing, error handling, multi-step reasoning) for free. Writing the
server was ~100 lines.

**Trade-off:** Subprocess-launches-subprocess depth (orchestrator →
claude -p → MCP server → claude -p for workers). Works fine but
debugging takes a moment to internalize.

---

## ADR-005 — Workers as discrete tools, not free-form delegation

**Context:** The boss needs to delegate. Options: (a) let the boss
spawn arbitrary `claude -p` calls with arbitrary system prompts, or
(b) restrict it to named workers.

**Decision:** Named workers only. The MCP tool `spawn_worker(name, prompt)`
has an `enum` of registered worker names; calling an unknown name
returns an error.

**Why:** The boss can't go rogue. Each worker has a vetted system
prompt and scoped tool access. Adding a new capability is an
intentional code change, not an emergent agent behavior. Debugging
who-said-what is easier when workers are typed.

**Trade-off:** Less flexibility. Mitigated with a `general` worker as
an escape hatch — generic Claude on Sonnet, no tools, for one-offs
that don't merit a specialist.

**Carries forward to v2** — the `spawn_worker(name, prompt)` enum-tool
still exists; only the call dispatch (MCP stdio → in-process) changed.

---

## ADR-006 — Memory as a tool, not eager-loaded

**Context:** Legacy Hermes injected ~3.5KB of memory + 60KB of skills
into every system prompt. The boss saw all of it regardless of need.
This was the documented $6/conversation bloat source.

**Decision:** Boss starts each turn with a tiny system prompt
(persona + delegation rules + tool descriptions, ~1KB). It calls
`search_memory` when it decides it needs context. No eager load.

**Why:** Most turns don't need memory ("hey what's up" doesn't need
the full corpus). Pre-loading wasted tokens on every turn. On-demand
queries cost ~50ms and ~0 system-prompt tokens.

**Trade-off:** Boss has to *know* to call search_memory when relevant.
Handled by the system prompt's delegation guide ("user references
prior conversations → search_memory FIRST").

---

## ADR-007 — Re-embed corpus with `mxbai-embed-large`

**Status:** **Carried forward** — `mxbai-embed-large` (1024-dim,
Cosine) remains artoo's embedding model in v2. The two-collection
setup (ADR-008) was retired; everything now lives in
`QDRANT_COLLECTION` (`artoo_memories` by default).

**Context:** The legacy memory_proxy fell back to zero vectors when
its embed model wasn't available. Most of the 373 existing points had
zero vectors, so "vector search" was actually keyword scroll-and-filter
in disguise.

**Decision:** Re-embed all points into a new collection using
`mxbai-embed-large` (1024-dim, Cosine).

**Why:** Real semantic search > keyword filter. The "who is rylee"
query is the canonical test — keyword search would never rank that
correctly because "who" doesn't appear in the memory.

**Model choice:** `mxbai-embed-large` is current SOTA in its size
class, runs on a 3060 Ti in ~7ms per embed, free (local), and has
1024 useful dimensions.

---

## ADR-008 — Two-collection setup for safe migration

**Status:** **Superseded 2026-05-15** — single-collection. The legacy
v1 proxy was decommissioned; the `memory_resync` cron was unregistered.

**Context:** Going from v1 (384-dim) to v2 (1024-dim) — different
vector spaces, can't just upsert in place.

**Decision:** New collection with same UUIDs and 1024-dim vectors.
Artoo reads from v2. The legacy proxy and Claude.ai's MCP connector
keep writing to v1. A daily cron syncs v1 → v2.

**Why:** Preserving UUIDs means the obsidian link graph still
resolves. Keeping v1 around means cross-session memory from
Claude.ai keeps working. Sync cron means artoo eventually sees those
writes.

**Trade-off:** Two collections to maintain. The legacy proxy could be
turned off, eliminating one write path; deferred until the v1
collection has zero non-resync writers.

---

## ADR-009 — In-process cron scheduler (not OS cron)

**Context:** Need scheduled jobs (spend_check, memory_inventory,
etc.). Options: (a) in-process scheduler in the same Python process
as the Telegram adapter, or (b) OS-level `cron` calling
`python -m artoo.crons.X`.

**Decision:** In-process via `croniter`. The scheduler is an asyncio
task in the same event loop as the Telegram adapter.

**Why:** Shared state. Crons can use the same workers, the same
memory, the same `notify` helper, and the same orchestrator setup as
the chat path. One process to manage. Hot-reload by restarting artoo.

**Trade-off:** If artoo crashes during a scheduled time, the cron
doesn't fire. systemd restarts artoo automatically; no
"catch-up-missed-jobs" logic exists. Acceptable for personal use.

---

## ADR-010 — Versioning: `v<overhaul>.<refactor>.<skill>`

**Context:** Standard semver felt wrong. The project is for one
person; "major / minor / patch" doesn't map to how it actually evolves.

**Decision:** `v<overhaul>.<refactor>.<skill>`.
- **overhaul:** full architecture rewrites. `v1` is artoo itself
  (the first overhaul of the previous hermes-based setup).
- **refactor:** structural changes to core modules (orchestrator,
  runtime, scheduler, memory layer).
- **skill:** counts up when a worker or cron is added.

**Why:** It's fun, and it maps to what's actually changing. "Bumped
the refactor digit" tells you the core got reshaped. "Bumped the
skill digit" tells you a new capability landed.

**Trade-off:** Non-standard. Anyone reading the version externally
won't know what it means. Documented here.

---

## ADR-011 — Drop `claude -p`; route all calls through OpenRouter

**Supersedes:** ADR-002.

**Context:** v1's `claude -p` subprocess interface locked artoo to
Anthropic, paid per-turn subprocess startup overhead, and shipped a
binary dependency on a logged-in `claude` CLI. Wanting to put the
boss on Kimi K2.6 (Fireworks-hosted) made this untenable.

**Decision:** Every model call goes through OpenRouter via a
Python-native chat-completions loop (`runtime.openrouter` for
one-shots, `agent_loop.run` for the boss tool-use loop). No
subprocess, no MCP stdio. Provider routing is per-prefix-pin
(`PROVIDER_PINS` in `runtime.py`).

**Why:**
- Multi-provider routing in one request shape.
- In-process tool dispatch eliminates the orchestrator → claude →
  MCP → claude depth.
- ZDR claim is enforceable via `data_collection: "deny"` + pinned
  provider + `allow_fallbacks: false`.
- Worker model swaps become one-line edits.

**Trade-off:** Per-call HTTP overhead instead of subprocess startup
(roughly a wash). We maintain our own retry/backoff. Cost is per-token
again — but Kimi via Fireworks is dramatically cheaper than Sonnet via
the Anthropic API for boss-shaped traffic.

---

## ADR-012 — In-process tool dispatch (MCP becomes optional)

**Supersedes:** ADR-004.

**Context:** With ADR-011 dropping the `claude -p` subprocess, the
MCP server's role as the boss's tool transport went away. Keeping it
in the hot path would mean orchestrator → agent_loop → MCP stdio →
Python — adding round-trips for no benefit.

**Decision:** `orchestrator._TOOLS` lists every tool the boss can
call (`search_memory`, `save_memory`, `delete_memory`,
`restore_memory`, `remind`, `generate_image`, `local`, `github`,
`browser_task`, `spawn_worker`). `_tool_handler` dispatches to the
Python implementation directly when `agent_loop` emits a tool call.
`mcp_server.py` survives as a standalone module for external clients
(Claude.ai's qdrant-memory connector, artoo-web) that want the same
tool surface over MCP stdio.

**Why:** Cuts latency, removes a transport layer to debug, and lets
us evolve the boss tool surface without breaking external MCP
consumers (because they pin to the MCP server's interface, not the
orchestrator's internals).

**Trade-off:** The boss and external MCP clients see *almost* the
same tools, but not exactly — the orchestrator exposes `local`,
`github`, and `browser_task`, which the MCP server intentionally
withholds from external callers. Contracts drift if we're not
careful; mitigated by keeping the memory + remind + spawn_worker
schemas in lockstep between the two surfaces.

---

## ADR-013 — Single Qdrant collection; retire v1 + memory_resync cron

**Supersedes:** ADR-008.

**Context:** Two-collection (v1 + v2) was a migration safety net. By
2026-05-15 the legacy proxy had zero non-resync writers, and the
Claude.ai connector now writes directly to v2 via the MCP server
(ADR-012).

**Decision:** Single collection (`QDRANT_COLLECTION`, default
`artoo_memories`). The `memory_resync` cron is unregistered in
`crons/__init__.py`. The migration script (`scripts/reembed.py`) and
the old resync module stay on disk as historical references; both
need clear "do not run in production" guardrails (see audit).

**Why:** One source of truth. No more sync lag. Memory hygiene
becomes a single three-way reconcile (Qdrant ↔ Obsidian pages ↔
link-nodes) instead of two-way sync + reconcile.

**Trade-off:** No more "isolated v1 archive" — if a write ever
corrupts a memory, the only safety net is Obsidian's filesystem
mirror. Acceptable for personal use given how rare writes are.
