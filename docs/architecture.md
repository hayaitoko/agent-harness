# Architecture

## The shape

```
   Telegram ──┐
   (CLI)   ──┼──> orchestrator ── agent_loop ──> OpenRouter
              │     │   (Python-native tool-use loop, sync)
              │     │
   Channels   │  SQLite                ├─ search_memory  ──> Qdrant + Obsidian
   to add  ──┘  conversation           ├─ save_memory    ──> Qdrant + Obsidian
                history                ├─ remind         ──> SQLite reminders
                                       ├─ generate_image ──> OpenRouter (image)
                                       ├─ local          ──> read/write/run/dev/build/restart
                                       ├─ github         ──> wrapped gh CLI
                                       ├─ browser_task   ──> headless-Chrome sub-agent
                                       └─ spawn_worker
                                                  │
                                                  ├─ deep      (DeepSeek v4 Pro)
                                                  ├─ general   (GLM-5.1)
                                                  ├─ designer  (Sonnet 4.6)
                                                  └─ quick     (Mercury-2)
```

The orchestrator is the only thing chat channels talk to. It's a (mostly)
stateless function: `respond(message, history) → text`. State
(conversation history, activity log, scheduled jobs) lives in the
surrounding Python process and SQLite.

## Why OpenRouter as the model interface

v1 used `claude -p` (Claude Code CLI) as the model interface, leaning on
Max OAuth quota. v2 replaced that with a Python-native chat-completions
loop against OpenRouter for three reasons:

1. **Multi-provider routing.** Per-family provider pins (`PROVIDER_PINS`
   in `runtime.py`) let the boss run on Kimi K2.6, workers on
   DeepSeek + Mercury, and design tasks on Sonnet — all from one
   request shape. `claude -p` only spoke to Anthropic.
2. **In-process tool dispatch.** The boss's `agent_loop.run()` calls
   Python functions directly. No subprocess startup overhead, no MCP
   stdio round-trip, no `bypassPermissions` flag. The MCP server still
   exists (`mcp_server.py`) but it's for external clients only.
3. **ZDR enforcement.** OR's `data_collection: "deny"` plus single-provider
   pinning (`allow_fallbacks: false`) keeps prompts off provider training
   sets. The boss path sets this explicitly; worker paths inherit the
   same provider pin.

The trade-off: per-call HTTP overhead instead of subprocess startup, and
we maintain our own retry/backoff (`_post_with_retry` in `runtime.py`).
Worth it for the routing flexibility.

## Why the orchestrator is stateless

We deliberately don't carry session state in the model interface — every
turn re-sends the conversation prefix from SQLite. Reasons:

- **Single-writer only.** A stateful model-side session corrupts under
  concurrent channels or restarts.
- **Recovery is annoying** when a session goes bad — there's no clean
  "roll back one turn" without filesystem surgery.
- **Prompt caching at the provider layer** amortizes the cost of
  re-sending the same prefix (Anthropic ephemeral + Fireworks automatic
  prefix caching).

Conversation history lives in SQLite (`storage.py`, WAL mode) and gets
prepended to the prompt each turn. The orchestrator is a pure
function-of-state — easy to debug, easy to replay, easy to test.

## Workers as discrete tools

The boss can't spawn arbitrary model calls with arbitrary prompts. It
calls `spawn_worker(name, prompt)` — a tool whose `name` arg is an enum
of registered workers (`general`, `deep`, `designer`, `fast`, `quick`).
Each worker declares its model, system prompt, scoped tools, and timeout
in a `WorkerConfig`.

This is the difference between "agent with access to a delegation
primitive" and "agent that can spawn rogue subprocesses." The former is
predictable, debuggable, and bounded.

The general-purpose escape hatch (`general` worker) exists so the boss
isn't forced to invent a specialist for every one-off task. Specialists
should be carved out when patterns repeat — not pre-designed.

## Model routing strategy

The boss runs on Kimi K2.6 (Fireworks-pinned). Picked 2026-05-17 from
the operator's 12-test eval: tied for top overall score (107/112), no errors,
~1/4 the input tokens of Sonnet on long-project orchestration.
Reasoning model — `agent_loop` runs with the higher `max_tokens`
default to leave room for reasoning + tool calls.

The boss self-escalates by calling `spawn_worker`:

| Trigger                                | Worker      | Model               |
|----------------------------------------|-------------|---------------------|
| Hard reasoning, complex synthesis      | `deep`      | DeepSeek v4 Pro     |
| Mid-weight scoped task                 | `general`   | GLM-5.1             |
| UX / design specialist                 | `designer`  | Sonnet 4.6          |
| Polish + cheap classify / extraction   | `quick`     | Mercury-2 (diffusion) |

The "right" choice happens during reasoning, not via a pre-classifier
that runs on every turn. Most turns resolve in the boss's own context
without invoking any worker. The boss only delegates when the task
genuinely needs it.

`/model` sets the per-chat boss preference. The legacy one-shot
overrides `/opus`, `/sonnet`, `/haiku`, `/kimi` were retired in v2.15.1.

## Memory as a tool, not a context prefix

The legacy Hermes setup pre-loaded ~3.5KB of memory + 60KB of skills
into every system prompt. The boss read all of it whether it needed any
of it. That was the $6/conversation bloat source.

Artoo's boss starts each turn with a tiny system prompt (persona +
delegation rules + tool descriptions, ~1KB). It *calls* `search_memory`
when it actually needs context. Memory queries are ~50ms and ~0 tokens
of system-prompt waste.

`search_memory` = dense vector search (mxbai-embed-large, 1024-dim, Qdrant) +
1-hop Obsidian link-graph, then a **rerank** pass: it over-fetches (top ~20) and
re-sorts to the best few. Recall was strong but top-1 was weak (recall@5 0.90
vs recall@1 0.35), so reranking lifts the right memory to the top (recall@1
0.35→0.65 on Llama-3.3-70B; `memory_rerank.py`). Reranker order is local
cross-encoder (`RERANK_URL`) → cloud LLM → identity fallback. (The active
filter is `must_not status==archived`, not `MatchExcept` — the latter silently
dropped every legacy field-less point, i.e. ~all memories, until 2026-05-31.)

## Memory hygiene

Memory lives in two layers: **Qdrant** (`artoo_memories`,
mxbai-embed-large, 1024-dim cosine) holds the content + vectors and is
the source of truth; the **Obsidian vault** is a derived link graph
(`<date>.<slug>.<uuid>.md` pages + `link-nodes/<uuid>.md` UUID-named
backlink anchors + `_index/uuid-map.md`).

The cron suite reconciles + tidies:

| Cron | Schedule | What it does |
|---|---|---|
| `memory_hygiene` | 03:30 nightly | Three-way reconcile: creates missing link-nodes, deletes orphan pages, deletes orphan link-nodes, scrubs dead-UUID lines from every link-node body, rewrites `_index/uuid-map.md` from current Qdrant state. Never touches Qdrant — only derived vault state. |
| `duplicate_digest` | 06:00 Sundays | Vector-search neighbor scan; pairs with cosine ≥ 0.95 posted to Telegram with Forget A / Forget B / Merge / Ignore inline buttons. Pending pairs persist in `data/duplicate_review.json` so callbacks survive restart; Ignore entries TTL 30 days. |
| `staleness_sweep` | 05:00 monthly | Soft-archives memories whose `last_retrieved_at` is older than 180 days. Grandfathers points missing the field (treats `created_at` as the anchor; falls through to "skip" if neither exists). Cap 100/run. |

Boss-facing tools:

- `delete_memory(uuid, hard=False)` — soft-archive by default (sets
  `payload.status = "archived"`; search filters it out). `hard=True`
  cascades: deletes Qdrant point + page file + link-node file + scrubs
  the UUID from every other link-node body.
- `restore_memory(uuid)` — flip archived → active. Hard-deletes can't
  be restored through this path.

Search filters `status == "archived"` by default. `mark_retrieved` is
called on every primary hit so the staleness sweep has a real "last
useful" signal to reason against. Files: `artoo/memory.py`,
`artoo/crons/memory_hygiene.py`, `artoo/crons/duplicate_digest.py`,
`artoo/crons/staleness_sweep.py`.

## Build conductor

`/build` and the boss's `local(op=build)` tool drive the **conductor**
(`artoo/conductor/`) — a Sonnet-driven agentic loop that replaced the old
round_pipeline (the PLAN→CODE→REVIEW→FIX→VERIFY stage machine, deleted
2026-05-30). One capable model driving tools, not a fixed stage sequence.

**The loop.** A conductor model (Sonnet 4.6 by default; `ARTOO_CONDUCTOR_MODEL`)
runs via `agent_loop.run` with a small scoped tool kit — `run`, `read`, `write`,
`search`, `delegate`, `update_progress`, `verify` — over a project dir until the
verify gate is green. `run_build` is an outer loop around `agent_loop.run`: each
cycle the conductor works, then the harness runs verify *itself* and re-engages
the conductor if it's red. The model cannot declare a red build "done" — the
exit code decides, not its opinion.

**Pillars** (from the 2026-05-30 design):

1. **Scoped tools** — ~7, not 40. Less off-rails, cheaper routing.
2. **Delegation as a cost/context firewall** — `delegate(worker, task)` hands
   bulk code-writing to a cheap worker (DeepSeek `deep`), applies its
   `ARTOO_FILE` output, and returns only a *summary* — raw code never enters the
   expensive conductor's context. Code-writing workers need a real output budget
   (`WorkerConfig.max_tokens`; deep = 32768) or DeepSeek truncates mid-codegen.
3. **Code-enforced verify gate** — `.artoo/verify.yml` (below) is run by the
   harness; green = the project's real checks pass. Inconclusive (timeout /
   couldn't-run) is NOT green.
4. **Spec-adherence review gate** — green ≠ spec-met (a model can write tests
   that pass by mocking the hard part). After verify is green, a cheap
   cross-family reviewer (Kimi K2.6 thinking; `conductor/review.py`) checks the
   build against the goal and blocks on faked requirements / hollow tests
   (plaintext auth where SQL was asked, a fanout test that mocks the fanout…).
   **Done = verify green AND review pass.**
5. **Durable state + budget breaker** — PROGRESS.md persists across cycles; a
   daily spend cap (`conductor/budget.py`, default $5, `/budget`) is a wallet
   stop-loss (cost only, never a quality halt).

A project opts into verification with `.artoo/verify.yml`, run by the gate:

```yaml
verify:
  - name: web-check
    cmd: cd web && pnpm install --frozen-lockfile && pnpm run check
    parse: svelte-check
  - name: backend-test
    cmd: .venv/bin/python -m pytest -q
    parse: pytest
```

Each command is shelled out and parsed with a tool-specific parser
(`svelte-check` / `vitest` / `vite` / `pytest` / `tsc` / `generic`); exit 0 with
no parsed errors = green. `agent_loop` caches the *growing conversation* (rolling
cache_control breakpoint), not just the system+tools prefix — essential for the
conductor's long tool loops, where re-billed tool output would otherwise dominate
cost.


## Channels are interchangeable

Telegram is the only channel today. Adding Discord / Signal / Slack /
HomeAssistant / web is a single new module under `artoo/channels/`
that:

1. Receives an inbound message
2. Calls `orchestrator.respond(text, history)`
3. Writes turns to `storage.append(channel_id, role, content)`
4. Sends the response text back

The orchestrator doesn't know what channel called it. That's the whole
point of putting it behind a function boundary.
