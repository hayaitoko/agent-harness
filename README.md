# artoo

Personal homelab AI agent. Telegram-first, memory-aware, multi-model routing
through OpenRouter with ZDR enforced end-to-end.

## Architecture

```
   Telegram ──> orchestrator ──> agent_loop ──> OpenRouter
                     │                ▲           pinned per PROVIDER_PINS
                     │ system prompt  │           ZDR enforced
                     │ + tool schemas │           cache_control passthrough
                     │ (cached)       │           (Anthropic providers)
                     │                │
                     ├──> search_memory  ─> Qdrant + Obsidian (link graph)
                     ├──> save_memory    ─> Qdrant + Obsidian + uuid map
                     ├──> remind         ─> SQLite reminder queue (Telegram)
                     ├──> generate_image ─> OpenRouter (Gemini Flash Image)
                     ├──> local          ─> read/write/run/dev/build/restart
                     ├──> github         ─> wrapped gh CLI
                     ├──> browser_task   ─> headless-Chrome sub-agent
                     └──> spawn_worker
                              │
                              ├─ deep      (DeepSeek v4 Pro,  heavy reasoning)
                              ├─ general   (GLM-5.1,          mid-weight)
                              ├─ designer  (Sonnet 4.6,       UX + design)
                              └─ quick     (Mercury-2,        polish + classify/extract)
```

The boss is a single `agent_loop.run()` per turn — Python-native chat
completions loop against OpenRouter. Tool calls are dispatched in-process;
no subprocess, no MCP stdio, no `claude -p`. Workers are one-shot OR calls.

`PROVIDER_PINS` (in `artoo/runtime.py`) sends every request to one provider
per model family so cache pools stay coherent. Anthropic + Google route via
`google-vertex`; Moonshot + DeepSeek via `fireworks`; Inception via
`inception`; Tencent via `siliconflow`. Unpinned models raise rather than
silently routing.

Conversation history lives in SQLite (WAL mode). The system prompt + tool
schemas are marked `cache_control: ephemeral` — the cache TTL keeps boss
turns cheap when conversation is active. The effect depends on the
provider's caching support (Anthropic: explicit ephemeral; Fireworks:
automatic prefix caching; others: pass-through).

## Models

| Slot              | Model (default)                          | Backend       | Notes                                        |
|-------------------|------------------------------------------|---------------|----------------------------------------------|
| orchestrator boss | `moonshotai/kimi-k2.6`                   | fireworks     | Default boss. Reasoning model.               |
| `deep`            | `deepseek/deepseek-v4-pro`               | fireworks     | Heavy reasoning, complex synthesis           |
| `general`         | `z-ai/glm-5.1`                           | deepinfra     | Mid-weight scoped tasks                      |
| `designer`        | `anthropic/claude-sonnet-4.6`            | google-vertex | UX + design specialist                       |
| `quick`           | `inception/mercury-2`                    | inception     | Polish + classify/extract (diffusion; `fast` merged in 2026-05-21) |
| title-naming      | `openai/gpt-oss-120b`                    | google-vertex | Used by /new to title archived sessions      |
| vision (step 1)   | `google/gemini-3.1-flash-lite`           | google-vertex | Image-to-text describe; output feeds boss    |
| image generation  | `google/gemini-3.1-flash-image-preview`  | google-vertex | Nano Banana 2; `/image` cmd + tool           |

Boss model is per-chat overridable via `/model`. Aliases registered in
`orchestrator._MODEL_ALIASES`: `kimi`, `sonnet`, `opus`, `haiku`, `gpt-oss`.

Each worker is a one-line `model=` edit in `artoo/workers/<name>.py`.
Add new providers by extending `PROVIDER_PINS`.

## Memory layer

- **Qdrant** holds content + 1024-dim semantic vectors (`mxbai-embed-large`
  via Ollama).
- **Obsidian vault** is a human-readable mirror + link graph. Each memory
  is a page at `<date>.<slug>.<uuid>.md` plus a `link-nodes/<uuid>.md`
  that holds backlinked UUIDs.
- Reads use vector search; results are augmented one hop through the link
  graph. Falls back to keyword scroll if Ollama is unreachable.
- Writes are LLM-judged: the boss picks title, tags, and `linked_uuids`
  after a `search_memory` to find what's related.
- Boss-facing memory tools: `search_memory`, `save_memory`, `delete_memory`
  (soft archive by default; `hard=True` cascades), `restore_memory`,
  `mark_retrieved` (auto-called on each search hit for staleness tracking).

## Caching & cost

The boss's system prompt + tool schemas sit behind a `cache_control:
ephemeral` marker. Providers that honor it (Anthropic via google-vertex)
return a cache-read on subsequent turns within the TTL; providers with
their own caching (Fireworks) apply prefix caching automatically.

`runtime.Result` surfaces `cache_read_tokens` and `cost_usd` per call;
`activity.log_turn` writes them to the per-day JSONL activity log
(`data/activity/<date>.jsonl`) for later analysis.

Re-verify cache behavior after upstream changes:
```bash
.venv/bin/python -m artoo.scripts.probe_cache
```

## Crons

Registered in `artoo/crons/__init__.py`:

| Job                | Schedule        | What it does                                                          |
|--------------------|-----------------|-----------------------------------------------------------------------|
| `spend_check`      | `0 9 * * *`     | Daily OpenRouter balance to Telegram                                  |
| `memory_hygiene`   | `30 3 * * *`    | Three-way reconcile of Qdrant ↔ Obsidian pages ↔ link-nodes           |
| `duplicate_digest` | `0 6 * * 0`     | Vector-search near-duplicates; weekly Telegram digest with inline kb  |
| `staleness_sweep`  | `0 5 1 * *`     | Soft-archive memories not retrieved in 180 days (cap 100/run)         |
| `reminders`        | `* * * * *`     | Fire scheduled one-shot reminders to Telegram                         |

## Setup

Prereqs:
- Python 3.11+
- A Qdrant instance (local or remote)
- An Ollama instance with `mxbai-embed-large` pulled
- An OpenRouter API key with ZDR set on per your privacy preferences
- (For Telegram) a bot token

```bash
# Install deps
uv venv --python 3.12
uv pip install -e .

# Configure
cp .env.example .env
# fill in: TELEGRAM_BOT_TOKEN, OPENROUTER_API_KEY,
#         QDRANT_URL/API_KEY, OBSIDIAN_VAULT, OLLAMA_HOST

# Smoke test in CLI mode
.venv/bin/python -m artoo

# Run as a systemd user service
cp systemd/artoo.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now artoo.service
```

## Telegram commands

- `/model` — pick the boss model for this chat
- `/image <prompt>` — generate an image via Nano Banana 2 (Gemini 3.1 Flash Image)
- `/remind <delta> <msg>` — schedule a one-shot reminder
- `/dev <task>` — autonomous code pipeline (DeepSeek writes, Sonnet reviews)
- `/build <goal>` — agentic build conductor (Sonnet drives, delegates to cheap workers, verify + review gates)
- `/budget` — show/adjust the conductor's daily spend cap
- `/gh <args>` — run a `gh` CLI command
- `/qa <url>` — drive a URL through the browser harness as a QA pass
- `/new` — archive conversation, start fresh
- `/clear` — wipe conversation (no archive)
- `/ctx` — show current context window usage
- `/update` — update to the latest upstream, preserving local customizations; gate on tests; self-restart
- `/update rollback [ref]` — revert the last update (restore its backup branch), re-test, restart
- `/help` — show commands

`/opus`, `/sonnet`, `/haiku`, `/kimi` one-shot overrides were retired in
v2.15.1 — use `/model` to set a per-chat boss preference.

`/update` (home-chat-only) works for **both** the private repo and the public
mirror. The public mirror is re-snapshotted as a fresh single-commit history on
every publish, so a plain `git pull --ff-only` can't fast-forward it. Instead
`artoo/self_update.py` keeps a local `artoo-upstream-base` ref tracking the last
snapshot and does a vendor-branch 3-way merge: it applies only the upstream
*delta* onto your tree, so **your customizations are preserved**. Any merge
conflicts are handed to Kimi K2.6-thinking to resolve. It backs up first
(`artoo-update-backup-<ts>`), commits a dirty tree so it participates, runs the
pytest gate, and only restarts when green. If the merge is clean but tests fail,
it keeps the merged code on disk and does **not** restart — talk to the bot to
fix it, or `/update rollback` to revert. After `git remote set-url origin <your
fork>`, this is the whole update story for a public deployment.

The boss can also call `generate_image` itself — it'll send a picture into the
chat whenever it sees fit (visual answer lands better than words, you ask for
"a render of…", etc.).

## Adding a worker

1. Drop a module under `artoo/workers/` exporting `CONFIG: WorkerConfig`
2. Register it in `artoo/workers/__init__.py` (`_REGISTERED` list)
3. Restart artoo — the boss sees the new option in the catalog automatically

## Adding a cron

1. Drop a module under `artoo/crons/` exporting `JOB: scheduler.Job`
2. Register it in `artoo/crons/__init__.py` (`register_all()`)
3. Restart artoo

## Adding a provider pin

When introducing a new model family, add a prefix entry to `PROVIDER_PINS`
in `artoo/runtime.py`:

```python
PROVIDER_PINS = {
    "anthropic/":  "google-vertex",
    "google/":     "google-vertex",
    "openai/":     "google-vertex",
    "moonshotai/": "fireworks",
    "inception/":  "inception",
    "deepseek/":   "fireworks",
    "tencent/":    "siliconflow",
    # add here
}
```

Per-model exceptions live in `MODEL_PROVIDER_OVERRIDES` (checked first).
Unpinned models raise from `_provider_for()` rather than silently
routing — keeps ZDR enforcement and cache-pool integrity honest.

## Operations

```bash
systemctl --user status artoo
systemctl --user restart artoo
journalctl --user -u artoo -f
```

Activity log (per-day JSONL, structured):
```bash
tail -f data/activity/$(date -u +%F).jsonl
```

## Standalone MCP server (optional)

`artoo/mcp_server.py` exposes the memory + remind + spawn_worker tools
over stdio MCP for external clients (Claude.ai's qdrant-memory connector,
artoo-web, etc.). The orchestrator no longer uses it — kept for external
integrations.

```bash
python -m artoo.mcp_server
```

## Versioning

`v<overhaul>.<feature>.<patch>`:
- **overhaul** bumps for full architecture pivots (e.g. 1.x → 2.x: drop
  `claude -p`, all-OR routing).
- **feature** bumps when a new capability lands — a worker, cron, tool,
  Telegram command, or subsystem (e.g. 2.17.0 browser integration, 2.19.0
  round_pipeline + github_cli, 2.24.0 the build conductor + customization-
  preserving `/update`).
- **patch** bumps for fixes and refinements within a feature line.

## License

Personal project. No license, do whatever — just don't expect support.
