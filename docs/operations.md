# Operations

## Running

Artoo runs as a `systemd` user service. The unit file lives at
`systemd/artoo.service` in the repo and gets installed to
`~/.config/systemd/user/artoo.service`.

```bash
# Install / reinstall the unit
cp systemd/artoo.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now artoo.service

# Verify
systemctl --user status artoo.service
```

The unit sets `PATH` explicitly to include `~/.local/bin` so user-installed
CLIs (`gh`, `pnpm`, `uv`, etc.) are reachable by the boss's `local` and
`github` tools.

For the service to start when you're logged out, linger must be on:
```bash
loginctl enable-linger $USER
```

## Logs

stdout/stderr are captured by journald:

```bash
# Live tail
journalctl --user -u artoo -f

# Recent errors only
journalctl --user -u artoo -p err -n 100
```

Per-day structured activity log (every chat turn, worker dispatch,
cron fire — JSONL, queryable with `jq`):
```bash
ls data/activity/
cat data/activity/$(date -u +%F).jsonl | jq .
```

`data/artoo.log` is a v1 artifact (when systemd appended stdout to a
file). New deployments don't write it. If yours exists, the
logrotate config in `systemd/artoo.logrotate` will tidy it.

## Restarting

```bash
systemctl --user restart artoo
```

This kills the orchestrator and the scheduler. **In-flight worker
HTTP calls are cancelled** — the OpenRouter request gets a TCP RST and
the response is lost. A restart mid-conversation means the user has to
resend their last message.

The boss's `local` tool has a `restart` op that schedules a delayed
self-restart via systemd. Prefer that over `systemctl restart` from
inside an active conversation — it lets the current turn complete
first.

## Configuration

All deployment-specific values live in `.env` (gitignored). The
`config.py` defaults are localhost-ish placeholders so the code can run
on a fresh machine without any env. See `.env.example` for the full
list.

Required for full function:
- `TELEGRAM_BOT_TOKEN` — bot to receive/send on
- `OPENROUTER_API_KEY` — primary inference backend
- `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION` — memory store
- `OBSIDIAN_VAULT` — path to the vault directory (CIFS, NFS, local — doesn't matter)
- `OLLAMA_HOST` — embedding endpoint with `mxbai-embed-large` loaded

Optional:
- `TELEGRAM_HOME_CHANNEL` — where crons push messages
- `ANTHROPIC_API_KEY` — only for the optional `anthropic_vision()` SDK fallback
- Browser-harness vars (`BROWSER_HARNESS_BIN`, `BU_CDP_URL`, `BU_NAME`,
  `BROWSER_TASK_MODEL`) — see the Browser automation section below
- `ARTOO_LOCAL_SAFE_ROOT` — override the boss's `local` tool safe root
- **Conductor** (`/build`): `ARTOO_CONDUCTOR_MODEL` (default
  `anthropic/claude-sonnet-4.6`), `ARTOO_CONDUCTOR_MAX_CYCLES` (default 6),
  `ARTOO_CONDUCTOR_DAILY_BUDGET_USD` (default 5; also adjustable via `/budget`)
- `ARTOO_BUILDS_DIR` — the build workshop where `/build <slug>` places scratch
  projects (default `~/builds`); kept separate from `~/projects` (real work)
- **Memory rerank**: `ARTOO_MEMORY_RERANK` (default on; `0` disables),
  `RERANK_URL` (local cross-encoder TEI endpoint; unset = cloud reranker
  Llama-3.3-70B carries it)

## Adding a worker

1. Drop a module under `artoo/workers/` exporting `CONFIG: WorkerConfig`:

```python
# artoo/workers/codebase_search.py
from ..runtime import WorkerConfig

CONFIG = WorkerConfig(
    name="codebase_search",
    description="Search a local codebase for a pattern, return paths + snippets.",
    model="openrouter:anthropic/claude-sonnet-4.6",
    system_prompt="You search code. Return file:line + the matching block. No commentary.",
    timeout=120,
)
```

2. Register in `artoo/workers/__init__.py`:
```python
from . import codebase_search
_REGISTERED = [general, deep, fast, quick, designer, codebase_search]
```

3. `systemctl --user restart artoo`

The boss sees the new worker in the spawn_worker catalog automatically —
no system prompt change needed.

## Adding a cron

1. Drop a module under `artoo/crons/` exporting `JOB: Job`:

```python
# artoo/crons/rack_check.py
from ..scheduler import Job
from .. import notify

async def run():
    # ping host list, build a status message
    await notify.to_telegram(status)

JOB = Job(name="rack_check", schedule="0 */4 * * *", fn=run, timeout=120)
```

2. Register in `artoo/crons/__init__.py`:
```python
from . import rack_check
scheduler.register(rack_check.JOB)
```

3. Restart

To test a cron immediately without waiting for its scheduled time:
```bash
.venv/bin/python -c "import asyncio; from artoo.crons import rack_check; asyncio.run(rack_check.run())"
```

## Debugging tips

- **Boss not seeing a tool / worker:** confirm `orchestrator._TOOLS`
  lists it (in-process tool dispatch — no MCP config file involved
  for the boss path). For external MCP clients, sanity-check the
  standalone server: `.venv/bin/python -m artoo.mcp_server` (it'll
  hang on stdio — that's correct).

- **Stuck OR call:** the boss path has `_post_with_retry` with up to
  3 retries (1/3/9s backoff). Check `journalctl --user -u artoo`
  for `OR call retry` lines.

- **gh / pnpm / uv not found by a tool:** the systemd unit's
  `Environment="PATH=..."` doesn't include the dir. Edit the unit
  and reload.

- **Memory search returns nothing:** check Ollama is reachable
  (`curl http://OLLAMA_HOST/api/version`); check the right collection
  is set (`QDRANT_COLLECTION` env); check Qdrant has data
  (`curl QDRANT_URL/collections/COLLECTION_NAME`).

- **Telegram "typing" indicator dies after 5s:** the typing refresher
  task in `channels/telegram.py` failed for some reason. Check
  `journalctl --user -u artoo` for the failure.

## Backups

The Obsidian vault is the durable copy of memory content — back it up
at the filesystem level (rsync, ZFS snapshots, restic, whatever).
Qdrant can be rebuilt from the vault if needed.

`data/artoo.db` holds conversation history + chat prefs. Back it up
under load with the SQLite online backup API:
```bash
sqlite3 data/artoo.db ".backup data/artoo.db.bak"
```

For Qdrant itself, hit `POST /collections/<name>/snapshots` to create
a snapshot, then download from `GET /collections/<name>/snapshots/<id>`.
A nightly cron for this is on the urgent list but not built yet.

## Browser automation (browser-harness)

The boss has two tools (`browser_run`, `browser_task`) that drive a real
Chromium via [browser-use/browser-harness](https://github.com/browser-use/browser-harness).
Chromium runs as its own systemd user unit, with CDP exposed on
`127.0.0.1:9222` and a persistent profile under
`~/.cache/artoo/chrome-profile`.

One-time setup:

```bash
# 1. Install Google Chrome stable (the unit ExecStart hardcodes this path)
wget -qO- https://dl.google.com/linux/linux_signing_key.pub | \
  sudo gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" | \
  sudo tee /etc/apt/sources.list.d/google-chrome.list
sudo apt update && sudo apt install -y google-chrome-stable

# 2. Install the browser-harness CLI globally
uv tool install browser-harness

# 3. Install + start the Chromium systemd user unit
cp systemd/chromium-harness.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now chromium-harness.service

# 4. Verify CDP is reachable
curl -s http://127.0.0.1:9222/json/version | head -1

# 5. Restart artoo so the new tools register
systemctl --user restart artoo
```

Logins are NOT pre-populated — the profile starts blank. To seed
cookies for a site Artoo needs (e.g. Gmail), stop the headless unit,
launch Chrome interactively against the same `--user-data-dir`, log in,
quit, then re-enable the unit. Cookies persist to disk.

Tunable env vars (set in `artoo/.env`):

- `BROWSER_HARNESS_BIN` — path override if `browser-harness` isn't on `$PATH`.
- `BU_CDP_URL` — defaults to `http://127.0.0.1:9222`. Point elsewhere
  for a remote browser.
- `BU_NAME` — daemon IPC namespace. Defaults to `artoo`.
- `BROWSER_TASK_MODEL` — model that drives the `browser_task` sub-loop.
  Defaults to `anthropic/claude-sonnet-4.6`.

## Log rotation

`data/activity/*.jsonl` rolls one file per day. `data/artoo.log` is the
legacy systemd-redirected log (new deployments don't create it).
Logrotate config at `systemd/artoo.logrotate`:

```bash
sudo cp systemd/artoo.logrotate /etc/logrotate.d/artoo
sudo logrotate -d /etc/logrotate.d/artoo   # dry-run to verify parse
```

journald handles its own retention for the systemd-captured logs (tune
via `/etc/systemd/journald.conf` if needed).

## Versioning

`v<overhaul>.<refactor>.<skill>` — see README. When you bump, update
`pyproject.toml`. Commits with code changes get squashed under a
version bump if they're part of one logical change.
