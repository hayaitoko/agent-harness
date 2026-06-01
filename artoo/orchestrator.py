"""Main agent loop.

State lives in the caller (CLI or telegram adapter) — the orchestrator is a
stateless function that takes a user message + conversation history and
returns Artoo's reply. The boss is one agent_loop.run() call per turn,
which handles tool dispatch (search_memory, save_memory, remind,
spawn_worker) against the OpenRouter chat-completions API.

The system prompt cache_control marker means the persona + delegation guide
+ tool catalog only burns full tokens once every ~5 minutes; every other
turn within that window reads them back at 1/10th cost.
"""
from __future__ import annotations

import contextvars
import json
import logging
import time

from . import activity, agent_loop, browser, github_cli, memory, reminders, runtime, persona, workers
from .image import generate as image_generate

_log = logging.getLogger("artoo.orchestrator")

# Per-turn image queue. Channel adapters (Telegram) pass a list via
# respond(..., image_queue=[...]); the generate_image tool appends to it;
# the adapter drains and sends after orchestrator.respond returns. The list
# is shared by reference across threads — asyncio.to_thread + contextvar
# isn't required, the caller just keeps a handle.
_pending_images: contextvars.ContextVar[list[tuple[bytes, str, str]] | None] = contextvars.ContextVar(
    "artoo_pending_images", default=None
)

_DELEGATION_GUIDE = """
YOU are the Artoo boss. You can answer directly or delegate to workers.

TOOLS:
- search_memory(query, limit): semantic search over Qdrant + obsidian link graph. Returns primary hits + linked memories.
- save_memory(text, title, tags, linked_uuids): persist a new memory. Call search_memory FIRST to find related uuids to link to.
- delete_memory(uuid, hard?): soft-archive (default) or hard-delete a memory by UUID. Use for stale or wrong facts: search_memory → identify the bad memory → delete_memory(uuid). Default soft-archive is reversible via restore_memory.
- restore_memory(uuid): undo a soft-archive. Hard-deleted memories cannot be restored.
- local(op, ...): read/write/run/dev/build against your own homelab VM (safe root /home/youruser/artoo). Use when the operator asks you to modify your own source, run a command, kick off a /dev iteration, or step a /build project. Ops: `read` (path), `write` (path, content), `run`/`shell` (cmd, cwd?, timeout_s?), `dev` (task, context?, max_rounds?, security_review?), `build` (project_dir, goal?, max_rounds?, autonomous?). Returns a structured dict; ok=False on any failure. Path traversal outside the safe root is refused. the operator trusts you with this — use it deliberately. Prefer `dev` over hand-coding multi-file edits via write; prefer `build` over manual round-stepping.
- remind(delta, text): schedule a one-shot reminder. delta is human ('30 minutes', '2h', '1d 4h'). Use whenever the operator asks to be reminded of something at a future time.
- cron(op, ...): create/list/delete RECURRING scheduled jobs. Each fire runs a prompt through your full agent loop (all tools) and sends the result to the operator. ops: create(schedule, prompt) where schedule is a 5-field cron expr ('0 8 * * 1-5' = 8am weekdays); list; delete(id). Use for "every morning/week, do X" requests. remind is one-shot; cron repeats.
- self_check(): ground truth about yourself — your live tool catalog, your repo's git branch/HEAD/clean state, whether pytest is installed, service status. Call it BEFORE claiming you lack a capability or are unsure about your own state.
- spawn_worker(name, prompt): delegate to a worker. The tool's description has the full catalog.
- github(command, cwd?): run the gh CLI on the operator's behalf (he's authed as youruser). Use for reading/writing PRs and issues, querying gh api, etc. Auth/config changes are refused.
- generate_image(prompt): make an image with Nano Banana 2 (Gemini 3.1 Flash Image) via OpenRouter and send it directly to the operator's chat. Use when he asks for a picture, or when a visual would land better than words. Delivery is automatic — don't describe what you generated after.

BEFORE PROPOSING FIXES TO ARTOO ITSELF (the conductor, telegram channel, runtime, etc.) — check recent commits first:
  github("api /repos/youruser/artoo/commits?per_page=20 --jq '.[] | {sha: .sha[:7], msg: .commit.message | split(\"\\n\")[0], date: .commit.author.date}'")
You don't see artoo's own source in your context. Many "fixes" you'd be tempted to suggest are already shipped — file-marker output format, retries with backoff, failure dumps to disk, the boss meta-review on persistent failure, etc. Read the last 10-20 commit subjects before pitching architectural changes, and pull a diff if a commit subject looks relevant (gh api /repos/youruser/artoo/commits/<sha>). Be honest when memory says "I designed X" — that draft may have been replaced; check before claiming X is the right move.

ROUTING — DECIDE PER TURN:
- Casual chat / answers you already have from this conversation: reply DIRECTLY, no tools.
- Need to recall something the operator told you, or context about his projects: search_memory FIRST.
- the operator shares a new fact, preference, decision, plan, or context that should outlive this conversation: search_memory for related → save_memory with title/tags/linked_uuids.
- Heavy reasoning (careful code review, complex synthesis, multi-step problem solving): spawn_worker('deep', ...).
- Trivial classification, field extraction, obvious rewording, short drafts, simple rewrites: spawn_worker('quick', ...).
- Mid-weight task that doesn't fit a specialist: spawn_worker('general', ...).

The instinct should be: "Can I do this in <10s of my own response? Yes → just answer. No → who's better?"

VOICE:
- Terse. the operator knows you. No restating context, no preamble like "Sure!" or "Great question!".
- When delegating, briefly note what you're doing ("Asking the deep worker..."), then return the worker's answer cleanly.

GROUND RULES — NON-NEGOTIABLE:
- BIAS TO ACTION — do the whole task before coming back. the operator's ask is authorization to reach the outcome, not just the first step. If getting there means SSHing into a box, running a chain of commands, reading files, or a dozen tool calls — DO them, end to end, then report what happened. Don't stop to ask "should I ssh in?" or "want me to run that?" — if the ask implied it, the answer is yes; that's what he meant. He had to coax you into an ssh once; never make him do that again. Exhaust everything you CAN do on your own first. A question to the operator is the LAST resort, only when you're truly blocked: a real ambiguity you can't resolve yourself, a credential/secret only he has, or an irreversible/destructive step (hard delete, force-push, restart, spending money). Everything else — just do it.
- YOU CAN SEE WEB PAGES AND SSH THE HOMELAB. Two capabilities you may not realize you have: (a) browser_see — you're text-only, so browser_run only surfaces a page's DOM text, but browser_see screenshots the current tab and a cheap vision model hands you back a full description. Use it to actually LOOK at a rendered dashboard, chart, canvas, image, or dialog when the DOM isn't enough. The harness Chromium already ignores self-signed certs, so internal/homelab https UIs load directly — there is no cert wall to click through. (b) Passwordless SSH — key auth is set up from this VM into the homelab, so reach hosts with local op=run (`ssh <user>@<host> '<cmd>'`); no password needed. Recall which hosts exist, their users, and what runs where via search_memory. Don't tell the operator you can't see a screen or can't SSH a box — you can.
- Don't fabricate. You have memory, worker delegation, the github tool, AND the `local` tool that gives you direct read/write/run/dev/build access to your own homelab VM (safe root /home/youruser/artoo). When the operator asks about your own state, your own source, or anything under /home/youruser/artoo — USE local. Don't say "I don't have filesystem access" — you do. Don't say "I can't run shell" — you can (local op=run). Don't say "I can't invoke /build" — you can (local op=build). Don't say "I can't modify my own code" — you can (local op=write or local op=dev). The only things outside your reach are: systemd commands (restarting yourself requires the operator), data outside /home/youruser/artoo, and anything requiring root.
- VERIFY BEFORE CLAIMING A LIMITATION. When you're unsure about your own state — what tools you have, whether something's installed, whether code is committed, whether the service is current — call self_check FIRST and answer from its output, not from memory or assumption. On 2026-05-26 you told the operator you had no filesystem access, that pytest wasn't installed, and that committed code was uncommitted — all three false, and all three caught instantly by self_check. Memory and intuition lie about your own state; the tools don't.
- The github tool CAN clone repos to disk (gh repo clone <repo> <path>), read repo trees (gh api /repos/<owner>/<repo>/contents/...), open PRs and issues, comment, etc. When the operator asks you to "go look at" or "clone" or "fetch" a repo, USE the tool — don't claim you can't.
- After self-modification via local.write or local.dev: the running process still has the OLD code in memory. To make a change live, follow the RESTART PLAYBOOK below. Don't pretend the change is already live before restarting.
- RESTART PLAYBOOK (use this exact order):
    1. Edit via local.write or local.dev.
    2. local.run cmd="pytest tests/ -q" to verify your change. If it fails: tell the operator, do not restart.
    3. Send one final message to the operator naming what you changed and saying "restarting now — back in ~5s."
    4. Call local.restart with reason="<short summary>". test_first=True (default) double-checks pytest before pulling the trigger. Don't set test_first=false unless the operator explicitly tells you to.
    5. Stop emitting tool calls after restart. The next message the operator sends will land in the freshly-restarted process.
  Never restart unprompted. Only restart after a code change that needs to land, or when the operator asks for it directly. Don't restart for memory hygiene, "to refresh context," or any other reason — the only legitimate trigger is "new code on disk that must enter the running process."
- Memory is a starting point, not ground truth — files and state change while memory doesn't auto-update. When the operator asks about *current* state of a file or system, prefer local.read (for your own source) or delegating to a worker that can verify (for elsewhere) over reciting memory.
- the operator acts on what you tell him. Get it right or say you don't know.
""".strip()

SYSTEM_PROMPT = f"{persona.ARTOO_PERSONA}\n\n{_DELEGATION_GUIDE}"

# Boss model. moonshotai/kimi-k2.6 in THINKING mode, pinned to parasail in
# runtime.MODEL_PROVIDER_OVERRIDES. Bench 2026-05-31 put k2.6-thinking in the
# boss seat over the standalone kimi-k2-thinking it replaces: same 262K ctx,
# but k2.6 is the faster, stronger base model, and gating its thinking on the
# OpenRouter reasoning param (BOSS_REASONING_EFFORT, below) gives the boss the
# delegation/planning reasoning it needs WITHOUT k2-thinking's habit of
# re-reasoning the whole history window every turn (the latency that made it
# feel slow on big contexts). CRITICAL: k2.6's thinking is NOT native — it
# only engages when respond()/respond_vision() pass reasoning_effort into
# agent_loop.run. Drop that param and the boss silently degrades to plain
# instruct k2.6.
DEFAULT_MODEL = "moonshotai/kimi-k2.6"

# Boss reasoning effort (low|medium|high) → OpenRouter reasoning.effort. This
# is what turns k2.6's thinking mode ON (the old k2-thinking reasoned
# natively and ignored this). respond() and respond_vision() pass it into
# agent_loop.run. Dial up for harder reasoning, down for snappier replies.
BOSS_REASONING_EFFORT = "medium"

# Long-context fallback. The boss (Kimi K2.6) caps at 262K. On the rare
# turn where estimated input (history + memory + tool catalog) blows past
# LONG_CONTEXT_THRESHOLD, respond() auto-switches to this 1M-ctx model.
# CONSTRAINT: it MUST support tool-calling — the boss always sends the tool
# catalog. Llama 4 Maverick (the prior pick) does NOT tool-call on OpenRouter
# — 404s with tools on every provider/ZDR combo, verified 2026-05-29 — which
# silently broke this path (it only fires on >250K-token turns, so it went
# unnoticed). Gemini 3.5 Flash: 1M ctx, tool-calling verified under
# data_collection=deny on google-vertex (ZDR), cheap. The switch only fires
# when the caller hasn't explicitly chosen a model — explicit /model picks
# always win, even if they'd truncate.
LONG_CONTEXT_MODEL = "google/gemini-3.5-flash"
LONG_CONTEXT_THRESHOLD = 250_000   # tokens; well below Kimi's 262K ceiling

# Boss failover chain. When the primary (Kimi K2.6 → parasail) exhausts its
# retry ladder on a transient error, the boss turn fails over to these in
# order, each on a DIFFERENT provider so a single-provider throttle doesn't
# follow. agent_loop only fails over before any tool side-effect has run (see
# base_msg_count there). All support OR tool-calling + the reasoning param.
#   1. GLM-5.1 → deepinfra   (the proven `general` worker model)
# (k2.6 used to sit here as a second fallback; it's the primary now, and
# respond() filters the primary out of this list anyway. Only one cross-
# provider fallback remains — add another non-k2-thinking lane here if more
# boss redundancy is wanted.)
# Each MUST be pinned in runtime.PROVIDER_PINS / MODEL_PROVIDER_OVERRIDES.
BOSS_FALLBACKS = [
    "z-ai/glm-5.1",
]

# Friendly aliases mapped to OR model ids. Used by /model and any caller
# that wants to pass a short name. Channel adapters resolve via
# _resolve_model() before invoking the boss.
_MODEL_ALIASES: dict[str, str] = {
    "kimi":    "moonshotai/kimi-k2.6",
    "sonnet":  "anthropic/claude-sonnet-4.6",
    "haiku":   "anthropic/claude-haiku-4.5",
    "gpt-oss": "openai/gpt-oss-120b",
    # 2026-05-21 additions for experimentation:
    "glm":     "z-ai/glm-5.1",
    "llama4":  "meta-llama/llama-4-maverick",
}


def _resolve_model(m: str) -> str:
    return _MODEL_ALIASES.get(m, m)


def _estimate_input_tokens(
    history: list[dict] | None,
    user_message: str,
) -> int:
    """Rough character-based token estimate for the boss input.

    Counts SYSTEM_PROMPT + tool schemas + history + user_message. Uses
    ~3 chars/token (Kimi tokenizer averages closer to 3.5 for English;
    we round low so the long-ctx switch fires before we'd actually
    exceed the model's window).
    """
    char_total = len(SYSTEM_PROMPT) + len(user_message)
    char_total += sum(len(json.dumps(t.parameters)) + len(t.description) for t in _TOOLS)
    if history:
        char_total += sum(len(turn.get("content", "")) for turn in history)
    return char_total // 3


_TOOLS: list[agent_loop.Tool] = [
    agent_loop.Tool(
        name="search_memory",
        description=(
            "Search Artoo's memory (Qdrant + obsidian link graph) for relevant past "
            "conversations, decisions, and notes about the user, their projects, and "
            "homelab. Use this whenever a user question references things they've told "
            "you before, or you need context you don't have from this conversation alone."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 5, "description": "Max primary results"},
            },
            "required": ["query"],
        },
    ),
    agent_loop.Tool(
        name="save_memory",
        description=(
            "Save a new memory to Artoo's knowledge base (qdrant + obsidian). Use when "
            "the user shares a fact, preference, decision, plan, or important context "
            "worth recalling in future conversations.\n\n"
            "RECOMMENDED FLOW: call search_memory FIRST to find related existing memories "
            "— their UUIDs go into linked_uuids so the graph stays connected. Choose a "
            "short title (3-7 words) and 2-5 tags."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text":  {"type": "string", "description": "The memory content (verbatim or paraphrased)"},
                "title": {"type": "string", "description": "Short title (3-7 words)"},
                "tags":  {"type": "array", "items": {"type": "string"}, "description": "2-5 tags"},
                "linked_uuids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "UUIDs of related memories (from search_memory). Optional but recommended.",
                },
            },
            "required": ["text"],
        },
    ),
    agent_loop.Tool(
        name="delete_memory",
        description=(
            "Delete a memory by UUID. Defaults to soft-archive: invisible to search "
            "but reversible via restore_memory. Use for stale or wrong facts: "
            "search_memory → identify the bad UUID → delete_memory(uuid).\n\n"
            "Set hard=true ONLY when the operator explicitly asks to forget something "
            "permanently. Soft-archive is the right default — it lets you recover "
            "if you misjudged which memory was wrong."
        ),
        parameters={
            "type": "object",
            "properties": {
                "uuid": {"type": "string", "description": "UUID of the memory to delete (from search_memory)"},
                "hard": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, irreversibly removes Qdrant point + page + link-node + backlinks. Default false.",
                },
            },
            "required": ["uuid"],
        },
    ),
    agent_loop.Tool(
        name="restore_memory",
        description=(
            "Reverse a soft-archive. Sets status back to active so it shows in normal "
            "search again. Hard-deleted memories cannot be restored this way."
        ),
        parameters={
            "type": "object",
            "properties": {
                "uuid": {"type": "string", "description": "UUID of the archived memory"},
            },
            "required": ["uuid"],
        },
    ),
    agent_loop.Tool(
        name="local",
        description=(
            "Unified read/write/run/dev/build against your own homelab VM "
            "(safe root /home/youruser/artoo). Use this when the operator asks you "
            "to modify your own source, run a shell command, kick off a "
            "/dev iteration, or step a /build project.\n\n"
            "ops:\n"
            "  read    — read(path)                          → {text, bytes}\n"
            "  write   — write(path, content)                → {bytes_written}\n"
            "  run     — run(cmd, cwd?, timeout_s?)          → {exit_code, stdout, stderr, duration_ms}\n"
            "  shell   — alias for run\n"
            "  dev     — dev(task, context?, max_rounds?, security_review?) → DevResult dict\n"
            "  build   — build(project_dir, goal?, max_rounds?, autonomous?) → round results\n"
            "  restart — restart(reason, delay_s=3, test_first=True) → schedules systemctl --user restart artoo.service\n\n"
            "Path traversal outside the safe root is refused. read/write "
            "cap at 5MB. run defaults timeout 60s (max 600s). dev caps "
            "max_rounds at 5; build at 10. restart runs pytest first by "
            "default and refuses if tests fail.\n\n"
            "Prefer `dev` for code generation tasks over hand-writing "
            "multi-file edits with `write`. Prefer `build` for project "
            "work over invoking pipelines manually."
        ),
        parameters={
            "type": "object",
            "properties": {
                "op": {
                    "type": "string",
                    "enum": ["read", "write", "run", "shell", "dev", "build"],
                    "description": "Which operation to perform",
                },
                "path":    {"type": "string", "description": "read/write: file path (rel to /home/youruser/artoo or absolute under it)"},
                "content": {"type": "string", "description": "write: file content"},
                "cmd":     {"type": "string", "description": "run/shell: command to execute via bash"},
                "cwd":     {"type": "string", "description": "run/shell: working directory (under safe root). Default = safe root."},
                "timeout_s": {"type": "integer", "description": "run/shell: timeout in seconds (default 60, max 600)"},
                "task":    {"type": "string", "description": "dev: what to build (one sentence to a short brief)"},
                "context": {"type": "string", "description": "dev: existing-code context the worker should reference"},
                "max_rounds": {"type": "integer", "description": "dev/build: round cap"},
                "security_review": {"type": "boolean", "description": "dev: run a post-approval security audit"},
                "project_dir": {"type": "string", "description": "build: project directory (abs path, or a bare slug → placed in the build workshop ~/builds)"},
                "goal":    {"type": "string", "description": "build: goal text (required for NEW projects to seed PROGRESS.md)"},
                "autonomous": {"type": "boolean", "description": "build: skip the human gate between rounds"},
                "reason":  {"type": "string", "description": "restart: short string for the audit log (e.g. 'orchestrator.py edit landed')"},
                "delay_s": {"type": "integer", "description": "restart: seconds to wait before the actual systemctl call (default 3, max 30) — gives your final message time to send"},
                "test_first": {"type": "boolean", "description": "restart: run pytest first and refuse on failure (default True; only set false if the operator explicitly says so)"},
            },
            "required": ["op"],
        },
    ),
    agent_loop.Tool(
        name="remind",
        description=(
            "Schedule a one-shot reminder. Parses a human time delta (e.g. '30 minutes', "
            "'1h', '2h 30m', '45m', '1 day') and sends `text` to the operator's Telegram at that "
            "time. Self-cleans after firing — no leftover state.\n\n"
            "Use whenever the operator asks to be reminded of something at a future time. "
            "Return the human-friendly confirmation to him."
        ),
        parameters={
            "type": "object",
            "properties": {
                "delta": {"type": "string", "description": "Time delta from now, e.g. '30 minutes', '2h', '1d 4h 15m'"},
                "text":  {"type": "string", "description": "The reminder message"},
            },
            "required": ["delta", "text"],
        },
    ),
    agent_loop.Tool(
        name="cron",
        description=(
            "Create, list, or delete RECURRING scheduled jobs. Unlike remind (one-shot), a "
            "cron job fires repeatedly on a cron schedule and, each time it fires, runs a "
            "PROMPT through your full agent loop — you reason with all your tools and the "
            "result is sent to the operator's Telegram. Use for recurring autonomous work: "
            "'every weekday 8am, check my GitHub notifications and summarize'.\n\n"
            "ops:\n"
            "  create(schedule, prompt) — schedule is a 5-field cron expression in server "
            "local time (e.g. '0 8 * * 1-5' = 8am Mon-Fri, '*/30 * * * *' = every 30 min, "
            "'0 9 * * 0' = 9am Sundays). prompt is what you'll be told to do when it fires — "
            "write it as an instruction to yourself.\n"
            "  list — show all jobs with their ids, schedules, and prompts.\n"
            "  delete(id) — remove a job by its id (from list).\n"
            "Persists across restarts. Return the human-friendly confirmation to the operator."
        ),
        parameters={
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["create", "list", "delete"], "description": "The operation"},
                "schedule": {"type": "string", "description": "create: 5-field cron expression, server local time"},
                "prompt": {"type": "string", "description": "create: the instruction to run each time the job fires"},
                "id": {"type": "string", "description": "delete: the job id to remove"},
            },
            "required": ["op"],
        },
    ),
    agent_loop.Tool(
        name="self_check",
        description=(
            "Return GROUND TRUTH about your own state: your live tool catalog, your repo's "
            "git branch/HEAD/clean-or-dirty, whether pytest is installed, and your service "
            "status. Takes no arguments.\n\n"
            "Call this BEFORE claiming you lack a capability or are unsure about your own "
            "state — e.g. before saying 'I don't have filesystem access', 'pytest isn't "
            "installed', 'that isn't committed', or 'I'm not sure if X is wired up'. It is "
            "cheap and authoritative. Don't guess about yourself when you can check."
        ),
        parameters={"type": "object", "properties": {}},
    ),
    agent_loop.Tool(
        name="spawn_worker",
        description=(
            "Dispatch a focused task to a specialist worker. Workers have their own model "
            "+ system prompt + scoped capabilities. Returns the worker's final text result.\n\n"
            f"Available workers:\n{workers.catalog()}"
        ),
        parameters={
            "type": "object",
            "properties": {
                "name":   {"type": "string", "enum": workers.names(), "description": "Worker name"},
                "prompt": {"type": "string", "description": "Task description for the worker"},
            },
            "required": ["name", "prompt"],
        },
    ),
    agent_loop.Tool(
        name="browser_run",
        description=(
            "Execute a Python snippet in a real Chromium session via browser-harness. "
            "Use for turn-by-turn browser work where YOU want to see each step's output "
            "and decide the next snippet — walkthroughs, debugging, reasoning-heavy tasks. "
            "Helpers (exact names — don't guess): goto_url(url), page_info(), "
            "js(expr) -> evaluates JS and returns the value, capture_screenshot(path=None, full=False), "
            "fill_input(selector, text), click_at_xy(x, y), press_key(key), scroll(x, y, dy=-300), "
            "wait(seconds), wait_for_load(), wait_for_element(selector), wait_for_network_idle(), "
            "new_tab(url), switch_tab(target), list_tabs(), upload_file(selector, path), "
            "http_get(url). Plus anything in agent-workspace/agent_helpers.py. "
            "Homelab web UIs (TrueNAS, Proxmox, PBS, etc.) use self-signed certs — the harness "
            "Chromium already launches with --ignore-certificate-errors, so just goto_url() the "
            "https URL directly; there is no cert interstitial to click through. State (open tab, "
            "daemon) persists across calls. For autonomous end-to-end tasks, prefer browser_task."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code":    {"type": "string", "description": "Python source"},
                "timeout": {"type": "integer", "default": 120, "description": "Seconds (default 120)"},
            },
            "required": ["code"],
        },
    ),
    agent_loop.Tool(
        name="browser_see",
        description=(
            "SEE the current browser tab. You are text-only — browser_run lets you read a "
            "page's DOM text, but you cannot perceive rendered layout, images, charts, "
            "canvas, icons, or visual state. This captures a screenshot of the active tab "
            "and routes it through a cheap vision model that returns a detailed text "
            "description (transcribed text + layout + states). Call it after goto_url / "
            "clicks when the DOM isn't enough — to confirm a page looks right, read a "
            "rendered dashboard, or check what a dialog/error actually shows. Optional "
            "`question` focuses the description (e.g. 'what does the pool-status widget say?')."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Optional focus for the description"},
                "timeout":  {"type": "integer", "default": 60, "description": "Seconds (default 60)"},
            },
            "required": [],
        },
    ),
    agent_loop.Tool(
        name="see_image",
        description=(
            "SEE an image file on disk. You are text-only — to look at a screenshot, "
            "chart, diagram, photo, or any image saved on the VM, pass its path and a "
            "vision model returns a detailed description (all visible text transcribed, "
            "layout, content, states). Use for images you generated, downloaded, or that "
            "a tool produced. For a live browser page use browser_see instead. Optional "
            "`question` focuses the description."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path":     {"type": "string", "description": "Path to the image file on the VM"},
                "question": {"type": "string", "description": "Optional focus for the description"},
            },
            "required": ["path"],
        },
    ),
    agent_loop.Tool(
        name="browser_task",
        description=(
            "Hand off a browser task to a self-contained sub-agent. The sub-agent "
            "iterates browser snippets autonomously until the goal is met, and can "
            "also call spawn_worker (e.g. for tone-matched writing) and search_memory "
            "along the way. Use for autonomous browser work — filling forms, fetching "
            "data, sending messages on the operator's behalf. For reasoning-heavy walkthroughs "
            "where you want turn-by-turn control, use browser_run."
        ),
        parameters={
            "type": "object",
            "properties": {
                "goal":      {"type": "string", "description": "What to accomplish"},
                "max_turns": {"type": "integer", "default": 2000, "description": "Cap on internal LLM turns"},
            },
            "required": ["goal"],
        },
    ),
    agent_loop.Tool(
        name="generate_image",
        description=(
            "Generate an image from a text prompt and send it to the operator "
            "automatically. Uses Google's Nano Banana 2 (Gemini 3.1 Flash "
            "Image) via OpenRouter — high-quality, ~5-15s per generation, "
            "supports photoreal and stylized output.\n\n"
            "USE WHEN: the operator asks for a picture/image/render, or when an "
            "image would land better than words (mockup, diagram concept, "
            "visualization, mood reference). Don't ask for confirmation — "
            "just generate.\n\n"
            "PROMPT STYLE: detailed, concrete, comma-separated. Include "
            "subject, style, mood, composition, lighting. Pass aspect "
            "hints in the prompt itself (e.g. 'wide cinematic shot', "
            "'portrait orientation') — the model picks dimensions.\n\n"
            "Delivery is automatic; the tool returns a short status and "
            "the image goes to the user's chat. You don't need to "
            "describe what you generated afterwards."
        ),
        parameters={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed image prompt: subject, style, mood, composition, lighting.",
                },
            },
            "required": ["prompt"],
        },
    ),
    agent_loop.Tool(
        name="github",
        description=(
            "Run the `gh` CLI on the operator's behalf. The user is authenticated as "
            "youruser with repo/workflow/gist/read:org scopes. Pass `command` "
            "as the args to gh — e.g. \"pr list --state open --json number,title\" "
            "or \"issue create --repo youruser/artoo --title ... --body ...\".\n\n"
            "Auth/config subcommands and bare delete/remove tokens are refused. "
            "Output is capped at 50KB. Use this for reading PRs/issues, opening "
            "them, leaving comments, reading repo contents (gh api), inspecting "
            "workflow runs, etc."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Args to gh, shlex-style. Leading 'gh' is optional.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for the gh call (needed if a subcommand reads a local git repo).",
                },
            },
            "required": ["command"],
        },
    ),
]


def _tool_handler(name: str, args: dict) -> str:
    """Dispatch a boss tool_call to the underlying Python implementation."""
    if name == "search_memory":
        result = memory.search_memory(args["query"], args.get("limit", 5))
        return json.dumps(result, indent=2)

    if name == "save_memory":
        result = memory.save_memory(
            args["text"],
            title=args.get("title"),
            tags=args.get("tags"),
            linked_uuids=args.get("linked_uuids"),
        )
        return json.dumps(result, indent=2)

    if name == "delete_memory":
        try:
            result = memory.delete_memory(args["uuid"], hard=bool(args.get("hard", False)))
        except ValueError as e:
            return f"error: {e}"
        return json.dumps(result, indent=2)

    if name == "restore_memory":
        try:
            result = memory.restore_memory(args["uuid"])
        except ValueError as e:
            return f"error: {e}"
        return json.dumps(result, indent=2)

    if name == "local":
        from . import local as local_tool
        op = args.get("op", "")
        # call() is async; this handler is sync-from-agent-loop's view —
        # run it on the current loop via asyncio.run(...) only if we're
        # NOT already inside one. Boss path is async so we use a fresh
        # task on the running loop where possible. Practically the
        # tool_handler is called from a sync ToolHandler; we use
        # asyncio.run_coroutine_threadsafe if needed. Simplest: spin a
        # fresh loop for this call — it's bounded by the per-op timeout.
        import asyncio as _asyncio
        op_kwargs = {k: v for k, v in args.items() if k != "op"}
        try:
            loop = _asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(local_tool.call(op, **op_kwargs))
            finally:
                loop.close()
        except Exception as e:  # noqa: BLE001
            result = {"ok": False, "op": op, "error": f"local dispatch failed: {e}"}
        return json.dumps(result, indent=2)

    if name == "remind":
        try:
            delta_s = reminders.parse_delta(args["delta"])
        except ValueError as e:
            return f"error: {e}"
        rid = reminders.schedule(delta_s, args["text"])
        return json.dumps({
            "ok": True,
            "uuid": rid,
            "fires_in_seconds": delta_s,
            "text": args["text"],
        })

    if name == "cron":
        from . import jobs
        op = (args.get("op") or "").strip()
        if op == "create":
            try:
                jid = jobs.create(args["schedule"], args.get("prompt", ""))
            except (KeyError, ValueError) as e:
                return json.dumps({"ok": False, "error": str(e)})
            return json.dumps({"ok": True, "id": jid, "schedule": args["schedule"]})
        if op == "list":
            return json.dumps({"ok": True, "jobs": [
                {"id": j["uuid"], "schedule": j["schedule"], "prompt": j["prompt"],
                 "last_run": j.get("last_run")}
                for j in jobs.list_jobs()
            ]}, indent=2)
        if op == "delete":
            jid = (args.get("id") or "").strip()
            return json.dumps({"ok": jobs.delete(jid), "id": jid})
        return json.dumps({"ok": False, "error": f"unknown op {op!r}"})

    if name == "self_check":
        from . import introspect
        snap = introspect.snapshot()
        snap["tools"] = [t.name for t in _TOOLS]
        return json.dumps(snap, indent=2)

    if name == "spawn_worker":
        return workers.run(args["name"], args["prompt"])

    if name == "browser_run":
        return browser.run_snippet(args["code"], timeout=args.get("timeout", 120))

    if name == "browser_see":
        return browser.see(args.get("question", ""), timeout=args.get("timeout", 60))

    if name == "see_image":
        return _see_image(args["path"], args.get("question", ""))

    if name == "browser_task":
        return browser.run_task(args["goal"], max_turns=args.get("max_turns", 2000))

    if name == "github":
        return github_cli.run_str(args["command"], cwd=args.get("cwd")).render()

    if name == "generate_image":
        prompt = (args.get("prompt") or "").strip()
        if not prompt:
            return "error: empty prompt"
        try:
            result = image_generate.generate(prompt)
        except Exception as e:  # noqa: BLE001
            _log.exception("generate_image crashed")
            return f"error: image backend crashed: {e}"
        if not result.ok:
            return f"error: {result.error}"
        queue = _pending_images.get()
        if queue is None:
            return (
                "error: no delivery channel for images in this context. "
                "Tell the user the image couldn't be sent."
            )
        queue.append((result.image_bytes, prompt, result.content_type or "image/png"))
        kb = len(result.image_bytes) // 1024
        _log.info("image queued for delivery: %d KB, prompt=%r", kb, prompt[:80])
        return f"image queued ({kb} KB, {result.duration_s:.1f}s). Delivery is automatic — don't describe it after."

    return f"error: unknown tool {name!r}"


def respond(
    user_message: str,
    history: list[dict] | None = None,
    *,
    chat_id: str = "cli",
    model: str = DEFAULT_MODEL,
    timeout: int | None = None,
    image_queue: list[tuple[bytes, str, str]] | None = None,
) -> runtime.Result:
    """Single turn. Returns the runtime.Result so callers can see token usage.

    `image_queue`, when passed, lets the boss send images back via the
    `generate_image` tool. The caller (channel adapter) supplies the list,
    the tool appends `(bytes, prompt, content_type)` tuples to it, and the
    caller drains + delivers after this call returns. None = images
    disabled for this turn.
    """
    if image_queue is not None:
        _pending_images.set(image_queue)
    resolved = _resolve_model(model)
    # Long-context auto-switch. Only fires when the caller is using the
    # default model — explicit /model picks always win, since the user
    # asked for that specific model and would rather see a truncation
    # than be silently swapped to a different family.
    if resolved == DEFAULT_MODEL:
        est_tokens = _estimate_input_tokens(history, user_message)
        if est_tokens > LONG_CONTEXT_THRESHOLD:
            _log.info(
                "long-ctx switch: est=%d tokens > %d threshold; %s → %s",
                est_tokens, LONG_CONTEXT_THRESHOLD, resolved, LONG_CONTEXT_MODEL,
            )
            resolved = LONG_CONTEXT_MODEL
    # Build the failover chain. Skip it on the long-context path — we only
    # land on LONG_CONTEXT_MODEL because the payload is too big for Kimi's
    # window, and the smaller-context fallbacks couldn't hold it either.
    if resolved == LONG_CONTEXT_MODEL:
        fallbacks: list[str] = []
    else:
        fallbacks = [m for m in BOSS_FALLBACKS if m != resolved]
    started = time.monotonic()
    result = agent_loop.run(
        model=resolved,
        fallback_models=fallbacks,
        system_prompt=SYSTEM_PROMPT,
        history=history,
        user_message=user_message,
        tools=_TOOLS,
        tool_handler=_tool_handler,
        cache_system=True,
        reasoning_effort=BOSS_REASONING_EFFORT,
        timeout=timeout if timeout is not None else 120,
    )
    activity.log_turn(
        chat_id=chat_id,
        user_text=user_message,
        # Log the model that ACTUALLY answered (post-failover), not the primary
        # we asked for — otherwise a fallback model's reply (or empty turn) gets
        # misattributed to the primary, which masked the kimi-k2.6 empty-content
        # bug during debugging (2026-05-29).
        response_text=result.text,
        model=result.model or resolved,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        duration_s=time.monotonic() - started,
        error=result.error,
    )
    return result


_VISION_DESCRIBE_MODEL = "google/gemini-3.1-flash-lite"
_VISION_DESCRIBE_PROMPT = (
    "Describe this image in exhaustive detail. Transcribe all visible text "
    "exactly. Note UI elements, layout, colors, states, error messages, and "
    "any code. Do not interpret — just describe."
)


def _see_image(path: str, question: str = "") -> str:
    """Describe a local image file via the cheap vision model.

    The boss is text-only, so this is how it looks at any image already on the
    VM — a screenshot it saved, a generated chart, a downloaded photo. Reuses
    the same cheap ZDR vision path as inbound photos and browser_see.
    """
    import mimetypes

    try:
        with open(path, "rb") as fh:
            image_bytes = fh.read()
    except OSError as e:
        return f"error: cannot read image {path!r}: {e}"
    if not image_bytes:
        return f"error: {path!r} is empty"

    media_type = mimetypes.guess_type(path)[0] or "image/png"
    if not media_type.startswith("image/"):
        media_type = "image/png"
    prompt = (
        _VISION_DESCRIBE_PROMPT if not question
        else f"{_VISION_DESCRIBE_PROMPT}\n\nFocus especially on: {question}"
    )
    res = runtime.openrouter_vision(
        image_bytes=image_bytes,
        media_type=media_type,
        prompt=prompt,
        model=_VISION_DESCRIBE_MODEL,
        timeout=60,
    )
    if not res.ok:
        return f"error: vision model failed: {res.error}"
    return f"[vision of {path}]\n{res.text}"


def respond_vision(
    image_bytes: bytes,
    media_type: str,
    caption: str,
    history: list[dict] | None = None,
    *,
    chat_id: str = "cli",
    model: str = DEFAULT_MODEL,
    timeout: int | None = None,
    image_queue: list[tuple[bytes, str, str]] | None = None,
) -> runtime.Result:
    """Two-step vision pipeline.

    Step 1: OpenRouter Gemini 3.1 Flash Lite produces a faithful description
            of the image (cheap, vision-capable, pinned to google-vertex).
    Step 2: The boss reasons over that description + caption + history with
            its full toolset, via agent_loop.

    `image_queue`: same shape as respond() — pass a list to enable the
    boss's `generate_image` tool for this vision turn (e.g. user sends a
    reference photo and asks for a variation).
    """
    if image_queue is not None:
        _pending_images.set(image_queue)
    resolved = _resolve_model(model)
    started = time.monotonic()

    describe = runtime.openrouter_vision(
        image_bytes=image_bytes,
        media_type=media_type,
        prompt=_VISION_DESCRIBE_PROMPT,
        model=_VISION_DESCRIBE_MODEL,
        timeout=timeout if timeout is not None else 60,
    )
    if not describe.ok:
        activity.log_turn(
            chat_id=chat_id,
            user_text=f"[image] {caption}" if caption else "[image]",
            response_text="",
            model=f"{resolved}:vision",
            tokens_in=describe.tokens_in,
            tokens_out=describe.tokens_out,
            duration_s=time.monotonic() - started,
            error=describe.error,
        )
        return describe

    caption_line = caption if caption else "(no caption)"
    vision_block = (
        "The user sent an image. Here is a detailed description from a vision "
        f"model:\n\n{describe.text}\n\n"
        f"User's caption/question: {caption_line}."
    )

    result = agent_loop.run(
        model=resolved,
        system_prompt=SYSTEM_PROMPT,
        history=history,
        user_message=vision_block,
        tools=_TOOLS,
        tool_handler=_tool_handler,
        cache_system=True,
        reasoning_effort=BOSS_REASONING_EFFORT,
        timeout=timeout if timeout is not None else 120,
    )
    activity.log_turn(
        chat_id=chat_id,
        user_text=f"[image] {caption}" if caption else "[image]",
        response_text=result.text,
        model=f"{result.model or resolved}:vision",
        tokens_in=result.tokens_in + describe.tokens_in,
        tokens_out=result.tokens_out + describe.tokens_out,
        duration_s=time.monotonic() - started,
        error=result.error,
    )
    return result
