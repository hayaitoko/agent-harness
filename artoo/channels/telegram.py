"""Telegram channel adapter.

Receives messages, dispatches to orchestrator.respond(), persists turns,
sends replies. python-telegram-bot v21 (async).

Run: `python -m artoo.channels.telegram`
Requires TELEGRAM_BOT_TOKEN in artoo/.env.
"""
from __future__ import annotations

import asyncio
import datetime
import logging
import re

from collections import deque

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    ApplicationHandlerStop,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    TypeHandler,
    filters,
)

from .. import chat_prefs, config, crons, dev_pipeline, github_cli, jobs, local, orchestrator, reminders, runtime, storage
from ..crons import duplicate_digest
from ..image import generate as image_generate
from ..scheduler import scheduler

log = logging.getLogger("artoo.telegram")

_TG_MAX = 4000  # under Telegram's 4096 hard limit, leaving headroom
_TYPING_INTERVAL = 4.0  # Telegram's "typing" lasts ~5s; refresh every 4

# Update_id dedup. Telegram redelivers an update when a handler raises or
# the bot fails to ACK in time. The orchestrator may already have run side
# effects (save_memory, local.write, /image $) before the failure, so a
# blind retry double-charges and double-applies. Track the last N
# update_ids we've started processing; reject duplicates before any
# handler runs. Set is bounded by deque length.
_DEDUP_CAP = 2048
_SEEN_UPDATE_IDS: set[int] = set()
_SEEN_UPDATE_ORDER: deque[int] = deque(maxlen=_DEDUP_CAP)


async def _dedup_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reject already-seen updates so retries can't re-trigger side effects.

    Registered as a TypeHandler in group=-1 so it runs before every other
    handler. Raising ApplicationHandlerStop tells PTB to stop dispatching
    this update entirely.
    """
    uid = getattr(update, "update_id", None)
    if uid is None:
        return  # nothing to dedup against; let it through
    if uid in _SEEN_UPDATE_IDS:
        log.info("dropping duplicate update_id=%s", uid)
        raise ApplicationHandlerStop
    if len(_SEEN_UPDATE_ORDER) == _DEDUP_CAP:
        _SEEN_UPDATE_IDS.discard(_SEEN_UPDATE_ORDER[0])
    _SEEN_UPDATE_ORDER.append(uid)
    _SEEN_UPDATE_IDS.add(uid)

# Context overflow warnings: fire once per threshold per session so the operator
# gets a heads-up before /new is forced. Cleared on /new; resets on service
# restart (so a process bounce may re-warn once — acceptable).
_CTX_LIMIT = 180_000
_CTX_THRESHOLDS = (50, 75, 90)
_CTX_WARNED: dict[str, set[int]] = {}

# Surfaced in Telegram's `/` menu via setMyCommands. One-shot slash model
# overrides (/opus, /sonnet, /haiku, /kimi) were retired in v2.15.1 —
# use /model to set a per-chat boss model preference.
_BOT_COMMANDS = [
    BotCommand("model", "Pick the boss model for this chat"),
    BotCommand("dev", "Autonomous code pipeline: /dev <task>"),
    BotCommand("build", "Multi-round project build: /build <goal>"),
    BotCommand("gh", "Run a gh CLI command: /gh pr list"),
    BotCommand("qa", "QA-test a site: /qa <url> [hints]"),
    BotCommand("image", "Generate an image: /image <prompt>"),
    BotCommand("remind", "Set a reminder: /remind 30m text"),
    BotCommand("new", "Archive conversation, start fresh"),
    BotCommand("clear", "Wipe conversation (no archive)"),
    BotCommand("ctx", "Show context window usage"),
    BotCommand("update", "Update to latest upstream (keeps your changes), test, restart"),
    BotCommand("help", "Show what I can do"),
]

# Matches leading delta tokens at the start of a string, e.g. "30m", "1d 4h".
_DELTA_PREFIX_RE = re.compile(r"^((?:\d+\s*[a-zA-Z]+\s*)+)")


def _model_for_chat(chat_id: str) -> str:
    """Resolve the boss model for this chat: per-chat preference or DEFAULT_MODEL."""
    pref = chat_prefs.get_model(chat_id)
    return pref or orchestrator.DEFAULT_MODEL


def _humanize_model(model_id: str) -> str:
    """Render a model id with its friendly alias if known: 'kimi (moonshotai/kimi-k2.6)'."""
    for alias, full in orchestrator._MODEL_ALIASES.items():
        if full == model_id:
            return f"{alias} ({model_id})"
    return model_id


# Display order + friendly labels for the /model picker. Order = button order.
# Labels are kept short so they don't overflow Telegram inline-button width.
# Opus dropped 2026-05-21 — never used as a /model choice and the alias
# was removed from orchestrator._MODEL_ALIASES. Opus is still wired in
# round_pipeline/escalation.py for wheelspin recovery (separate path).
_MODEL_PICKER: list[tuple[str, str]] = [
    ("kimi",    "Kimi K2.6"),
    ("sonnet",  "Sonnet 4.6"),
    ("glm",     "GLM-5.1"),
    ("llama4",  "Llama 4 Maverick"),
    ("haiku",   "Haiku 4.5"),
    ("gpt-oss", "GPT-OSS 120B"),
]


def _model_keyboard(current: str) -> InlineKeyboardMarkup:
    """Inline keyboard for /model. ✓ marks the current selection."""
    aliases = orchestrator._MODEL_ALIASES
    buttons = []
    for alias, label in _MODEL_PICKER:
        full = aliases.get(alias)
        if full is None:
            continue
        marker = "✓ " if current == full else ""
        buttons.append(InlineKeyboardButton(f"{marker}{label}", callback_data=f"model:set:{alias}"))
    # Pack 2 per row; trailing odd buttons get their own row.
    rows = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
    rows.append([InlineKeyboardButton("↺ Reset to global default", callback_data="model:reset")])
    return InlineKeyboardMarkup(rows)


async def _typing_loop(bot, chat_id: int) -> None:
    """Keep refreshing the typing indicator while orchestrator runs."""
    try:
        while True:
            try:
                await bot.send_chat_action(chat_id=chat_id, action="typing")
            except Exception as e:  # noqa: BLE001 — best-effort
                log.debug("typing refresh failed: %s", e)
            await asyncio.sleep(_TYPING_INTERVAL)
    except asyncio.CancelledError:
        return


_TITLE_MODEL = "openai/gpt-oss-120b"   # via fireworks per PROVIDER_PINS — cheap, fast
_TITLE_SYSTEM = (
    "You title conversations. Read the conversation, reply with ONLY a short "
    "descriptive title (5 words max). No quotes, no trailing punctuation, no "
    "preamble, no commentary."
)
_TITLE_PROMPT_TEMPLATE = "Conversation:\n\n{conv}\n\nTitle:"


def _format_turns_for_title(turns: list[tuple[str, str]]) -> str:
    """Compact turn rendering for the title prompt — capped per-turn to keep
    the prompt small. 500 chars per turn × 10 turns = 5K chars, well below
    any sensible context budget."""
    return "\n".join(f"{role}: {content[:500]}" for role, content in turns)


async def cmd_new(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Archive the current session with an AI-generated title and open a new one.

    Old conversations stay in the DB (queryable for review) but stop bleeding
    into new context. If there's nothing to archive (empty session), we still
    rotate so the warning counters reset cleanly.
    """
    chat_id = str(update.effective_chat.id)
    last_turns = storage.raw_history(chat_id, limit=10)

    title: str | None = None
    if last_turns:
        prompt = _TITLE_PROMPT_TEMPLATE.format(conv=_format_turns_for_title(last_turns))
        try:
            # Direct OR call — no boss persona, no tool schemas. Going through
            # orchestrator.respond used to bring the full delegation guide +
            # 6 tools along, eating tokens and occasionally tool-calling
            # instead of replying with a title.
            result = await asyncio.to_thread(
                runtime.openrouter,
                prompt,
                model=_TITLE_MODEL,
                system_prompt=_TITLE_SYSTEM,
                max_tokens=32,
                timeout=30,
            )
            if not result.error and result.text:
                # Defensive cleanup: model sometimes adds quotes / trailing
                # punctuation / extra lines despite instructions.
                title = result.text.strip().splitlines()[0].strip().strip('"').strip("'")
                title = title.rstrip(".!?,;:").strip()
                if not title:
                    title = None
                else:
                    title = title[:100]
        except Exception as e:  # noqa: BLE001 — best-effort
            log.error("title gen failed[%s]: %s", chat_id, e)

    storage.new_session(chat_id, title=title)
    _CTX_WARNED.pop(chat_id, None)
    log.info("session archived[%s] title=%r turns=%d", chat_id, title, len(last_turns))

    if not last_turns:
        await update.message.reply_text("(new session started — nothing to archive)")
    elif title:
        await update.message.reply_text(
            f"📁 Archived: \"{title}\"\n\n(new session started)"
        )
    else:
        await update.message.reply_text(
            "(new session started — couldn't generate a title, prior turns archived untitled)"
        )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/clear — wipe the current session entirely. No archive, no title.

    Unlike /new (which archives the conversation under a generated title for
    later review), /clear deletes the turns + session row outright. Prior
    archived sessions and saved memories (Qdrant/Obsidian) are untouched.
    """
    chat_id = str(update.effective_chat.id)
    deleted = storage.clear_session(chat_id)
    _CTX_WARNED.pop(chat_id, None)
    log.info("session cleared[%s] turns_deleted=%d", chat_id, deleted)
    if deleted == 0:
        await update.message.reply_text("(nothing to clear — no active session)")
    else:
        plural = "s" if deleted != 1 else ""
        await update.message.reply_text(
            f"🧹 Wiped {deleted} turn{plural}. Fresh start, no archive."
        )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lines = ["I'm Artoo. Just talk to me — I'll route to the right model.", ""]
    for c in _BOT_COMMANDS:
        lines.append(f"/{c.command} — {c.description}")
    lines += [
        "",
        "Default boss: Kimi K2.6. Switch with /model.",
        "Workers: DeepSeek (deep), Mercury (fast/quick), Claude Sonnet (general).",
        "/dev runs an autonomous code pipeline (DeepSeek writes, Sonnet reviews).",
    ]
    await update.message.reply_text("\n".join(lines))


async def cmd_remind(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/remind <delta> <text> — schedule a one-shot reminder.

    Examples:
        /remind 30m call mum
        /remind 1h 30m check the oven
        /remind 2d submit invoice
    """
    msg = update.effective_message
    chat_id = str(update.effective_chat.id)

    if not context.args:
        await msg.reply_text(
            "Usage: /remind <time> <text>\n"
            "e.g. /remind 30m call mum\n"
            "     /remind 1h check the oven\n"
            "     /remind 2d submit invoice"
        )
        return

    full = " ".join(context.args)
    m = _DELTA_PREFIX_RE.match(full)
    if not m:
        await msg.reply_text("Couldn't parse a time from that. Try: /remind 30m your text here")
        return

    delta_str = m.group(1).strip()
    reminder_text = full[m.end():].strip()

    if not reminder_text:
        await msg.reply_text("Need some text after the time. e.g. /remind 30m call mum")
        return

    try:
        delta_s = reminders.parse_delta(delta_str)
    except ValueError as e:
        await msg.reply_text(f"Couldn't parse time {delta_str!r}: {e}")
        return

    reminders.schedule(delta_s, reminder_text, chat_id=chat_id)

    fire_at = datetime.datetime.now() + datetime.timedelta(seconds=delta_s)
    fire_str = fire_at.strftime("%-I:%M %p").lower()
    if delta_s >= 86400:
        fire_str = fire_at.strftime("%b %-d at %-I:%M %p")

    await msg.reply_text(f"⏰ Reminder set for {fire_str} — \"{reminder_text}\"")


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/model — show current boss model + inline keyboard to switch.

    Persists per-chat in artoo.db (chat_prefs table). Use ↺ Reset to
    revert to the global DEFAULT_MODEL.
    """
    msg = update.effective_message
    chat_id_str = str(update.effective_chat.id)
    current = _model_for_chat(chat_id_str)
    is_global = chat_prefs.get_model(chat_id_str) is None

    body = (
        f"🤖 Boss model for this chat\n"
        f"  current: {_humanize_model(current)}"
        f"{' (using global default)' if is_global else ''}\n\n"
        f"Tap to switch:"
    )
    await msg.reply_text(body, reply_markup=_model_keyboard(current))


async def handle_model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process inline keyboard taps from /model. Edits the message in place."""
    query = update.callback_query
    if query is None or query.message is None:
        return
    chat_id_str = str(query.message.chat.id)
    data = query.data or ""

    if data == "model:reset":
        chat_prefs.set_model(chat_id_str, None)
        new_model = orchestrator.DEFAULT_MODEL
        await query.answer("Reset to global default")
    elif data.startswith("model:set:"):
        alias = data.split(":", 2)[2]
        full = orchestrator._MODEL_ALIASES.get(alias)
        if full is None:
            await query.answer(f"unknown alias {alias!r}", show_alert=True)
            return
        chat_prefs.set_model(chat_id_str, full)
        new_model = full
        await query.answer(f"Boss is now {alias}")
    else:
        await query.answer("unknown action", show_alert=True)
        return

    is_global = chat_prefs.get_model(chat_id_str) is None
    body = (
        f"🤖 Boss model for this chat\n"
        f"  current: {_humanize_model(new_model)}"
        f"{' (using global default)' if is_global else ''}\n\n"
        f"Tap to switch:"
    )
    try:
        await query.edit_message_text(body, reply_markup=_model_keyboard(new_model))
    except Exception as e:  # noqa: BLE001 — message may have been edited concurrently
        log.debug("/model edit_message_text failed: %s", e)


# Tasks under this many chars route through propose_spec first — Sonnet
# drafts a 3-5 bullet spec for confirmation. Long, detailed briefs run
# straight through. Borrowed from obra/superpowers' brainstorming pattern.
_DEV_SPEC_THRESHOLD = 150

# Flags consumed by /dev — must appear before the task body.
_DEV_FLAGS = {"--sec", "-s"}


def _parse_dev_args(args: list[str]) -> tuple[str, bool]:
    """Strip leading /dev flags from args; return (task, security_review)."""
    security = False
    i = 0
    while i < len(args) and args[i] in _DEV_FLAGS:
        if args[i] in ("--sec", "-s"):
            security = True
        i += 1
    return " ".join(args[i:]), security


async def cmd_dev(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/dev [--sec] <task> — run the autonomous code pipeline.

    DeepSeek v4 Pro writes, executor runs, Sonnet 4.6 reviews, loops until
    APPROVED or max_rounds (5). The boss is bypassed — this is a direct
    pipeline invocation. Returns the final code (inline if short, file if
    long) plus a summary line.

    Short tasks (< 150 chars) go through a spec-confirm step first: Sonnet
    drafts a tight spec, user taps ▶ Run or ✕ Cancel. Long tasks run
    directly.

    --sec / -s enables an OWASP/STRIDE security audit on the final code
    (one extra Sonnet call, ~$0.001). Findings are reported alongside
    the code; they don't gate APPROVED.
    """
    msg = update.effective_message
    if not context.args:
        await msg.reply_text(
            "Usage: /dev [--sec] <task>\n"
            "e.g. /dev write a Python function that flattens a nested list\n"
            "     /dev --sec write a function that hashes passwords\n\n"
            "DeepSeek writes the code, executor runs it, Sonnet reviews each "
            "round, iterates until approved or 5 rounds. Short tasks get a "
            "spec-confirm prompt first. --sec adds an OWASP/STRIDE pass."
        )
        return

    task, security_review = _parse_dev_args(list(context.args))
    if not task:
        await msg.reply_text("❌ /dev needs a task after the flags")
        return

    if len(task) < _DEV_SPEC_THRESHOLD:
        await msg.reply_text("📝 drafting spec to confirm before burning pipeline rounds…")
        try:
            spec, _ = await asyncio.to_thread(dev_pipeline.propose_spec, task)
        except Exception as e:  # noqa: BLE001
            log.exception("spec proposer crashed")
            await msg.reply_text(f"❌ spec proposer crashed: {e} — running directly")
            await _run_dev_pipeline(msg, task, spec="", security_review=security_review)
            return

        if not spec:
            # Sonnet judged task already clear → straight to pipeline.
            await _run_dev_pipeline(msg, task, spec="", security_review=security_review)
            return

        # Stash in per-user data so the button callback can recover it.
        context.user_data["pending_dev"] = {
            "task": task,
            "spec": spec,
            "security_review": security_review,
        }
        sec_chip = "  [security audit on]" if security_review else ""
        await msg.reply_text(
            f"🧭 Proposed spec for: \"{task}\"{sec_chip}\n\n{spec}\n\n"
            "Tap ▶ Run to start, or ✕ Cancel and re-issue /dev with a more "
            "specific task.",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("▶ Run", callback_data="dev:run"),
                InlineKeyboardButton("✕ Cancel", callback_data="dev:cancel"),
            ]]),
        )
        return

    await _run_dev_pipeline(msg, task, spec="", security_review=security_review)


async def _run_dev_pipeline(msg, task: str, *, spec: str, security_review: bool = False) -> None:
    """Start the dev pipeline and reply with the result. `spec` (if set)
    is passed as `context=` so the worker sees the agreed brief.
    `security_review` toggles the post-approval OWASP/STRIDE pass.
    """
    preview = task if len(task) <= 100 else task[:97] + "..."
    chip = "  🛡 security audit on" if security_review else ""
    await msg.reply_text(f"⚙️ dev pipeline started: \"{preview}\"{chip}\nThis can take a few minutes on harder tasks.")

    try:
        result = await asyncio.to_thread(
            dev_pipeline.run, task, context=spec, security_review=security_review,
        )
    except Exception as e:  # noqa: BLE001
        log.exception("dev pipeline crashed")
        await msg.reply_text(f"❌ pipeline crashed: {e}")
        return

    status = (
        f"✅ approved in {result.rounds} round{'s' if result.rounds != 1 else ''}"
        if result.approved
        else f"❌ unapproved after {result.rounds} rounds"
    )
    summary = f"{status} — cost ${result.cost_usd:.5f}"
    if result.error:
        summary += f"\nerror: {result.error}"

    code = result.code or "(no code produced)"

    body_with_summary = f"{summary}\n\n```\n{code}\n```"
    if len(body_with_summary) <= _TG_MAX:
        try:
            await msg.reply_text(body_with_summary, parse_mode="Markdown")
        except Exception:
            await msg.reply_text(body_with_summary)
    else:
        from io import BytesIO
        buf = BytesIO(code.encode("utf-8"))
        buf.name = "dev_output.py"
        await msg.reply_document(buf, caption=summary)

    if result.security is not None:
        chip = "✅ clean" if result.security.clean else "⚠️ findings"
        sec_body = f"🛡 security audit: {chip}\n\n{result.security.report}"
        for i in range(0, len(sec_body), _TG_MAX):
            await msg.reply_text(sec_body[i:i + _TG_MAX])


async def handle_dev_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Inline-keyboard callbacks from the /dev spec-confirm prompt."""
    query = update.callback_query
    if query is None or query.message is None:
        return
    data = query.data or ""
    pending = context.user_data.get("pending_dev") if context.user_data else None

    if data == "dev:cancel":
        context.user_data.pop("pending_dev", None)
        await query.answer("cancelled")
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception as e:  # noqa: BLE001
            log.debug("/dev cancel edit failed: %s", e)
        return

    if data == "dev:run":
        if not pending:
            await query.answer("nothing pending — re-issue /dev", show_alert=True)
            return
        await query.answer("starting…")
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception as e:  # noqa: BLE001
            log.debug("/dev run edit failed: %s", e)
        context.user_data.pop("pending_dev", None)
        await _run_dev_pipeline(
            query.message,
            pending["task"],
            spec=pending["spec"],
            security_review=pending.get("security_review", False),
        )
        return

    await query.answer("unknown action", show_alert=True)


# ── /build: multi-task round-gated build pipeline ─────────────────────
#
# Autonomous-with-interrupts. The engine self-halts on planner-requested
# human input, ≥5 reviewer errors in a round, fixer failure, or thrashing.
# User can /build abort to cancel cooperatively. After a halt the chat can
# /build resume [notes] to inject guidance and resume — notes are
# appended to PROGRESS.md (the planner reads its tail every round).

import json as _json
import pathlib as _pathlib
from ..conductor.progress import create_initial, read_progress as _read_progress

_BUILD_PROJECTS_ROOT = config.BUILDS_DIR  # Artoo's build workshop (see config.BUILDS_DIR)
_BUILD_SPEC_THRESHOLD = 150     # shorter goals get a Sonnet-drafted spec first


_BUILD_CHAT_STATE = config.DATA_DIR / "build_chats.json"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str, max_words: int = 6) -> str:
    words = _SLUG_RE.sub("-", text.lower()).strip("-").split("-")
    words = [w for w in words if w][:max_words] or ["project"]
    return "-".join(words)[:60]


def _unique_project_dir(goal: str) -> _pathlib.Path:
    _BUILD_PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    base = _slugify(goal)
    candidate = _BUILD_PROJECTS_ROOT / base
    i = 2
    while candidate.exists():
        candidate = _BUILD_PROJECTS_ROOT / f"{base}-{i}"
        i += 1
    return candidate


def _load_chat_state() -> dict:
    if not _BUILD_CHAT_STATE.exists():
        return {}
    try:
        return _json.loads(_BUILD_CHAT_STATE.read_text())
    except (_json.JSONDecodeError, OSError):
        return {}


def _save_chat_state(chat_id: int, *, project_dir: _pathlib.Path, goal: str) -> None:
    state = _load_chat_state()
    state[str(chat_id)] = {
        "project_dir": str(project_dir),
        "goal": goal,
        "updated": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    _BUILD_CHAT_STATE.write_text(_json.dumps(state, indent=2))


def _get_chat_project(chat_id: int) -> _pathlib.Path | None:
    entry = _load_chat_state().get(str(chat_id))
    if not entry:
        return None
    p = _pathlib.Path(entry["project_dir"])
    return p if p.exists() else None


def _enumerate_projects() -> list[dict]:
    """Return one entry per project directory under _BUILD_PROJECTS_ROOT
    that has a project_state.json (= a real build project, not a stray
    directory). Sorted by most-recently-modified state file first so the
    menu surfaces the active work at the top.

    Entry shape:
        {slug, project_dir, round, halted, halt_reason, updated_ts, goal}

    `goal` is best-effort: pulled from data/build_chats.json if any chat
    has this dir as its last project, else empty. The menu shows what's
    available regardless.
    """
    if not _BUILD_PROJECTS_ROOT.exists():
        return []
    chat_state = _load_chat_state()
    # Map project_dir → goal (last-seen across all chats).
    dir_to_goal: dict[str, str] = {}
    for entry in chat_state.values():
        if isinstance(entry, dict):
            pd = entry.get("project_dir")
            goal = entry.get("goal", "")
            if pd:
                dir_to_goal.setdefault(pd, goal)

    out: list[dict] = []
    for child in _BUILD_PROJECTS_ROOT.iterdir():
        if not child.is_dir():
            continue
        # A project is any dir with a PROGRESS.md (conductor) — or a legacy
        # project_state.json that still parses (corrupt-only dirs are skipped).
        progress_path = child / "PROGRESS.md"
        state_path = child / "project_state.json"
        round_n, halted = 0, False
        if progress_path.is_file():
            marker = progress_path
            if state_path.is_file():  # legacy metadata, best-effort
                try:
                    raw = _json.loads(state_path.read_text())
                    round_n = int(raw.get("current_round", 0) or 0)
                    halted = bool(raw.get("halted", False))
                except (_json.JSONDecodeError, OSError):
                    pass
        elif state_path.is_file():
            try:
                raw = _json.loads(state_path.read_text())
                round_n = int(raw.get("current_round", 0) or 0)
                halted = bool(raw.get("halted", False))
            except (_json.JSONDecodeError, OSError):
                continue  # corrupt legacy-only dir — not a usable project
            marker = state_path
        else:
            continue
        out.append({
            "slug": child.name,
            "project_dir": child,
            "round": round_n,
            "halted": halted,
            "updated_ts": marker.stat().st_mtime,
            "goal": dir_to_goal.get(str(child), ""),
        })
    out.sort(key=lambda e: e["updated_ts"], reverse=True)
    return out


def _format_project_list(entries: list[dict]) -> str:
    if not entries:
        return (
            "no /build projects yet. Issue `/build <goal>` to start one, "
            "or `/build clone <owner/repo> <task>` to start against a repo."
        )
    lines = [f"📁 *{len(entries)} build project(s)*", ""]
    for e in entries:
        status = "🛑 halted" if e["halted"] else "▶️ in flight"
        head = f"• `{e['slug']}` — round {e['round']} · {status}"
        if e["goal"]:
            head += f"\n     goal: {e['goal'][:80]}"
        lines.append(head)
    lines.append("")
    lines.append("Tap one with `/build switch`, or `/build resume` to pick up the last one for this chat.")
    return "\n".join(lines)


# Cap on the inline keyboard size — Telegram clients render long button
# stacks poorly on mobile, and >12 projects is a sign you want the text
# `/build list` view anyway.
_BUILD_SWITCH_MAX_BUTTONS = 12


def _switch_keyboard(entries: list[dict]) -> InlineKeyboardMarkup:
    """Inline keyboard for /build switch. One button per project, two
    columns. Bottom row is a Cancel button so taps don't leave the
    chat in a "pending action" feel.
    """
    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for e in entries[:_BUILD_SWITCH_MAX_BUTTONS]:
        # Slug + round number is enough — full goal is in the body text.
        # Halted marker so you can tell at a glance.
        marker = "🛑" if e["halted"] else "▶️"
        label = f"{marker} {e['slug'][:24]} (r{e['round']})"
        row.append(InlineKeyboardButton(label, callback_data=f"build:switch:{e['slug']}"))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("✕ Cancel", callback_data="build:switch_cancel")])
    return InlineKeyboardMarkup(rows)


async def _drive_conductor(msg, project_dir, goal: str) -> None:
    """Run the agentic conductor (one long run to verified-green) and post a
    compact result. Unlike _drive_build there are no discrete rounds to stream;
    the conductor self-bounds via cycles + a budget cap, and the verify gate is
    the definition of done."""
    import asyncio as _asyncio

    from ..conductor import run_build

    await msg.reply_text(
        f"🎼 conductor building\nGoal: {goal[:200]}\nDir: {project_dir}\n"
        "Sonnet drives, cheap workers do the bulk, done = verify green. Working…"
    )
    try:
        res = await _asyncio.to_thread(run_build, str(project_dir), goal)
    except Exception as e:  # noqa: BLE001
        await msg.reply_text(f"💥 conductor crashed: {type(e).__name__}: {e}")
        return

    head = "✅ verified green" if res.ok else f"⛔ halted: {res.halt_reason}"
    files = ", ".join(res.files_written[:20]) or "(none)"
    body = (
        f"{head}\nDir: {res.project_dir}\n"
        f"cycles: {res.cycles}   cost: ~${res.cost_usd:.3f}\n"
        f"files: {files}\n\n{(res.verify_summary or '')[:1500]}"
    )
    if res.error:
        body += f"\n\nerror: {res.error[:300]}"

    markup = None
    if not res.ok and res.halt_reason == "budget":
        from ..conductor import budget as _budget
        body += f"\n\n{_budget.status_line()}"
        markup = InlineKeyboardMarkup([[
            InlineKeyboardButton("➕ Add $5 & resume", callback_data="budget:add5_resume"),
        ]])
    await msg.reply_text(body[:4000], reply_markup=markup)


async def _start_new_build(msg, goal: str, spec: str) -> None:
    project_dir = _unique_project_dir(goal)
    project_dir.mkdir(parents=True, exist_ok=True)
    spec_body = f"# Spec\n\nGoal: {goal}\n"
    if spec:
        spec_body += f"\n{spec}\n"
    (project_dir / "SPEC.md").write_text(spec_body)
    # Conductor reads SPEC.md/PROGRESS.md off disk; seed PROGRESS.md and go.
    create_initial(project_dir, goal)
    _save_chat_state(msg.chat_id, project_dir=project_dir, goal=goal)
    await _drive_conductor(msg, project_dir, goal)


_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


async def cmd_build(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/build <goal>                     — build/dev a project via the conductor
    /build clone <owner/repo> <task>  — clone repo first, then work against it
    /build status                     — show this chat's project + PROGRESS tail
    /build resume [notes]             — re-run the conductor on this chat's project
    /build list                       — list every project on disk
    /build switch                     — pick a different project for this chat
    """
    msg = update.effective_message
    args = list(context.args or [])

    if not args:
        await msg.reply_text(
            "Usage:\n"
            "  /build <goal>                       — build/dev a project (agentic conductor)\n"
            "  /build clone <owner/repo> <task>    — clone a repo, then work against it\n"
            "  /build status                       — show this chat's project + PROGRESS tail\n"
            "  /build resume [notes]               — re-run the conductor on this chat's project\n"
            "  /build list                         — list every project in the build workshop\n"
            "  /build switch                       — pick a different project for this chat\n\n"
            "Conductor: Sonnet drives · cheap workers do the bulk · done = verify green.\n"
            "Spend is capped by a daily budget — see /budget.\n"
            f"Projects land in {_BUILD_PROJECTS_ROOT}/<slug>/"
        )
        return

    sub = args[0].lower()
    chat_id = msg.chat_id

    if sub == "abort":
        await msg.reply_text(
            "conductor runs aren't round-interruptible — they self-bound by cycle "
            "+ daily budget caps. Use /budget to cap spend."
        )
        return

    if sub == "status":
        project_dir = _get_chat_project(chat_id)
        if project_dir is None:
            await msg.reply_text("no project for this chat yet. `/build <goal>` to start one.")
            return
        tail = (_read_progress(project_dir) or "(no PROGRESS.md yet)")[-1500:]
        await msg.reply_text(f"project: {project_dir}\n\n— PROGRESS.md tail —\n{tail}")
        return

    if sub == "list":
        entries = _enumerate_projects()
        await msg.reply_text(_format_project_list(entries), parse_mode="Markdown")
        return

    if sub == "switch":
        entries = _enumerate_projects()
        if not entries:
            await msg.reply_text("no projects yet. Issue `/build <goal>` to start one.")
            return
        head = f"📂 *Switch active project for this chat* ({len(entries)} available)\n"
        if len(entries) > _BUILD_SWITCH_MAX_BUTTONS:
            head += (
                f"\nShowing the {_BUILD_SWITCH_MAX_BUTTONS} most-recent — "
                f"use `/build list` to see all."
            )
        head += "\n\nTap one to make it the *resume target* for this chat."
        await msg.reply_text(
            head, parse_mode="Markdown", reply_markup=_switch_keyboard(entries),
        )
        return

    if sub == "resume":
        project_dir = _get_chat_project(chat_id)
        if project_dir is None:
            await msg.reply_text("no prior project for this chat. `/build <goal>` to start one.")
            return
        notes = " ".join(args[1:]).strip()
        # The conductor re-orients from PROGRESS.md/SPEC.md; notes (if any) steer it.
        goal = notes or "Continue this project per PROGRESS.md and SPEC.md until verify is green."
        await _drive_conductor(msg, project_dir, goal)
        return

    if sub == "clone":
        if len(args) < 2 or not _REPO_RE.match(args[1]):
            await msg.reply_text(
                "Usage: /build clone <owner/repo> <task description>\n"
                "e.g. /build clone youruser/agent-interface read HANDOFF.md and continue the work"
            )
            return
        repo = args[1]
        task = " ".join(args[2:]).strip()
        if not task:
            await msg.reply_text("need a task description after the repo")
            return

        repo_name = repo.split("/", 1)[1]
        project_dir = _BUILD_PROJECTS_ROOT / repo_name
        _BUILD_PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)

        if (project_dir / ".git").exists():
            await msg.reply_text(f"📦 {repo} already cloned at {project_dir} — skipping clone")
        else:
            await msg.reply_text(f"📥 cloning {repo} → {project_dir}…")
            clone_res = await asyncio.to_thread(
                github_cli.run, ["repo", "clone", repo, str(project_dir)], timeout=180,
            )
            if not clone_res.ok:
                await msg.reply_text(f"❌ clone failed:\n{clone_res.render()[:_TG_MAX - 50]}")
                return

        if not (project_dir / "PROGRESS.md").exists():
            create_initial(project_dir, task)
        _save_chat_state(chat_id, project_dir=project_dir, goal=task)
        await _drive_conductor(msg, project_dir, task)
        return

    # New run.
    goal = " ".join(args).strip()
    if len(goal) < _BUILD_SPEC_THRESHOLD:
        await msg.reply_text("📝 drafting spec to confirm before starting…")
        try:
            spec, _ = await asyncio.to_thread(dev_pipeline.propose_spec, goal)
        except Exception:  # noqa: BLE001
            log.exception("build spec proposer crashed")
            spec = ""

        if not spec:
            await _start_new_build(msg, goal, spec="")
            return

        context.user_data["pending_build"] = {"goal": goal, "spec": spec}
        await msg.reply_text(
            f"🧭 Proposed spec:\n\n{spec[:_TG_MAX - 200]}\n\nTap ▶ Run to start.",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("▶ Run", callback_data="build:run"),
                InlineKeyboardButton("✕ Cancel", callback_data="build:cancel"),
            ]]),
        )
        return

    await _start_new_build(msg, goal, spec="")


async def handle_build_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None or query.message is None:
        return
    data = query.data or ""
    pending = context.user_data.get("pending_build") if context.user_data else None

    if data == "build:cancel":
        context.user_data.pop("pending_build", None)
        await query.answer("cancelled")
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception as e:  # noqa: BLE001
            log.debug("/build cancel edit failed: %s", e)
        return

    if data == "build:run":
        if not pending:
            await query.answer("nothing pending — re-issue /build", show_alert=True)
            return
        await query.answer("starting…")
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception as e:  # noqa: BLE001
            log.debug("/build run edit failed: %s", e)
        context.user_data.pop("pending_build", None)
        await _start_new_build(query.message, pending["goal"], spec=pending["spec"])
        return

    if data == "build:switch_cancel":
        await query.answer("cancelled")
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception as e:  # noqa: BLE001
            log.debug("/build switch cancel edit failed: %s", e)
        return

    if data.startswith("build:switch:"):
        slug = data[len("build:switch:"):]
        chat_id = query.message.chat.id
        candidate = _BUILD_PROJECTS_ROOT / slug
        is_project = candidate.is_dir() and (
            (candidate / "PROGRESS.md").is_file()
            or (candidate / "project_state.json").is_file()
        )
        if not is_project:
            await query.answer(f"project not found: {slug}", show_alert=True)
            return
        # Preserve the existing goal text if we have one in chat-state
        # for this dir, else carry an empty string. Goal is purely for
        # /build status display; it doesn't change planner behavior.
        goal = ""
        for entry in _load_chat_state().values():
            if isinstance(entry, dict) and entry.get("project_dir") == str(candidate):
                goal = entry.get("goal", "")
                break
        _save_chat_state(chat_id, project_dir=candidate, goal=goal)
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception as e:  # noqa: BLE001
            log.debug("/build switch select edit failed: %s", e)
        await query.answer(f"switched to {slug}")
        await query.message.reply_text(
            f"✅ active project for this chat is now `{slug}`\n"
            f"`/build resume` to continue · `/build status` to see state",
            parse_mode="Markdown",
        )
        return

    await query.answer("unknown action", show_alert=True)


async def handle_duplicate_digest_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process Forget-A / Forget-B / Merge / Ignore taps from the
    weekly duplicate-digest keyboard.

    Callback data shape: ``dup:<action>:<pair_id>`` — set up by
    `crons/duplicate_digest._format_keyboards`. Resolution is delegated
    to `duplicate_digest.resolve_action` so the cron module owns the
    pending-state file format.
    """
    query = update.callback_query
    if query is None or query.message is None:
        return
    parts = (query.data or "").split(":", 2)
    if len(parts) != 3 or parts[0] != "dup":
        await query.answer("unrecognized action", show_alert=True)
        return
    _, action, pair_id = parts

    if action not in {"forget_a", "forget_b", "merge", "ignore"}:
        await query.answer(f"unknown action: {action}", show_alert=True)
        return

    result = await asyncio.to_thread(duplicate_digest.resolve_action, action, pair_id)

    if not result.get("ok"):
        await query.answer(result.get("error", "failed"), show_alert=True)
        return

    # Disable the keyboard on the resolved pair so it can't be tapped
    # again. Best-effort — non-fatal if Telegram rejects the edit (e.g.
    # message too old).
    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except Exception as e:  # noqa: BLE001
        log.debug("duplicate digest edit failed: %s", e)

    if action == "forget_a":
        await query.answer("A deleted")
        confirm = f"🗑 forgot A (`{pair_id}`). Kept B: `{result.get('kept', '')[:8]}`."
    elif action == "forget_b":
        await query.answer("B deleted")
        confirm = f"🗑 forgot B (`{pair_id}`). Kept A: `{result.get('kept', '')[:8]}`."
    elif action == "merge":
        await query.answer("merged into A")
        confirm = (
            f"🔗 merged `{pair_id}` into A (`{result.get('kept', '')[:8]}`). "
            "B hard-deleted; if you wanted A's content updated with B's text, "
            "tell the boss and it'll call save_memory."
        )
    else:  # ignore
        await query.answer("ignored 30d")
        confirm = f"⏭ ignoring `{pair_id}` for 30 days."

    await query.message.reply_text(confirm, parse_mode="Markdown")


async def cmd_budget(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/budget — show today's conductor spend/cap.
    /budget +5      — raise today's cap by $5
    /budget set 10  — set today's cap to $10
    """
    from ..conductor import budget

    msg = update.effective_message
    args = list(context.args or [])
    if not args:
        await msg.reply_text(
            budget.status_line() + "\n\n`/budget +5` to add · `/budget set 10` to override",
            parse_mode="Markdown",
        )
        return

    a = args[0].lower().lstrip("+")
    try:
        if args[0].startswith("+"):
            cap = budget.bump_cap(float(a or (args[1] if len(args) > 1 else "")))
        elif a == "add" and len(args) > 1:
            cap = budget.bump_cap(float(args[1]))
        elif a == "set" and len(args) > 1:
            cap = budget.set_cap(float(args[1]))
        else:
            await msg.reply_text("usage: `/budget` · `/budget +5` · `/budget set 10`",
                                 parse_mode="Markdown")
            return
    except (ValueError, IndexError):
        await msg.reply_text("amount must be a number — e.g. `/budget +5`", parse_mode="Markdown")
        return
    await msg.reply_text(f"✅ daily cap now ${cap:.2f}\n{budget.status_line()}")


async def handle_budget_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None or query.message is None:
        return
    if query.data != "budget:add5_resume":
        await query.answer("unknown action", show_alert=True)
        return
    from ..conductor import budget

    chat_id = query.message.chat.id
    project_dir = _get_chat_project(chat_id)
    if project_dir is None:
        await query.answer("no project for this chat to resume", show_alert=True)
        return
    new_cap = budget.bump_cap(5)
    await query.answer(f"added $5 → cap ${new_cap:.2f}, resuming")
    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except Exception as e:  # noqa: BLE001
        log.debug("budget resume button edit failed: %s", e)
    await _drive_conductor(
        query.message, project_dir,
        "Continue this project per PROGRESS.md and SPEC.md until verify is green.",
    )


# ── /gh: direct gh CLI access ──────────────────────────────────────────

async def cmd_gh(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/gh <args> — run `gh <args>` and reply with the output.

    Auth/config subcommands and bare delete/remove tokens are refused.
    Output capped at 50KB. Example: `/gh pr list --state open`.
    """
    msg = update.effective_message
    if not context.args:
        await msg.reply_text(
            "Usage: /gh <args>\n"
            "e.g. /gh pr list --state open\n"
            "     /gh issue create --repo youruser/artoo --title 'foo' --body 'bar'\n"
            "     /gh api user --jq .login\n\n"
            "Refused: auth/config subcommands, bare delete/remove tokens."
        )
        return

    cmd_preview = " ".join(context.args)[:120]
    await msg.reply_text(f"$ gh {cmd_preview}")
    result = await asyncio.to_thread(github_cli.run, list(context.args))
    body = result.render()
    if not body.strip():
        body = "(no output)"
    for i in range(0, len(body), _TG_MAX):
        chunk = body[i:i + _TG_MAX]
        try:
            await msg.reply_text(f"```\n{chunk}\n```", parse_mode="Markdown")
        except Exception:
            await msg.reply_text(chunk)


async def cmd_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/image <prompt> — text-to-image via Nano Banana 2 (Gemini 3.1 Flash Image)."""
    msg = update.effective_message
    if not context.args:
        await msg.reply_text(
            "Usage: /image <prompt>\n"
            "e.g. /image a homelab rack at golden hour, photoreal, cinematic\n\n"
            "Uses Google Nano Banana 2 (Gemini 3.1 Flash Image) via OpenRouter. "
            "Pass aspect/composition cues in the prompt itself."
        )
        return

    prompt = " ".join(context.args)
    preview = prompt if len(prompt) <= 120 else prompt[:117] + "..."
    await msg.reply_text(f"🎨 Generating image: \"{preview}\"\nThis takes ~5-15s.")

    try:
        result = await asyncio.to_thread(image_generate.generate, prompt)
    except Exception as e:  # noqa: BLE001
        log.exception("image generation crashed")
        await msg.reply_text(f"❌ image error: {e}")
        return

    await _send_image_result(msg, result, fallback_caption="image")


async def _send_image_result(msg, result, *, fallback_caption: str) -> None:
    """Common rendering for an ImageResult — handles success, errors, and file send."""
    if not result.ok:
        await msg.reply_text(f"❌ {result.error}")
        return

    from io import BytesIO
    ext = "jpg" if "jpeg" in (result.content_type or "") else "png"
    buf = BytesIO(result.image_bytes)
    buf.name = f"artoo_{fallback_caption}.{ext}"
    caption = result.model
    caption += f"  ({result.duration_s:.1f}s)"
    await msg.reply_photo(buf, caption=caption)


async def _flush_image_queue(msg, image_queue: list[tuple[bytes, str, str]]) -> None:
    """Send images that the boss queued via the generate_image tool.

    Each entry is (bytes, prompt, content_type). The prompt is used as
    the caption (truncated to Telegram's 1024-char limit).
    """
    from io import BytesIO
    for img_bytes, img_prompt, ct in image_queue:
        try:
            ext = "jpg" if "jpeg" in (ct or "") else "png"
            buf = BytesIO(img_bytes)
            buf.name = f"artoo_image.{ext}"
            caption = img_prompt[:1024] if img_prompt else None
            await msg.reply_photo(buf, caption=caption)
        except Exception:  # noqa: BLE001
            log.exception("boss-generated image send failed")
    image_queue.clear()


_QA_GOAL_TEMPLATE = (
    "QA-test the site at {url_repr}.\n\n"
    "1. goto_url({url_repr}); call page_info() to see what loaded.\n"
    "2. Exercise the golden path — whatever the primary CTA / main feature "
    "appears to be (sign-up form, search, checkout, etc.). If multiple "
    "primary actions exist, try each briefly.\n"
    "3. Take screenshot()s of: the landing state, mid-flow, and any error "
    "or success state you reach. Save them ONLY under {shots_dir} as "
    "screenshot-N.png so they can be attached. Do NOT write screenshots "
    "anywhere else.\n"
    "4. Note any of the following: JS console errors (you can read them "
    "from the harness), broken images, dead links visible above the fold, "
    "obvious accessibility misses (missing alt text on key images, "
    "buttons without labels, low-contrast text), slow loads (>3s), layout "
    "issues at default viewport.{extra}\n\n"
    "Return a structured report:\n"
    "  SUMMARY: one sentence on what the site does and overall health.\n"
    "  WORKED: bullet list of what you successfully exercised.\n"
    "  ISSUES: bullet list, severity-tagged (BLOCKER / MAJOR / MINOR / NIT).\n"
    "  SCREENSHOTS: paths of the files you saved under {shots_dir}.\n"
    "Keep it under ~1500 chars total."
)

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


async def cmd_qa(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/qa <url> [extra-instructions] — drive a URL through browser_task as a QA pass.

    Spawns the browser sub-agent with a QA-flavored goal: open the site,
    exercise the golden path, capture screenshots, return a structured
    report (SUMMARY / WORKED / ISSUES / SCREENSHOTS). Anything after the
    URL is appended to the goal as extra instructions, so:

        /qa https://example.com pay particular attention to the search bar

    Borrowed in spirit from garrytan/gstack's /qa skill, but routed through
    artoo's browser_task sub-agent instead of a separate browser stack.
    """
    msg = update.effective_message
    if not context.args:
        await msg.reply_text(
            "Usage: /qa <url> [extra-instructions]\n"
            "e.g. /qa https://example.com\n"
            "     /qa https://artoo.dev pay particular attention to /login\n\n"
            "Drives the URL through browser_task: opens the site, exercises "
            "the golden path, captures screenshots, returns a structured "
            "report. Takes 30-180s depending on site complexity."
        )
        return

    url = context.args[0]
    if not _URL_RE.match(url):
        await msg.reply_text(f"❌ first arg must be an http(s) URL, got: {url!r}")
        return
    extra = " ".join(context.args[1:]).strip()
    extra_clause = f"\n5. Extra instructions: {extra}" if extra else ""

    preview = url if len(url) <= 80 else url[:77] + "..."
    await msg.reply_text(f"🧪 QA started against {preview}\nThis can take 30-180s.")

    # Per-call dir scopes screenshots to this invocation only. json.dumps()
    # gives a quote-safe URL literal so a hostile URL can't break out of
    # the goto_url(...) call we ask the sub-agent to make.
    import glob
    import json
    import shutil
    import tempfile

    shots_dir = tempfile.mkdtemp(prefix="artoo-qa-")
    goal = _QA_GOAL_TEMPLATE.format(
        url_repr=json.dumps(url),
        shots_dir=shots_dir,
        extra=extra_clause,
    )
    try:
        try:
            report = await asyncio.to_thread(orchestrator.browser.run_task, goal, max_turns=12)
        except Exception as e:  # noqa: BLE001
            log.exception("/qa run_task crashed")
            await msg.reply_text(f"❌ QA crashed: {e}")
            return

        chunks = [report[i:i + _TG_MAX] for i in range(0, len(report), _TG_MAX)] or ["(empty report)"]
        for chunk in chunks:
            await msg.reply_text(chunk)

        for shot in sorted(glob.glob(f"{shots_dir}/*.png")):
            try:
                with open(shot, "rb") as f:
                    await msg.reply_photo(f, caption=shot)
            except Exception as e:  # noqa: BLE001
                log.debug("qa screenshot attach failed for %s: %s", shot, e)
    finally:
        shutil.rmtree(shots_dir, ignore_errors=True)


async def cmd_ctx(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = str(update.effective_chat.id)
    stats = storage.context_stats(chat_id)
    sess = storage.session_stats(chat_id)
    kept = stats["kept"]
    total = stats["total"]
    est = stats["est_tokens"]
    pct = est / _CTX_LIMIT * 100
    session_line = (
        f"Session {sess['current_index']} of {sess['total']} total"
        if sess["current_index"]
        else f"No active session yet ({sess['total']} archived)"
    )
    msg = (
        f"{session_line}\n"
        f"Context: {kept}/{total} turns active, ~{est:,} tokens "
        f"({pct:.1f}% of 180K limit)"
    )
    if stats["dropped"]:
        msg += " (oldest turns dropped — history exceeded limit)"
    await update.message.reply_text(msg)


def _is_home_chat(chat_id: str) -> bool:
    """True if this chat may run privileged ops like /update.

    Gated to TELEGRAM_HOME_CHANNEL when set. If unset (e.g. the public
    template default), the guard is open — the bot is token-gated to a
    single operator anyway.
    """
    home = config.optional("TELEGRAM_HOME_CHANNEL").strip()
    return (not home) or (chat_id == home)


# Remembers the last update's backup branch per chat so `/update rollback`
# works after a tests-red merge. In-memory: a process restart clears it, but the
# backup BRANCH persists in git, and `/update rollback <ref>` takes an explicit ref.
_LAST_UPDATE_BACKUP: dict[str, str] = {}


async def cmd_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Update to the latest upstream, preserving local customizations.

    Unlike a plain `git pull --ff-only` (which can't fast-forward the public
    mirror's orphan-snapshot history), this uses artoo.self_update: a vendor-
    branch 3-way merge that folds the upstream delta into the user's tree,
    routes any conflicts to Kimi K2.6-thinking, gates on pytest, and only
    restarts when green. Customizations are kept; a backup branch is the escape
    hatch. `/update rollback [ref]` reverts.
    """
    chat_id = str(update.effective_chat.id)
    if not _is_home_chat(chat_id):
        await update.message.reply_text("⛔ /update is restricted to the home channel.")
        return

    args = list(context.args or [])
    if args and args[0].lower() == "rollback":
        await _update_rollback(update, chat_id, args[1:])
        return

    await update.message.reply_text(
        "🔄 Updating — fetching upstream, folding changes into your customizations "
        "(Kimi resolves any conflicts), then testing before any restart…"
    )

    from .. import self_update

    res = await asyncio.to_thread(self_update.run_update, str(config.ROOT))

    # Up-to-date / first-run baseline: nothing to restart.
    if res.status == self_update.STATUS_UP_TO_DATE:
        await update.message.reply_text("✅ Already on the latest upstream.")
        return
    if res.status == self_update.STATUS_BASELINE:
        await update.message.reply_text("✅ " + res.summary)
        return

    if res.backup_ref:
        _LAST_UPDATE_BACKUP[chat_id] = res.backup_ref

    # Hard failure (rolled back) — customizations intact, nothing changed.
    if not res.ok and res.status in (self_update.STATUS_ROLLED_BACK, self_update.STATUS_ERROR):
        body = f"❌ Update failed — {res.error}\n\n{res.summary}"
        await update.message.reply_text(body[:3500])
        return

    # Merged but tests RED: per design, do NOT restart or roll back — talk it out.
    if res.status == self_update.STATUS_TESTS_FAILED:
        ai = f"\nKimi resolved: {', '.join(res.ai_resolved)}" if res.ai_resolved else ""
        body = (
            "⚠️ I merged the update but the test suite is RED, so I did NOT restart — "
            f"you're still running the old code.{ai}\n\n"
            f"Backup: `{res.backup_ref}`\n\n"
            "Tell me what to do — we can debug the failures together (I can read/edit/run "
            "via the boss), or `/update rollback` to revert. Failures:\n\n"
            f"```\n{(res.detail or '')[-1400:]}\n```"
        )
        await update.message.reply_text(body[:4000], parse_mode="Markdown")
        return

    # Clean update, tests green → restart onto the new code.
    ai = f" Kimi resolved {len(res.ai_resolved)} conflict(s)." if res.ai_resolved else ""
    await update.message.reply_text(f"✅ {res.summary}{ai}\n🧪 Tests green — restarting…")
    restart = await local.call(
        op="restart", reason=f"/update: {res.status}", test_first=False,  # engine already gated
    )
    if not restart.get("ok"):
        await update.message.reply_text(
            "⚠️ Update applied and tested green, but the restart call failed:\n"
            f"{(restart.get('error') or '')[:400]}\n\nRestart manually on the box."
        )
        return
    await update.message.reply_text(
        f"🚀 Restarting in ~{restart.get('delay_s', 3)}s onto the new code — back shortly."
    )


async def _update_rollback(update: Update, chat_id: str, extra: list[str]) -> None:
    """`/update rollback [ref]` — restore the backup branch from the last update."""
    from .. import self_update

    ref = extra[0] if extra else _LAST_UPDATE_BACKUP.get(chat_id, "")
    if not ref:
        await update.message.reply_text(
            "No backup remembered for this chat. Pass one explicitly: "
            "`/update rollback <branch>` (see `git branch --list 'artoo-update-backup-*'`).",
            parse_mode="Markdown",
        )
        return
    res = await asyncio.to_thread(self_update.rollback, str(config.ROOT), ref)
    if not res.ok:
        await update.message.reply_text(f"❌ Rollback failed: {res.error}")
        return
    await update.message.reply_text(
        f"↩️ {res.summary}\n🧪 Re-testing and restarting onto the restored code…"
    )
    restart = await local.call(op="restart", reason=f"/update rollback {ref}", test_first=True)
    if not restart.get("ok"):
        tests = restart.get("tests") or {}
        tail = (tests.get("stdout_tail") or tests.get("stderr_tail") or restart.get("error") or "")[:800]
        await update.message.reply_text(
            "⚠️ Rolled back on disk, but the post-rollback test/restart failed:\n" + tail
        )
        return
    await update.message.reply_text(
        f"🚀 Restored. Restarting in ~{restart.get('delay_s', 3)}s — back shortly."
    )


async def _maybe_warn_context(bot, chat_id_str: str, chat_id_int: int) -> None:
    """Fire a one-shot warning when context crosses 50 / 75 / 90% of the limit.

    Each threshold fires at most once per session. State lives in the
    module-level `_CTX_WARNED` dict and is cleared on /new. We pick the
    HIGHEST unwarned threshold the current usage crosses so one big jump
    (e.g. a long doc dump from 40% → 80%) skips the 50% warning rather
    than firing both back-to-back.
    """
    stats = storage.context_stats(chat_id_str)
    pct = stats["est_tokens"] / _CTX_LIMIT * 100
    warned = _CTX_WARNED.setdefault(chat_id_str, set())

    # Walk thresholds high → low so we send only the most urgent one this turn.
    for threshold in reversed(_CTX_THRESHOLDS):
        if pct >= threshold and threshold not in warned:
            warned.add(threshold)
            if threshold == 50:
                msg = "📊 Context at 50% — getting busy"
            elif threshold == 75:
                msg = "⚠️ Context at 75% — consider /new soon"
            else:
                msg = "🚨 Context at 90% — almost full, /new to start fresh"
            try:
                await bot.send_message(chat_id=chat_id_int, text=msg)
            except Exception as e:  # noqa: BLE001 — best-effort
                log.debug("context warning send failed[%s]: %s", chat_id_str, e)
            return


# Internal prompt fired on context overflow. Not from the user — Artoo runs
# this as a fresh turn (no history) so it doesn't recurse, asking the model
# to checkpoint dropped context into long-term memory via save_memory.
_COMPACTION_PROMPT = (
    "SYSTEM TASK (not from user): The conversation history just exceeded "
    "the context window. The oldest turns have been dropped. Here is a "
    "summary task: search your memory for this conversation, then save a "
    "memory summarizing the key facts, decisions, and context from this "
    "conversation that are worth preserving long-term. Be thorough — this "
    "is your only chance to retain what was dropped."
)


async def _compact_history(bot, chat_id: str) -> None:
    """Fire-and-forget compaction: snapshot dropped context into memory.

    Called when storage trimming reports that turns were dropped. Runs the
    orchestrator with NO history (so it doesn't recurse into the same
    overflow) and discards the result — we only care about the tool calls
    (memory writes) the model makes during the turn. Notifies the user via
    `bot` when the background task succeeds or fails so they know the
    compaction state without watching logs.
    """
    try:
        result = await asyncio.to_thread(
            orchestrator.respond, _COMPACTION_PROMPT, chat_id=chat_id
        )
        if result.error:
            log.error("compaction error[%s]: %s", chat_id, result.error)
            try:
                await bot.send_message(
                    chat_id=int(chat_id),
                    text="Context compaction failed — some history may have been lost.",
                )
            except Exception as notify_err:  # noqa: BLE001
                log.debug("compaction error notify failed[%s]: %s", chat_id, notify_err)
        else:
            log.info(
                "compaction completed[%s] tokens=%d/%d",
                chat_id, result.tokens_in, result.tokens_out,
            )
            try:
                await bot.send_message(
                    chat_id=int(chat_id),
                    text="Context compacted. Oldest turns saved to memory.",
                )
            except Exception as notify_err:  # noqa: BLE001
                log.debug("compaction success notify failed[%s]: %s", chat_id, notify_err)
    except Exception as e:  # noqa: BLE001 — best-effort background task
        log.error("compaction failed[%s]: %s", chat_id, e)
        try:
            await bot.send_message(
                chat_id=int(chat_id),
                text="Context compaction failed — some history may have been lost.",
            )
        except Exception as notify_err:  # noqa: BLE001
            log.debug("compaction failure notify failed[%s]: %s", chat_id, notify_err)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return

    chat_id_str = str(update.effective_chat.id)
    chat_id_int = update.effective_chat.id
    user_text = msg.text.strip()

    if not user_text:
        return

    model = _model_for_chat(chat_id_str)
    log.info("user[%s] (model=%s): %s", chat_id_str, model, user_text[:80])

    typing_task = asyncio.create_task(_typing_loop(context.bot, chat_id_int))
    image_queue: list[tuple[bytes, str, str]] = []
    try:
        history, hist_stats = storage.history_tokens_with_stats(chat_id_str)
        result = await asyncio.to_thread(
            orchestrator.respond, user_text, history,
            chat_id=chat_id_str, model=model, image_queue=image_queue,
        )
    finally:
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

    if result.error:
        log.error("orchestrator error[%s]: %s", chat_id_str, result.error)
        await msg.reply_text(f"[error: {result.error}]")
        return

    storage.append(chat_id_str, "user", user_text)
    storage.append(chat_id_str, "assistant", result.text)
    log.info("artoo[%s] tokens=%d/%d: %s", chat_id_str, result.tokens_in, result.tokens_out, result.text[:80])

    await _maybe_warn_context(context.bot, chat_id_str, chat_id_int)

    # Overflow → kick off async compaction. Don't block the user's reply;
    # the model checkpoints dropped context into long-term memory in the
    # background. The check uses pre-turn stats — the new turn we just
    # appended isn't part of what got dropped, so this is the right signal.
    if hist_stats["dropped"]:
        log.warning(
            "context overflow[%s]: %d/%d turns active — kicking off compaction",
            chat_id_str, hist_stats["kept"], hist_stats["total"],
        )
        await msg.reply_text("Context limit hit — saving key context to memory now...")
        asyncio.create_task(_compact_history(context.bot, chat_id_str))

    text = result.text or "(empty response)"
    chunks = [text[i:i + _TG_MAX] for i in range(0, len(text), _TG_MAX)]
    for chunk in chunks:
        await msg.reply_text(chunk)

    if image_queue:
        await _flush_image_queue(msg, image_queue)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Photo turn: download largest size, route via the vision path.

    Telegram serves photos in multiple resolutions; the last entry of `msg.photo`
    is the largest. Telegram always re-encodes uploaded photos as JPEG, so we
    hard-code media_type.
    """
    msg = update.effective_message
    if not msg or not msg.photo:
        return

    chat_id_str = str(update.effective_chat.id)
    chat_id_int = update.effective_chat.id
    caption = (msg.caption or "").strip()
    model = _model_for_chat(chat_id_str)

    log.info(
        "user[%s] photo (model=%s, caption=%s)",
        chat_id_str, model, caption[:80] if caption else "<none>",
    )

    photo = msg.photo[-1]  # largest size
    try:
        tg_file = await context.bot.get_file(photo.file_id)
        image_bytes = bytes(await tg_file.download_as_bytearray())
    except Exception as e:  # noqa: BLE001
        log.error("photo download failed[%s]: %s", chat_id_str, e)
        await msg.reply_text(f"[error downloading image: {e}]")
        return

    typing_task = asyncio.create_task(_typing_loop(context.bot, chat_id_int))
    image_queue: list[tuple[bytes, str, str]] = []
    try:
        history = storage.history_tokens(chat_id_str)
        result = await asyncio.to_thread(
            orchestrator.respond_vision,
            image_bytes,
            "image/jpeg",
            caption,
            history,
            chat_id=chat_id_str,
            model=model,
            image_queue=image_queue,
        )
    finally:
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

    if result.error:
        log.error("vision error[%s]: %s", chat_id_str, result.error)
        await msg.reply_text(f"[error: {result.error}]")
        return

    user_log_text = f"[image] {caption}" if caption else "[image]"
    storage.append(chat_id_str, "user", user_log_text)
    storage.append(chat_id_str, "assistant", result.text)
    log.info(
        "artoo[%s] vision tokens=%d/%d: %s",
        chat_id_str, result.tokens_in, result.tokens_out, result.text[:80],
    )

    await _maybe_warn_context(context.bot, chat_id_str, chat_id_int)

    text = result.text or "(empty response)"
    chunks = [text[i:i + _TG_MAX] for i in range(0, len(text), _TG_MAX)]
    for chunk in chunks:
        await msg.reply_text(chunk)

    if image_queue:
        await _flush_image_queue(msg, image_queue)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Document turn: image-mime documents route via the vision path.

    the operator often sends screenshots as files (not photos) so Telegram doesn't
    re-encode them as JPEG — keeps text readable. Non-image documents are
    ignored silently.
    """
    msg = update.effective_message
    if not msg or not msg.document:
        return

    mime_type = msg.document.mime_type or ""
    if not mime_type.startswith("image/"):
        return

    chat_id_str = str(update.effective_chat.id)
    chat_id_int = update.effective_chat.id
    caption = (msg.caption or "").strip()
    model = _model_for_chat(chat_id_str)

    log.info(
        "user[%s] document (mime=%s, model=%s, caption=%s)",
        chat_id_str, mime_type, model, caption[:80] if caption else "<none>",
    )

    try:
        tg_file = await context.bot.get_file(msg.document.file_id)
        image_bytes = bytes(await tg_file.download_as_bytearray())
    except Exception as e:  # noqa: BLE001
        log.error("document download failed[%s]: %s", chat_id_str, e)
        await msg.reply_text(f"[error downloading image: {e}]")
        return

    typing_task = asyncio.create_task(_typing_loop(context.bot, chat_id_int))
    image_queue: list[tuple[bytes, str, str]] = []
    try:
        history = storage.history_tokens(chat_id_str)
        result = await asyncio.to_thread(
            orchestrator.respond_vision,
            image_bytes,
            mime_type,
            caption,
            history,
            chat_id=chat_id_str,
            model=model,
            image_queue=image_queue,
        )
    finally:
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

    if result.error:
        log.error("vision error[%s]: %s", chat_id_str, result.error)
        await msg.reply_text(f"[error: {result.error}]")
        return

    user_log_text = f"[image] {caption}" if caption else "[image]"
    storage.append(chat_id_str, "user", user_log_text)
    storage.append(chat_id_str, "assistant", result.text)
    log.info(
        "artoo[%s] vision tokens=%d/%d: %s",
        chat_id_str, result.tokens_in, result.tokens_out, result.text[:80],
    )

    await _maybe_warn_context(context.bot, chat_id_str, chat_id_int)

    text = result.text or "(empty response)"
    chunks = [text[i:i + _TG_MAX] for i in range(0, len(text), _TG_MAX)]
    for chunk in chunks:
        await msg.reply_text(chunk)

    if image_queue:
        await _flush_image_queue(msg, image_queue)


async def _post_init(application: Application) -> None:
    await application.bot.set_my_commands(_BOT_COMMANDS)
    log.info("bot commands set: %s", [c.command for c in _BOT_COMMANDS])
    crons.register_all()
    jobs.load_all()  # register persisted dynamic cron jobs (artoo/jobs.py)
    scheduler.start()


def main() -> None:
    token = config.require("TELEGRAM_BOT_TOKEN")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log.info("artoo telegram adapter starting")
    app = (
        Application.builder()
        .token(token)
        .concurrent_updates(True)
        .post_init(_post_init)
        .build()
    )
    # Update_id dedup runs in group=-1 so it precedes every other handler.
    # Duplicates raise ApplicationHandlerStop and never reach a real handler.
    app.add_handler(TypeHandler(Update, _dedup_update), group=-1)
    app.add_handler(CommandHandler("new", cmd_new))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("remind", cmd_remind))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("dev", cmd_dev))
    app.add_handler(CommandHandler("build", cmd_build))
    app.add_handler(CommandHandler("budget", cmd_budget))
    app.add_handler(CommandHandler("gh", cmd_gh))
    app.add_handler(CommandHandler("image", cmd_image))
    app.add_handler(CommandHandler("qa", cmd_qa))
    app.add_handler(CommandHandler("ctx", cmd_ctx))
    app.add_handler(CommandHandler("update", cmd_update))
    app.add_handler(CallbackQueryHandler(handle_model_callback, pattern=r"^model:"))
    app.add_handler(CallbackQueryHandler(handle_dev_callback, pattern=r"^dev:"))
    app.add_handler(CallbackQueryHandler(handle_build_callback, pattern=r"^build:"))
    app.add_handler(CallbackQueryHandler(handle_budget_callback, pattern=r"^budget:"))
    app.add_handler(CallbackQueryHandler(handle_duplicate_digest_callback, pattern=r"^dup:"))
    # Image documents (uncompressed screenshots) also route to vision.
    # Registered before PHOTO so image/* docs are claimed here, not by the
    # generic text/photo handlers.
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_document))
    # Photos route through the vision path (Gemini describe → agent_loop reason).
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Everything else — plain text + retired one-shot model overrides
    # (which the boss now just sees as the literal command since /model
    # replaced them in v2.15.1).
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.run_polling()


if __name__ == "__main__":
    main()
