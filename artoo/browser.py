"""Browser-harness integration.

Drives a local headless Chromium via the browser-use/browser-harness CLI
(https://github.com/browser-use/browser-harness). Two public entry points
power the boss's `browser_run` and `browser_task` tools.

  run_snippet(code, *, timeout=120) -> str
      Pipe `code` (Python) into `browser-harness`, return combined
      stdout+stderr. The boss uses this for turn-by-turn control:
      one snippet, see result, decide the next snippet.

  run_task(goal, *, model=None, max_turns=10) -> str
      Self-contained sub-agent. A worker LLM iterates browser snippets
      against `goal` and can also call spawn_worker (e.g. for tone-matched
      writing) and search_memory mid-task. Returns the sub-agent's final
      summary.

Connection: chromium-harness.service launches Chromium with
`--remote-debugging-port=9222 --user-data-dir=$HOME/.cache/artoo/chrome-profile`.
We export BU_CDP_URL so browser-harness skips local discovery, and BU_NAME
so its daemon IPC is namespaced. Both env vars are overridable in .env.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess

from . import agent_loop, config, memory, runtime, workers

_log = logging.getLogger("artoo.browser")

_BH_BIN = config.optional("BROWSER_HARNESS_BIN") or shutil.which("browser-harness") or "browser-harness"
_BH_ENV = {
    "BU_CDP_URL": config.optional("BU_CDP_URL", "http://127.0.0.1:9222"),
    "BU_NAME":    config.optional("BU_NAME", "artoo"),
}
# The sub-agent is a tool-calling loop (browser_python, browser_see, ...), so
# the model MUST support OR function-calling. Llama 4 Maverick (the prior pick)
# does NOT tool-call on OpenRouter — OR returns "no endpoints found" the moment
# a tools array is sent, so browser_task silently 404'd on turn 0. Gemini 3.5
# Flash: 1M ctx for long browser transcripts, tool-calling verified, cheap, ZDR
# via google-vertex. Native multimodality is no longer needed here — browser_see
# routes screenshots through the cheap vision model instead.
_TASK_MODEL = config.optional("BROWSER_TASK_MODEL") or "google/gemini-3.5-flash"

# Cheap multimodal model that acts as the (text-only) boss's eyes on a page.
# Same model + ZDR path (google-vertex) the inbound-image pipeline already uses
# in orchestrator.respond_vision — see browser.see().
_VISION_MODEL = config.optional("BROWSER_VISION_MODEL") or "google/gemini-3.1-flash-lite"
_SHOT_MARKER = "__ARTOO_SHOT_B64__"
_SEE_PROMPT = (
    "Describe this web-page screenshot in exhaustive detail. Transcribe all "
    "visible text exactly. Note layout, buttons, forms, menus, tables, charts, "
    "icons, selected/active states, and any error or status messages. Do not "
    "interpret or advise — just describe what is on screen."
)


def see(question: str = "", *, timeout: int = 60) -> str:
    """Screenshot the current browser tab and return a vision-model description.

    The boss model is text-only, so a raw screenshot is useless to it. This is
    a two-step bridge mirroring orchestrator.respond_vision: step 1 grabs a PNG
    of the live tab via the harness, step 2 runs it through a cheap multimodal
    model (google/gemini-3.1-flash-lite, ZDR via google-vertex) and returns the
    description as text the boss can reason over. `question` narrows the focus.
    """
    snippet = (
        "import base64 as _b64\n"
        "_p = capture_screenshot(max_dim=1536)\n"
        f"print('{_SHOT_MARKER}' + _b64.b64encode(open(_p, 'rb').read()).decode())\n"
    )
    out = run_snippet(snippet, timeout=min(timeout, 60))
    b64 = next(
        (ln[len(_SHOT_MARKER):].strip() for ln in out.splitlines() if ln.startswith(_SHOT_MARKER)),
        None,
    )
    if not b64:
        return f"error: could not capture a screenshot. harness output:\n{out}"
    try:
        image_bytes = base64.b64decode(b64)
    except (ValueError, base64.binascii.Error) as e:
        return f"error: bad screenshot data ({e})"

    prompt = _SEE_PROMPT if not question else f"{_SEE_PROMPT}\n\nFocus especially on: {question}"
    res = runtime.openrouter_vision(
        image_bytes=image_bytes,
        media_type="image/png",
        prompt=prompt,
        model=_VISION_MODEL,
        timeout=timeout,
    )
    if not res.ok:
        return f"error: vision model failed: {res.error}"
    return f"[vision of current tab]\n{res.text}"


def run_snippet(code: str, *, timeout: int = 120) -> str:
    """Execute `code` through browser-harness, return its output.

    State (open tab, daemon, helpers) persists across calls within a Chrome
    session — the harness daemon stays up between invocations.
    """
    env = {**os.environ, **_BH_ENV}
    try:
        proc = subprocess.run(
            [_BH_BIN],
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except FileNotFoundError:
        return (
            f"error: browser-harness binary not found ({_BH_BIN!r}). "
            "Install with `uv tool install browser-harness` and ensure "
            "chromium-harness.service is running."
        )
    except subprocess.TimeoutExpired:
        return f"error: snippet exceeded {timeout}s wall clock"

    parts: list[str] = []
    if proc.stdout.strip():
        parts.append(proc.stdout.rstrip())
    if proc.stderr.strip():
        parts.append(f"[stderr]\n{proc.stderr.rstrip()}")
    if proc.returncode != 0:
        parts.append(f"[exit {proc.returncode}]")
    return "\n".join(parts) if parts else "(no output)"


_TASK_SYSTEM = (
    "You are Artoo's browser-task sub-agent. You drive a real Chromium browser "
    "via the `browser_python` tool — it executes Python with browser-harness "
    "helpers (exact names: goto_url, page_info, js, capture_screenshot, "
    "fill_input, click_at_xy, press_key, scroll, wait, wait_for_load, "
    "wait_for_element, new_tab, switch_tab, list_tabs, plus anything in "
    "agent-workspace/agent_helpers.py).\n\n"
    "How to work:\n"
    "  1. Open relevant pages with goto_url; call page_info() to see what's there.\n"
    "  2. To SEE the rendered page — layout, charts, images, a dialog, visual state "
    "     the DOM text doesn't capture — call the `browser_see` tool. It screenshots "
    "     the current tab and returns a vision model's description. Don't guess at a "
    "     page's look from HTML alone when browser_see can just show you.\n"
    "  3. Homelab/internal https UIs use self-signed certs, but the harness ignores "
    "     cert errors — goto_url the https URL directly, there's no interstitial.\n"
    "  4. If a helper is missing, write one — edit agent-workspace/agent_helpers.py "
    "     directly via browser_python (plain Python file IO). The harness will pick "
    "     it up on the next snippet.\n"
    "  5. Iterate snippets until the goal is achieved.\n"
    "  6. If you need natural-language output in the operator's voice (drafting a reply, "
    "     a post, a message), call `spawn_worker` for tone-matched text, then paste "
    "     it into the page.\n"
    "  7. If you need prior context about the operator or his projects, call `search_memory`.\n\n"
    "Stop when the goal is done. Return a 1-3 sentence summary plus any data the "
    "caller asked for."
)


def _task_tools() -> list[agent_loop.Tool]:
    return [
        agent_loop.Tool(
            name="browser_python",
            description=(
                "Execute Python in the browser-harness session. Helpers (exact names — "
                "don't guess): goto_url(url), page_info(), js(expr), "
                "capture_screenshot(path=None, full=False), fill_input(selector, text), "
                "click_at_xy(x, y), press_key(key), scroll(x, y, dy=-300), wait(seconds), "
                "wait_for_load(), wait_for_element(selector), new_tab(url), "
                "switch_tab(target), list_tabs(), plus anything in "
                "agent-workspace/agent_helpers.py. State persists across calls. "
                "Returns combined stdout + stderr. To SEE the page rather than read its "
                "DOM, use the browser_see tool."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code":    {"type": "string", "description": "Python source"},
                    "timeout": {"type": "integer", "default": 120, "description": "Seconds"},
                },
                "required": ["code"],
            },
        ),
        agent_loop.Tool(
            name="browser_see",
            description=(
                "SEE the current browser tab. Screenshots the active tab and returns a "
                "vision model's detailed description (transcribed text, layout, buttons, "
                "charts, images, states, errors). Use when the rendered look matters and "
                "DOM text from browser_python isn't enough. Optional `question` focuses "
                "the description."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Optional focus for the description"},
                    "timeout":  {"type": "integer", "default": 60, "description": "Seconds"},
                },
                "required": [],
            },
        ),
        agent_loop.Tool(
            name="spawn_worker",
            description=(
                "Delegate to a specialist worker (e.g. for drafting text in the operator's "
                "voice). Returns the worker's final text.\n\n"
                f"Available workers:\n{workers.catalog()}"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name":   {"type": "string", "enum": workers.names()},
                    "prompt": {"type": "string"},
                },
                "required": ["name", "prompt"],
            },
        ),
        agent_loop.Tool(
            name="search_memory",
            description="Search Artoo's memory for prior context about the operator.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        ),
    ]


def _task_handler(name: str, args: dict) -> str:
    if name == "browser_python":
        return run_snippet(args["code"], timeout=args.get("timeout", 120))
    if name == "browser_see":
        return see(args.get("question", ""), timeout=args.get("timeout", 60))
    if name == "spawn_worker":
        return workers.run(args["name"], args["prompt"])
    if name == "search_memory":
        return json.dumps(memory.search_memory(args["query"], args.get("limit", 5)), indent=2)
    return f"error: unknown tool {name!r}"


def run_task(
    goal: str,
    *,
    model: str | None = None,
    max_turns: int = 2000,
) -> str:
    """Drive the browser autonomously toward `goal`. Returns the sub-agent's
    final text. The internal loop can call browser_python, spawn_worker,
    and search_memory.
    """
    result = agent_loop.run(
        model=model or _TASK_MODEL,
        system_prompt=_TASK_SYSTEM,
        history=None,
        user_message=f"Goal: {goal}",
        tools=_task_tools(),
        tool_handler=_task_handler,
        cache_system=True,
        max_turns=max_turns,
    )
    if not result.ok:
        return f"error: {result.error}"
    return result.text
