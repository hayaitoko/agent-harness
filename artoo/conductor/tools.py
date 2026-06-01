"""Conductor tool kit.

Seven tools, exposed upfront (the conductor's surface is small, so a discovery
mechanism would only add a round-trip — see the toolset discussion). Every tool
is project-scoped: file/run ops are rooted at the project_dir and reject any
path that escapes it (mirrors local.py's safe-root discipline, but the root is
the project being built, not Artoo's own tree — local.py's ops can't be reused
because they're locked to SAFE_ROOT).

The `delegate` tool is the cost/context firewall: a cheap worker writes the
code, the markers are applied here, and only a short summary goes back to the
(expensive) conductor — raw code never enters its context.

Tool handlers are SYNCHRONOUS (agent_loop calls them synchronously) and never
raise — errors come back as strings so the conductor can read and recover.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .. import agent_loop, workers
from . import verify as verify_mod

_log = logging.getLogger("artoo.conductor.tools")

# Output handed back to the conductor is capped so a giant log can't blow its
# context. Failures usually surface in the tail; we keep head + tail.
_OUT_HEAD = 1500
_OUT_TAIL = 4500
_RUN_TIMEOUT_DEFAULT = 120
_RUN_TIMEOUT_MAX = 600
_MAX_WRITE_BYTES = 5 * 1024 * 1024

# ARTOO_FILE marker parser — replicated (not imported from round_pipeline.engine)
# so the conductor stays decoupled from the /build engine we're retiring.
_FILE_BLOCK = re.compile(
    # `\n?` before the END marker tolerates a worker that omits the final
    # newline — otherwise the whole block silently fails to match.
    r"===\s*ARTOO_FILE:\s*(?P<path>[^\n=]+?)\s*===\n(?P<body>.*?)\n?===\s*ARTOO_END\s*===",
    re.DOTALL,
)
_SUMMARY_LINE = re.compile(r"^SUMMARY:\s*(.+)$", re.MULTILINE)

_DELEGATE_CODE_FORMAT = """

OUTPUT FORMAT — return COMPLETE file contents using these exact markers, nothing else:

SUMMARY: <one line on what you did>

=== ARTOO_FILE: relative/path/to/file.py ===
<complete raw file contents — no markdown fences, no escaping>
=== ARTOO_END ===

One block per file. Emit every file you create or change in full."""


@dataclass
class ConductorState:
    """Mutable run state shared across tool calls (for logging + the report)."""
    project_dir: Path
    files_written: set[str] = field(default_factory=set)
    delegations: int = 0
    last_verify_green: bool = False
    last_verify_summary: str = ""


def _truncate(text: str) -> str:
    if len(text) <= _OUT_HEAD + _OUT_TAIL:
        return text
    return f"{text[:_OUT_HEAD]}\n...[{len(text) - _OUT_HEAD - _OUT_TAIL} chars truncated]...\n{text[-_OUT_TAIL:]}"


def _resolve(project_dir: Path, rel: str) -> Path | None:
    """Resolve `rel` to an absolute path. Relative paths resolve under the
    project dir; absolute paths are honored as-is.

    NOTE: by design (the operator's call 2026-05-30) there is NO project-dir
    containment here — the conductor has full-system file access, same as its
    `run` shell. The guardrail is in the system prompt (be careful with Artoo's
    own source / the live trading-agent), with git as the backstop. Returns
    None only for an empty/unresolvable path.
    """
    if not rel:
        return None
    cand = Path(rel).expanduser() if Path(rel).is_absolute() else (project_dir / rel).expanduser()
    try:
        return cand.resolve()
    except (OSError, RuntimeError):
        return None


def _rel_label(target: Path, project_dir: Path) -> str:
    """Display path: relative to the project if inside it, else absolute."""
    try:
        return str(target.relative_to(project_dir.resolve()))
    except ValueError:
        return str(target)


# ── verify gate (synchronous reimpl of verify_mod.run_verify) ───────────────

def run_verify_sync(project_dir: Path) -> list[verify_mod.VerificationReport]:
    """Run the project's configured verify commands and return their reports.

    Reuses verify_mod's config loader + parsers (pytest/tsc/vitest/svelte/vite/
    generic) but runs the subprocesses synchronously so the conductor stays a
    plain sync loop with no nested event loop.
    """
    entries = verify_mod.load_verify_config(project_dir)
    reports: list[verify_mod.VerificationReport] = []
    for entry in entries:
        cwd = (project_dir / entry.cwd).resolve() if entry.cwd else project_dir
        if not cwd.is_dir():
            reports.append(verify_mod.VerificationReport(
                name=entry.name, command=entry.cmd, cwd=str(cwd),
                inconclusive=True, inconclusive_reason=f"cwd does not exist: {cwd}",
            ))
            continue
        try:
            proc = subprocess.run(
                ["bash", "-lc", entry.cmd],
                cwd=str(cwd), capture_output=True, text=True,
                timeout=entry.timeout_s,
                env={**os.environ, **entry.env},  # honor per-entry env from verify.yml
            )
        except subprocess.TimeoutExpired:
            reports.append(verify_mod.VerificationReport(
                name=entry.name, command=entry.cmd, cwd=str(cwd),
                inconclusive=True, inconclusive_reason=f"timed out after {entry.timeout_s}s",
            ))
            continue
        except OSError as e:
            reports.append(verify_mod.VerificationReport(
                name=entry.name, command=entry.cmd, cwd=str(cwd),
                inconclusive=True, inconclusive_reason=f"could not start: {e}",
            ))
            continue
        output = (proc.stdout or "") + (proc.stderr or "")
        parser = verify_mod.PARSERS.get(entry.parse, verify_mod.PARSERS["generic"])
        errors, warnings = parser.parse(output, proc.returncode)
        reports.append(verify_mod.VerificationReport(
            name=entry.name, command=entry.cmd, cwd=str(cwd),
            exit_code=proc.returncode, errors=errors, warnings=warnings,
            raw_output=output,
        ))
    return reports


def verify_is_green(reports: list[verify_mod.VerificationReport]) -> bool:
    """Green iff there is ≥1 report, NONE are inconclusive, and all are green.

    An inconclusive check (timeout / binary-not-found / couldn't start) means we
    do NOT know it passed — it can never read as green. Otherwise a timed-out
    test suite alongside a green lint would masquerade as "done", defeating the
    whole gate. If a check shouldn't block, don't configure it.
    """
    if not reports:
        return False
    if any(r.inconclusive for r in reports):
        return False
    return all(r.green for r in reports)


def summarize_verify(reports: list[verify_mod.VerificationReport]) -> str:
    if not reports:
        return ("NO VERIFY CONFIG — there is no .artoo/verify.yml and nothing "
                "auto-detected. Create .artoo/verify.yml so 'done' has meaning.")
    lines: list[str] = []
    for r in reports:
        if r.inconclusive:
            lines.append(f"• {r.name}: INCONCLUSIVE ({r.inconclusive_reason})")
        elif r.green:
            lines.append(f"• {r.name}: GREEN (exit 0)")
        else:
            lines.append(f"• {r.name}: RED (exit {r.exit_code}, {len(r.errors)} error(s))")
            for e in r.errors[:8]:
                loc = f" {e.file}:{e.line}" if e.file else ""
                lines.append(f"    - {e.message[:160]}{loc}")
    return "\n".join(lines)


# ── tool implementations ────────────────────────────────────────────────────

def _t_run(st: ConductorState, args: dict) -> str:
    cmd = (args.get("cmd") or "").strip()
    if not cmd:
        return "error: cmd is required"
    timeout = max(1, min(int(args.get("timeout_s") or _RUN_TIMEOUT_DEFAULT), _RUN_TIMEOUT_MAX))
    cwd = st.project_dir
    if args.get("cwd"):
        resolved = _resolve(st.project_dir, args["cwd"])
        if resolved is None or not resolved.is_dir():
            return f"error: cwd outside project or not a directory: {args['cwd']}"
        cwd = resolved
    try:
        proc = subprocess.run(
            ["bash", "-lc", cmd], cwd=str(cwd),
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {timeout}s: {cmd}"
    except OSError as e:
        return f"error: could not start command: {e}"
    out = _truncate(((proc.stdout or "") + (proc.stderr or "")).strip())
    return f"exit_code: {proc.returncode}\n{out or '(no output)'}"


def _t_read(st: ConductorState, args: dict) -> str:
    target = _resolve(st.project_dir, args.get("path") or "")
    if target is None:
        return f"error: invalid path: {args.get('path')!r}"
    if not target.is_file():
        return f"error: not a file: {args.get('path')!r}"
    try:
        data = target.read_text(errors="replace")
    except OSError as e:
        return f"error: read failed: {e}"
    return _truncate(data)


def _t_write(st: ConductorState, args: dict) -> str:
    path = args.get("path") or ""
    content = args.get("content")
    if content is None:
        return "error: content is required"
    if len(content.encode("utf-8")) > _MAX_WRITE_BYTES:
        return "error: content exceeds 5MB; split the write or delegate"
    target = _resolve(st.project_dir, path)
    if target is None:
        return f"error: invalid path: {path!r}"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    except OSError as e:
        return f"error: write failed: {e}"
    rel = _rel_label(target, st.project_dir)
    st.files_written.add(rel)
    return f"wrote {rel} ({len(content.splitlines())} lines)"


def _t_search(st: ConductorState, args: dict) -> str:
    pattern = (args.get("pattern") or "").strip()
    if not pattern:
        return "error: pattern is required"
    try:
        proc = subprocess.run(
            ["bash", "-lc", f"grep -rnI -- {_shquote(pattern)} . | head -100"],
            cwd=str(st.project_dir), capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return f"error: search failed: {e}"
    out = (proc.stdout or "").strip()
    return out or "(no matches)"


def _shquote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def _t_delegate(st: ConductorState, args: dict) -> str:
    worker = (args.get("worker") or "general").strip()
    task = (args.get("task") or "").strip()
    apply_files = bool(args.get("apply_files", True))
    if not task:
        return "error: task is required"
    if worker not in workers.names():
        return f"error: unknown worker {worker!r}; available: {workers.names()}"
    prompt = task + (_DELEGATE_CODE_FORMAT if apply_files else "")
    st.delegations += 1
    out = workers.run(worker, prompt)
    if out.startswith("[worker ") and "error:" in out[:40]:
        return out  # worker dispatch error, surfaced verbatim
    if not apply_files:
        return _truncate(out)

    files, summary = _parse_markers(out)
    if not files:
        return ("worker returned no ARTOO_FILE blocks (apply_files was true). "
                "Either retry with a clearer spec, or set apply_files=false and "
                f"handle the output yourself. Raw head:\n{out[:800]}")
    applied: list[str] = []
    for rel, body in files.items():
        target = _resolve(st.project_dir, rel)
        if target is None:
            applied.append(f"SKIPPED {rel} (unresolvable path)")
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body)
        except OSError as e:
            applied.append(f"FAILED {rel} ({e})")
            continue
        st.files_written.add(_rel_label(target, st.project_dir))
        applied.append(f"{rel} ({len(body.splitlines())} lines)")
    head = f"delegated to {worker}. summary: {summary or '(none)'}"
    return head + "\napplied:\n" + "\n".join(f"  - {a}" for a in applied)


def _parse_markers(text: str) -> tuple[dict[str, str], str]:
    files: dict[str, str] = {}
    for m in _FILE_BLOCK.finditer(text or ""):
        p = m.group("path").strip()
        if p:
            files[p] = m.group("body")
    sm = _SUMMARY_LINE.search(text or "")
    return files, (sm.group(1).strip() if sm else "")


def _t_update_progress(st: ConductorState, args: dict) -> str:
    note = (args.get("note") or "").strip()
    if not note:
        return "error: note is required"
    p = st.project_dir / "PROGRESS.md"
    try:
        existing = p.read_text() if p.exists() else "# Project Progress\n"
        p.write_text(existing.rstrip() + "\n\n" + note.strip() + "\n")
    except OSError as e:
        return f"error: could not update PROGRESS.md: {e}"
    return "PROGRESS.md updated"


def _t_verify(st: ConductorState, args: dict) -> str:
    reports = run_verify_sync(st.project_dir)
    green = verify_is_green(reports)
    summary = summarize_verify(reports)
    st.last_verify_green = green
    st.last_verify_summary = summary
    verdict = "GREEN ✓ — verify passes." if green else "RED ✗ — not done yet."
    return f"{verdict}\n{summary}"


_HANDLERS = {
    "run": _t_run,
    "read": _t_read,
    "write": _t_write,
    "search": _t_search,
    "delegate": _t_delegate,
    "update_progress": _t_update_progress,
    "verify": _t_verify,
}


def make_tool_handler(state: ConductorState) -> agent_loop.ToolHandler:
    def handler(name: str, args: dict) -> str:
        fn = _HANDLERS.get(name)
        if fn is None:
            return f"error: unknown tool {name!r}; available: {sorted(_HANDLERS)}"
        return fn(state, args)
    return handler


def build_tools() -> list[agent_loop.Tool]:
    worker_list = ", ".join(workers.names())
    return [
        agent_loop.Tool(
            name="run",
            description=("Run a shell command (bash -lc) inside the project dir. Returns exit_code "
                         "+ output (truncated). Use for ls/find, running tests ad-hoc, checking an "
                         "import, booting the app. Default timeout 120s (max 600)."),
            parameters={"type": "object", "properties": {
                "cmd": {"type": "string", "description": "command to run"},
                "cwd": {"type": "string", "description": "subdir under project (optional)"},
                "timeout_s": {"type": "integer", "description": "timeout seconds (default 120, max 600)"},
            }, "required": ["cmd"]},
        ),
        agent_loop.Tool(
            name="read",
            description="Read a file in the project (truncated if large).",
            parameters={"type": "object", "properties": {
                "path": {"type": "string", "description": "file path relative to project root"},
            }, "required": ["path"]},
        ),
        agent_loop.Tool(
            name="write",
            description=("Write a file in the project (creates dirs). Use for SMALL/surgical edits "
                         "and config files. For bulk code, prefer `delegate`."),
            parameters={"type": "object", "properties": {
                "path": {"type": "string", "description": "file path relative to project root"},
                "content": {"type": "string", "description": "full file content"},
            }, "required": ["path", "content"]},
        ),
        agent_loop.Tool(
            name="search",
            description="grep -rn across the project for a pattern (first 100 matches).",
            parameters={"type": "object", "properties": {
                "pattern": {"type": "string", "description": "string/regex to search for"},
            }, "required": ["pattern"]},
        ),
        agent_loop.Tool(
            name="delegate",
            description=(
                "Hand a bounded coding/analysis mission to a CHEAP worker, keeping your own "
                f"context lean. Workers available: {worker_list}. The worker can't see this "
                "conversation — give it a complete, self-contained spec.\n"
                "apply_files=true (default): the worker writes complete files; they're applied "
                "to the project for you and you get back a SUMMARY (filenames + line counts), not "
                "the raw code. Use for new modules, features, big rewrites, boilerplate.\n"
                "apply_files=false: returns the worker's text (use for analysis/answers)."),
            parameters={"type": "object", "properties": {
                "worker": {"type": "string", "description": f"worker name ({worker_list})"},
                "task": {"type": "string", "description": "self-contained mission spec for the worker"},
                "apply_files": {"type": "boolean", "description": "apply ARTOO_FILE output (default true)"},
            }, "required": ["worker", "task"]},
        ),
        agent_loop.Tool(
            name="update_progress",
            description=("Append a section to PROGRESS.md — the durable state a fresh conductor reads "
                         "to resume. Record your plan, what's done, what's left, decisions, blockers."),
            parameters={"type": "object", "properties": {
                "note": {"type": "string", "description": "markdown to append"},
            }, "required": ["note"]},
        ),
        agent_loop.Tool(
            name="verify",
            description=("Run the project's REAL verification (tests/type-check/build per "
                         ".artoo/verify.yml or auto-detect) and return the truth. This is the "
                         "definition of done — green here, nowhere else. If there's no verify "
                         "config, write .artoo/verify.yml first."),
            parameters={"type": "object", "properties": {}},
        ),
    ]
