"""Project verification — run real tool output through the round loop.

The orchestrator's job is to catch inconsistencies. Before this module
existed, the only "is the code broken" signal was the LLM reviewer's
emitted severity enum (engine.py:509). That count fed the persistent-
failure detector and the boss meta-review trigger — so when the LLM
hallucinated "ERROR" severity, the whole control loop reasoned over
garbage. Rounds 33-41 on agent-interface burned nine rounds thrashing
on a "broken" subsystem that wasn't actually broken because no real
check tool ever ran (node_modules wasn't installed).

This module is the fix. A project declares its verification commands
in `.artoo/verify.yml`; the engine runs them after CODE and after FIX,
parses the output into a structured `VerificationReport`, and uses the
*tool's* error count as the source of truth for stuck-detection and
halt logic. The LLM reviewer is downgraded to a code-quality opiner —
it still emits bugs, but those augment (not replace) the tool output.

Parsers live here (one per common tool). New tools get a new parser
registered in `PARSERS`. A `generic` parser falls back to "exit code 0
= green; otherwise N errors with no detail" so an unknown tool still
contributes a signal without the orchestrator needing per-tool code.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_log = logging.getLogger("artoo.conductor.verify")


# ── Dataclasses ─────────────────────────────────────────────────────

@dataclass
class VerifyIssue:
    """A single error or warning emitted by a verification tool."""
    file: str = ""
    line: Optional[int] = None
    column: Optional[int] = None
    severity: str = "error"  # "error" | "warning"
    message: str = ""
    code: str = ""  # tool-specific error code (e.g. TS2345)

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "severity": self.severity,
            "message": self.message,
            "code": self.code,
        }


@dataclass
class VerificationReport:
    """One verification command's full result.

    The orchestrator treats this as ground truth:
    - `exit_code == 0 and not errors` → green
    - `errors` → the canonical "what's actually broken" list
    - `inconclusive` → the command itself failed (timeout, not found,
      crashed) and the result should NOT feed stuck-detection logic
    """
    name: str
    command: str
    cwd: str
    exit_code: int = -1
    errors: list[VerifyIssue] = field(default_factory=list)
    warnings: list[VerifyIssue] = field(default_factory=list)
    raw_output: str = ""  # truncated to RAW_OUTPUT_LIMIT in to_dict()
    duration_ms: int = 0
    inconclusive: bool = False
    inconclusive_reason: str = ""

    @property
    def green(self) -> bool:
        return not self.inconclusive and self.exit_code == 0 and not self.errors

    @property
    def error_count(self) -> int:
        """Used as the ground-truth signal in halt/stuck logic. Inconclusive
        runs return 0 so a misconfigured verify doesn't false-trigger halts.
        """
        if self.inconclusive:
            return 0
        return len(self.errors)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "command": self.command,
            "cwd": self.cwd,
            "exit_code": self.exit_code,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "raw_output": self.raw_output[:RAW_OUTPUT_LIMIT],
            "duration_ms": self.duration_ms,
            "inconclusive": self.inconclusive,
            "inconclusive_reason": self.inconclusive_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VerificationReport":
        return cls(
            name=d.get("name", ""),
            command=d.get("command", ""),
            cwd=d.get("cwd", ""),
            exit_code=d.get("exit_code", -1),
            errors=[VerifyIssue(**e) for e in d.get("errors", [])],
            warnings=[VerifyIssue(**w) for w in d.get("warnings", [])],
            raw_output=d.get("raw_output", ""),
            duration_ms=d.get("duration_ms", 0),
            inconclusive=d.get("inconclusive", False),
            inconclusive_reason=d.get("inconclusive_reason", ""),
        )


@dataclass
class RoundVerifySnapshot:
    """All verification reports for one round, plus the totals used by
    stuck-detection. Stored in a sliding window on ProjectState.
    """
    round_number: int
    reports: list[VerificationReport] = field(default_factory=list)
    ts: float = 0.0  # unix timestamp

    @property
    def total_errors(self) -> int:
        return sum(r.error_count for r in self.reports)

    @property
    def total_warnings(self) -> int:
        return sum(len(r.warnings) for r in self.reports if not r.inconclusive)

    @property
    def all_green(self) -> bool:
        return bool(self.reports) and all(r.green for r in self.reports)

    @property
    def any_inconclusive(self) -> bool:
        return any(r.inconclusive for r in self.reports)

    def to_dict(self) -> dict:
        return {
            "round_number": self.round_number,
            "reports": [r.to_dict() for r in self.reports],
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RoundVerifySnapshot":
        return cls(
            round_number=d.get("round_number", 0),
            reports=[VerificationReport.from_dict(r) for r in d.get("reports", [])],
            ts=d.get("ts", 0.0),
        )


# ── Constants ───────────────────────────────────────────────────────

# Raw output is kept for diagnostic rounds but truncated when persisted
# to project_state.json so the file stays human-readable.
RAW_OUTPUT_LIMIT = 20_000

# Verify-history sliding window. The stuck detector needs a few rounds
# of trend to fire; 20 is plenty for any sensible window without
# bloating project_state.json.
VERIFY_HISTORY_MAX = 20

# Per-command timeout. Most projects' check + test + build runs in
# under 2 min. 600s leaves headroom for an integration suite while
# bounding stuck-on-IO failure modes. Overridable per-entry in verify.yml.
DEFAULT_TIMEOUT_S = 600


# ── Parsers ─────────────────────────────────────────────────────────

class VerifyParser:
    """Subclass and implement `parse()` to register a new tool format.

    Implementations are stateless — one instance is reused for every
    invocation of that tool. Parse from the *combined* stdout+stderr
    captured by the runner.
    """

    name = "abstract"

    def parse(self, output: str, exit_code: int) -> tuple[list[VerifyIssue], list[VerifyIssue]]:
        """Return (errors, warnings). Default falls back to the generic
        signal: exit code != 0 emits one synthetic error covering the
        whole run. Subclasses replace this with format-specific parsing.
        """
        if exit_code == 0:
            return [], []
        return [VerifyIssue(
            severity="error",
            message=f"command exited with code {exit_code}",
        )], []


class SvelteCheckParser(VerifyParser):
    """svelte-check output:

        1779314388500 ERROR "src/lib/stores/messages.ts" 3:20 "Cannot find module..."
        1779314388500 WARNING "src/lib/components/X.svelte" 12:1 "no-static-..."
        1779314388500 COMPLETED 246 FILES 11 ERRORS 1 WARNINGS 8 FILES_WITH_PROBLEMS
    """

    name = "svelte-check"

    _LINE = re.compile(
        r"^\d+\s+(?P<level>ERROR|WARNING)\s+"
        r'"(?P<file>[^"]+)"\s+'
        r"(?P<line>\d+):(?P<col>\d+)\s+"
        r'"(?P<msg>.*?)"\s*$',
        re.MULTILINE,
    )

    def parse(self, output: str, exit_code: int) -> tuple[list[VerifyIssue], list[VerifyIssue]]:
        errors: list[VerifyIssue] = []
        warnings: list[VerifyIssue] = []
        for m in self._LINE.finditer(output):
            issue = VerifyIssue(
                file=m.group("file"),
                line=int(m.group("line")),
                column=int(m.group("col")),
                severity="error" if m.group("level") == "ERROR" else "warning",
                message=m.group("msg"),
            )
            if issue.severity == "error":
                errors.append(issue)
            else:
                warnings.append(issue)
        # If svelte-check exited non-zero but we parsed no diagnostics,
        # surface that as one synthetic error so the round signals
        # broken-but-no-detail rather than mysteriously green.
        if exit_code != 0 and not errors and not warnings:
            errors.append(VerifyIssue(
                severity="error",
                message=f"svelte-check exited {exit_code} with no parseable diagnostics",
            ))
        return errors, warnings


class VitestParser(VerifyParser):
    """vitest output lines we care about:

        FAIL  src/lib/stores/threads.test.ts > select sets activeThreadId
          AssertionError: expected 'a' to be 'b'
        Test Files  1 failed | 3 passed (4)
              Tests  1 failed | 29 passed (30)
    """

    name = "vitest"

    _FAIL_LINE = re.compile(
        r"^\s*(?:FAIL|×)\s+(?P<file>\S+\.(?:test|spec)\.[jt]sx?)"
        r"(?:\s+>\s+(?P<test>.+?))?\s*$",
        re.MULTILINE,
    )
    _SUMMARY = re.compile(
        r"^\s*Tests\s+(?P<failed>\d+)\s+failed",
        re.MULTILINE,
    )

    def parse(self, output: str, exit_code: int) -> tuple[list[VerifyIssue], list[VerifyIssue]]:
        errors: list[VerifyIssue] = []
        for m in self._FAIL_LINE.finditer(output):
            test_name = m.group("test") or "(unnamed)"
            errors.append(VerifyIssue(
                file=m.group("file"),
                severity="error",
                message=f"test failed: {test_name}",
            ))
        # Fallback: if exit code is non-zero but no FAIL lines parsed,
        # check the summary for a failed count.
        if not errors and exit_code != 0:
            sm = self._SUMMARY.search(output)
            if sm and int(sm.group("failed")) > 0:
                errors.append(VerifyIssue(
                    severity="error",
                    message=f"{sm.group('failed')} test(s) failed (no per-file detail parsed)",
                ))
            else:
                errors.append(VerifyIssue(
                    severity="error",
                    message=f"vitest exited {exit_code} with no parseable failures",
                ))
        return errors, []


class ViteBuildParser(VerifyParser):
    """Vite/sveltekit build failures — usually one big error block, sometimes
    with a stack trace. We surface the first error line + any subsequent
    'at file:line:col' frames so the orchestrator sees what blew up.

        Error: The following routes were marked as prerenderable, but ...
            at file:///.../prerender.js:76:25
    """

    name = "vite"

    _ERROR_HEAD = re.compile(r"^(?:Error|RollupError|ParseError):\s*(?P<msg>.+?)$", re.MULTILINE)
    _AT_FRAME = re.compile(r"\bat\s+(?:file://)?(?P<file>[^\s:]+):(?P<line>\d+)(?::(?P<col>\d+))?")

    def parse(self, output: str, exit_code: int) -> tuple[list[VerifyIssue], list[VerifyIssue]]:
        errors: list[VerifyIssue] = []
        for m in self._ERROR_HEAD.finditer(output):
            msg = m.group("msg").strip()
            tail = output[m.end():m.end() + 2000]
            frame = self._AT_FRAME.search(tail)
            issue = VerifyIssue(
                severity="error",
                message=msg[:500],
                file=frame.group("file") if frame else "",
                line=int(frame.group("line")) if frame else None,
                column=int(frame.group("col")) if frame and frame.group("col") else None,
            )
            errors.append(issue)
        if exit_code != 0 and not errors:
            errors.append(VerifyIssue(
                severity="error",
                message=f"vite build exited {exit_code} with no parseable error",
            ))
        return errors, []


class PytestParser(VerifyParser):
    """pytest -q output:

        tests/test_x.py::test_foo FAILED                                       [50%]
        FAILED tests/test_x.py::test_foo - AssertionError: ...
        =========== 1 failed, 14 passed in 0.42s ===========
    """

    name = "pytest"

    _FAIL_TAIL = re.compile(
        r"^FAILED\s+(?P<file>[^\s:]+)::(?P<test>\S+)(?:\s+-\s+(?P<msg>.+?))?$",
        re.MULTILINE,
    )
    _ERROR_TAIL = re.compile(
        r"^ERROR\s+(?P<file>[^\s:]+)::(?P<test>\S+)(?:\s+-\s+(?P<msg>.+?))?$",
        re.MULTILINE,
    )
    _SUMMARY = re.compile(
        r"=+\s*"
        r"(?:(?P<failed>\d+)\s+failed,?\s*)?"
        r"(?:(?P<errors>\d+)\s+errors?,?\s*)?"
        r"(?:\d+\s+passed)?",
    )

    def parse(self, output: str, exit_code: int) -> tuple[list[VerifyIssue], list[VerifyIssue]]:
        errors: list[VerifyIssue] = []
        for m in self._FAIL_TAIL.finditer(output):
            errors.append(VerifyIssue(
                file=m.group("file"),
                severity="error",
                message=f"test failed: {m.group('test')}"
                        + (f" — {m.group('msg')[:300]}" if m.group("msg") else ""),
            ))
        for m in self._ERROR_TAIL.finditer(output):
            errors.append(VerifyIssue(
                file=m.group("file"),
                severity="error",
                message=f"test error: {m.group('test')}"
                        + (f" — {m.group('msg')[:300]}" if m.group("msg") else ""),
            ))
        if exit_code != 0 and not errors:
            sm = self._SUMMARY.search(output)
            n = 0
            if sm:
                n = (int(sm.group("failed") or 0) + int(sm.group("errors") or 0))
            errors.append(VerifyIssue(
                severity="error",
                message=(
                    f"pytest exited {exit_code}, summary={n} failed/error"
                    if n else f"pytest exited {exit_code} with no parseable failures"
                ),
            ))
        return errors, []


class TscParser(VerifyParser):
    """tsc --noEmit output:

        src/foo.ts(12,5): error TS2345: Argument of type 'string' ...
    """

    name = "tsc"

    _LINE = re.compile(
        r"^(?P<file>[^\s(]+)\((?P<line>\d+),(?P<col>\d+)\):\s+"
        r"(?P<level>error|warning)\s+(?P<code>TS\d+):\s+(?P<msg>.+?)$",
        re.MULTILINE,
    )

    def parse(self, output: str, exit_code: int) -> tuple[list[VerifyIssue], list[VerifyIssue]]:
        errors: list[VerifyIssue] = []
        warnings: list[VerifyIssue] = []
        for m in self._LINE.finditer(output):
            issue = VerifyIssue(
                file=m.group("file"),
                line=int(m.group("line")),
                column=int(m.group("col")),
                severity="error" if m.group("level") == "error" else "warning",
                message=m.group("msg"),
                code=m.group("code"),
            )
            (errors if issue.severity == "error" else warnings).append(issue)
        if exit_code != 0 and not errors and not warnings:
            errors.append(VerifyIssue(
                severity="error",
                message=f"tsc exited {exit_code} with no parseable diagnostics",
            ))
        return errors, warnings


class GenericParser(VerifyParser):
    """Fallback parser for tools we don't have format-specific code for.
    Exit code 0 → green. Non-zero → one synthetic error with a short
    summary of the tail of stdout/stderr."""

    name = "generic"

    def parse(self, output: str, exit_code: int) -> tuple[list[VerifyIssue], list[VerifyIssue]]:
        if exit_code == 0:
            return [], []
        # Surface the last 300 chars of output so the orchestrator's
        # error list isn't entirely opaque.
        tail = output.strip().splitlines()[-5:] if output.strip() else []
        snippet = " | ".join(line.strip() for line in tail)[:400]
        return [VerifyIssue(
            severity="error",
            message=f"exit code {exit_code}: {snippet}" if snippet else f"exit code {exit_code}",
        )], []


PARSERS: dict[str, VerifyParser] = {
    SvelteCheckParser.name: SvelteCheckParser(),
    VitestParser.name: VitestParser(),
    ViteBuildParser.name: ViteBuildParser(),
    PytestParser.name: PytestParser(),
    TscParser.name: TscParser(),
    GenericParser.name: GenericParser(),
}


# ── Verify-config schema ────────────────────────────────────────────

@dataclass
class VerifyEntry:
    """One verification command from .artoo/verify.yml."""
    name: str
    cmd: str
    parse: str = "generic"
    cwd: str = ""            # relative to project_dir; empty = project_dir itself
    timeout_s: int = DEFAULT_TIMEOUT_S
    env: dict[str, str] = field(default_factory=dict)


# ── Runner ──────────────────────────────────────────────────────────

async def run_verify(
    entries: list[VerifyEntry],
    project_dir: Path,
) -> list[VerificationReport]:
    """Run every configured verify entry sequentially and return their
    reports. Sequential (not parallel) because verifiers tend to
    contend for the same dependency caches (pnpm-store, .venv) and
    parallel installs corrupt those caches.

    Each command runs through `bash -lc` so PATH includes user-local
    shims (fnm, uv, cargo, pyenv). Failures during command setup itself
    (binary not found, timeout) are surfaced as inconclusive reports —
    they don't count against the project's error budget but they DO
    appear in the report list so the orchestrator can flag misconfig.
    """
    reports: list[VerificationReport] = []
    for entry in entries:
        report = await _run_one(entry, project_dir)
        reports.append(report)
        _log.info(
            "verify %s: exit=%d errors=%d warnings=%d duration=%dms%s",
            entry.name,
            report.exit_code,
            len(report.errors),
            len(report.warnings),
            report.duration_ms,
            " (inconclusive)" if report.inconclusive else "",
        )
    return reports


async def _run_one(entry: VerifyEntry, project_dir: Path) -> VerificationReport:
    cwd = (project_dir / entry.cwd).resolve() if entry.cwd else project_dir
    started = time.monotonic()

    if not cwd.is_dir():
        return VerificationReport(
            name=entry.name,
            command=entry.cmd,
            cwd=str(cwd),
            inconclusive=True,
            inconclusive_reason=f"cwd does not exist: {cwd}",
        )

    try:
        proc = await asyncio.create_subprocess_exec(
            "bash", "-lc", entry.cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**_inherited_env(), **entry.env},
        )
    except OSError as e:
        return VerificationReport(
            name=entry.name,
            command=entry.cmd,
            cwd=str(cwd),
            inconclusive=True,
            inconclusive_reason=f"could not start subprocess: {e}",
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    try:
        stdout_bytes, _ = await asyncio.wait_for(
            proc.communicate(), timeout=entry.timeout_s,
        )
    except asyncio.TimeoutError:
        proc.kill()
        try:
            await proc.wait()
        except Exception:  # noqa: BLE001
            pass
        return VerificationReport(
            name=entry.name,
            command=entry.cmd,
            cwd=str(cwd),
            inconclusive=True,
            inconclusive_reason=f"timed out after {entry.timeout_s}s",
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    output = stdout_bytes.decode("utf-8", errors="replace")
    parser = PARSERS.get(entry.parse, PARSERS["generic"])
    errors, warnings = parser.parse(output, proc.returncode or 0)

    return VerificationReport(
        name=entry.name,
        command=entry.cmd,
        cwd=str(cwd),
        exit_code=proc.returncode or 0,
        errors=errors,
        warnings=warnings,
        raw_output=output,
        duration_ms=int((time.monotonic() - started) * 1000),
    )


def _inherited_env() -> dict[str, str]:
    """The subprocess needs PATH, HOME, and the user's shell envs to find
    user-installed toolchains (pnpm, uv, fnm). bash -lc handles profile
    sourcing but we still need to seed the base env.
    """
    import os
    keep = {"PATH", "HOME", "USER", "LANG", "LC_ALL", "SHELL", "TERM",
            "XDG_RUNTIME_DIR", "XDG_DATA_HOME", "XDG_CONFIG_HOME",
            "NODE_OPTIONS", "PNPM_HOME", "FNM_DIR"}
    return {k: v for k, v in os.environ.items() if k in keep or k.startswith("FNM_")}


# ── Config loader ───────────────────────────────────────────────────

def load_verify_config(project_dir: Path) -> list[VerifyEntry]:
    """Read `.artoo/verify.yml`. Returns [] if no config and no auto-detect
    signal — caller treats empty list as "verification disabled for this
    project" and skips the VERIFY phase.

    Format:

        verify:
          - name: web-check
            cmd: cd web && pnpm install --frozen-lockfile && pnpm run check
            parse: svelte-check
            timeout_s: 300

        # optional global defaults
        defaults:
          timeout_s: 300
    """
    cfg_path = project_dir / ".artoo" / "verify.yml"
    if cfg_path.exists():
        return _parse_yaml_config(cfg_path)
    # Auto-detect when no config: surface a sensible default so a new
    # project gets ground-truth verification without explicit setup.
    return _auto_detect(project_dir)


def _parse_yaml_config(path: Path) -> list[VerifyEntry]:
    try:
        import yaml  # type: ignore
    except ImportError:
        _log.warning(
            "PyYAML not installed; .artoo/verify.yml present but cannot be "
            "parsed. Install PyYAML or remove the file to use auto-detection."
        )
        return []
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        _log.warning("invalid YAML in %s: %s", path, e)
        return []

    defaults = data.get("defaults", {}) or {}
    default_timeout = int(defaults.get("timeout_s", DEFAULT_TIMEOUT_S))
    entries: list[VerifyEntry] = []
    for raw in data.get("verify", []) or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", "")).strip()
        cmd = str(raw.get("cmd", "")).strip()
        if not name or not cmd:
            _log.warning("verify entry skipped (missing name or cmd): %r", raw)
            continue
        entries.append(VerifyEntry(
            name=name,
            cmd=cmd,
            parse=str(raw.get("parse", "generic")).strip(),
            cwd=str(raw.get("cwd", "")).strip(),
            timeout_s=int(raw.get("timeout_s", default_timeout)),
            env={str(k): str(v) for k, v in (raw.get("env") or {}).items()},
        ))
    return entries


def _auto_detect(project_dir: Path) -> list[VerifyEntry]:
    """Best-effort defaults for unfamiliar projects. Returns empty if
    nothing recognizable; the engine then runs without verify (existing
    behavior continues — opt-in only).
    """
    entries: list[VerifyEntry] = []

    # pyproject + tests/ → pytest
    if (project_dir / "pyproject.toml").exists() and (project_dir / "tests").is_dir():
        entries.append(VerifyEntry(
            name="pytest",
            cmd="uv run pytest -q",
            parse="pytest",
        ))
    elif (project_dir / "pytest.ini").exists() and (project_dir / "tests").is_dir():
        entries.append(VerifyEntry(
            name="pytest",
            cmd="pytest -q",
            parse="pytest",
        ))

    # Recurse one level for nested package.json (covers monorepo `web/`
    # layouts like agent-interface). Auto-detect is best-effort; users
    # with real needs should write verify.yml.
    for pkg_json in [project_dir / "package.json"] + list(project_dir.glob("*/package.json")):
        if not pkg_json.exists():
            continue
        sub_dir = pkg_json.parent.relative_to(project_dir) if pkg_json.parent != project_dir else Path(".")
        cwd = "" if sub_dir == Path(".") else str(sub_dir)
        scripts = _read_npm_scripts(pkg_json)
        prefix = "pnpm" if (pkg_json.parent / "pnpm-lock.yaml").exists() else "npm"
        if "check" in scripts:
            entries.append(VerifyEntry(
                name=f"{cwd or 'root'}-check",
                cmd=f"{prefix} install --frozen-lockfile && {prefix} run check",
                parse="svelte-check",
                cwd=cwd,
            ))
        if "test" in scripts:
            entries.append(VerifyEntry(
                name=f"{cwd or 'root'}-test",
                cmd=f"{prefix} run test",
                parse="vitest",
                cwd=cwd,
            ))

    return entries


def _read_npm_scripts(pkg_json: Path) -> set[str]:
    import json
    try:
        data = json.loads(pkg_json.read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    scripts = data.get("scripts") or {}
    return set(scripts.keys()) if isinstance(scripts, dict) else set()
