"""GitHub access via the `gh` CLI.

The user is already authenticated locally (`gh auth status` shows logged
in). This wrapper shells out to `gh` and captures structured output for
Artoo to consume — no API keys, no Python SDK dependency.

Two deny mechanisms are layered:

1. Hard deny — patterns that touch auth or destroy state. These cannot
   run via this wrapper at all (e.g. `gh auth logout`).
2. Output cap — stdout/stderr trimmed to 50KB so a `gh repo list --json
   ...` against a giant org can't blow up the model context.

Callers:
- orchestrator._TOOLS exposes this as the `github` tool so the boss can
  use it during conversation.
- channels/telegram.py /gh exposes it directly for shell-style use.
- round_pipeline workers can import run() if they need to peek at issues
  or open PRs as part of a build.
"""
from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from typing import Sequence

# Hard deny — any token in the args list matches one of these prefixes,
# the call is refused before subprocess. Conservative on purpose; the
# user can override by running gh directly in a shell.
_DENY_TOKEN_PREFIXES: tuple[str, ...] = (
    "logout",          # gh auth logout
    "delete",          # gh repo delete, gh secret delete, etc.
    "remove",          # gh ssh-key remove, gh gpg-key remove
    "revoke",          # gh auth refresh --revoke
)

# Subcommands that should never run from this wrapper.
_DENY_SUBCOMMANDS: frozenset[str] = frozenset({
    "auth",        # all auth state changes — login/logout/refresh/token
    "config",      # local gh config mutations
})

# Subcommands whose first verb is dangerous even if it's not in _DENY_TOKEN_PREFIXES.
# Per-subcommand allowlist of verbs that mutate state but are OK to use.
_OK_MUTATING: dict[str, frozenset[str]] = {
    "pr": frozenset({"create", "edit", "comment", "review", "merge", "close", "reopen", "ready"}),
    "issue": frozenset({"create", "edit", "comment", "close", "reopen", "lock", "unlock", "pin", "unpin", "transfer"}),
    "release": frozenset({"create", "edit", "upload"}),
    "repo": frozenset({"create", "edit", "fork", "clone", "sync", "rename", "archive", "unarchive"}),
    "gist": frozenset({"create", "edit", "clone"}),
    "label": frozenset({"create", "edit"}),
    "workflow": frozenset({"run", "enable", "disable"}),
    "run": frozenset({"rerun", "cancel", "watch"}),
}

# Stdout/stderr cap per call. Above this we truncate with a marker.
_OUTPUT_CAP_BYTES = 50_000


@dataclass
class GhResult:
    ok: bool
    stdout: str
    stderr: str
    exit_code: int
    truncated: bool = False
    error: str | None = None  # set when refused or subprocess failed to launch

    def render(self) -> str:
        """Human-readable summary suitable for chat output or LLM context."""
        if self.error:
            return f"❌ gh refused: {self.error}"
        parts: list[str] = []
        parts.append(f"exit={self.exit_code}")
        if self.stdout:
            parts.append(f"\nstdout:\n{self.stdout}")
        if self.stderr:
            parts.append(f"\nstderr:\n{self.stderr}")
        if self.truncated:
            parts.append("\n[output truncated]")
        return "".join(parts).strip() or f"exit={self.exit_code} (no output)"


def _check_args(args: Sequence[str]) -> str | None:
    """Return an error message if args should be refused, else None."""
    if not args:
        return "no gh subcommand supplied"

    first = args[0].lower()
    if first in _DENY_SUBCOMMANDS:
        return f"subcommand {first!r} is denied (auth/config changes are out-of-band)"

    for tok in args:
        low = tok.lower()
        for deny in _DENY_TOKEN_PREFIXES:
            if low == deny:
                # Allow if the subcommand+verb combo is on the allowlist.
                # e.g. `gh issue close 42` has close — but it's allowed for
                # the issue subcommand.
                if len(args) >= 2 and first in _OK_MUTATING and args[1].lower() in _OK_MUTATING[first]:
                    continue
                return f"token {tok!r} matches deny-list ({deny}); use the gh CLI directly if intended"
    return None


def run(args: Sequence[str], *, cwd: str | None = None, timeout: int = 120) -> GhResult:
    """Run `gh <args>` and return a structured result.

    `args` is a list of pre-split tokens (no shell expansion). Pass it the
    way you'd pass argv: `["pr", "list", "--state", "open"]`.
    """
    err = _check_args(args)
    if err:
        return GhResult(ok=False, stdout="", stderr="", exit_code=-1, error=err)

    cmd = ["gh", *args]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except FileNotFoundError:
        return GhResult(ok=False, stdout="", stderr="", exit_code=-1,
                        error="gh binary not on PATH")
    except subprocess.TimeoutExpired:
        return GhResult(ok=False, stdout="", stderr="", exit_code=-1,
                        error=f"gh timed out after {timeout}s")

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    truncated = False
    if len(stdout) > _OUTPUT_CAP_BYTES:
        stdout = stdout[:_OUTPUT_CAP_BYTES] + "\n…[truncated]"
        truncated = True
    if len(stderr) > _OUTPUT_CAP_BYTES:
        stderr = stderr[:_OUTPUT_CAP_BYTES] + "\n…[truncated]"
        truncated = True

    return GhResult(
        ok=proc.returncode == 0,
        stdout=stdout,
        stderr=stderr,
        exit_code=proc.returncode,
        truncated=truncated,
    )


def run_str(command: str, *, cwd: str | None = None, timeout: int = 120) -> GhResult:
    """Convenience for the orchestrator tool: take a single command string,
    shlex-split it, then call run(). The leading `gh` is stripped if present
    so the boss can write either form.
    """
    try:
        tokens = shlex.split(command)
    except ValueError as e:
        return GhResult(ok=False, stdout="", stderr="", exit_code=-1,
                        error=f"could not parse command: {e}")
    if tokens and tokens[0].lower() == "gh":
        tokens = tokens[1:]
    return run(tokens, cwd=cwd, timeout=timeout)


def who_am_i() -> str:
    """Return the authenticated GitHub user, or an error string."""
    res = run(["api", "user", "--jq", ".login"])
    if not res.ok:
        return f"(unauth: {res.stderr or res.error or 'unknown'})"
    return res.stdout.strip()
