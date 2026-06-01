"""Self-update engine that survives the public-mirror's orphan-snapshot model.

The public mirror (see .github/workflows/publish-public.yml) is re-snapshotted as
a FRESH single-commit history on every publish — `rm -rf .git && git init`. So a
public user tracking it has no shared history with the next snapshot, and a plain
`git pull --ff-only` (what cmd_update did) can NEVER fast-forward. Worse, a naive
`git merge --allow-unrelated-histories` against an orphan tree turns every file
into a conflict and clobbers the user's customizations.

This module solves both with a VENDOR-BRANCH model:

  - A local ref `refs/artoo/upstream-base` tracks the last upstream snapshot we
    synced. It lives under refs/artoo/ (NOT refs/heads/) so it can never collide
    with a branch name the user owns.
  - Each update fetches the new snapshot, commits it as a child of that ref
    (giving us a linear upstream line: V0 -> V1 -> V2 ...), then does a real
    3-way merge into the user's branch with the PREVIOUS snapshot as the base.
  - That applies only the upstream DELTA (prev->new) onto the user's tree, so
    their customizations are preserved and conflicts arise only where they edited
    the same lines upstream also changed.

Conflicts (the only part needing judgment) are handed to a resolver — by default
Kimi K2.6-thinking (cross-checked, ZDR-pinned). Everything else is deterministic
git plumbing so an unattended run can't wander.

Safety:
  - A backup branch (artoo-update-backup-<ts>) is created AFTER any dirty-tree
    commit, so it captures the user's uncommitted edits too — rollback restores
    them.
  - The engine refuses to start on a mid-merge / rebase / cherry-pick repo.
  - On an unresolvable conflict or a hard git error it restores the backup and
    cleans any files the merge created.
  - On a CLEAN merge whose tests then FAIL it does NOT roll back and does NOT
    restart — it leaves the merged tree on disk (recoverable via the backup) so
    the operator can fix it with the bot.
  - The engine NEVER restarts a service and NEVER pushes anything. Backups and
    the vendor ref stay local.
"""
from __future__ import annotations

import datetime
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import subprocess

_log = logging.getLogger("artoo.self_update")

# Internal refs under refs/artoo/ — deliberately NOT refs/heads/* so they can
# never clobber or be clobbered by a user's branch of the same name.
VENDOR_REF = "refs/artoo/upstream-base"     # last upstream snapshot we merged
FETCHED_REF = "refs/artoo/upstream-fetch"   # scratch: where each fetch lands
BACKUP_PREFIX = "artoo-update-backup-"      # user-visible branch (for /update rollback)

# Identity for the synthetic upstream commits + merge commits we author.
_GIT_ENV = {
    "GIT_AUTHOR_NAME": "artoo-update",
    "GIT_AUTHOR_EMAIL": "update@artoo.local",
    "GIT_COMMITTER_NAME": "artoo-update",
    "GIT_COMMITTER_EMAIL": "update@artoo.local",
}

# A conflict resolver takes (path, conflicted_text_with_markers) and returns the
# fully resolved file text, or None if it cannot resolve it. Injectable for tests.
ConflictResolver = Callable[[str, str], Optional[str]]

# A REAL git conflict needs BOTH a start marker (`<<<<<<<`) and an end marker
# (`>>>>>>>`) — each 7 bracket chars at line start, followed by a space+label or
# (for a resolver that left a bare marker) end-of-line. We require BOTH so a lone
# `=======` (a Markdown H1 underline, an RST section rule) never trips a false
# positive; 7 consecutive < or > at line start ~never occurs in real source, and
# a `>>> ` Python REPL prompt is only 3 chars so it can't match `>{7}`.
_CONFLICT_START_RE = re.compile(r"^<{7}(?: |$)", re.MULTILINE)
_CONFLICT_END_RE = re.compile(r"^>{7}(?: |$)", re.MULTILINE)
_FENCE_RE = re.compile(r"^\s*```[^\n]*\n(.*)\n```\s*$", re.DOTALL)
# In-progress operations that make it unsafe to start mutating the repo.
_INPROGRESS_MARKERS = ("MERGE_HEAD", "rebase-merge", "rebase-apply",
                       "CHERRY_PICK_HEAD", "REVERT_HEAD", "BISECT_LOG")


@dataclass
class UpdateResult:
    ok: bool
    status: str = ""           # see STATUS_* below
    applied: bool = False      # did the working tree actually change
    conflicts: list[str] = field(default_factory=list)      # files that conflicted
    ai_resolved: list[str] = field(default_factory=list)    # files the resolver fixed
    unresolved: list[str] = field(default_factory=list)     # files it could NOT fix
    backup_ref: str = ""       # branch to roll back to
    tests_passed: Optional[bool] = None
    summary: str = ""
    detail: str = ""
    error: Optional[str] = None


# status values
STATUS_UP_TO_DATE = "up-to-date"
STATUS_BASELINE = "baseline-recorded"   # first run: nothing merged, baseline saved
STATUS_UPDATED = "updated"              # merged clean (+ tests if run)
STATUS_TESTS_FAILED = "merged-tests-failed"  # merged on disk, tests red, NOT rolled back
STATUS_ERROR = "error"                  # rolled back to backup (or refused before mutating)
STATUS_ROLLED_BACK = "rolled-back"


def _git(repo: Path, *args: str, check: bool = True, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    """Run a git command in `repo` (list form — never a shell). Raises
    CalledProcessError on non-zero when check=True."""
    full_env = None
    if env is not None:
        import os
        full_env = {**os.environ, **env}
    return subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True,
        check=check, env=full_env,
    )


def _out(repo: Path, *args: str) -> str:
    return _git(repo, *args).stdout.strip()


def _ref_exists(repo: Path, ref: str) -> bool:
    return _git(repo, "rev-parse", "--verify", "--quiet", ref, check=False).returncode == 0


def _current_branch(repo: Path) -> Optional[str]:
    name = _out(repo, "rev-parse", "--abbrev-ref", "HEAD")
    return None if name == "HEAD" else name  # detached → None


def _is_dirty(repo: Path) -> bool:
    return bool(_out(repo, "status", "--porcelain"))


def _untracked(repo: Path) -> set[str]:
    """Untracked, non-ignored files (so we never touch .env / data / .venv)."""
    out = _out(repo, "ls-files", "--others", "--exclude-standard")
    return {line for line in out.splitlines() if line.strip()}


def _in_progress_op(repo: Path) -> Optional[str]:
    """Name of an in-progress git operation (merge/rebase/...) or None. Starting
    an update on top of one would corrupt both."""
    try:
        git_dir = Path(_out(repo, "rev-parse", "--git-dir"))
    except subprocess.CalledProcessError:
        return None
    if not git_dir.is_absolute():
        git_dir = (repo / git_dir).resolve()
    for marker in _INPROGRESS_MARKERS:
        if (git_dir / marker).exists():
            return marker
    return None


def _has_conflict_markers(text: str) -> bool:
    return bool(_CONFLICT_START_RE.search(text) and _CONFLICT_END_RE.search(text))


def _strip_fences(text: str) -> str:
    """Unwrap a single ```lang ... ``` fence if the resolver wrapped the whole
    file in one. Only strips a CLEAN block (fence is the entire output); if the
    output merely starts with a fence but has trailing prose, we leave it as-is
    so the caller's validation rejects it (safer than guessing). A file that is
    not fence-wrapped (e.g. Markdown that legitimately contains code fences) is
    returned untouched."""
    s = text.strip()
    if not s.startswith("```"):
        return text
    m = _FENCE_RE.match(s)
    if not m:
        return text  # starts with a fence but isn't a clean block → caller rejects
    inner = m.group(1)
    return inner if inner.endswith("\n") else inner + "\n"


def _upstream_url(repo: Path, explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    res = _git(repo, "remote", "get-url", "origin", check=False)
    return res.stdout.strip() or None


def _unique_backup_name(repo: Path, stamp: str) -> str:
    base = f"{BACKUP_PREFIX}{stamp}"
    name = base
    i = 2
    while _ref_exists(repo, f"refs/heads/{name}"):
        name = f"{base}-{i}"
        i += 1
    return name


def run_update(
    repo_dir: str | Path,
    *,
    upstream_url: Optional[str] = None,
    upstream_branch: str = "main",
    conflict_resolver: Optional[ConflictResolver] = None,
    test_cmd: Optional[list[str]] = None,
    run_tests: bool = True,
    _now: Optional[datetime.datetime] = None,
) -> UpdateResult:
    """Pull the latest upstream snapshot into `repo_dir`, preserving local
    customizations via a vendor-branch 3-way merge. Never raises — failures come
    back in the UpdateResult. Does NOT restart anything (caller's job).

    `conflict_resolver` defaults to Kimi (kimi_resolver). `test_cmd` defaults to
    the project's pytest; pass run_tests=False to skip the gate.
    """
    repo = Path(repo_dir).expanduser().resolve()
    resolver = conflict_resolver or kimi_resolver
    stamp = (_now or datetime.datetime.now()).strftime("%Y%m%d-%H%M%S")

    if not (repo / ".git").exists():
        return UpdateResult(ok=False, status=STATUS_ERROR, error=f"not a git repo: {repo}")

    branch = _current_branch(repo)
    if branch is None:
        return UpdateResult(ok=False, status=STATUS_ERROR,
                            error="HEAD is detached — check out a branch before updating")

    inprog = _in_progress_op(repo)
    if inprog:
        return UpdateResult(ok=False, status=STATUS_ERROR,
                            error=f"a git {inprog} is in progress — finish or abort it first",
                            summary="Refused to update: the repo is mid-operation. Nothing changed.")

    url = _upstream_url(repo, upstream_url)
    if not url:
        return UpdateResult(ok=False, status=STATUS_ERROR,
                            error="no upstream URL (pass upstream_url or set an 'origin' remote)")

    backup = _unique_backup_name(repo, stamp)
    pre_untracked: set[str] = set()
    backup_made = False
    try:
        # 1. Commit a dirty tree FIRST so the backup (next) captures the user's
        #    uncommitted edits — otherwise a rollback would discard them.
        if _is_dirty(repo):
            _git(repo, "add", "-A", env=_GIT_ENV)
            _git(repo, "commit", "-m", f"artoo: local changes before update {stamp}",
                 "--no-verify", env=_GIT_ENV)

        # 2. Back up the (now fully-committed) state. branch with NO -f: the name
        #    is timestamp-unique (checked) so this won't clobber anything.
        _git(repo, "branch", backup, "HEAD", env=_GIT_ENV)
        backup_made = True
        pre_untracked = _untracked(repo)

        # 3. Fetch the new upstream snapshot into an explicit ref (NOT FETCH_HEAD,
        #    which accumulates across fetches and can resolve to a stale tree).
        _git(repo, "fetch", "--no-tags", "--force", url,
             f"{upstream_branch}:{FETCHED_REF}", env=_GIT_ENV)
        new_tree = _out(repo, "rev-parse", f"{FETCHED_REF}^{{tree}}")

        # 4. Bootstrap vs incremental.
        if not _ref_exists(repo, VENDOR_REF):
            return _bootstrap(repo, new_tree, backup)

        prev_vendor = _out(repo, "rev-parse", VENDOR_REF)
        prev_tree = _out(repo, "rev-parse", f"{VENDOR_REF}^{{tree}}")
        if new_tree == prev_tree:
            return UpdateResult(ok=True, status=STATUS_UP_TO_DATE, applied=False,
                                backup_ref=backup, summary="Already on the latest upstream.")

        # 5. Synthesize the new vendor commit (child of the previous snapshot) so
        #    a real 3-way merge has prev_vendor as its base.
        new_vendor = _commit_tree(repo, new_tree, parent=prev_vendor,
                                  message=f"upstream snapshot {stamp}")

        # 6. Merge upstream delta into the user's branch.
        merge = _git(repo, "merge", "--no-ff", "-m", f"artoo update {stamp}",
                     new_vendor, check=False, env=_GIT_ENV)
        conflicts: list[str] = []
        ai_resolved: list[str] = []
        unresolved: list[str] = []
        if merge.returncode != 0:
            conflicts = _conflicted_files(repo)
            if not conflicts:
                _git(repo, "merge", "--abort", check=False, env=_GIT_ENV)
                _restore(repo, backup, pre_untracked)
                return UpdateResult(ok=False, status=STATUS_ROLLED_BACK, backup_ref=backup,
                                    error=f"merge failed: {(merge.stderr or merge.stdout)[:400]}",
                                    summary="Update aborted; restored your previous state.")
            for rel in conflicts:
                if _try_resolve(repo, rel, resolver) is None:
                    unresolved.append(rel)
                else:
                    ai_resolved.append(rel)
            if unresolved:
                _git(repo, "merge", "--abort", check=False, env=_GIT_ENV)
                _restore(repo, backup, pre_untracked)
                return UpdateResult(
                    ok=False, status=STATUS_ROLLED_BACK, backup_ref=backup,
                    conflicts=conflicts, ai_resolved=ai_resolved, unresolved=unresolved,
                    error=f"could not auto-resolve: {', '.join(unresolved)}",
                    summary=("Hit merge conflicts the resolver couldn't settle; restored "
                             "your previous state (your customizations are intact). These "
                             "files need a human:\n  - " + "\n  - ".join(unresolved)),
                )
            _git(repo, "commit", "--no-edit", "--no-verify", env=_GIT_ENV)

        # 7. Advance the vendor ref ONLY after a successful merge commit.
        _git(repo, "update-ref", VENDOR_REF, new_vendor, env=_GIT_ENV)

        # 8. Test gate.
        if not run_tests:
            return UpdateResult(ok=True, status=STATUS_UPDATED, applied=True, backup_ref=backup,
                                conflicts=conflicts, ai_resolved=ai_resolved, tests_passed=None,
                                summary=_merged_summary(ai_resolved, tested=False))

        passed, test_detail = _run_tests(repo, test_cmd)
        if passed:
            return UpdateResult(ok=True, status=STATUS_UPDATED, applied=True, backup_ref=backup,
                                conflicts=conflicts, ai_resolved=ai_resolved, tests_passed=True,
                                summary=_merged_summary(ai_resolved, tested=True),
                                detail=test_detail)
        # Tests red on a clean merge. Per design: DON'T roll back — keep the merged
        # tree on disk so the operator can fix it with the bot. Backup is the hatch.
        return UpdateResult(
            ok=False, status=STATUS_TESTS_FAILED, applied=True, backup_ref=backup,
            conflicts=conflicts, ai_resolved=ai_resolved, tests_passed=False,
            summary=(
                "Merged the update, but the test suite is RED — I did NOT restart, so "
                "you're still running the old code. The merged code is on disk so we can "
                "fix it together. Talk to me about the failures below, or roll back with "
                f"`/update rollback` (restores `{backup}`)."
            ),
            detail=test_detail,
        )
    except subprocess.CalledProcessError as e:
        try:
            _git(repo, "merge", "--abort", check=False, env=_GIT_ENV)
            if backup_made and _ref_exists(repo, f"refs/heads/{backup}"):
                _restore(repo, backup, pre_untracked)
        except Exception:  # noqa: BLE001
            _log.exception("restore after failed update also failed")
        return UpdateResult(ok=False, status=STATUS_ERROR, backup_ref=backup if backup_made else "",
                            error=f"git error: {(e.stderr or str(e))[:400]}",
                            summary="Update failed; restored your previous state.")


def _bootstrap(repo: Path, new_tree: str, backup: str) -> UpdateResult:
    """First-ever update: no prior snapshot to diff against, so we can't safely
    3-way merge. Record the current upstream as the baseline (keeping the user's
    tree exactly via `-s ours`) so EVERY FUTURE update merges real deltas. The
    user's running version is untouched — no restart needed."""
    v0 = _commit_tree(repo, new_tree, parent=None, message="upstream baseline")
    _git(repo, "update-ref", VENDOR_REF, v0, env=_GIT_ENV)
    # Make the baseline an ancestor of the user's branch WITHOUT changing their
    # tree, so future merge-base resolution finds it.
    _git(repo, "merge", "-s", "ours", "--allow-unrelated-histories",
         "-m", "artoo: record upstream baseline", v0, env=_GIT_ENV)
    return UpdateResult(
        ok=True, status=STATUS_BASELINE, applied=False, backup_ref=backup,
        summary=("First update on this checkout — recorded the current upstream as your "
                 "baseline without changing your code. From now on `/update` folds upstream "
                 "changes into your customizations automatically."),
    )


def _commit_tree(repo: Path, tree: str, *, parent: Optional[str], message: str) -> str:
    args = ["commit-tree", tree, "-m", message]
    if parent:
        args += ["-p", parent]
    return _git(repo, *args, env=_GIT_ENV).stdout.strip()


def _conflicted_files(repo: Path) -> list[str]:
    out = _out(repo, "diff", "--name-only", "--diff-filter=U")
    return [line for line in out.splitlines() if line.strip()]


def _try_resolve(repo: Path, rel: str, resolver: ConflictResolver) -> Optional[str]:
    """Hand one conflicted file to the resolver; write + stage on success.
    Returns the resolved text, or None if unresolved (markers/fences left, empty,
    raised, etc.) — in which case the caller rolls back cleanly."""
    path = repo / rel
    try:
        conflicted = path.read_text(errors="replace")
    except OSError:
        return None
    try:
        resolved = resolver(rel, conflicted)
    except Exception as e:  # noqa: BLE001 — a resolver must never crash the update
        _log.warning("conflict resolver raised for %s: %s", rel, e)
        return None
    if not resolved or not resolved.strip():
        return None
    resolved = _strip_fences(resolved)
    # Reject anything that still looks unfinished: leftover conflict markers, or a
    # stray code fence the resolver wrapped around prose. Better a clean rollback
    # than committing broken source.
    if _has_conflict_markers(resolved):
        return None
    if resolved.lstrip().startswith("```") or re.search(r"^```\s*$", resolved, re.MULTILINE):
        return None
    try:
        path.write_text(resolved)
        _git(repo, "add", "--", rel, env=_GIT_ENV)
    except (OSError, subprocess.CalledProcessError):
        return None
    return resolved


def _restore(repo: Path, backup: str, pre_untracked: set[str]) -> None:
    """Return the tree+index to the backup, and remove only files the engine's
    merge created (never the user's pre-existing untracked files)."""
    _git(repo, "reset", "--hard", backup, check=False, env=_GIT_ENV)
    for rel in _untracked(repo) - pre_untracked:
        try:
            (repo / rel).unlink()
        except OSError:
            pass


def _run_tests(repo: Path, test_cmd: Optional[list[str]]) -> tuple[bool, str]:
    cmd = test_cmd or _default_test_cmd(repo)
    try:
        proc = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"could not run tests ({cmd}): {e}"
    tail = ((proc.stdout or "") + (proc.stderr or "")).strip()[-2000:]
    return proc.returncode == 0, tail


def _default_test_cmd(repo: Path) -> list[str]:
    venv_py = repo / ".venv" / "bin" / "python"
    py = str(venv_py) if venv_py.exists() else "python3"
    return [py, "-m", "pytest", "-q"]


def _merged_summary(ai_resolved: list[str], *, tested: bool) -> str:
    parts = ["Updated to the latest upstream, customizations preserved."]
    if ai_resolved:
        parts.append(f"Resolved {len(ai_resolved)} conflict(s): {', '.join(ai_resolved)}.")
    parts.append("Tests green." if tested else "Tests skipped.")
    return " ".join(parts)


# ── default resolver: Kimi K2.6-thinking ────────────────────────────────────

_RESOLVER_MODEL = "moonshotai/kimi-k2.6"  # cross-checked, ZDR-pinned (parasail) in runtime
_RESOLVER_SYS = """You resolve a single git merge conflict in a source file.

The file contains conflict markers:
  <<<<<<< ours        ← the USER's version (their customizations — preserve their intent)
  =======
  >>>>>>> theirs      ← the new UPSTREAM version (the update — fold in its improvements)

Produce a single coherent file that keeps the user's customizations AND incorporates
the upstream change. When they touch unrelated things, include both. When they
genuinely conflict, prefer the user's intent but adopt upstream's bug/security fixes.

Output ONLY the complete resolved file contents. No conflict markers, no markdown
fences, no commentary — just the file as it should be written to disk."""


def kimi_resolver(path: str, conflicted_text: str) -> Optional[str]:
    """Resolve a conflict with Kimi K2.6-thinking. Returns None on any failure
    (the engine then treats the file as unresolved and rolls back safely)."""
    from . import runtime
    prompt = (
        f"File: {path}\n\n"
        f"Conflicted contents (resolve every marker):\n\n{conflicted_text}"
    )
    r = runtime.openrouter(
        prompt, model=_RESOLVER_MODEL, system_prompt=_RESOLVER_SYS,
        reasoning_effort="high", max_tokens=32768,
    )
    if not r.ok or not r.text:
        _log.warning("kimi conflict resolver failed for %s: %s", path, r.error)
        return None
    return r.text


def rollback(repo_dir: str | Path, backup_ref: str) -> UpdateResult:
    """Restore a backup branch created by a prior run_update. Used by
    `/update rollback`."""
    repo = Path(repo_dir).expanduser().resolve()
    if not _ref_exists(repo, backup_ref):
        return UpdateResult(ok=False, status=STATUS_ERROR, error=f"no such backup: {backup_ref}")
    branch = _current_branch(repo) or "HEAD"
    try:
        _git(repo, "reset", "--hard", backup_ref, env=_GIT_ENV)
    except subprocess.CalledProcessError as e:
        return UpdateResult(ok=False, status=STATUS_ERROR,
                            error=f"rollback failed: {(e.stderr or str(e))[:300]}")
    return UpdateResult(ok=True, status=STATUS_ROLLED_BACK, backup_ref=backup_ref,
                        summary=f"Rolled back `{branch}` to `{backup_ref}`. Restart to run the old code.")
