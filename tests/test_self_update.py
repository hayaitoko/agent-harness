"""Self-update engine tests — real git repos in tmp_path, fake upstream snapshots,
no network, fake conflict resolver. Proves the property that matters: upstream
deltas land WITHOUT clobbering the user's customizations, conflicts route to the
resolver, and failure paths restore (or, for tests-red, deliberately don't)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from artoo import self_update as su


def _git(repo: Path, *args: str) -> str:
    env = {**su._GIT_ENV}
    import os
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True,
                          check=True, env={**os.environ, **env}).stdout.strip()


def _init_repo(path: Path, files: dict[str, str]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", "main")
    _write(path, files)
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "initial")
    return path


def _write(path: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        p = path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def _make_upstream(tmp: Path, name: str, files: dict[str, str]) -> Path:
    """A bare-ish 'mirror' repo we fetch from (orphan snapshot each time, like the
    real publish workflow does)."""
    up = tmp / name
    up.mkdir(parents=True, exist_ok=True)
    _git(up, "init", "-q", "-b", "main")
    _write(up, files)
    _git(up, "add", "-A")
    _git(up, "commit", "-q", "-m", f"snapshot {name}")
    return up


# ── bootstrap ────────────────────────────────────────────────────────────────

def test_bootstrap_records_baseline_without_changing_user_tree(tmp_path):
    user = _init_repo(tmp_path / "user", {"app.py": "print('user custom')\n"})
    up = _make_upstream(tmp_path, "up1", {"app.py": "print('upstream')\n"})

    res = su.run_update(user, upstream_url=str(up), run_tests=False)

    assert res.ok and res.status == su.STATUS_BASELINE
    assert not res.applied
    # User's code is UNTOUCHED on first run.
    assert (user / "app.py").read_text() == "print('user custom')\n"
    assert su._ref_exists(user, su.VENDOR_REF)


# ── clean incremental update preserves customizations ────────────────────────

def test_update_applies_upstream_delta_keeping_user_edits(tmp_path):
    # Upstream v1 == what the user started from; user then customizes a DIFFERENT
    # file; upstream v2 changes its own file. No overlap → no conflict, both kept.
    shared = "def core():\n    return 1\n"
    user = _init_repo(tmp_path / "user", {"core.py": shared, "mine.py": "X = 1\n"})
    up1 = _make_upstream(tmp_path, "up1", {"core.py": shared})
    su.run_update(user, upstream_url=str(up1), run_tests=False)  # baseline

    # user customizes mine.py
    _write(user, {"mine.py": "X = 999  # my tweak\n"})
    _git(user, "add", "-A"); _git(user, "commit", "-q", "-m", "customize")

    # upstream v2 ships a new core.py + a brand-new file
    up2 = _make_upstream(tmp_path, "up2",
                         {"core.py": "def core():\n    return 2  # upstream fix\n",
                          "feature.py": "NEW = True\n"})
    res = su.run_update(user, upstream_url=str(up2), run_tests=False)

    assert res.ok and res.status == su.STATUS_UPDATED and res.applied
    assert "return 2" in (user / "core.py").read_text()          # upstream change landed
    assert (user / "mine.py").read_text() == "X = 999  # my tweak\n"  # customization kept
    assert (user / "feature.py").read_text() == "NEW = True\n"   # new upstream file added


def test_up_to_date_is_noop(tmp_path):
    files = {"a.py": "1\n"}
    user = _init_repo(tmp_path / "user", files)
    up = _make_upstream(tmp_path, "up1", files)
    su.run_update(user, upstream_url=str(up), run_tests=False)  # baseline
    # same upstream again → nothing to do
    res = su.run_update(user, upstream_url=str(up), run_tests=False)
    assert res.ok and res.status == su.STATUS_UP_TO_DATE and not res.applied


# ── conflict resolution via the (faked) resolver ─────────────────────────────

def test_conflict_routed_to_resolver_and_applied(tmp_path):
    base = "VALUE = 1\n"
    user = _init_repo(tmp_path / "user", {"conf.py": base})
    up1 = _make_upstream(tmp_path, "up1", {"conf.py": base})
    su.run_update(user, upstream_url=str(up1), run_tests=False)  # baseline

    # BOTH sides edit the same line → conflict.
    _write(user, {"conf.py": "VALUE = 1  # user\n"})
    _git(user, "add", "-A"); _git(user, "commit", "-q", "-m", "user edit")
    up2 = _make_upstream(tmp_path, "up2", {"conf.py": "VALUE = 2  # upstream\n"})

    seen = {}
    def resolver(path, text):
        seen["path"] = path
        assert "<<<<<<<" in text  # got real conflict markers
        return "VALUE = 2  # user + upstream merged\n"

    res = su.run_update(user, upstream_url=str(up2), conflict_resolver=resolver, run_tests=False)

    assert res.ok and res.status == su.STATUS_UPDATED
    assert seen["path"] == "conf.py"
    assert res.ai_resolved == ["conf.py"]
    assert (user / "conf.py").read_text() == "VALUE = 2  # user + upstream merged\n"
    assert not _git(user, "status", "--porcelain")  # merge committed, tree clean


def test_unresolvable_conflict_rolls_back(tmp_path):
    base = "K = 1\n"
    user = _init_repo(tmp_path / "user", {"x.py": base})
    up1 = _make_upstream(tmp_path, "up1", {"x.py": base})
    su.run_update(user, upstream_url=str(up1), run_tests=False)

    _write(user, {"x.py": "K = 1  # mine\n"})
    _git(user, "add", "-A"); _git(user, "commit", "-q", "-m", "mine")
    head_before = _git(user, "rev-parse", "HEAD")
    up2 = _make_upstream(tmp_path, "up2", {"x.py": "K = 2  # theirs\n"})

    # Resolver gives up (returns None) → engine must abort + restore.
    res = su.run_update(user, upstream_url=str(up2), conflict_resolver=lambda p, t: None,
                        run_tests=False)
    assert not res.ok and res.status == su.STATUS_ROLLED_BACK
    assert res.unresolved == ["x.py"]
    assert (user / "x.py").read_text() == "K = 1  # mine\n"     # customization intact
    assert _git(user, "rev-parse", "HEAD") == head_before        # branch restored
    assert not _git(user, "status", "--porcelain")


def test_resolver_leaving_markers_is_rejected(tmp_path):
    base = "P = 0\n"
    user = _init_repo(tmp_path / "user", {"p.py": base})
    up1 = _make_upstream(tmp_path, "up1", {"p.py": base})
    su.run_update(user, upstream_url=str(up1), run_tests=False)
    _write(user, {"p.py": "P = 0  # mine\n"})
    _git(user, "add", "-A"); _git(user, "commit", "-q", "-m", "mine")
    up2 = _make_upstream(tmp_path, "up2", {"p.py": "P = 1\n"})

    # Resolver echoes junk that still has markers → must be rejected → rollback.
    res = su.run_update(user, upstream_url=str(up2),
                        conflict_resolver=lambda p, t: "<<<<<<< still broken\nP=1\n>>>>>>>\n",
                        run_tests=False)
    assert not res.ok and res.status == su.STATUS_ROLLED_BACK


def test_resolver_fences_are_stripped(tmp_path):
    base = "Q = 0\n"
    user = _init_repo(tmp_path / "user", {"q.py": base})
    up1 = _make_upstream(tmp_path, "up1", {"q.py": base})
    su.run_update(user, upstream_url=str(up1), run_tests=False)
    _write(user, {"q.py": "Q = 0  # mine\n"})
    _git(user, "add", "-A"); _git(user, "commit", "-q", "-m", "mine")
    up2 = _make_upstream(tmp_path, "up2", {"q.py": "Q = 9\n"})

    res = su.run_update(user, upstream_url=str(up2),
                        conflict_resolver=lambda p, t: "```python\nQ = 9  # merged\n```",
                        run_tests=False)
    assert res.ok and (user / "q.py").read_text() == "Q = 9  # merged\n"


# ── test gate behavior ───────────────────────────────────────────────────────

def test_tests_red_keeps_merge_and_does_not_roll_back(tmp_path):
    base = "v = 1\n"
    user = _init_repo(tmp_path / "user", {"m.py": base})
    up1 = _make_upstream(tmp_path, "up1", {"m.py": base})
    su.run_update(user, upstream_url=str(up1), run_tests=False)
    up2 = _make_upstream(tmp_path, "up2", {"m.py": "v = 2\n"})

    res = su.run_update(user, upstream_url=str(up2), run_tests=True,
                        test_cmd=["bash", "-c", "exit 1"])  # forced red
    assert not res.ok and res.status == su.STATUS_TESTS_FAILED
    assert res.applied and res.tests_passed is False
    # Per design: merged code STAYS on disk (not rolled back).
    assert (user / "m.py").read_text() == "v = 2\n"
    assert res.backup_ref  # but a backup exists for /update rollback


def test_tests_green_completes(tmp_path):
    base = "v = 1\n"
    user = _init_repo(tmp_path / "user", {"m.py": base})
    up1 = _make_upstream(tmp_path, "up1", {"m.py": base})
    su.run_update(user, upstream_url=str(up1), run_tests=False)
    up2 = _make_upstream(tmp_path, "up2", {"m.py": "v = 2\n"})

    res = su.run_update(user, upstream_url=str(up2), run_tests=True,
                        test_cmd=["bash", "-c", "exit 0"])
    assert res.ok and res.status == su.STATUS_UPDATED and res.tests_passed is True


# ── rollback ─────────────────────────────────────────────────────────────────

def test_rollback_restores_backup(tmp_path):
    base = "v = 1\n"
    user = _init_repo(tmp_path / "user", {"m.py": base})
    up1 = _make_upstream(tmp_path, "up1", {"m.py": base})
    su.run_update(user, upstream_url=str(up1), run_tests=False)
    up2 = _make_upstream(tmp_path, "up2", {"m.py": "v = 2\n"})
    res = su.run_update(user, upstream_url=str(up2), run_tests=False)
    assert (user / "m.py").read_text() == "v = 2\n"

    rb = su.rollback(user, res.backup_ref)
    assert rb.ok and rb.status == su.STATUS_ROLLED_BACK
    assert (user / "m.py").read_text() == "v = 1\n"  # back to pre-update


def test_rollback_unknown_ref_errors(tmp_path):
    user = _init_repo(tmp_path / "user", {"a.py": "1\n"})
    rb = su.rollback(user, "no-such-backup")
    assert not rb.ok and rb.status == su.STATUS_ERROR


# ── guards ───────────────────────────────────────────────────────────────────

def test_not_a_repo_errors(tmp_path):
    d = tmp_path / "plain"
    d.mkdir()
    res = su.run_update(d, upstream_url="x", run_tests=False)
    assert not res.ok and res.status == su.STATUS_ERROR


def test_dirty_tree_is_committed_and_preserved(tmp_path):
    base = "a = 1\n"
    user = _init_repo(tmp_path / "user", {"a.py": base})
    up1 = _make_upstream(tmp_path, "up1", {"a.py": base})
    su.run_update(user, upstream_url=str(up1), run_tests=False)
    # leave an UNCOMMITTED edit, then update from a non-conflicting upstream change
    _write(user, {"a.py": "a = 1  # uncommitted\n"})
    up2 = _make_upstream(tmp_path, "up2", {"a.py": base, "b.py": "new\n"})
    res = su.run_update(user, upstream_url=str(up2), run_tests=False)
    assert res.ok
    # uncommitted customization survived the update
    assert "uncommitted" in (user / "a.py").read_text()
    assert (user / "b.py").read_text() == "new\n"


# ── B1: ref-name collisions must not clobber the user's branches ─────────────

def test_user_branch_named_like_vendor_ref_is_untouched(tmp_path):
    # A user who happens to own a branch literally named like our old vendor ref
    # must not lose it — the vendor ref lives under refs/artoo/, not refs/heads/.
    user = _init_repo(tmp_path / "user", {"a.py": "1\n"})
    _git(user, "branch", "artoo-upstream-base")  # user's precious branch
    before = _git(user, "rev-parse", "artoo-upstream-base")
    up = _make_upstream(tmp_path, "up1", {"a.py": "1\n"})
    res = su.run_update(user, upstream_url=str(up), run_tests=False)
    assert res.ok
    # The user's branch still exists and still points where it did.
    assert _git(user, "rev-parse", "artoo-upstream-base") == before
    # And our internal ref is under refs/artoo/, not a heads branch.
    assert su._ref_exists(user, su.VENDOR_REF)


# ── B2: refuse to run on a mid-operation repo ────────────────────────────────

def test_refuses_when_merge_in_progress(tmp_path):
    user = _init_repo(tmp_path / "user", {"a.py": "1\n"})
    up = _make_upstream(tmp_path, "up1", {"a.py": "1\n"})
    head = _git(user, "rev-parse", "HEAD")
    # Simulate an interrupted merge.
    (user / ".git" / "MERGE_HEAD").write_text(head + "\n")
    res = su.run_update(user, upstream_url=str(up), run_tests=False)
    assert not res.ok and res.status == su.STATUS_ERROR
    assert "in progress" in (res.error or "")
    # Nothing was mutated — no backup branch created.
    assert not _git(user, "branch", "--list", "artoo-update-backup-*")


# ── S1: a Markdown '=======' underline is NOT a conflict marker ──────────────

def test_markdown_underline_not_treated_as_conflict_marker(tmp_path):
    base = "# Title\n=======\nbody\n"
    user = _init_repo(tmp_path / "user", {"README.md": base})
    up1 = _make_upstream(tmp_path, "up1", {"README.md": base})
    su.run_update(user, upstream_url=str(up1), run_tests=False)
    _write(user, {"README.md": "# Title\n=======\nbody (mine)\n"})
    _git(user, "add", "-A"); _git(user, "commit", "-q", "-m", "mine")
    up2 = _make_upstream(tmp_path, "up2", {"README.md": "# Title\n=======\nbody (upstream)\n"})

    # Resolver returns a clean merge that legitimately contains a '=======' line.
    merged = "# Title\n=======\nbody (mine + upstream)\n"
    res = su.run_update(user, upstream_url=str(up2),
                        conflict_resolver=lambda p, t: merged, run_tests=False)
    assert res.ok and res.status == su.STATUS_UPDATED  # NOT falsely rejected
    assert (user / "README.md").read_text() == merged


def test_has_conflict_markers_helper():
    assert su._has_conflict_markers("<<<<<<< ours\nx\n=======\ny\n>>>>>>> theirs\n")
    assert su._has_conflict_markers("<<<<<<<\nx\n=======\ny\n>>>>>>>\n")  # bare markers
    assert not su._has_conflict_markers("# Heading\n=======\ntext\n")   # md underline
    assert not su._has_conflict_markers(">>> python_repl_prompt()\n")    # repl (3 >)
    assert not su._has_conflict_markers("a clean file\n")
    assert not su._has_conflict_markers("<<<<<<< ours\nonly a start, no end\n")  # need both


# ── S2: resolver output with trailing prose after a fence is rejected ────────

def test_resolver_trailing_prose_after_fence_is_rejected(tmp_path):
    base = "P = 0\n"
    user = _init_repo(tmp_path / "user", {"p.py": base})
    up1 = _make_upstream(tmp_path, "up1", {"p.py": base})
    su.run_update(user, upstream_url=str(up1), run_tests=False)
    _write(user, {"p.py": "P = 0  # mine\n"})
    _git(user, "add", "-A"); _git(user, "commit", "-q", "-m", "mine")
    head_before = _git(user, "rev-parse", "HEAD")
    up2 = _make_upstream(tmp_path, "up2", {"p.py": "P = 1\n"})

    # Fenced code followed by chatter → not a clean block → must be rejected.
    junk = "```python\nP = 1  # merged\n```\nHope this helps!"
    res = su.run_update(user, upstream_url=str(up2),
                        conflict_resolver=lambda p, t: junk, run_tests=False)
    assert not res.ok and res.status == su.STATUS_ROLLED_BACK
    assert _git(user, "rev-parse", "HEAD") == head_before
    assert "```" not in (user / "p.py").read_text()  # no junk written


# ── S3: an UNCOMMITTED edit is recoverable after an unresolvable conflict ─────

def test_uncommitted_edit_recoverable_after_rollback(tmp_path):
    base = "K = 1\n"
    user = _init_repo(tmp_path / "user", {"k.py": base})
    up1 = _make_upstream(tmp_path, "up1", {"k.py": base})
    su.run_update(user, upstream_url=str(up1), run_tests=False)
    # UNCOMMITTED customization on the same line upstream will change → conflict.
    _write(user, {"k.py": "K = 1  # my precious uncommitted edit\n"})
    up2 = _make_upstream(tmp_path, "up2", {"k.py": "K = 2  # upstream\n"})

    res = su.run_update(user, upstream_url=str(up2),
                        conflict_resolver=lambda p, t: None, run_tests=False)
    assert not res.ok and res.status == su.STATUS_ROLLED_BACK
    # The edit must survive in the working tree (it was committed into the backup
    # before the merge, then restored).
    assert "my precious" in (user / "k.py").read_text()
    assert res.backup_ref and su._ref_exists(user, res.backup_ref)


# ── S4: engine-created untracked files are cleaned on rollback ────────────────

def test_rollback_cleans_merge_created_untracked(tmp_path):
    base = "K = 1\n"
    user = _init_repo(tmp_path / "user", {"k.py": base})
    up1 = _make_upstream(tmp_path, "up1", {"k.py": base})
    su.run_update(user, upstream_url=str(up1), run_tests=False)
    _write(user, {"k.py": "K = 1  # mine\n"})
    _git(user, "add", "-A"); _git(user, "commit", "-q", "-m", "mine")
    # A pre-existing untracked file the USER owns — must NOT be deleted.
    (user / "user_scratch.txt").write_text("keep me\n")
    up2 = _make_upstream(tmp_path, "up2", {"k.py": "K = 2\n", "newfile.py": "x\n"})

    res = su.run_update(user, upstream_url=str(up2),
                        conflict_resolver=lambda p, t: None, run_tests=False)
    assert res.status == su.STATUS_ROLLED_BACK
    assert (user / "user_scratch.txt").read_text() == "keep me\n"  # user's file kept
    assert not (user / "newfile.py").exists()  # merge-created file cleaned
