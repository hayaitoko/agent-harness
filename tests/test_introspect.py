"""Contract test for ground-truth self-inspection (artoo/introspect.py).

Asserts shape, not environment-specific values, so it stays green anywhere.
"""
from __future__ import annotations

from artoo import introspect


def test_snapshot_shape():
    snap = introspect.snapshot()
    assert set(snap) == {"repo", "git", "pytest_installed", "service"}
    assert isinstance(snap["pytest_installed"], bool)
    assert set(snap["git"]) == {"branch", "head", "clean", "uncommitted_files"}
    assert isinstance(snap["git"]["clean"], bool)
    assert isinstance(snap["git"]["uncommitted_files"], list)
    assert set(snap["service"]) == {"active", "started"}


def test_git_state_reports_this_repo():
    # We're running inside the artoo repo, so git should resolve a real HEAD.
    state = introspect.git_state()
    assert state["head"]  # non-empty "shortsha subject"
    assert state["branch"]
