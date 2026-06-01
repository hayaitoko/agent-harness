"""Guard the active-memory filter — it must exclude ONLY archived points and
pass field-less legacy points (regression guard for the 2026-05-31 bug where a
MatchExcept-in-`must` filter silently dropped ~all 392 memories from search)."""
from __future__ import annotations

from artoo import memory


def test_active_filter_excludes_only_archived_via_must_not():
    f = memory._active_filter()
    # Must NOT be a `must`-based filter (the old broken form that required the
    # status field to exist on every point).
    assert not f.must, "active filter must not use `must` (drops field-less points)"
    assert f.must_not and len(f.must_not) == 1
    cond = f.must_not[0]
    assert cond.key == "status"
    assert cond.match.value == memory.STATUS_ARCHIVED
