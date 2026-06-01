"""/build switch + /build list — project enumeration and keyboard shape.

The Telegram CallbackQueryHandler itself is hard to unit-test without
a mock bot, so the tests focus on the pure-function pieces: the
project enumerator (the hard part to get right) + keyboard
construction. End-to-end switch behavior is covered by integration
via the live channel.
"""
import json
from pathlib import Path

import pytest

from artoo.channels import telegram as tg


def _make_project(root: Path, slug: str, *, round_n: int = 0, halted: bool = False,
                  mtime: float | None = None) -> Path:
    pdir = root / slug
    pdir.mkdir(parents=True, exist_ok=True)
    state = {
        "current_round": round_n,
        "total_completed": max(0, round_n - 1),
        "halted": halted,
        "autonomous": False,
        "max_autonomous": 10,
    }
    state_path = pdir / "project_state.json"
    state_path.write_text(json.dumps(state))
    if mtime is not None:
        import os
        os.utime(state_path, (mtime, mtime))
    return pdir


def test_enumerate_empty_root(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(tg, "_BUILD_PROJECTS_ROOT", tmp_path)
    monkeypatch.setattr(tg, "_BUILD_CHAT_STATE", tmp_path / "chat_state.json")
    assert tg._enumerate_projects() == []


def test_enumerate_skips_dirs_without_project_state(tmp_path: Path, monkeypatch):
    """A random subdirectory (e.g. a stray git clone) shouldn't appear
    in the menu — only directories that look like build projects."""
    monkeypatch.setattr(tg, "_BUILD_PROJECTS_ROOT", tmp_path)
    monkeypatch.setattr(tg, "_BUILD_CHAT_STATE", tmp_path / "chat_state.json")
    (tmp_path / "random-dir").mkdir()
    (tmp_path / "random-dir" / "readme.md").write_text("hi")
    _make_project(tmp_path, "real-project", round_n=3)
    entries = tg._enumerate_projects()
    slugs = [e["slug"] for e in entries]
    assert slugs == ["real-project"]


def test_enumerate_sorts_by_recency(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(tg, "_BUILD_PROJECTS_ROOT", tmp_path)
    monkeypatch.setattr(tg, "_BUILD_CHAT_STATE", tmp_path / "chat_state.json")
    _make_project(tmp_path, "older", mtime=1000.0)
    _make_project(tmp_path, "newer", mtime=2000.0)
    _make_project(tmp_path, "newest", mtime=3000.0)
    slugs = [e["slug"] for e in tg._enumerate_projects()]
    assert slugs == ["newest", "newer", "older"]


def test_enumerate_surfaces_round_and_halted(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(tg, "_BUILD_PROJECTS_ROOT", tmp_path)
    monkeypatch.setattr(tg, "_BUILD_CHAT_STATE", tmp_path / "chat_state.json")
    _make_project(tmp_path, "halted-p", round_n=41, halted=True)
    _make_project(tmp_path, "running-p", round_n=3, halted=False)
    by_slug = {e["slug"]: e for e in tg._enumerate_projects()}
    assert by_slug["halted-p"]["halted"] is True
    assert by_slug["halted-p"]["round"] == 41
    assert by_slug["running-p"]["halted"] is False
    assert by_slug["running-p"]["round"] == 3


def test_enumerate_recovers_goal_from_chat_state(tmp_path: Path, monkeypatch):
    """If a chat has this project in its persistent pointer, surface
    the original goal in the listing so it's easy to recognize."""
    monkeypatch.setattr(tg, "_BUILD_PROJECTS_ROOT", tmp_path)
    chat_state = tmp_path / "chat_state.json"
    monkeypatch.setattr(tg, "_BUILD_CHAT_STATE", chat_state)
    pdir = _make_project(tmp_path, "agent-interface")
    chat_state.write_text(json.dumps({
        "12345": {"project_dir": str(pdir), "goal": "build the agent UI", "updated": "x"},
    }))
    entries = tg._enumerate_projects()
    assert entries[0]["goal"] == "build the agent UI"


def test_enumerate_ignores_corrupt_state_files(tmp_path: Path, monkeypatch):
    """A truncated/corrupt project_state.json shouldn't crash the listing."""
    monkeypatch.setattr(tg, "_BUILD_PROJECTS_ROOT", tmp_path)
    monkeypatch.setattr(tg, "_BUILD_CHAT_STATE", tmp_path / "chat_state.json")
    bad = tmp_path / "bad-project"
    bad.mkdir()
    (bad / "project_state.json").write_text("{not json")
    _make_project(tmp_path, "good-project")
    slugs = [e["slug"] for e in tg._enumerate_projects()]
    assert "good-project" in slugs
    assert "bad-project" not in slugs


def test_format_project_list_empty(tmp_path: Path):
    out = tg._format_project_list([])
    assert "no /build projects yet" in out


def test_format_project_list_renders_halted_and_running(tmp_path: Path):
    entries = [
        {"slug": "a", "project_dir": tmp_path, "round": 5,
         "halted": True, "updated_ts": 1, "goal": "build A"},
        {"slug": "b", "project_dir": tmp_path, "round": 2,
         "halted": False, "updated_ts": 2, "goal": ""},
    ]
    out = tg._format_project_list(entries)
    assert "🛑" in out and "▶️" in out
    assert "round 5" in out and "round 2" in out
    assert "build A" in out


def test_switch_keyboard_caps_at_max_buttons(tmp_path: Path):
    """Beyond _BUILD_SWITCH_MAX_BUTTONS, the rest must be hidden — they
    surface via /build list instead so the menu stays scannable."""
    entries = [
        {"slug": f"proj-{i}", "project_dir": tmp_path, "round": i,
         "halted": False, "updated_ts": float(i), "goal": ""}
        for i in range(20)
    ]
    kb = tg._switch_keyboard(entries)
    # Flatten buttons; cancel row is always present at the end.
    flat = [btn for row in kb.inline_keyboard for btn in row]
    project_buttons = [b for b in flat if b.callback_data and b.callback_data.startswith("build:switch:")]
    assert len(project_buttons) == tg._BUILD_SWITCH_MAX_BUTTONS
    cancel = [b for b in flat if b.callback_data == "build:switch_cancel"]
    assert len(cancel) == 1


def test_switch_keyboard_includes_round_and_halt_marker(tmp_path: Path):
    entries = [
        {"slug": "halted-x", "project_dir": tmp_path, "round": 41,
         "halted": True, "updated_ts": 1, "goal": ""},
        {"slug": "running-y", "project_dir": tmp_path, "round": 2,
         "halted": False, "updated_ts": 2, "goal": ""},
    ]
    kb = tg._switch_keyboard(entries)
    labels = [b.text for row in kb.inline_keyboard for b in row]
    assert any("🛑" in lbl and "halted-x" in lbl and "r41" in lbl for lbl in labels)
    assert any("▶️" in lbl and "running-y" in lbl and "r2" in lbl for lbl in labels)
