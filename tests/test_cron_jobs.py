"""Tests for the dynamic agentic-cron subsystem (artoo/jobs.py).

Covers persistence, scheduler registration, validation, delete/unregister,
startup load, and the fire body (boss agent turn -> Telegram delivery) with
the LLM + Telegram calls stubbed out.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from artoo import jobs, runtime
from artoo.scheduler import Scheduler


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point JOBS_DIR at a temp dir and give jobs a fresh scheduler so tests
    don't touch the real on-disk jobs or the module-level singleton."""
    jobs_dir = tmp_path / "cron_jobs"
    jobs_dir.mkdir()
    monkeypatch.setattr(jobs, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(jobs, "scheduler", Scheduler())
    return jobs_dir


def test_create_persists_and_registers(isolated):
    jid = jobs.create("0 8 * * 1-5", "summarize my github notifications")
    # Persisted to disk.
    path = isolated / f"{jid}.json"
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["schedule"] == "0 8 * * 1-5"
    assert payload["prompt"] == "summarize my github notifications"
    assert payload["enabled"] is True
    assert payload["last_run"] is None
    # Registered into the live scheduler under the namespaced name.
    assert jobs._job_name(jid) in jobs.scheduler.jobs


def test_create_rejects_bad_schedule(isolated):
    with pytest.raises(ValueError):
        jobs.create("not a cron expr", "do something")
    # Nothing persisted.
    assert list(isolated.glob("*.json")) == []


def test_create_rejects_empty_prompt(isolated):
    with pytest.raises(ValueError):
        jobs.create("*/5 * * * *", "   ")


def test_list_jobs_newest_first(isolated):
    a = jobs.create("0 9 * * *", "job A")
    b = jobs.create("0 10 * * *", "job B")
    listed = jobs.list_jobs()
    ids = [j["uuid"] for j in listed]
    assert set(ids) == {a, b}
    # Sorted by created_at descending — b was created last.
    assert ids[0] == b


def test_delete_removes_file_and_unregisters(isolated):
    jid = jobs.create("0 9 * * *", "job")
    name = jobs._job_name(jid)
    assert name in jobs.scheduler.jobs

    assert jobs.delete(jid) is True
    assert not (isolated / f"{jid}.json").exists()
    assert name not in jobs.scheduler.jobs

    # Deleting again / unknown / malformed id is a harmless False.
    assert jobs.delete(jid) is False
    assert jobs.delete("nope") is False
    assert jobs.delete("../etc/passwd") is False


def test_load_all_registers_persisted_and_skips_invalid(isolated):
    good = jobs.create("0 7 * * *", "good job")
    # Drop a fresh scheduler so load_all has to re-register from disk.
    jobs.scheduler = Scheduler()
    # Write a corrupt job file that must be skipped, not crash load_all.
    (isolated / "deadbeef.json").write_text(json.dumps(
        {"uuid": "deadbeef", "schedule": "bogus", "prompt": "x"}
    ))

    count = jobs.load_all()
    assert count == 1
    assert jobs._job_name(good) in jobs.scheduler.jobs


def test_fire_runs_boss_turn_and_delivers(isolated, monkeypatch):
    """The registered job's fire fn runs orchestrator.respond and ships the
    result text to Telegram, then records last_run."""
    import artoo.orchestrator as orchestrator

    captured: dict = {}

    def fake_respond(prompt, *, chat_id="cli", **kw):
        captured["prompt"] = prompt
        captured["chat_id"] = chat_id
        return runtime.Result(text="here is your summary", cost_usd=0.01)

    async def fake_to_telegram(text, chat_id=None):
        captured["delivered"] = text
        captured["delivered_chat"] = chat_id

    monkeypatch.setattr(orchestrator, "respond", fake_respond)
    monkeypatch.setattr(jobs.notify, "to_telegram", fake_to_telegram)

    jid = jobs.create("0 8 * * *", "summarize the news")
    fire_fn = jobs.scheduler.jobs[jobs._job_name(jid)].fn
    asyncio.run(fire_fn())

    assert captured["prompt"] == "summarize the news"
    assert captured["delivered"] == "here is your summary"
    # last_run was recorded on disk.
    payload = json.loads((isolated / f"{jid}.json").read_text())
    assert payload["last_run"] is not None


def test_fire_reports_agent_failure(isolated, monkeypatch):
    """If the boss turn raises, the job tells the operator instead of dying silently."""
    import artoo.orchestrator as orchestrator

    captured: dict = {}

    def boom(prompt, *, chat_id="cli", **kw):
        raise RuntimeError("model exploded")

    async def fake_to_telegram(text, chat_id=None):
        captured["delivered"] = text

    monkeypatch.setattr(orchestrator, "respond", boom)
    monkeypatch.setattr(jobs.notify, "to_telegram", fake_to_telegram)

    jid = jobs.create("0 8 * * *", "do the thing")
    fire_fn = jobs.scheduler.jobs[jobs._job_name(jid)].fn
    asyncio.run(fire_fn())

    assert "scheduled job failed" in captured["delivered"]
    assert "model exploded" in captured["delivered"]
