"""Conductor tests — no live API. agent_loop.run and workers.run are faked.

Focus: the things that make the conductor trustworthy — the code-enforced verify
gate (it cannot return success while red), the delegate context-firewall, and
project-scoped path safety.
"""
from __future__ import annotations

import pytest

from artoo import agent_loop, config, runtime, workers
from artoo.conductor import budget, conductor, tools
from artoo.conductor import verify as verify_mod
from artoo.conductor.tools import ConductorState, make_tool_handler


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Isolate side effects: throwaway budget ledger (fresh daily cap, no real
    file), and a stubbed review gate that PASSES by default — so green-path tests
    don't make a real DeepSeek call. Review-gate tests override review_build."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "_budget_data")
    from artoo.conductor.review import ReviewResult
    monkeypatch.setattr(conductor, "review_build", lambda pd, g: ReviewResult(passed=True, cost_usd=0.0))


def _sentinel_verify(monkeypatch):
    """Verify config = 'pass iff a DONE file exists in the project'."""
    entry = verify_mod.VerifyEntry(name="sentinel", cmd="test -f DONE", parse="generic")
    monkeypatch.setattr(verify_mod, "load_verify_config", lambda _pd: [entry])


# ── verify gate ─────────────────────────────────────────────────────────────

def test_verify_gate_red_then_green(tmp_path, monkeypatch):
    _sentinel_verify(monkeypatch)
    # No DONE file → red.
    reports = tools.run_verify_sync(tmp_path)
    assert not tools.verify_is_green(reports)
    assert "RED" in tools.summarize_verify(reports)
    # Create DONE → green.
    (tmp_path / "DONE").write_text("x")
    reports = tools.run_verify_sync(tmp_path)
    assert tools.verify_is_green(reports)
    assert "GREEN" in tools.summarize_verify(reports)


def test_no_verify_config_is_not_green(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_mod, "load_verify_config", lambda _pd: [])
    reports = tools.run_verify_sync(tmp_path)
    # No conclusive reports → cannot be green (can't fake done by having no checks).
    assert not tools.verify_is_green(reports)


def test_inconclusive_check_is_never_green(tmp_path):
    # A green check alongside an INCONCLUSIVE one (e.g. a timed-out test suite)
    # must NOT read as green — "tests didn't run" is not "tests passed".
    green = verify_mod.VerificationReport(name="lint", command="x", cwd=".", exit_code=0)
    timed_out = verify_mod.VerificationReport(
        name="tests", command="pytest", cwd=".",
        inconclusive=True, inconclusive_reason="timed out",
    )
    assert green.green and not timed_out.green
    assert not tools.verify_is_green([green, timed_out])
    assert tools.verify_is_green([green])  # all conclusive + green → green


# ── project-scoped tools + path safety ──────────────────────────────────────

def test_write_read_run_within_project(tmp_path):
    st = ConductorState(project_dir=tmp_path)
    h = make_tool_handler(st)
    assert "wrote" in h("write", {"path": "pkg/m.py", "content": "x = 1\n"})
    assert "x = 1" in h("read", {"path": "pkg/m.py"})
    out = h("run", {"cmd": "echo hello"})
    assert "exit_code: 0" in out and "hello" in out
    assert "pkg/m.py" in st.files_written


def test_yolo_allows_out_of_project_writes(tmp_path):
    # Full-system access by design: relative paths land in the project; absolute
    # paths are honored. (The guardrail is the system prompt, not a sandbox.)
    proj = tmp_path / "proj"
    proj.mkdir()
    st = ConductorState(project_dir=proj)
    h = make_tool_handler(st)
    outside = tmp_path / "sibling.txt"
    assert "wrote" in h("write", {"path": str(outside), "content": "ok"})
    assert outside.read_text() == "ok"
    # relative path stays in the project
    assert "wrote" in h("write", {"path": "inside.txt", "content": "x"})
    assert (proj / "inside.txt").exists()


def test_run_reports_nonzero_exit(tmp_path):
    st = ConductorState(project_dir=tmp_path)
    h = make_tool_handler(st)
    out = h("run", {"cmd": "exit 3"})
    assert "exit_code: 3" in out


# ── delegate: context firewall ──────────────────────────────────────────────

def test_delegate_applies_files_and_returns_summary_only(tmp_path, monkeypatch):
    st = ConductorState(project_dir=tmp_path)
    h = make_tool_handler(st)
    worker_output = (
        "SUMMARY: added the widget module\n\n"
        "=== ARTOO_FILE: src/widget.py ===\n"
        "def widget():\n    return 42\n"
        "=== ARTOO_END ===\n"
    )
    monkeypatch.setattr(workers, "names", lambda: ["general", "deep"])
    monkeypatch.setattr(workers, "run", lambda name, prompt: worker_output)

    result = h("delegate", {"worker": "general", "task": "build a widget"})

    # File was applied to disk...
    assert (tmp_path / "src/widget.py").read_text().startswith("def widget()")
    assert "src/widget.py" in st.files_written
    # ...but the raw code did NOT come back to the conductor — only a summary.
    assert "added the widget module" in result
    assert "def widget()" not in result
    assert "return 42" not in result


def test_delegate_unknown_worker(tmp_path, monkeypatch):
    st = ConductorState(project_dir=tmp_path)
    h = make_tool_handler(st)
    monkeypatch.setattr(workers, "names", lambda: ["general"])
    assert "unknown worker" in h("delegate", {"worker": "nope", "task": "x"})


# ── full outer loop: the enforced gate ──────────────────────────────────────

def _fake_loop(writes_done: bool, cost: float = 0.001):
    """Build a fake agent_loop.run that optionally has the 'conductor' create
    the DONE sentinel via its tool handler, then returns a final Result."""
    def fake(*, tool_handler, **kwargs):
        if writes_done:
            tool_handler("write", {"path": "DONE", "content": "x"})
        return runtime.Result(text="I believe it's done.", cost_usd=cost, model="fake/conductor")
    return fake


def test_run_build_succeeds_only_when_verify_green(tmp_path, monkeypatch):
    _sentinel_verify(monkeypatch)
    monkeypatch.setattr(agent_loop, "run", _fake_loop(writes_done=True))
    res = conductor.run_build(tmp_path, goal="make it green")
    assert res.ok
    assert res.halt_reason == "verified-green"
    assert res.cycles == 1
    assert "DONE" in res.files_written


def test_run_build_refuses_to_finish_red(tmp_path, monkeypatch):
    _sentinel_verify(monkeypatch)
    # Conductor claims done ("I believe it's done.") but never creates DONE.
    monkeypatch.setattr(agent_loop, "run", _fake_loop(writes_done=False))
    res = conductor.run_build(tmp_path, goal="x", max_cycles=2)
    # The model SAID done, but the gate is red → run_build must NOT report ok.
    assert not res.ok
    assert res.halt_reason == "max-cycles"
    assert res.cycles == 2


def test_budget_breaker_trips_on_cost(tmp_path, monkeypatch):
    _sentinel_verify(monkeypatch)
    monkeypatch.setattr(agent_loop, "run", _fake_loop(writes_done=False, cost=5.0))
    res = conductor.run_build(tmp_path, goal="x", budget_usd=1.0, max_cycles=10)
    assert not res.ok
    assert res.halt_reason == "budget"
    assert res.cycles == 1  # tripped after the first (expensive) cycle


def test_runs_even_when_already_green(tmp_path, monkeypatch):
    # A green project + a real goal must STILL run — developing a passing
    # codebase is the common case. No short-circuit on already-green; the gate
    # is verify-green AFTER the work. (Regression: the conductor used to bail
    # with cycles=0 and do nothing, making it useless for dev tasks.)
    _sentinel_verify(monkeypatch)
    (tmp_path / "DONE").write_text("x")  # verify already green
    calls = {"n": 0}

    def fake(**kwargs):
        calls["n"] += 1
        return runtime.Result(text="done", cost_usd=0.001, model="fake")

    monkeypatch.setattr(agent_loop, "run", fake)
    res = conductor.run_build(tmp_path, goal="add a /health endpoint")
    assert calls["n"] >= 1        # the conductor actually ran a cycle
    assert res.ok                 # verify still green afterward → success
    assert res.halt_reason == "verified-green"
    assert res.cycles == 1


def test_conductor_error_halts_without_spinning(tmp_path, monkeypatch):
    _sentinel_verify(monkeypatch)

    def erroring(**kwargs):
        return runtime.Result(text="", error="openrouter http error", cost_usd=0.0)

    monkeypatch.setattr(agent_loop, "run", erroring)
    res = conductor.run_build(tmp_path, goal="x", max_cycles=5)
    assert not res.ok
    assert res.halt_reason == "conductor-error"
    assert res.cycles == 1  # didn't spin all 5 cycles on a hard error


# ── daily budget ledger ─────────────────────────────────────────────────────

def test_budget_ledger_controls(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert budget.cap_today() == budget.DEFAULT_DAILY_CAP_USD
    assert budget.spent_today() == 0.0
    budget.add_spend(1.5)
    assert budget.spent_today() == 1.5
    assert budget.remaining_today() == budget.DEFAULT_DAILY_CAP_USD - 1.5
    assert budget.bump_cap(5) == budget.DEFAULT_DAILY_CAP_USD + 5      # the +$5 button
    assert budget.set_cap(2) == 2.0                                    # manual override
    assert budget.remaining_today() == 0.5                            # 2.0 cap - 1.5 spent


# ── review gate ─────────────────────────────────────────────────────────────

def test_review_pass_completes(tmp_path, monkeypatch):
    from artoo.conductor.review import ReviewResult
    _sentinel_verify(monkeypatch)
    monkeypatch.setattr(agent_loop, "run", _fake_loop(writes_done=True))
    monkeypatch.setattr(conductor, "review_build",
                        lambda pd, g: ReviewResult(passed=True, cost_usd=0.001))
    res = conductor.run_build(tmp_path, goal="x")
    assert res.ok and res.halt_reason == "verified-green"


def test_review_gaps_block_completion(tmp_path, monkeypatch):
    # Verify is green every cycle, but the reviewer always finds gaps → the build
    # must NOT be reported done (green is necessary, not sufficient).
    from artoo.conductor.review import ReviewResult
    _sentinel_verify(monkeypatch)
    monkeypatch.setattr(agent_loop, "run", _fake_loop(writes_done=True))
    monkeypatch.setattr(conductor, "review_build",
                        lambda pd, g: ReviewResult(passed=False, gaps="VERDICT: GAPS\n- faked auth",
                                                   cost_usd=0.001))
    res = conductor.run_build(tmp_path, goal="x", max_cycles=2)
    assert not res.ok
    assert res.halt_reason == "max-cycles"
    assert "review gaps" in res.verify_summary.lower()


def test_review_gap_then_fixed_completes(tmp_path, monkeypatch):
    from artoo.conductor.review import ReviewResult
    _sentinel_verify(monkeypatch)
    monkeypatch.setattr(agent_loop, "run", _fake_loop(writes_done=True))
    calls = {"n": 0}

    def fake_review(pd, g):
        calls["n"] += 1
        if calls["n"] == 1:
            return ReviewResult(passed=False, gaps="VERDICT: GAPS\n- fix it", cost_usd=0.001)
        return ReviewResult(passed=True, cost_usd=0.001)

    monkeypatch.setattr(conductor, "review_build", fake_review)
    res = conductor.run_build(tmp_path, goal="x", max_cycles=3)
    assert res.ok and res.halt_reason == "verified-green"
    assert res.cycles == 2  # gap on cycle 1, fixed + re-reviewed clean on cycle 2


def test_review_skipped_does_not_block(tmp_path, monkeypatch):
    # Reviewer outage degrades safe — verify-green stands, build not blocked.
    from artoo.conductor.review import ReviewResult
    _sentinel_verify(monkeypatch)
    monkeypatch.setattr(agent_loop, "run", _fake_loop(writes_done=True))
    monkeypatch.setattr(conductor, "review_build",
                        lambda pd, g: ReviewResult(passed=True, skipped=True, error="reviewer down"))
    res = conductor.run_build(tmp_path, goal="x")
    assert res.ok and res.halt_reason == "verified-green"


def test_run_build_halts_when_daily_budget_exhausted(tmp_path, monkeypatch):
    _sentinel_verify(monkeypatch)
    monkeypatch.setattr(budget, "remaining_today", lambda: 0.0)

    def boom(**kwargs):
        raise AssertionError("must not run a cycle when the daily budget is spent")

    monkeypatch.setattr(agent_loop, "run", boom)
    res = conductor.run_build(tmp_path, goal="x")
    assert not res.ok
    assert res.halt_reason == "budget"
    assert res.cycles == 0
