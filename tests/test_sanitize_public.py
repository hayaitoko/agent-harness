"""Sanitizer for the public mirror (scripts/sanitize_public.py).

The leak scan is the safety net that stops CI from publishing a tree that
still carries personal/secret tokens, so its substitution + scan logic is
worth pinning down.
"""
import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "sanitize_public", _ROOT / "scripts" / "sanitize_public.py"
)
sanitize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sanitize)


def test_substitute_scrubs_all_identifiers():
    src = "Lukas runs /home/hayai/artoo from github.com/hayaitoko/artoo — by Lukas Threlkeld"
    out = sanitize.substitute(src)
    for token in ("Lukas", "Threlkeld", "hayaitoko", "/home/hayai"):
        assert token not in out, f"{token!r} survived substitution"
    assert "/home/youruser/artoo" in out
    assert "youruser/artoo" in out


def test_full_name_replaced_before_bare_first_name():
    # Ordering matters: "Lukas Threlkeld" must resolve cleanly, not leave a
    # dangling surname from the bare "Lukas" rule.
    out = sanitize.substitute('name = "Lukas Threlkeld"')
    assert out == 'name = "Artoo contributors"'


def test_scan_flags_personal_tokens():
    assert sanitize.scan("f.py", "ping Lukas at /home/hayai/artoo") != []
    assert sanitize.scan("f.py", "nothing sensitive here") == []


def test_scan_catches_credentials():
    assert sanitize.scan("c.py", "OPENROUTER_API_KEY=sk-or-abcd1234ef") != []
    assert sanitize.scan("c.py", "ANTHROPIC_API_KEY=sk-ant-abcd1234ef") != []
    # Real Telegram bot-token shape: 8-10 digits, colon, exactly 35 token chars.
    tg = "12345678:" + "A" * 35
    assert sanitize.scan("c.py", f'token = "{tg}"') != []


def test_substituted_then_scanned_is_clean():
    # The real pipeline: substitute, THEN scan. A scrubbed line must pass.
    dirty = "Lukas at /home/hayai/artoo (github hayaitoko)"
    assert sanitize.scan("f.py", sanitize.substitute(dirty)) == []


def test_bare_username_handle_is_scrubbed_and_flagged():
    # The raw unix-username handle leaks via systemd logrotate (`su hayai
    # hayai`) and the build-bot git email (`artoo-build@hayai.local`),
    # neither of which match the /home/hayai or hayaitoko rules.
    assert sanitize.scan("artoo.logrotate", "    su hayai hayai") != []
    assert sanitize.scan("engine.py", "artoo-build@hayai.local") != []
    # And the substitute→scan pipeline clears both.
    for dirty in ("    su hayai hayai", "artoo-build@hayai.local"):
        cleaned = sanitize.substitute(dirty)
        assert "hayai" not in cleaned
        assert sanitize.scan("f", cleaned) == []
