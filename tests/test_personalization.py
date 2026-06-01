"""Public-template personalization knobs (config + persona).

These hold whether the deploy sets the ARTOO_* vars (private) or not
(public template default), so they pass in both contexts.
"""
import os

from artoo import config, persona


def test_repo_url_derives_from_slug_when_unset():
    # With ARTOO_REPO_URL unset, REPO_URL is built from the slug. This deploy
    # doesn't set ARTOO_REPO_URL, so the live invariant must hold.
    if not config.optional("ARTOO_REPO_URL"):
        assert config.REPO_URL == f"https://github.com/{config.REPO_SLUG}"


def test_default_persona_template_names_owner():
    assert config.OWNER in persona._DEFAULT_PERSONA


def test_persona_honors_env_or_falls_back_to_template():
    override = os.environ.get("ARTOO_PERSONA")
    if override:
        assert persona.ARTOO_PERSONA == override
    else:
        assert persona.ARTOO_PERSONA == persona._DEFAULT_PERSONA


def test_safe_root_is_resolved_path():
    # local._safe_root() must return config.SAFE_ROOT when no override is set,
    # so a fresh clone (no ARTOO_LOCAL_SAFE_ROOT) still has a valid safe root.
    from artoo import local
    if not os.environ.get("ARTOO_LOCAL_SAFE_ROOT"):
        assert local._safe_root() == config.SAFE_ROOT
