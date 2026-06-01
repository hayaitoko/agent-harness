"""Environment + path configuration."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"missing required env var: {name}")
    return val


def optional(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


# All deployment-specific values come from .env (gitignored). The defaults
# below are sensible local-machine starters — override via env for real use.
QDRANT_URL = optional("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = optional("QDRANT_API_KEY")
QDRANT_COLLECTION = optional("QDRANT_COLLECTION", "artoo_memories")
OBSIDIAN_VAULT = Path(optional("OBSIDIAN_VAULT", str(ROOT / "vault")))

# Embedding endpoint — points at any Ollama with an embed model loaded.
OLLAMA_HOST = optional("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = optional("EMBED_MODEL", "mxbai-embed-large")
EMBED_DIM = int(optional("EMBED_DIM", "1024"))
# Local cross-encoder rerank endpoint (TEI-style POST /rerank). Empty = no local
# reranker yet → memory rerank uses the cloud failover. Set this once a small
# cross-encoder (e.g. bge-reranker-base) is stood up on the GPU box; the 8B-LLM
# rerank was too slow on the shared 8GB card (timed out, 2026-05-31).
RERANK_URL = optional("RERANK_URL", "")
DATA_DIR = Path(optional("ARTOO_DATA_DIR", str(ROOT / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Artoo's build workshop — the ONE place /build and the conductor put scratch
# projects, forever. Deliberately separate from ~/projects (real / graduated
# work) so conductor runs never clutter it: a relative `/build <slug>` lands at
# BUILDS_DIR/<slug>. Override with ARTOO_BUILDS_DIR. Absolute project paths
# passed to the conductor are still honoured as-is (so you can point a build at
# a real repo on purpose).
BUILDS_DIR = Path(optional("ARTOO_BUILDS_DIR", str(Path.home() / "builds")))
BUILDS_DIR.mkdir(parents=True, exist_ok=True)

# --- Public-template knobs ---
# Personalization that differs between a private deploy and the public
# mirror. Defaults are generic so the published source reads neutrally; a
# private deploy overrides them in .env (gitignored). OWNER feeds the
# persona + boss prompt, REPO_URL feeds OpenRouter's HTTP-Referer header.
OWNER = optional("ARTOO_OWNER", "the operator")
REPO_SLUG = optional("ARTOO_REPO_SLUG", "youruser/artoo")
REPO_URL = optional("ARTOO_REPO_URL") or f"https://github.com/{REPO_SLUG}"

# Safe root for the boss's `local` self-modification tool. Defaults to the
# repo root so a fresh clone works without config; override via
# ARTOO_LOCAL_SAFE_ROOT. local._safe_root() reads the env var lazily (for
# test monkeypatching) and falls back to this.
SAFE_ROOT = Path(optional("ARTOO_LOCAL_SAFE_ROOT") or str(ROOT)).resolve()
