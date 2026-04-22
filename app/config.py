"""Mobius RAG configuration.

2026-04-21 hardening: Ollama fallback removed. All LLM completions
route through the shared mobius-chat LLM Manager so Thompson-bandit
routing + quality telemetry cover rag-side calls too. Local dev needs
the same Vertex credentials production uses (via ADC); there is no
longer a local-first "just run Ollama" option.

Startup gates (enforced at module load):

  * ``DATABASE_URL`` — required, no localhost fallback. Cloud SQL
    only.
  * ``VERTEX_PROJECT_ID`` — required. Used for embeddings and as the
    ADC-authenticated project for Vertex SDK calls.
  * ``CHAT_INTERNAL_LLM_URL`` + ``MOBIUS_SKILL_LLM_INTERNAL_KEY`` —
    required in hosted mode (``ENV=prod``). These point at chat's
    ``/internal/skill-llm`` proxy so rag-side LLM calls participate
    in the same Thompson bandit. Dev may omit them to use the local
    path, but hosted deploy must set both.

Kept intentionally small — every knob is one env var so deploys can
tune without code changes.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from mobius-rag root so DATABASE_URL is correct even if
# the shell already set it (e.g. special chars in password).
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path, override=True)

# ── Environment ──────────────────────────────────────────────────────
ENV = os.getenv("ENV", "dev")  # dev | prod
_IS_HOSTED = ENV.strip().lower() in ("prod", "staging")

# ── Database ─────────────────────────────────────────────────────────
# Cloud SQL only; no localhost fallback. Set DATABASE_URL in .env:
# postgresql+asyncpg://postgres:PASSWORD@<host>:5432/mobius_rag
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL is required. Point to Cloud SQL "
        "(e.g. postgresql+asyncpg://postgres:PASSWORD@34.135.72.145:5432/mobius_rag). "
        "See docs/MIGRATE_LOCAL_TO_CLOUD.md"
    )

# ── GCS ──────────────────────────────────────────────────────────────
GCS_BUCKET = os.getenv("GCS_BUCKET", "mobius-rag-uploads-mobiusos")

# ── LLM routing ──────────────────────────────────────────────────────
# 2026-04-21: Ollama removed. All LLM generation calls go through
# mobius-chat's /internal/skill-llm proxy so Thompson-bandit routing
# and llm_calls analytics cover rag-side calls uniformly.
#
# Dev fallback: when CHAT_INTERNAL_LLM_URL is unset AND ENV=dev, the
# llm_manager_client falls back to calling Vertex directly (so
# local dev without a running chat instance still works for simple
# smoke tests). Hosted mode requires the proxy URL + key.
CHAT_INTERNAL_LLM_URL = os.getenv("CHAT_INTERNAL_LLM_URL")
MOBIUS_SKILL_LLM_INTERNAL_KEY = os.getenv("MOBIUS_SKILL_LLM_INTERNAL_KEY")

# Vertex settings used for:
#  * embeddings (gemini-embedding-001)
#  * direct-call fallback in local dev (no LLM manager proxy)
#  * ADC project hint for the Vertex SDK
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-flash")

# Critique retry: retry extraction when score < threshold (0.0-1.0).
# User can override via ?threshold=.
CRITIQUE_RETRY_THRESHOLD = float(os.getenv("CRITIQUE_RETRY_THRESHOLD", "0.6"))

# ── Uploads ──────────────────────────────────────────────────────────
# Hard cap on single-file upload size. Enforced in /upload before we
# allocate a full in-memory read. Default 100 MB — large enough for a
# multi-hundred-page provider manual PDF, small enough to keep a single
# Cloud Run instance from OOM on adversarial traffic.
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# ── Admin auth ───────────────────────────────────────────────────────
# All /admin/* routes require X-Admin-Key to match this value in hosted
# mode. If unset in hosted mode, /admin/* is blocked entirely (503) —
# the whole point is that prod without a key configured should be safer
# than prod with an open admin surface. Dev (ENV=dev) bypasses the
# check unless ADMIN_API_KEY is explicitly set.
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

# ── Embeddings ───────────────────────────────────────────────────────
# Defaults to Vertex when VERTEX_PROJECT_ID is set. OpenAI path kept
# as an alternative; no Ollama fallback. Must stay at 1536 dims to
# match the Chroma collection chat reads from.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER") or (
    "vertex" if VERTEX_PROJECT_ID else "openai"
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or (
    "gemini-embedding-001" if EMBEDDING_PROVIDER == "vertex" else "text-embedding-3-small"
)
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

# ── Google Drive OAuth (optional) ────────────────────────────────────
DRIVE_API_ENABLED = os.getenv("DRIVE_API_ENABLED", "false").lower() in ("true", "1", "yes")
GOOGLE_DRIVE_CLIENT_ID = os.getenv("GOOGLE_DRIVE_CLIENT_ID")
GOOGLE_DRIVE_CLIENT_SECRET = os.getenv("GOOGLE_DRIVE_CLIENT_SECRET")
GOOGLE_DRIVE_REDIRECT_URI = os.getenv("GOOGLE_DRIVE_REDIRECT_URI")
RAG_FRONTEND_URL = os.getenv("RAG_FRONTEND_URL", "http://localhost:8001")


# ── Startup validation ───────────────────────────────────────────────


class ConfigError(RuntimeError):
    """Raised when required config is missing or inconsistent."""


def assert_hosted_config() -> None:
    """Fail-fast at startup in hosted envs. No-op in ``ENV=dev``.

    Refuses to boot on misconfigured prod/staging rather than waiting
    for the first LLM call to surface the issue. Collects every
    problem at once so operators fix one batch instead of iterating.
    """
    if not _IS_HOSTED:
        return

    problems: list[str] = []

    if not VERTEX_PROJECT_ID:
        problems.append(
            "VERTEX_PROJECT_ID is unset. Required in hosted mode for "
            "ADC-authenticated Vertex calls (embeddings + fallback LLM)."
        )

    if not CHAT_INTERNAL_LLM_URL:
        problems.append(
            "CHAT_INTERNAL_LLM_URL is unset. Hosted mode requires rag's "
            "LLM calls to route through chat's /internal/skill-llm proxy "
            "so Thompson-bandit telemetry covers the full stack. Point "
            "at e.g. https://mobius-chat-<hash>-uc.a.run.app/internal/skill-llm"
        )

    if not MOBIUS_SKILL_LLM_INTERNAL_KEY:
        problems.append(
            "MOBIUS_SKILL_LLM_INTERNAL_KEY is unset. Required when "
            "CHAT_INTERNAL_LLM_URL is set — chat's /internal/skill-llm "
            "rejects requests without the matching X-Mobius-Skill-LLM-Key "
            "header. Pull from Secret Manager."
        )

    # Legacy Ollama env vars: if set, refuse to boot so operators know
    # the support is gone and update their deploy config.
    stale = [k for k in ("LLM_PROVIDER", "OLLAMA_BASE_URL", "OLLAMA_MODEL",
                         "OLLAMA_NUM_PREDICT") if os.getenv(k)]
    if stale:
        problems.append(
            f"Stale Ollama env vars present ({stale}). Ollama support "
            "was removed 2026-04-21. Remove these from your deploy "
            "config; LLM calls now route through CHAT_INTERNAL_LLM_URL."
        )

    if problems:
        header = (
            f"Refusing to start in ENV={ENV!r}: "
            f"{len(problems)} config problem(s):"
        )
        bullets = "\n  - " + "\n  - ".join(problems)
        hint = (
            "\n\nFix the env and restart. If this is local dev, unset "
            "ENV (or set ENV=dev) to disable this gate."
        )
        raise ConfigError(header + bullets + hint)
