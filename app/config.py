import os
import sys
from pathlib import Path

# Load env: module .env first, then global mobius-config/.env (same helper as mobius-chat)
_repo_root = Path(__file__).resolve().parent.parent
_config_dir = _repo_root.parent / "mobius-config"
if _config_dir.exists() and str(_config_dir) not in sys.path:
    sys.path.insert(0, str(_config_dir))
try:
    from env_helper import load_env
    load_env(_repo_root)
except ImportError:
    from dotenv import load_dotenv
    load_dotenv(_repo_root / ".env", override=True)

# Resolve relative GOOGLE_APPLICATION_CREDENTIALS to absolute (repo root = parent of app/)
# Skip placeholders (e.g. /path/to/your-service-account.json) — env_helper already cleared those
_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if _creds and "/path/to/" not in _creds and "your-service-account" not in _creds:
    if not Path(_creds).is_absolute():
        _abs = (_repo_root / _creds).resolve()
        if _abs.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_abs)
    elif not Path(_creds).expanduser().is_file():
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

# Environment
ENV = os.getenv("ENV", "dev")  # dev or prod

# Database
if ENV == "prod":
    DATABASE_URL = os.getenv("DATABASE_URL")  # e.g., postgresql+asyncpg://user:pass@host/db
else:
    # Dev defaults
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/mobius_rag"
    )

# GCS
GCS_BUCKET = os.getenv("GCS_BUCKET", "mobius-rag-uploads-mobiusos")

# LLM Provider Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "vertex"

# Ollama settings (for local development)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "8192"))

# Vertex AI (Gemini) settings — recommended first option for production
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-1.5-pro")

# Critique retry: retry extraction when score < threshold (0.0–1.0). User can override via ?threshold=.
CRITIQUE_RETRY_THRESHOLD = float(os.getenv("CRITIQUE_RETRY_THRESHOLD", "0.6"))

# Embedding provider (openai | vertex) – defaults to vertex when VERTEX_PROJECT_ID is set
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER") or ("vertex" if os.getenv("VERTEX_PROJECT_ID") else "openai")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or (
    "gemini-embedding-001" if EMBEDDING_PROVIDER == "vertex" else "text-embedding-3-small"
)
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
