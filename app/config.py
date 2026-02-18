import os
from dotenv import load_dotenv

load_dotenv()

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

# Google Drive OAuth (optional; enables Drive import when set)
DRIVE_API_ENABLED = os.getenv("DRIVE_API_ENABLED", "false").lower() in ("true", "1", "yes")
GOOGLE_DRIVE_CLIENT_ID = os.getenv("GOOGLE_DRIVE_CLIENT_ID")
GOOGLE_DRIVE_CLIENT_SECRET = os.getenv("GOOGLE_DRIVE_CLIENT_SECRET")
GOOGLE_DRIVE_REDIRECT_URI = os.getenv("GOOGLE_DRIVE_REDIRECT_URI")  # e.g. http://localhost:8001/drive/callback
RAG_FRONTEND_URL = os.getenv("RAG_FRONTEND_URL", "http://localhost:8001")  # For post-OAuth redirect
