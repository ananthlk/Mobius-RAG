"""LLM config loader: load versioned LLM config from files and build provider from config dict."""
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from app.services.llm_provider import OllamaProvider, VertexAIProvider, LLMProvider

logger = logging.getLogger(__name__)

# Directory containing LLM config YAML files (app/llm_configs/)
_LLM_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "llm_configs"


def get_llm_config(version_or_name: str) -> Optional[Dict[str, Any]]:
    """
    Load LLM config by version or name (e.g. "default", "production", "v1").
    Returns dict with keys: provider, model, options, and provider-specific (ollama, vertex).
    Returns None if file not found.
    """
    # Try exact filename first (default.yaml, production.yaml)
    path = _LLM_CONFIGS_DIR / f"{version_or_name}.yaml"
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return None
        return dict(data)
    except Exception as e:
        logger.warning(f"Failed to load LLM config {version_or_name}: {e}")
        return None


def get_llm_provider_from_config(config: Dict[str, Any]) -> LLMProvider:
    """
    Build an LLMProvider instance from a config dict (from get_llm_config).
    Config must have "provider" ("ollama" or "vertex"), "model", and provider-specific keys.
    For ollama: ollama.base_url or env OLLAMA_BASE_URL; options.num_predict.
    For vertex: vertex.project_id (or VERTEX_PROJECT_ID env), vertex.location.
    """
    provider = (config.get("provider") or "").lower()
    model = config.get("model") or ""
    options = config.get("options") or {}

    if provider == "ollama":
        ollama = config.get("ollama") or {}
        base_url = ollama.get("base_url")
        if base_url is None:
            from app.config import OLLAMA_BASE_URL
            base_url = OLLAMA_BASE_URL
        num_predict = options.get("num_predict")
        if num_predict is None:
            from app.config import OLLAMA_NUM_PREDICT
            num_predict = OLLAMA_NUM_PREDICT
        return OllamaProvider(
            base_url=base_url,
            model=model or "llama3.1:8b",
            num_predict=int(num_predict),
        )
    elif provider == "vertex":
        vertex = config.get("vertex") or {}
        project_id = vertex.get("project_id")
        if project_id is None:
            from app.config import VERTEX_PROJECT_ID
            project_id = VERTEX_PROJECT_ID
        if not project_id:
            raise ValueError("Vertex AI requires project_id (vertex.project_id or VERTEX_PROJECT_ID)")
        location = vertex.get("location") or "us-central1"
        return VertexAIProvider(
            project_id=project_id,
            location=location,
            model=model or "gemini-1.5-pro",
        )
    else:
        raise ValueError(f"Unknown LLM provider in config: {provider}")
