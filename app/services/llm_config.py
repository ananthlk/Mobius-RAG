"""LLM config loader: load versioned LLM config from DB or files and build provider from config dict.

Uses the LLM provider registry (llm_provider.register_provider) so any registered
provider can be selected via YAML or DB without code changes here.

Resolution order: DB first (if session given), then YAML file.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import LlmConfig
from app.services.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

# Directory containing LLM config YAML files (app/llm_configs/)
_LLM_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "llm_configs"


def _row_to_config(row: LlmConfig) -> Dict[str, Any]:
    """Turn LlmConfig ORM row into the same dict shape as YAML (provider, model, options, ollama, vertex, openai)."""
    return {
        "provider": row.provider or "",
        "model": row.model,
        "version": row.version_label or row.name,
        "options": dict(row.options) if row.options else {},
        "ollama": dict(row.ollama) if row.ollama else {},
        "vertex": dict(row.vertex) if row.vertex else {},
        "openai": dict(row.openai) if row.openai else {},
    }


async def get_llm_config_from_db(db: AsyncSession, name: str) -> Optional[Dict[str, Any]]:
    """
    Load one LLM config from the database by name.
    Returns dict with keys provider, model, options, ollama, vertex, openai, or None if not found.
    """
    result = await db.execute(select(LlmConfig).where(LlmConfig.name == name))
    row = result.scalar_one_or_none()
    if row is None:
        return None
    return _row_to_config(row)


async def list_llm_config_names_from_db(db: AsyncSession) -> List[str]:
    """Return sorted list of LLM config names stored in the database."""
    result = await db.execute(select(LlmConfig.name).order_by(LlmConfig.name))
    return [r[0] for r in result.fetchall()]


async def save_llm_config(
    db: AsyncSession,
    name: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Upsert an LLM config in the database. config dict can have provider, model, version,
    options, ollama, vertex, openai. Returns the saved config as dict (same shape as get).
    """
    provider = (config.get("provider") or "").strip() or None
    if not provider:
        raise ValueError("Config must specify 'provider'")
    model = config.get("model")
    version_label = config.get("version")
    options = config.get("options")
    ollama = config.get("ollama")
    vertex = config.get("vertex")
    openai = config.get("openai")
    if options is not None and not isinstance(options, dict):
        options = {}
    if ollama is not None and not isinstance(ollama, dict):
        ollama = {}
    if vertex is not None and not isinstance(vertex, dict):
        vertex = {}
    if openai is not None and not isinstance(openai, dict):
        openai = {}

    result = await db.execute(select(LlmConfig).where(LlmConfig.name == name))
    row = result.scalar_one_or_none()
    if row is None:
        row = LlmConfig(
            name=name,
            provider=provider,
            model=model,
            version_label=version_label,
            options=options or {},
            ollama=ollama or {},
            vertex=vertex or {},
            openai=openai or {},
        )
        db.add(row)
    else:
        row.provider = provider
        row.model = model
        row.version_label = version_label
        row.options = options if options is not None else (row.options or {})
        row.ollama = ollama if ollama is not None else (row.ollama or {})
        row.vertex = vertex if vertex is not None else (row.vertex or {})
        row.openai = openai if openai is not None else (row.openai or {})
    await db.flush()
    await db.refresh(row)
    return _row_to_config(row)


async def get_llm_config_resolved(version_or_name: str, db: AsyncSession) -> Optional[Dict[str, Any]]:
    """
    Load LLM config by version/name: try DB first, then YAML file.
    Use this from API and worker when a DB session is available.
    """
    cfg = await get_llm_config_from_db(db, version_or_name)
    if cfg is not None:
        return cfg
    return get_llm_config(version_or_name)


def get_llm_config(version_or_name: str) -> Optional[Dict[str, Any]]:
    """
    Load LLM config by version or name from YAML only (e.g. "default", "production", "v1").
    Returns dict with keys: provider, model, options, and provider-specific (ollama, vertex, etc.).
    Returns None if file not found.
    """
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


def list_llm_config_names_yaml() -> List[str]:
    """Return sorted list of config names from YAML files (stems of *.yaml)."""
    names = []
    if _LLM_CONFIGS_DIR.is_dir():
        for p in _LLM_CONFIGS_DIR.iterdir():
            if p.suffix == ".yaml" and p.stem:
                names.append(p.stem)
    return sorted(names) if names else []


def get_llm_provider_from_config(config: Dict[str, Any]) -> LLMProvider:
    """
    Build an LLMProvider instance from a config dict (from get_llm_config).
    Uses the provider registry: config["provider"] must match a registered name
    (e.g. "ollama", "vertex"). Provider-specific keys (ollama.base_url, vertex.project_id, etc.)
    are passed through to the registered factory.
    """
    from app.services.llm_provider import _PROVIDER_REGISTRY, list_providers

    provider = (config.get("provider") or "").lower().strip()
    if not provider:
        raise ValueError("Config must specify 'provider' (e.g. ollama, vertex)")
    factory = _PROVIDER_REGISTRY.get(provider)
    if not factory:
        raise ValueError(f"Unknown LLM provider: {provider}. Registered: {list_providers()}")
    return factory(config)
