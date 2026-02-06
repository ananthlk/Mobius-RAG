"""Embedding provider abstraction for supporting multiple embedding backends.

Providers: openai, vertex.
Config: from app.config or config dict (embedding_model, dimensions, provider-specific keys).
"""
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any
import asyncio
import logging
import os

logger = logging.getLogger(__name__)

_PROVIDER_REGISTRY: Dict[str, Callable[[Dict[str, Any]], "EmbeddingProvider"]] = {}


def register_provider(name: str, factory: Callable[[Dict[str, Any]], "EmbeddingProvider"]) -> None:
    name = (name or "").lower().strip()
    if not name:
        raise ValueError("Provider name must be non-empty")
    _PROVIDER_REGISTRY[name] = factory


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for a list of texts. Synchronous; caller may wrap in asyncio.to_thread for async."""
        pass


def _openai_embed(texts: list[str], model: str, api_key: str | None, base_url: str | None) -> list[list[float]]:
    """Blocking OpenAI embeddings call."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "OpenAI package required for embeddings. Install with: pip install -e '.[openai]'"
        ) from e
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)
    # OpenAI accepts list of strings; max 2048 inputs per request
    batch_size = 100
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(input=batch, model=model)
        for d in resp.data:
            all_embeddings.append(d.embedding)
    return all_embeddings


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: dict):
        self.model = config.get("model") or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.api_key = config.get("api_key") or config.get("openai", {}).get("api_key")
        self.base_url = config.get("base_url") or config.get("openai", {}).get("base_url")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return _openai_embed(texts, self.model, self.api_key, self.base_url)


def _vertex_embed(
    texts: list[str],
    model: str,
    project: str,
    location: str,
    output_dimensionality: int | None = None,
) -> list[list[float]]:
    """Blocking Vertex AI embeddings call. Uses same credentials as LLM (GOOGLE_APPLICATION_CREDENTIALS)."""
    try:
        import vertexai
        from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
    except ImportError as e:
        raise ImportError(
            "Vertex AI package required for embeddings. Either install it: pip install -e '.[vertex]' "
            "(from mobius-rag repo), or use OpenAI instead by setting EMBEDDING_PROVIDER=openai and OPENAI_API_KEY."
        ) from e
    vertexai.init(project=project, location=location)
    emb_model = TextEmbeddingModel.from_pretrained(model)
    all_embeddings: list[list[float]] = []
    kwargs = dict(output_dimensionality=output_dimensionality) if output_dimensionality else {}
    # gemini-embedding-001 takes 1 input per request; text-embedding-* can batch
    batch_size = 1 if model == "gemini-embedding-001" else 5
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = [TextEmbeddingInput(t, task_type="RETRIEVAL_DOCUMENT") for t in batch]
        resp = emb_model.get_embeddings(inputs, **kwargs)
        for r in resp:
            all_embeddings.append(list(r.values))
    return all_embeddings


class VertexEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: dict):
        self.model = config.get("model") or os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        self.dimensions = config.get("dimensions") or int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
        v = config.get("vertex", {}) or {}
        self.project = v.get("project_id") or os.getenv("VERTEX_PROJECT_ID")
        self.location = v.get("location") or os.getenv("VERTEX_LOCATION", "us-central1")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.project:
            raise ValueError("VERTEX_PROJECT_ID not set")
        return _vertex_embed(
            texts, self.model, self.project, self.location,
            output_dimensionality=self.dimensions if self.model == "gemini-embedding-001" else None,
        )


def _openai_factory(config: dict) -> EmbeddingProvider:
    return OpenAIEmbeddingProvider(config)


def _vertex_factory(config: dict) -> EmbeddingProvider:
    return VertexEmbeddingProvider(config)


register_provider("openai", _openai_factory)
register_provider("vertex", _vertex_factory)


def get_embedding_provider(config: dict | None = None) -> EmbeddingProvider:
    """Return embedding provider from config. If config is None, use app.config env vars."""
    if config is None:
        from app.config import EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS
        config = {
            "provider": EMBEDDING_PROVIDER,
            "model": EMBEDDING_MODEL,
            "dimensions": EMBEDDING_DIMENSIONS,
        }
    provider_name = (config.get("provider") or "openai").lower().strip()
    factory = _PROVIDER_REGISTRY.get(provider_name)
    if not factory:
        raise ValueError(f"Unknown embedding provider: {provider_name}. Available: {list(_PROVIDER_REGISTRY.keys())}")
    return factory(config)


async def embed_async(texts: list[str], provider: EmbeddingProvider | None = None) -> list[list[float]]:
    """Async wrapper: run embed in thread to avoid blocking."""
    p = provider or get_embedding_provider()
    return await asyncio.to_thread(p.embed, texts)
