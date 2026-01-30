"""LLM Provider abstraction for supporting multiple LLM backends.

To add a new LLM module:
1. Implement a class that subclasses LLMProvider and implements stream_generate() and generate().
2. Call register_provider("name", factory) where factory is a callable (config_dict) -> LLMProvider.
3. Add a YAML under app/llm_configs/ with provider: "name" and any provider-specific keys.
See app/llm_configs/README.md for details.
"""
from abc import ABC, abstractmethod
from typing import AsyncIterator, Callable, Dict, Any
import asyncio
import json
import logging
import queue
import threading
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

# Registry: provider name -> factory(config: dict) -> LLMProvider
_PROVIDER_REGISTRY: Dict[str, Callable[[Dict[str, Any]], "LLMProvider"]] = {}


def register_provider(name: str, factory: Callable[[Dict[str, Any]], "LLMProvider"]) -> None:
    """Register an LLM provider. factory(config_dict) must return an LLMProvider instance."""
    name = (name or "").lower().strip()
    if not name:
        raise ValueError("Provider name must be non-empty")
    _PROVIDER_REGISTRY[name] = factory


def list_providers() -> list[str]:
    """Return registered provider names."""
    return sorted(_PROVIDER_REGISTRY.keys())


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream LLM response tokens."""
        pass
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate complete LLM response (non-streaming)."""
        pass


def _ollama_stream_producer(
    base_url: str, model: str, prompt: str, opts: dict, out: queue.Queue
) -> None:
    """Runs in a thread. Reads Ollama stream line-by-line, puts each chunk into out (threading.Queue).
    Puts None when done; puts ('error', msg) on failure."""
    req_data = {"model": model, "prompt": prompt, "stream": True, "options": opts}
    data = json.dumps(req_data).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if "response" in d:
                        out.put(d["response"])
                    if d.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
        out.put(None)
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = ""
        out.put(("error", f"Ollama API error: {e.code} - {err_body}"))
    except Exception as e:
        out.put(("error", str(e)))


def _ollama_request(base_url: str, model: str, prompt: str, stream: bool, **kwargs) -> tuple[str | None, list[str] | None]:
    """Blocking Ollama HTTP request. Returns (error_msg, None) on failure or (None, chunks) on success.
    For stream=True, chunks is list of response strings; for stream=False, chunks is [full_response].
    """
    req_data = {"model": model, "prompt": prompt, "stream": stream, **kwargs}
    data = json.dumps(req_data).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            if stream:
                chunks = []
                for line in resp:
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        if "response" in d:
                            chunks.append(d["response"])
                        if d.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
                return (None, chunks)
            else:
                body = resp.read().decode("utf-8")
                d = json.loads(body)
                return (None, [d.get("response", "")])
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = ""
        return (f"Ollama API error: {e.code} - {err_body}", None)
    except Exception as e:
        return (str(e), None)


class OllamaProvider(LLMProvider):
    """Ollama provider for local development. Uses urllib (no aiohttp)."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1:8b", num_predict: int = 8192):
        self.base_url = base_url
        self.model = model
        self.num_predict = num_predict
    
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream responses from Ollama. Yields chunks as they arrive (live streaming)."""
        opts = {"num_predict": self.num_predict}
        if "options" in kwargs:
            opts = {**opts, **kwargs.pop("options")}
        loop = asyncio.get_event_loop()
        q: queue.Queue = queue.Queue()

        def get() -> object:
            return q.get()

        t = threading.Thread(
            target=_ollama_stream_producer,
            args=(self.base_url, self.model, prompt, opts, q),
            daemon=True,
        )
        t.start()
        while True:
            item = await loop.run_in_executor(None, get)
            if item is None:
                break
            if isinstance(item, tuple) and item[0] == "error":
                raise Exception(item[1])
            yield item
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate complete response from Ollama."""
        opts = {"num_predict": self.num_predict}
        if "options" in kwargs:
            opts = {**opts, **kwargs.pop("options")}
        loop = asyncio.get_event_loop()
        err, chunks = await loop.run_in_executor(
            None,
            lambda: _ollama_request(self.base_url, self.model, prompt, False, options=opts, **kwargs),
        )
        if err:
            raise Exception(err)
        return (chunks or [""])[0]


def _vertex_stream_producer(model_name: str, prompt: str, gen_config: dict, out: queue.Queue) -> None:
    """Runs in a thread. Streams Vertex AI (Gemini) response into out. Puts None when done; puts ('error', msg) on failure."""
    try:
        from vertexai.generative_models import GenerativeModel
        model = GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=gen_config,
            stream=True,
        )
        for chunk in response:
            if chunk.text:
                out.put(chunk.text)
        out.put(None)
    except Exception as e:
        out.put(("error", str(e)))


def _vertex_generate_sync(model_name: str, prompt: str, gen_config: dict) -> str:
    """Blocking Vertex AI (Gemini) generate. Run via asyncio.to_thread to avoid blocking the event loop."""
    from vertexai.generative_models import GenerativeModel
    model = GenerativeModel(model_name)
    response = model.generate_content(prompt, generation_config=gen_config)
    return response.text or ""


class VertexAIProvider(LLMProvider):
    """Vertex AI (Gemini) provider for production. Sync SDK calls run off the event loop."""
    
    def __init__(self, project_id: str, location: str = "us-central1", model: str = "gemini-1.5-pro"):
        try:
            import vertexai
            vertexai.init(project=project_id, location=location)
            self.model_name = model
        except ImportError:
            raise ImportError("google-cloud-aiplatform is required for Vertex AI provider. Install with: pip install -e \".[vertex]\"")
        except Exception as e:
            raise Exception(f"Failed to initialize Vertex AI: {str(e)}")
    
    def _generation_config(self, **kwargs) -> dict:
        return {"temperature": 0.1, **kwargs}
    
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream responses from Vertex AI (Gemini). Sync iterator runs in a thread."""
        gen_config = self._generation_config(**kwargs)
        loop = asyncio.get_event_loop()
        q: queue.Queue = queue.Queue()

        def get() -> object:
            return q.get()

        t = threading.Thread(
            target=_vertex_stream_producer,
            args=(self.model_name, prompt, gen_config, q),
            daemon=True,
        )
        t.start()
        while True:
            item = await loop.run_in_executor(None, get)
            if item is None:
                break
            if isinstance(item, tuple) and item[0] == "error":
                raise Exception(item[1])
            yield item

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate complete response from Vertex AI (Gemini). Sync call runs in a thread."""
        gen_config = self._generation_config(**kwargs)
        return await asyncio.to_thread(_vertex_generate_sync, self.model_name, prompt, gen_config)


def _ollama_factory(config: Dict[str, Any]) -> "LLMProvider":
    """Build OllamaProvider from config dict (for registry)."""
    ollama = config.get("ollama") or {}
    from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_NUM_PREDICT
    base_url = ollama.get("base_url") or OLLAMA_BASE_URL
    model = config.get("model") or OLLAMA_MODEL
    options = config.get("options") or {}
    num_predict = options.get("num_predict")
    if num_predict is None:
        num_predict = OLLAMA_NUM_PREDICT
    return OllamaProvider(base_url=base_url, model=model, num_predict=int(num_predict))


def _vertex_factory(config: Dict[str, Any]) -> "LLMProvider":
    """Build VertexAIProvider from config dict (for registry)."""
    vertex = config.get("vertex") or {}
    from app.config import VERTEX_PROJECT_ID, VERTEX_LOCATION, VERTEX_MODEL
    project_id = vertex.get("project_id") or VERTEX_PROJECT_ID
    if not project_id:
        raise ValueError("Vertex AI requires project_id (vertex.project_id or VERTEX_PROJECT_ID)")
    location = vertex.get("location") or VERTEX_LOCATION
    model = config.get("model") or VERTEX_MODEL
    return VertexAIProvider(project_id=project_id, location=location, model=model)


# Optional: OpenAI provider (register only if openai package is available)
def _openai_factory(config: Dict[str, Any]) -> "LLMProvider":
    """Build OpenAIProvider from config dict (for registry). Requires: pip install openai."""
    try:
        from app.services.llm_provider_openai import OpenAIProvider  # noqa: F401
    except ImportError:
        raise ImportError("OpenAI provider requires: pip install openai")
    openai_config = config.get("openai") or {}
    api_key = openai_config.get("api_key")
    if not api_key or (isinstance(api_key, str) and api_key.strip() == "***"):
        import os
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI requires api_key (openai.api_key or OPENAI_API_KEY)")
    model = config.get("model") or "gpt-4o-mini"
    base_url = openai_config.get("base_url")
    return OpenAIProvider(api_key=api_key, model=model, base_url=base_url)


try:
    from app.services.llm_provider_openai import OpenAIProvider  # noqa: F401
    register_provider("openai", _openai_factory)
except ImportError:
    pass

# Register built-in providers so config-driven and future modules can resolve by name
register_provider("ollama", _ollama_factory)
register_provider("vertex", _vertex_factory)


def get_llm_provider():
    """Get LLM provider based on environment configuration (config.py)."""
    from app.config import (
        LLM_PROVIDER,
        OLLAMA_BASE_URL,
        OLLAMA_MODEL,
        OLLAMA_NUM_PREDICT,
        VERTEX_PROJECT_ID,
        VERTEX_LOCATION,
        VERTEX_MODEL
    )
    # Use registry when possible so env behaves like a single built-in config
    cfg = {
        "provider": LLM_PROVIDER,
        "model": OLLAMA_MODEL if LLM_PROVIDER == "ollama" else VERTEX_MODEL,
        "options": {"num_predict": OLLAMA_NUM_PREDICT} if LLM_PROVIDER == "ollama" else {},
        "ollama": {"base_url": OLLAMA_BASE_URL},
        "vertex": {"project_id": VERTEX_PROJECT_ID, "location": VERTEX_LOCATION},
    }
    provider_name = (LLM_PROVIDER or "").lower()
    factory = _PROVIDER_REGISTRY.get(provider_name)
    if factory:
        return factory(cfg)
    raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}. Registered: {list_providers()}")
