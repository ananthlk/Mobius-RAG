"""LLM Provider abstraction for supporting multiple LLM backends."""
from abc import ABC, abstractmethod
from typing import AsyncIterator
import asyncio
import json
import logging
import queue
import threading
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


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


class VertexAIProvider(LLMProvider):
    """Vertex AI provider for production."""
    
    def __init__(self, project_id: str, location: str = "us-central1", model: str = "gemini-1.5-pro"):
        try:
            import vertexai
            vertexai.init(project=project_id, location=location)
            self.model_name = model
        except ImportError:
            raise ImportError("google-cloud-aiplatform is required for Vertex AI provider")
        except Exception as e:
            raise Exception(f"Failed to initialize Vertex AI: {str(e)}")
    
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream responses from Vertex AI."""
        from vertexai.generative_models import GenerativeModel
        
        model = GenerativeModel(self.model_name)
        generation_config = {
            "temperature": 0.1,
            **kwargs
        }
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            stream=True
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate complete response from Vertex AI."""
        from vertexai.generative_models import GenerativeModel
        
        model = GenerativeModel(self.model_name)
        generation_config = {
            "temperature": 0.1,
            **kwargs
        }
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return response.text


def get_llm_provider():
    """Get LLM provider based on configuration."""
    from app.config import (
        LLM_PROVIDER,
        OLLAMA_BASE_URL,
        OLLAMA_MODEL,
        OLLAMA_NUM_PREDICT,
        VERTEX_PROJECT_ID,
        VERTEX_LOCATION,
        VERTEX_MODEL
    )
    
    if LLM_PROVIDER == "ollama":
        return OllamaProvider(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            num_predict=OLLAMA_NUM_PREDICT
        )
    elif LLM_PROVIDER == "vertex":
        if not VERTEX_PROJECT_ID:
            raise ValueError("VERTEX_PROJECT_ID is required for Vertex AI provider")
        return VertexAIProvider(
            project_id=VERTEX_PROJECT_ID,
            location=VERTEX_LOCATION,
            model=VERTEX_MODEL
        )
    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")
