"""OpenAI API LLM provider (optional). Requires: pip install openai."""
from typing import AsyncIterator
import logging

from openai import AsyncOpenAI

from app.services.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (chat completions)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        temperature: float = 0.1,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature

    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream response from OpenAI chat completions."""
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        stream = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=kwargs.get("temperature", self.temperature),
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate full response from OpenAI chat completions."""
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            temperature=kwargs.get("temperature", self.temperature),
        )
        if resp.choices and resp.choices[0].message.content:
            return resp.choices[0].message.content
        return ""
