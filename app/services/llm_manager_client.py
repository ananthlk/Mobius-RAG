"""Client for mobius-chat's LLM Manager (``/internal/skill-llm``).

2026-04-21 hardening: rag's extraction + critique calls go through
here instead of hitting Vertex directly. Benefits:

* **Unified Thompson-bandit**. Chat's ``llm_manager.generate`` routes
  across Groq / Anthropic / Vertex and records per-turn quality +
  latency + cost into ``llm_calls`` + ``adjudication_scores``.
  Rag-side calls now show up in the same dashboards.

* **Single source of model-selection truth**. No drift between
  "which model chat picks for this stage" vs "which model rag picks".
  Both ask the same router.

* **Shared circuit breakers + rate limits**. If Vertex starts
  throttling, both chat and rag back off together instead of
  fighting for quota.

Contract (mirrors mobius-chat's ``SkillLLMRequest``):

    POST {CHAT_INTERNAL_LLM_URL}
    Headers:
        Content-Type: application/json
        X-Mobius-Skill-LLM-Key: <MOBIUS_SKILL_LLM_INTERNAL_KEY>
    Body:
        {"system": "...", "user": "...", "stage": "<one of the allowed stages>",
         "max_tokens": N, "correlation_id": "...", "thread_id": "...",
         "mode": "factual|creative|..."}

    Response:
        {"text": "...", "usage": {...}}

Dev fallback: when ``CHAT_INTERNAL_LLM_URL`` is unset AND ``ENV=dev``,
we hit Vertex directly via the legacy ``get_llm_provider()``. Hosted
mode requires the proxy URL (enforced by ``assert_hosted_config``).

Streaming: this client is non-streaming. The previous
``stream_generate`` path in rag existed to emit token-by-token
progress to the frontend SSE stream. Routing through chat's
manager is worth the streaming loss because unified bandit data
beats tokens-while-you-wait UX, and we can add a streaming variant
to the manager later if it matters.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from app.config import (
    CHAT_INTERNAL_LLM_URL,
    ENV,
    MOBIUS_SKILL_LLM_INTERNAL_KEY,
)

logger = logging.getLogger(__name__)


# Stages accepted by chat's /internal/skill-llm. Keep in sync with
# mobius-chat/app/main.py _SKILL_LLM_ALLOWED_STAGES. Typos here mean
# the endpoint returns HTTP 400; we fail-loud rather than silently
# route to a wrong stage.
RAG_STAGES: frozenset[str] = frozenset({
    "rag_extraction",
    "rag_critique",
    "rag_lexicon_triage",
})


# Reasonable default. Extraction on a 20-page chunk can take ~20s;
# critique is typically ~5s. Caller can override.
_DEFAULT_TIMEOUT_S = 120


class LLMManagerError(RuntimeError):
    """Raised when the LLM manager proxy returns an error we can't recover from."""


def generate_sync(
    *,
    system: str = "",
    user: str = "",
    stage: str,
    max_tokens: int = 4096,
    correlation_id: str | None = None,
    thread_id: str | None = None,
    mode: str | None = None,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> tuple[str, dict[str, Any]]:
    """Synchronous ``generate`` via chat's LLM manager.

    Returns ``(text, usage_dict)``. ``usage_dict`` carries whatever
    chat's router attached (model id, tokens, latency, cost, etc.) —
    callers typically attach this to their ``ExtractedFact`` /
    ``HierarchicalChunk`` rows for provenance.

    Raises ``LLMManagerError`` on non-2xx from the proxy. Dev
    fallback path is called only when ``CHAT_INTERNAL_LLM_URL`` is
    unset AND ``ENV != "prod"`` — hosted mode always uses the proxy.
    """
    if stage not in RAG_STAGES:
        raise LLMManagerError(
            f"Unknown stage {stage!r}. Allowed: {sorted(RAG_STAGES)}. "
            "Add new stages to both this client's RAG_STAGES set AND "
            "mobius-chat/app/main.py _SKILL_LLM_ALLOWED_STAGES."
        )

    if not CHAT_INTERNAL_LLM_URL or not MOBIUS_SKILL_LLM_INTERNAL_KEY:
        if ENV.lower() in ("prod", "staging"):
            raise LLMManagerError(
                "CHAT_INTERNAL_LLM_URL / MOBIUS_SKILL_LLM_INTERNAL_KEY "
                "missing in hosted mode. Check Secret Manager wiring."
            )
        return _dev_fallback(
            system=system, user=user, stage=stage, max_tokens=max_tokens,
        )

    body = json.dumps({
        "system": system,
        "user": user,
        "stage": stage,
        "max_tokens": max_tokens,
        "correlation_id": correlation_id,
        "thread_id": thread_id,
        "mode": mode,
    }).encode("utf-8")

    req = urllib.request.Request(
        CHAT_INTERNAL_LLM_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Mobius-Skill-LLM-Key": MOBIUS_SKILL_LLM_INTERNAL_KEY,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise LLMManagerError(
            f"LLM manager returned HTTP {exc.code} for stage={stage!r}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise LLMManagerError(
            f"LLM manager unreachable at {CHAT_INTERNAL_LLM_URL!r}: {exc}"
        ) from exc

    text = payload.get("text") or ""
    usage = payload.get("usage") or {}
    return text, usage


# ── Dev fallback ─────────────────────────────────────────────────────


def _dev_fallback(
    *, system: str, user: str, stage: str, max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    """Hit Vertex directly when no proxy is configured.

    Only runs in ``ENV=dev`` (the caller guards the hosted path).
    Logs a loud warning so dev mistakes don't accidentally ship —
    if someone sees ``dev_fallback`` in their prod log, they know
    the manager proxy is misconfigured.
    """
    logger.warning(
        "llm_manager_client: dev fallback — hitting Vertex directly "
        "(stage=%s, prompt_chars=%d). Set CHAT_INTERNAL_LLM_URL to "
        "route through the shared LLM Manager.",
        stage, len(system) + len(user),
    )
    from app.services.llm_provider import get_llm_provider
    import asyncio

    provider = get_llm_provider()
    prompt = f"{system}\n\n{user}" if system else user

    async def _call() -> str:
        return await provider.generate(prompt, max_tokens=max_tokens)

    try:
        text = asyncio.run(_call())
    except RuntimeError:
        # Already inside an event loop (FastAPI route context).
        # Create a nested task via get_event_loop().run_until_complete
        # equivalent — simplest is a thread.
        import concurrent.futures as _cf
        with _cf.ThreadPoolExecutor(max_workers=1) as ex:
            text = ex.submit(lambda: asyncio.new_event_loop().run_until_complete(_call())).result()
    return text, {"provider": "vertex_dev_fallback", "model": "unknown"}


async def generate(
    *,
    system: str = "",
    user: str = "",
    stage: str,
    max_tokens: int = 4096,
    correlation_id: str | None = None,
    thread_id: str | None = None,
    mode: str | None = None,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> tuple[str, dict[str, Any]]:
    """Async wrapper for ``generate_sync``. Runs the sync HTTP call
    in a thread so the caller's event loop isn't blocked.
    """
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: generate_sync(
            system=system, user=user, stage=stage, max_tokens=max_tokens,
            correlation_id=correlation_id, thread_id=thread_id, mode=mode,
            timeout_s=timeout_s,
        ),
    )
