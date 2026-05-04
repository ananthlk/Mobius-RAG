"""Skills router — POST /api/skills/v1/corpus_search.

Exposes the corpus retrieval pipeline (BM25 + pgvector + RRF + rerank)
as a versioned skill endpoint.  Consumers:

  * mobius-rag Repository UI  (direct call from the frontend)
  * mobius-chat retriever      (replaces the split BM25/vector/RRF stack
                                in Phase 3 of the extraction plan)

Contract is frozen at v1; breaking changes bump the version prefix.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.corpus_search import (
    CorpusSearchRequest,
    CorpusSearchResponse,
    corpus_search,
)
from app.services.corpus_search_agent import (
    CorpusSearchAgentRequest,
    CorpusSearchAgentResponse,
    corpus_search_agent,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/skills/v1", tags=["skills"])


@router.post("/corpus_search", response_model=CorpusSearchResponse)
async def skill_corpus_search(
    body: CorpusSearchRequest,
    db: AsyncSession = Depends(get_db),
    x_caller: str | None = Header(default=None, alias="X-Caller"),
    x_caller_id: str | None = Header(default=None, alias="X-Caller-Id"),
) -> CorpusSearchResponse:
    """Run BM25 / semantic / hybrid corpus search and return ranked chunks.

    **mode** — search arm strategy:
    - ``corpus``    — hybrid BM25 ⊕ pgvector, RRF-fused, reranked (default)
    - ``precision`` — BM25-only; best for exact codes / HCPCS lookups
    - ``recall``    — pgvector-only; best for "what do we know about X"

    **tag_mode** — how lexicon-extracted j/d/p tags filter candidates.
    Pick based on user intent:
    - ``auto`` (default) — strict first, fall through to relaxed if 0 hits.
                            Good general-purpose default.
    - ``strict`` — require ``documents.payer/state/program`` to match
                    the query's jurisdiction. Returns empty if no
                    authoring docs exist. Use when grounding a specific
                    claim about a specific authority (e.g. "What does
                    AHCA say about X?", "What is Sunshine Health's
                    appeal window?"). Authoring-truth, not body-mention.
    - ``relaxed`` — skip jurisdiction filter; OR across domain/process
                     body tags. Use for exploratory / cross-payer
                     questions ("What does the literature say about
                     prior authorization?", "How do payers handle X?").
    - ``none``   — no tag filter at all. Use for code lookups (FL.UM.51,
                     H0019), brand names, or when filters would hurt
                     recall.

    **filters** narrow the candidate pool to a specific payer/state/
    program/authority_level before ranking. Use sparingly — explicit
    filters override automatic tag inference.

    **include_document_ids** pins the search to specific document UUIDs
    (used by the chat instant-rag path for uploaded documents).

    Response includes ``telemetry`` with per-arm hit counts, timing,
    and (when applicable) the actual ``tag_mode_used`` after auto-fallback.
    """
    if not (body.query or "").strip():
        raise HTTPException(status_code=400, detail="query must be non-empty")

    try:
        return await corpus_search(
            db, body,
            caller=(x_caller or "api").strip()[:64] or "api",
            caller_id=(x_caller_id or None),
        )
    except Exception as exc:
        logger.error("skill_corpus_search: unhandled error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/corpus_search_agent", response_model=CorpusSearchAgentResponse)
async def skill_corpus_search_agent(
    body: CorpusSearchAgentRequest,
    db: AsyncSession = Depends(get_db),
    x_caller: str | None = Header(default=None, alias="X-Caller"),
    x_caller_id: str | None = Header(default=None, alias="X-Caller-Id"),
) -> CorpusSearchAgentResponse:
    """RAG-as-agent — single endpoint that internally orchestrates retrieval.

    The chat planner calls THIS instead of choosing between
    ``corpus_search`` / ``precision_search`` / ``explore_search``. The
    agent runs a deterministic mini-loop:

      1. Classify the query (regex literal patterns + J/P/D lexicon match)
      2. Rewrite per strategy (hybrid / phrase_strict / vector_broad)
      3. Run strategies in adaptive order based on QueryProfile
      4. Evaluate each strategy against ITS OWN success criterion
      5. Return best chunks + confidence + improvement_hint

    The planner consumes ``confidence`` and ``improvement_hint.suggestion``
    instead of reasoning about which retrieval arm to use. Determinism:
    same query in → same chunks out (within one corpus revision).

    Response shape adds:
      * ``confidence``         — "high" | "medium" | "low"
      * ``query_profile``      — what the classifier learned
      * ``strategies_tried``   — per-strategy outcome history (auditable)
      * ``improvement_hint``   — reframing suggestion when confidence < high

    See ``app.services.corpus_search_agent`` for the design rationale.

    Note (Phase 1): no external escalation yet. When all corpus
    strategies fail, the response includes ``confidence: low`` and a
    hint suggesting the user reframe or look outside the corpus. Phase
    2 will fold ``lookup_authoritative_sources`` / ``google_search`` /
    ``web_scrape`` into the agent's internal loop.
    """
    if not (body.query or "").strip():
        raise HTTPException(status_code=400, detail="query must be non-empty")

    try:
        return await corpus_search_agent(
            db, body,
            caller=(x_caller or "api").strip()[:64] or "api",
            caller_id=(x_caller_id or None),
        )
    except Exception as exc:
        logger.error(
            "skill_corpus_search_agent: unhandled error: %s", exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
