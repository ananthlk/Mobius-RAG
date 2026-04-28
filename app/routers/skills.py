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

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.corpus_search import (
    CorpusSearchRequest,
    CorpusSearchResponse,
    corpus_search,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/skills/v1", tags=["skills"])


@router.post("/corpus_search", response_model=CorpusSearchResponse)
async def skill_corpus_search(
    body: CorpusSearchRequest,
    db: AsyncSession = Depends(get_db),
) -> CorpusSearchResponse:
    """Run BM25 / semantic / hybrid corpus search and return ranked chunks.

    **mode** options:
    - ``corpus``    — hybrid BM25 ⊕ pgvector, RRF-fused, reranked (default)
    - ``precision`` — BM25-only; best for exact codes / HCPCS lookups
    - ``recall``    — pgvector-only; best for "what do we know about X"

    **filters** narrow the candidate pool to a specific payer/state/program/
    authority_level before ranking.

    **include_document_ids** pins the search to specific document UUIDs
    (used by the chat instant-rag path for uploaded documents).

    Response includes ``telemetry`` with per-arm hit counts and timing
    for diagnostics.
    """
    if not (body.query or "").strip():
        raise HTTPException(status_code=400, detail="query must be non-empty")

    try:
        return await corpus_search(db, body)
    except Exception as exc:
        logger.error("skill_corpus_search: unhandled error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
