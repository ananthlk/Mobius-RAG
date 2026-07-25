"""Dataclass contracts for the pool module (Step 2 of the answer engine).

PoolResult is the output of one pool build (pool.run_pool_for_query()) --
the unioned, deduped candidate set every downstream filler reads PURELY
(no DB handle, no re-embed, no re-scan -- gate (b), the whole reason Pool
exists). See docs/rag-agents/pool-schematic-spec.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ScopeContext:
    """Which federated source an adapter is allowed to draw from.

    "federate, never leak" (retriever-meet-old-plan.md P0) -- scope rides
    the adapter, baked in at construction; Pool's orchestration core never
    branches on it directly. public=global is the only scope built today
    (pool-schematic-spec.md S1/S3.0) -- org=tenant, instant_rag=user+PHI
    are placeholders for when those adapters land, not consumed yet.
    """

    kind: str  # "public" | "org" | "instant_rag" | "cache"
    tenant_id: str | None = None
    user_id: str | None = None


@dataclass
class PoolCandidate:
    """One chunk in the pool -- a match (from a strategy) or a neighbor.

    source_type/document_status are passthrough fields required by
    Product-Awareness's reality-gating (target-structure-spec.md S10) so
    "planned" vs "live" survives pool -> shape -> synthesis -> contract.
    page_number/paragraph_index are needed by neighbor assembly
    (_fetch_sibling_chunks_batch's windowing), not just display.
    """

    chunk_id: str
    document_id: str
    text: str
    is_neighbor: bool
    source_arm: str  # "tag_select" | "vector" | "inherited" | "" for neighbors
    score: float | None
    tags: dict = field(default_factory=dict)
    document_status: str | None = None
    source_type: str | None = None
    content_sha: str | None = None
    page_number: int | None = None
    paragraph_index: int | None = None
    # Additive ranking signal for Filler a (BM25), added 2026-07-23 -- Fillers
    # can't make DB calls (gate b), so Pool supplies this instead of Filler a
    # re-querying. ts_rank_cd(search_vec, plainto_tsquery(...), 32) against
    # every match candidate regardless of which arm surfaced it -- Filler a
    # ranks across the whole pool, not just one arm. None for neighbors
    # (Retriever's explicit call): they weren't retrieved by any term match at
    # all, so a BM25 score would measure something incidental to why they're
    # in the pool -- same "None means no signal" convention `score` already
    # established for neighbors.
    bm25_score: float | None = None
    # Authority level for reranking (Filler a reranking signal). Values:
    # "contract_source_of_truth" (1.0), "operational" (0.5), "fyi" (0.2), None (0.0)
    authority_level: str | None = None


@dataclass
class PoolResult:
    """Everything Pool produced for one rewritten_query (Step 2 output)."""

    query: str = ""
    candidates: list[PoolCandidate] = field(default_factory=list)
    # doc_narrow_ms (S3.1 step 2 cascade), tag_select_ms (step 4 chunk query),
    # embed_ms, vector_ms, inherited_ms, dedup_ms, neighbor_ms -- every
    # DB/retrieval segment split per TECH's clarification 2026-07-23, gate (d).
    segment_ms: dict = field(default_factory=dict)
    strategy_hint: str = ""  # which arm(s) actually contributed
    fallback_triggered: bool = False
    pool_ms: int = 0
    # Added 2026-07-23 (Filler s / Payor Platform request, Retriever-relayed
    # and independently verified): Pool's vector arm already computes this
    # via gemini-embedding-001 @ output_dimensionality=1536 -- confirmed
    # live to match the Payor Fact Store's own vector(1536) schema exactly
    # (mobius-payor/migrations/006_payor_fact_store.sql,
    # mobius-payor/app/fact_embed.py). Reuse saves Filler s a redundant
    # embed call. None when the vector arm didn't run for this query (e.g.
    # no-retrieval postures) or the embed call itself failed -- Filler s
    # falls back to its own bounded call in that case, always correct
    # either way, this is purely an optimization.
    query_embedding: list[float] | None = None


class SourceAdapter(ABC):
    """Generic seam every federated source implements (pool-schematic-spec.md S3.0).

    Pool's orchestration core (union/dedup/neighbor-assembly) calls these
    four methods generically -- it never branches on which adapter it's
    talking to. public = adapter #1, the only one built today; org/
    instant-rag land later behind this exact interface, no touching
    Pool's core (Ananth's explicit plug-and-play requirement, 2026-07-23).

    ``inherited()`` is deliberately optional in EFFECT, not signature --
    AHCA inheritance is a public-corpus-specific concept; adapters with no
    equivalent return [] without violating the interface.
    """

    scope: ScopeContext

    @abstractmethod
    async def tag_select(
        self, query: str, d_codes: list[str], j_codes: list[str], p_codes: list[str], width: int
    ) -> tuple[list[PoolCandidate], dict]:
        """Takes RAW matched tag codes (unclassified) -- required/boosted/drop
        bucketing happens INSIDE the adapter, not the orchestrator, because
        tag selectivity is corpus-specific (computed against each adapter's
        own backing tag table, e.g. public's document_tags). Same ownership
        line as "each adapter owns bm25+vector+inheritance+neighbors for
        its section" (retriever-meet-old-plan.md). ``query`` is needed only
        for the additive `bm25_score` field (Filler a, 2026-07-23), not for
        the tag-coverage selection logic itself.

        Returns (candidates, segment_ms) -- segment_ms carries doc_narrow_ms
        + tag_select_ms as two distinct buckets (TECH's 2026-07-23 split)."""
        ...

    @abstractmethod
    async def vector_search(
        self, query: str, width: int
    ) -> tuple[list[PoolCandidate], dict, list[float] | None]:
        """Returns (candidates, segment_ms, query_embedding) -- segment_ms
        carries embed_ms + vector_ms. query_embedding is the raw embedding
        computed for this query (None if the embed call itself failed),
        surfaced so Pool can populate PoolResult.query_embedding for reuse
        by downstream Fillers with matching embedding-space needs."""
        ...

    @abstractmethod
    async def inherited(self, query: str, payor_codes: list[str], width: int) -> tuple[list[PoolCandidate], dict]:
        """Returns (candidates, segment_ms) -- segment_ms carries inherited_ms.
        No-ops ([], {}) for adapters/queries with no inheritance concept.
        ``query`` is needed only for the additive `bm25_score` field."""
        ...

    @abstractmethod
    async def neighbors(self, candidates: list[PoolCandidate]) -> tuple[list[PoolCandidate], dict]:
        """Returns (matches + neighbors, segment_ms) -- segment_ms carries neighbor_ms."""
        ...
