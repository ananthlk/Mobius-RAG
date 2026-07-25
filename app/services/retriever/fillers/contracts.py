"""Dataclass contracts for the fillers module (Step 3 of the answer engine).

FilledShape is the output of filling one slot assignment phase --
slots pre-built by Shape, filled with candidates from Pool, ranked by filler strategy.
See docs/rag-agents/fillers-schematic-spec.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Filler c's own assignment_reason values for LLM-citation chunks (see
# filler_c.py's _CHUNK_SHAPE_BY_STATUS) -- the single source of truth.
# observer.py and synthesis.py both key logic off these; previously each
# hand-copied the "llm_partial_match" string independently (real drift
# risk, caught by Tech Review 2026-07-24) -- now imported downward instead.
ASSIGNMENT_REASON_LLM_RETRIEVED = "llm_retrieved"
ASSIGNMENT_REASON_LLM_RETRIEVED_EXTERNAL = "llm_retrieved_external"
ASSIGNMENT_REASON_LLM_PARTIAL_MATCH = "llm_partial_match"


@dataclass
class FilledChunk:
    """One chunk assigned to a slot.

    For internal (Pool-sourced) chunks: document_id is populated, url is None.
    For external chunks (Web Search, LLM Retrieval): url is populated, document_id is None.
    Chat integrates url directly into SourceRef without renaming.
    """

    chunk_id: str
    document_id: str | None = None  # internal ref (Pool-sourced), None for external
    text: str = ""
    url: str | None = None  # external ref (Web Search, LLM Retrieval, etc.), None for internal
    title: str | None = None  # display title (DuckDuckGo passage titles, best-effort)
    source_type: str | None = None  # "internal" | "external" | None
    document_status: str | None = None
    # Real authority signal, carried through from Pool (2026-07-24, Retriever
    # + Ananth: the grounding badge was inferring authority from source_type,
    # a chunking-STRATEGY label with nothing to do with authority). This is
    # the DB's `document_authority_level` — contract_source_of_truth /
    # payer_policy / payer_manual / fee_schedule / operational_suggested /
    # fyi_not_citable / '' — which Pool already reads into
    # PoolCandidate.authority_level (public_adapter.py) and filler_a even
    # uses for scoring, but which was DROPPED here at the FilledChunk
    # boundary. Populated by the Pool-sourced fillers (a/b) from
    # candidate.authority_level; None for external/fact-store chunks (c/d/s),
    # which _infer_authority classifies by source_type as before. Additive/
    # optional, same contract-change class as quote_verified/title (DB-
    # stewarded). Synthesis's _infer_authority (Synthesizer) consumes this as
    # the precise signal when present. Gates adjudicator mode (b) — see
    # adjudicator-calibration-spec.md §8.
    authority_level: str | None = None
    content_sha: str | None = None
    page_number: int | None = None  # location within source (from Pool or fact store)
    paragraph_index: int | None = None  # paragraph granularity (from Pool)
    quote_verified: bool | None = None  # True=LLM quote matched, False=hallucination, None=no quote
    tags: dict = field(default_factory=dict)
    is_neighbor: bool = False
    original_score: float | None = None
    assignment_reason: str = ""  # e.g., "score_rank", "semantic_match", "fallback"


@dataclass
class FilledSlot:
    """One slot after chunk assignment."""

    slot_id: str
    slot_semantics: str  # "direct_answer" | "thematic_exploration" | "external_context"
    capacity: int
    required: bool  # True=required slot, False=optional/fallback (from AnswerSlot.required)
    chunks: list[FilledChunk] = field(default_factory=list)
    occupancy: int = 0
    under_filled: bool = False
    over_filled: bool = False


@dataclass
class FilledShape:
    """Output of the filling phase -- all slots assigned with chunks."""

    slots: list[FilledSlot] = field(default_factory=list)
    total_chunks_assigned: int = 0
    filling_strategy: str = ""  # e.g., "bm25" for filler a
    emit: dict = field(default_factory=dict)
