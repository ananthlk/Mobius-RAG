"""Filler a — BM25 ranking strategy (Step 3a of the answer engine).

Pure Python strategy: consumes Pool's candidates (pre-scored by Pool's BM25 logic),
ranks by BM25 score, assigns top-N to each slot based on capacity.

Constraint: read-only, zero DB/embed calls (gate b, single-pool principle).
Pool provides PoolCandidate.bm25_score (ts_rank_cd via plainto_tsquery);
Filler a ranks by that field and assigns non-overlapping chunks to slots.

CRITICAL: Read PoolCandidate.bm25_score, NOT PoolCandidate.score.
PoolCandidate.score holds different signals per source_arm (vector similarity, tag-coverage, etc.).
PoolCandidate.bm25_score is the BM25-specific ranking signal for Filler a.

See docs/rag-agents/fillers-schematic-spec.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.services.retriever.pool.contracts import PoolCandidate, PoolResult
from app.services.retriever.shape.slots import AnswerShapeResult
from app.services.retriever.fillers.contracts import (
    FilledChunk,
    FilledSlot,
    FilledShape,
)

logger = logging.getLogger(__name__)

# Authority level weights for reranking signal composition.
# Imported from corpus_search.py (canonical source of truth) to ensure alignment.
# Values are converted to lowercase before lookup (canonical pattern).
_AUTHORITY_WEIGHTS = {
    "contract_source_of_truth": 1.0,   # provider/member/billing manuals, UM/auth policies
    "payer_website": 0.75,             # docs sourced directly from payor's website
    "operational_suggested": 0.65,     # operationally useful, suggested reading
    "payer_policy": 0.50,              # published policy docs — citable source
    "fyi_not_citable": 0.20,           # informational, not authoritative
}
_AUTHORITY_DEFAULT = 0.10  # Untagged docs get small default weight


def _compute_authority_score(authority_level: str | None) -> float:
    """Map authority_level to [0, 1] score, matching corpus_search.py's canonical weighting.

    Converts to lowercase before lookup to match database storage convention.
    Default 0.10 for None (untagged docs score above zero but below any tagged value).
    """
    if not authority_level:
        return _AUTHORITY_DEFAULT
    return _AUTHORITY_WEIGHTS.get((authority_level or "").strip().lower(), _AUTHORITY_DEFAULT)


def _compute_tag_coverage_score(tags: dict) -> float:
    """Compute tag coverage quality [0, 1]. More tags = more relevant."""
    if not tags:
        return 0.0
    # Simple heuristic: normalize tag count (cap at 10 for saturation)
    tag_count = len(tags)
    return min(1.0, tag_count / 10.0)


def _compute_length_score(text: str) -> float:
    """Compute text quality [0, 1] based on length. Prefer 100-500 chars."""
    if not text:
        return 0.0
    length = len(text)
    # Penalize stubs < 50 chars, prefer 100-500, plateau after 500
    if length < 50:
        return length / 50.0 * 0.5  # [0, 0.5)
    elif length < 100:
        return 0.5 + (length - 50) / 50.0 * 0.25  # [0.5, 0.75)
    elif length <= 500:
        return 0.75 + (length - 100) / 400.0 * 0.25  # [0.75, 1.0]
    else:
        return 1.0


def _compute_rerank_score(candidate: PoolCandidate, query: str = "") -> float:
    """
    Compose multiple signals into a unified rerank score [0, 1].

    Weights (Filler a v0.2 — Retriever's reranking principle):
      bm25 (0.65) + authority (0.10) + tag_coverage (0.10) + length (0.10) + misc (0.05)

    BM25 is primary (0.65 weight) because Pool pre-scores candidates by arm.
    Secondary signals (authority, coverage, length) provide tie-breaking and
    quality lift when BM25 scores are close, but don't override BM25 when it
    diverges significantly.

    CRITICAL: This is Filler a's PRIMARY signal composition. Pool provides
    candidates pre-scored by arm (bm25_score, vector score, etc.) but does
    NOT do multi-signal fusion across all signals (gate b, read-only). Each
    Filler composes its own signal set for its own arm.
    """
    bm25_sig = candidate.bm25_score or 0.0  # Already [0, 1]
    auth_sig = _compute_authority_score(candidate.authority_level)
    cov_sig = _compute_tag_coverage_score(candidate.tags)
    len_sig = _compute_length_score(candidate.text)

    # Weighted sum (already normalized [0, 1])
    composite = (
        0.65 * bm25_sig +
        0.10 * auth_sig +
        0.10 * cov_sig +
        0.10 * len_sig +
        0.05 * 0.5  # Misc (neutral baseline)
    )
    return composite


@dataclass
class RoutingLadder:
    """Strategy sequence per slot (from Router/two-phase design)."""

    slot_id: str
    strategy_sequence: list[str]  # e.g., ["a", "b", "c"]


def fill_shape_bm25(
    pool_result: PoolResult,
    shape_result: AnswerShapeResult,
    routing_ladders: list[RoutingLadder] | None = None,
) -> FilledShape:
    """
    Fill pre-built slots by ranking Pool's candidates via BM25 score.

    Algorithm (Phase 2 of fillers spec):
    1. Sort candidates by PoolCandidate.bm25_score (Pool's ts_rank_cd via plainto_tsquery) descending.
    2. For each slot in priority order:
       - Filter candidates if needed (by slot semantics / source_type).
       - Take top-N where N = slot.capacity (non-overlapping).
       - Assign to slot, remove from pool.
    3. Track occupancy, under_fill, over_fill.

    BM25 scoring details:
    - Pool computes PoolCandidate.bm25_score: ts_rank_cd(search_vec, plainto_tsquery('english', :query), 32)
      where plainto_tsquery handles punctuation safely (production-proven in corpus_search).
    - Score range: [0, 1] due to flag 32 (log normalization).
    - Neighbors get bm25_score=None (positional adjacency, no search-term match).
    - CRITICAL: Read PoolCandidate.bm25_score, NOT PoolCandidate.score (which holds arm-specific signals).

    Args:
        pool_result: Output from Pool (Step 2).
        shape_result: Output from Shape (Step 1).
        routing_ladders: Optional per-slot strategy sequences (unused v1).

    Returns:
        FilledShape with all slots assigned.
    """

    # Compose multi-signal rerank score and sort descending.
    # Primary: bm25_score (Pool's BM25 ranking). Secondary: authority, tag_coverage, length.
    # Neighbors (positional adjacency only) get bm25_score=None and are filtered out.
    scored_candidates = [c for c in pool_result.candidates if c.bm25_score is not None]
    # Compute composite rerank score per candidate
    scored_candidates = [
        (c, _compute_rerank_score(c, pool_result.query))
        for c in scored_candidates
    ]
    # Sort by composite score descending
    scored_candidates.sort(key=lambda pair: pair[1], reverse=True)
    # Unwrap candidates (keep only PoolCandidate, discard score tuple)
    scored_candidates = [c for c, _ in scored_candidates]

    filled_slots: list[FilledSlot] = []
    total_assigned = 0
    remaining_candidates = scored_candidates.copy()

    for slot in shape_result.slots:
        filled_slot = FilledSlot(
            slot_id=slot.slot_id,
            slot_semantics=slot.slot_semantics,
            capacity=slot.capacity,
            required=slot.required,
        )

        # Filter candidates by slot semantics.
        candidates_for_slot = _filter_by_slot_semantics(
            remaining_candidates, slot
        )

        # Take top-N (capacity), remove from remaining pool.
        assigned_chunks = []
        chunk_ids_assigned = set()
        for candidate in candidates_for_slot[: slot.capacity]:
            assigned_chunks.append(
                FilledChunk(
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    text=candidate.text,
                    document_status=candidate.document_status,
                    content_sha=candidate.content_sha,
                    source_type=candidate.source_type,
                    tags=candidate.tags or {},
                    is_neighbor=candidate.is_neighbor,
                    original_score=candidate.bm25_score,
                    assignment_reason="score_rank",
                )
            )
            chunk_ids_assigned.add(candidate.chunk_id)

        # Remove assigned candidates from remaining pool (non-overlapping).
        remaining_candidates = [
            c for c in remaining_candidates if c.chunk_id not in chunk_ids_assigned
        ]

        filled_slot.chunks = assigned_chunks
        filled_slot.occupancy = len(assigned_chunks)
        filled_slot.under_filled = filled_slot.occupancy < slot.capacity
        filled_slot.over_filled = False  # N/A for simple top-N

        filled_slots.append(filled_slot)
        total_assigned += filled_slot.occupancy

    # Diagnostic emit.
    emit = {
        "fillers_decision": "bm25_rank_assign",
        "slots_filled": len([s for s in filled_slots if s.occupancy > 0]),
        "empty_slots": len([s for s in filled_slots if s.occupancy == 0]),
        "under_filled": len([s for s in filled_slots if s.under_filled]),
        "total_chunks_assigned": total_assigned,
        "per_slot_details": [
            {
                "slot_id": s.slot_id,
                "slot_semantics": s.slot_semantics,
                "occupancy": s.occupancy,
                "capacity": s.capacity,
            }
            for s in filled_slots
        ],
    }

    return FilledShape(
        slots=filled_slots,
        total_chunks_assigned=total_assigned,
        filling_strategy="bm25",
        emit=emit,
    )


def _filter_by_slot_semantics(
    candidates: list[PoolCandidate],
    slot,  # AnswerSlot from shape_result
) -> list[PoolCandidate]:
    """
    Filter candidates by slot semantics (phase 2, line 88-91 of spec).

    For v1, all filtering is done upstream (Shape/Gate).
    Fillers just consumes the pool as-is and ranks by score.
    Future: slot-semantic-specific filtering (direct_answer vs thematic_exploration).
    """
    # v1: No filtering, return all (scoring handles the ranking).
    # Future: could filter by source_type for external_context slots, etc.
    return candidates
