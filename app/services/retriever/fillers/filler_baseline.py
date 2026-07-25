"""Filler baseline — Uniform top-N ranking (no slot semantics).

Used for before/after calibration comparison against Filler a (BM25).
This is the control: just take top-N by bm25_score regardless of slot semantics.

See docs/rag-agents/filler-a-calibration-plan.md.
"""

from __future__ import annotations

from app.services.retriever.pool.contracts import PoolCandidate, PoolResult
from app.services.retriever.shape.slots import AnswerShapeResult
from app.services.retriever.fillers.contracts import (
    FilledChunk,
    FilledSlot,
    FilledShape,
)


def fill_shape_uniform_topn(
    pool_result: PoolResult,
    shape_result: AnswerShapeResult,
) -> FilledShape:
    """
    Baseline filler: uniform top-N by bm25_score, NO slot-semantic filtering.

    This is the control for calibration: pure score-rank without any semantic logic.
    Allows isolating whether semantic filtering actually improves over simple ranking.

    Algorithm:
    1. Sort candidates by bm25_score descending.
    2. For each slot in order:
       - Take top-N (capacity) from remaining pool.
       - NO filtering by slot semantics, source_type, or posture.
       - Remove assigned candidates from remaining.
    3. Track occupancy, under_fill.

    Args:
        pool_result: Output from Pool (Step 2).
        shape_result: Output from Shape (Step 1).

    Returns:
        FilledShape with slots assigned uniformly by score rank.
    """

    # Sort candidates by BM25 score descending.
    scored_candidates = [c for c in pool_result.candidates if c.bm25_score is not None]
    scored_candidates.sort(key=lambda c: c.bm25_score, reverse=True)

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

        # Take top-N, no semantic filtering (control).
        assigned_chunks = []
        chunk_ids_assigned = set()
        for candidate in remaining_candidates[: slot.capacity]:
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

        # Remove assigned candidates from remaining pool.
        remaining_candidates = [
            c for c in remaining_candidates if c.chunk_id not in chunk_ids_assigned
        ]

        filled_slot.chunks = assigned_chunks
        filled_slot.occupancy = len(assigned_chunks)
        filled_slot.under_filled = filled_slot.occupancy < slot.capacity
        filled_slot.over_filled = False

        filled_slots.append(filled_slot)
        total_assigned += filled_slot.occupancy

    # Diagnostic emit.
    emit = {
        "fillers_decision": "uniform_topn_baseline",
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
        filling_strategy="uniform_topn",
        emit=emit,
    )
