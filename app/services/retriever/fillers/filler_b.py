"""Filler b — vector-search ranking strategy (Step 3b of the answer engine).

Pure Python strategy: consumes Pool's candidates (pre-scored by Pool's vector
similarity logic), reranks with the legacy multi-signal composite (see
below), assigns top-N per slot based on capacity.

Constraint: read-only, zero DB/embed calls (gate b, single-pool principle).

REPLICATED FROM LEGACY (`app/services/corpus_search.py::_rerank()`,
reranker v1.2/v1.3), reused directly (`_authority_score`, `_length_score`,
`_classify_jpd`, `_jpd_signal` imported, not reimplemented — same "heavy
reuse" convention Pool's own PublicSourceAdapter follows):

  score = (W_SIM*sim + W_AUTH*auth + W_LEN*length + W_JPD*jpd) / MAX_WEIGHT

  - sim    (0.25): PoolCandidate.score itself. Legacy's own `_best_arm_sim`
    only rescales cosine similarity in RRF/hybrid mode (`arm_scores` set);
    Filler b is single-arm ("similarity already holds the raw arm score" —
    legacy's own bypass branch), so raw `.score` is the CORRECT legacy
    behavior here already, not an approximation of it.
  - authority (0.10): `_authority_score(candidate.authority_level)`.
    PoolCandidate.authority_level is now real (Pool's public_adapter.py
    selects `document_authority_level`, verified live against
    information_schema — the column exists, the SELECT list previously
    had a typo referencing bare `authority_level`, fixed 2026-07-23 while
    building this; broke every Pool call, tag_select/vector/inherited
    alike, until fixed). Uses legacy's REAL `_authority_score`/
    `_AUTHORITY_WEIGHTS` (5-key canonical enum, verified against
    `metadata_canonical.py::_AUTHORITY_LEVEL_CANONICAL`, the actual
    single source of truth) — NOT Filler a's own `_AUTHORITY_WEIGHTS`
    reimplementation in filler_a.py, which uses different key strings
    ("operational"/"fyi") than the real canonical values
    ("operational_suggested"/"fyi_not_citable") and is missing
    "payer_website"/"payer_policy" entirely — those real values will
    silently fall through to Filler a's 0.0 default. Flagged upstream,
    not fixed here (not this file's scope).
  - length (0.05): `_length_score(candidate.text)` — real, computed.
  - jpd    (0.20): `_classify_jpd(query)` once, `_jpd_signal(query_cats,
    candidate.text)` per candidate — pure keyword/regex text matching, no
    DB or embed call, real, computed.
  - Per-category decay floor (legacy: drop chunks scoring < 0.6x the best
    score in their `{arm}_{source_type}` category): replicated. Category
    collapses to `source_type` alone here since every candidate already
    shares arm="vector".

STOPGAP MITIGATIONS (2026-07-23, added after Retriever reproduced a
100%-junk result set live against the Sunshine Health timely-filing query —
the composite above still lost to junk because sim's 0.25 weight beats
length's 0.05-weighted penalty by too much margin on real data). These are
explicitly interim defense-in-depth, NOT a fix for corpus quality — a
60-char junk chunk would sail through both untouched. The real fix is the
min-content-length gate at ingestion (already flagged to Curation/
Maintaining). Both use only data already on PoolCandidate — no new seam:

  - Hard length floor (`_MIN_CHUNK_LENGTH = 50`, same threshold
    `_length_score` already uses as its own zero-point): candidates below
    it are dropped BEFORE ranking, not just soft-penalized inside the
    composite. This is what actually stops the reproduced case — a soft
    0.05-weighted penalty was provably insufficient; an outright filter is
    not weight-tunable around.
  - Exact-text dedup: candidates sharing byte-identical `.text` collapse to
    one representative (their best rerank_score) before slot assignment.
    Verified directly against the DB that `content_sha` does NOT catch this
    — 15 rows with identical text all had distinct content_sha values (it's
    evidently salted per-document, not a pure content hash) — so this dedup
    is on raw `.text`, not `content_sha`. Prevents the "same boilerplate
    line, 8 different documents" pattern from consuming multiple slot
    positions even when each instance individually clears the length floor.

NOT REPLICATED — genuinely blocked, not skipped for convenience:
  - coverage (0.55 — the LARGEST weight in the legacy formula):
    `required_phrases`/`required_phrase_weights`/`required_phrase_tag_codes`
    come from Gate's lexicon expansion + DB-computed selectivity
    (`selectivity_for_tag`). Filler b has zero access to GateResult under
    its current contract (PoolResult + AnswerShapeResult + RoutingLadder
    only) and cannot call the DB itself (gate b). Setting W_COV=0 is
    legacy's OWN documented fallback for `has_tag_cov=False` (see
    `_rerank()`'s own branch), not a new approximation. Wiring this for
    real needs a new seam — either Pool attaches a per-candidate coverage
    score at build time, or Fillers' input contract grows a GateResult
    passthrough. Flagged upstream, not solved here.
  - chunk_dtag_boost multiplier: same blocker as coverage (needs query
    d-tag codes from Gate). Never triggers without `phrase_tag_codes`, so
    intentionally not implemented (dead code otherwise).
  - Neighbor score dampening (legacy: 0.5x a neighbor's seed's score):
    neighbors carry `source_arm=""` and are excluded by Filler b's
    source_arm=="vector" filter before reranking ever runs — same as
    before this change. No seed-to-neighbor back-reference exists on
    PoolCandidate to dampen against even if neighbors were let through;
    that's a contract gap for Pool to close, not Filler b's to invent.

See docs/rag-agents/fillers-schematic-spec.md and
docs/rag-agents/filler-b-vector-kickoff.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.services.corpus_search import (
    _authority_score,
    _classify_jpd,
    _jpd_signal,
    _length_score,
)
from app.services.retriever.pool.contracts import PoolCandidate, PoolResult
from app.services.retriever.shape.slots import AnswerShapeResult
from app.services.retriever.fillers.contracts import (
    FilledChunk,
    FilledSlot,
    FilledShape,
)

logger = logging.getLogger(__name__)

# Weights match legacy reranker v1.2/v1.3 (corpus_search.py::_rerank())
# exactly, so the two systems stay directly comparable signal-for-signal.
_W_SIM = 0.25
_W_AUTH = 0.10
_W_LEN = 0.05
_W_JPD = 0.20  # zeroed per-query below when the query has no JPD category hits

# Per-category decay floor, same threshold as legacy: drop candidates
# scoring below 0.6x the best score in their category.
_DECAY_FLOOR_RATIO = 0.6

# Stopgap hard length floor (see module docstring) -- same threshold
# _length_score already treats as its zero-point, but enforced as an
# outright filter here, not a soft weighted penalty.
_MIN_CHUNK_LENGTH = 50


@dataclass
class RoutingLadder:
    """Strategy sequence per slot (from Router/two-phase design)."""

    slot_id: str
    strategy_sequence: list[str]  # e.g., ["a", "b", "c"]


def _rerank_vector_candidates(
    candidates: list[PoolCandidate], query: str
) -> tuple[list[tuple[PoolCandidate, float]], dict]:
    """Composite-rerank vector-arm candidates, legacy formula (see module
    docstring for exactly which signals are real vs. structurally-inert
    pending upstream data), plus the two stopgap mitigations (hard length
    floor, exact-text dedup). Returns (survivors sorted descending, stats).
    """
    query_cats = _classify_jpd(query) if query else {}
    has_jpd = bool(query_cats)
    w_jpd = _W_JPD if has_jpd else 0.0
    max_weight = _W_SIM + _W_AUTH + _W_LEN + w_jpd

    # Stopgap 1: hard length floor -- drop before ranking, not a soft penalty.
    length_filtered = [c for c in candidates if len((c.text or "").strip()) < _MIN_CHUNK_LENGTH]
    candidates = [c for c in candidates if len((c.text or "").strip()) >= _MIN_CHUNK_LENGTH]

    scored: list[tuple[PoolCandidate, float]] = []
    for c in candidates:
        sim = c.score or 0.0
        auth = _authority_score(c.authority_level)
        length = _length_score(c.text)
        jpd, _jpd_tags = _jpd_signal(query_cats, c.text) if has_jpd else (0.0, [])

        raw = (_W_SIM * sim) + (_W_AUTH * auth) + (_W_LEN * length) + (w_jpd * jpd)
        rerank_score = raw / max_weight if max_weight > 0 else raw
        scored.append((c, rerank_score))

    # Stopgap 2: exact-text dedup -- collapse byte-identical text to its
    # best-scoring instance. content_sha does NOT catch this (verified live:
    # distinct per-document, not a pure content hash), so dedup on raw text.
    best_by_text: dict[str, tuple[PoolCandidate, float]] = {}
    dedup_collapsed = 0
    for c, score in scored:
        key = (c.text or "").strip()
        existing = best_by_text.get(key)
        if existing is None:
            best_by_text[key] = (c, score)
        else:
            dedup_collapsed += 1
            if score > existing[1]:
                best_by_text[key] = (c, score)
    deduped = list(best_by_text.values())

    # Per-category decay floor. Category = source_type alone (arm is
    # constant "vector" for every candidate reaching this function).
    cat_best: dict[str, float] = {}
    for c, score in deduped:
        cat = c.source_type or "unknown"
        cat_best[cat] = max(cat_best.get(cat, 0.0), score)

    survivors = []
    decay_dropped = 0
    for c, score in deduped:
        cat = c.source_type or "unknown"
        best = cat_best.get(cat, 0.0)
        if best > 0 and score < _DECAY_FLOOR_RATIO * best:
            decay_dropped += 1
            continue
        survivors.append((c, score))

    survivors.sort(key=lambda pair: pair[1], reverse=True)
    stats = {
        "length_filtered": len(length_filtered),
        "dedup_collapsed": dedup_collapsed,
        "decay_dropped": decay_dropped,
    }
    return survivors, stats


def fill_shape_vector(
    pool_result: PoolResult,
    shape_result: AnswerShapeResult,
    routing_ladders: list[RoutingLadder] | None = None,
) -> FilledShape:
    """
    Fill pre-built slots by reranking Pool's vector-arm candidates with the
    legacy multi-signal composite (sim/authority/length/jpd + decay floor —
    see module docstring for what's real vs. structurally-inert here) and
    assigning top-N per slot.

    Algorithm (Phase 2 of fillers spec):
    1. Filter to candidates with source_arm == "vector" (excludes tag_select's
       coverage-count score, inherited's None score, and neighbors).
    2. Rerank via `_rerank_vector_candidates` (legacy composite + decay floor).
    3. For each slot in priority order:
       - Filter candidates if needed (by slot semantics / source_type).
       - Take top-N where N = slot.capacity (non-overlapping).
       - Assign to slot, remove from pool.
    4. Track occupancy, under_fill, over_fill.

    Args:
        pool_result: Output from Pool (Step 2).
        shape_result: Output from Shape (Step 1).
        routing_ladders: Optional per-slot strategy sequences (unused v1).

    Returns:
        FilledShape with all slots assigned.
    """

    # Filter to vector-arm candidates with a real similarity score first --
    # neighbors (source_arm="") and non-vector arms are excluded, same as
    # before. Their .score field means something else entirely (coverage
    # count, None) and reranking them would be meaningless.
    vector_candidates = [
        c for c in pool_result.candidates
        if c.source_arm == "vector" and c.score is not None
    ]
    reranked, rerank_stats = _rerank_vector_candidates(vector_candidates, pool_result.query)
    rerank_score_by_id = {c.chunk_id: score for c, score in reranked}
    scored_candidates = [c for c, _score in reranked]

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
                    original_score=rerank_score_by_id[candidate.chunk_id],
                    assignment_reason="vector_rerank",
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
        "fillers_decision": "vector_rerank_assign",
        "slots_filled": len([s for s in filled_slots if s.occupancy > 0]),
        "empty_slots": len([s for s in filled_slots if s.occupancy == 0]),
        "under_filled": len([s for s in filled_slots if s.under_filled]),
        "total_chunks_assigned": total_assigned,
        "length_filtered": rerank_stats["length_filtered"],
        "dedup_collapsed": rerank_stats["dedup_collapsed"],
        "decay_dropped": rerank_stats["decay_dropped"],
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
        filling_strategy="vector_rerank",
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
