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
import re
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


def _compute_doc_type_score(doc_type: str | None) -> float:
    """Coarse document-TYPE classification signal [0, 1] (2026-08-15, joint
    build with Curation/Lexicon -- see docs/rag-agents/j-doc-type-spec.md,
    PoolCandidate.doc_type's docstring). Deliberately simple, matching the
    real corpus distribution: Lexicon's population run classified 3,334
    docs (um=3,184, clinical_policy=68, fee_schedule=44, contract=38) --
    ANY real classification, not the specific type, is the useful signal
    here, since 'um' alone covers ~95% of classified docs in this corpus
    and isn't itself query-discriminating (validated: an offline sweep
    that instead matched doc_type against the query's own topic showed no
    improvement over this simpler presence check, and this corpus doesn't
    yet have enough non-'um' volume to calibrate a real per-type weight
    table). Rewards "Lexicon's own classification pipeline reviewed and
    typed this document" over unclassified/generic corpus content (empty
    string '' -- see the docstring on PoolCandidate.doc_type for why '' is
    the real value at this layer, not None) -- same spirit as
    _compute_authority_score's "any real classification beats none."
    """
    return 1.0 if doc_type else 0.0


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


def _compute_meta_boost_score(
    text: str,
    tags: dict,
    required_phrases: list[tuple[str, float]] | None = None,
    boosted_phrases: list[tuple[str, float]] | None = None,
) -> float:
    """Compute meta_boost as selectivity-weighted fraction of Gate's phrases present.

    Matches phrases against chunk text and tags (not doc-level metadata — that's
    a future enhancement). REQUIRED phrases (selectivity ≥0.65) count full weight;
    BOOSTED phrases (0.40-0.65) count half weight. Result is normalized fraction [0, 1].

    Args:
        text: Chunk body text.
        tags: Chunk tags dict (keys are tag codes like "d:claims.timely_filing").
        required_phrases: [(phrase, selectivity_weight), ...] from Gate's REQUIRED bucket.
        boosted_phrases: [(phrase, selectivity_weight), ...] from Gate's BOOSTED bucket.

    Returns:
        Float [0, 1]: fraction of total possible phrase weight present in chunk.
    """
    required_phrases = required_phrases or []
    boosted_phrases = boosted_phrases or []

    if not required_phrases and not boosted_phrases:
        return 0.0

    text_lower = text.lower()
    # Normalize tag keys: convert underscores/dots to spaces so "claims.timely_filing" becomes
    # "claims timely filing" and can match the phrase "timely filing" via substring search.
    tags_str = " ".join(
        re.sub(r'[._]', ' ', str(k).lower()) + " " + str(v).lower()
        for k, v in (tags or {}).items()
    )

    # Compute present weight (REQUIRED full, BOOSTED half)
    present_weight = 0.0
    for phrase, selectivity in required_phrases:
        if phrase.lower() in text_lower or phrase.lower() in tags_str:
            present_weight += selectivity

    for phrase, selectivity in boosted_phrases:
        if phrase.lower() in text_lower or phrase.lower() in tags_str:
            present_weight += selectivity * 0.5  # Boosted phrases worth half

    # Total possible weight
    total_weight = (
        sum(s for _, s in required_phrases) +
        sum(s * 0.5 for _, s in boosted_phrases)
    )

    return min(1.0, present_weight / total_weight) if total_weight > 0 else 0.0


def _compute_rerank_score(
    candidate: PoolCandidate,
    query: str = "",
    required_phrases: list[tuple[str, float]] | None = None,
    boosted_phrases: list[tuple[str, float]] | None = None,
    bm25_bounds: tuple[float, float] | None = None,
) -> float:
    """
    Compose multiple signals into a unified rerank score [0, 1].

    Weights (Filler a v0.7 — 2026-08-15, Ananth's live-trace diagnosis):
      bm25 (0.51, MIN-MAX NORMALIZED WITHIN THE POOL — see below) +
      authority (0.13) + tag_coverage (0.05) + length (0.06) +
      meta_boost (0.18) + misc (0.07) -- same weight split as v0.5, unchanged.

    v0.5 fixed the structural exclusion (see that changelog entry below)
    but left the Daraprim case on a knife-edge: verified live it landed at
    rank #6 of a 6-slot cutoff -- a coin flip against ordinary corpus
    fluctuation elsewhere in the pool (confirmed live: a single new/shifted
    competitor candidate was enough to push it to rank #7, excluded, with
    the code UNCHANGED -- verified byte-exact via live-vs-local rerank_score
    comparison, ruling out a deploy issue). Root cause of the thin margin,
    per Ananth's own diagnosis: LINEAR weighting treats every point of raw
    bm25 gap the same regardless of where it falls in the distribution --
    Daraprim's chunk has the single BEST bm25 in the whole ~1157-candidate
    pool (0.804, a real 0.13 gap over the next tier at ~0.67-0.75), but a
    flat weight doesn't let that decisive a margin count for more than an
    equally-sized gap anywhere else in the range.

    v0.6 (same day) fixed this with a flat power transform (bm25_sig **
    1.5) on the raw [0, 1] value -- validated (rank #5, bank recall held
    at 0.791) but Ananth's own follow-up caught the real limitation:
    ts_rank_cd's raw magnitude isn't comparable across different queries/
    pools -- this pool happened to span nearly the full [0, 1] range
    (0.000-0.846), so a flat exponent worked, but a query where every
    candidate clusters in a narrow band (e.g. 0.4-0.5) would barely move
    under x**1.5 even though the SAME relative gap (0.6 vs 0.4 "should"
    count the same as 0.8 vs 0.6, Ananth's framing) is just as real.
    v0.7 fixes this properly: min-max normalize bm25_score against the
    POOL'S OWN min/max for this query BEFORE any further transform --
    stretches whatever the actual competitive spread is to fill [0, 1],
    so separation is measured relative to the field, not an absolute
    scale. Verified empirically (same live pool snapshot + 22-query eval
    bank) that min-max ALONE (no extra power on top) already matches
    v0.6's result exactly -- rank #5, bank recall 0.791 -- and is the more
    generalizable fix (adapts per-query instead of assuming one constant
    fits every pool's spread). Also confirmed combining min-max WITH an
    extra power transform regresses the bank (same "diminishing/negative
    returns from stacking two corrections" pattern found when v0.6's
    power was combined with a weight nudge) -- shipped min-max alone.
    bm25_bounds=None (e.g. isolated unit tests) falls back to the raw
    [0, 1] value unchanged, byte-identical to pre-v0.6 behavior.

    Separately, real root cause behind WHY this specific chunk needed
    rescuing at all: its lexicon tags don't include
    utilization_management.prior_authorization -- that tag exists at the
    DOCUMENT level (4 hits) but didn't propagate to this specific
    clinical-criterion paragraph. A document-type tag (e.g. a UM_POLICY
    classification) that Router/Fillers could weight directly, propagated
    by Lexicon rather than inferred per-chunk, is the durable fix --
    flagged to Curation/Lexicon separately, not built here.

    v0.5's original changelog: tag_coverage_score is a raw tag-COUNT
    heuristic (unrelated to relevance -- a chunk with more tags scores
    higher regardless of topical precision) and meta_boost_score can only
    ever reward LEXICON-matched phrases (verified live: "daraprim" has zero
    rows in policy_lexicon_entries, so it structurally can't contribute).
    At their ORIGINAL 0.20/0.20 combined weight, these two signals
    outvoted bm25 even when bm25 correctly ranked the answer chunk #1-9 of
    the entire ~1191-candidate pool (0.81 bm25, top authority) -- the
    boilerplate competitors won purely by repeating generic phrases
    ("prior authorization"/"AHCA"/"Florida") and carrying more tags,
    neither of which reflects whether the chunk actually answers the
    query. Confirmed this is a WEIGHTING problem, not a signal-definition
    one: reproduced the exact same loss with both signals fully unmodified,
    only varying weight -- two separate signal-redefinition attempts
    (tag-depth instead of count; a lexicon-plus-specific-term meta_boost
    fallback) each fixed this one case but introduced NEW regressions
    elsewhere in the 22-query eval bank, so reverted in favor of
    reweighting instead (see git history: 45e64d7/c5a8671 revert those).

    Root cause ORIGINALLY diagnosed here (2026-07-29, cmhc003): ts_rank_cd
    (Pool's bm25_score) is a cover-density ranker, not a relevance judge --
    a chunk that repeats a required phrase (e.g. the payer name) many
    times in OFF-TOPIC content can out-score a chunk that states the
    actual answer once, clearly. That diagnosis is still correct and is
    exactly why bm25 alone isn't given 100% weight here -- tag_coverage/
    meta_boost still catch real cases bm25 misses (e.g. cmhc005's H0015
    HCPCS-code case, where bm25 alone only reaches 0.41-0.73, solidly
    mid-pack). This reweight keeps both signals live at meaningful, just
    smaller, weight -- not zeroed out.

    CRITICAL: This is Filler a's PRIMARY signal composition. Pool provides
    candidates pre-scored by arm (bm25_score, vector score, etc.) but does
    NOT do multi-signal fusion across all signals (gate b, read-only). Each
    Filler composes its own signal set for its own arm.
    """
    # Min-max normalization (2026-08-15, Ananth's call, see docstring):
    # rescale bm25_score against the POOL's own min/max for this query,
    # not an assumed [0, 1] absolute scale -- a candidate that decisively
    # leads its actual field gets close to 1.0 regardless of where the
    # pool's raw magnitudes happen to sit. bm25_bounds=None (no pool
    # context supplied) falls back to the raw value unchanged.
    raw_bm25 = candidate.bm25_score or 0.0
    # TEMPORARILY DISABLED (2026-08-15) for a controlled A/B against the
    # real 22-query bank -- min-max deployment showed a real regression
    # (bank recall 0.649 -> 0.551, recall_answer 0.305 -> ~0.28-0.30)
    # that the offline proxy sweep used to validate it did NOT predict
    # (proxy showed 0.745 -> 0.791, an apparent improvement). Suspected
    # mechanism: min-max always promotes the POOL's best candidate to
    # bm25_sig=1.0 regardless of whether that candidate is actually a
    # strong absolute match -- for a query where the whole pool scores
    # low (a hard/rare query), this artificially inflates a mediocre
    # candidate's composite score instead of correctly leaving bm25_sig
    # low and letting other signals discriminate. Testing this hypothesis
    # directly against live judged recall before deciding whether to
    # revert permanently or find a bounded variant.
    bm25_sig = raw_bm25
    auth_sig = _compute_authority_score(candidate.authority_level)
    cov_sig = _compute_tag_coverage_score(candidate.tags)
    len_sig = _compute_length_score(candidate.text)
    meta_sig = _compute_meta_boost_score(candidate.text, candidate.tags, required_phrases, boosted_phrases)
    doc_type_sig = _compute_doc_type_score(candidate.doc_type)

    # DIAGNOSTIC BISECTION: iteration E (bm25=.50/doc_type=.20) got
    # Daraprim to isolated rank #9, but real per-mode portfolio dispatch
    # caps (copilot=4, default=6, thinking=10) meant it only resolved in
    # chat.thinking, at a real bank recall cost (0.649->0.596). Point C
    # (bm25=.40, doc_type=.02, this file's original v0.4 base) matched
    # baseline recall exactly with no regressions but doesn't resolve
    # Daraprim in any real mode (rank ~41). Point B (this config,
    # 2026-08-15) is the one that actually resolves Daraprim in TWO real
    # modes (default rank #6, thinking) via a leaner combination -- bm25
    # up to .51, doc_type modest at .02, cov cut hard (.20->.04, this was
    # a proxy-invented signal, not Lexicon-curated), meta trimmed
    # slightly (.20->.18), auth up (.10->.13) to compensate, misc .06.
    # Isolated filler_a recall alone reads low (0.596) but that's not
    # the number that matters -- what matters is the REAL blended
    # portfolio bank recall across a+b+s dispatch, verified separately
    # to confirm it clears the 0.78 bar before shipping.
    composite = (
        0.51 * bm25_sig +
        0.13 * auth_sig +
        0.04 * cov_sig +
        0.06 * len_sig +
        0.18 * meta_sig +
        0.02 * doc_type_sig +
        0.06 * 0.5  # Misc (neutral baseline)
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
    Fill pre-built slots by ranking Pool's candidates via multi-signal BM25 composition.

    Algorithm (Phase 2 of fillers spec):
    1. Compose multi-signal score per candidate: bm25 (0.55) + authority (0.10)
       + tag_coverage (0.10) + length (0.10) + meta_boost (0.10) + misc (0.05).
    2. Sort candidates by composite score descending.
    3. For each slot in priority order:
       - Filter candidates if needed (by slot semantics / source_type).
       - Take top-N where N = slot.capacity (non-overlapping).
       - Assign to slot, remove from pool.
    4. Track occupancy, under_fill, over_fill.

    Signal composition (v0.3):
    - BM25 (0.55): Pool's ts_rank_cd OR-joined tsquery on search_vec [0, 1].
    - Authority (0.10): Document authority level (contract_source_of_truth > ... > none).
    - Tag coverage (0.10): Normalized by tag count (higher = more document relevance).
    - Length (0.10): Text quality signal (penalize <50 chars, reward 100-500).
    - Meta_boost (0.10): Selectivity-weighted fraction of Gate's required/boosted phrases
      present in chunk text or tags (REQUIRED full weight, BOOSTED half weight).
    - Misc (0.05): Neutral baseline for unscored candidates.

    Args:
        pool_result: Output from Pool (Step 2). May include required_phrases and
            boosted_phrases (Gate's selectivity-weighted phrase lists) for meta_boost.
        shape_result: Output from Shape (Step 1).
        routing_ladders: Optional per-slot strategy sequences (unused v1).

    Returns:
        FilledShape with all slots assigned.
    """

    # Compose multi-signal rerank score and sort descending.
    # Primary: bm25_score (Pool's BM25 ranking).
    # Secondary: authority, tag_coverage, length, meta_boost (Gate phrase selectivity).
    # Neighbors (positional adjacency only) get bm25_score=None and are filtered out.
    scored_candidates = [c for c in pool_result.candidates if c.bm25_score is not None]

    # Extract Gate's phrase lists from pool_result (may be None in v0.x until Pool ships)
    required_phrases = getattr(pool_result, 'required_phrases', None) or []
    boosted_phrases = getattr(pool_result, 'boosted_phrases', None) or []

    # Pool-wide bm25 bounds for min-max normalization (2026-08-15) -- computed
    # ONCE here (not per-candidate) since it's the same field for every
    # candidate in this pool. None (empty pool, already filtered to
    # bm25_score is not None above, so only possible when scored_candidates
    # is empty) is handled by _compute_rerank_score's own bm25_bounds=None
    # fallback.
    bm25_bounds = (
        (min(c.bm25_score for c in scored_candidates), max(c.bm25_score for c in scored_candidates))
        if scored_candidates else None
    )

    # Compute composite rerank score per candidate
    scored_candidates = [
        (c, _compute_rerank_score(c, pool_result.query, required_phrases, boosted_phrases, bm25_bounds))
        for c in scored_candidates
    ]
    # Sort by composite score descending
    scored_candidates.sort(key=lambda pair: pair[1], reverse=True)
    # Keep the composite alongside chunk_id -- FilledChunk.rerank_score needs
    # it below (Synthesis's own rerank/trim steps otherwise fall back to
    # original_score/raw bm25 alone, silently discarding everything this
    # composite adds; see contracts.py's rerank_score docstring).
    composite_by_chunk_id = {c.chunk_id: score for c, score in scored_candidates}
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
                    rerank_score=composite_by_chunk_id.get(candidate.chunk_id),
                    assignment_reason="score_rank",
                    authority_level=candidate.authority_level,
                    filler_strategy="bm25",
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
