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

  - sim    (0.25): PoolCandidate.vector_similarity (NOT raw `.score` — see
    "STRUCTURAL FIX 2026-07-30" below for why). Legacy's own `_best_arm_sim`
    only rescales cosine similarity in RRF/hybrid mode (`arm_scores` set);
    Filler b is single-arm, so the raw similarity value is used as-is, not
    an approximation of legacy's rescaling.
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
  - meta_boost (0.20, NEW 2026-07-30): `_compute_meta_boost_score(text, tags,
    required_phrases, boosted_phrases)` — imported directly from
    `filler_a.py` (reused, not reimplemented; this creates a cross-filler
    import, a real bit of debt worth moving to a shared module later, not
    blocking now). Fed by `PoolResult.required_phrases`/`.boosted_phrases`
    (selectivity-weighted phrase lists from Gate's lexicon expansion),
    which Pool already populates — Filler b just wasn't consuming them
    until now. Weight of 0.20 chosen to match Filler a's own live-trace-
    validated number for this identical signal (not legacy's untested 0.55
    — Filler a's real calibration on this corpus already found 0.55-class
    weights cause the same "superficial match beats real relevance"
    problem legacy's own formula was trying to avoid). Diagnosed directly
    against a live 22-query trace (2026-07-30, forced_strategy=b, mean
    recall 0.43): partial-recall failures consistently missed *specific*
    facts (e.g. cmhc002's H0019, cmhc017's GT/95 modifiers) that generic
    `sim`/`jpd` couldn't discriminate — `meta_boost` is aimed directly at
    that gap.
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
  - literal-code/modifier specificity (e.g. "GT"/"95" modifiers, POS codes
    "02"/"10", HCPCS-adjacent codes like "H0019"): confirmed with Lexicon
    directly (2026-07-30) that the lexicon is topic-based BY DESIGN and
    deliberately excludes bare short/digit tokens (false-match/retrieval-
    noise risk) — `meta_boost` above has nothing to reward for these even
    once populated. Needs a separate query-conditional literal-match
    signal (regex/exact match of codes present IN THE QUERY against chunk
    text) — flagged as shared, cross-filler infrastructure for someone
    else to build, not this file's scope.
  - chunk_dtag_boost multiplier: needs query d-tag codes from Gate in a
    shape `meta_boost` doesn't cover. Never triggers without
    `phrase_tag_codes`, so intentionally not implemented (dead code
    otherwise).
  - `_classify_jpd`'s hardcoded pattern dict (`corpus_search.py`) drifts
    from the real lexicon — confirmed directly with Lexicon: "referral" has
    zero JPD pattern coverage despite being fully tagged in the lexicon
    (`utilization_management.referrals`). Lexicon's own recommendation:
    wire jpd off the real lexicon expansion instead of the hardcoded dict,
    so it can't drift again. Bigger architectural change, treated as a
    separate follow-up, not bundled into this pass.
  - Neighbor score dampening (legacy: 0.5x a neighbor's seed's score): NOW
    POSSIBLY RELEVANT, NOT IMPLEMENTED — since the structural fix below
    means neighbors can enter ranking for the first time, un-dampened.
    Pool's own docstring on `vector_similarity` takes the position that a
    neighbor's similarity is "meaningful ... regardless of why it's in the
    pool" (their explicit design call, not assumed here), and no seed-to-
    neighbor back-reference exists on PoolCandidate to dampen against even
    if we wanted to. Not adding unrequested defensive logic on top of
    Pool's documented design decision — flagging as an open question to
    revisit with real data if neighbors turn out to crowd out real matches
    in practice, not pre-solving a problem with no evidence yet.

STRUCTURAL FIX (2026-07-30, Pool + Retriever): real bug, not a Filler-b-
local one — `dedup_candidates()` is first-arm-wins on chunk_id collision
(union order tag_select -> vector -> inherited). When a chunk was found by
BOTH tag_select and vector_search, the deduped entry kept tag_select's
provenance, so Filler b's OLD `source_arm=="vector"` filter never saw it —
even when vector_search independently found the same chunk with a strong
score. Confirmed live on cmhc017: the correct answer chunk sat in the pool
tagged source_arm=tag_select while a standalone vector_search() call found
the identical chunk at rank 165/1000, similarity 0.823. Real, silent recall
loss, not a one-off — and it made Filler b an outlier: Filler a never
filtered by source_arm (bm25_score is computed for every arm uniformly), so
it never had this problem.

Fix, mirroring bm25_score's own precedent: Pool now computes
`PoolCandidate.vector_similarity` for EVERY final candidate (matches AND
neighbors) in one batch query after dedup, independent of which arm's
provenance survived the union (`public_adapter.py::attach_vector_similarity`).
Filler b now ranks the WHOLE deduped pool by this field, same as Filler a
already does with bm25_score — no more source_arm filter. `score` stays
arm-overloaded (tag_select's coverage count, vector's old raw similarity,
inherited's None) and must never be read as a similarity value here anymore.

Real gotcha, not hypothetical: `vector_similarity` is `None` for every
candidate globally if the query embedding call itself failed (Pool's
`attach_vector_similarity` short-circuits `if not query_embedding`), not
just per-candidate — same "filter out and move on" handling as `.score or
0.0` used before, not a special case to add extra logic for.

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
from app.services.retriever.fillers.filler_a import _compute_meta_boost_score
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
# NEW 2026-07-30 -- weight matches Filler a's own live-trace-validated
# number for the identical signal, not legacy's untested 0.55 (see module
# docstring). Zeroed below when the query has no required/boosted phrases.
_W_META = 0.20

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
    candidates: list[PoolCandidate],
    query: str,
    required_phrases: list[tuple[str, float]] | None = None,
    boosted_phrases: list[tuple[str, float]] | None = None,
) -> tuple[list[tuple[PoolCandidate, float]], dict]:
    """Composite-rerank candidates by vector_similarity (Pool's uniform
    per-candidate field, not source_arm-restricted -- see module docstring's
    "STRUCTURAL FIX"), legacy formula otherwise (see module docstring for
    exactly which signals are real vs. structurally-inert pending upstream
    data), plus the two stopgap mitigations (hard length floor, exact-text
    dedup). Returns (survivors sorted descending, stats).
    """
    required_phrases = required_phrases or []
    boosted_phrases = boosted_phrases or []
    has_meta = bool(required_phrases or boosted_phrases)

    query_cats = _classify_jpd(query) if query else {}
    has_jpd = bool(query_cats)
    w_jpd = _W_JPD if has_jpd else 0.0
    w_meta = _W_META if has_meta else 0.0
    max_weight = _W_SIM + _W_AUTH + _W_LEN + w_jpd + w_meta

    # Stopgap 1: hard length floor -- drop before ranking, not a soft penalty.
    length_filtered = [c for c in candidates if len((c.text or "").strip()) < _MIN_CHUNK_LENGTH]
    candidates = [c for c in candidates if len((c.text or "").strip()) >= _MIN_CHUNK_LENGTH]

    scored: list[tuple[PoolCandidate, float]] = []
    for c in candidates:
        sim = c.vector_similarity or 0.0
        auth = _authority_score(c.authority_level)
        length = _length_score(c.text)
        jpd, _jpd_tags = _jpd_signal(query_cats, c.text) if has_jpd else (0.0, [])
        meta = (
            _compute_meta_boost_score(c.text, c.tags, required_phrases, boosted_phrases)
            if has_meta else 0.0
        )

        raw = (
            (_W_SIM * sim) + (_W_AUTH * auth) + (_W_LEN * length)
            + (w_jpd * jpd) + (w_meta * meta)
        )
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
    Fill pre-built slots by reranking Pool's WHOLE deduped candidate pool
    with the legacy multi-signal composite (sim/authority/length/jpd/
    meta_boost + decay floor — see module docstring for what's real vs.
    structurally-inert here) and assigning top-N per slot.

    Algorithm (Phase 2 of fillers spec):
    1. Filter to candidates with a real `vector_similarity` (populated by
       Pool for every candidate regardless of which arm's provenance
       survived dedup — see "STRUCTURAL FIX" in the module docstring for
       why this is NOT a source_arm=="vector" filter anymore).
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

    # Rank the WHOLE deduped pool, not just source_arm=="vector" -- Pool's
    # vector_similarity backfill (2026-07-30) means a candidate whose
    # provenance says "tag_select" can still carry a real, independently-
    # computed vector similarity if vector_search also found it. Filtering
    # by source_arm here was the original bug (see module docstring).
    scorable_candidates = [
        c for c in pool_result.candidates
        if c.vector_similarity is not None
    ]
    # Same getattr-with-default pattern as Filler a's meta_boost consumption
    # (filler_a.py) -- these fields may be absent on older PoolResult shapes.
    required_phrases = getattr(pool_result, "required_phrases", None) or []
    boosted_phrases = getattr(pool_result, "boosted_phrases", None) or []
    reranked, rerank_stats = _rerank_vector_candidates(
        scorable_candidates, pool_result.query, required_phrases, boosted_phrases
    )
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
                    # page_number/paragraph_index: threaded 2026-08-19. Same
                    # threading-gap class as authority_level above -- the field
                    # existed on FilledChunk ("location within source (from
                    # Pool)"), PoolCandidate carried the value, and this filler
                    # simply never passed it along, so every chunk it served
                    # reached the contract with page_number=None. Found from a
                    # live trace: page-proximity passenger-table attachment
                    # joins on (document_id, page_number), so a null page
                    # silently resolves no table -- and because that path fails
                    # open, nothing anywhere reports the miss.
                    page_number=candidate.page_number,
                    paragraph_index=candidate.paragraph_index,
                    document_id=candidate.document_id,
                    text=candidate.text,
                    document_status=candidate.document_status,
                    content_sha=candidate.content_sha,
                    source_type=candidate.source_type,
                    # Real bug found live, 2026-08-04 (Ananth caught it from
                    # the trace UI's new authority_score field): this filler
                    # READS candidate.authority_level for its own ranking
                    # score (_authority_score(c.authority_level) above) but
                    # never threaded it onto the output FilledChunk --
                    # synthesis.py's _infer_authority then had nothing but
                    # the coarse source_type fallback to go on, and
                    # source_type="hierarchical" isn't in
                    # _AUTHORITATIVE_SOURCE_TYPES ({"internal","fact_store"}),
                    # so every b-served chunk was mislabeled "external" even
                    # when it was our own real ingested, authoritative
                    # corpus content. filler_a already threads this
                    # correctly (contracts.py:302) -- b was the outlier.
                    authority_level=candidate.authority_level,
                    tags=candidate.tags or {},
                    is_neighbor=candidate.is_neighbor,
                    original_score=rerank_score_by_id[candidate.chunk_id],
                    assignment_reason="vector_rerank",
                    filler_strategy="vector_rerank",
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
