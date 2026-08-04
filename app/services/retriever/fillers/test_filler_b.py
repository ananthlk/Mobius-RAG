"""Unit tests for Filler b (vector-similarity reranking strategy).

Covers the composite reranker (`_rerank_vector_candidates` — legacy
sim/authority/length/jpd/meta_boost signals + decay floor + the two stopgap
mitigations, see filler_b.py's module docstring) and the slot-filling
wrapper (`fill_shape_vector`).

STRUCTURAL FIX (2026-07-30): Filler b now ranks the WHOLE deduped Pool
candidate set by `PoolCandidate.vector_similarity` (Pool-computed for every
candidate regardless of which arm's provenance survived dedup), not just
`source_arm=="vector"` candidates. The old `source_arm` filter silently
dropped candidates that vector_search found but that got dedup-stamped with
a different arm's provenance — see module docstring's "STRUCTURAL FIX" for
the real bug this closes.
"""

import pytest

from app.services.retriever.fillers.filler_b import (
    fill_shape_vector,
    _rerank_vector_candidates,
)
from app.services.retriever.pool.contracts import PoolCandidate, PoolResult
from app.services.retriever.shape.slots import AnswerShapeResult, AnswerSlot


def _vec_candidate(chunk_id, *, score, text="A" * 200, source_type="document",
                    source_arm="vector", is_neighbor=False, **kw):
    """`score` sets `vector_similarity` -- the field Filler b actually reads
    now. The legacy `.score` field (arm-overloaded: tag_select's coverage
    count, vector's old raw similarity, inherited's None) is left at its
    dataclass default (None) unless a test explicitly needs to prove it's
    ignored, since Filler b must never read it anymore."""
    return PoolCandidate(
        chunk_id=chunk_id,
        document_id=kw.pop("document_id", f"doc_{chunk_id}"),
        text=text,
        score=kw.pop("legacy_score", None),
        vector_similarity=score,
        bm25_score=kw.pop("bm25_score", 0.0),  # must never affect ranking here
        source_arm=source_arm,
        is_neighbor=is_neighbor,
        source_type=source_type,
        **kw,
    )


@pytest.fixture
def sample_candidates():
    """Sample pool candidates: some with vector_similarity (rankable
    regardless of source_arm), some without (must be excluded)."""
    return [
        _vec_candidate("chunk_1", score=0.95, text="Policy on medical necessity " * 20),
        _vec_candidate("chunk_2", score=0.87, text="Prior authorization requirements " * 20),
        _vec_candidate("chunk_3", score=0.72, text="Coverage limitations " * 20),
        # tag_select provenance, but vector_search ALSO found it (Pool's
        # backfill) -- this is exactly the dedup-masking case, and it
        # SHOULD be rankable now.
        PoolCandidate(
            chunk_id="tag_select_with_vector_sim",
            document_id="doc_2",
            text="Dual-found chunk: real content matched by both arms " * 5,
            score=5.0,  # legacy coverage count -- irrelevant to Filler b now
            vector_similarity=0.80,
            bm25_score=0.65,
            source_arm="tag_select",
            is_neighbor=False,
            source_type="document",
        ),
        # inherited provenance, no vector_similarity -- genuinely excluded
        # (vector_search never scored it, unlike the case above).
        PoolCandidate(
            chunk_id="inherited_no_vector_sim",
            document_id="doc_3",
            text="AHCA-inherited authority doc " * 20,
            score=None,
            vector_similarity=None,
            bm25_score=0.55,
            source_arm="inherited",
            is_neighbor=False,
            source_type="document",
        ),
        # neighbor with NO vector_similarity -- excluded (mirrors the
        # "embed call failed" / "not in the batch" case).
        PoolCandidate(
            chunk_id="neighbor_no_vector_sim",
            document_id="doc_3",
            text="Neighbor context with no similarity computed " * 5,
            score=None,
            vector_similarity=None,
            bm25_score=None,
            source_arm="",
            is_neighbor=True,
            source_type="document",
        ),
    ]


@pytest.fixture
def sample_shape():
    """Sample shape result with slots."""
    return AnswerShapeResult(
        slots=[
            AnswerSlot(
                slot_id="direct_answer",
                slot_semantics="direct_answer",
                capacity=2,
                required=True,
                priority=0,
            ),
            AnswerSlot(
                slot_id="thematic_0",
                slot_semantics="thematic_exploration",
                capacity=1,
                required=False,
                priority=0,
            ),
        ]
    )


class TestRerankSignals:
    """Direct tests of the composite reranker math."""

    def test_sim_still_drives_ranking_when_other_signals_equal(self):
        """With length/jpd held equal (no jpd hits, same-length text), a
        higher vector_similarity should still win -- sim is 0.25 of the
        composite, not zeroed out."""
        candidates = [
            _vec_candidate("low_sim", score=0.10, text="Neutral generic text. " * 20),
            _vec_candidate("high_sim", score=0.95, text="Neutral generic text. " * 20),
        ]
        reranked, _stats = _rerank_vector_candidates(candidates, query="what is the deadline")
        ids = [c.chunk_id for c, _ in reranked]
        assert ids[0] == "high_sim"

    def test_bm25_score_never_used(self):
        """CRITICAL: bm25_score must have zero effect on Filler b's ranking
        -- it's Filler a's signal, computed uniformly but semantically
        irrelevant here (see filler_a's own field-confusion near-miss)."""
        candidates = [
            _vec_candidate("real_winner", score=0.90, bm25_score=0.01,
                            text="Neutral generic text. " * 20),
            _vec_candidate("bm25_decoy", score=0.20, bm25_score=0.99,
                            text="Neutral generic text. " * 20),
        ]
        reranked, _stats = _rerank_vector_candidates(candidates, query="what is the deadline")
        assert reranked[0][0].chunk_id == "real_winner", (
            "bm25_score leaked into vector reranking"
        )

    def test_legacy_score_field_never_used(self):
        """CRITICAL, structural-fix-specific: the arm-overloaded `.score`
        field (tag_select's coverage count, etc.) must have zero effect --
        only `vector_similarity` drives ranking now."""
        candidates = [
            _vec_candidate("real_winner", score=0.90, legacy_score=0.01,
                            text="Neutral generic text. " * 20),
            _vec_candidate("legacy_score_decoy", score=0.20, legacy_score=99.0,
                            text="Neutral generic text. " * 20, source_arm="tag_select"),
        ]
        reranked, _stats = _rerank_vector_candidates(candidates, query="what is the deadline")
        assert reranked[0][0].chunk_id == "real_winner", (
            "legacy .score field leaked into vector_similarity-based reranking"
        )

    def test_hard_length_floor_filters_junk_entirely(self):
        """Stopgap 1: a sub-50-char chunk must be dropped OUTRIGHT before
        ranking, not merely soft-penalized -- this is the mitigation added
        after the soft 0.05-weighted length penalty was proven insufficient
        (junk's raw-similarity margin beat it on real data, see Retriever's
        live repro)."""
        candidates = [
            _vec_candidate("junk_stub", score=0.99, text="Florida Medicaid o CHIP"),  # 23 chars
            _vec_candidate("real_content", score=0.50, text="Real substantive policy content here. " * 5),
        ]
        reranked, stats = _rerank_vector_candidates(candidates, query="policy question")
        ids = [c.chunk_id for c, _ in reranked]
        assert "junk_stub" not in ids, "sub-50-char junk was not hard-filtered despite a high raw score"
        assert "real_content" in ids
        assert stats["length_filtered"] == 1

    def test_length_soft_penalty_still_applies_above_floor(self):
        """Two candidates both clearing the hard floor (>=50 chars) should
        still have length as a real, if soft, tie-breaking signal -- the
        hard floor doesn't replace the graded _length_score contribution
        for genuinely-substantive-but-shorter chunks."""
        candidates = [
            _vec_candidate("short_but_valid", score=0.80, text="A" * 55),
            _vec_candidate("longer", score=0.79, text="Real policy content here. " * 20),
        ]
        reranked, _stats = _rerank_vector_candidates(candidates, query="policy question")
        ids = [c.chunk_id for c, _ in reranked]
        assert ids[0] == "longer", "length signal did not tip a close race above the hard floor"

    def test_exact_text_dedup_collapses_duplicates(self):
        """Stopgap 2: candidates sharing byte-identical text (e.g. the same
        boilerplate line repeated across many documents) must collapse to
        ONE representative, not each independently win a slot position."""
        candidates = [
            _vec_candidate(f"dup_{i}", score=0.90, text="Repeated boilerplate line content here. " * 3)
            for i in range(8)
        ] + [
            _vec_candidate("distinct", score=0.85, text="Genuinely different substantive content. " * 3),
        ]
        reranked, stats = _rerank_vector_candidates(candidates, query="content")
        ids = [c.chunk_id for c, _ in reranked]
        dup_survivors = [i for i in ids if i.startswith("dup_")]
        assert len(dup_survivors) == 1, "exact-text dedup did not collapse identical-text candidates"
        assert "distinct" in ids
        assert stats["dedup_collapsed"] == 7  # 8 duplicates -> 1 survivor = 7 collapsed

    def test_dedup_keeps_highest_scoring_duplicate(self):
        """When duplicates have different underlying scores (e.g. from
        different documents), the survivor should be the best-scoring one,
        not an arbitrary one."""
        candidates = [
            _vec_candidate("dup_low", score=0.60, text="Identical duplicate text here. " * 3),
            _vec_candidate("dup_high", score=0.95, text="Identical duplicate text here. " * 3),
        ]
        reranked, _stats = _rerank_vector_candidates(candidates, query="content")
        assert len(reranked) == 1
        assert reranked[0][0].chunk_id == "dup_high"

    def test_reproduces_retriever_failure_case_now_fixed(self):
        """Direct regression test for the exact failure shape Retriever
        reproduced live: many candidates sharing one short duplicate
        boilerplate chunk, scoring higher than real content, dominating a
        10-capacity slot. Confirms the stopgaps fix it -- occupancy should
        now be real, distinct content, not 10 copies of junk."""
        junk_candidates = [
            _vec_candidate(f"junk_{i}", score=0.797, text="Florida Medicaid o CHIP",
                            document_id=f"junk_doc_{i}")
            for i in range(12)  # more than capacity, as in the real repro (191 vector candidates)
        ]
        real_candidates = [
            _vec_candidate(f"real_{i}", score=0.60 - (i * 0.01),
                            text=f"Timely filing deadline policy section {i}: participating providers "
                                 f"must submit claims within 180 days of service. " * 2,
                            document_id=f"real_doc_{i}")
            for i in range(10)
        ]
        pool_result = PoolResult(
            query="What is the timely filing deadline for Sunshine Health FL Medicaid claims?",
            candidates=junk_candidates + real_candidates,
            pool_ms=100,
        )
        shape = AnswerShapeResult(
            slots=[AnswerSlot(slot_id="direct_answer", slot_semantics="direct_answer",
                               capacity=10, required=True, priority=0)]
        )
        filled = fill_shape_vector(pool_result, shape)

        assigned_ids = [c.chunk_id for c in filled.slots[0].chunks]
        assert filled.slots[0].occupancy == 10
        assert all(cid.startswith("real_") for cid in assigned_ids), (
            f"junk still won slot positions despite higher raw score: {assigned_ids}"
        )
        assert len(set(c.text for c in filled.slots[0].chunks)) == 10, "assigned chunks are not distinct content"

    def test_jpd_signal_boosts_topical_match(self):
        """A chunk whose text matches the query's JPD category (e.g. prior
        authorization) should outrank an off-topic chunk with equal sim."""
        candidates = [
            _vec_candidate(
                "on_topic", score=0.70,
                text="This policy requires prior authorization and medical necessity review. " * 5,
            ),
            _vec_candidate(
                "off_topic", score=0.70,
                text="This document describes unrelated general information. " * 5,
            ),
        ]
        reranked, _stats = _rerank_vector_candidates(
            candidates, query="does this require prior authorization"
        )
        ids = [c.chunk_id for c, _ in reranked]
        assert ids[0] == "on_topic", "jpd signal did not reward the topically-matching chunk"

    def test_jpd_inert_when_query_has_no_category_hits(self):
        """A query with no JPD keyword hits should leave ranking to
        sim/length alone -- W_JPD zeroes out (mirrors legacy's has_jpd
        fallback), not a crash or a default boost."""
        candidates = [
            _vec_candidate("higher_sim", score=0.85, text="Some content here. " * 20),
            _vec_candidate("lower_sim", score=0.40, text="Some other content. " * 20),
        ]
        # Query has no words matching any _JPD_PATTERNS category.
        reranked, _stats = _rerank_vector_candidates(candidates, query="xyzzy plugh qwerty")
        assert reranked[0][0].chunk_id == "higher_sim"

    def test_meta_boost_rewards_required_phrase_presence(self):
        """A candidate containing a Gate-derived required phrase should
        outrank an equal-sim candidate that doesn't -- the whole point of
        wiring meta_boost in (2026-07-30, diagnosed against real cmhc002/
        cmhc017 failures where generic on-topic chunks beat the chunk that
        actually contained the specific needed fact)."""
        candidates = [
            _vec_candidate("has_phrase", score=0.70,
                            text="Prior authorization is required for H0019 residential treatment. " * 3),
            _vec_candidate("no_phrase", score=0.70,
                            text="General information about behavioral health services offered. " * 3),
        ]
        required_phrases = [("h0019", 0.9)]
        reranked, _stats = _rerank_vector_candidates(
            candidates, query="does this require prior auth for H0019",
            required_phrases=required_phrases,
        )
        assert reranked[0][0].chunk_id == "has_phrase", (
            "meta_boost did not reward the chunk containing the required phrase"
        )

    def test_meta_boost_uses_boosted_phrases_at_half_weight(self):
        """Boosted-bucket phrases count for half weight vs required-bucket
        (mirrors Filler a's _compute_meta_boost_score exactly, reused not
        reimplemented) -- a chunk hitting only a boosted phrase should score
        between a no-phrase chunk and a required-phrase chunk."""
        no_phrase = _vec_candidate("no_phrase", score=0.70, text="Unrelated content here. " * 5)
        boosted_only = _vec_candidate("boosted_only", score=0.70, text="Discusses telehealth modifiers. " * 5)
        required_present = _vec_candidate("required_present", score=0.70, text="Discusses telehealth GT modifier requirements. " * 5)

        required_phrases = [("gt modifier", 0.9)]
        boosted_phrases = [("telehealth modifiers", 0.5)]

        reranked, stats = _rerank_vector_candidates(
            [no_phrase, boosted_only, required_present], query="telehealth modifier requirements",
            required_phrases=required_phrases, boosted_phrases=boosted_phrases,
        )
        scores = {c.chunk_id: s for c, s in reranked}
        assert scores["required_present"] > scores["boosted_only"], (
            "required-bucket phrase should outscore boosted-bucket (full vs half weight)"
        )
        # no_phrase legitimately fails the decay floor here -- meta_boost
        # widens the score gap enough that it's genuinely > 0.6x the best in
        # its category, not a bug. Confirm it was dropped, not silently lost.
        assert "no_phrase" not in scores
        assert stats["decay_dropped"] == 1

    def test_meta_boost_inert_when_no_phrases_provided(self):
        """No required/boosted phrases (the common case until upstream
        callers pass them) -- W_META zeroes out, same as pre-meta_boost
        behavior, not a crash or a default penalty/boost."""
        candidates = [
            _vec_candidate("a", score=0.85, text="Some content over here, padded. " * 20),
            _vec_candidate("b", score=0.40, text="Different content over here, padded. " * 20),
        ]
        reranked, _stats = _rerank_vector_candidates(candidates, query="query")
        assert reranked[0][0].chunk_id == "a"

    def test_decay_floor_drops_low_scoring_candidates(self):
        """A candidate scoring well below the best-in-category (< 0.6x)
        should be dropped entirely, not just deprioritized -- same
        threshold as legacy's per-category decay."""
        candidates = [
            _vec_candidate("best", score=0.95, text="Strong match content. " * 20),
            _vec_candidate("far_behind", score=0.05, text="Weak match content. " * 20),
        ]
        reranked, _stats = _rerank_vector_candidates(candidates, query="strong match")
        ids = [c.chunk_id for c, _ in reranked]
        assert "best" in ids
        assert "far_behind" not in ids, "decay floor should have dropped a far-lower scorer"

    def test_decay_floor_is_per_source_type_category(self):
        """A low scorer in a DIFFERENT source_type category should survive
        even though it would've been dropped against a much-better
        same-category peer -- category = source_type here."""
        candidates = [
            _vec_candidate("doc_best", score=0.95, text="Strong doc content. " * 20, source_type="document"),
            _vec_candidate("web_only", score=0.30, text="Web content, own category. " * 20, source_type="web"),
        ]
        reranked, _stats = _rerank_vector_candidates(candidates, query="content")
        ids = [c.chunk_id for c, _ in reranked]
        assert "doc_best" in ids
        assert "web_only" in ids, "decay floor incorrectly cross-contaminated across source_type categories"

    def test_authority_default_is_uniform_noop(self):
        """Two candidates with no authority_level set (the dataclass
        default, None) should both fall back to legacy's own
        _AUTHORITY_DEFAULT uniformly -- not introduce bias between
        otherwise-identical candidates."""
        # Distinct text (same length) so exact-text dedup doesn't collapse
        # them -- this test is about the authority signal, not dedup.
        candidates = [
            _vec_candidate("a", score=0.5, text="Some content over here, padded. " * 20),
            _vec_candidate("b", score=0.5, text="Other content over here, padded. " * 20),
        ]
        reranked, _stats = _rerank_vector_candidates(candidates, query="content")
        scores = [score for _, score in reranked]
        assert scores[0] == pytest.approx(scores[1]), "authority default introduced ranking bias"

    def test_authority_level_differentiates_ranking_when_populated(self):
        """Real authority_level (Pool's document_authority_level
        passthrough) should outrank an equal-sim candidate with a lower
        authority tier -- proves the signal is actually wired in, not just
        structurally present."""
        candidates = [
            _vec_candidate("high_authority", score=0.70, text="Policy content. " * 20,
                            authority_level="contract_source_of_truth"),
            _vec_candidate("low_authority", score=0.70, text="Different policy content. " * 20,
                            authority_level="fyi_not_citable"),
        ]
        reranked, _stats = _rerank_vector_candidates(candidates, query="policy")
        assert reranked[0][0].chunk_id == "high_authority"

    def test_authority_level_empty_string_treated_as_missing(self):
        """DB default for document_authority_level is '' (NOT NULL,
        default=''), not None -- must fall back to _AUTHORITY_DEFAULT same
        as None, not KeyError or a mismatched-key silent 0.0 (the exact bug
        class found in filler_a.py's own reimplementation, flagged
        separately -- see module docstring)."""
        candidates = [
            _vec_candidate("empty_string_auth", score=0.5, text="Some content over here, padded. " * 20,
                            authority_level=""),
            _vec_candidate("none_auth", score=0.5, text="Other content over here, padded. " * 20,
                            authority_level=None),
        ]
        reranked, _stats = _rerank_vector_candidates(candidates, query="content")
        scores = [score for _, score in reranked]
        assert scores[0] == pytest.approx(scores[1]), (
            "empty-string authority_level was not treated the same as None"
        )

    def test_returns_empty_for_empty_input(self):
        reranked, stats = _rerank_vector_candidates([], query="anything")
        assert reranked == []
        assert stats == {"length_filtered": 0, "dedup_collapsed": 0, "decay_dropped": 0}


class TestFillerBBasic:
    """fill_shape_vector()-level behavior."""

    def test_includes_dedup_masked_candidate_with_vector_similarity(self, sample_candidates, sample_shape):
        """THE core structural-fix regression test: a candidate whose Pool
        provenance says source_arm=="tag_select" (dedup's first-arm-wins
        outcome) but that carries a real Pool-backfilled vector_similarity
        must be rankable and assignable -- this is exactly the cmhc017 bug
        (correct chunk masked by tag_select's provenance, invisible to the
        OLD source_arm=="vector" filter)."""
        pool_result = PoolResult(
            query="medical necessity", candidates=sample_candidates, pool_ms=150,
        )
        filled = fill_shape_vector(pool_result, sample_shape)
        all_assigned_ids = {c.chunk_id for slot in filled.slots for c in slot.chunks}
        assert "tag_select_with_vector_sim" in all_assigned_ids, (
            "dedup-masked candidate with a real vector_similarity was not considered -- "
            "the structural fix regressed"
        )

    def test_excludes_candidates_without_vector_similarity(self, sample_candidates, sample_shape):
        """Candidates with vector_similarity=None must never be selected,
        regardless of source_arm or what the legacy .score field holds --
        the new filter is vector_similarity-based, not arm-based."""
        pool_result = PoolResult(
            query="medical necessity", candidates=sample_candidates, pool_ms=150,
        )
        filled = fill_shape_vector(pool_result, sample_shape)

        all_assigned_ids = {c.chunk_id for slot in filled.slots for c in slot.chunks}
        assert "inherited_no_vector_sim" not in all_assigned_ids
        assert "neighbor_no_vector_sim" not in all_assigned_ids

    def test_under_fill_detection(self, sample_shape):
        pool_result = PoolResult(
            query="medical necessity",
            candidates=[_vec_candidate("only_one", score=0.9, text="content " * 20)],
            pool_ms=150,
        )
        filled = fill_shape_vector(pool_result, sample_shape)

        direct_slot = filled.slots[0]
        assert direct_slot.occupancy == 1
        assert direct_slot.under_filled

        thematic_slot = filled.slots[1]
        assert thematic_slot.occupancy == 0
        assert thematic_slot.under_filled

    def test_empty_pool(self, sample_shape):
        pool_result = PoolResult(query="medical necessity", candidates=[], pool_ms=150)
        filled = fill_shape_vector(pool_result, sample_shape)
        for slot in filled.slots:
            assert slot.occupancy == 0
            assert len(slot.chunks) == 0
            assert slot.under_filled

    def test_all_candidates_missing_vector_similarity_yields_empty(self, sample_shape):
        """Mirrors Pool's real fallback: if the query embedding call failed,
        Pool's attach_vector_similarity leaves vector_similarity=None on
        EVERY candidate, not just some -- Filler b must handle that as a
        clean empty result, not crash."""
        candidates = [
            PoolCandidate(
                chunk_id="no_sim_1", document_id="doc_x", text="Context chunk " * 10,
                score=None, vector_similarity=None, source_arm="vector", is_neighbor=False,
                source_type="document",
            ),
        ]
        pool_result = PoolResult(query="query", candidates=candidates, pool_ms=50)
        filled = fill_shape_vector(pool_result, sample_shape)
        for slot in filled.slots:
            assert slot.occupancy == 0

    def test_capacity_respected(self, sample_candidates, sample_shape):
        small_shape = AnswerShapeResult(
            slots=[AnswerSlot(slot_id="slot_1", slot_semantics="direct_answer",
                               capacity=1, required=True, priority=0)]
        )
        pool_result = PoolResult(query="query", candidates=sample_candidates, pool_ms=150)
        filled = fill_shape_vector(pool_result, small_shape)
        assert filled.slots[0].occupancy == 1
        assert len(filled.slots[0].chunks) == 1

    def test_deterministic_order(self, sample_candidates, sample_shape):
        pool_result = PoolResult(
            query="medical necessity", candidates=sample_candidates.copy(), pool_ms=150,
        )
        filled_1 = fill_shape_vector(pool_result, sample_shape)
        filled_2 = fill_shape_vector(pool_result, sample_shape)
        for slot_1, slot_2 in zip(filled_1.slots, filled_2.slots):
            assert [c.chunk_id for c in slot_1.chunks] == [c.chunk_id for c in slot_2.chunks]

    def test_emit_diagnostics(self, sample_candidates, sample_shape):
        pool_result = PoolResult(
            query="medical necessity", candidates=sample_candidates, pool_ms=150,
        )
        filled = fill_shape_vector(pool_result, sample_shape)

        assert filled.filling_strategy == "vector_rerank"
        assert filled.emit["fillers_decision"] == "vector_rerank_assign"
        for key in ("slots_filled", "empty_slots", "under_filled", "total_chunks_assigned",
                    "length_filtered", "dedup_collapsed", "decay_dropped", "per_slot_details"):
            assert key in filled.emit

    def test_field_preservation(self, sample_shape):
        candidate = _vec_candidate(
            "test_chunk", score=0.9, text="Test text content. " * 20,
            document_status="live", content_sha="sha123",
            tags={"d:domain": 1, "p:process": 2},
        )
        pool_result = PoolResult(query="test", candidates=[candidate], pool_ms=100)
        filled = fill_shape_vector(pool_result, sample_shape)

        chunk = filled.slots[0].chunks[0]
        assert chunk.chunk_id == "test_chunk"
        assert chunk.document_id == "doc_test_chunk"
        assert chunk.is_neighbor is False
        assert chunk.source_type == "document"
        assert chunk.document_status == "live"
        assert chunk.content_sha == "sha123"
        assert chunk.tags == {"d:domain": 1, "p:process": 2}
        assert chunk.assignment_reason == "vector_rerank"
        assert 0.0 <= chunk.original_score <= 1.0

    def test_pool_result_required_phrases_flow_through_end_to_end(self, sample_shape):
        """PoolResult.required_phrases must actually reach meta_boost via
        fill_shape_vector -- not just work in the unit-tested
        _rerank_vector_candidates function directly."""
        has_phrase = _vec_candidate("has_phrase", score=0.70,
                                     text="Discusses the H0019 code directly here. " * 5)
        no_phrase = _vec_candidate("no_phrase", score=0.70,
                                    text="Generic unrelated content over here. " * 5)
        pool_result = PoolResult(
            query="H0019 billing question",
            candidates=[no_phrase, has_phrase],
            pool_ms=100,
            required_phrases=[("h0019", 0.9)],
        )
        small_shape = AnswerShapeResult(
            slots=[AnswerSlot(slot_id="direct_answer", slot_semantics="direct_answer",
                               capacity=1, required=True, priority=0)]
        )
        filled = fill_shape_vector(pool_result, small_shape)
        assert filled.slots[0].chunks[0].chunk_id == "has_phrase", (
            "required_phrases on PoolResult did not reach meta_boost through fill_shape_vector"
        )

    def test_many_candidates_many_slots(self):
        candidates = [
            _vec_candidate(f"chunk_{i}", score=1.0 - (i * 0.01),
                            text=f"Content number {i} with enough length to clear the stub floor. " * 3)
            for i in range(100)
        ]
        slots = [
            AnswerSlot(slot_id=f"slot_{i}", slot_semantics="direct_answer",
                       capacity=5, required=True, priority=i)
            for i in range(10)
        ]
        shape = AnswerShapeResult(slots=slots)
        pool_result = PoolResult(query="test", candidates=candidates, pool_ms=200)
        filled = fill_shape_vector(pool_result, shape)

        total_assigned = sum(s.occupancy for s in filled.slots)
        assert total_assigned == filled.total_chunks_assigned
        assert total_assigned <= 100
        # Every slot up to available supply should be filled (decay floor may
        # drop some low-scoring tail candidates -- not asserting exactly 100).
        assert filled.slots[0].occupancy == 5
