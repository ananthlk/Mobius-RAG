"""Unit tests for Filler b (vector-search reranking strategy).

Covers the composite reranker (`_rerank_vector_candidates` — legacy
sim/authority/length/jpd signals + decay floor + the two stopgap
mitigations added after Retriever reproduced a 100%-junk result set live,
see filler_b.py's module docstring) and the slot-filling wrapper
(`fill_shape_vector`).
"""

import pytest

from app.services.retriever.fillers.filler_b import (
    fill_shape_vector,
    _rerank_vector_candidates,
)
from app.services.retriever.pool.contracts import PoolCandidate, PoolResult
from app.services.retriever.shape.slots import AnswerShapeResult, AnswerSlot


def _vec_candidate(chunk_id, *, score, text="A" * 200, source_type="document", **kw):
    return PoolCandidate(
        chunk_id=chunk_id,
        document_id=kw.pop("document_id", f"doc_{chunk_id}"),
        text=text,
        score=score,
        bm25_score=kw.pop("bm25_score", 0.0),  # must never affect ranking here
        source_arm="vector",
        is_neighbor=False,
        source_type=source_type,
        **kw,
    )


@pytest.fixture
def sample_candidates():
    """Sample pool candidates mixing vector, tag_select, and inherited arms."""
    return [
        _vec_candidate("chunk_1", score=0.95, text="Policy on medical necessity " * 20),
        _vec_candidate("chunk_2", score=0.87, text="Prior authorization requirements " * 20),
        _vec_candidate("chunk_3", score=0.72, text="Coverage limitations " * 20),
        PoolCandidate(
            chunk_id="tag_select_high_score",
            document_id="doc_2",
            text="Tag-matched chunk with a high coverage COUNT, not a similarity " * 5,
            score=5.0,  # Coverage count -- NOT comparable to cosine similarity.
            bm25_score=0.65,
            source_arm="tag_select",
            is_neighbor=False,
            source_type="document",
        ),
        PoolCandidate(
            chunk_id="inherited_1",
            document_id="doc_3",
            text="AHCA-inherited authority doc " * 20,
            score=None,  # inherited arm never sets a score.
            bm25_score=0.55,
            source_arm="inherited",
            is_neighbor=False,
            source_type="document",
        ),
        PoolCandidate(
            chunk_id="neighbor_1",
            document_id="doc_3",
            text="Neighbor context " * 20,
            score=None,
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
        higher raw .score should still win -- sim is 0.25 of the composite,
        not zeroed out."""
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
        same-category peer -- category = source_type here (arm is
        constant 'vector')."""
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
        """Real authority_level (now populated from Pool's
        document_authority_level passthrough) should outrank an equal-sim
        candidate with a lower authority tier -- proves the signal is
        actually wired in, not just structurally present."""
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

    def test_excludes_non_vector_arms(self, sample_candidates, sample_shape):
        """tag_select and inherited candidates must never be selected, even
        when their .score (a different signal entirely) would outrank every
        vector candidate."""
        pool_result = PoolResult(
            query="medical necessity", candidates=sample_candidates, pool_ms=150,
        )
        filled = fill_shape_vector(pool_result, sample_shape)

        all_assigned_ids = {c.chunk_id for slot in filled.slots for c in slot.chunks}
        assert "tag_select_high_score" not in all_assigned_ids
        assert "inherited_1" not in all_assigned_ids
        assert "neighbor_1" not in all_assigned_ids

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

    def test_all_neighbors_no_scores(self, sample_shape):
        neighbors_only = [
            PoolCandidate(
                chunk_id="neighbor_1", document_id="doc_x", text="Context chunk",
                score=None, source_arm="", is_neighbor=True, source_type="document",
            ),
        ]
        pool_result = PoolResult(query="query", candidates=neighbors_only, pool_ms=50)
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
