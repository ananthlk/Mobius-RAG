"""Unit tests for Filler a (BM25 ranking strategy)."""

import pytest

from app.services.retriever.fillers.filler_a import fill_shape_bm25
from app.services.retriever.pool.contracts import PoolCandidate, PoolResult
from app.services.retriever.shape.slots import AnswerShapeResult, AnswerSlot


@pytest.fixture
def sample_candidates():
    """Sample pool candidates with BM25 scores (bm25_score field)."""
    return [
        PoolCandidate(
            chunk_id="chunk_1",
            document_id="doc_1",
            text="Policy on medical necessity",
            score=0.50,  # Generic score (e.g., tag-coverage signal)
            bm25_score=0.95,  # BM25 score (what Filler a sorts by)
            source_arm="tag_select",
            is_neighbor=False,
            source_type="document",
        ),
        PoolCandidate(
            chunk_id="chunk_2",
            document_id="doc_1",
            text="Prior authorization requirements",
            score=0.48,
            bm25_score=0.87,
            source_arm="tag_select",
            is_neighbor=False,
            source_type="document",
        ),
        PoolCandidate(
            chunk_id="chunk_3",
            document_id="doc_2",
            text="Coverage limitations",
            score=0.76,  # Vector search similarity
            bm25_score=0.72,
            source_arm="vector",
            is_neighbor=False,
            source_type="document",
        ),
        PoolCandidate(
            chunk_id="chunk_4",
            document_id="doc_2",
            text="Related policy statement",
            score=0.65,
            bm25_score=0.65,
            source_arm="inherited",
            is_neighbor=False,
            source_type="document",
        ),
        PoolCandidate(
            chunk_id="chunk_5",
            document_id="doc_3",
            text="Neighbor context",
            score=None,  # Neighbor, no generic score
            bm25_score=None,  # Neighbor, no BM25 score (positional adjacency only)
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


class TestFillerABasic:
    """Basic filling algorithm tests."""

    def test_fill_by_bm25_score(self, sample_candidates, sample_shape):
        """Candidates should be ranked by BM25 score (descending)."""
        pool_result = PoolResult(
            query="medical necessity",
            candidates=sample_candidates,
            pool_ms=150,
        )

        filled = fill_shape_bm25(pool_result, sample_shape)

        # First slot: direct_answer, capacity 2
        # Should get top-2 scored candidates (excluding None score).
        direct_slot = filled.slots[0]
        assert direct_slot.occupancy == 2
        assert direct_slot.chunks[0].chunk_id == "chunk_1"  # score 0.95
        assert direct_slot.chunks[1].chunk_id == "chunk_2"  # score 0.87
        assert direct_slot.chunks[0].original_score == 0.95
        assert direct_slot.chunks[1].original_score == 0.87

    def test_under_fill_detection(self, sample_candidates, sample_shape):
        """Under-fill should be detected when occupancy < capacity."""
        # Remove some candidates to trigger under-fill.
        reduced_candidates = sample_candidates[:2]
        pool_result = PoolResult(
            query="medical necessity",
            candidates=reduced_candidates,
            pool_ms=150,
        )

        filled = fill_shape_bm25(pool_result, sample_shape)

        # First slot: capacity 2, but only 2 candidates total (one scored).
        direct_slot = filled.slots[0]
        assert direct_slot.occupancy == 2
        assert not direct_slot.under_filled

        # Second slot: capacity 1, no candidates left.
        thematic_slot = filled.slots[1]
        assert thematic_slot.occupancy == 0
        assert thematic_slot.under_filled

    def test_empty_pool(self, sample_shape):
        """Empty pool should result in empty slots."""
        pool_result = PoolResult(
            query="medical necessity",
            candidates=[],
            pool_ms=150,
        )

        filled = fill_shape_bm25(pool_result, sample_shape)

        for slot in filled.slots:
            assert slot.occupancy == 0
            assert len(slot.chunks) == 0
            assert slot.under_filled

    def test_all_neighbors_no_scores(self, sample_shape):
        """Pool with only neighbors (no scores) should result in empty slots."""
        neighbors_only = [
            PoolCandidate(
                chunk_id="neighbor_1",
                document_id="doc_x",
                text="Context chunk",
                score=None,
                source_arm="",
                is_neighbor=True,
                source_type="document",
            ),
        ]
        pool_result = PoolResult(
            query="query",
            candidates=neighbors_only,
            pool_ms=50,
        )

        filled = fill_shape_bm25(pool_result, sample_shape)

        for slot in filled.slots:
            assert slot.occupancy == 0

    def test_capacity_respected(self, sample_candidates, sample_shape):
        """Slots should never exceed capacity."""
        # Modify shape to have very small capacities.
        small_shape = AnswerShapeResult(
            slots=[
                AnswerSlot(
                    slot_id="slot_1",
                    slot_semantics="direct_answer",
                    capacity=1,
                    required=True,
                    priority=0,
                ),
            ]
        )

        pool_result = PoolResult(
            query="query",
            candidates=sample_candidates,
            pool_ms=150,
        )

        filled = fill_shape_bm25(pool_result, small_shape)

        assert filled.slots[0].occupancy == 1
        assert len(filled.slots[0].chunks) == 1

    def test_deterministic_order(self, sample_candidates, sample_shape):
        """Same inputs should produce same output (deterministic)."""
        pool_result = PoolResult(
            query="medical necessity",
            candidates=sample_candidates.copy(),
            pool_ms=150,
        )

        filled_1 = fill_shape_bm25(pool_result, sample_shape)
        filled_2 = fill_shape_bm25(pool_result, sample_shape)

        for slot_1, slot_2 in zip(filled_1.slots, filled_2.slots):
            assert [c.chunk_id for c in slot_1.chunks] == [
                c.chunk_id for c in slot_2.chunks
            ]

    def test_emit_diagnostics(self, sample_candidates, sample_shape):
        """Emit should contain diagnostic information."""
        pool_result = PoolResult(
            query="medical necessity",
            candidates=sample_candidates,
            pool_ms=150,
        )

        filled = fill_shape_bm25(pool_result, sample_shape)

        assert filled.filling_strategy == "bm25"
        assert "fillers_decision" in filled.emit
        assert filled.emit["fillers_decision"] == "bm25_rank_assign"
        assert "slots_filled" in filled.emit
        assert "empty_slots" in filled.emit
        assert "under_filled" in filled.emit
        assert "total_chunks_assigned" in filled.emit
        assert "per_slot_details" in filled.emit


class TestFillerAEdgeCases:
    """Edge case and characterization tests."""

    def test_reads_bm25_score_not_generic_score(self, sample_shape):
        """CRITICAL: Filler a must read bm25_score, not the generic score field.

        This catches the bug where code accidentally reads PoolCandidate.score
        (which holds arm-specific signals) instead of PoolCandidate.bm25_score
        (the BM25-specific ranking for Filler a).

        If this test fails, the sorting is reading the wrong field.
        """
        # Create candidates where .score and .bm25_score have OPPOSITE rankings.
        candidates = [
            PoolCandidate(
                chunk_id="highest_bm25",
                document_id="doc_1",
                text="Best by BM25",
                score=0.10,  # Low generic score
                bm25_score=0.95,  # High BM25 score (what we should sort by)
                source_arm="vector",
                is_neighbor=False,
                source_type="document",
            ),
            PoolCandidate(
                chunk_id="lowest_bm25",
                document_id="doc_2",
                text="Worst by BM25",
                score=0.90,  # High generic score
                bm25_score=0.15,  # Low BM25 score (what we should sort by)
                source_arm="vector",
                is_neighbor=False,
                source_type="document",
            ),
        ]

        pool_result = PoolResult(query="test", candidates=candidates, pool_ms=100)
        filled = fill_shape_bm25(pool_result, sample_shape)

        # If sorting is correct (by bm25_score), first chunk should be "highest_bm25".
        # If sorting is wrong (by score), first chunk would be "lowest_bm25".
        assert filled.slots[0].chunks[0].chunk_id == "highest_bm25", \
            "CRITICAL BUG: Filler a is reading PoolCandidate.score instead of .bm25_score"
        assert filled.slots[0].chunks[0].original_score == 0.95, \
            "original_score should reflect bm25_score (0.95), not generic score (0.10)"

    def test_mixed_scored_and_unscored(self, sample_shape):
        """Mixed scored and unscored candidates should filter out None bm25_scores."""
        candidates = [
            PoolCandidate(
                chunk_id="scored_1",
                document_id="doc_1",
                text="Scored",
                score=0.7,  # Generic score
                bm25_score=0.8,  # BM25 score (Filler a sorts by this)
                source_arm="vector",
                is_neighbor=False,
                source_type="document",
            ),
            PoolCandidate(
                chunk_id="unscored_1",
                document_id="doc_2",
                text="Unscored (neighbor)",
                score=None,  # No generic score
                bm25_score=None,  # No BM25 score
                source_arm="",
                is_neighbor=True,
                source_type="document",
            ),
            PoolCandidate(
                chunk_id="scored_2",
                document_id="doc_3",
                text="Scored again",
                score=0.5,
                bm25_score=0.6,
                source_arm="vector",
                is_neighbor=False,
                source_type="document",
            ),
        ]

        pool_result = PoolResult(query="test", candidates=candidates, pool_ms=100)
        filled = fill_shape_bm25(pool_result, sample_shape)

        # First slot should get scored_1 (0.8) and scored_2 (0.6), not the unscored neighbor.
        direct_slot = filled.slots[0]
        assert direct_slot.occupancy == 2
        assert direct_slot.chunks[0].chunk_id == "scored_1"
        assert direct_slot.chunks[0].original_score == 0.8
        assert direct_slot.chunks[1].chunk_id == "scored_2"
        assert direct_slot.chunks[1].original_score == 0.6

    def test_many_candidates_many_slots(self):
        """Large-scale test: many candidates, many slots."""
        candidates = [
            PoolCandidate(
                chunk_id=f"chunk_{i}",
                document_id=f"doc_{i % 10}",
                text=f"Content {i}",
                score=0.5 + (i * 0.001),  # Generic score (slowly increasing)
                bm25_score=1.0 - (i * 0.01),  # BM25 score (Filler a sorts by this, decreasing)
                source_arm="vector",
                is_neighbor=False,
                source_type="document",
            )
            for i in range(100)
        ]

        slots = [
            AnswerSlot(
                slot_id=f"slot_{i}",
                slot_semantics="direct_answer",
                capacity=5,
                required=True,
                priority=i,
            )
            for i in range(10)
        ]
        shape = AnswerShapeResult(slots=slots)

        pool_result = PoolResult(query="test", candidates=candidates, pool_ms=200)
        filled = fill_shape_bm25(pool_result, shape)

        # Should fill slots in order by bm25_score (descending).
        # slot_0 gets chunk_0-4 (bm25_score 1.0, 0.99, 0.98, 0.97, 0.96)
        # slot_1 gets chunk_5-9 (bm25_score 0.95, 0.94, 0.93, 0.92, 0.91)
        expected_chunks = 0
        for i, slot in enumerate(filled.slots):
            if expected_chunks < 100:
                expected_occupancy = min(5, 100 - expected_chunks)
                assert slot.occupancy == expected_occupancy
                expected_chunks += slot.occupancy
            else:
                assert slot.occupancy == 0

    def test_field_preservation(self, sample_candidates, sample_shape):
        """All PoolCandidate fields should be preserved in FilledChunk."""
        candidate = PoolCandidate(
            chunk_id="test_chunk",
            document_id="test_doc",
            text="Test text",
            score=0.7,  # Generic score
            bm25_score=0.9,  # BM25 score (what should be preserved in original_score)
            authority_level="contract_source_of_truth",  # Should be threaded through to FilledChunk
            source_arm="vector",
            is_neighbor=False,
            source_type="document",
            document_status="live",
            content_sha="sha123",
            tags={"d:domain": 1, "p:process": 2},
        )

        pool_result = PoolResult(query="test", candidates=[candidate], pool_ms=100)
        filled = fill_shape_bm25(pool_result, sample_shape)

        chunk = filled.slots[0].chunks[0]
        assert chunk.chunk_id == "test_chunk"
        assert chunk.document_id == "test_doc"
        assert chunk.text == "Test text"
        assert chunk.original_score == 0.9  # Should be bm25_score, not generic score
        assert chunk.is_neighbor is False
        assert chunk.source_type == "document"
        assert chunk.document_status == "live"
        assert chunk.content_sha == "sha123"
        assert chunk.tags == {"d:domain": 1, "p:process": 2}
        assert chunk.authority_level == "contract_source_of_truth"  # Verify authority_level is threaded through

    def test_multi_signal_reranking(self):
        """Verify Filler a composes BM25 + authority + tag_coverage + length signals."""
        from app.services.retriever.fillers.filler_a import _compute_rerank_score

        # Candidate A: high BM25 + good signals (should win)
        candidate_a = PoolCandidate(
            chunk_id="a",
            document_id="doc_a",
            text="Medical necessity policy statement for prior authorization requirements and coverage limitations.",
            bm25_score=0.9,  # High BM25
            authority_level="contract_source_of_truth",  # Max authority (1.0 weight)
            tags={"d:domain": 1, "p:process": 2, "j:jurisdiction": 3},  # Many tags
            source_arm="tag_select",
            is_neighbor=False,
            score=0.5,
        )

        # Candidate B: lower BM25 + worse signals (should lose)
        candidate_b = PoolCandidate(
            chunk_id="b",
            document_id="doc_b",
            text="Short text",  # Too short, penalized
            bm25_score=0.7,  # Lower BM25
            authority_level="fyi_not_citable",  # Low authority (0.20 weight)
            tags={"d:domain": 1},  # Few tags
            source_arm="vector",
            is_neighbor=False,
            score=0.5,
        )

        score_a = _compute_rerank_score(candidate_a, "test query")
        score_b = _compute_rerank_score(candidate_b, "test query")

        # A should rank higher (better BM25 + better signals)
        assert score_a > score_b, f"A ({score_a}) should > B ({score_b}) with better composite signals"

        # Verify they're both in valid [0, 1] range
        assert 0.0 <= score_a <= 1.0
        assert 0.0 <= score_b <= 1.0

        # Test secondary signals win tie-breaks when BM25 is very similar
        candidate_c = PoolCandidate(
            chunk_id="c",
            document_id="doc_c",
            text="A medium length document with some policy details.",
            bm25_score=0.75,  # Close to D
            authority_level="contract_source_of_truth",  # Max authority
            tags={"d:domain": 1, "p:process": 2},
            source_arm="tag_select",
            is_neighbor=False,
            score=0.5,
        )

        candidate_d = PoolCandidate(
            chunk_id="d",
            document_id="doc_d",
            text="Short",
            bm25_score=0.75,  # Same BM25 score
            authority_level=None,
            tags={},
            source_arm="vector",
            is_neighbor=False,
            score=0.5,
        )

        score_c = _compute_rerank_score(candidate_c, "test query")
        score_d = _compute_rerank_score(candidate_d, "test query")

        # C should rank higher on secondary signals when BM25 is equal
        assert score_c > score_d, f"C ({score_c}) should > D ({score_d}) when BM25 is equal but C has better authority/coverage"

    def test_meta_boost_signal(self):
        """Verify meta_boost signal lifts chunks containing Gate's required/boosted phrases."""
        from app.services.retriever.fillers.filler_a import _compute_meta_boost_score

        # Candidate with required phrases present
        candidate_with_phrase = PoolCandidate(
            chunk_id="with_phrase",
            document_id="doc1",
            text="Timely filing deadline for claims is 180 days from service date.",
            bm25_score=0.5,
            tags={"d:claims": 1},
            source_arm="tag_select",
            is_neighbor=False,
            score=0.5,
        )

        # Candidate without required phrases
        candidate_without_phrase = PoolCandidate(
            chunk_id="without_phrase",
            document_id="doc2",
            text="Provider must submit clean claims.",
            bm25_score=0.5,
            tags={"d:claims": 1},
            source_arm="tag_select",
            is_neighbor=False,
            score=0.5,
        )

        # Gate's phrases
        required_phrases = [("timely filing", 0.93), ("180 days", 0.85)]
        boosted_phrases = [("claims", 0.55)]

        score_with = _compute_meta_boost_score(
            candidate_with_phrase.text, candidate_with_phrase.tags,
            required_phrases, boosted_phrases
        )
        score_without = _compute_meta_boost_score(
            candidate_without_phrase.text, candidate_without_phrase.tags,
            required_phrases, boosted_phrases
        )

        # Chunk with required phrases should score higher
        assert score_with > score_without, f"Chunk with phrases ({score_with}) should > without ({score_without})"
        assert score_with > 0.0, "Chunk with phrases should score nonzero"
        assert 0.0 <= score_with <= 1.0, "Meta boost should be normalized [0, 1]"

    def test_meta_boost_tag_matching_with_underscores_dots(self):
        """Regression: ensure tag keys with underscores/dots match phrases correctly.

        Bug found 2026-07-23: tag key 'claims.timely_filing' should match phrase
        'timely filing' via normalized substring matching (convert dots/underscores to spaces).
        """
        from app.services.retriever.fillers.filler_a import _compute_meta_boost_score

        # Chunk with tag "claims.timely_filing" (dots/underscores)
        candidate_with_tag = PoolCandidate(
            chunk_id="with_timely_tag",
            document_id="doc1",
            text="General information about claims.",
            bm25_score=0.5,
            tags={"d:claims.timely_filing": 1, "d:claims.general": 1},
            source_arm="tag_select",
            is_neighbor=False,
            score=0.5,
        )

        # Gate's required phrases (should match the tag even though key has dots/underscores)
        required_phrases = [("timely filing", 0.93), ("filing deadline", 0.88)]
        boosted_phrases = []

        score = _compute_meta_boost_score(
            candidate_with_tag.text, candidate_with_tag.tags,
            required_phrases, boosted_phrases
        )

        # Should match "timely filing" in tag key "claims.timely_filing" after normalization
        assert score > 0.0, f"Should match normalized tag (got {score})"
        # The phrase "timely filing" (0.93 weight) should contribute; "filing deadline" (0.88) won't match
        # So score should be approximately 0.93 / (0.93 + 0.88) ≈ 0.514
        assert 0.5 < score < 0.6, f"Score should be ~0.51 for one-of-two phrases (got {score})"


class TestMetaBoostSpecificTermFallback:
    """Real bug fix (2026-08-15, live trace: "prior authorization criteria
    for Daraprim at AHCA Florida"). meta_boost's phrase pool comes
    exclusively from Gate's matched lexicon codes -- a rare, specific query
    term with zero lexicon representation (verified live via direct DB
    query against policy_lexicon_entries: 0 rows for "daraprim"/
    "pyrimethamine"/"toxoplasmosis") could never generate a phrase, so
    meta_boost scored 0 on the correct answer chunk regardless of content.
    Validated net-positive on the live 22-query cmhc bank via an offline
    weight sweep (zero regressions, +0.20 recall on cmhc005's H0015 case)
    before shipping -- see mobius-rag scratchpad sweep script/results,
    Retriever session 2026-08-15."""

    def test_capitalized_uncovered_term_gets_credit(self):
        from app.services.retriever.fillers.filler_a import _compute_meta_boost_score

        chunk_with_term = "Trial and failure of generic Pyrimethamine before Daraprim is required."
        chunk_without_term = "Standard toxoplasmosis treatment documentation is required."

        required_phrases = [("prior authorization", 0.93), ("florida", 0.85)]
        boosted_phrases = []
        query = "prior authorization criteria for Daraprim at AHCA Florida"

        score_with = _compute_meta_boost_score(
            chunk_with_term, {}, required_phrases, boosted_phrases, query
        )
        score_without = _compute_meta_boost_score(
            chunk_without_term, {}, required_phrases, boosted_phrases, query
        )
        assert score_with > score_without
        assert 0.0 <= score_with <= 1.0

    def test_already_lexicon_covered_term_not_double_counted(self):
        """A capitalized term already covered by an existing required/
        boosted phrase (e.g. a payer name) must not be treated as a NEW
        uncovered term -- would double-weight it and distort the fraction."""
        from app.services.retriever.fillers.filler_a import (
            _compute_meta_boost_score, _extract_uncovered_specific_terms,
        )

        required_phrases = [("sunshine health", 0.93)]
        boosted_phrases = []
        query = "How do I submit a corrected claim to Sunshine Health Florida?"

        specific = _extract_uncovered_specific_terms(query, required_phrases, boosted_phrases)
        assert "sunshine" not in specific
        assert "health" not in specific

    def test_sentence_initial_capitalization_not_treated_as_specific(self):
        from app.services.retriever.fillers.filler_a import _extract_uncovered_specific_terms

        specific = _extract_uncovered_specific_terms("What is the deadline?", [], [])
        assert specific == []

    def test_no_query_degrades_to_original_behavior(self):
        """Backward-compat: omitting `query` (default "") must behave
        identically to before this fix -- every existing caller/test that
        doesn't pass it stays byte-identical."""
        from app.services.retriever.fillers.filler_a import _compute_meta_boost_score

        required_phrases = [("timely filing", 0.93)]
        score = _compute_meta_boost_score(
            "Timely filing is 180 days.", {}, required_phrases, [],
        )
        assert score == 1.0

    def test_lowercase_specific_term_gets_no_fallback_credit(self):
        """Honest scope limit: the heuristic relies on capitalization, so
        an all-lowercase query gets no specific-term fallback -- documented
        limitation, not silently pretending to solve the general case."""
        from app.services.retriever.fillers.filler_a import _extract_uncovered_specific_terms

        specific = _extract_uncovered_specific_terms(
            "prior authorization criteria for daraprim at ahca florida", [], []
        )
        assert specific == []
