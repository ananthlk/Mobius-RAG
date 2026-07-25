"""Integration tests for Filler a with real Pool output.

These tests work with Pool's PoolCandidate.bm25_score field (distinct from .score).
Run with: pytest app/services/retriever/fillers/test_filler_a_integration.py -v
"""

import pytest
from app.services.retriever.fillers.filler_a import fill_shape_bm25
from app.services.retriever.pool.contracts import PoolCandidate, PoolResult
from app.services.retriever.shape.slots import AnswerShapeResult, AnswerSlot


@pytest.fixture
def real_pool_result():
    """
    Simulates a real PoolResult from Pool (Step 2).

    CRITICAL: Uses PoolCandidate.bm25_score (what Filler a sorts by), distinct from
    PoolCandidate.score (arm-specific signals like vector similarity or tag-coverage).
    """
    return PoolResult(
        query="prior authorization requirements",
        candidates=[
            # tag_select arm candidates
            PoolCandidate(
                chunk_id="chunk_tag_1",
                document_id="doc_ahca_001",
                text="AHCA prior authorization policy section 1: all inpatient stays require authorization.",
                score=0.88,  # Tag-coverage signal
                bm25_score=0.92,  # BM25 ranking (Filler a sorts by this)
                source_arm="tag_select",
                is_neighbor=False,
                source_type="document",
                document_status="live",
                tags={"d:utilization_management": 1, "p:prior_authorization": 2, "j:regulatory_authority.ahca": 1},
            ),
            PoolCandidate(
                chunk_id="chunk_tag_2",
                document_id="doc_ahca_001",
                text="AHCA prior authorization policy section 2: exceptions for emergency services.",
                score=0.85,  # Tag-coverage signal
                bm25_score=0.88,
                source_arm="tag_select",
                is_neighbor=False,
                source_type="document",
                document_status="live",
                tags={"d:utilization_management": 1, "p:prior_authorization": 2, "j:regulatory_authority.ahca": 1},
            ),
            # vector search candidates (semantic matches)
            PoolCandidate(
                chunk_id="chunk_vec_1",
                document_id="doc_sunshine_042",
                text="Sunshine Health authorization request process for elective procedures.",
                score=0.82,  # Vector similarity
                bm25_score=0.76,
                source_arm="vector",
                is_neighbor=False,
                source_type="document",
                document_status="live",
                tags={"j:payor.sunshine_health": 1, "p:prior_authorization": 1},
            ),
            # inherited arm candidate (authority-derived)
            PoolCandidate(
                chunk_id="chunk_inherited_1",
                document_id="doc_dod_tricare",
                text="TRICARE authorization thresholds for behavioral health services.",
                score=0.60,  # Inherited authority signal
                bm25_score=0.65,
                source_arm="inherited",
                is_neighbor=False,
                source_type="document",
                document_status="live",
                tags={"j:regulatory_authority.dod": 1},
            ),
            # Neighbors (no scores, from neighbor-assembly step)
            PoolCandidate(
                chunk_id="chunk_neighbor_1",
                document_id="doc_ahca_001",
                text="AHCA prior authorization policy section 1.5: appeal process overview.",
                score=None,  # Neighbor: no generic score
                bm25_score=None,  # Neighbor: no BM25 score (positional adjacency only)
                source_arm="",
                is_neighbor=True,
                source_type="document",
                document_status="live",
            ),
        ],
        segment_ms={
            "doc_narrow_ms": 42,
            "tag_select_ms": 118,
            "embed_ms": 890,
            "vector_ms": 156,
            "inherited_ms": 64,
            "neighbor_ms": 23,
            "dedup_ms": 8,
        },
        strategy_hint="tag_select+vector+inherited+neighbors",
        fallback_triggered=False,
        pool_ms=1301,
    )


@pytest.fixture
def answer_shape_for_policy_query():
    """
    Simulates AnswerShapeResult for a policy-oriented query.

    Shape already determined: DIRECT_ANSWER + FAN_OUT(2 themes) pattern.
    """
    return AnswerShapeResult(
        slots=[
            AnswerSlot(
                slot_id="direct_answer",
                slot_semantics="direct_answer",
                capacity=2,
                required=True,
                priority=0,
                rewritten_query="",
            ),
            AnswerSlot(
                slot_id="fanout_regulatory",
                slot_semantics="thematic_exploration",
                capacity=2,
                required=True,
                priority=1,
                rewritten_query="",
            ),
            AnswerSlot(
                slot_id="external_context",
                slot_semantics="external_context",
                capacity=1,
                required=False,
                priority=2,
                rewritten_query="",
            ),
        ]
    )


class TestFillerAIntegration:
    """Integration tests: Filler a + Pool + Shape."""

    def test_fill_realistic_pool_output(self, real_pool_result, answer_shape_for_policy_query):
        """Filler a should rank realistic Pool output by bm25_score and fill all slots."""
        filled = fill_shape_bm25(real_pool_result, answer_shape_for_policy_query)

        # direct_answer: should get the two highest bm25_score chunks
        direct_slot = filled.slots[0]
        assert direct_slot.occupancy == 2
        assert direct_slot.chunks[0].chunk_id == "chunk_tag_1"  # bm25_score 0.92 (highest)
        assert direct_slot.chunks[0].original_score == 0.92  # Verify it's the bm25_score
        assert direct_slot.chunks[1].chunk_id == "chunk_tag_2"  # bm25_score 0.88 (second)
        assert direct_slot.chunks[1].original_score == 0.88

        # fanout_regulatory: should get next-best (vector + inherited, by bm25_score)
        fanout_slot = filled.slots[1]
        assert fanout_slot.occupancy == 2
        assert fanout_slot.chunks[0].chunk_id == "chunk_vec_1"  # bm25_score 0.76
        assert fanout_slot.chunks[0].original_score == 0.76
        assert fanout_slot.chunks[1].chunk_id == "chunk_inherited_1"  # bm25_score 0.65
        assert fanout_slot.chunks[1].original_score == 0.65

        # external_context: no more candidates (neighbor is filtered out)
        external_slot = filled.slots[2]
        assert external_slot.occupancy == 0
        assert external_slot.under_filled

    def test_diagnostics_from_pool(self, real_pool_result, answer_shape_for_policy_query):
        """Emit should surface Pool's segment_ms and strategy_hint."""
        filled = fill_shape_bm25(real_pool_result, answer_shape_for_policy_query)

        # v1: Fillers doesn't re-emit segment_ms; Router/Observer might.
        # Just verify Filler's own diagnostics are present.
        assert filled.emit["total_chunks_assigned"] == 4
        assert filled.emit["under_filled"] == 1  # external_context
        assert filled.emit["slots_filled"] == 2  # direct + fanout

    def test_passthrough_fields_intact(self, real_pool_result, answer_shape_for_policy_query):
        """All Pool fields should survive: document_status, tags, content_sha, etc."""
        filled = fill_shape_bm25(real_pool_result, answer_shape_for_policy_query)

        # Check the first assigned chunk retains all fields.
        chunk = filled.slots[0].chunks[0]
        assert chunk.document_status == "live"
        assert "d:utilization_management" in chunk.tags
        assert "p:prior_authorization" in chunk.tags
        assert "j:regulatory_authority.ahca" in chunk.tags
        assert chunk.source_type == "document"
        # CRITICAL: original_score should be bm25_score, not generic score
        assert chunk.original_score == 0.92  # bm25_score, not the generic score (0.88)

    def test_neighbors_filtered_by_none_score(self, real_pool_result, answer_shape_for_policy_query):
        """Neighbors with score=None should not be assigned (filtered before ranking)."""
        filled = fill_shape_bm25(real_pool_result, answer_shape_for_policy_query)

        # Walk all filled chunks; none should be the neighbor chunk (score=None).
        all_chunk_ids = []
        for slot in filled.slots:
            all_chunk_ids.extend([c.chunk_id for c in slot.chunks])

        assert "chunk_neighbor_1" not in all_chunk_ids  # Neighbor filtered out


class TestFillerAEvalReadiness:
    """Setup for Eval's NUMBER-MOVING calibration (cmhc 26-query bank).

    Once Eval is ready to grade Filler a, these structures prepare the harness.
    Eval will inject queries → shape → pool → filler_a and measure occupancy/coverage.
    """

    @pytest.mark.skip(reason="Awaiting Eval calibration setup (post-sign-off)")
    def test_eval_cmhc_26_query_bank(self):
        """
        Eval will call this with 26 CMHC queries from the standard bank.

        Metrics to track per query:
        - per-slot occupancy (actual filled / capacity requested)
        - under_fill_count (slots that didn't reach capacity)
        - total_coverage (% of Pool candidates that got assigned)
        - per-arm contribution (which arms filled which slots)

        Baseline: compare against Pool's solo performance (all arms unioned, unranked).
        """
        pass

    @pytest.mark.skip(reason="Awaiting Eval calibration setup (post-sign-off)")
    def test_eval_before_after_pool_bm25_field(self):
        """
        Eval verifies that adding ts_rank_cd to Pool didn't regress Pool's recall.

        Filler a's job is precision (top-N from Pool), not recall (Pool's job).
        But we want to confirm Pool's BM25 addition doesn't hurt upstream metrics.
        """
        pass
