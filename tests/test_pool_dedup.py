"""Tests for Pool's union/dedup logic (app/services/retriever/pool/dedup.py).

Pure unit tests only -- dedup_candidates() is pure compute, zero DB calls,
same pattern test_shape_structure.py established for Structure. This is
Pool's actual novel contribution (target-structure-spec.md S1: legacy
strategies never unioned), so it earns real coverage, not just a
characterization test.
"""

from __future__ import annotations

from app.services.retriever.pool.contracts import PoolCandidate
from app.services.retriever.pool.dedup import dedup_candidates


def _cand(chunk_id: str, source_arm: str = "vector", content_sha: str | None = None, text: str = "") -> PoolCandidate:
    return PoolCandidate(
        chunk_id=chunk_id,
        document_id="doc1",
        text=text,
        is_neighbor=False,
        source_arm=source_arm,
        score=0.5,
        content_sha=content_sha,
    )


class TestIdDedup:
    def test_same_chunk_id_from_two_strategies_kept_once(self):
        a = _cand("c1", source_arm="tag_select")
        b = _cand("c1", source_arm="vector")
        out = dedup_candidates([a, b])
        assert len(out) == 1
        assert out[0].source_arm == "tag_select"  # first wins

    def test_distinct_ids_all_kept(self):
        out = dedup_candidates([_cand("c1"), _cand("c2"), _cand("c3")])
        assert {c.chunk_id for c in out} == {"c1", "c2", "c3"}

    def test_empty_input_gives_empty_output(self):
        assert dedup_candidates([]) == []


class TestContentDedup:
    def test_same_content_sha_different_ids_deduped(self):
        a = _cand("c1", content_sha="sha-abc")
        b = _cand("c2", content_sha="sha-abc")
        out = dedup_candidates([a, b])
        assert len(out) == 1
        assert out[0].chunk_id == "c1"  # first wins

    def test_different_content_sha_both_kept(self):
        a = _cand("c1", content_sha="sha-abc")
        b = _cand("c2", content_sha="sha-xyz")
        out = dedup_candidates([a, b])
        assert len(out) == 2

    def test_missing_content_sha_falls_back_to_normalized_body(self):
        a = _cand("c1", text="  The Provider   must confirm eligibility  ")
        b = _cand("c2", text="the provider must confirm eligibility")
        out = dedup_candidates([a, b])
        assert len(out) == 1

    def test_two_empty_text_candidates_not_falsely_deduped(self):
        # Empty content key ("") must never match another empty content key --
        # two genuinely distinct near-empty chunks shouldn't collapse into one.
        a = _cand("c1", text="")
        b = _cand("c2", text="")
        out = dedup_candidates([a, b])
        assert len(out) == 2


class TestOrderPreservation:
    def test_preserves_input_order(self):
        out = dedup_candidates([_cand("c3"), _cand("c1"), _cand("c2")])
        assert [c.chunk_id for c in out] == ["c3", "c1", "c2"]

    def test_strategy_precedence_is_caller_controlled(self):
        # dedup_candidates itself has no strategy preference -- whichever
        # list position comes first wins. Ordering the union
        # (tag_select + vector + inherited) is the caller's (pool.py's) job.
        tag_first = dedup_candidates([_cand("c1", source_arm="tag_select"), _cand("c1", source_arm="vector")])
        vector_first = dedup_candidates([_cand("c1", source_arm="vector"), _cand("c1", source_arm="tag_select")])
        assert tag_first[0].source_arm == "tag_select"
        assert vector_first[0].source_arm == "vector"
