"""Unit tests for Observer (Step 4e -- per-slot "would this rung benefit
from another turn?" verdicts feeding Router's decide_continuation()).
"""

from app.services.retriever import observer
from app.services.retriever.fillers.contracts import FilledChunk, FilledSlot
from app.services.router.continuation import (
    VERDICT_EXHAUSTED_ATTEMPTS,
    VERDICT_SATISFIED,
    VERDICT_WOULD_BENEFIT,
)


def _slot(*, occupancy, capacity, chunks=None, required=True):
    chunks = chunks if chunks is not None else [
        FilledChunk(chunk_id=f"c{i}") for i in range(occupancy)
    ]
    return FilledSlot(
        slot_id="s1", slot_semantics="direct_answer", capacity=capacity,
        required=required, chunks=chunks, occupancy=occupancy,
        under_filled=occupancy < capacity, over_filled=False,
    )


def _chunk(assignment_reason="score_rank"):
    return FilledChunk(chunk_id="c", assignment_reason=assignment_reason)


def _llm_chunk(assignment_reason, quote_verified):
    return FilledChunk(
        chunk_id="c", assignment_reason=assignment_reason, quote_verified=quote_verified,
    )


def _scored_chunk(score):
    return FilledChunk(chunk_id="c", original_score=score)


class TestDispatch:
    def test_unknown_strategy_is_exhausted(self):
        verdict, reason = observer.evaluate("z", _slot(occupancy=1, capacity=1))
        assert verdict == VERDICT_EXHAUSTED_ATTEMPTS
        assert "no_handler" in reason


class TestDeterminism:
    def test_every_handler_strategy_has_a_determinism_entry(self):
        assert set(observer._HANDLERS) == set(observer._IS_DETERMINISTIC)

    def test_deterministic_strategies(self):
        for strategy in ("a", "b", "s"):
            assert observer.is_deterministic(strategy) is True

    def test_non_deterministic_strategies(self):
        for strategy in ("c", "d"):
            assert observer.is_deterministic(strategy) is False

    def test_unknown_strategy_defaults_to_deterministic(self):
        assert observer.is_deterministic("z") is True


class TestFactStore:
    def test_hit_is_never_satisfied_alone_even_at_capacity_one(self):
        # Eval's sharper rule (2026-07-24, round 2): s never independently
        # satisfies a slot, regardless of capacity -- capacity==1 is not a
        # real signal that s "owns" a dedicated slot (no such slot type
        # exists in the AnswerSlot contract), so even a capacity=1 hit
        # still supplements rather than substitutes.
        verdict, reason = observer.evaluate(
            "s", _slot(occupancy=1, capacity=1), attempt_number=1, max_attempts=2,
        )
        assert verdict == VERDICT_WOULD_BENEFIT
        assert reason == "fact_store_hit_supplementary_not_sole_answer"

    def test_hit_on_a_larger_slot_would_benefit_not_satisfied(self):
        # BUG FIX regression test (2026-07-24, real live evidence via
        # Retriever's calibration harness, cmhc001): s shares the SAME
        # direct_answer slot as a/b/c/d -- capacity reflects the query's
        # whole needed breadth, not a dedicated capacity-1 fact slot. A
        # single fact-store hit against capacity=10 must NOT short-circuit
        # the loop -- it's real but supplementary evidence, the next rung
        # must still deploy to fill the rest.
        verdict, reason = observer.evaluate(
            "s", _slot(occupancy=1, capacity=10), attempt_number=1, max_attempts=2,
        )
        assert verdict == VERDICT_WOULD_BENEFIT
        assert reason == "fact_store_hit_supplementary_not_sole_answer"

    def test_hit_exhausted_once_attempts_used_up(self):
        verdict, reason = observer.evaluate(
            "s", _slot(occupancy=1, capacity=10), attempt_number=2, max_attempts=2,
        )
        assert verdict == VERDICT_EXHAUSTED_ATTEMPTS
        assert reason == "fact_store_hit_supplementary_not_sole_answer_attempts_exhausted"

    def test_miss_would_benefit_when_attempts_remain(self):
        verdict, reason = observer.evaluate(
            "s", _slot(occupancy=0, capacity=1), attempt_number=1, max_attempts=2,
        )
        assert verdict == VERDICT_WOULD_BENEFIT
        assert "miss" in reason

    def test_miss_exhausted_when_attempts_used_up(self):
        verdict, reason = observer.evaluate(
            "s", _slot(occupancy=0, capacity=1), attempt_number=2, max_attempts=2,
        )
        assert verdict == VERDICT_EXHAUSTED_ATTEMPTS
        assert reason.endswith("attempts_exhausted")


class TestBm25AndVector:
    def test_filled_to_capacity_is_satisfied(self):
        for strategy in ("a", "b"):
            verdict, _ = observer.evaluate(strategy, _slot(occupancy=3, capacity=3))
            assert verdict == VERDICT_SATISFIED

    def test_under_filled_would_benefit(self):
        for strategy in ("a", "b"):
            verdict, reason = observer.evaluate(
                strategy, _slot(occupancy=1, capacity=3), attempt_number=1, max_attempts=3,
            )
            assert verdict == VERDICT_WOULD_BENEFIT
            assert "under_filled_1_of_3" in reason

    def test_empty_is_would_benefit_not_exhausted_when_attempts_remain(self):
        # Deterministic rerank of a fixed Pool -- but WOULD_BENEFIT here means
        # "deploy the slot's next planned rung", not "retry this same one".
        verdict, _ = observer.evaluate(
            "a", _slot(occupancy=0, capacity=3), attempt_number=1, max_attempts=2,
        )
        assert verdict == VERDICT_WOULD_BENEFIT

    def test_under_filled_exhausted_once_attempts_used_up(self):
        verdict, reason = observer.evaluate(
            "b", _slot(occupancy=0, capacity=3), attempt_number=2, max_attempts=2,
        )
        assert verdict == VERDICT_EXHAUSTED_ATTEMPTS
        assert reason.endswith("attempts_exhausted")


class TestLlmRetrieval:
    def test_all_verified_and_full_is_satisfied(self):
        chunks = [
            _llm_chunk("llm_retrieved", quote_verified=True),
            _llm_chunk("llm_retrieved_external", quote_verified=True),
        ]
        slot = _slot(occupancy=2, capacity=2, chunks=chunks)
        verdict, reason = observer.evaluate("c", slot)
        assert verdict == VERDICT_SATISFIED
        assert reason == "all_citations_verified_filled_to_capacity"

    def test_hallucinated_quote_would_benefit(self):
        chunks = [
            _llm_chunk("llm_retrieved", quote_verified=True),
            _llm_chunk("llm_partial_match", quote_verified=False),
        ]
        slot = _slot(occupancy=2, capacity=2, chunks=chunks)
        verdict, reason = observer.evaluate("c", slot, attempt_number=1, max_attempts=2)
        assert verdict == VERDICT_WOULD_BENEFIT
        assert reason == "1_of_2_citations_unverified"

    def test_quote_less_citation_is_not_treated_as_verified(self):
        # BUG FIX regression test (2026-07-24): a citation with NO quote at
        # all gets assignment_reason="llm_retrieved" (nothing to falsify)
        # but quote_verified=None (nothing was confirmed either) -- must
        # NOT read as verified just because it isn't the "llm_partial_match"
        # reason string.
        chunks = [_llm_chunk("llm_retrieved", quote_verified=None)]
        slot = _slot(occupancy=1, capacity=1, chunks=chunks)
        verdict, reason = observer.evaluate("c", slot, attempt_number=1, max_attempts=2)
        assert verdict == VERDICT_WOULD_BENEFIT
        assert reason == "1_of_1_citations_unverified"

    def test_verified_but_under_filled_would_benefit(self):
        chunks = [_llm_chunk("llm_retrieved", quote_verified=True)]
        slot = _slot(occupancy=1, capacity=3, chunks=chunks)
        verdict, reason = observer.evaluate("c", slot, attempt_number=1, max_attempts=2)
        assert verdict == VERDICT_WOULD_BENEFIT
        assert "under_filled_1_of_3_verified" == reason

    def test_empty_would_benefit(self):
        verdict, reason = observer.evaluate(
            "c", _slot(occupancy=0, capacity=2), attempt_number=1, max_attempts=2,
        )
        assert verdict == VERDICT_WOULD_BENEFIT
        assert reason == "llm_retrieval_empty"

    def test_unverified_exhausted_once_attempts_used_up(self):
        chunks = [_llm_chunk("llm_partial_match", quote_verified=False)]
        slot = _slot(occupancy=1, capacity=1, chunks=chunks)
        verdict, reason = observer.evaluate("c", slot, attempt_number=2, max_attempts=2)
        assert verdict == VERDICT_EXHAUSTED_ATTEMPTS
        assert reason == "1_of_1_citations_unverified_attempts_exhausted"

    def test_non_llm_chunk_with_no_assignment_reason_never_counts_as_unverified(self):
        # A slot filled by some other path (e.g. neighbor-completion) with
        # an assignment_reason outside the LLM-citation set must never be
        # swept into the unverified count just because quote_verified
        # defaults to None -- the scoping check (assignment_reason in
        # _LLM_CITATION_REASONS) must gate this, not quote_verified alone.
        chunks = [_llm_chunk("llm_retrieved", quote_verified=True), _chunk("score_rank")]
        slot = _slot(occupancy=2, capacity=2, chunks=chunks)
        verdict, reason = observer.evaluate("c", slot)
        assert verdict == VERDICT_SATISFIED
        assert reason == "all_citations_verified_filled_to_capacity"


class TestWebSearch:
    def test_under_filled_would_benefit_regardless_of_scores(self):
        chunks = [_scored_chunk(0.9)]
        verdict, reason = observer.evaluate(
            "d", _slot(occupancy=1, capacity=2, chunks=chunks), attempt_number=1, max_attempts=2,
        )
        assert verdict == VERDICT_WOULD_BENEFIT
        assert "under_filled" in reason

    def test_filled_to_capacity_uniform_scores_is_satisfied(self):
        chunks = [_scored_chunk(0.7), _scored_chunk(0.65)]
        verdict, reason = observer.evaluate("d", _slot(occupancy=2, capacity=2, chunks=chunks))
        assert verdict == VERDICT_SATISFIED
        assert reason == "web_search_filled_to_capacity_no_decay_floor_violation"

    def test_filled_to_capacity_with_decay_floor_violation_would_benefit(self):
        # top=0.8, floor=0.6*0.8=0.48 -- 0.1 is well below the floor, same
        # "nominally full but padded with junk near the bottom" shape as
        # filler_b's own reproduced case.
        chunks = [_scored_chunk(0.8), _scored_chunk(0.1)]
        verdict, reason = observer.evaluate(
            "d", _slot(occupancy=2, capacity=2, chunks=chunks), attempt_number=1, max_attempts=2,
        )
        assert verdict == VERDICT_WOULD_BENEFIT
        assert reason == "web_search_1_of_2_chunks_below_decay_floor"

    def test_missing_score_data_fails_open_to_satisfied(self):
        chunks = [FilledChunk(chunk_id="c1"), FilledChunk(chunk_id="c2")]
        verdict, reason = observer.evaluate("d", _slot(occupancy=2, capacity=2, chunks=chunks))
        assert verdict == VERDICT_SATISFIED
        assert reason == "web_search_filled_to_capacity_no_score_data"

    def test_decay_floor_violation_exhausted_once_attempts_used_up(self):
        chunks = [_scored_chunk(0.8), _scored_chunk(0.1)]
        verdict, reason = observer.evaluate(
            "d", _slot(occupancy=2, capacity=2, chunks=chunks), attempt_number=2, max_attempts=2,
        )
        assert verdict == VERDICT_EXHAUSTED_ATTEMPTS
        assert reason.endswith("attempts_exhausted")


class TestEveryVerdictHasAReason:
    def test_no_empty_reason_across_all_strategies_and_paths(self):
        cases = [
            ("s", _slot(occupancy=0, capacity=1)),
            ("s", _slot(occupancy=1, capacity=1)),
            ("a", _slot(occupancy=0, capacity=2)),
            ("a", _slot(occupancy=2, capacity=2)),
            ("b", _slot(occupancy=0, capacity=2)),
            ("b", _slot(occupancy=2, capacity=2)),
            ("c", _slot(occupancy=0, capacity=2)),
            ("c", _slot(occupancy=2, capacity=2, chunks=[_chunk("llm_retrieved")] * 2)),
            ("d", _slot(occupancy=0, capacity=2)),
            ("d", _slot(occupancy=2, capacity=2)),
        ]
        for strategy, slot in cases:
            verdict, reason = observer.evaluate(strategy, slot, attempt_number=1, max_attempts=3)
            assert verdict in (VERDICT_WOULD_BENEFIT, VERDICT_SATISFIED, VERDICT_EXHAUSTED_ATTEMPTS)
            assert reason  # Eval's non-negotiable ask -- always a reason string
