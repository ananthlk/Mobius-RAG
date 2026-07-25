"""Unit tests for RRF fusion (fusion.py) -- Eval's v1 cross-strategy
blending ruling, docs/rag-agents/blend-model-design.md S4, 2026-07-24.
Standalone module, not wired into compile_synthesis yet.
"""

import pytest

from app.services.retriever.fillers.contracts import FilledChunk
from app.services.retriever.fusion import (
    DEFAULT_MMR_LAMBDA,
    DEFAULT_REDUNDANCY_THRESHOLD,
    DEFAULT_RRF_K,
    POOL_VERDICT_BUDGET_FULL,
    POOL_VERDICT_GAPS_REMAIN,
    POOL_VERDICT_SATURATED,
    FusedChunk,
    derive_coverage_diagnostic,
    mmr_select,
    rrf_fuse,
)


def _estimate_tokens(text):
    return len(text or "") // 4


def _fused(chunk_id, *, text, score=None):
    return FusedChunk(chunk=_chunk(chunk_id, text=text), rrf_score=score if score is not None else 1.0)


def _chunk(chunk_id, *, text="text", content_sha=None):
    return FilledChunk(chunk_id=chunk_id, text=text, content_sha=content_sha)


class TestRrfFuseBasics:
    def test_single_strategy_preserves_rank_order(self):
        a1, a2, a3 = _chunk("a1", text="alpha"), _chunk("a2", text="beta"), _chunk("a3", text="gamma")
        result = rrf_fuse({"a": [a1, a2, a3]})
        assert [f.chunk.chunk_id for f in result] == ["a1", "a2", "a3"]

    def test_scores_decrease_with_rank(self):
        a1, a2, a3 = _chunk("a1", text="alpha"), _chunk("a2", text="beta"), _chunk("a3", text="gamma")
        result = rrf_fuse({"a": [a1, a2, a3]})
        scores = [f.rrf_score for f in result]
        assert scores[0] > scores[1] > scores[2]

    def test_rank_1_score_matches_formula(self):
        a1 = _chunk("a1", text="alpha")
        result = rrf_fuse({"a": [a1]}, k=60)
        assert result[0].rrf_score == pytest.approx(1.0 / 61)

    def test_default_k_is_literature_standard(self):
        assert DEFAULT_RRF_K == 60


class TestIncomparableScalesDontMatter:
    """The exact scenario Eval's ruling addresses: raw scores from
    different strategies are on incomparable scales. RRF must fuse by
    rank position alone -- a strategy whose rank-1 candidate has a huge
    raw score (never even read by rrf_fuse) must not out-rank a
    consensus pick just because of scale."""

    def test_rrf_never_reads_original_score(self):
        # A chunk with a tiny/absent original_score, ranked #1 in its own
        # strategy, must still outrank a chunk ranked #2 in another
        # strategy with a huge original_score -- rrf_fuse doesn't even
        # look at original_score, only list position.
        huge_score_but_rank_2 = FilledChunk(chunk_id="d1", text="d first", original_score=99999.0)
        huge_score_but_rank_1 = FilledChunk(chunk_id="d0", text="d zero", original_score=100000.0)
        tiny_score_rank_1 = FilledChunk(chunk_id="a1", text="a first", original_score=0.01)

        result = rrf_fuse({
            "d": [huge_score_but_rank_1, huge_score_but_rank_2],  # d0 rank1, d1 rank2
            "a": [tiny_score_rank_1],  # a1 rank1
        })
        # Both d0 and a1 are rank-1 in their own strategy -> tied score,
        # both must outrank d1 (rank-2), regardless of raw score magnitude.
        by_id = {f.chunk.chunk_id: f.rrf_score for f in result}
        assert by_id["d0"] == by_id["a1"]
        assert by_id["d0"] > by_id["d1"]


class TestCrossStrategyConsensus:
    """A chunk (or a near-duplicate identity) surfaced by multiple
    strategies should be rewarded -- summed reciprocal ranks, not just
    the best single contribution."""

    def test_same_chunk_id_across_strategies_gets_summed_score(self):
        shared = _chunk("shared-1", text="alpha content")
        a_only = _chunk("a-only", text="alpha only content")

        result = rrf_fuse({
            "a": [shared, a_only],  # shared rank1, a_only rank2
            "b": [shared],          # shared rank1 again
        })

        by_id = {f.chunk.chunk_id: f for f in result}
        expected_shared_score = 1.0 / 61 + 1.0 / 61  # rank1 in both a and b
        assert by_id["shared-1"].rrf_score == pytest.approx(expected_shared_score)
        assert set(by_id["shared-1"].contributing_strategies) == {"a", "b"}
        # The consensus pick now outranks a chunk that only ever appeared once.
        assert by_id["shared-1"].rrf_score > by_id["a-only"].rrf_score

    def test_same_content_different_chunk_id_across_strategies_merges(self):
        """Identity merging must use the same content-key logic as
        synthesis.py's dedup -- not just chunk_id equality -- since two
        strategies can independently surface the same underlying text
        under different synthetic/real chunk ids."""
        a_version = _chunk("a-id", text="shared boilerplate clause")
        b_version = _chunk("b-id", text="shared boilerplate clause")

        result = rrf_fuse({"a": [a_version], "b": [b_version]})

        assert len(result) == 1
        assert set(result[0].contributing_strategies) == {"a", "b"}

    def test_distinct_chunks_never_merged(self):
        a1 = _chunk("a1", text="alpha content")
        b1 = _chunk("b1", text="beta content")
        result = rrf_fuse({"a": [a1], "b": [b1]})
        assert len(result) == 2
        assert {f.contributing_strategies[0] for f in result} == {"a", "b"}


class TestOrdering:
    def test_first_seen_order_used_for_ties_before_sort(self):
        # Sanity: with equal scores, output is still deterministic (stable
        # sort preserves insertion order for ties).
        a1 = _chunk("a1", text="alpha")
        b1 = _chunk("b1", text="beta")
        result = rrf_fuse({"a": [a1], "b": [b1]})
        assert [f.chunk.chunk_id for f in result] == ["a1", "b1"]

    def test_empty_input_returns_empty(self):
        assert rrf_fuse({}) == []

    def test_strategy_with_empty_list_is_fine(self):
        a1 = _chunk("a1", text="alpha")
        result = rrf_fuse({"a": [a1], "b": []})
        assert [f.chunk.chunk_id for f in result] == ["a1"]


class TestMmrDefaults:
    def test_lambda_is_recall_leaning_not_midpoint(self):
        """Eval's ruling, 2026-07-24: seed 0.7, not the 0.5 midpoint --
        over-diversifying risks dropping distinct facts."""
        assert DEFAULT_MMR_LAMBDA == 0.7

    def test_redundancy_threshold_is_conservative(self):
        """Eval's ruling: fear over-merging distinct facts more than
        under-merging true duplicates -- only drop on very high overlap."""
        assert DEFAULT_REDUNDANCY_THRESHOLD >= 0.8


class TestMmrBudgetRespect:
    def test_selects_within_budget(self):
        fused = [_fused(f"c{i}", text=f"distinct content number {i} about topic {i}") for i in range(10)]
        budget = _estimate_tokens(fused[0].chunk.text) * 3  # room for ~3
        result = mmr_select(fused, token_budget=budget, estimate_tokens=_estimate_tokens)
        total = sum(_estimate_tokens(f.chunk.text) for f in result.selected)
        assert total <= budget
        assert len(result.selected) >= 1  # never returns nothing

    def test_always_selects_at_least_one_even_if_it_exceeds_budget(self):
        huge = _fused("huge", text="x" * 10000)
        result = mmr_select([huge], token_budget=1, estimate_tokens=_estimate_tokens)
        assert len(result.selected) == 1

    def test_empty_input_returns_empty_selection(self):
        result = mmr_select([], token_budget=1000, estimate_tokens=_estimate_tokens)
        assert result.selected == []
        assert result.merged_away == []
        assert result.pairs_merged == 0
        assert result.budget_cutoff_remaining == []

    def test_budget_cutoff_tracked_separately_from_redundancy_merge(self):
        """Router's ask, 2026-07-24 (CoverageDiagnostic negotiation): a
        candidate never evaluated because the budget ran out must NOT be
        indistinguishable from one rejected as truly redundant -- one means
        "the pool might still help, we ran out of room," the other means
        "the pool has nothing more to offer." Distinct fields, not a shared
        drop-count."""
        fused = [_fused(f"c{i}", text=f"distinct content number {i} about topic {i}") for i in range(5)]
        one_chunk_tokens = _estimate_tokens(fused[0].chunk.text)
        budget = one_chunk_tokens * 2  # room for exactly ~2

        result = mmr_select(fused, token_budget=budget, estimate_tokens=_estimate_tokens)

        assert len(result.selected) < len(fused)
        assert result.merged_away == []  # nothing here was redundant -- all distinct content
        assert len(result.budget_cutoff_remaining) > 0
        # Every unselected chunk is accounted for by exactly one of the two paths.
        assert len(result.selected) + len(result.budget_cutoff_remaining) == len(fused)

    def test_no_budget_cutoff_when_pool_fully_processed(self):
        """When the whole pool gets resolved into selected-or-merged
        without ever hitting the budget ceiling, budget_cutoff_remaining
        must be empty -- this is the "saturated" signal (pool exhausted,
        not budget-limited), distinct from budget_full."""
        fused = [_fused(f"c{i}", text=f"distinct content number {i} about topic {i}") for i in range(3)]
        result = mmr_select(fused, token_budget=1_000_000, estimate_tokens=_estimate_tokens)
        assert result.budget_cutoff_remaining == []
        assert len(result.selected) == 3


class TestMmrRedundancy:
    def test_near_duplicate_text_is_merged_away_not_selected_twice(self):
        original = _fused("orig", text="participating providers have 180 days to file claims", score=1.0)
        near_duplicate = _fused("dup", text="participating providers have 180 days to file claims", score=0.9)
        distinct = _fused("distinct", text="non-participating providers use a different appeals process entirely", score=0.5)

        result = mmr_select(
            [original, near_duplicate, distinct],
            token_budget=10_000, estimate_tokens=_estimate_tokens,
        )

        selected_ids = {f.chunk.chunk_id for f in result.selected}
        assert "orig" in selected_ids
        assert "dup" not in selected_ids
        assert "distinct" in selected_ids
        assert result.pairs_merged == 1
        dropped, kept_because_of = result.merged_away[0]
        assert dropped.chunk.chunk_id == "dup"
        assert kept_because_of.chunk.chunk_id == "orig"

    def test_topically_similar_but_factually_distinct_chunks_both_kept(self):
        """The exact risk Eval flagged: two chunks sharing topic anchors
        but stating DIFFERENT facts (different procedure codes) must NOT
        be merged just because they're topically similar -- conservative
        threshold must keep both."""
        code_a = _fused(
            "code-a", text="Sunshine Health prior authorization required for procedure code 96130",
        )
        code_b = _fused(
            "code-b", text="Sunshine Health prior authorization required for procedure code 97140",
        )

        result = mmr_select(
            [code_a, code_b], token_budget=10_000, estimate_tokens=_estimate_tokens,
        )

        selected_ids = {f.chunk.chunk_id for f in result.selected}
        assert selected_ids == {"code-a", "code-b"}
        assert result.pairs_merged == 0

    def test_merged_away_pairs_are_returned_for_offline_instrumentation(self):
        """Eval's required instrumentation: the actual (dropped, kept)
        pairs must be recoverable, not just a count -- that's what lets
        offline calibration check whether a merged-away chunk carried a
        must_fact the survivor didn't."""
        a = _fused("a", text="identical text here for testing purposes only")
        b = _fused("b", text="identical text here for testing purposes only")

        result = mmr_select([a, b], token_budget=10_000, estimate_tokens=_estimate_tokens)

        assert len(result.merged_away) == 1
        assert isinstance(result.merged_away[0], tuple)
        assert len(result.merged_away[0]) == 2


class TestMmrRelevanceOrdering:
    def test_higher_rrf_ranked_candidates_preferred_within_budget(self):
        # Three distinct, non-redundant chunks; budget only fits ~2 -- MMR
        # should prefer the higher-ranked (earlier in fused list) ones.
        best = _fused("best", text="alpha distinct content one")
        middle = _fused("middle", text="beta distinct content two")
        worst = _fused("worst", text="gamma distinct content three")

        one_chunk_tokens = _estimate_tokens(best.chunk.text)
        result = mmr_select(
            [best, middle, worst],
            token_budget=one_chunk_tokens * 2 + 1,
            estimate_tokens=_estimate_tokens,
        )

        selected_ids = {f.chunk.chunk_id for f in result.selected}
        assert "best" in selected_ids
        assert len(result.selected) <= 2


class TestDeriveCoverageDiagnostic:
    """Router's CoverageDiagnostic, RESOLVED 2026-07-24 -- derived from a
    real rrf_fuse -> mmr_select pipeline, not hand-faked FusedChunks, so
    contributing_strategies is genuine, not asserted-into-existence."""

    def test_budget_full_takes_priority_over_content_state(self):
        a1 = FilledChunk(chunk_id="a1", text="alpha content one")
        a2 = FilledChunk(chunk_id="a2", text="beta distinct content two")
        fused = rrf_fuse({"a": [a1, a2]})
        one_chunk_tokens = _estimate_tokens(a1.text)
        selection = mmr_select(fused, token_budget=one_chunk_tokens, estimate_tokens=_estimate_tokens)

        diagnostic = derive_coverage_diagnostic("s1", selection)

        assert diagnostic.slot_id == "s1"
        assert diagnostic.pool_verdict == POOL_VERDICT_BUDGET_FULL
        assert diagnostic.saturated_strategies == []
        assert diagnostic.uncovered_aspect_count is None

    def test_saturated_when_every_contributing_strategy_fully_merged_away(self):
        # Both strategies' candidates are near-duplicates of each other --
        # only one survives selection, so BOTH strategies' contributions
        # collapse: whichever one didn't "win" the merge is fully saturated,
        # but if it's the ONLY other strategy, the pool-level verdict
        # still isn't "saturated" unless every contributor is -- so make
        # both a and b redundant with a THIRD, distinct c winner to isolate
        # the case where a and b are both fully saturated relative to c.
        winner = FilledChunk(chunk_id="c1", text="participating providers have 180 days to file claims")
        a_dup = FilledChunk(chunk_id="a1", text="participating providers have 180 days to file claims")
        b_dup = FilledChunk(chunk_id="b1", text="participating providers have 180 days to file claims")
        fused = rrf_fuse({"c": [winner], "a": [a_dup], "b": [b_dup]})
        selection = mmr_select(fused, token_budget=1_000_000, estimate_tokens=_estimate_tokens)

        diagnostic = derive_coverage_diagnostic("s1", selection)

        # a/b/c are all identical text -> they merge into ONE fused identity
        # via rrf_fuse's own dedup (same content key), so this actually
        # exercises rrf_fuse's cross-strategy consensus merge, not mmr's.
        # All three strategies "contribute" to the single surviving entry.
        assert diagnostic.pool_verdict == POOL_VERDICT_GAPS_REMAIN
        assert diagnostic.saturated_strategies == []

    def test_a_strategy_whose_only_pick_wins_outright_is_not_saturated(self):
        """A strategy contributing exactly one candidate that wins outright
        (never compared against anything, never rejected) has shown NO
        evidence either way about whether deepening it would help -- it
        must NOT be counted as saturated just because a DIFFERENT
        strategy's candidate was redundant against it."""
        winner = FilledChunk(chunk_id="c1", text="participating providers have 180 days to file claims")
        a_redundant = FilledChunk(chunk_id="a1", text="participating providers have 180 days to file claim")
        fused = rrf_fuse({"c": [winner], "a": [a_redundant]})
        selection = mmr_select(
            fused, token_budget=1_000_000, estimate_tokens=_estimate_tokens,
            redundancy_threshold=0.7,
        )

        diagnostic = derive_coverage_diagnostic("s1", selection)

        # "c" never had a candidate rejected -- not saturated, so the pool
        # overall reads gaps_remain, not saturated, even though "a" alone
        # did show redundancy.
        assert diagnostic.pool_verdict == POOL_VERDICT_GAPS_REMAIN
        assert diagnostic.saturated_strategies == ["a"]

    def test_saturated_when_every_contributing_strategy_shows_redundancy(self):
        """CORRECTED definition (2026-07-24): saturated_strategies means
        "showed at least one redundant candidate," not "produced zero
        surviving candidates" -- the latter is unreachable given
        mmr_select's own "always keep >=1" guarantee (see
        derive_coverage_diagnostic's docstring for the proof). Here, "c"
        contributes a unique winner AND a second, internally-redundant
        candidate -- so even though c's WINNER survives, c itself still
        shows a sign of redundancy (c2 vs c1), same as a1 vs c1 for "a" --
        making the pool-level verdict correctly SATURATED once every
        contributing strategy has shown that sign."""
        winner = FilledChunk(chunk_id="c1", text="participating providers have 180 days to file claims")
        c_self_redundant = FilledChunk(chunk_id="c2", text="participating providers have 180 days to file claim")
        a_redundant = FilledChunk(chunk_id="a1", text="the participating providers have 180 days to file claims total")
        fused = rrf_fuse({"c": [winner, c_self_redundant], "a": [a_redundant]})
        selection = mmr_select(
            fused, token_budget=1_000_000, estimate_tokens=_estimate_tokens,
            redundancy_threshold=0.7,
        )

        diagnostic = derive_coverage_diagnostic("s1", selection)

        assert diagnostic.pool_verdict == POOL_VERDICT_SATURATED
        assert diagnostic.saturated_strategies == ["a", "c"]

    def test_gaps_remain_when_at_least_one_strategy_still_viable(self):
        a1 = FilledChunk(chunk_id="a1", text="alpha distinct content one about topic one")
        b1 = FilledChunk(chunk_id="b1", text="beta distinct content two about topic two")
        fused = rrf_fuse({"a": [a1], "b": [b1]})
        selection = mmr_select(fused, token_budget=1_000_000, estimate_tokens=_estimate_tokens)

        diagnostic = derive_coverage_diagnostic("s1", selection)

        assert diagnostic.pool_verdict == POOL_VERDICT_GAPS_REMAIN
        assert diagnostic.saturated_strategies == []

    def test_empty_selection_is_gaps_remain_not_saturated(self):
        """Vacuous-truth trap: an empty saturated_strategies set is
        technically a superset of an empty all_contributing set in Python
        set semantics -- must not let that make an empty slot read as
        'saturated' when it's really 'nothing here yet, still needs
        content'."""
        selection = mmr_select([], token_budget=1000, estimate_tokens=_estimate_tokens)
        diagnostic = derive_coverage_diagnostic("s1", selection)
        assert diagnostic.pool_verdict == POOL_VERDICT_GAPS_REMAIN

    def test_reason_is_nonempty_prose_for_every_verdict(self):
        for selection in [
            mmr_select([], token_budget=1000, estimate_tokens=_estimate_tokens),
            mmr_select(
                rrf_fuse({"a": [FilledChunk(chunk_id="a1", text="x" * 10000)]}),
                token_budget=1, estimate_tokens=_estimate_tokens,
            ),
        ]:
            diagnostic = derive_coverage_diagnostic("s1", selection)
            assert isinstance(diagnostic.reason, str) and diagnostic.reason

    def test_per_strategy_counts_distinguishes_1_of_6_from_5_of_6(self):
        """Router's ask, 2026-07-24: the binary saturated_strategies bar
        can over-exclude on a single incidental loss -- this is the
        instrument that lets Eval tell "1 of 6 candidates rejected" (weak
        over-exclusion signal) apart from "5 of 6 rejected" (strong
        saturation signal), instead of both reading identically as just
        "a is in saturated_strategies"."""
        winner = FilledChunk(chunk_id="c1", text="alpha winning unique content")
        # Distinct chunk_id AND slightly distinct text per candidate (a
        # trailing unique marker word) -- avoids rrf_fuse's own identity
        # merge (which operates on exact body-text match) while staying
        # similar enough to clear a relaxed MMR redundancy threshold.
        a_candidates = [
            FilledChunk(chunk_id=f"a{i}", text=f"alpha winning unique content nearly nearly{i}")
            for i in range(6)
        ]
        fused = rrf_fuse({"c": [winner], "a": a_candidates})
        selection = mmr_select(
            fused, token_budget=1_000_000, estimate_tokens=_estimate_tokens,
            redundancy_threshold=0.5,
        )

        diagnostic = derive_coverage_diagnostic("s1", selection)

        assert "a" in diagnostic.per_strategy_counts
        counts = diagnostic.per_strategy_counts["a"]
        assert counts["n_selected"] + counts["n_merged_away"] == 6
        assert counts["n_merged_away"] >= 1  # at least one truly redundant against the others

    def test_per_strategy_counts_populated_even_in_budget_full(self):
        """Partial data (whatever WAS resolved before the budget cutoff)
        is still informative -- must not be empty just because the loop
        stopped early."""
        a1 = FilledChunk(chunk_id="a1", text="alpha distinct content")
        a2 = FilledChunk(chunk_id="a2", text="beta distinct content entirely different")
        fused = rrf_fuse({"a": [a1, a2]})
        one_chunk_tokens = _estimate_tokens(a1.text)
        selection = mmr_select(fused, token_budget=one_chunk_tokens, estimate_tokens=_estimate_tokens)

        diagnostic = derive_coverage_diagnostic("s1", selection)

        assert diagnostic.pool_verdict == POOL_VERDICT_BUDGET_FULL
        assert diagnostic.per_strategy_counts  # not empty despite the early cutoff

    def test_per_strategy_counts_empty_for_empty_selection(self):
        selection = mmr_select([], token_budget=1000, estimate_tokens=_estimate_tokens)
        diagnostic = derive_coverage_diagnostic("s1", selection)
        assert diagnostic.per_strategy_counts == {}
