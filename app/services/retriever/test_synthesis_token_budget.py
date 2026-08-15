"""Unit tests for Synthesis's token-budget enforcement (2026-07-24).

Real incident this closes: neighbor completion has no budget awareness of
its own and can multiply a bounded initial retrieval well past the
caller's real token_budget before it ever reaches Contract/Chat -- traced
to a live ~473K-character prompt that exhausted Vertex's per-minute token
quota. `_trim_to_token_budget` is pure/sync (no DB), tested directly here
rather than in test_synthesis.py to avoid colliding with that actively-
edited shared file (Synthesizer's own test suite).
"""

from app.services.retriever.synthesis import _estimate_tokens, _trim_to_token_budget
from app.services.retriever.synthesis_contracts import CompiledCitation, CompiledSlot


def _citation(index, *, text="x" * 40, is_neighbor=False, original_score=1.0, slot_id="direct_answer"):
    return CompiledCitation(
        index=index, chunk_id=f"c{index}", document_name="doc", text=text,
        source_type="internal", document_status="live",
        is_neighbor=is_neighbor, original_score=original_score, slot_id=slot_id,
    )


class TestEstimateTokens:
    def test_chars_over_four_approximation(self):
        assert _estimate_tokens("x" * 40) == 10

    def test_none_text_is_zero(self):
        assert _estimate_tokens(None) == 0


class TestTrimToTokenBudget:
    def test_under_budget_trims_nothing(self):
        citations = [_citation(1, text="x" * 40)]
        slots = {"direct_answer": CompiledSlot(
            slot_id="direct_answer", slot_semantics="direct_answer", capacity=5,
            required=True, citations=list(citations),
        )}
        dropped = _trim_to_token_budget(citations, slots, token_budget=1000)
        assert dropped == 0
        assert len(citations) == 1
        assert len(slots["direct_answer"].citations) == 1

    def test_over_budget_drops_neighbors_first(self):
        primary = _citation(1, text="x" * 400, is_neighbor=False, original_score=0.9)
        neighbor = _citation(2, text="y" * 400, is_neighbor=True, original_score=0.9)
        citations = [primary, neighbor]
        slots = {"direct_answer": CompiledSlot(
            slot_id="direct_answer", slot_semantics="direct_answer", capacity=5,
            required=True, citations=list(citations),
        )}
        # total tokens = 100 + 100 = 200; budget only fits one.
        dropped = _trim_to_token_budget(citations, slots, token_budget=150)
        assert dropped == 1
        assert citations == [primary]
        assert slots["direct_answer"].citations == [primary]

    def test_over_budget_drops_lowest_score_among_non_neighbors_when_all_primary(self):
        strong = _citation(1, text="x" * 400, is_neighbor=False, original_score=0.9)
        weak = _citation(2, text="y" * 400, is_neighbor=False, original_score=0.1)
        citations = [strong, weak]
        slots = {"direct_answer": CompiledSlot(
            slot_id="direct_answer", slot_semantics="direct_answer", capacity=5,
            required=True, citations=list(citations),
        )}
        dropped = _trim_to_token_budget(citations, slots, token_budget=150)
        assert dropped == 1
        assert citations == [strong]

    def test_reindexes_survivors_contiguously(self):
        c1 = _citation(1, text="x" * 400, is_neighbor=True)
        c2 = _citation(2, text="y" * 40, is_neighbor=False)
        c3 = _citation(3, text="z" * 40, is_neighbor=False)
        citations = [c1, c2, c3]
        slots = {"direct_answer": CompiledSlot(
            slot_id="direct_answer", slot_semantics="direct_answer", capacity=5,
            required=True, citations=list(citations),
        )}
        _trim_to_token_budget(citations, slots, token_budget=20)
        assert [c.index for c in citations] == list(range(1, len(citations) + 1))

    def test_never_drops_below_budget_when_impossible_keeps_cheapest(self):
        """If the single remaining citation alone exceeds budget, it must
        still survive -- a required slot with real evidence is never
        trimmed down to literally nothing (mirrors mmr_select's own
        "always keep at least one selection" guarantee in fusion.py).
        Real bug fix 2026-08-14: the old loop dropped this last citation
        too, and this test's own name/docstring claimed the opposite of
        what its assertion (`dropped in (0, 1)`) actually enforced --
        exactly how a live incident (portfolio dispatch synthesizing a
        required slot down to zero citations despite fillers delivering
        real content) went uncaught."""
        c1 = _citation(1, text="x" * 4000, is_neighbor=True, original_score=0.5)
        citations = [c1]
        slots = {"direct_answer": CompiledSlot(
            slot_id="direct_answer", slot_semantics="direct_answer", capacity=5,
            required=True, citations=list(citations),
        )}
        dropped = _trim_to_token_budget(citations, slots, token_budget=10)
        assert dropped == 0
        assert citations == [c1]
        assert slots["direct_answer"].citations == [c1]

    def test_multiple_oversized_citations_keeps_at_least_the_strongest(self):
        """Real incident shape: several citations that, even combined,
        exceed budget -- e.g. one primary plus several neighbors added by
        completion, all individually large. Must still keep at least the
        single strongest (last-to-drop) survivor rather than trim to zero."""
        weak_neighbor = _citation(1, text="a" * 4000, is_neighbor=True, original_score=0.2)
        another_neighbor = _citation(2, text="b" * 4000, is_neighbor=True, original_score=0.3)
        strongest_primary = _citation(3, text="c" * 4000, is_neighbor=False, original_score=0.9)
        citations = [weak_neighbor, another_neighbor, strongest_primary]
        slots = {"direct_answer": CompiledSlot(
            slot_id="direct_answer", slot_semantics="direct_answer", capacity=5,
            required=True, citations=list(citations),
        )}
        dropped = _trim_to_token_budget(citations, slots, token_budget=10)
        assert dropped == 2
        assert citations == [strongest_primary]
        assert slots["direct_answer"].citations == [strongest_primary]
