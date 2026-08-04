"""Unit tests for Synthesis (Step 5 -- compile Fillers' FilledShape into a
reranked, deduped, neighbor-complete, cited package for Chat/Eval).
"""

import pytest

from app.services.retriever import synthesis
from app.services.retriever.fillers.contracts import FilledChunk, FilledShape, FilledSlot
from app.services.retriever.synthesis_contracts import (
    POOL_VERDICT_BUDGET_FULL,
    POOL_VERDICT_GAPS_REMAIN,
    POOL_VERDICT_SATURATED,
    SlotVerdict,
)
from app.services.router.continuation import VERDICT_SATISFIED, VERDICT_WOULD_BENEFIT


def _chunk(
    chunk_id, *, document_id=None, url=None, text="text", score=1.0,
    is_neighbor=False, content_sha=None, assignment_reason="score_rank",
    document_status=None, source_type=None, quote_verified=None,
):
    return FilledChunk(
        chunk_id=chunk_id, document_id=document_id, text=text, url=url,
        source_type=source_type or ("internal" if document_id else "external"),
        document_status=document_status, content_sha=content_sha,
        is_neighbor=is_neighbor, original_score=score,
        assignment_reason=assignment_reason, quote_verified=quote_verified,
    )


def _slot(slot_id, chunks, *, semantics="direct_answer", capacity=None, required=True):
    capacity = capacity if capacity is not None else len(chunks)
    return FilledSlot(
        slot_id=slot_id, slot_semantics=semantics, capacity=capacity,
        required=required, chunks=chunks, occupancy=len(chunks),
        under_filled=len(chunks) < capacity, over_filled=False,
    )


class TestRerankSlotChunks:
    def test_primary_matches_sort_before_neighbors(self):
        primary = _chunk("p", score=0.5)
        neighbor = _chunk("n", score=0.9, is_neighbor=True)
        out = synthesis._rerank_slot_chunks([neighbor, primary])
        assert [c.chunk_id for c in out] == ["p", "n"]

    def test_higher_score_first_within_same_neighbor_tier(self):
        low = _chunk("low", score=0.2)
        high = _chunk("high", score=0.8)
        out = synthesis._rerank_slot_chunks([low, high])
        assert [c.chunk_id for c in out] == ["high", "low"]

    def test_none_score_treated_as_lowest(self):
        none_score = _chunk("none", score=None)
        scored = _chunk("scored", score=0.1)
        out = synthesis._rerank_slot_chunks([none_score, scored])
        assert [c.chunk_id for c in out] == ["scored", "none"]


class TestDedupCrossSlot:
    def test_same_chunk_id_across_slots_deduped_first_wins(self):
        s1 = _slot("s1", [_chunk("dup", text="from s1")])
        s2 = _slot("s2", [_chunk("dup", text="from s2")])
        items = [(s1, s1.chunks[0]), (s2, s2.chunks[0])]
        out, removed = synthesis._dedup_cross_slot(items)
        assert removed == 1
        assert len(out) == 1
        assert out[0][1].text == "from s1"

    def test_same_content_sha_different_ids_deduped(self):
        s1 = _slot("s1", [_chunk("a", content_sha="sha1")])
        s2 = _slot("s2", [_chunk("b", content_sha="sha1")])
        items = [(s1, s1.chunks[0]), (s2, s2.chunks[0])]
        out, removed = synthesis._dedup_cross_slot(items)
        assert removed == 1
        assert len(out) == 1

    def test_distinct_chunks_all_kept(self):
        s1 = _slot("s1", [_chunk("a", text="alpha content"), _chunk("b", text="beta content")])
        items = [(s1, c) for c in s1.chunks]
        out, removed = synthesis._dedup_cross_slot(items)
        assert removed == 0
        assert len(out) == 2

    def test_identical_text_different_content_sha_deduped(self):
        """Eval's correctness finding, 2026-07-24: content_sha is salted
        per-document in this schema (not a pure content hash) -- the same
        fact/boilerplate appearing in two different payer manuals gets
        byte-identical text but two DIFFERENT content_sha values. A dedup
        that only falls back to body-text when content_sha is ABSENT would
        let this duplicate survive; body-text must be checked regardless of
        whether content_sha is also present."""
        s1 = _slot("s1", [_chunk("a", text="shared boilerplate clause", content_sha="sha-doc-1")])
        s2 = _slot("s2", [_chunk("b", text="shared boilerplate clause", content_sha="sha-doc-2")])
        items = [(s1, s1.chunks[0]), (s2, s2.chunks[0])]
        out, removed = synthesis._dedup_cross_slot(items)
        assert removed == 1
        assert len(out) == 1


class TestFallbackDocumentName:
    def test_url_path_humanized(self):
        c = _chunk("x", url="https://www.example.gov/providers/prior-authorization-requirements")
        assert synthesis._fallback_document_name(c) == "Prior Authorization Requirements"

    def test_url_strips_known_extension(self):
        c = _chunk("x", url="https://www.example.gov/docs/member-handbook.pdf")
        assert synthesis._fallback_document_name(c) == "Member Handbook"

    def test_url_with_no_path_falls_back_to_domain(self):
        c = _chunk("x", url="https://www.cms.gov")
        assert synthesis._fallback_document_name(c) == "www.cms.gov"

    def test_document_id_falls_back_to_short_label(self):
        c = _chunk("x", document_id="abcdef1234567890")
        assert synthesis._fallback_document_name(c) == "document abcdef12"

    def test_neither_falls_back_to_generic_source(self):
        c = FilledChunk(chunk_id="x")
        assert synthesis._fallback_document_name(c) == "source"


class TestInferAuthority:
    """Chat's sign-off gap, 2026-07-24: SourceRef.authority feeds the
    grounding badge ("grounded" iff every source is authoritative)."""

    def test_internal_is_authoritative(self):
        c = _chunk("x", document_id="doc1")
        assert synthesis._infer_authority(c, "internal") == "authoritative"

    def test_external_is_external(self):
        c = _chunk("x", url="https://example.com/page")
        assert synthesis._infer_authority(c, "external") == "external"

    def test_planned_overrides_source_type(self):
        c = _chunk("x", document_id="doc1", document_status="planned")
        assert synthesis._infer_authority(c, "internal") == "planned"

    def test_fact_store_is_authoritative_not_external(self):
        """Retriever's finding, 2026-07-24: a fact_store hit is a
        certified, pre-verified Payor Platform fact -- not a lower-
        confidence web result. The old `source_type == "internal"` check
        would have mislabeled it "external"."""
        c = _chunk("x", document_id="fact_ab4ddde8d8adb299", source_type="fact_store")
        assert synthesis._infer_authority(c, "fact_store") == "authoritative"


class TestCompileSynthesis:
    @pytest.mark.asyncio
    async def test_compiles_citations_with_resolved_names_and_verdicts(self, monkeypatch):
        async def fake_resolve(db, chunk_ids):
            return {"c1": "Real Document Name"}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        slot = _slot("s1", [_chunk("c1", document_id="doc1", score=0.9)])
        shape = FilledShape(slots=[slot], total_chunks_assigned=1)

        result = await synthesis.compile_synthesis(
            "query", shape, db=None,
            verdicts={"s1": SlotVerdict(VERDICT_SATISFIED, "bm25_filled_to_capacity")},
        )

        assert len(result.citations) == 1
        cite = result.citations[0]
        assert cite.index == 1
        assert cite.document_name == "Real Document Name"
        assert cite.source_type == "internal"
        assert cite.authority == "authoritative"
        assert cite.verified is True

        assert len(result.slots) == 1
        assert result.slots[0].verdict == VERDICT_SATISFIED
        assert result.slots[0].verdict_reason == "bm25_filled_to_capacity"

        assert result.telemetry.chunks_in == 1
        assert result.telemetry.chunks_out == 1
        assert result.telemetry.document_name_resolved == 1
        assert result.telemetry.document_name_fallback == 0

        # DB's ask, 2026-07-24: compile_ms is the TOTAL across every named
        # segment below (not name-lookup specifically), and segment_ms
        # breaks out where the time actually goes -- same convention as
        # Pool's segment_ms. `budget_enforcement_ms` added by Retriever's
        # later _trim_to_token_budget patch (real incident: unbounded
        # neighbor completion blew a 473K-char prompt past Vertex's quota).
        assert set(result.telemetry.segment_ms) == {
            "rerank_ms", "neighbor_completion_ms", "dedup_ms",
            "name_resolution_ms", "citation_build_ms", "budget_enforcement_ms",
        }
        assert all(ms >= 0 for ms in result.telemetry.segment_ms.values())

    @pytest.mark.asyncio
    async def test_unresolved_name_falls_back_and_is_counted(self, monkeypatch):
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        slot = _slot("s1", [_chunk("c1", document_id="doc1")])
        shape = FilledShape(slots=[slot])

        result = await synthesis.compile_synthesis("query", shape, db=None)

        assert result.telemetry.document_name_fallback == 1
        assert result.citations[0].document_name.startswith("document ")

    @pytest.mark.asyncio
    async def test_unverified_llm_citation_flagged(self, monkeypatch):
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        slot = _slot("s1", [_chunk("c1", document_id="doc1", assignment_reason="llm_partial_match")])
        shape = FilledShape(slots=[slot])

        result = await synthesis.compile_synthesis("query", shape, db=None)

        assert result.citations[0].verified is False
        assert result.telemetry.unverified_citations == 1

    @pytest.mark.asyncio
    async def test_llm_citation_with_no_quote_given_reads_unverified(self, monkeypatch):
        """Eval's regression request, 2026-07-24: a filler-c citation where
        no quote was given at all (quote_verified=None, collapsed upstream
        into the same assignment_reason="llm_retrieved" bucket as a real
        quote match) must compile to verified=False, not True -- the exact
        overstated-confidence gap Filler c found and Eval ruled on."""
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        slot = _slot("s1", [_chunk(
            "c1", document_id="doc1", assignment_reason="llm_retrieved", quote_verified=None,
        )])
        shape = FilledShape(slots=[slot])

        result = await synthesis.compile_synthesis("query", shape, db=None)

        assert result.citations[0].verified is False
        assert result.telemetry.unverified_citations == 1

    @pytest.mark.asyncio
    async def test_llm_citation_with_matched_quote_reads_verified(self, monkeypatch):
        """The real-confirmation case -- quote given AND matched -- must
        still read verified=True, or the fail-closed fix would have
        overcorrected into never trusting an LLM citation at all."""
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        slot = _slot("s1", [_chunk(
            "c1", document_id="doc1", assignment_reason="llm_retrieved", quote_verified=True,
        )])
        shape = FilledShape(slots=[slot])

        result = await synthesis.compile_synthesis("query", shape, db=None)

        assert result.citations[0].verified is True
        assert result.telemetry.unverified_citations == 0

    @pytest.mark.asyncio
    async def test_non_llm_chunk_verified_regardless_of_quote_verified_default(self, monkeypatch):
        """Regression guard: applying `quote_verified is True` UNCONDITIONALLY
        across every chunk (the naive literal reading of the fix) would
        misread every a/b/s chunk as unverified, since none of them ever
        touch `quote_verified` and it defaults to None. A BM25/vector/
        fact-store match is a direct corpus/service hit, never subject to
        quote verification, and must stay verified=True."""
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        slot = _slot("s1", [_chunk(
            "c1", document_id="doc1", assignment_reason="score_rank", quote_verified=None,
        )])
        shape = FilledShape(slots=[slot])

        result = await synthesis.compile_synthesis("query", shape, db=None)

        assert result.citations[0].verified is True
        assert result.telemetry.unverified_citations == 0

    @pytest.mark.asyncio
    async def test_planned_status_survives_and_is_counted(self, monkeypatch):
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        slot = _slot("s1", [_chunk("c1", document_id="doc1", document_status="planned")])
        shape = FilledShape(slots=[slot])

        result = await synthesis.compile_synthesis("query", shape, db=None)

        assert result.citations[0].document_status == "planned"
        assert result.telemetry.planned_status_citations == 1

    @pytest.mark.asyncio
    async def test_live_status_survives_and_is_not_counted_as_planned(self, monkeypatch):
        """Product-Awareness's sign-off condition (2026-07-24): spot-check
        both "planned" and "live" round-trip, don't just check one."""
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        slot = _slot("s1", [_chunk("c1", document_id="doc1", document_status="live")])
        shape = FilledShape(slots=[slot])

        result = await synthesis.compile_synthesis("query", shape, db=None)

        assert result.citations[0].document_status == "live"
        assert result.telemetry.planned_status_citations == 0

    @pytest.mark.asyncio
    async def test_none_status_survives_as_none_not_coerced(self, monkeypatch):
        """None is a legitimate document_status (e.g. external chunks with
        no document row) -- must round-trip as None, not get coerced to a
        falsy string or dropped."""
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        slot = _slot("s1", [_chunk("c1", url="https://example.com/page", document_status=None)])
        shape = FilledShape(slots=[slot])

        result = await synthesis.compile_synthesis("query", shape, db=None)

        assert result.citations[0].document_status is None
        assert result.telemetry.planned_status_citations == 0

    @pytest.mark.asyncio
    async def test_missing_verdict_defaults_to_empty_not_fabricated(self, monkeypatch):
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        slot = _slot("s1", [_chunk("c1", document_id="doc1")])
        shape = FilledShape(slots=[slot])

        result = await synthesis.compile_synthesis("query", shape, db=None)

        assert result.slots[0].verdict == ""
        assert result.slots[0].verdict_reason == ""

    @pytest.mark.asyncio
    async def test_cross_slot_duplicate_counted_in_telemetry(self, monkeypatch):
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        s1 = _slot("s1", [_chunk("dup", document_id="doc1")])
        s2 = _slot("s2", [_chunk("dup", document_id="doc1")])
        shape = FilledShape(slots=[s1, s2])

        result = await synthesis.compile_synthesis("query", shape, db=None)

        assert len(result.citations) == 1
        assert result.telemetry.duplicates_removed == 1
        assert result.telemetry.chunks_in == 2

    @pytest.mark.asyncio
    async def test_empty_slot_is_not_dropped_from_output(self, monkeypatch):
        """Real bug found in Ananth's QA pass, 2026-07-24: iterating only
        chunk-level (slot, chunk) pairs meant a slot with zero chunks never
        appeared in the loop at all, so it silently vanished from
        result.slots -- Chat had no way to caveat a fully-exhausted slot
        because it never even saw it. A vanished slot must be
        indistinguishable from neither existing NOR from being reported."""
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        filled_slot = _slot("s1", [_chunk("c1", document_id="doc1")])
        empty_slot = FilledSlot(
            slot_id="s2", slot_semantics="thematic_exploration", capacity=3,
            required=False, chunks=[], occupancy=0, under_filled=True, over_filled=False,
        )
        shape = FilledShape(slots=[filled_slot, empty_slot])

        result = await synthesis.compile_synthesis(
            "query", shape, db=None,
            verdicts={"s2": SlotVerdict(VERDICT_WOULD_BENEFIT, "no_implemented_strategy_or_pool")},
        )

        assert [s.slot_id for s in result.slots] == ["s1", "s2"]
        empty = result.slots[1]
        assert empty.citations == []
        assert empty.occupancy == 0
        assert empty.verdict == VERDICT_WOULD_BENEFIT
        assert empty.verdict_reason == "no_implemented_strategy_or_pool"

        # Blend model (2026-07-24): every slot, including empty ones, must
        # get a real CoverageDiagnostic for Router's decide_continuation --
        # a missing entry would be ambiguous ("no data" vs "not
        # applicable"). An empty slot correctly reads gaps_remain, not
        # saturated (the vacuous-truth trap fusion.py's own tests guard).
        assert "s2" in result.coverage_diagnostics
        assert result.coverage_diagnostics["s2"].pool_verdict == POOL_VERDICT_GAPS_REMAIN


class TestFusionIntegration:
    """Blend model, retention landing 2026-07-24: compile_synthesis's real
    per-slot pipeline now runs RRF+MMR fusion across a slot's chunks
    (potentially from several strategies, since retention unions rungs
    instead of discarding). These tests exercise the ACTUAL wiring, not
    fusion.py's algorithms in isolation (already covered in test_fusion.py)."""

    @pytest.mark.asyncio
    async def test_cross_strategy_redundant_chunks_get_fused_not_just_literally_deduped(self, monkeypatch):
        """The scenario retention exists for: two DIFFERENT strategies
        (different chunk_id, different assignment_reason) surface the SAME
        underlying fact, worded slightly differently -- literal chunk_id/
        content-key dedup (_dedup_cross_slot) would NOT catch this (no
        shared identity), but RRF+MMR's TF-IDF redundancy check does."""
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        # Measured directly (not guessed): TF-IDF cosine similarity ~0.92,
        # comfortably above the default 0.85 redundancy threshold -- near-
        # paraphrase, not literally identical text (which would instead be
        # caught by rrf_fuse's own identity-merge before mmr_select's
        # redundancy check ever runs, a different mechanism this test isn't
        # targeting).
        a_chunk = _chunk(
            "a1", document_id="doc-a",
            text="participating providers have 180 days to file claims for services rendered",
            assignment_reason="score_rank", score=0.9,
        )
        b_chunk = _chunk(
            "b1", document_id="doc-b",
            text="participating providers have 180 days to file claims for the services rendered",
            assignment_reason="vector_rerank", score=0.9,
        )
        slot = _slot("s1", [a_chunk, b_chunk])
        shape = FilledShape(slots=[slot])

        result = await synthesis.compile_synthesis("query", shape, db=None)

        # Literal cross-slot dedup alone would never have caught this (no
        # shared chunk_id/content_sha/exact body-text match) -- only real
        # fusion redundancy checking does.
        assert result.telemetry.duplicates_removed == 0
        assert result.telemetry.fusion_dropped_redundant >= 1
        assert len(result.citations) == 1

    @pytest.mark.asyncio
    async def test_distinct_strategies_both_survive_when_genuinely_different(self, monkeypatch):
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        a_chunk = _chunk(
            "a1", document_id="doc-a", text="alpha content about topic one entirely",
            assignment_reason="score_rank",
        )
        b_chunk = _chunk(
            "b1", document_id="doc-b", text="beta content about a completely different topic",
            assignment_reason="vector_rerank",
        )
        slot = _slot("s1", [a_chunk, b_chunk])
        shape = FilledShape(slots=[slot])

        result = await synthesis.compile_synthesis("query", shape, db=None)

        assert len(result.citations) == 2
        assert result.telemetry.fusion_dropped_redundant == 0

    @pytest.mark.asyncio
    async def test_coverage_diagnostic_populated_for_every_slot(self, monkeypatch):
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        slot = _slot("s1", [_chunk("a1", document_id="doc-a", assignment_reason="score_rank")])
        shape = FilledShape(slots=[slot])

        result = await synthesis.compile_synthesis("query", shape, db=None)

        assert "s1" in result.coverage_diagnostics
        assert result.coverage_diagnostics["s1"].pool_verdict in {
            POOL_VERDICT_GAPS_REMAIN, POOL_VERDICT_SATURATED, POOL_VERDICT_BUDGET_FULL,
        }

    @pytest.mark.asyncio
    async def test_per_slot_payload_tokens_splits_budget_across_slots(self, monkeypatch):
        """A slot with a much smaller payload-token share must end up with
        a tighter effective MMR budget than one with a much larger share --
        proves per_slot_payload_tokens is actually used as split weights,
        not ignored."""
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        # Each slot gets 5 substantial, mutually-distinct chunks -- enough
        # that a tight budget forces real trimming and a generous one doesn't.
        def _distinct_chunks(prefix, n):
            return [
                _chunk(
                    f"{prefix}{i}", document_id=f"doc-{prefix}{i}",
                    text=f"{prefix} distinct substantive content block number {i} " * 10,
                    assignment_reason="score_rank",
                )
                for i in range(n)
            ]

        rich_slot = _slot("rich", _distinct_chunks("rich", 5), capacity=5)
        poor_slot = _slot("poor", _distinct_chunks("poor", 5), capacity=5)
        shape = FilledShape(slots=[rich_slot, poor_slot])

        one_chunk_tokens = synthesis._estimate_tokens(_distinct_chunks("rich", 1)[0].text)
        overall_budget = one_chunk_tokens * 5  # enough for ~5 total, split unevenly below

        result = await synthesis.compile_synthesis(
            "query", shape, db=None,
            token_budget=overall_budget,
            per_slot_payload_tokens={"rich": 9000, "poor": 1000},
        )

        rich_count = len([c for c in result.citations if c.slot_id == "rich"])
        poor_count = len([c for c in result.citations if c.slot_id == "poor"])

        assert rich_count > poor_count

    @pytest.mark.asyncio
    async def test_legacy_call_without_per_slot_payload_tokens_still_works(self, monkeypatch):
        """None-safe: omitting per_slot_payload_tokens must not crash or
        change existing single-strategy behavior (falls back to an
        effectively-unbounded per-slot MMR budget, relying on the existing
        global _trim_to_token_budget pass for overall enforcement)."""
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        slot = _slot("s1", [_chunk("a1", document_id="doc-a")])
        shape = FilledShape(slots=[slot])

        result = await synthesis.compile_synthesis("query", shape, db=None, token_budget=1_000_000)

        assert len(result.citations) == 1
        assert result.telemetry.fusion_dropped_budget == 0

    @pytest.mark.asyncio
    async def test_reconciliation_holds_when_fusion_drops_a_chunk(self, monkeypatch):
        """The reconciliation guard's identity must account for fusion-level
        drops now, not just dedup/neighbor-completion -- otherwise it would
        false-positive on every query where fusion legitimately rejects a
        redundant cross-strategy chunk."""
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        a_chunk = _chunk(
            "a1", document_id="doc-a",
            text="participating providers have 180 days to file claims for services rendered",
            assignment_reason="score_rank",
        )
        b_chunk = _chunk(
            "b1", document_id="doc-b",
            text="participating providers have 180 days to file claims for the services rendered",
            assignment_reason="vector_rerank",
        )
        slot = _slot("s1", [a_chunk, b_chunk])
        shape = FilledShape(slots=[slot])

        result = await synthesis.compile_synthesis("query", shape, db=None)

        # No "reconciliation failed" ERROR should have been necessary --
        # the real proof is that chunks_out matches the identity exactly.
        expected = (
            result.telemetry.chunks_in
            - result.telemetry.fusion_dropped_redundant
            - result.telemetry.fusion_dropped_budget
            - result.telemetry.fusion_content_merged
            - result.telemetry.duplicates_removed
            + result.telemetry.neighbors_added
        )
        assert result.telemetry.chunks_out == expected

    @pytest.mark.asyncio
    async def test_same_strategy_content_identical_chunks_from_two_documents_get_merged_and_counted(self, monkeypatch):
        """Root-caused live 2026-07-29 (Sunshine Health timely-filing query,
        dev DB): rrf_fuse's own content-identity merge (chunk_identity.py's
        content_keys(), body-text match) folds two chunks into one canonical
        FusedChunk even WITHIN a single strategy group -- e.g. two separately-
        ingested copies of the same document repeating an identical
        paragraph. Before this fix, that merge was invisible to every
        telemetry counter, so the reconciliation guard's identity (which
        assumed every input chunk became a real neighbor-completion seed)
        false-alarmed on every run of this query. `fusion_content_merged`
        must count it and the identity must hold."""
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        shared_text = (
            "To send claims electronically to Sunshine Health, all EDI "
            "claims must first be forwarded to the clearinghouse"
        )
        a_chunk = _chunk(
            "bf6efa88-b68f-43a6-87d4-14018ba5bbe9", document_id="d9721756-d1b1-4cf4-845b-f44652c5fcf9",
            text=shared_text, assignment_reason="score_rank",
        )
        b_chunk = _chunk(
            "38d60941-5d11-470a-b5ed-2b84b9063c02", document_id="8fba1cb5-2203-49ca-b381-58ea32c4d86c",
            text=shared_text, assignment_reason="score_rank",
        )
        slot = _slot("s1", [a_chunk, b_chunk])
        shape = FilledShape(slots=[slot])

        result = await synthesis.compile_synthesis("query", shape, db=None)

        assert result.telemetry.fusion_content_merged == 1
        assert result.telemetry.duplicates_removed == 0
        assert result.telemetry.fusion_dropped_redundant == 0
        assert len(result.citations) == 1

        expected = (
            result.telemetry.chunks_in
            - result.telemetry.fusion_dropped_redundant
            - result.telemetry.fusion_dropped_budget
            - result.telemetry.fusion_content_merged
            - result.telemetry.duplicates_removed
            + result.telemetry.neighbors_added
        )
        assert result.telemetry.chunks_out == expected


class TestDataCollectionMode:
    """Data-collection posture (Ananth, 2026-07-24): Router forces a single
    strategy per slot during recall-at-K calibration; real MMR selection
    must be bypassed so a forced strategy's true top-X reaches Chat/Eval
    uncontaminated by MMR's still-uncalibrated redundancy_threshold/budget-
    cutoff. Gated on an EXPLICIT `data_collection_mode` flag, not shape-
    inference -- see synthesis.py's docstring for why (a slot organically
    having one strategy's chunks also happens in normal blend-mode
    operation and must still get real per-slot MMR budget-splitting)."""

    @staticmethod
    def _distinct_chunks(prefix, n):
        return [
            _chunk(
                f"{prefix}{i}", document_id=f"doc-{prefix}{i}",
                text=f"{prefix} distinct substantive content block number {i} " * 10,
                assignment_reason="score_rank",
            )
            for i in range(n)
        ]

    @pytest.mark.asyncio
    async def test_data_collection_mode_keeps_all_chunks_despite_skewed_per_slot_share(self, monkeypatch):
        """The whole point of this mode: a slot given a tiny per_slot_
        payload_tokens share (which would starve it under real per-slot MMR
        budget-splitting -- see
        test_per_slot_payload_tokens_splits_budget_across_slots, the normal-
        mode counterpart of this exact setup) must NOT lose chunks at the
        fusion stage -- that's exactly the contamination Retriever flagged.
        Overall token_budget is deliberately generous (comfortably covers
        all 10 real chunks) so the SEPARATE downstream global trim doesn't
        confound this -- this test isolates the per-slot MMR bypass
        specifically."""
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        rich_slot = _slot("rich", self._distinct_chunks("rich", 5), capacity=5)
        poor_slot = _slot("poor", self._distinct_chunks("poor", 5), capacity=5)
        shape = FilledShape(slots=[rich_slot, poor_slot])

        one_chunk_tokens = synthesis._estimate_tokens(self._distinct_chunks("rich", 1)[0].text)
        overall_budget = one_chunk_tokens * 20  # generous: 2x the real total (10 chunks)

        result = await synthesis.compile_synthesis(
            "query", shape, db=None,
            token_budget=overall_budget,
            per_slot_payload_tokens={"rich": 9500, "poor": 500},  # poor's share alone would starve it under real MMR
            data_collection_mode=True,
        )

        rich_count = len([c for c in result.citations if c.slot_id == "rich"])
        poor_count = len([c for c in result.citations if c.slot_id == "poor"])

        assert rich_count == 5
        assert poor_count == 5  # NOT starved, despite the skewed 500-token share
        assert result.telemetry.fusion_dropped_budget == 0
        assert result.telemetry.fusion_dropped_redundant == 0

    @pytest.mark.asyncio
    async def test_data_collection_mode_reports_budget_full_without_dropping(self, monkeypatch):
        """coverage_diagnostics still reports a genuine budget_full signal
        for the starved slot (Router/Eval's fill-depth measurement) even
        though nothing was actually dropped from the real output -- same
        setup as the bypass test above."""
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        rich_slot = _slot("rich", self._distinct_chunks("rich", 5), capacity=5)
        poor_slot = _slot("poor", self._distinct_chunks("poor", 5), capacity=5)
        shape = FilledShape(slots=[rich_slot, poor_slot])

        one_chunk_tokens = synthesis._estimate_tokens(self._distinct_chunks("rich", 1)[0].text)
        overall_budget = one_chunk_tokens * 20

        result = await synthesis.compile_synthesis(
            "query", shape, db=None,
            token_budget=overall_budget,
            per_slot_payload_tokens={"rich": 9500, "poor": 500},
            data_collection_mode=True,
        )

        poor_count = len([c for c in result.citations if c.slot_id == "poor"])

        assert result.coverage_diagnostics["poor"].pool_verdict == POOL_VERDICT_BUDGET_FULL
        assert poor_count == 5  # the signal fired, but nothing was actually cut

    @pytest.mark.asyncio
    async def test_data_collection_mode_never_reports_saturated_logs_detected_not_dropped(self, monkeypatch):
        """Near-duplicate chunks within the one forced strategy would
        trigger MMR's real redundancy drop in normal mode -- here they must
        survive (both reach citations) while the probe still measures the
        near-duplicate density as a log-only counter, per Retriever/Eval's
        calibration ask."""
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        # Same near-duplicate pair measured in
        # test_cross_strategy_redundant_chunks_get_fused_not_just_literally_deduped
        # (~0.92 TF-IDF cosine similarity, above the 0.85 default threshold).
        a_chunk = _chunk(
            "a1", document_id="doc-a",
            text="participating providers have 180 days to file claims for services rendered",
            assignment_reason="score_rank",
        )
        a_chunk2 = _chunk(
            "a2", document_id="doc-a2",
            text="participating providers have 180 days to file claims for the services rendered",
            assignment_reason="score_rank",
        )
        slot = _slot("s1", [a_chunk, a_chunk2])
        shape = FilledShape(slots=[slot])

        result = await synthesis.compile_synthesis(
            "query", shape, db=None, data_collection_mode=True,
        )

        assert len(result.citations) == 2  # neither one dropped
        assert result.telemetry.fusion_dropped_redundant == 0
        assert result.telemetry.fusion_redundant_detected_not_dropped >= 1
        assert result.coverage_diagnostics["s1"].pool_verdict != POOL_VERDICT_SATURATED
        assert result.coverage_diagnostics["s1"].saturated_strategies == []

    @pytest.mark.asyncio
    async def test_multi_strategy_slot_in_data_collection_mode_still_bypasses_and_warns(self, monkeypatch, caplog):
        """Defensive branch: if a slot unexpectedly has more than one
        strategy while the caller declared data_collection_mode=True (should
        never happen per Router's design, but a silent wrong-branch bug
        would be worse than a log line), the bypass still applies -- and a
        warning is logged rather than the assumption failing silently."""
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        a_chunk = _chunk("a1", document_id="doc-a", text="alpha content", assignment_reason="score_rank")
        b_chunk = _chunk("b1", document_id="doc-b", text="beta content", assignment_reason="vector_rerank")
        slot = _slot("s1", [a_chunk, b_chunk])
        shape = FilledShape(slots=[slot])

        with caplog.at_level("WARNING", logger="app.services.retriever.synthesis"):
            result = await synthesis.compile_synthesis(
                "query", shape, db=None, data_collection_mode=True,
            )

        assert len(result.citations) == 2
        assert any("data_collection_mode=True but slot_id=s1" in r.message for r in caplog.records)


class TestNeighborInheritsDocumentStatus:
    """Tech Review's finding, 2026-07-24: a fresh neighbor chunk previously
    hardcoded document_status=None regardless of its seed's real status --
    so a neighbor of a PLANNED document read as authoritative via
    _infer_authority (None != "planned"), even though the seed itself
    correctly read "planned". Neighbors must inherit the seed's status."""

    @pytest.mark.asyncio
    async def test_neighbor_inherits_planned_status_from_seed(self, monkeypatch):
        async def fake_expand(db, seeds, *, paragraph_window, page_window):
            seed = seeds[0]
            neighbor_row = {
                "id": "neighbor-1", "document_id": seed["document_id"],
                "text": "neighbor text", "page_number": 2, "paragraph_index": 1,
                "content_sha": None, "rerank_score": 0.1,
            }
            return seeds + [neighbor_row], {"requested": True}

        monkeypatch.setattr(synthesis, "_expand_with_neighbors", fake_expand)

        seed_chunk = FilledChunk(
            chunk_id="seed-1", document_id="planned-doc", text="seed text",
            document_status="planned", page_number=1, paragraph_index=0,
            source_type="internal", assignment_reason="score_rank", original_score=0.9,
        )
        slot = _slot("s1", [seed_chunk])

        out, added, skipped = await synthesis._complete_neighbors(
            db=None, slot_items=[(slot, seed_chunk)],
        )

        assert added == 1
        neighbor = next(c for _, c in out if c.chunk_id == "neighbor-1")
        assert neighbor.document_status == "planned"
        # Downstream consequence this fix actually protects against:
        assert synthesis._infer_authority(neighbor, "internal") == "planned"

    @pytest.mark.asyncio
    async def test_neighbor_inherits_none_status_when_seed_has_none(self, monkeypatch):
        async def fake_expand(db, seeds, *, paragraph_window, page_window):
            seed = seeds[0]
            neighbor_row = {
                "id": "neighbor-2", "document_id": seed["document_id"],
                "text": "neighbor text", "page_number": 2, "paragraph_index": 1,
                "content_sha": None, "rerank_score": 0.1,
            }
            return seeds + [neighbor_row], {"requested": True}

        monkeypatch.setattr(synthesis, "_expand_with_neighbors", fake_expand)

        seed_chunk = FilledChunk(
            chunk_id="seed-2", document_id="live-doc", text="seed text",
            document_status=None, page_number=1, paragraph_index=0,
            source_type="internal", assignment_reason="score_rank", original_score=0.9,
        )
        slot = _slot("s1", [seed_chunk])

        out, added, skipped = await synthesis._complete_neighbors(
            db=None, slot_items=[(slot, seed_chunk)],
        )

        neighbor = next(c for _, c in out if c.chunk_id == "neighbor-2")
        assert neighbor.document_status is None
        assert synthesis._infer_authority(neighbor, "internal") == "authoritative"


class TestFactStoreExclusion:
    """Retriever's finding, 2026-07-24, confirmed live (crashed EVERY
    fact-store-sourced answer, not a corner case): filler_s's chunk_id is
    ALWAYS a synthetic `fact_<hash>` string (never a UUID), and document_id
    is either the fact store's own source doc_id or that same synthetic
    string. Treating "document_id truthy, url falsy" as "real internal
    document" sent this straight into a `CAST(:ids AS uuid[])` query,
    raising asyncpg.InvalidTextRepresentationError. These tests use the
    REAL (unmocked) _resolve_document_names/_complete_neighbors with
    db=None -- if the fix were wrong, this would crash with an
    AttributeError on db.execute, not just fail an assertion."""

    @pytest.mark.asyncio
    async def test_fact_store_only_shape_does_not_touch_db(self, caplog):
        fact_chunk = _chunk(
            "fact_ab4ddde8d8adb299", document_id="fact_ab4ddde8d8adb299",
            source_type="fact_store", assignment_reason="fact_store_hit",
            text="Sunshine Health prior auth answer",
        )
        slot = _slot("s1", [fact_chunk])
        shape = FilledShape(slots=[slot])

        # No monkeypatch -- db=None would raise AttributeError the moment
        # either helper actually tried to call db.execute(...). Real
        # (unmocked) code path proves the fact_store chunk never reaches it.
        with caplog.at_level("WARNING", logger="app.services.retriever.synthesis"):
            result = await synthesis.compile_synthesis("query", shape, db=None)

        assert len(result.citations) == 1
        cite = result.citations[0]
        assert cite.source_type == "fact_store"
        assert cite.authority == "authoritative"
        assert cite.document_name  # some non-empty fallback label, not a crash
        assert result.telemetry.neighbors_skipped_no_anchor == 1
        assert result.telemetry.document_name_resolved == 0
        assert result.telemetry.document_name_fallback == 1
        # Retriever's polish catch, 2026-07-24: a fact_store chunk was
        # never eligible for a documents-table row -- must not fire the
        # "may be missing" ops-integrity alarm meant for genuine internal misses.
        assert result.telemetry.document_name_lookup_missed == 0
        assert not any("lookup missed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_fact_store_mixed_with_internal_chunk_only_resolves_internal(self, monkeypatch):
        seen_ids = {}

        async def spy_resolve(db, chunk_ids):
            seen_ids["ids"] = list(chunk_ids)
            return {"c1": "Real Internal Doc"}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", spy_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        internal_chunk = _chunk("c1", document_id="doc1", text="internal text")
        fact_chunk = _chunk(
            "fact_xyz", document_id="fact_xyz", source_type="fact_store",
            assignment_reason="fact_store_hit", text="fact answer",
        )
        slot = _slot("s1", [internal_chunk, fact_chunk])
        shape = FilledShape(slots=[slot])

        result = await synthesis.compile_synthesis("query", shape, db=None)

        assert seen_ids["ids"] == ["c1"]  # the fact_store chunk_id never reaches the batched query
        by_id = {c.chunk_id: c for c in result.citations}
        assert by_id["c1"].document_name == "Real Internal Doc"
        assert by_id["fact_xyz"].authority == "authoritative"


class TestChunkCountReconciliation:
    """Eval's ask, 2026-07-24: every input chunk's fate must be accounted
    for by exactly one counter, so chunks_out == chunks_in -
    duplicates_removed + neighbors_added holds exactly. If that identity
    breaks, a chunk was silently dropped or double-counted -- the failure
    mode that's invisible in a real calibration pass without this check
    ("chunks_out looks low" vs. "counter X is the leak")."""

    @pytest.mark.asyncio
    async def test_counts_reconcile_with_dedup_and_neighbor_addition_together(self, monkeypatch, caplog):
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            # Simulate 2 fresh neighbor chunks added on top of the 3 seeds.
            s1, s2, s3 = slot_items[0][0], slot_items[1][0], slot_items[2][0]
            new_items = list(slot_items) + [
                (s1, _chunk("n1", document_id="doc1", is_neighbor=True, text="neighbor one")),
                (s2, _chunk("n2", document_id="doc2", is_neighbor=True, text="neighbor two")),
            ]
            return new_items, 2, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        # 3 chunks in, one of which (c-dup) is a cross-slot duplicate of s1's chunk.
        s1 = _slot("s1", [_chunk("c1", document_id="doc1", text="alpha content")])
        s2 = _slot("s2", [_chunk("c-dup", document_id="doc1", text="alpha content")])
        s3 = _slot("s3", [_chunk("c3", document_id="doc3", text="gamma content")])
        shape = FilledShape(slots=[s1, s2, s3])

        with caplog.at_level("ERROR", logger="app.services.retriever.synthesis"):
            result = await synthesis.compile_synthesis("query", shape, db=None)

        assert result.telemetry.chunks_in == 3
        assert result.telemetry.duplicates_removed == 1
        assert result.telemetry.neighbors_added == 2
        expected_chunks_out = (
            result.telemetry.chunks_in
            - result.telemetry.duplicates_removed
            + result.telemetry.neighbors_added
        )
        assert result.telemetry.chunks_out == expected_chunks_out == 4
        assert not any("reconciliation failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_reconciliation_failure_is_logged_not_raised(self, monkeypatch, caplog):
        """A deliberately-lying _complete_neighbors (claims 0 added but
        actually adds 2) should be CAUGHT by the reconciliation check --
        logged loudly, but never raised (telemetry must not break the
        user's request)."""
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            s1 = slot_items[0][0]
            new_items = list(slot_items) + [
                (s1, _chunk("n1", document_id="doc1", is_neighbor=True, text="sneaky neighbor")),
            ]
            return new_items, 0, 0  # lies: claims 0 added, actually added 1

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        slot = _slot("s1", [_chunk("c1", document_id="doc1", text="alpha content")])
        shape = FilledShape(slots=[slot])

        with caplog.at_level("ERROR", logger="app.services.retriever.synthesis"):
            result = await synthesis.compile_synthesis("query", shape, db=None)

        assert result.telemetry.chunks_out == 2  # the compile itself is unaffected
        assert any("reconciliation failed" in r.message for r in caplog.records)


class TestTechnicalFailureRetry:
    """Ananth's standing principle, 2026-07-24: force a retry on technical
    failures before degrading -- "ask once and get the answer," try our
    best for first-pass resolution. Retriever's ruling on the 3-layer retry
    design (per-rung -> whole-loop -> this module's per-helper retry, finest
    scope first): after this function's own retry is exhausted, it must
    RE-RAISE, not degrade to an empty/partial result -- otherwise the
    orchestrator's whole-loop retry never gets a chance to recover a
    systemic failure, and Synthesis would quietly return a degraded result
    forever instead."""

    @pytest.mark.asyncio
    async def test_name_resolution_retries_once_then_succeeds(self, monkeypatch):
        calls = {"n": 0}

        async def flaky_execute(query, params):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("simulated transient DB failure")
            class _Result:
                def mappings(self):
                    class _M:
                        def all(self):
                            return [{"id": "c1", "document_display_name": "Recovered Name", "document_filename": None}]
                    return _M()
            return _Result()

        class _FakeDB:
            execute = staticmethod(flaky_execute)

        names = await synthesis._resolve_document_names(_FakeDB(), ["c1"])

        assert calls["n"] == 2  # one failure + one retry
        assert names == {"c1": "Recovered Name"}

    @pytest.mark.asyncio
    async def test_name_resolution_reraises_after_retry_exhausted(self, monkeypatch):
        async def always_fails(query, params):
            raise ConnectionError("simulated persistent DB failure")

        class _FakeDB:
            execute = staticmethod(always_fails)

        with pytest.raises(ConnectionError):
            await synthesis._resolve_document_names(_FakeDB(), ["c1"])

    @pytest.mark.asyncio
    async def test_compile_synthesis_propagates_name_lookup_failure(self, monkeypatch):
        """A technical failure that survives its own retry must escalate
        all the way out of compile_synthesis -- not get caught and turned
        into a degraded SynthesisResult."""
        async def fake_resolve(db, chunk_ids):
            raise ConnectionError("simulated persistent DB failure")

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        slot = _slot("s1", [_chunk("c1", document_id="doc1")])
        shape = FilledShape(slots=[slot])

        with pytest.raises(ConnectionError):
            await synthesis.compile_synthesis("query", shape, db=None)

    @pytest.mark.asyncio
    async def test_compile_synthesis_propagates_neighbor_completion_failure(self, monkeypatch):
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            raise ConnectionError("simulated persistent DB failure")

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        slot = _slot("s1", [_chunk("c1", document_id="doc1")])
        shape = FilledShape(slots=[slot])

        with pytest.raises(ConnectionError):
            await synthesis.compile_synthesis("query", shape, db=None)


class TestAttributionPassthrough:
    """Eval's non-negotiable ask (2026-07-24): ride_along, verdict/reason
    strings, and per-slot model_trace must survive compilation intact --
    never dropped, never normalized/collapsed."""

    @pytest.mark.asyncio
    async def test_ride_along_survives_on_slot_and_telemetry(self, monkeypatch):
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        slot = _slot("s1", [_chunk("c1", document_id="doc1")])
        shape = FilledShape(slots=[slot])

        result = await synthesis.compile_synthesis(
            "query", shape, db=None,
            verdicts={"s1": SlotVerdict(VERDICT_SATISFIED, "reason", ride_along=True)},
        )

        assert result.slots[0].ride_along is True
        assert result.telemetry.per_slot_ride_along == {"s1": True}

    @pytest.mark.asyncio
    async def test_exhausted_budget_and_exhausted_attempts_not_collapsed(self, monkeypatch):
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        s1 = _slot("s1", [_chunk("a", document_id="doc1", text="alpha content")])
        s2 = _slot("s2", [_chunk("b", document_id="doc2", text="beta content")])
        shape = FilledShape(slots=[s1, s2])

        result = await synthesis.compile_synthesis(
            "query", shape, db=None,
            verdicts={
                "s1": SlotVerdict("EXHAUSTED_BUDGET", "clock_cutoff"),
                "s2": SlotVerdict("EXHAUSTED_ATTEMPTS", "strategy_limit_reached"),
            },
        )

        verdict_by_slot = {s.slot_id: s.verdict for s in result.slots}
        assert verdict_by_slot == {"s1": "EXHAUSTED_BUDGET", "s2": "EXHAUSTED_ATTEMPTS"}

    @pytest.mark.asyncio
    async def test_model_trace_threaded_through_when_supplied(self, monkeypatch):
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        slot = _slot("s1", [_chunk("c1", document_id="doc1", assignment_reason="llm_retrieved")])
        shape = FilledShape(slots=[slot])

        result = await synthesis.compile_synthesis(
            "query", shape, db=None,
            filler_emit_by_slot={"s1": {"model_used": "claude-sonnet-5", "llm_call_id": "call-123"}},
        )

        assert result.slots[0].model_trace == {
            "model_used": "claude-sonnet-5", "llm_call_id": "call-123",
        }

    @pytest.mark.asyncio
    async def test_model_trace_includes_stage_when_supplied(self, monkeypatch):
        """Eval's ruling, 2026-07-24: stage is preserve-don't-drop attribution
        (the filler already knows it, same class as model_used/llm_call_id),
        not speculative machinery -- kept so a future "does model quality
        diverge by stage" analysis stays possible without a re-plumb."""
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        slot = _slot("s1", [_chunk("c1", document_id="doc1", assignment_reason="llm_retrieved")])
        shape = FilledShape(slots=[slot])

        result = await synthesis.compile_synthesis(
            "query", shape, db=None,
            filler_emit_by_slot={"s1": {
                "stage": "rag_strategy_c_llm_retrieval",
                "model_used": "claude-sonnet-5", "llm_call_id": "call-123",
                "latency_ms": 842,  # deliberately supplied but must NOT be copied through
            }},
        )

        assert result.slots[0].model_trace == {
            "stage": "rag_strategy_c_llm_retrieval",
            "model_used": "claude-sonnet-5", "llm_call_id": "call-123",
        }
        assert "latency_ms" not in result.slots[0].model_trace  # Timing's attempt_spans owns this, not model_trace

    @pytest.mark.asyncio
    async def test_model_trace_empty_when_not_supplied(self, monkeypatch):
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        slot = _slot("s1", [_chunk("c1", document_id="doc1")])
        shape = FilledShape(slots=[slot])

        result = await synthesis.compile_synthesis("query", shape, db=None)

        assert result.slots[0].model_trace == {}

    @pytest.mark.asyncio
    async def test_document_name_lookup_miss_logged_and_counted(self, monkeypatch, caplog):
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        slot = _slot("s1", [_chunk("c1", document_id="doc-missing")])
        shape = FilledShape(slots=[slot])

        with caplog.at_level("WARNING", logger="app.services.retriever.synthesis"):
            result = await synthesis.compile_synthesis("query", shape, db=None)

        assert result.telemetry.document_name_lookup_missed == 1
        assert any("document_name lookup missed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_external_chunk_name_miss_not_logged_as_ops_signal(self, monkeypatch, caplog):
        """An external (url-based) chunk has no documents-table row by
        design -- its fallback isn't a data-integrity signal, unlike an
        internal chunk's document_id miss."""
        async def fake_resolve(db, chunk_ids):
            return {}

        async def fake_neighbors(db, slot_items):
            return slot_items, 0, 0

        monkeypatch.setattr(synthesis, "_resolve_document_names", fake_resolve)
        monkeypatch.setattr(synthesis, "_complete_neighbors", fake_neighbors)

        slot = _slot("s1", [_chunk("c1", url="https://example.com/some-page")])
        shape = FilledShape(slots=[slot])

        with caplog.at_level("WARNING", logger="app.services.retriever.synthesis"):
            result = await synthesis.compile_synthesis("query", shape, db=None)

        assert result.telemetry.document_name_lookup_missed == 0
        assert not any("document_name lookup missed" in r.message for r in caplog.records)
