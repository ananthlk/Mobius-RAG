"""DB-backed integration tests for Pool (Step 2): app/services/retriever/pool/.

Same two-layer structure as test_shape_reformat.py: pure unit coverage
lives in test_pool_dedup.py; this file is real run_gate() + real
PublicSourceAdapter + real run_pool_for_query() against live dev data,
pinned assertions -- not just a manual smoke script.
"""

from __future__ import annotations

import pytest

from app.database import AsyncSessionLocal
from app.services.retriever.shape.contracts import ResourcePosture
from app.services.retriever.shape.gate import run_gate
from app.services.retriever.pool.public_adapter import PublicSourceAdapter
from app.services.retriever.pool.pool import run_pool_for_query, run_pool_fanout


def _rp(breadth: int = 5) -> ResourcePosture:
    return ResourcePosture(breadth=breadth, confidence_bar=0.7, max_attempts=1, speed_budget="interactive")


class TestPlanScopedQuery:
    """cmhc001 -- a real j:payor.* match, exercises all three strategies."""

    async def _run(self):
        query = "What is the timely filing deadline for Sunshine Health FL Medicaid claims?"
        async with AsyncSessionLocal() as db:
            gate = await run_gate(db, query)
            adapter = PublicSourceAdapter(db)
            return await run_pool_for_query(db, query, gate, _rp(), adapter)

    async def test_all_three_strategies_contribute(self):
        result = await self._run()
        assert "tag_select" in result.strategy_hint
        assert "vector" in result.strategy_hint
        assert "inherited" in result.strategy_hint

    async def test_no_duplicate_chunk_ids(self):
        result = await self._run()
        ids = [c.chunk_id for c in result.candidates]
        assert len(ids) == len(set(ids))

    async def test_neighbor_expansion_bounded_not_truncating_matches(self):
        """Regression test for the real bug found 2026-07-23: passing the
        full wide union (378 candidates) directly to _expand_with_neighbors
        silently truncated seeds down to its shared _NEIGHBOR_TOTAL_CAP=50,
        dropping 328 real matches. Pool must bound the neighbor-expansion
        input and preserve the rest of the union untouched."""
        result = await self._run()
        non_neighbor = [c for c in result.candidates if not c.is_neighbor]
        # All match candidates must survive -- none silently dropped by the
        # shared neighbor-cap machinery.
        assert len(non_neighbor) > 100

    async def test_every_segment_timed(self):
        result = await self._run()
        for key in ("doc_narrow_ms", "tag_select_ms", "embed_ms", "vector_ms", "inherited_ms", "dedup_ms", "neighbor_ms"):
            assert key in result.segment_ms, f"missing segment: {key}"

    async def test_not_fallback(self):
        result = await self._run()
        assert result.fallback_triggered is False

    async def test_bm25_score_present_on_matches_none_on_neighbors(self):
        """Filler a's additive ranking field (2026-07-23, Retriever-confirmed
        contract): every match candidate regardless of source arm gets a
        real bm25_score; neighbors get None (no term-match reason for being
        in the pool, same convention as the existing `score` field)."""
        result = await self._run()
        matches = [c for c in result.candidates if not c.is_neighbor]
        neighbors = [c for c in result.candidates if c.is_neighbor]
        assert matches, "expected at least one match candidate"
        assert all(c.bm25_score is not None for c in matches)
        assert all(isinstance(c.bm25_score, float) for c in matches)
        assert neighbors, "expected at least one neighbor candidate"
        assert all(c.bm25_score is None for c in neighbors)
        # Regression guard for the real bug found 2026-07-23 (Payor-Policy's
        # live-trace report): a bare plainto_tsquery AND-joins every content
        # word in the raw question, so ts_rank_cd against an arbitrary
        # (non-full-text-filtered) candidate set reads a flat 0.0 for nearly
        # everything -- confirmed live, 0/581 real candidates scored nonzero
        # before the OR-tsquery fix. "not None" alone doesn't catch this.
        nonzero = [c for c in matches if c.bm25_score > 0]
        assert len(nonzero) > 10, (
            f"only {len(nonzero)}/{len(matches)} matches scored nonzero bm25 -- "
            "suspect the AND-tsquery regression (see comment above)"
        )

    async def test_query_embedding_exposed_for_reuse(self):
        """Filler s's (Payor Platform) reuse request, 2026-07-23, Retriever-
        relayed and independently verified: gemini-embedding-001 @
        output_dimensionality=1536 matches the Payor Fact Store's own
        vector(1536) schema exactly, so PoolResult should expose the
        already-computed query embedding rather than force a second,
        redundant embed call."""
        result = await self._run()
        assert result.query_embedding is not None
        assert len(result.query_embedding) == 1536
        assert all(isinstance(x, float) for x in result.query_embedding)

    async def test_phrase_buckets_exclude_drop_and_shape_matches_filler_a(self):
        """Filler A's meta_boost signal (2026-07-23, shape verified directly
        against their filler_a.py, not just relayed): required_phrases/
        boosted_phrases must be list[tuple[str, float]], DROP-bucket phrases
        (bare "claims"/"medicaid", selectivity < 0.40) excluded entirely --
        confirmed live these were exactly the generic terms diluting bm25
        ranking before this fix."""
        result = await self._run()
        assert result.required_phrases, "expected at least one REQUIRED phrase"
        assert result.boosted_phrases, "expected at least one BOOSTED phrase"
        for phrase, weight in result.required_phrases + result.boosted_phrases:
            assert isinstance(phrase, str)
            assert isinstance(weight, float)
            assert 0.0 <= weight <= 1.0
        required_terms = {p for p, _ in result.required_phrases}
        boosted_terms = {p for p, _ in result.boosted_phrases}
        assert "timely filing" in required_terms
        assert "sunshine health" in required_terms
        # "medicaid" only ever comes from j:program.medicaid (DROP, sel=0.387)
        # -- unlike "claims", which also legitimately belongs to a separate,
        # more-selective code (d:claims.general, BOOSTED) and correctly
        # survives via this module's max-selectivity-across-codes resolution.
        assert "medicaid" not in required_terms and "medicaid" not in boosted_terms
        # REQUIRED and BOOSTED must be selectivity-ordered (REQUIRED >= 0.65 > BOOSTED >= 0.40).
        assert all(w >= 0.65 for _, w in result.required_phrases)
        assert all(0.40 <= w < 0.65 for _, w in result.boosted_phrases)

    async def test_vector_arm_ef_search_matches_width(self):
        """Regression test for the real bug found 2026-07-23 (Filler b's
        live-trace report): app/database.py sets hnsw.ef_search=100 as a
        connection default, but vector_search()'s LIMIT (width) can run up
        to breadth*100 -- 10x what ef_search=100 explores during HNSW graph
        traversal. Confirmed live: at ef_search=100, the TRUE best semantic
        match for this exact query was completely ABSENT from the top-1000
        results -- HNSW silently backfills with worse matches rather than
        just returning fewer good ones. Fixed via SET LOCAL hnsw.ef_search
        scaled to width before the query runs."""
        result = await self._run()
        vector_matches = [c for c in result.candidates if c.source_arm == "vector"]
        best = max(vector_matches, key=lambda c: c.score or 0)
        assert best.score > 0.85, (
            f"best vector match scored only {best.score} -- suspect ef_search "
            "regression (true best match for this query scores ~0.886-0.889)"
        )
        assert "180 day" in (best.text or "").lower() or "365 day" in (best.text or "").lower()

    async def test_vector_similarity_attached_regardless_of_provenance(self):
        """Regression test for the real structural bug found 2026-07-23
        (Retriever's live-trace report): dedup_candidates() is first-arm-wins
        on chunk_id collision (union order tag_select -> vector -> inherited),
        so a chunk found by BOTH tag_select and vector_search only kept
        tag_select's provenance -- any filler that filters to
        source_arm=="vector" (e.g. Filler b) never saw it, even when
        vector_search independently found the identical chunk with a strong
        similarity score. Every final candidate -- matches AND neighbors --
        must carry a real vector_similarity regardless of which arm's
        provenance survived the union."""
        result = await self._run()
        non_vector_matches = [c for c in result.candidates if not c.is_neighbor and c.source_arm != "vector"]
        neighbors = [c for c in result.candidates if c.is_neighbor]
        assert non_vector_matches, "expected at least one tag_select/inherited candidate"
        assert neighbors, "expected at least one neighbor candidate"
        assert all(c.vector_similarity is not None for c in non_vector_matches)
        assert all(isinstance(c.vector_similarity, float) for c in non_vector_matches)
        assert all(0.0 <= c.vector_similarity <= 1.0 for c in non_vector_matches)
        assert all(c.vector_similarity is not None for c in neighbors)


class TestCrossPayerExclusion:
    """Regression test for the real correctness bug found 2026-07-23
    (Payor-Policy's live-trace report): vector_search() had ZERO payer
    scoping, unlike tag_select. Confirmed live: a Sunshine-Health-specific
    query surfaced 10 candidates in the top-1000 tagged ONLY with
    payor.aetna/payor.molina_healthcare (no Sunshine/Centene co-tag) --
    genuinely different payers' policies shown as if relevant. Fixed with a
    NEGATIVE exclusion (chunks tagged with a DIFFERENT payer than the one
    the query names are excluded), not a positive scope filter -- generic/
    no-payor content must still pass through untouched."""

    async def test_no_cross_payer_contamination(self):
        query = "How do I submit a corrected claim to Sunshine Health Florida?"
        async with AsyncSessionLocal() as db:
            gate = await run_gate(db, query)
            assert "j:payor.sunshine_health" in gate.j_codes
            adapter = PublicSourceAdapter(db)
            candidates, _segment_ms, _embedding = await adapter.vector_search(
                query, gate.expansion_phrases, gate.j_codes, width=1000
            )
            # NOTE (caught in this test, not the SQL fix): PoolCandidate.tags
            # merges chunk_d_tags/chunk_p_tags/chunk_j_tags into one dict,
            # losing which namespace each key came from. "payor.hmo_plan" is
            # a D-tag (content ABOUT HMO plans generally), not a J-tag payer
            # identity, despite sharing the "payor." prefix -- confirmed by
            # querying chunk_d_tags/chunk_j_tags separately. The SQL exclusion
            # only ever checked chunk_j_tags (correctly), so it never touched
            # this D-tag; only this test's flattened-tags check needed the
            # exclusion, which is why the code was right and the first draft
            # of this test wasn't.
            wrong_payer = [
                c for c in candidates
                if any(
                    k.startswith("payor.") and k not in ("payor.sunshine_health", "payor.centene", "payor.hmo_plan")
                    for k in (c.tags or {})
                )
            ]
            assert wrong_payer == [], f"found {len(wrong_payer)} candidates tagged with a different payer"

    async def test_generic_no_payor_content_still_passes_through(self):
        """The exclusion must not become an accidental positive scope
        filter -- content with no payor tag at all (AHCA-authority docs,
        generic policy) still needs to reach the vector arm."""
        query = "How do I submit a corrected claim to Sunshine Health Florida?"
        async with AsyncSessionLocal() as db:
            gate = await run_gate(db, query)
            adapter = PublicSourceAdapter(db)
            candidates, _segment_ms, _embedding = await adapter.vector_search(
                query, gate.expansion_phrases, gate.j_codes, width=1000
            )
            no_payor = [c for c in candidates if not any(k.startswith("payor.") for k in (c.tags or {}))]
            assert len(no_payor) > 100

    async def test_no_exclusion_when_query_names_no_payor(self):
        """A query with no matched payer tag gets no exclusion at all --
        j_payors is empty, the SQL's `CAST(:j_payors AS text[]) = '{}'`
        branch short-circuits."""
        query = "What are the general prior authorization requirements for behavioral health services?"
        async with AsyncSessionLocal() as db:
            gate = await run_gate(db, query)
            assert not any(c.startswith("j:payor.") for c in gate.j_codes)
            adapter = PublicSourceAdapter(db)
            candidates, _segment_ms, _embedding = await adapter.vector_search(
                query, gate.expansion_phrases, gate.j_codes, width=500
            )
            any_payor_tagged = [c for c in candidates if any(k.startswith("payor.") for k in (c.tags or {}))]
            assert any_payor_tagged, "expected at least some payor-tagged content when no exclusion is active"


class TestZeroBreadthQueryEmbedding:
    async def test_none_when_vector_arm_never_runs(self):
        async with AsyncSessionLocal() as db:
            gate = await run_gate(db, "asdkjaslkdj nonsense query zzz")
            adapter = PublicSourceAdapter(db)
            result = await run_pool_for_query(db, "q", gate, _rp(breadth=0), adapter)
            assert result.query_embedding is None


class TestNoPayorQuery:
    """A query with a d-tag but no j:payor.* match -- inherited() must
    correctly no-op (the "default to AHCA" behavior lives in tag_select's
    own cascade substitution, not in inherited() -- pool-schematic-spec.md
    S1 correction)."""

    async def test_inherited_contributes_nothing_without_payor_tag(self):
        query = "What are the general prior authorization requirements for behavioral health services?"
        async with AsyncSessionLocal() as db:
            gate = await run_gate(db, query)
            assert not any(c.startswith("j:payor.") for c in gate.j_codes), (
                "test query must have no payor tag to exercise this path -- "
                f"got j_codes={gate.j_codes}"
            )
            adapter = PublicSourceAdapter(db)
            candidates, segment_ms = await adapter.inherited(query, gate.expansion_phrases, gate.j_codes, width=100)
            assert candidates == []
            assert segment_ms["inherited_ms"] == 0


class TestZeroBreadthPosture:
    """CLARIFY/DECLINE postures reach Pool with an all-zero ResourcePosture
    (same convention Structure uses) -- must short-circuit cleanly, no DB
    calls, not an error."""

    async def test_zero_breadth_returns_empty_fallback(self):
        async with AsyncSessionLocal() as db:
            gate = await run_gate(db, "asdkjaslkdj nonsense query zzz")
            adapter = PublicSourceAdapter(db)
            result = await run_pool_for_query(db, "q", gate, _rp(breadth=0), adapter)
            assert result.candidates == []
            assert result.fallback_triggered is True
            assert result.segment_ms == {}


class TestFanOut:
    """Multiple rewritten_queries run concurrently -- FAN_OUT resolution
    (pool-schematic-spec.md S1: gemini-embedding-001 caps at 1/call, so
    concurrency across queries is the lever, not batching within one)."""

    async def test_two_queries_both_produce_results(self):
        queries = [
            "What is the timely filing deadline for Sunshine Health FL Medicaid claims?",
            "What are the general prior authorization requirements for behavioral health services?",
        ]
        async with AsyncSessionLocal() as db:
            gate = await run_gate(db, queries[0])
            adapter = PublicSourceAdapter(db)
            results = await run_pool_fanout(db, queries, gate, _rp(), adapter)
            assert len(results) == 2
            assert results[0].query == queries[0]
            assert results[1].query == queries[1]
