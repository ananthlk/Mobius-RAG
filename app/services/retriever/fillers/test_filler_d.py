"""Unit tests for Filler d (Web Search).

Real behavior verification, not "it looks like the original" — pure-logic
pieces (query enrichment, reranking, chunk mapping) are tested directly;
network-touching pieces (_search_web_vertex/_search_web/_fetch_and_extract)
are monkeypatched at the module level so the integration test exercises the
real slot-assignment/emit logic without a live network call.
"""

from __future__ import annotations

import pytest

from app.services.retriever.fillers import filler_d
from app.services.retriever.fillers.filler_d import (
    _Passage,
    _SearchHit,
    _boost_phrases,
    _chunk_from_passage,
    _dedup_hits,
    _dedup_passages_by_text,
    _embed_search_operators,
    _most_specific_d_tag,
    _normalize_bm25_query,
    _rerank_hits,
    _score_bm25,
    _stable_chunk_id,
    _tag_to_phrase,
    build_authoritative_query,
    fill_shape_external,
    prescreen_search,
    should_prescreen_search,
)
from app.services.retriever.fillers.payer_context import PayerContext
from app.services.retriever.pool.contracts import PoolResult
from app.services.retriever.shape.slots import AnswerShapeResult, AnswerSlot


# ---------------------------------------------------------------------------
# Query enrichment
# ---------------------------------------------------------------------------


class TestMostSpecificDTag:
    def test_picks_most_dotted_specific_tag(self):
        tags = ["d:claims.general", "d:claims.timely_filing", "j:payor.sunshine_health"]
        assert _most_specific_d_tag(tags) == "d:claims.timely_filing"

    def test_excludes_generic_leaves(self):
        tags = ["d:claims.general", "d:eligibility.info"]
        assert _most_specific_d_tag(tags) is None

    def test_no_d_tags_returns_none(self):
        assert _most_specific_d_tag(["j:payor.aetna", "p:florida"]) is None

    def test_none_input(self):
        assert _most_specific_d_tag(None) is None

    def test_ties_break_on_longest_code(self):
        tags = ["d:aaa.bb", "d:aaa.longer_leaf"]
        assert _most_specific_d_tag(tags) == "d:aaa.longer_leaf"

    def test_single_segment_generic_tag_excluded(self):
        """Regression guard for a second real bug found during porting
        (same class as _tag_to_phrase's): legacy's ``t.split(".")[-1] not
        in _GENERIC_D_TAG_LEAVES`` check compared the WHOLE single-segment
        string (prefix attached, e.g. "d:general") against a bare-word
        exclusion set -- it never matched, so single-segment generic tags
        silently slipped through as "specific" candidates. Fixed via
        _bare_leaf; this must now correctly exclude them."""
        assert _most_specific_d_tag(["d:general"]) is None
        assert _most_specific_d_tag(["d:general", "d:claims.timely_filing"]) == "d:claims.timely_filing"


class TestTagToPhrase:
    def test_strips_prefix_and_underscores(self):
        assert _tag_to_phrase("d:utilization_management.prior_authorization") == "prior authorization"

    def test_single_segment_strips_prefix(self):
        """Regression guard for a real bug found during porting: legacy's
        ``split(".")[-1]`` only strips "d:" incidentally when a dot exists;
        for a bare single-segment tag it left "d:" attached, producing a
        broken search-quote term. Fixed in this port, not inherited."""
        assert _tag_to_phrase("d:eligibility") == "eligibility"

    def test_generalized_to_p_and_j_prefixes(self):
        """Helper now serves boost-phrase extraction across all three tag
        kinds (2026-07-23), not just d-tags -- verify p:/j: single-segment
        codes strip correctly too, same fix as the d: case above."""
        assert _tag_to_phrase("p:process") == "process"
        assert _tag_to_phrase("j:contact") == "contact"
        assert _tag_to_phrase("j:payor.sunshine_health") == "sunshine health"


class TestBuildAuthoritativeQuery:
    def test_no_domain_no_exact_terms(self):
        """Simplified vs legacy: without a verified domain, never add an
        unanchored exact term (no selectivity DB lookup in this module)."""
        query, domain, terms = build_authoritative_query(
            "timely filing deadline", ["d:claims.timely_filing"], None,
        )
        assert query == "timely filing deadline"
        assert domain is None
        assert terms == []

    def test_domain_present_adds_specific_d_tag_phrase(self):
        query, domain, terms = build_authoritative_query(
            "timely filing deadline", ["d:claims.timely_filing"], "sunshinehealth.com",
        )
        assert domain == "sunshinehealth.com"
        assert terms == ["timely filing"]

    def test_domain_present_but_only_generic_tag(self):
        query, domain, terms = build_authoritative_query(
            "eligibility info", ["d:eligibility.general"], "sunshinehealth.com",
        )
        assert terms == []

    def test_raw_query_never_mutated(self):
        query, _, _ = build_authoritative_query(
            "some query", ["d:claims.timely_filing"], "example.com",
        )
        assert query == "some query"

    def test_p_and_j_tags_never_reach_the_query_string(self):
        """build_authoritative_query only ever contributes a d-tag exact
        term (see module docstring) -- p:/j: signal goes to reranking
        instead (_boost_phrases), never into the live query itself."""
        _, _, terms = build_authoritative_query(
            "q", ["d:claims.timely_filing", "p:process", "j:payor.sunshine_health"], "example.com",
        )
        assert terms == ["timely filing"]


class TestBoostPhrases:
    def test_extracts_all_three_tag_kinds(self):
        phrases = _boost_phrases(["d:claims.timely_filing", "p:prior_authorization", "j:payor.sunshine_health"])
        assert phrases == ["timely filing", "prior authorization", "sunshine health"]

    def test_excludes_generic_leaves(self):
        phrases = _boost_phrases(["d:claims.general", "p:info"])
        assert phrases == []

    def test_ignores_non_dpj_prefixes(self):
        phrases = _boost_phrases(["x:something", "d:claims.timely_filing"])
        assert phrases == ["timely filing"]

    def test_dedupes_preserving_order(self):
        phrases = _boost_phrases(["d:claims.timely_filing", "p:claims.timely_filing"])
        assert phrases == ["timely filing"]

    def test_empty_or_none(self):
        assert _boost_phrases(None) == []
        assert _boost_phrases([]) == []


class TestDedupHits:
    def test_merges_preserving_first_seen_order(self):
        a = [_SearchHit("a1", "", "http://a.com"), _SearchHit("a2", "", "http://b.com")]
        b = [_SearchHit("b1", "", "http://c.com")]
        result = _dedup_hits(a, b)
        assert [h.url for h in result] == ["http://a.com", "http://b.com", "http://c.com"]

    def test_dedupes_across_lists_keeping_first_occurrence(self):
        a = [_SearchHit("first title", "", "http://a.com")]
        b = [_SearchHit("second title (should be dropped)", "", "http://a.com")]
        result = _dedup_hits(a, b)
        assert len(result) == 1
        assert result[0].title == "first title"

    def test_empty_lists(self):
        assert _dedup_hits([], []) == []


class TestDedupPassagesByText:
    def test_keeps_first_occurrence_drops_later_identical_text(self):
        p1 = _Passage(url="http://mirror-a.com", title="", snippet="", text="identical body text", fetch_status="ok", fetch_ms=10)
        p2 = _Passage(url="http://mirror-b.com", title="", snippet="", text="identical body text", fetch_status="ok", fetch_ms=10)
        result = _dedup_passages_by_text([p1, p2])
        assert len(result) == 1
        assert result[0].url == "http://mirror-a.com"

    def test_different_text_both_kept(self):
        p1 = _Passage(url="http://a.com", title="", snippet="", text="text one", fetch_status="ok", fetch_ms=10)
        p2 = _Passage(url="http://b.com", title="", snippet="", text="text two", fetch_status="ok", fetch_ms=10)
        result = _dedup_passages_by_text([p1, p2])
        assert len(result) == 2

    def test_whitespace_only_difference_still_dedupes(self):
        """Compares .strip()'d text -- leading/trailing whitespace
        differences (e.g. from extraction quirks) shouldn't create a false
        non-duplicate."""
        p1 = _Passage(url="http://a.com", title="", snippet="", text="same text", fetch_status="ok", fetch_ms=10)
        p2 = _Passage(url="http://b.com", title="", snippet="", text="  same text  ", fetch_status="ok", fetch_ms=10)
        result = _dedup_passages_by_text([p1, p2])
        assert len(result) == 1

    def test_empty_list(self):
        assert _dedup_passages_by_text([]) == []


class TestRerankHits:
    def test_no_signals_returns_original_order(self):
        hits = [_SearchHit("a", "", "http://x.com"), _SearchHit("b", "", "http://y.com")]
        assert _rerank_hits(hits, None, []) == hits

    def test_boost_terms_alone_can_promote(self):
        """boost_terms (p:/j: tag signal, not embedded in the query) can
        promote a hit on their own, even with no domain/exact_terms."""
        hits = [
            _SearchHit("no match", "generic", "http://a.com"),
            _SearchHit("prior authorization rules", "", "http://b.com"),
        ]
        reranked = _rerank_hits(hits, None, [], boost_terms=["prior authorization"])
        assert reranked[0].url == "http://b.com"

    def test_boost_terms_dont_double_count_exact_terms(self):
        """A term present in BOTH exact_terms and boost_terms (e.g. the
        d-tag that made it into the query also appears in _boost_phrases)
        must only score once. Constructed so double-counting would actually
        change the winner (not just a same-outcome coincidence): hit_a
        matches the overlapping term once; hit_b matches two DISTINCT
        boost-only terms. Correct (deduped) scores: hit_a=1 (exact only,
        boost contribution suppressed), hit_b=2 (boost only) -- hit_b wins.
        If dedup were broken, hit_a would double-count to 2, tying hit_b,
        and the stable sort would keep hit_a first (original order) --
        hit_a would incorrectly win.
        """
        hit_a = _SearchHit("timely filing rules", "", "http://a.com")
        hit_b = _SearchHit("prior authorization and sunshine health plan", "", "http://b.com")
        reranked = _rerank_hits(
            [hit_a, hit_b], None, ["timely filing"],
            boost_terms=["timely filing", "prior authorization", "sunshine health"],
        )
        assert reranked[0].url == "http://b.com"

    def test_domain_match_promoted(self):
        hits = [
            _SearchHit("generic", "", "http://seo-farm.com/a"),
            _SearchHit("real", "", "http://sunshinehealth.com/a"),
        ]
        reranked = _rerank_hits(hits, "sunshinehealth.com", [])
        assert reranked[0].url == "http://sunshinehealth.com/a"

    def test_exact_term_match_promoted(self):
        hits = [
            _SearchHit("no match", "generic content", "http://a.com"),
            _SearchHit("timely filing rules", "", "http://b.com"),
        ]
        reranked = _rerank_hits(hits, None, ["timely filing"])
        assert reranked[0].url == "http://b.com"

    def test_stable_sort_ties_keep_order(self):
        hits = [_SearchHit("x", "", "http://a.com"), _SearchHit("x", "", "http://b.com")]
        reranked = _rerank_hits(hits, None, ["nomatch"])
        assert [h.url for h in reranked] == ["http://a.com", "http://b.com"]


class TestEmbedSearchOperators:
    def test_no_site_no_exact(self):
        assert _embed_search_operators("q", None, []) == "q"

    def test_exact_terms_quoted(self):
        assert _embed_search_operators("q", None, ["timely filing"]) == 'q "timely filing"'

    def test_site_appended(self):
        assert _embed_search_operators("q", "example.com", []) == "q site:example.com"

    def test_both_combined_in_order(self):
        result = _embed_search_operators("q", "example.com", ["a", "b"])
        assert result == 'q "a" "b" site:example.com'


# ---------------------------------------------------------------------------
# Passage -> FilledChunk
# ---------------------------------------------------------------------------


class TestStableChunkId:
    def test_deterministic(self):
        assert _stable_chunk_id("http://a.com") == _stable_chunk_id("http://a.com")

    def test_different_urls_different_ids(self):
        assert _stable_chunk_id("http://a.com") != _stable_chunk_id("http://b.com")

    def test_prefix(self):
        assert _stable_chunk_id("http://a.com").startswith("ext:")


class TestChunkFromPassage:
    def test_maps_fields_correctly_with_real_bm25_score(self):
        p = _Passage(url="http://a.com", title="Title", snippet="snip", text="real content", fetch_status="ok", fetch_ms=100)
        chunk = _chunk_from_passage(p, 0.42)
        assert chunk.text == "real content"
        assert chunk.source_type == "external"
        assert chunk.is_neighbor is False
        assert chunk.original_score == 0.42
        assert chunk.assignment_reason == "external_fetch"
        assert chunk.chunk_id == _stable_chunk_id("http://a.com")
        # url field landed (DB, 2026-07-23): external chunks carry url,
        # NOT a synthetic document_id -- see contracts.py's FilledChunk
        # docstring convention (internal: document_id set, url None;
        # external: url set, document_id None).
        assert chunk.url == "http://a.com"
        assert chunk.document_id is None

    def test_falls_back_to_1_0_when_bm25_score_is_none(self):
        """bm25_score=None means _score_bm25 failed closed (DB error) for
        this URL -- every chunk must still get SOME score, not go unranked."""
        p = _Passage(url="http://a.com", title="", snippet="", text="content", fetch_status="ok", fetch_ms=10)
        chunk = _chunk_from_passage(p, None)
        assert chunk.original_score == 1.0


# ---------------------------------------------------------------------------
# BM25 scoring (real ts_rank_cd against fetched text, inherited from Pool's
# exact ranking function -- see filler_d.py's _score_bm25 docstring)
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _FakeDB:
    """Records the executed SQL/params, returns a scripted result (or
    raises) -- same fake-injection pattern as Filler s's _FakeClient."""

    def __init__(self, rows: list[dict] | None = None, exc: Exception | None = None):
        self._rows = rows or []
        self._exc = exc
        self.calls: list[tuple[str, dict]] = []

    async def execute(self, stmt, params):
        self.calls.append((str(stmt), params))
        if self._exc is not None:
            raise self._exc
        return _FakeResult(self._rows)


class TestNormalizeBm25Query:
    def test_strips_question_lead_phrase(self):
        assert _normalize_bm25_query("What is the timely filing deadline") == "timely filing deadline"

    def test_strips_noise_quantifiers(self):
        """Verified against the real function output, not the source
        docstring's worked example -- that example shows the END-TO-END
        result after Postgres's OWN stopword removal inside plainto_tsquery
        (which drops "an"), a separate stage from this Python-level
        string normalization (which does not touch "an" -- it's neither a
        _QUESTION_LEAD phrase nor a _BM25_NOISE word)."""
        assert _normalize_bm25_query("how many days do I have to file an appeal") == "days file an appeal"

    def test_never_returns_empty(self):
        """Stripping everything would leave nothing useful to query with --
        must fall back to the original text instead."""
        assert _normalize_bm25_query("how many") == "how many"

    def test_preserves_content_words(self):
        result = _normalize_bm25_query("What is the timely filing deadline for Sunshine Health Medicaid claims in Florida?")
        for word in ("timely", "filing", "deadline", "Sunshine", "Health", "Medicaid", "claims", "Florida"):
            assert word in result


class TestScoreBm25:
    @pytest.mark.asyncio
    async def test_empty_passages_no_query(self):
        db = _FakeDB()
        result = await _score_bm25(db, "query", [])
        assert result == {}
        assert db.calls == []  # no DB round trip for zero passages

    @pytest.mark.asyncio
    async def test_batches_all_passages_in_one_call(self):
        db = _FakeDB(rows=[
            {"url": "http://a.com", "score": 0.8},
            {"url": "http://b.com", "score": 0.3},
        ])
        passages = [
            _Passage(url="http://a.com", title="", snippet="", text="alpha", fetch_status="ok", fetch_ms=0),
            _Passage(url="http://b.com", title="", snippet="", text="beta", fetch_status="ok", fetch_ms=0),
        ]
        result = await _score_bm25(db, "query", passages)
        assert result == {"http://a.com": 0.8, "http://b.com": 0.3}
        assert len(db.calls) == 1  # one batched query, not N round trips

    @pytest.mark.asyncio
    async def test_uses_or_semantics_not_plain_and(self):
        """Real bug found + fixed live (not theoretical): a straight
        plainto_tsquery (AND) scored a genuinely relevant fetched passage
        as a hard 0.0, because it didn't repeat every query word verbatim.
        Fixed by rebuilding an OR-joined to_tsquery from plainto_tsquery's
        safely-stemmed/tokenized output (never a hand-built to_tsquery
        string, which throws on raw user text containing !&|():*'). Both
        functions must appear -- plainto_tsquery for safe parsing, wrapped
        in to_tsquery for the OR rebuild."""
        db = _FakeDB(rows=[])
        await _score_bm25(db, "query", [_Passage(url="http://a.com", title="", snippet="", text="t", fetch_status="ok", fetch_ms=0)])
        sql = db.calls[0][0]
        assert "plainto_tsquery" in sql
        assert "to_tsquery('english', replace(" in sql
        assert "' & ', ' | '" in sql  # AND -> OR operator swap

    @pytest.mark.asyncio
    async def test_db_error_fails_closed_not_crashes(self):
        db = _FakeDB(exc=RuntimeError("connection lost"))
        result = await _score_bm25(
            db, "query", [_Passage(url="http://a.com", title="", snippet="", text="t", fetch_status="ok", fetch_ms=0)],
        )
        assert result == {}  # clean fallback, not an exception propagating


# ---------------------------------------------------------------------------
# Integration: fill_shape_external (network calls monkeypatched)
# ---------------------------------------------------------------------------


def _shape(capacity: int = 3, slot_id: str = "external_context") -> AnswerShapeResult:
    return AnswerShapeResult(
        slots=[
            AnswerSlot(
                slot_id=slot_id, slot_semantics="external_context",
                capacity=capacity, required=False, priority=0,
            ),
        ],
    )


def _ok_passage(url: str, text: str | None = None) -> _Passage:
    # Default text is UNIQUE PER URL (embeds the url itself) and
    # comfortably >= _MIN_PASSAGE_LENGTH (50 chars) -- two real fixture
    # bugs found fixing this (2026-07-24): (1) a too-short default got
    # silently caught by the length floor; (2) a SHARED default text
    # across multiple _ok_passage(url) calls for different URLs made
    # _dedup_passages_by_text collapse them into one, breaking every test
    # simulating multiple distinct passages. Pass an explicit `text=` when
    # a test genuinely wants identical content across URLs (e.g. testing
    # the dedup itself).
    if text is None:
        text = f"real fetched content from {url}, comfortably well above the length floor threshold"
    assert len(text) >= 50 + 15, "keep real margin above _MIN_PASSAGE_LENGTH, not a boundary value"
    return _Passage(url=url, title=f"Title for {url}", snippet="", text=text, fetch_status="ok", fetch_ms=50)


async def _fake_ddg_empty(query, *, n, site=None, exact=None):
    """DDG now runs CONCURRENTLY with Vertex (2026-07-23), not only as a
    last-resort fallback -- every test exercising fill_shape_external must
    patch it too, or it hits the real network (only silently safe today
    because CHAT_SKILLS_GOOGLE_SEARCH_URL happens to be unset in this test
    environment; not something to rely on)."""
    return []


# ---------------------------------------------------------------------------
# Speculative pre-fetch: should_prescreen_search / prescreen_search /
# fill_shape_external's prescreened param (Ananth green-lit, 2026-07-23)
# ---------------------------------------------------------------------------


class TestShouldPrescreenSearch:
    def test_any_allows_prescreen(self):
        assert should_prescreen_search("any") is True

    def test_citable_required_blocks_prescreen(self):
        assert should_prescreen_search("citable_required") is False

    def test_unknown_value_defaults_to_allow(self):
        """Only the exact citable_required string blocks -- anything else
        (including a value Structure/Router might add later) fails open to
        allow, matching the fail-open convention this fleet uses elsewhere
        (payer_crawlable=None, etc.) rather than a closed allowlist."""
        assert should_prescreen_search("some_future_value") is True


class TestPrescreenSearch:
    @pytest.mark.asyncio
    async def test_returns_same_hits_as_inline_search_did(self, monkeypatch):
        """prescreen_search must be a pure extraction -- same search logic,
        same merge/dedup/rerank, not a behavior change."""
        async def fake_vertex(query, *, n, site, exact):
            return [_SearchHit("t1", "", "http://a.com"), _SearchHit("t2", "", "http://b.com")]

        monkeypatch.setattr(filler_d, "_search_web_vertex", fake_vertex)
        monkeypatch.setattr(filler_d, "_search_web", _fake_ddg_empty)

        result = await prescreen_search("test query", tag_matches=[])

        assert [h.url for h in result.hits] == ["http://a.com", "http://b.com"]
        assert result.search_backend == "vertex"
        assert result.n_vertex_hits == 2
        assert result.n_ddg_hits == 0

    @pytest.mark.asyncio
    async def test_no_fetch_no_db_touched(self, monkeypatch):
        """Search-only -- must never call _fetch_and_extract or _score_bm25."""
        calls = {"fetch": 0, "bm25": 0}

        async def fake_vertex(query, *, n, site, exact):
            return [_SearchHit("t", "", "http://a.com")]

        async def fake_fetch(hit):
            calls["fetch"] += 1
            return _ok_passage(hit.url)

        async def fake_bm25(db, q, passages):
            calls["bm25"] += 1
            return {}

        monkeypatch.setattr(filler_d, "_search_web_vertex", fake_vertex)
        monkeypatch.setattr(filler_d, "_search_web", _fake_ddg_empty)
        monkeypatch.setattr(filler_d, "_fetch_and_extract", fake_fetch)
        monkeypatch.setattr(filler_d, "_score_bm25", fake_bm25)

        await prescreen_search("test query")

        assert calls == {"fetch": 0, "bm25": 0}


class TestFillShapeExternalPrescreened:
    @pytest.mark.asyncio
    async def test_prescreened_result_skips_search_entirely(self, monkeypatch):
        """The whole point: when a PrescreenedSearch is passed in, neither
        _search_web_vertex nor _search_web should be called at all -- the
        ~5-10s Vertex cost is already sunk."""
        search_calls = {"vertex": 0, "ddg": 0}

        async def fake_vertex(query, *, n, site, exact):
            search_calls["vertex"] += 1
            return []

        async def fake_ddg(query, *, n, site=None, exact=None):
            search_calls["ddg"] += 1
            return []

        async def fake_fetch(hit):
            return _ok_passage(hit.url)

        monkeypatch.setattr(filler_d, "_search_web_vertex", fake_vertex)
        monkeypatch.setattr(filler_d, "_search_web", fake_ddg)
        monkeypatch.setattr(filler_d, "_fetch_and_extract", fake_fetch)

        prescreened = filler_d.PrescreenedSearch(
            hits=[_SearchHit("t", "", "http://prescreened.com")],
            search_backend="vertex", site_domain=None, exact_terms=[], boost_terms=[],
            search_ms=6500, n_vertex_hits=1, n_vertex_unconstrained_hits=0, n_ddg_hits=0,
        )

        result = await fill_shape_external(
            PoolResult(), _shape(capacity=1), "test query", db=None,
            prescreened=prescreened,
        )

        assert search_calls == {"vertex": 0, "ddg": 0}
        assert result.slots[0].occupancy == 1
        assert result.slots[0].chunks[0].chunk_id == filler_d._stable_chunk_id("http://prescreened.com")
        assert result.emit["prescreened"] is True
        assert result.emit["search_ms"] == 6500  # the sunk cost, carried through for observability

    @pytest.mark.asyncio
    async def test_none_prescreened_runs_search_as_before(self, monkeypatch):
        """Default (no prescreened result) -- fully backward compatible,
        runs search inline exactly as before this param existed."""
        async def fake_vertex(query, *, n, site, exact):
            return [_SearchHit("t", "", "http://fresh.com")]

        async def fake_fetch(hit):
            return _ok_passage(hit.url)

        monkeypatch.setattr(filler_d, "_search_web_vertex", fake_vertex)
        monkeypatch.setattr(filler_d, "_search_web", _fake_ddg_empty)
        monkeypatch.setattr(filler_d, "_fetch_and_extract", fake_fetch)

        result = await fill_shape_external(
            PoolResult(), _shape(capacity=1), "test query", db=None,
        )

        assert result.emit["prescreened"] is False
        assert result.slots[0].chunks[0].chunk_id == filler_d._stable_chunk_id("http://fresh.com")


class TestFillShapeExternal:
    @pytest.mark.asyncio
    async def test_tiny_passage_filtered_by_length_floor(self, monkeypatch):
        async def fake_vertex(query, *, n, site, exact):
            return [_SearchHit("t1", "", "http://good.com"), _SearchHit("t2", "", "http://tiny.com")]

        async def fake_fetch(hit):
            if "tiny" in hit.url:
                return _Passage(url=hit.url, title="", snippet="", text="x", fetch_status="ok", fetch_ms=10)
            return _ok_passage(hit.url)

        monkeypatch.setattr(filler_d, "_search_web_vertex", fake_vertex)
        monkeypatch.setattr(filler_d, "_search_web", _fake_ddg_empty)
        monkeypatch.setattr(filler_d, "_fetch_and_extract", fake_fetch)

        result = await fill_shape_external(PoolResult(), _shape(), "test query", db=None)

        assert result.total_chunks_assigned == 1
        assert result.slots[0].chunks[0].chunk_id == filler_d._stable_chunk_id("http://good.com")
        assert result.emit["n_below_length_floor"] == 1
        assert result.emit["n_ok"] == 2  # n_ok counts fetch success, not post-quality-filter survivors

    @pytest.mark.asyncio
    async def test_duplicate_text_across_urls_deduped(self, monkeypatch):
        async def fake_vertex(query, *, n, site, exact):
            return [_SearchHit("t1", "", "http://mirror-a.com"), _SearchHit("t2", "", "http://mirror-b.com")]

        async def fake_fetch(hit):
            return _ok_passage(hit.url, text="identical mirrored content, present on both hosts, well above the length floor")

        monkeypatch.setattr(filler_d, "_search_web_vertex", fake_vertex)
        monkeypatch.setattr(filler_d, "_search_web", _fake_ddg_empty)
        monkeypatch.setattr(filler_d, "_fetch_and_extract", fake_fetch)

        result = await fill_shape_external(PoolResult(), _shape(), "test query", db=None)

        assert result.total_chunks_assigned == 1
        assert result.emit["n_text_duplicates_dropped"] == 1

    @pytest.mark.asyncio
    async def test_vertex_hit_assigns_chunks(self, monkeypatch):
        async def fake_vertex(query, *, n, site, exact):
            return [_SearchHit("t1", "", "http://a.com"), _SearchHit("t2", "", "http://b.com")]

        async def fake_fetch(hit):
            return _ok_passage(hit.url)

        monkeypatch.setattr(filler_d, "_search_web_vertex", fake_vertex)
        monkeypatch.setattr(filler_d, "_search_web", _fake_ddg_empty)
        monkeypatch.setattr(filler_d, "_fetch_and_extract", fake_fetch)

        result = await fill_shape_external(
            PoolResult(), _shape(capacity=3), "test query", db=None, tag_matches=[],
        )

        assert result.total_chunks_assigned == 2
        assert result.slots[0].occupancy == 2
        assert result.slots[0].under_filled is True  # capacity=3, got 2
        assert result.emit["search_backend"] == "vertex"
        assert len(result.emit["passages"]) == 2
        assert result.emit["passages"][0]["url"] == "http://a.com"

    @pytest.mark.asyncio
    async def test_bm25_reorders_by_relevance_not_fetch_order(self, monkeypatch):
        """The first-fetched hit isn't necessarily the most relevant one --
        BM25 scoring (real text relevance) must be able to override raw
        fetch order when deciding which chunk lands in a capacity-limited
        slot, and the winning chunk's original_score must be the real score."""
        async def fake_vertex(query, *, n, site, exact):
            return [_SearchHit("t1", "", "http://a.com"), _SearchHit("t2", "", "http://b.com")]

        async def fake_fetch(hit):
            return _ok_passage(hit.url)

        async def fake_bm25(db, raw_query, passages):
            # b.com fetched SECOND but scores higher -- must still win capacity=1.
            return {"http://a.com": 0.1, "http://b.com": 0.9}

        monkeypatch.setattr(filler_d, "_search_web_vertex", fake_vertex)
        monkeypatch.setattr(filler_d, "_search_web", _fake_ddg_empty)
        monkeypatch.setattr(filler_d, "_fetch_and_extract", fake_fetch)
        monkeypatch.setattr(filler_d, "_score_bm25", fake_bm25)

        result = await fill_shape_external(
            PoolResult(), _shape(capacity=1), "test query", db=None,
        )

        assert result.slots[0].occupancy == 1
        winner = result.slots[0].chunks[0]
        assert winner.chunk_id == filler_d._stable_chunk_id("http://b.com")
        assert winner.original_score == 0.9

    @pytest.mark.asyncio
    async def test_p_and_j_tags_boost_reranking_not_the_query(self, monkeypatch):
        """End-to-end: p:/j: tags reach _rerank_hits as boost_terms and can
        change which hit sorts first, but never get embedded in the actual
        search query/prompt (only the d-tag, and only when domain-anchored)."""
        captured_query = {}

        async def fake_vertex(query, *, n, site, exact):
            captured_query["query"] = query
            captured_query["exact"] = exact
            return [
                _SearchHit("no boost match", "generic content", "http://a.com"),
                _SearchHit("prior authorization for sunshine health", "", "http://b.com"),
            ]

        async def fake_fetch(hit):
            return _ok_passage(hit.url)

        monkeypatch.setattr(filler_d, "_search_web_vertex", fake_vertex)
        monkeypatch.setattr(filler_d, "_search_web", _fake_ddg_empty)
        monkeypatch.setattr(filler_d, "_fetch_and_extract", fake_fetch)

        result = await fill_shape_external(
            PoolResult(), _shape(capacity=2), "coverage question", db=None,
            tag_matches=["p:prior_authorization", "j:payor.sunshine_health"],
            payer_context=None,  # no domain -- so p:/j: must NOT reach the query
        )

        # No domain -> build_authoritative_query contributes zero exact_terms
        # (p:/j: tags never feed the query string, only reranking).
        assert captured_query["exact"] == []
        assert captured_query["query"] == "coverage question"
        # But reranking still picked up the boost signal and promoted b.com.
        assert result.slots[0].chunks[0].chunk_id == filler_d._stable_chunk_id("http://b.com")

    @pytest.mark.asyncio
    async def test_empty_vertex_still_gets_ddg_hits(self, monkeypatch):
        """Vertex and DDG now run CONCURRENTLY (2026-07-23), not
        DDG-only-as-last-resort -- verify DDG hits still surface a chunk
        even when Vertex returns nothing, and the backend label reflects
        DDG as a real contributor, not a "fallback" label."""
        calls = {"vertex": 0, "ddg": 0}

        async def fake_vertex(query, *, n, site, exact):
            calls["vertex"] += 1
            return []

        async def fake_ddg(query, *, n, site=None, exact=None):
            calls["ddg"] += 1
            return [_SearchHit("t", "", "http://fallback.com")]

        async def fake_fetch(hit):
            return _ok_passage(hit.url)

        monkeypatch.setattr(filler_d, "_search_web_vertex", fake_vertex)
        monkeypatch.setattr(filler_d, "_search_web", fake_ddg)
        monkeypatch.setattr(filler_d, "_fetch_and_extract", fake_fetch)

        result = await fill_shape_external(
            PoolResult(), _shape(), "test query", db=None,
        )

        assert calls["vertex"] == 1
        assert calls["ddg"] >= 1
        assert result.emit["search_backend"] == "ddg"
        assert result.total_chunks_assigned == 1

    @pytest.mark.asyncio
    async def test_no_hits_at_all_leaves_slot_empty(self, monkeypatch):
        async def fake_vertex(query, *, n, site, exact):
            return []

        async def fake_ddg(query, *, n, site=None, exact=None):
            return []

        monkeypatch.setattr(filler_d, "_search_web_vertex", fake_vertex)
        monkeypatch.setattr(filler_d, "_search_web", fake_ddg)

        result = await fill_shape_external(PoolResult(), _shape(), "test query", db=None)

        assert result.total_chunks_assigned == 0
        assert result.slots[0].occupancy == 0
        assert result.slots[0].under_filled is True

    @pytest.mark.asyncio
    async def test_failed_fetches_excluded_from_chunks(self, monkeypatch):
        async def fake_vertex(query, *, n, site, exact):
            return [_SearchHit("t1", "", "http://good.com"), _SearchHit("t2", "", "http://bad.com")]

        async def fake_fetch(hit):
            if "bad" in hit.url:
                return _Passage(url=hit.url, title="", snippet="", text="", fetch_status="http_404", fetch_ms=20)
            return _ok_passage(hit.url)

        monkeypatch.setattr(filler_d, "_search_web_vertex", fake_vertex)
        monkeypatch.setattr(filler_d, "_search_web", _fake_ddg_empty)
        monkeypatch.setattr(filler_d, "_fetch_and_extract", fake_fetch)

        result = await fill_shape_external(PoolResult(), _shape(), "test query", db=None)

        assert result.total_chunks_assigned == 1
        assert result.emit["n_ok"] == 1
        assert result.emit["n_fetched"] == 2  # both attempted, only 1 usable

    @pytest.mark.asyncio
    async def test_capacity_truncates_across_slots(self, monkeypatch):
        """Two slots, capacity 1 each -- 3 chunks fetched, non-overlapping assignment."""
        async def fake_vertex(query, *, n, site, exact):
            return [_SearchHit(f"t{i}", "", f"http://{i}.com") for i in range(3)]

        async def fake_fetch(hit):
            return _ok_passage(hit.url)

        monkeypatch.setattr(filler_d, "_search_web_vertex", fake_vertex)
        monkeypatch.setattr(filler_d, "_search_web", _fake_ddg_empty)
        monkeypatch.setattr(filler_d, "_fetch_and_extract", fake_fetch)

        shape = AnswerShapeResult(slots=[
            AnswerSlot(slot_id="s0", slot_semantics="external_context", capacity=1, required=False),
            AnswerSlot(slot_id="s1", slot_semantics="external_context", capacity=1, required=False),
        ])
        result = await fill_shape_external(PoolResult(), shape, "test query", db=None)

        assert result.slots[0].occupancy == 1
        assert result.slots[1].occupancy == 1
        assigned_ids = {c.chunk_id for s in result.slots for c in s.chunks}
        assert len(assigned_ids) == 2  # non-overlapping

    @pytest.mark.asyncio
    async def test_payer_context_threads_into_query_not_reresolved(self, monkeypatch):
        """payer_context is a param, not re-resolved -- verify site_domain
        flows into the CONSTRAINED search call (site_domain present ->
        also fires a second, genuinely-unconstrained Vertex call in
        parallel per the 2026-07-23 diversification change; both calls are
        captured separately here, not overwritten in a shared dict)."""
        captured_calls = []

        async def fake_vertex(query, *, n, site, exact):
            captured_calls.append({"site": site, "exact": exact})
            return []

        async def fake_ddg(query, *, n, site=None, exact=None):
            return []

        monkeypatch.setattr(filler_d, "_search_web_vertex", fake_vertex)
        monkeypatch.setattr(filler_d, "_search_web", fake_ddg)

        pc = PayerContext(slug="sunshine_health", display_name="Sunshine Health", site_domain="sunshinehealth.com", crawlable=True)
        await fill_shape_external(
            PoolResult(), _shape(), "timely filing", db=None,
            tag_matches=["d:claims.timely_filing"], payer_context=pc,
        )

        # Two Vertex calls now: one constrained (site_domain + exact_terms),
        # one deliberately unconstrained (site=None, exact=[]) run in
        # parallel for diversity -- both must be present.
        assert len(captured_calls) == 2
        constrained = next(c for c in captured_calls if c["site"] is not None)
        unconstrained = next(c for c in captured_calls if c["site"] is None)
        assert constrained["site"] == "sunshinehealth.com"
        assert constrained["exact"] == ["timely filing"]
        assert unconstrained["exact"] == []

    @pytest.mark.asyncio
    async def test_none_payer_context_fails_open(self, monkeypatch):
        """No payer_context (Gate found no payer tag, or orchestrator
        skipped resolution) -- must not crash, must not add a domain."""
        captured = {}

        async def fake_vertex(query, *, n, site, exact):
            captured["site"] = site
            return []

        async def fake_ddg(query, *, n, site=None, exact=None):
            return []

        monkeypatch.setattr(filler_d, "_search_web_vertex", fake_vertex)
        monkeypatch.setattr(filler_d, "_search_web", fake_ddg)

        result = await fill_shape_external(
            PoolResult(), _shape(), "generic query", db=None, payer_context=None,
        )

        assert captured["site"] is None
        assert result.slots[0].occupancy == 0  # no crash, clean empty result

    @pytest.mark.asyncio
    async def test_required_field_threaded_from_slot(self, monkeypatch):
        async def fake_vertex(query, *, n, site, exact):
            return []

        async def fake_ddg(query, *, n, site=None, exact=None):
            return []

        monkeypatch.setattr(filler_d, "_search_web_vertex", fake_vertex)
        monkeypatch.setattr(filler_d, "_search_web", fake_ddg)

        shape = AnswerShapeResult(slots=[
            AnswerSlot(slot_id="s0", slot_semantics="external_context", capacity=2, required=True),
        ])
        result = await fill_shape_external(PoolResult(), shape, "q", db=None)
        assert result.slots[0].required is True

    @pytest.mark.asyncio
    async def test_deterministic_chunk_ids_across_repeat_calls(self, monkeypatch):
        """Same URL in, same chunk_id out -- across two separate calls (not
        just within one), matching Filler s's 3x-repeat determinism lesson."""
        async def fake_vertex(query, *, n, site, exact):
            return [_SearchHit("t", "", "http://stable.com")]

        async def fake_fetch(hit):
            return _ok_passage(hit.url)

        monkeypatch.setattr(filler_d, "_search_web_vertex", fake_vertex)
        monkeypatch.setattr(filler_d, "_search_web", _fake_ddg_empty)
        monkeypatch.setattr(filler_d, "_fetch_and_extract", fake_fetch)

        r1 = await fill_shape_external(PoolResult(), _shape(), "q", db=None)
        r2 = await fill_shape_external(PoolResult(), _shape(), "q", db=None)

        assert r1.slots[0].chunks[0].chunk_id == r2.slots[0].chunks[0].chunk_id
