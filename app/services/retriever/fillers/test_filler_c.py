"""Unit tests for Filler c (LLM Retrieval strategy).

Covers the deterministic, I/O-free pieces directly (JSON parsing, citation
coercion, title tokenization/overlap, ValidatedCitation -> FilledChunk
reshaping) plus the slot-orchestration logic in fill_shape_llm_retrieval
(monkeypatching the real LLM+DB call, _run_llm_retrieval, so the test stays
fast/deterministic -- the live call itself is exercised separately, see
module docstring in filler_c.py for the real end-to-end run performed
2026-07-23 against this dev environment's real DB + real Vertex fallback).
"""

import pytest

from app.services.retriever.fillers import filler_c
from app.services.retriever.fillers.filler_c import (
    CitationCandidate,
    ValidatedCitation,
    _LocateResult,
    _chunk_from_citation,
    _coerce_citation,
    _overlap_coefficient,
    _parse_llm_json,
    _quote_present,
    _run_llm_retrieval,
    _section_topic,
    _tokenize_title,
    fill_shape_llm_retrieval,
)
from app.services.retriever.shape.slots import AnswerShapeResult, AnswerSlot


# ---------------------------------------------------------------------------
# _parse_llm_json
# ---------------------------------------------------------------------------


def test_parse_llm_json_clean():
    raw = '{"answer": "yes", "citations": []}'
    assert _parse_llm_json(raw) == {"answer": "yes", "citations": []}


def test_parse_llm_json_markdown_fence():
    raw = '```json\n{"answer": "yes", "citations": []}\n```'
    assert _parse_llm_json(raw) == {"answer": "yes", "citations": []}


def test_parse_llm_json_extracts_first_object_on_garbage_prefix():
    raw = 'Sure, here is the answer:\n{"answer": "yes", "citations": []}'
    assert _parse_llm_json(raw) == {"answer": "yes", "citations": []}


def test_parse_llm_json_unparseable_sets_parse_error():
    result = _parse_llm_json("not json at all")
    assert result["_parse_error"] is True
    assert result["answer"] == ""
    assert result["citations"] == []


# ---------------------------------------------------------------------------
# _coerce_citation
# ---------------------------------------------------------------------------


def test_coerce_citation_requires_title_or_url():
    assert _coerce_citation({"quote": "some quote", "page": 3}) is None


def test_coerce_citation_valid():
    c = _coerce_citation({
        "document_title": "Sunshine Provider Manual",
        "page": "12",
        "section": "3.2",
        "url": None,
        "quote": "a verbatim quote",
    })
    assert c == CitationCandidate(
        document_title="Sunshine Provider Manual", page=12, section="3.2",
        url=None, quote="a verbatim quote",
    )


def test_coerce_citation_non_dict_returns_none():
    assert _coerce_citation("not a dict") is None


# ---------------------------------------------------------------------------
# Title tokenization / overlap
# ---------------------------------------------------------------------------


def test_tokenize_title_drops_stopwords_and_short_tokens():
    tokens = _tokenize_title("The Sunshine Health Provider Manual of Florida")
    assert "the" not in tokens
    assert "of" not in tokens
    assert "sunshine" in tokens
    assert "provider" in tokens


def test_overlap_coefficient_full_credit_when_shorter_side_subset():
    a = {"sunshine", "manual"}
    b = {"sunshine", "health", "provider", "manual", "florida", "2023"}
    assert _overlap_coefficient(a, b) == 1.0


def test_overlap_coefficient_empty_sets():
    assert _overlap_coefficient(set(), {"x"}) == 0.0
    assert _overlap_coefficient({"x"}, set()) == 0.0


# ---------------------------------------------------------------------------
# _quote_present
# ---------------------------------------------------------------------------


def test_quote_present_true_on_substring_match():
    assert _quote_present(
        "Some preamble text. Claims must be received within 365 calendar days from the date of service. More text.",
        "Claims must be received within 365 calendar days from the date of service.",
    )


def test_quote_present_false_when_topically_different():
    assert not _quote_present(
        "To send claims electronically, all EDI claims must first be forwarded to a clearinghouse.",
        "Claims must be received within 365 calendar days from the date of service.",
    )


def test_quote_present_false_on_empty_inputs():
    assert not _quote_present(None, "some quote")
    assert not _quote_present("some text", None)
    assert not _quote_present("", "")


# ---------------------------------------------------------------------------
# _section_topic
# ---------------------------------------------------------------------------


def test_section_topic_strips_leading_numbers():
    assert _section_topic("10.1 Electronic Claims Submission") == "Electronic Claims Submission"
    assert _section_topic(None) == ""
    assert _section_topic("") == ""


# ---------------------------------------------------------------------------
# ValidatedCitation -> FilledChunk reshaping (the core design work)
# ---------------------------------------------------------------------------


def _citation(status: str, **kwargs) -> ValidatedCitation:
    return ValidatedCitation(candidate=CitationCandidate(), status=status, **kwargs)


def test_reshape_retrieved_produces_chunk_with_real_document_id():
    v = _citation(
        "retrieved", document_id="doc-1", matched_chunk_text="real chunk text",
        matched_page=5, locate_method="title_strict(overlap=0.90)",
    )
    chunk = _chunk_from_citation(v)
    assert chunk is not None
    assert chunk.document_id == "doc-1"
    assert chunk.text == "real chunk text"
    assert chunk.source_type == "llm_hinted_retrieval"
    assert chunk.original_score == 1.0
    assert chunk.assignment_reason == "llm_retrieved"
    assert chunk.is_neighbor is False


def test_reshape_threads_quote_verified_true_for_confirmed_citation():
    """Regression for the producer gap Eval caught 2026-07-23: v already
    carried the correct tri-state, but _chunk_from_citation silently dropped
    it building FilledChunk, so a genuinely quote-confirmed citation was
    ending up verified=False downstream in Synthesis. Both directions of
    that bug covered here and below."""
    v = _citation(
        "retrieved", document_id="doc-1", matched_chunk_text="real chunk text",
        matched_page=5, quote_verified=True,
    )
    chunk = _chunk_from_citation(v)
    assert chunk.quote_verified is True


def test_reshape_threads_quote_verified_none_when_no_quote_given():
    v = _citation(
        "retrieved", document_id="doc-1", matched_chunk_text="real chunk text",
        matched_page=5, quote_verified=None,
    )
    chunk = _chunk_from_citation(v)
    assert chunk.quote_verified is None


def test_reshape_retrieved_external_produces_chunk_with_url_not_document_id():
    v = _citation(
        "retrieved_external", document_id=None, matched_chunk_text="web passage text",
        discovered_source_url="https://example.com/policy",
    )
    chunk = _chunk_from_citation(v)
    assert chunk is not None
    # contracts.py's stated convention: external chunks carry url, not document_id.
    assert chunk.document_id is None
    assert chunk.url == "https://example.com/policy"
    assert chunk.source_type == "external_validated"
    assert chunk.original_score == 0.9
    assert chunk.assignment_reason == "llm_retrieved_external"


def test_reshape_internal_chunk_has_no_url():
    v = _citation(
        "retrieved", document_id="doc-1", matched_chunk_text="real chunk text", matched_page=5,
    )
    chunk = _chunk_from_citation(v)
    assert chunk is not None
    assert chunk.document_id == "doc-1"
    assert chunk.url is None
    assert chunk.page_number == 5


def test_reshape_doc_found_section_missing_produces_medium_confidence_chunk():
    v = _citation(
        "doc_found_section_missing", document_id="doc-2", matched_chunk_text="first chunk of doc",
    )
    chunk = _chunk_from_citation(v)
    assert chunk is not None
    assert chunk.original_score == 0.5
    assert chunk.assignment_reason == "llm_partial_match"


@pytest.mark.parametrize("status", [
    "doc_in_sitemap_not_ingested", "doc_robots_blocked", "doc_not_found",
])
def test_reshape_unretrievable_statuses_produce_no_chunk(status):
    v = _citation(status)  # no matched_chunk_text
    assert _chunk_from_citation(v) is None


def test_reshape_missing_text_produces_no_chunk_even_for_retrieved_status():
    # Defensive: status alone doesn't guarantee a chunk -- matched_chunk_text
    # must actually be present (mirrors legacy corpus_search_agent.py's own
    # `if v.matched_chunk_text and v.status in (...)` guard).
    v = _citation("retrieved", document_id="doc-1", matched_chunk_text=None)
    assert _chunk_from_citation(v) is None


def test_reshape_chunk_id_is_deterministic_for_same_citation():
    v = _citation("retrieved", document_id="doc-1", matched_chunk_text="same text", matched_page=5)
    c1 = _chunk_from_citation(v)
    c2 = _chunk_from_citation(v)
    assert c1.chunk_id == c2.chunk_id


# ---------------------------------------------------------------------------
# fill_shape_llm_retrieval orchestration (monkeypatched _run_llm_retrieval --
# the real LLM+DB call is exercised live separately, not here)
# ---------------------------------------------------------------------------


class _FakeDB:
    """Never actually touched -- _run_llm_retrieval is monkeypatched below."""


# ---------------------------------------------------------------------------
# Regression test: quote-verification gate (real bug found via a live call
# 2026-07-23 -- by_user_query BM25 within doc returned a non-empty but
# topically wrong chunk; ANOTHER real live call after the fix confirmed the
# fallback correctly finds the right one instead. See filler_c.py's
# _run_llm_retrieval / _quote_present for the fix itself; this reproduces
# the exact shape deterministically without needing a live DB/LLM per test run.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_llm_retrieval_rejects_topically_wrong_user_query_match(monkeypatch):
    wrong_topic_text = (
        "To send claims electronically, all EDI claims must first be forwarded to a clearinghouse."
    )
    correct_text = "Claims must be received within 365 calendar days from the date of service."
    quote = correct_text  # LLM's cited quote matches the correct chunk verbatim

    async def fake_ask_llm(query, *, correlation_id):
        return (
            "Sunshine Health claims must be received within 365 calendar days.",
            {"citations": [{
                "document_title": "Sunshine Health Provider Manual",
                "page": 122, "section": "10.1.1 Timely Filing",
                "url": None, "quote": quote,
            }]},
            {"llm_ms": 1, "llm_meta": {"model": "test"}, "parse_error": False},
        )

    async def fake_locate_citation(db, cand):
        return _LocateResult(
            document_id="doc-1", display_name="Sunshine Health Provider Manual",
            sitemap_kind="doc_ingested", locate_method="title_strict(overlap=1.00)",
        )

    async def fake_retrieve_in_doc_by_query(db, document_id, user_query, section):
        # Simulates the real observed failure: BM25-by-raw-query returns a
        # non-empty but WRONG chunk (never tries section_topic because this
        # already returned something, matching the real function's own
        # early-return-on-first-hit behavior).
        return wrong_topic_text, 122, "by_user_query"

    async def fake_retrieve_at_section_page(db, document_id, page, section, quote_arg):
        # The section/quote-anchored fallback chain finds the RIGHT chunk.
        return correct_text, 122, "by_quote_tokens"

    monkeypatch.setattr(filler_c, "_ask_llm", fake_ask_llm)
    monkeypatch.setattr(filler_c, "_locate_citation", fake_locate_citation)
    monkeypatch.setattr(filler_c, "_retrieve_in_doc_by_query", fake_retrieve_in_doc_by_query)
    monkeypatch.setattr(filler_c, "_retrieve_at_section_page", fake_retrieve_at_section_page)

    answer, citations, telemetry = await _run_llm_retrieval(_FakeDB(), "what is the timely filing deadline", agent_id="test")

    assert len(citations) == 1
    v = citations[0]
    # Must have fallen through to the verified chunk, NOT accepted the
    # topically-wrong non-empty result from by_user_query.
    assert v.matched_chunk_text == correct_text
    assert v.status == "retrieved"
    assert v.quote_verified is True  # per Eval's tri-state ruling, synthesis-module-spec.md §9.1


@pytest.mark.asyncio
async def test_run_llm_retrieval_downgrades_status_when_nothing_verifies(monkeypatch):
    quote = "a very specific claim the LLM cited"

    async def fake_ask_llm(query, *, correlation_id):
        return "answer", {"citations": [{
            "document_title": "Some Manual", "page": 1, "section": "1.1",
            "url": None, "quote": quote,
        }]}, {"llm_ms": 1, "llm_meta": {}, "parse_error": False}

    async def fake_locate_citation(db, cand):
        return _LocateResult(document_id="doc-1", sitemap_kind="doc_ingested", locate_method="title_strict(overlap=1.00)")

    async def fake_retrieve_in_doc_by_query(db, document_id, user_query, section):
        return "completely unrelated content", 1, "by_user_query"

    async def fake_retrieve_at_section_page(db, document_id, page, section, quote_arg):
        return "still completely unrelated content", 1, "by_section"

    monkeypatch.setattr(filler_c, "_ask_llm", fake_ask_llm)
    monkeypatch.setattr(filler_c, "_locate_citation", fake_locate_citation)
    monkeypatch.setattr(filler_c, "_retrieve_in_doc_by_query", fake_retrieve_in_doc_by_query)
    monkeypatch.setattr(filler_c, "_retrieve_at_section_page", fake_retrieve_at_section_page)

    answer, citations, telemetry = await _run_llm_retrieval(_FakeDB(), "some query", agent_id="test")

    v = citations[0]
    # Neither attempt contained the quote -- must NOT silently claim
    # "retrieved" for unverified content.
    assert v.status == "doc_found_section_missing"
    assert "not found" in v.notes.lower()
    assert v.quote_verified is False  # given-but-not-matched, distinct from no-quote-given


@pytest.mark.asyncio
async def test_run_llm_retrieval_no_quote_given_yields_none_not_false(monkeypatch):
    """The tri-state's whole point: a citation with no quote at all must read
    as quote_verified=None ("nothing was checked"), NOT False ("checked and
    failed") -- collapsing these was exactly the gap Eval's ruling caught."""

    async def fake_ask_llm(query, *, correlation_id):
        return "answer", {"citations": [{
            "document_title": "Some Manual", "page": 1, "section": None,
            "url": None, "quote": None,  # no quote supplied at all
        }]}, {"llm_ms": 1, "llm_meta": {}, "parse_error": False}

    async def fake_locate_citation(db, cand):
        return _LocateResult(document_id="doc-1", sitemap_kind="doc_ingested", locate_method="title_strict(overlap=1.00)")

    async def fake_retrieve_in_doc_by_query(db, document_id, user_query, section):
        return "whatever chunk text was found", 1, "by_user_query"

    async def fake_retrieve_at_section_page(db, document_id, page, section, quote_arg):
        return None, None, ""

    monkeypatch.setattr(filler_c, "_ask_llm", fake_ask_llm)
    monkeypatch.setattr(filler_c, "_locate_citation", fake_locate_citation)
    monkeypatch.setattr(filler_c, "_retrieve_in_doc_by_query", fake_retrieve_in_doc_by_query)
    monkeypatch.setattr(filler_c, "_retrieve_at_section_page", fake_retrieve_at_section_page)

    answer, citations, telemetry = await _run_llm_retrieval(_FakeDB(), "some query", agent_id="test")

    v = citations[0]
    assert v.status == "retrieved"  # unchanged behavior -- nothing to verify against
    assert v.quote_verified is None  # NOT False -- this is the distinction Synthesis needs


@pytest.mark.asyncio
async def test_fill_shape_threads_required_and_respects_capacity(monkeypatch):
    async def fake_run_llm_retrieval(db, query, *, agent_id, correlation_id=None):
        citations = [
            ValidatedCitation(
                candidate=CitationCandidate(), status="retrieved",
                document_id=f"doc-{i}", matched_chunk_text=f"chunk text {i}", matched_page=i,
            )
            for i in range(5)
        ]
        return "narrative answer", citations, {
            "llm_ms": 100, "validate_ms": 50, "total_ms": 150,
            "model_used": "gemini-2.5-flash", "llm_call_id": "call-123",
            "parse_error": False, "outcome_counts": {},
        }

    monkeypatch.setattr(filler_c, "_run_llm_retrieval", fake_run_llm_retrieval)

    slot = AnswerSlot(slot_id="direct_answer", slot_semantics="direct_answer", capacity=2, required=True)
    shape = AnswerShapeResult(query="q", posture=None, slots=[slot], reason="", slots_ms=0)

    result = await fill_shape_llm_retrieval(None, shape, "q", db=_FakeDB())

    assert len(result.slots) == 1
    filled = result.slots[0]
    assert filled.required is True  # threaded through, per Eval's contracts.py finding
    assert filled.occupancy == 2  # truncated to slot.capacity, not all 5 citations
    assert filled.under_filled is False
    assert result.filling_strategy == "llm_retrieval"
    assert result.emit["per_slot_details"][0]["llm_answer"] == "narrative answer"  # diagnostic-only, not in any chunk
    for chunk in filled.chunks:
        assert chunk.text != "narrative answer"


@pytest.mark.asyncio
async def test_fill_shape_uses_slot_rewritten_query_over_raw_query(monkeypatch):
    seen_queries = []

    async def fake_run_llm_retrieval(db, query, *, agent_id, correlation_id=None):
        seen_queries.append(query)
        return "", [], {
            "llm_ms": 0, "validate_ms": 0, "total_ms": 0,
            "model_used": None, "llm_call_id": None,
            "parse_error": False, "outcome_counts": {},
        }

    monkeypatch.setattr(filler_c, "_run_llm_retrieval", fake_run_llm_retrieval)

    slot = AnswerSlot(
        slot_id="fanout_0", slot_semantics="thematic_exploration", capacity=3,
        rewritten_query="theme-specific rewritten query", required=True,
    )
    shape = AnswerShapeResult(query="original top-level query", posture=None, slots=[slot], reason="", slots_ms=0)

    await fill_shape_llm_retrieval(None, shape, "original top-level query", db=_FakeDB())

    assert seen_queries == ["theme-specific rewritten query"]


@pytest.mark.asyncio
async def test_fill_shape_survives_per_slot_exception_without_crashing(monkeypatch):
    async def fake_run_llm_retrieval(db, query, *, agent_id, correlation_id=None):
        raise RuntimeError("simulated LLM/DB failure")

    monkeypatch.setattr(filler_c, "_run_llm_retrieval", fake_run_llm_retrieval)

    slot = AnswerSlot(slot_id="direct_answer", slot_semantics="direct_answer", capacity=5, required=True)
    shape = AnswerShapeResult(query="q", posture=None, slots=[slot], reason="", slots_ms=0)

    result = await fill_shape_llm_retrieval(None, shape, "q", db=_FakeDB())

    assert result.slots[0].occupancy == 0
    assert result.slots[0].under_filled is True
