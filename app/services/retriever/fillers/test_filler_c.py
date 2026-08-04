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
# Direct-relay design (2026-08-03): _run_llm_retrieval no longer locates or
# verifies a citation against our own corpus at all -- rag_strategy_c_validate
# is locked to Perplexity (sonar-pro), whose citations are independently
# live-fetched, not parametric LLM memory. Requiring them to ALSO exist in
# our own (necessarily incomplete) ingested corpus was discarding correct,
# WebFetch-verified answers just because our ingestion hadn't caught up
# (confirmed live on cmhc001). The old locate/verify pipeline
# (_locate_citation/_retrieve_in_doc_by_query/_retrieve_at_section_page/
# _quote_present) still exists but is no longer called from here.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_llm_retrieval_relays_citation_with_quote_directly(monkeypatch):
    quote = "Claims must be received within 365 calendar days from the date of service."

    async def fake_ask_llm(query, *, correlation_id):
        return (
            "Sunshine Health claims must be received within 365 calendar days.",
            {"citations": [{
                "document_title": "Sunshine Health Provider Manual",
                "payer": "Sunshine Health",
                "page": 122, "section": "10.1.1 Timely Filing",
                "url": "https://example.com/manual.pdf", "quote": quote,
            }]},
            {"llm_ms": 1, "llm_meta": {"model": "test"}, "parse_error": False},
        )

    monkeypatch.setattr(filler_c, "_ask_llm", fake_ask_llm)

    answer, citations, telemetry = await _run_llm_retrieval(_FakeDB(), "what is the timely filing deadline", agent_id="test")

    assert len(citations) == 1
    v = citations[0]
    # Relayed as-is (payer-prefixed, see the dedicated payer-prefix test
    # below) -- no corpus locate/verify step, no document_id (external).
    assert v.status == "retrieved_external"
    assert v.matched_chunk_text == f"[Sunshine Health] {quote}"
    assert v.discovered_source_url == "https://example.com/manual.pdf"
    assert v.document_id is None
    assert v.locate_method == "llm_direct_relay"


@pytest.mark.asyncio
async def test_run_llm_retrieval_no_quote_given_is_not_relayed(monkeypatch):
    """A citation with no quote at all has nothing to relay -- must be
    dropped (doc_not_found), not served as an empty/fabricated chunk."""

    async def fake_ask_llm(query, *, correlation_id):
        return "answer", {"citations": [{
            "document_title": "Some Manual", "payer": "Sunshine Health",
            "page": 1, "section": None, "url": None, "quote": None,
        }]}, {"llm_ms": 1, "llm_meta": {}, "parse_error": False}

    monkeypatch.setattr(filler_c, "_ask_llm", fake_ask_llm)

    answer, citations, telemetry = await _run_llm_retrieval(_FakeDB(), "some query", agent_id="test")

    v = citations[0]
    assert v.status == "doc_not_found"
    assert _chunk_from_citation(v) is None  # no usable chunk text


@pytest.mark.asyncio
async def test_run_llm_retrieval_prefixes_payer_onto_relayed_text(monkeypatch):
    """The relayed chunk text is just the isolated quote, which rarely
    restates the payer by name -- must be prefixed so must_facts like
    'Sunshine Health is the payer' can be graded from the chunk alone."""

    async def fake_ask_llm(query, *, correlation_id):
        return "answer", {"citations": [{
            "document_title": "Manual", "payer": "Sunshine Health",
            "page": 1, "section": None, "url": "https://example.com/x",
            "quote": "within 180 days of the date of service",
        }]}, {"llm_ms": 1, "llm_meta": {}, "parse_error": False}

    monkeypatch.setattr(filler_c, "_ask_llm", fake_ask_llm)

    answer, citations, telemetry = await _run_llm_retrieval(_FakeDB(), "some query", agent_id="test")

    assert citations[0].matched_chunk_text == "[Sunshine Health] within 180 days of the date of service"


@pytest.mark.asyncio
async def test_chunk_from_citation_hashes_url_and_quote_together(monkeypatch):
    """Real bug fixed 2026-08-03: chunk_id was hash(url) ALONE, so every
    citation from the same page collapsed to the SAME id and a downstream
    dedup silently kept only 1 of N distinct quotes per source (confirmed
    live: 10 real citations -> only 2 survived). Must be hash(url, quote)."""

    async def fake_ask_llm(query, *, correlation_id):
        return "answer", {"citations": [
            {"document_title": "Manual", "payer": "Sunshine Health", "url": "https://example.com/x",
             "quote": "First distinct fact from the same page."},
            {"document_title": "Manual", "payer": "Sunshine Health", "url": "https://example.com/x",
             "quote": "Second, completely different fact from the same page."},
        ]}, {"llm_ms": 1, "llm_meta": {}, "parse_error": False}

    monkeypatch.setattr(filler_c, "_ask_llm", fake_ask_llm)

    answer, citations, telemetry = await _run_llm_retrieval(_FakeDB(), "some query", agent_id="test")
    chunks = [_chunk_from_citation(v) for v in citations]

    assert len(citations) == 2
    assert chunks[0].chunk_id != chunks[1].chunk_id


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
