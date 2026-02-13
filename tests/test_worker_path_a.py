"""Unit tests for app.worker.path_a — helper functions + mocked process_paragraph()."""
from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.worker.path_a import (
    _find_fact_span_in_markdown,
    _normalize_whitespace,
    _sanitize_fact_for_db,
)


# ---------------------------------------------------------------------------
# _normalize_whitespace
# ---------------------------------------------------------------------------

def test_normalize_whitespace_collapses():
    assert _normalize_whitespace("  hello   world  ") == "hello world"


def test_normalize_whitespace_empty():
    assert _normalize_whitespace("") == ""
    assert _normalize_whitespace(None) == ""


# ---------------------------------------------------------------------------
# _sanitize_fact_for_db
# ---------------------------------------------------------------------------

def test_sanitize_removes_category_scores():
    data = {"fact_text": "x", "category_scores": {"a": 1}}
    safe = _sanitize_fact_for_db(data)
    assert "category_scores" not in safe
    assert safe["fact_text"] == "x"


def test_sanitize_nan_to_none():
    data = {"confidence": float("nan"), "fact_text": "y"}
    safe = _sanitize_fact_for_db(data)
    assert safe["confidence"] is None


def test_sanitize_inf_to_none():
    data = {"confidence": float("inf")}
    safe = _sanitize_fact_for_db(data)
    assert safe["confidence"] is None


def test_sanitize_bool_to_string():
    data = {"is_verified": True, "is_eligibility_related": False}
    safe = _sanitize_fact_for_db(data)
    assert safe["is_verified"] == "true"
    assert safe["is_eligibility_related"] == "false"


def test_sanitize_none_stays_none():
    data = {"fact_text": None}
    safe = _sanitize_fact_for_db(data)
    assert safe["fact_text"] is None


# ---------------------------------------------------------------------------
# _find_fact_span_in_markdown
# ---------------------------------------------------------------------------

def test_find_span_exact_fallback():
    md = "Hello world, this is a test."
    # _find_fact_span_in_markdown(fact_text, page_md, fallback_start, fallback_end)
    start, end = _find_fact_span_in_markdown("this is a test", md, 13, 27)
    assert start == 13
    assert end == 27


def test_find_span_search_when_offsets_wrong():
    md = "Hello world, this is a test."
    # Wrong offsets — function should find the fact via string search
    start, end = _find_fact_span_in_markdown("this is a test", md, 0, 5)
    assert start is not None
    assert md[start:end] == "this is a test"


def test_find_span_no_match_returns_fallback():
    md = "Hello world."
    start, end = _find_fact_span_in_markdown("xyz not here", md, 0, 5)
    assert start == 0
    assert end == 5


def test_find_span_empty_fact():
    # Empty fact text → returns fallback
    start, end = _find_fact_span_in_markdown("", "some md", None, None)
    assert start is None
    assert end is None


def test_find_span_empty_page():
    # Empty page text → returns fallback
    start, end = _find_fact_span_in_markdown("some fact", "", None, None)
    assert start is None
    assert end is None


# ---------------------------------------------------------------------------
# process_paragraph() — mocked per-paragraph call
# ---------------------------------------------------------------------------

def _make_ctx():
    from app.worker.context import ChunkingRunContext
    db = AsyncMock()
    return ChunkingRunContext(
        db=db,
        document_id="doc-a",
        doc_uuid=uuid4(),
        job_id="job-a",
        total_paragraphs=1,
        total_pages=1,
    )


def _make_chunk():
    return SimpleNamespace(
        id=uuid4(), extraction_status="pending", critique_status="pending", summary=None,
    )


@pytest.mark.asyncio
async def test_process_paragraph_happy_path():
    """process_paragraph: mock extraction + critique; verify chunk update and persistence."""
    from app.worker.path_a import process_paragraph

    ctx = _make_ctx()
    ctx.emit = AsyncMock()
    ctx.send_status = AsyncMock()
    chunk = _make_chunk()

    extraction_result = {
        "summary": "Enrollment requirement",
        "facts": [{"fact_text": "Members must be enrolled", "fact_type": "rule"}],
    }
    critique_result = {"pass": True, "score": 0.9, "feedback": "Good"}

    with patch("app.worker.path_a.stream_extract_facts") as mock_extract, \
         patch("app.worker.path_a.parse_json_response", return_value=extraction_result), \
         patch("app.worker.path_a.critique_extraction", new_callable=AsyncMock, return_value=critique_result), \
         patch("app.worker.path_a.db_handler") as mock_db:

        async def fake_stream(*a, **kw):
            yield '{"summary":"Enrollment requirement","facts":[{"fact_text":"Members must be enrolled"}]}'
        mock_extract.side_effect = fake_stream

        mock_db.persist_facts = AsyncMock(return_value=1)
        mock_db.safe_commit = AsyncMock()
        mock_db.write_embeddable_unit = AsyncMock()

        await process_paragraph(
            ctx, chunk, "1_0", "Members must be enrolled.",
            "## Section\n\nMembers must be enrolled.",
            section_path="Section",
            page_number=1, para_idx=0,
            threshold=0.6, critique_enabled=True, max_retries=0,
            llm=MagicMock(),
        )

    # Chunk should be updated
    assert chunk.extraction_status == "extracted"
    assert chunk.critique_status == "passed"
    assert chunk.summary == "Enrollment requirement"

    # Paragraph result recorded
    assert "1_0" in ctx.results_paragraphs
    assert ctx.results_paragraphs["1_0"]["status"] == "passed"

    # Events emitted
    event_types = [c[0][0] for c in ctx.emit.call_args_list]
    assert "extraction_start" in event_types
    assert "extraction_complete" in event_types
    assert "critique_start" in event_types
    assert "critique_complete" in event_types


@pytest.mark.asyncio
async def test_process_paragraph_extraction_failure_marks_chunk_failed():
    """When extraction fails, chunk status should be 'failed'."""
    from app.worker.path_a import process_paragraph

    ctx = _make_ctx()
    ctx.emit = AsyncMock()
    ctx.send_status = AsyncMock()
    chunk = _make_chunk()

    with patch("app.worker.path_a.stream_extract_facts") as mock_extract, \
         patch("app.worker.path_a.parse_json_response", side_effect=ValueError("bad")), \
         patch("app.worker.path_a.parse_json_response_best_effort", return_value=None), \
         patch("app.worker.path_a.record_paragraph_error", new_callable=AsyncMock), \
         patch("app.worker.path_a.db_handler") as mock_db:

        async def fake_stream(*a, **kw):
            yield "not json"
        mock_extract.side_effect = fake_stream
        mock_db.safe_commit = AsyncMock()

        await process_paragraph(
            ctx, chunk, "1_0", "Some text.", "Some text.",
            page_number=1, para_idx=0, llm=MagicMock(),
        )

    assert chunk.extraction_status == "failed"
    assert ctx.results_paragraphs["1_0"]["status"] == "failed"


@pytest.mark.asyncio
async def test_process_paragraph_no_facts_records_no_facts():
    """When extraction succeeds but returns 0 facts, status should be 'no_facts'."""
    from app.worker.path_a import process_paragraph

    ctx = _make_ctx()
    ctx.emit = AsyncMock()
    ctx.send_status = AsyncMock()
    chunk = _make_chunk()

    extraction_result = {"summary": "Nothing found", "facts": []}

    with patch("app.worker.path_a.stream_extract_facts") as mock_extract, \
         patch("app.worker.path_a.parse_json_response", return_value=extraction_result), \
         patch("app.worker.path_a.db_handler") as mock_db:

        async def fake_stream(*a, **kw):
            yield '{"summary":"Nothing found","facts":[]}'
        mock_extract.side_effect = fake_stream
        mock_db.safe_commit = AsyncMock()

        await process_paragraph(
            ctx, chunk, "1_0", "Some text.", "Some text.",
            page_number=1, para_idx=0, critique_enabled=False, llm=MagicMock(),
        )

    assert chunk.extraction_status == "extracted"
    assert ctx.results_paragraphs["1_0"]["status"] == "no_facts"
