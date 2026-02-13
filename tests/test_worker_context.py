"""Unit tests for app.worker.context.ChunkingRunContext."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.worker.context import ChunkingRunContext


def _make_ctx(*, total_paragraphs=10, total_pages=2) -> ChunkingRunContext:
    db = AsyncMock()
    return ChunkingRunContext(
        db=db,
        document_id="doc-123",
        doc_uuid=uuid4(),
        job_id="job-456",
        total_paragraphs=total_paragraphs,
        total_pages=total_pages,
    )


# ---------------------------------------------------------------------------
# record_paragraph_result
# ---------------------------------------------------------------------------

def test_record_paragraph_result_stores_entry():
    ctx = _make_ctx()
    ctx.record_paragraph_result("1_0", status="passed", facts=[{"fact_text": "f"}], summary="s")
    assert "1_0" in ctx.results_paragraphs
    assert ctx.results_paragraphs["1_0"]["status"] == "passed"
    assert ctx.results_paragraphs["1_0"]["facts"] == [{"fact_text": "f"}]
    assert ctx.results_paragraphs["1_0"]["summary"] == "s"


def test_record_paragraph_result_failed():
    ctx = _make_ctx()
    ctx.record_paragraph_result("2_0", status="failed", error="boom")
    assert ctx.results_paragraphs["2_0"]["status"] == "failed"
    assert ctx.results_paragraphs["2_0"]["error"] == "boom"


# ---------------------------------------------------------------------------
# completed_count / progress_percent
# ---------------------------------------------------------------------------

def test_completed_count_empty():
    ctx = _make_ctx()
    assert ctx.completed_count == 0


def test_completed_count_after_records():
    ctx = _make_ctx(total_paragraphs=5)
    ctx.record_paragraph_result("1_0", status="passed")
    ctx.record_paragraph_result("1_1", status="failed", error="e")
    assert ctx.completed_count == 2


def test_progress_percent():
    ctx = _make_ctx(total_paragraphs=4)
    ctx.record_paragraph_result("1_0", status="passed")
    ctx.record_paragraph_result("1_1", status="passed")
    assert ctx.progress_percent == 50.0


def test_progress_percent_zero_total():
    ctx = _make_ctx(total_paragraphs=0)
    assert ctx.progress_percent == 0.0


# ---------------------------------------------------------------------------
# emit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_calls_write_event():
    ctx = _make_ctx()
    with patch("app.worker.context.write_event", new_callable=AsyncMock) as mock_we:
        await ctx.emit(
            "test_event",
            message="tech msg",
            user_message="user msg",
            paragraph_id="1_0",
            extra={"key": "val"},
        )
        mock_we.assert_awaited_once()
        call_args = mock_we.call_args
        assert call_args[0][1] == ctx.doc_uuid
        assert call_args[0][2] == "test_event"
        data = call_args[0][3]
        assert data["message"] == "tech msg"
        assert data["user_message"] == "user msg"
        assert data["paragraph_id"] == "1_0"
        assert data["key"] == "val"


# ---------------------------------------------------------------------------
# send_status
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_status_emits_status_message():
    ctx = _make_ctx()
    with patch("app.worker.context.write_event", new_callable=AsyncMock) as mock_we:
        await ctx.send_status("tech", "user friendly")
        mock_we.assert_awaited_once()
        call_args = mock_we.call_args
        assert call_args[0][2] == "status_message"
        data = call_args[0][3]
        assert data["message"] == "tech"
        assert data["user_message"] == "user friendly"


@pytest.mark.asyncio
async def test_send_status_defaults_user_message():
    ctx = _make_ctx()
    with patch("app.worker.context.write_event", new_callable=AsyncMock) as mock_we:
        await ctx.send_status("same for both")
        data = mock_we.call_args[0][3]
        assert data["user_message"] == "same for both"


# ---------------------------------------------------------------------------
# upsert_progress
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upsert_progress_calls_upsert():
    ctx = _make_ctx(total_paragraphs=5, total_pages=2)
    ctx.record_paragraph_result("1_0", status="passed")
    with patch("app.worker.context.upsert_chunking_result", new_callable=AsyncMock, return_value=True) as mock_u:
        result = await ctx.upsert_progress("in_progress")
    assert result is True
    mock_u.assert_awaited_once()
    kw = mock_u.call_args
    assert kw[1]["status"] == "in_progress"
    assert kw[1]["total_paragraphs"] == 5
    assert kw[1]["total_pages"] == 2


# ---------------------------------------------------------------------------
# emit_progress
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_progress_includes_counts():
    ctx = _make_ctx(total_paragraphs=4, total_pages=1)
    ctx.record_paragraph_result("1_0", status="passed")
    ctx.record_paragraph_result("1_1", status="passed")
    with patch("app.worker.context.write_event", new_callable=AsyncMock) as mock_we:
        await ctx.emit_progress("1_1", current_page=1)
        data = mock_we.call_args[0][3]
        assert data["completed_paragraphs"] == 2
        assert data["total_paragraphs"] == 4
        assert data["progress_percent"] == 50.0
        assert data["current_paragraph"] == "1_1"


@pytest.mark.asyncio
async def test_emit_progress_with_error():
    ctx = _make_ctx(total_paragraphs=2)
    with patch("app.worker.context.write_event", new_callable=AsyncMock) as mock_we:
        await ctx.emit_progress("1_0", current_page=1, error="oops")
        data = mock_we.call_args[0][3]
        assert data["error"] == "oops"
