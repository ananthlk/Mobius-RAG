"""Unit tests for app.worker.errors.record_paragraph_error."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.worker.context import ChunkingRunContext
from app.worker.errors import record_paragraph_error


def _make_ctx() -> ChunkingRunContext:
    db = AsyncMock()
    return ChunkingRunContext(
        db=db,
        document_id="doc-err-1",
        doc_uuid=uuid4(),
        job_id="job-err-1",
        total_paragraphs=5,
        total_pages=1,
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extraction_json_error_classified():
    ctx = _make_ctx()
    err = json.JSONDecodeError("bad", "", 0)
    with patch("app.worker.errors.log_error", new_callable=AsyncMock) as mock_log, \
         patch("app.worker.errors.classify_error", return_value=("critical", "extraction")):
        # Patch emit so it doesn't actually write
        ctx.emit = AsyncMock()
        await record_paragraph_error(ctx, "1_0", err, context_label="extraction")
        mock_log.assert_awaited_once()
        call_kw = mock_log.call_args[1]
        assert call_kw["error_type"] == "json_parse_error"


@pytest.mark.asyncio
async def test_extraction_llm_failure_classified():
    ctx = _make_ctx()
    err = RuntimeError("LLM timeout")
    with patch("app.worker.errors.log_error", new_callable=AsyncMock) as mock_log, \
         patch("app.worker.errors.classify_error", return_value=("critical", "extraction")):
        ctx.emit = AsyncMock()
        await record_paragraph_error(ctx, "1_0", err, context_label="extraction")
        call_kw = mock_log.call_args[1]
        assert call_kw["error_type"] == "llm_failure"


@pytest.mark.asyncio
async def test_persistence_error_classified():
    ctx = _make_ctx()
    err = RuntimeError("db write fail")
    with patch("app.worker.errors.log_error", new_callable=AsyncMock) as mock_log, \
         patch("app.worker.errors.classify_error", return_value=("warning", "persistence")):
        ctx.emit = AsyncMock()
        await record_paragraph_error(ctx, "1_0", err, context_label="persistence")
        call_kw = mock_log.call_args[1]
        assert call_kw["error_type"] == "persistence_error"


@pytest.mark.asyncio
async def test_other_error_classified():
    ctx = _make_ctx()
    err = ValueError("unexpected")
    with patch("app.worker.errors.log_error", new_callable=AsyncMock) as mock_log, \
         patch("app.worker.errors.classify_error", return_value=("warning", "other")):
        ctx.emit = AsyncMock()
        await record_paragraph_error(ctx, "1_0", err, context_label="other")
        call_kw = mock_log.call_args[1]
        assert call_kw["error_type"] == "other"


# ---------------------------------------------------------------------------
# Records in context
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_records_failed_in_context():
    ctx = _make_ctx()
    err = RuntimeError("fail")
    with patch("app.worker.errors.log_error", new_callable=AsyncMock), \
         patch("app.worker.errors.classify_error", return_value=("warning", "other")):
        ctx.emit = AsyncMock()
        await record_paragraph_error(ctx, "2_0", err)
    assert ctx.results_paragraphs["2_0"]["status"] == "failed"
    assert "fail" in ctx.results_paragraphs["2_0"]["error"]


# ---------------------------------------------------------------------------
# Emits paragraph_failed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emits_paragraph_failed_event():
    ctx = _make_ctx()
    err = RuntimeError("oops")
    with patch("app.worker.errors.log_error", new_callable=AsyncMock), \
         patch("app.worker.errors.classify_error", return_value=("warning", "other")):
        ctx.emit = AsyncMock()
        await record_paragraph_error(ctx, "3_0", err)
    ctx.emit.assert_awaited()
    # Find the paragraph_failed call
    calls = [c for c in ctx.emit.call_args_list if c[0][0] == "paragraph_failed"]
    assert len(calls) == 1
    kw = calls[0][1]
    assert "oops" in kw["message"]
    assert kw["paragraph_id"] == "3_0"


# ---------------------------------------------------------------------------
# Resilience: log_error failure doesn't prevent recording
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_log_error_failure_still_records():
    ctx = _make_ctx()
    err = RuntimeError("x")
    with patch("app.worker.errors.log_error", new_callable=AsyncMock, side_effect=RuntimeError("log broke")), \
         patch("app.worker.errors.classify_error", return_value=("warning", "other")):
        ctx.emit = AsyncMock()
        await record_paragraph_error(ctx, "4_0", err)
    # Should still record the failure
    assert ctx.results_paragraphs["4_0"]["status"] == "failed"
