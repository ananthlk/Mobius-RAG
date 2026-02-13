"""Unit tests for app.worker.coordinator — shared paragraph loop."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.worker.config import WorkerConfig

_RAW_MD = "app.services.page_to_markdown.raw_page_to_markdown"


def _make_pages(n=1):
    pages = []
    for i in range(n):
        p = SimpleNamespace(
            page_number=i + 1,
            text_markdown=f"## Section {i+1}\n\nParagraph {i+1} text here.",
            text=f"Paragraph {i+1} text here.",
        )
        pages.append(p)
    return pages


def _mock_chunk():
    return SimpleNamespace(
        id=uuid4(), extraction_status="pending", critique_status="pending", summary=None,
    )


# ---------------------------------------------------------------------------
# Dispatch: Path A vs Path B
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_calls_path_a_process_paragraph_when_extraction_enabled():
    from app.worker.coordinator import run_chunking_loop

    doc_uuid = uuid4()
    pages = _make_pages(1)

    with patch("app.worker.coordinator.path_a") as mock_a, \
         patch("app.worker.coordinator.path_b") as mock_b, \
         patch("app.worker.coordinator.db_handler") as mock_db, \
         patch(_RAW_MD, side_effect=lambda t: t), \
         patch("app.worker.coordinator.split_paragraphs_from_markdown",
               return_value=[{"text": "T"}]):

        mock_a.process_paragraph = AsyncMock()
        mock_b.process_paragraph = AsyncMock()
        mock_b.clear_policy_data = AsyncMock()
        mock_b.prepare_resources = MagicMock(return_value=None)
        mock_b.finalise = AsyncMock()
        mock_db.clear_embeddable_units = AsyncMock(return_value=0)
        mock_db.safe_commit = AsyncMock()
        mock_db.safe_rollback = AsyncMock()
        mock_db.persist_chunk = AsyncMock(return_value=_mock_chunk())

        cfg = WorkerConfig(job_timeout_seconds=None)
        success = await run_chunking_loop(
            str(doc_uuid), doc_uuid, str(uuid4()), pages, AsyncMock(),
            extraction_enabled=True, llm=MagicMock(), worker_cfg=cfg,
        )

    assert success is True
    mock_a.process_paragraph.assert_awaited_once()
    mock_b.process_paragraph.assert_not_awaited()


@pytest.mark.asyncio
async def test_calls_path_b_process_paragraph_when_extraction_disabled():
    from app.worker.coordinator import run_chunking_loop

    doc_uuid = uuid4()
    pages = _make_pages(1)

    with patch("app.worker.coordinator.path_a") as mock_a, \
         patch("app.worker.coordinator.path_b") as mock_b, \
         patch("app.worker.coordinator.db_handler") as mock_db, \
         patch(_RAW_MD, side_effect=lambda t: t), \
         patch("app.worker.coordinator.split_paragraphs_from_markdown",
               return_value=[{"text": "T"}]):

        mock_a.process_paragraph = AsyncMock()
        mock_b.process_paragraph = AsyncMock()
        mock_b.clear_policy_data = AsyncMock()
        mock_b.prepare_resources = MagicMock(return_value=None)
        mock_b.finalise = AsyncMock()
        mock_db.clear_embeddable_units = AsyncMock(return_value=0)
        mock_db.safe_commit = AsyncMock()
        mock_db.safe_rollback = AsyncMock()
        mock_db.persist_chunk = AsyncMock(return_value=_mock_chunk())

        cfg = WorkerConfig(job_timeout_seconds=None)
        success = await run_chunking_loop(
            str(doc_uuid), doc_uuid, str(uuid4()), pages, AsyncMock(),
            extraction_enabled=False, worker_cfg=cfg,
        )

    assert success is True
    mock_b.process_paragraph.assert_awaited_once()
    mock_a.process_paragraph.assert_not_awaited()


# ---------------------------------------------------------------------------
# Shared HierarchicalChunk created before path dispatch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_persist_chunk_called_before_path_dispatch():
    from app.worker.coordinator import run_chunking_loop

    doc_uuid = uuid4()
    pages = _make_pages(1)
    call_order = []

    async def mock_persist(*a, **kw):
        call_order.append("persist_chunk")
        return _mock_chunk()

    async def mock_path_a(*a, **kw):
        call_order.append("path_a")

    with patch("app.worker.coordinator.path_a") as mock_a, \
         patch("app.worker.coordinator.path_b") as mock_b, \
         patch("app.worker.coordinator.db_handler") as mock_db, \
         patch(_RAW_MD, side_effect=lambda t: t), \
         patch("app.worker.coordinator.split_paragraphs_from_markdown",
               return_value=[{"text": "T"}]):

        mock_a.process_paragraph = mock_path_a
        mock_b.clear_policy_data = AsyncMock()
        mock_b.prepare_resources = MagicMock(return_value=None)
        mock_b.finalise = AsyncMock()
        mock_db.clear_embeddable_units = AsyncMock(return_value=0)
        mock_db.safe_commit = AsyncMock()
        mock_db.safe_rollback = AsyncMock()
        mock_db.persist_chunk = mock_persist

        cfg = WorkerConfig(job_timeout_seconds=None)
        await run_chunking_loop(
            str(doc_uuid), doc_uuid, str(uuid4()), pages, AsyncMock(),
            extraction_enabled=True, llm=MagicMock(), worker_cfg=cfg,
        )

    assert call_order == ["persist_chunk", "path_a"]


# ---------------------------------------------------------------------------
# Events: job_start + paragraph events + chunking_complete
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emits_job_start_paragraph_events_and_chunking_complete():
    from app.worker.coordinator import run_chunking_loop

    doc_uuid = uuid4()
    pages = _make_pages(1)
    emitted = []

    with patch("app.worker.coordinator.path_a") as mock_a, \
         patch("app.worker.coordinator.path_b") as mock_b, \
         patch("app.worker.coordinator.db_handler") as mock_db, \
         patch(_RAW_MD, side_effect=lambda t: t), \
         patch("app.worker.coordinator.split_paragraphs_from_markdown",
               return_value=[{"text": "T"}]), \
         patch("app.worker.coordinator.ChunkingRunContext") as MockCtx:

        mock_ctx = AsyncMock()
        mock_ctx.completed_count = 1
        mock_ctx.total_paragraphs = 1
        mock_ctx.total_pages = 1
        mock_ctx.results_paragraphs = {}
        mock_ctx.progress_percent = 100.0
        mock_ctx.upsert_progress = AsyncMock()
        mock_ctx.emit_progress = AsyncMock()

        async def capture_emit(event_type, **kw):
            emitted.append(event_type)
        mock_ctx.emit = capture_emit

        MockCtx.return_value = mock_ctx
        mock_a.process_paragraph = AsyncMock()
        mock_b.clear_policy_data = AsyncMock()
        mock_b.prepare_resources = MagicMock(return_value=None)
        mock_b.finalise = AsyncMock()
        mock_db.clear_embeddable_units = AsyncMock(return_value=0)
        mock_db.safe_commit = AsyncMock()
        mock_db.safe_rollback = AsyncMock()
        mock_db.persist_chunk = AsyncMock(return_value=_mock_chunk())

        cfg = WorkerConfig(job_timeout_seconds=None)
        await run_chunking_loop(
            str(doc_uuid), doc_uuid, str(uuid4()), pages, AsyncMock(),
            extraction_enabled=True, llm=MagicMock(), worker_cfg=cfg,
        )

    assert "job_start" in emitted
    assert "paragraph_start" in emitted
    assert "paragraph_complete" in emitted
    assert "chunking_complete" in emitted


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_timeout_returns_false():
    from app.worker.coordinator import run_chunking_loop
    import asyncio

    doc_uuid = uuid4()
    pages = _make_pages(1)

    async def slow_process(*a, **kw):
        await asyncio.sleep(10)

    with patch("app.worker.coordinator.path_a") as mock_a, \
         patch("app.worker.coordinator.path_b") as mock_b, \
         patch("app.worker.coordinator.db_handler") as mock_db, \
         patch(_RAW_MD, side_effect=lambda t: t), \
         patch("app.worker.coordinator.split_paragraphs_from_markdown",
               return_value=[{"text": "T"}]):

        mock_a.process_paragraph = slow_process
        mock_b.clear_policy_data = AsyncMock()
        mock_b.prepare_resources = MagicMock(return_value=None)
        mock_b.finalise = AsyncMock()
        mock_db.clear_embeddable_units = AsyncMock(return_value=0)
        mock_db.safe_commit = AsyncMock()
        mock_db.safe_rollback = AsyncMock()
        mock_db.persist_chunk = AsyncMock(return_value=_mock_chunk())

        cfg = WorkerConfig(job_timeout_seconds=0.05)
        success = await run_chunking_loop(
            str(doc_uuid), doc_uuid, str(uuid4()), pages, AsyncMock(),
            extraction_enabled=True, llm=MagicMock(), worker_cfg=cfg,
        )

    assert success is False


# ---------------------------------------------------------------------------
# Path B: calls clear_policy_data and finalise
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_path_b_calls_clear_and_finalise():
    from app.worker.coordinator import run_chunking_loop

    doc_uuid = uuid4()
    pages = _make_pages(1)

    with patch("app.worker.coordinator.path_a") as mock_a, \
         patch("app.worker.coordinator.path_b") as mock_b, \
         patch("app.worker.coordinator.db_handler") as mock_db, \
         patch(_RAW_MD, side_effect=lambda t: t), \
         patch("app.worker.coordinator.split_paragraphs_from_markdown",
               return_value=[{"text": "T"}]):

        mock_a.process_paragraph = AsyncMock()
        mock_b.process_paragraph = AsyncMock()
        mock_b.clear_policy_data = AsyncMock()
        mock_b.prepare_resources = MagicMock(return_value="resources")
        mock_b.finalise = AsyncMock()
        mock_db.clear_embeddable_units = AsyncMock(return_value=0)
        mock_db.safe_commit = AsyncMock()
        mock_db.safe_rollback = AsyncMock()
        mock_db.persist_chunk = AsyncMock(return_value=_mock_chunk())

        cfg = WorkerConfig(job_timeout_seconds=None)
        await run_chunking_loop(
            str(doc_uuid), doc_uuid, str(uuid4()), pages, AsyncMock(),
            extraction_enabled=False, worker_cfg=cfg,
        )

    mock_b.clear_policy_data.assert_awaited_once()
    mock_b.finalise.assert_awaited_once()


# ---------------------------------------------------------------------------
# Exception in paragraph processing doesn't crash the loop
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_paragraph_error_does_not_crash_loop():
    from app.worker.coordinator import run_chunking_loop

    doc_uuid = uuid4()
    pages = _make_pages(1)

    with patch("app.worker.coordinator.path_a") as mock_a, \
         patch("app.worker.coordinator.path_b") as mock_b, \
         patch("app.worker.coordinator.db_handler") as mock_db, \
         patch("app.worker.coordinator.record_paragraph_error", new_callable=AsyncMock), \
         patch(_RAW_MD, side_effect=lambda t: t), \
         patch("app.worker.coordinator.split_paragraphs_from_markdown",
               return_value=[{"text": "T1"}, {"text": "T2"}]):

        call_count = 0

        async def failing_then_ok(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("boom")

        mock_a.process_paragraph = failing_then_ok
        mock_b.clear_policy_data = AsyncMock()
        mock_b.prepare_resources = MagicMock(return_value=None)
        mock_b.finalise = AsyncMock()
        mock_db.clear_embeddable_units = AsyncMock(return_value=0)
        mock_db.safe_commit = AsyncMock()
        mock_db.safe_rollback = AsyncMock()
        mock_db.persist_chunk = AsyncMock(return_value=_mock_chunk())

        cfg = WorkerConfig(job_timeout_seconds=None)
        success = await run_chunking_loop(
            str(doc_uuid), doc_uuid, str(uuid4()), pages, AsyncMock(),
            extraction_enabled=True, llm=MagicMock(), worker_cfg=cfg,
        )

    # Loop continues past the first paragraph error
    assert success is True
    assert call_count == 2
