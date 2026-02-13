"""Unit tests for app.worker.db — all DB functions tested with mocked AsyncSession."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.worker.config import WorkerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_db(
    *,
    scalar_one_or_none=None,
    scalars_all=None,
    execute_side_effect=None,
):
    """Build a mock AsyncSession with common patterns."""
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()

    if execute_side_effect is not None:
        db.execute = AsyncMock(side_effect=execute_side_effect)
    else:
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = scalar_one_or_none
        if scalars_all is not None:
            result_mock.scalars.return_value.all.return_value = scalars_all
        else:
            result_mock.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(return_value=result_mock)

    return db


# ---------------------------------------------------------------------------
# write_event
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_event_adds_and_commits():
    from app.worker.db import write_event

    db = _mock_db()
    doc = uuid4()
    await write_event(db, doc, "test_event", {"key": "val"})
    db.add.assert_called_once()
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_write_event_rolls_back_on_error():
    from app.worker.db import write_event

    db = _mock_db()
    db.commit = AsyncMock(side_effect=RuntimeError("boom"))
    await write_event(db, uuid4(), "fail", {})
    db.rollback.assert_awaited_once()


# ---------------------------------------------------------------------------
# safe_commit / safe_rollback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_safe_commit_success():
    from app.worker.db import safe_commit

    db = _mock_db()
    assert await safe_commit(db) is True
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_safe_commit_failure_rolls_back():
    from app.worker.db import safe_commit

    db = _mock_db()
    db.commit = AsyncMock(side_effect=RuntimeError("fail"))
    assert await safe_commit(db) is False
    db.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_safe_rollback_never_raises():
    from app.worker.db import safe_rollback

    db = _mock_db()
    db.rollback = AsyncMock(side_effect=RuntimeError("rb fail"))
    await safe_rollback(db)  # should not raise


# ---------------------------------------------------------------------------
# set_config_snapshot
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_set_config_snapshot_basic():
    from app.worker.db import set_config_snapshot

    db = _mock_db()
    job = SimpleNamespace(
        threshold="0.6",
        critique_enabled="true",
        max_retries=2,
        extraction_enabled="true",
        generator_id="A",
        prompt_versions={"extraction": "v1"},
        llm_config_version="default",
        chunking_config_snapshot=None,
    )
    snap = await set_config_snapshot(db, job)
    assert snap["threshold"] == "0.6"
    assert snap["generator_id"] == "A"
    assert "snapshot_at" in snap
    assert job.chunking_config_snapshot is snap


@pytest.mark.asyncio
async def test_set_config_snapshot_with_worker_config():
    from app.worker.db import set_config_snapshot

    db = _mock_db()
    job = SimpleNamespace(
        threshold="0.7", critique_enabled=None, max_retries=None,
        extraction_enabled=None, generator_id="B", prompt_versions=None,
        llm_config_version=None, chunking_config_snapshot=None,
    )
    cfg = WorkerConfig(default_threshold=0.7, path_b_cap_ngrams=300)
    snap = await set_config_snapshot(db, job, worker_config=cfg)
    assert "worker_defaults" in snap
    assert snap["worker_defaults"]["default_threshold"] == 0.7
    assert snap["worker_defaults"]["path_b_cap_ngrams"] == 300


# ---------------------------------------------------------------------------
# persist_chunk
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_persist_chunk_creates_new():
    from app.worker.db import persist_chunk

    db = _mock_db(scalar_one_or_none=None)  # no existing chunk
    doc = uuid4()
    chunk = await persist_chunk(db, doc, 1, 0, "Hello world", section_path="Section 1")
    db.add.assert_called_once()
    db.flush.assert_awaited()
    assert chunk.text == "Hello world"
    assert chunk.text_length == 11
    assert chunk.section_path == "Section 1"


@pytest.mark.asyncio
async def test_persist_chunk_returns_existing():
    from app.worker.db import persist_chunk

    existing = SimpleNamespace(
        id=uuid4(), start_offset_in_page=None, text="existing",
    )
    db = _mock_db(scalar_one_or_none=existing)
    doc = uuid4()
    chunk = await persist_chunk(db, doc, 1, 0, "existing", start_offset_in_page=42)
    # Should patch offset
    assert existing.start_offset_in_page == 42
    db.flush.assert_awaited()


# ---------------------------------------------------------------------------
# enqueue_embedding_job
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enqueue_embedding_job_inserts_when_none_pending():
    from app.worker.db import enqueue_embedding_job

    db = _mock_db(scalar_one_or_none=None)
    result = await enqueue_embedding_job(db, uuid4(), "A")
    assert result is True
    db.add.assert_called_once()


@pytest.mark.asyncio
async def test_enqueue_embedding_job_skips_when_pending_exists():
    from app.worker.db import enqueue_embedding_job

    existing = SimpleNamespace(id=uuid4())
    db = _mock_db(scalar_one_or_none=existing)
    result = await enqueue_embedding_job(db, uuid4(), "A")
    assert result is False
    db.add.assert_not_called()


# ---------------------------------------------------------------------------
# write_embeddable_unit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_embeddable_unit_adds_row():
    from app.worker.db import write_embeddable_unit

    db = _mock_db()
    doc = uuid4()
    src = uuid4()
    await write_embeddable_unit(db, doc, "A", "chunk", src, "some text", page_number=1)
    db.add.assert_called_once()
    db.flush.assert_awaited()


# ---------------------------------------------------------------------------
# clear_embeddable_units
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_clear_embeddable_units():
    from app.worker.db import clear_embeddable_units

    result_mock = MagicMock()
    result_mock.rowcount = 5
    db = _mock_db()
    db.execute = AsyncMock(return_value=result_mock)
    count = await clear_embeddable_units(db, uuid4(), generator_id="B")
    assert count == 5
    db.flush.assert_awaited()


# ---------------------------------------------------------------------------
# log_processing_error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_log_processing_error_adds_row():
    from app.worker.db import log_processing_error

    db = _mock_db()
    await log_processing_error(
        db, "doc-id", "1_0", "llm_failure", "critical", "Something broke"
    )
    db.add.assert_called_once()
    db.flush.assert_awaited()


@pytest.mark.asyncio
async def test_log_processing_error_never_raises():
    from app.worker.db import log_processing_error

    db = _mock_db()
    db.add = MagicMock(side_effect=RuntimeError("boom"))
    # Should not raise
    await log_processing_error(db, "doc-id", "1_0", "other", "info", "msg")
