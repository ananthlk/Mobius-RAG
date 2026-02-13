"""Unit tests for app.worker.path_b — process_paragraph, clear, finalise."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


def _make_ctx():
    from app.worker.context import ChunkingRunContext
    db = AsyncMock()
    return ChunkingRunContext(
        db=db,
        document_id="doc-b",
        doc_uuid=uuid4(),
        job_id="job-b",
        total_paragraphs=2,
        total_pages=1,
    )


def _make_chunk():
    return SimpleNamespace(
        id=uuid4(), extraction_status="skipped", critique_status="skipped", summary=None,
    )


# ---------------------------------------------------------------------------
# clear_policy_data
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_clear_policy_data():
    from app.worker.path_b import clear_policy_data

    ctx = _make_ctx()
    with patch("app.worker.path_b.db_handler") as mock_db:
        mock_db.clear_policy_for_document = AsyncMock()
        await clear_policy_data(ctx)
    mock_db.clear_policy_for_document.assert_awaited_once()


# ---------------------------------------------------------------------------
# prepare_resources
# ---------------------------------------------------------------------------

def test_prepare_resources_loads_phrase_map():
    from app.worker.path_b import prepare_resources
    snapshot = [{"phrase": "test", "p_tag": "t1"}]

    with patch("app.services.policy_path_b.get_phrase_to_tag_map", return_value={"test": "t1"}):
        res = prepare_resources(snapshot)

    assert res.phrase_map == {"test": "t1"}
    assert res.lexicon_snapshot is snapshot


def test_prepare_resources_none_lexicon():
    from app.worker.path_b import prepare_resources

    res = prepare_resources(None)
    assert res.phrase_map is None
    assert res.lexicon_snapshot is None


# ---------------------------------------------------------------------------
# process_paragraph
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_process_paragraph_builds_policy_data():
    from app.worker.path_b import process_paragraph, PathBResources

    ctx = _make_ctx()
    ctx.emit = AsyncMock()
    chunk = _make_chunk()

    para_obj = SimpleNamespace(id=uuid4())
    line_obj = SimpleNamespace(id=uuid4(), text="Test line", p_tags=None, d_tags=None)

    res = PathBResources(
        build_paragraph_and_lines=AsyncMock(return_value=(para_obj, [line_obj])),
        apply_lexicon_to_lines=AsyncMock(return_value=1),
        aggregate_line_tags_to_paragraph=AsyncMock(),
        phrase_map={"test": "t1"},
    )

    with patch("app.worker.path_b.db_handler") as mock_db:
        mock_db.write_embeddable_unit = AsyncMock()
        mock_db.safe_commit = AsyncMock()

        await process_paragraph(
            ctx, chunk, "1_0", "Test line",
            page_number=1, para_idx=0,
            path_b_resources=res,
        )

    res.build_paragraph_and_lines.assert_awaited_once()
    res.apply_lexicon_to_lines.assert_awaited_once()
    res.aggregate_line_tags_to_paragraph.assert_awaited_once()

    assert "1_0" in ctx.results_paragraphs
    assert ctx.results_paragraphs["1_0"]["status"] == "skipped"


@pytest.mark.asyncio
async def test_process_paragraph_writes_embeddable_units():
    from app.worker.path_b import process_paragraph, PathBResources

    ctx = _make_ctx()
    ctx.emit = AsyncMock()
    chunk = _make_chunk()

    para_obj = SimpleNamespace(id=uuid4())
    line1 = SimpleNamespace(id=uuid4(), text="Line one", p_tags={"a": 1}, d_tags=None)
    line2 = SimpleNamespace(id=uuid4(), text="Line two", p_tags=None, d_tags={"b": 1})

    res = PathBResources(
        build_paragraph_and_lines=AsyncMock(return_value=(para_obj, [line1, line2])),
        apply_lexicon_to_lines=AsyncMock(return_value=2),
        aggregate_line_tags_to_paragraph=AsyncMock(),
        phrase_map={"x": "y"},
    )

    with patch("app.worker.path_b.db_handler") as mock_db:
        mock_db.write_embeddable_unit = AsyncMock()
        mock_db.safe_commit = AsyncMock()

        await process_paragraph(
            ctx, chunk, "1_0", "Line one\nLine two",
            page_number=1, para_idx=0,
            path_b_resources=res,
        )

    assert mock_db.write_embeddable_unit.await_count == 2


@pytest.mark.asyncio
async def test_process_paragraph_tag_agg_failure_non_fatal():
    from app.worker.path_b import process_paragraph, PathBResources

    ctx = _make_ctx()
    ctx.emit = AsyncMock()
    chunk = _make_chunk()

    para_obj = SimpleNamespace(id=uuid4())
    line_obj = SimpleNamespace(id=uuid4(), text="Test", p_tags=None, d_tags=None)

    res = PathBResources(
        build_paragraph_and_lines=AsyncMock(return_value=(para_obj, [line_obj])),
        apply_lexicon_to_lines=AsyncMock(return_value=0),
        aggregate_line_tags_to_paragraph=AsyncMock(side_effect=RuntimeError("agg fail")),
        phrase_map=None,
    )

    with patch("app.worker.path_b.db_handler") as mock_db:
        mock_db.write_embeddable_unit = AsyncMock()
        mock_db.safe_commit = AsyncMock()

        # Should not raise
        await process_paragraph(
            ctx, chunk, "1_0", "Test",
            page_number=1, para_idx=0,
            path_b_resources=res,
        )

    # Still recorded as skipped
    assert ctx.results_paragraphs["1_0"]["status"] == "skipped"


# ---------------------------------------------------------------------------
# finalise
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_finalise_runs_doc_aggregation_and_candidates():
    from app.worker.path_b import finalise, PathBResources

    ctx = _make_ctx()
    ctx.emit = AsyncMock()

    res = PathBResources(
        aggregate_paragraph_tags_to_document=AsyncMock(),
        extract_candidates_for_document=AsyncMock(),
        get_phrase_to_tag_map=MagicMock(return_value={"a": "b"}),
        lexicon_snapshot=[{"phrase": "x"}],
        phrase_map={"a": "b"},
    )

    with patch("app.worker.path_b.db_handler") as mock_db:
        mock_db.safe_commit = AsyncMock()
        mock_db.safe_rollback = AsyncMock()

        await finalise(ctx, res)

    res.aggregate_paragraph_tags_to_document.assert_awaited_once()
    res.extract_candidates_for_document.assert_awaited_once()
