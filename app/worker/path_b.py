"""
Path B pipeline: deterministic policy chunking + lexicon tags + candidates.

Called per-paragraph by the coordinator.  Receives an already-persisted
:class:`HierarchicalChunk` and enriches it with PolicyParagraph, PolicyLines,
lexicon tags, and embeddable units.

No LLM calls.  All DB writes via ``worker.db``, events via ``ctx.emit``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.worker.context import ChunkingRunContext
from app.worker.errors import record_paragraph_error
from app.worker import db as db_handler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resources prepared once per job (not per paragraph)
# ---------------------------------------------------------------------------

@dataclass
class PathBResources:
    """Pre-loaded lexicon data, reused across paragraphs."""
    lexicon_snapshot: Any | None = None
    phrase_map: dict | None = None
    # Service functions (lazy-imported once)
    build_paragraph_and_lines: Any = None
    apply_lexicon_to_lines: Any = None
    get_phrase_to_tag_map: Any = None
    extract_candidates_for_document: Any = None
    aggregate_line_tags_to_paragraph: Any = None
    aggregate_paragraph_tags_to_document: Any = None


def prepare_resources(lexicon_snapshot: Any | None) -> PathBResources:
    """Import services and build phrase map — called once per job."""
    from app.services.policy_path_b import (
        build_paragraph_and_lines,
        apply_lexicon_to_lines,
        get_phrase_to_tag_map,
        extract_candidates_for_document,
        aggregate_line_tags_to_paragraph,
        aggregate_paragraph_tags_to_document,
    )

    res = PathBResources(
        lexicon_snapshot=lexicon_snapshot,
        build_paragraph_and_lines=build_paragraph_and_lines,
        apply_lexicon_to_lines=apply_lexicon_to_lines,
        get_phrase_to_tag_map=get_phrase_to_tag_map,
        extract_candidates_for_document=extract_candidates_for_document,
        aggregate_line_tags_to_paragraph=aggregate_line_tags_to_paragraph,
        aggregate_paragraph_tags_to_document=aggregate_paragraph_tags_to_document,
    )

    if lexicon_snapshot is not None:
        res.phrase_map = get_phrase_to_tag_map(lexicon_snapshot)

    return res


# ---------------------------------------------------------------------------
# Pre-processing: clear existing policy data
# ---------------------------------------------------------------------------

async def clear_policy_data(ctx: ChunkingRunContext) -> None:
    """Remove existing policy paragraphs/lines for this document (idempotent)."""
    await db_handler.clear_policy_for_document(ctx.db, ctx.doc_uuid)


# ---------------------------------------------------------------------------
# Per-paragraph processor (called by coordinator)
# ---------------------------------------------------------------------------

async def process_paragraph(
    ctx: ChunkingRunContext,
    chunk,  # HierarchicalChunk — already persisted
    para_id: str,
    paragraph_text: str,
    *,
    section_path: str | None = None,
    page_number: int,
    para_idx: int,
    path_b_resources: PathBResources | None = None,
) -> None:
    """Enrich an existing HierarchicalChunk with policy paragraph/lines + tags.

    Writes PolicyParagraph, PolicyLines, applies lexicon tags, triggers
    forward tag propagation (line -> paragraph), and writes embeddable units.
    Does not raise — errors are recorded via ``record_paragraph_error``.
    """
    doc_id = ctx.document_id
    doc_uuid = ctx.doc_uuid
    db = ctx.db
    res = path_b_resources or prepare_resources(None)

    try:
        # ── Build PolicyParagraph + PolicyLines ───────────────────────
        para_obj, line_objs = await res.build_paragraph_and_lines(
            db, doc_uuid, page_number, para_idx, section_path, paragraph_text,
        )

        # ── Apply lexicon tags to lines ───────────────────────────────
        n_with_tags = 0
        if res.phrase_map and line_objs:
            n_with_tags = await res.apply_lexicon_to_lines(line_objs, res.phrase_map)

        # ── Forward propagation: line tags -> paragraph tags ──────────
        try:
            await res.aggregate_line_tags_to_paragraph(db, para_obj.id)
        except Exception as agg_err:
            logger.warning("[%s] [%s] tag aggregation (non-fatal): %s", doc_id, para_id, agg_err)

        # ── Write embeddable units for each policy line ───────────────
        try:
            for ln in line_objs:
                ln_text = getattr(ln, "text", None) or ""
                if ln_text.strip():
                    await db_handler.write_embeddable_unit(
                        db, doc_uuid, "B", "policy_line", ln.id,
                        ln_text.strip(),
                        page_number=page_number,
                        paragraph_index=para_idx,
                        section_path=section_path,
                        metadata={
                            "p_tags": getattr(ln, "p_tags", None),
                            "d_tags": getattr(ln, "d_tags", None),
                        },
                    )
        except Exception as eu_err:
            logger.warning("[%s] [%s] embeddable_unit write (non-fatal): %s", doc_id, para_id, eu_err)

        await db_handler.safe_commit(db)
        logger.info(
            "[%s] Path B: built 1 paragraph, %s lines for %s%s",
            doc_id, len(line_objs), para_id,
            f" ({n_with_tags} with tags)" if n_with_tags else "",
        )

        # --- Emit user-friendly result event ---
        n_lines = len(line_objs)
        # Gather distinct tag names from tagged lines
        tag_names: list[str] = []
        for ln in line_objs:
            p_tags = getattr(ln, "p_tags", None) or {}
            if isinstance(p_tags, dict):
                tag_names.extend(p_tags.keys())
        unique_tags = sorted(set(tag_names))

        if n_with_tags and unique_tags:
            tag_preview = ", ".join(unique_tags[:5])
            if len(unique_tags) > 5:
                tag_preview += f" (+{len(unique_tags) - 5} more)"
            user_msg = f"Page {page_number}: analyzed {n_lines} lines, {n_with_tags} matched policy terms: {tag_preview}"
        elif n_with_tags:
            user_msg = f"Page {page_number}: analyzed {n_lines} lines, {n_with_tags} matched policy terms"
        else:
            user_msg = f"Page {page_number}: analyzed {n_lines} lines, no policy term matches"

        await ctx.send_status(
            message=f"Path B: {n_lines} lines, {n_with_tags} tagged for {para_id}",
            user_message=user_msg,
        )

    except Exception as policy_err:
        logger.warning("[%s] [%s] Path B policy build/tag (non-fatal): %s", doc_id, para_id, policy_err, exc_info=True)
        await db_handler.safe_rollback(db)

    # Record result (Path B paragraphs are always "skipped" for extraction)
    ctx.record_paragraph_result(para_id, status="skipped", facts=[])


# ---------------------------------------------------------------------------
# Post-processing: document-level aggregation + candidate extraction
# ---------------------------------------------------------------------------

async def finalise(ctx: ChunkingRunContext, resources: PathBResources | None) -> None:
    """Run after all paragraphs: aggregate paragraph tags -> document tags,
    and extract lexicon candidates.
    """
    doc_id = ctx.document_id
    doc_uuid = ctx.doc_uuid
    db = ctx.db
    res = resources or prepare_resources(None)

    # ── Forward propagation: paragraph tags -> document tags ──────────
    await ctx.send_status(
        message="Aggregating paragraph tags to document level...",
        user_message="Aggregating policy tags across all sections into a document summary...",
    )
    try:
        await res.aggregate_paragraph_tags_to_document(db, doc_uuid)
        await db_handler.safe_commit(db)
        logger.info("[%s] Path B: document-level tag aggregation complete", doc_id)
        await ctx.send_status(
            message="Document tag aggregation complete.",
            user_message="Document-level policy tag summary built successfully.",
        )
    except Exception as doc_agg_err:
        logger.warning("[%s] Path B document tag aggregation (non-fatal): %s", doc_id, doc_agg_err, exc_info=True)
        await db_handler.safe_rollback(db)

    # ── Extract lexicon candidates ────────────────────────────────────
    if res.lexicon_snapshot is not None:
        await ctx.send_status(
            message="Extracting lexicon candidates...",
            user_message="Identifying new candidate terms for the policy lexicon...",
        )
        try:
            pm = res.phrase_map or res.get_phrase_to_tag_map(res.lexicon_snapshot)
            await res.extract_candidates_for_document(db, doc_uuid, run_id=None, phrase_map=pm)
            logger.info("[%s] Path B: candidate extraction complete", doc_id)
            await ctx.send_status(
                message="Candidate extraction complete.",
                user_message="Lexicon candidate extraction complete.",
            )
        except Exception as cand_err:
            logger.warning("[%s] Path B candidate extraction (non-fatal): %s", doc_id, cand_err, exc_info=True)
