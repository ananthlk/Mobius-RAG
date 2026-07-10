"""
Path B pipeline: deterministic policy chunking + lexicon tags + candidates.

Called per-paragraph by the coordinator.  Receives an already-persisted
:class:`HierarchicalChunk` and enriches it with PolicyParagraph, PolicyLines,
lexicon tags, and embeddable units.

No LLM calls.  All DB writes via ``worker.db``, events via ``ctx.emit``.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.worker.context import ChunkingRunContext
from app.worker.errors import record_paragraph_error
from app.worker import db as db_handler

logger = logging.getLogger(__name__)


async def _trigger_lexicon_cleanup(ctx: ChunkingRunContext, document_id: str, doc_log_id: str) -> None:
    """Best-effort inline candidate cleanup via lexicon-maintenance, surfaced as a
    visible pipeline step. Runs the fast pass (catalog propagation + deterministic
    rules) synchronously so we can report real counts in the status stream, then
    fires the bounded LLM triage of the residue in the background. Never raises."""
    from app import config
    base = getattr(config, "LEXICON_MAINTENANCE_URL", None)
    if not base:
        return
    url = base.rstrip("/") + "/policy/candidates/process-document"
    headers = {"Content-Type": "application/json"}
    if getattr(config, "ADMIN_API_KEY", None):
        headers["X-Admin-Key"] = config.ADMIN_API_KEY
    await ctx.send_status(
        message="Cleaning up lexicon candidates...",
        user_message="Cleaning up new candidate terms (junk removal + dedup)...",
    )
    try:
        import httpx
        async with httpx.AsyncClient(timeout=45.0) as client:
            # Fast pass — catalog + deterministic rules, returns counts.
            resp = await client.post(
                url, json={"document_id": document_id, "llm_chunks": 0, "sync": True}, headers=headers,
            )
            data = resp.json() if resp.status_code == 200 else {}
            cat = int(data.get("catalog_resolved", 0) or 0)
            det = int(data.get("deterministic_rejected_phrases", 0) or 0)
            rem = int(data.get("remaining_proposed", 0) or 0)
            await ctx.send_status(
                message=f"Lexicon cleanup: {cat} known + {det} junk auto-resolved, {rem} remaining.",
                user_message=f"Cleaned {cat + det} candidate terms automatically; {rem} left for LLM review.",
            )
            logger.info("[%s] lexicon cleanup fast-pass: HTTP %s cat=%s det=%s rem=%s",
                        doc_log_id, resp.status_code, cat, det, rem)
            # Background LLM triage of the genuinely-new residue (fire-and-forget).
            try:
                await client.post(url, json={"document_id": document_id, "llm_chunks": 2}, headers=headers)
            except Exception:
                pass
    except Exception as e:
        logger.warning("[%s] lexicon cleanup (non-fatal): %s", doc_log_id, e)
        await ctx.send_status(
            message="Lexicon cleanup skipped (non-fatal).",
            user_message="Candidate cleanup will run in the next batch pass.",
        )


# ---------------------------------------------------------------------------
# Per-doc timing accumulator (lives on PathBResources)
# ---------------------------------------------------------------------------

# Stage labels — short so log lines stay grep-able
_STAGES = (
    "build",      # build_paragraph_and_lines  (1 paragraph + N lines flush)
    "lexicon",    # apply_lexicon_to_lines (in-memory regex over lines)
    "aggregate",  # aggregate_line_tags_to_paragraph (1 SQL UPDATE)
    "bulk_eu",    # write_embeddable_units_bulk (1 multi-VALUES INSERT)
    "commit",     # safe_commit (paragraph-scoped commit)
    "status",     # send_status -> chunking_events INSERT
)


@dataclass
class StageTimings:
    """Accumulates per-stage wall time across all paragraphs in a doc."""
    totals_ms: dict[str, float] = field(default_factory=lambda: {s: 0.0 for s in _STAGES})
    counts: dict[str, int] = field(default_factory=lambda: {s: 0 for s in _STAGES})
    paragraphs: int = 0
    lines: int = 0

    def add(self, stage: str, dt_ms: float) -> None:
        self.totals_ms[stage] = self.totals_ms.get(stage, 0.0) + dt_ms
        self.counts[stage] = self.counts.get(stage, 0) + 1

    def summary(self) -> str:
        n = max(self.paragraphs, 1)
        parts = []
        total = 0.0
        for s in _STAGES:
            t = self.totals_ms.get(s, 0.0)
            total += t
            parts.append(f"{s}={t:.0f}ms({t/n:.1f}/p)")
        return (
            f"paragraphs={self.paragraphs} lines={self.lines} "
            f"total={total:.0f}ms ({total/n:.1f}ms/p) | " + " ".join(parts)
        )


# ---------------------------------------------------------------------------
# Resources prepared once per job (not per paragraph)
# ---------------------------------------------------------------------------

@dataclass
class PathBResources:
    """Pre-loaded lexicon data, reused across paragraphs."""
    lexicon_snapshot: Any | None = None
    phrase_map: dict | None = None
    refuted_map: dict | None = None
    # Service functions (lazy-imported once)
    build_paragraph_and_lines: Any = None
    apply_lexicon_to_lines: Any = None
    get_phrase_to_tag_map: Any = None
    extract_candidates_for_document: Any = None
    aggregate_line_tags_to_paragraph: Any = None
    aggregate_paragraph_tags_to_document: Any = None
    # Per-doc timing accumulator (populated by process_paragraph,
    # logged in finalise()).
    timings: StageTimings = field(default_factory=StageTimings)


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
        res.phrase_map, res.refuted_map = get_phrase_to_tag_map(lexicon_snapshot)

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

    timings = res.timings  # alias for terseness

    try:
        # ── Build PolicyParagraph + PolicyLines ───────────────────────
        t0 = time.perf_counter()
        para_obj, line_objs = await res.build_paragraph_and_lines(
            db, doc_uuid, page_number, para_idx, section_path, paragraph_text,
        )
        timings.add("build", (time.perf_counter() - t0) * 1000.0)

        # ── Apply lexicon tags to lines ───────────────────────────────
        n_with_tags = 0
        t0 = time.perf_counter()
        if res.phrase_map and line_objs:
            n_with_tags = await res.apply_lexicon_to_lines(line_objs, res.phrase_map, res.refuted_map)
        timings.add("lexicon", (time.perf_counter() - t0) * 1000.0)

        # ── Forward propagation: line tags -> paragraph tags ──────────
        # Pass the in-memory ORM objects to skip a SELECT against the
        # 589k-row policy_lines table. Measured 2026-05-01: that SELECT
        # was ~9.7s/paragraph (97% of Path B wall time) — the single
        # biggest cost in chunking. With this kwarg the stage drops to
        # <1ms (pure Python dict merge over 5–80 lines already loaded).
        t0 = time.perf_counter()
        try:
            await res.aggregate_line_tags_to_paragraph(
                db, para_obj.id, lines=line_objs, paragraph=para_obj,
            )
        except Exception as agg_err:
            logger.warning("[%s] [%s] tag aggregation (non-fatal): %s", doc_id, para_id, agg_err)
        timings.add("aggregate", (time.perf_counter() - t0) * 1000.0)

        # ── Write embeddable units for each policy line (bulk) ────────
        t0 = time.perf_counter()
        try:
            bulk_rows: list[dict[str, Any]] = []
            for ln in line_objs:
                ln_text = (getattr(ln, "text", None) or "").strip()
                if not ln_text:
                    continue
                bulk_rows.append(
                    {
                        "document_id": doc_uuid,
                        "generator_id": "B",
                        "source_type": "policy_line",
                        "source_id": ln.id,
                        "text": ln_text,
                        "page_number": page_number,
                        "paragraph_index": para_idx,
                        "section_path": section_path,
                        "metadata": {
                            "p_tags": getattr(ln, "p_tags", None),
                            "d_tags": getattr(ln, "d_tags", None),
                        },
                    }
                )
            if bulk_rows:
                await db_handler.write_embeddable_units_bulk(db, bulk_rows)
        except Exception as eu_err:
            logger.warning("[%s] [%s] embeddable_unit write (non-fatal): %s", doc_id, para_id, eu_err)
        timings.add("bulk_eu", (time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        await db_handler.safe_commit(db)
        timings.add("commit", (time.perf_counter() - t0) * 1000.0)

        timings.paragraphs += 1
        timings.lines += len(line_objs)

        # Emit a running [PATHB_TIMING] line every 50 paragraphs so we
        # can read latency on the wire mid-doc without waiting for
        # finalise() on 1000-paragraph docs.
        if timings.paragraphs % 50 == 0:
            logger.info("[%s] [PATHB_TIMING] %s", doc_id, timings.summary())

        logger.info(
            "[%s] Path B: built 1 paragraph, %s lines for %s%s",
            doc_id, len(line_objs), para_id,
            f" ({n_with_tags} with tags)" if n_with_tags else "",
        )

        # --- Emit user-friendly result event (env-gated) ---
        # Per-paragraph status_message events were costing ~65 ms/p
        # (53% of total paragraph time as of 2026-05-01) — every call
        # is one INSERT into chunking_events, a 600k+ row hot table.
        # Path B has no consumer that reads per-paragraph status events
        # (the dashboard reads doc-level events from the coordinator
        # and pulls progress numbers from policy_paragraphs counts).
        # Gate behind the same EMIT_PARAGRAPH_EVENTS env var that
        # already gates paragraph_start/paragraph_complete in
        # context.emit. Set EMIT_PARAGRAPH_EVENTS=1 to re-enable for
        # debugging.
        import os as _os
        if _os.environ.get("EMIT_PARAGRAPH_EVENTS", "").strip() in ("1", "true", "yes"):
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

            t0 = time.perf_counter()
            await ctx.send_status(
                message=f"Path B: {n_lines} lines, {n_with_tags} tagged for {para_id}",
                user_message=user_msg,
            )
            timings.add("status", (time.perf_counter() - t0) * 1000.0)
        else:
            # Still account a 0ms entry so the timing summary keeps the
            # 'status' column populated for log-table parsing.
            timings.add("status", 0.0)

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

    # ── Emit per-stage timing summary for the doc ────────────────────
    # Single grep-friendly line per doc. Look for "[PATHB_TIMING]" in
    # Cloud Logging to compare stage costs across docs.
    if res.timings.paragraphs > 0:
        logger.info("[%s] [PATHB_TIMING] %s", doc_id, res.timings.summary())

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

    # ── Stamp lexicon revision on document_tags ───────────────────────
    try:
        lex_rev = None
        if res.lexicon_snapshot and hasattr(res.lexicon_snapshot, "meta"):
            lex_rev = res.lexicon_snapshot.meta.get("revision")
        if lex_rev is not None:
            from datetime import datetime as _dt
            from sqlalchemy import text as _text
            await db.execute(
                _text(
                    "UPDATE document_tags SET lexicon_revision = :rev, tagged_at = :ts "
                    "WHERE document_id = :doc_id"
                ),
                {"rev": int(lex_rev), "ts": _dt.utcnow(), "doc_id": doc_uuid},
            )
            await db_handler.safe_commit(db)
            logger.info("[%s] Path B: stamped lexicon_revision=%s on document_tags", doc_id, lex_rev)
    except Exception as rev_err:
        logger.warning("[%s] Path B lexicon revision stamp (non-fatal): %s", doc_id, rev_err, exc_info=True)
        await db_handler.safe_rollback(db)

    # ── Extract lexicon candidates ────────────────────────────────────
    if res.lexicon_snapshot is not None:
        await ctx.send_status(
            message="Extracting lexicon candidates...",
            user_message="Identifying new candidate terms for the policy lexicon...",
        )
        try:
            if res.phrase_map:
                pm = res.phrase_map
            else:
                pm, _refuted = res.get_phrase_to_tag_map(res.lexicon_snapshot)
            await res.extract_candidates_for_document(db, doc_uuid, run_id=None, phrase_map=pm)
            # Commit so the freshly-extracted candidates are durable + visible to
            # the lexicon-maintenance cleanup service (separate connection).
            await db.commit()
            logger.info("[%s] Path B: candidate extraction complete", doc_id)
            await ctx.send_status(
                message="Candidate extraction complete.",
                user_message="Lexicon candidate extraction complete.",
            )
            # Fire-and-forget inline cleanup: deterministic rules + catalog +
            # bounded LLM triage on this document's candidates. Best-effort —
            # never blocks or fails ingestion.
            # Skipped on the inline instant pipeline (ctx.skip_lexicon_cleanup=True)
            # where the ~6s HTTP round-trip is unacceptable. Candidate mining
            # still runs at corpus-promotion / batch re-process time.
            if not getattr(ctx, "skip_lexicon_cleanup", False):
                await _trigger_lexicon_cleanup(ctx, str(doc_uuid), doc_id)
        except Exception as cand_err:
            logger.warning("[%s] Path B candidate extraction (non-fatal): %s", doc_id, cand_err, exc_info=True)
