"""
Integration test: Run a small document through Path A and Path B using the
refactored worker modules, then inspect every table and report results.

Usage:  cd mobius-rag && .venv/bin/python3 tests/integration_chunking_test.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import textwrap
import traceback
from datetime import datetime, timezone
from uuid import uuid4, UUID

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("integration")

# LLM config: use Vertex AI (production) by default.
# Set LLM_PROVIDER=ollama to use local Ollama instead.
import os

TEST_MARKDOWN = textwrap.dedent("""\
# Member Eligibility

## Section 1: General

Members must be enrolled in the plan to receive benefits. Eligibility is determined at the time of service. Prior authorization may be required for certain procedures.

## Section 2: Verification

The member must provide proof of eligibility upon request. Verification can be completed online or by phone. Failure to verify may result in denial of claims.

## Section 3: Effective Date

Coverage begins on the effective date shown on the member's card. Termination of eligibility ends benefits as of the last day of the month. Appeals must be filed within 60 days.
""")


def _now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ── Report helpers ─────────────────────────────────────────────────────────

class Report:
    def __init__(self):
        self.sections: list[str] = []
        self._current: list[str] = []

    def header(self, title: str):
        self._flush()
        self.sections.append(f"\n{'='*72}\n  {title}\n{'='*72}")
        self._current = []

    def sub(self, title: str):
        self._flush()
        self.sections.append(f"\n  --- {title} ---")
        self._current = []

    def line(self, text: str = ""):
        self._current.append(text)

    def table(self, headers: list[str], rows: list[list]):
        widths = [max(len(str(h)), *(len(str(r[i])) for r in rows))
                  for i, h in enumerate(headers)] if rows else [len(h) for h in headers]
        sep = "  ".join("-" * w for w in widths)
        hdr = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
        self._current.append(hdr)
        self._current.append(sep)
        for row in rows:
            self._current.append("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
        if not rows:
            self._current.append("  (no rows)")

    def _flush(self):
        if self._current:
            self.sections.append("\n".join(self._current))
            self._current = []

    def dump(self):
        self._flush()
        return "\n".join(self.sections)


# ── DB setup / teardown ───────────────────────────────────────────────────

async def create_test_document(db, suffix=""):
    """Insert a Document + 1 DocumentPage with the test markdown."""
    from app.models import Document, DocumentPage

    doc_id = uuid4()
    doc = Document(
        id=doc_id,
        filename=f"integration_test_document{suffix}.md",
        display_name=f"Integration Test Document {suffix}",
        file_hash=f"integration_test_{uuid4().hex[:12]}",
        file_path=f"gs://test-bucket/integration_test_document{suffix}.md",
        status="uploaded",
    )
    db.add(doc)
    await db.flush()

    page = DocumentPage(
        document_id=doc_id,
        page_number=1,
        text=TEST_MARKDOWN,
        text_markdown=TEST_MARKDOWN,
        extraction_status="success",
        text_length=len(TEST_MARKDOWN),
    )
    db.add(page)
    await db.flush()
    await db.commit()
    logger.info("Created test document %s with 1 page (%s chars)", doc_id, len(TEST_MARKDOWN))
    return doc_id


async def create_chunking_job(db, doc_id, *, generator_id="A"):
    from app.models import ChunkingJob

    job = ChunkingJob(
        document_id=doc_id,
        status="pending",
        threshold="0.6",
        generator_id=generator_id,
        extraction_enabled="true" if generator_id == "A" else "false",
        critique_enabled="true" if generator_id == "A" else "false",
        max_retries=1 if generator_id == "A" else 0,
    )
    db.add(job)
    await db.flush()
    await db.commit()
    logger.info("Created chunking job %s (generator=%s) for doc %s", job.id, generator_id, doc_id)
    return job


# ── Inspection queries ────────────────────────────────────────────────────

async def inspect(db, doc_id, report: Report, label: str):
    from sqlalchemy import select, text
    from app.models import (
        ChunkingJob, ChunkingResult, ChunkingEvent, HierarchicalChunk,
        ExtractedFact, EmbeddingJob,
    )

    # Refresh session to see latest
    await db.commit()

    # ── ChunkingJob ──
    report.sub(f"{label}: ChunkingJob")
    jobs = (await db.execute(select(ChunkingJob).where(ChunkingJob.document_id == doc_id))).scalars().all()
    rows = []
    for j in jobs:
        rows.append([str(j.id)[:8], j.status, j.generator_id or "-", j.worker_id or "-",
                      str(j.started_at)[:19] if j.started_at else "-",
                      str(j.completed_at)[:19] if j.completed_at else "-",
                      (j.error_message or "-")[:50],
                      "YES" if j.chunking_config_snapshot else "NO"])
    report.table(["id", "status", "gen", "worker", "started", "completed", "error", "config_snap"], rows)

    for j in jobs:
        if j.chunking_config_snapshot:
            report.line(f"\n  Config snapshot (job {str(j.id)[:8]}):")
            for k, v in j.chunking_config_snapshot.items():
                report.line(f"    {k}: {v}")

    # ── ChunkingResult ──
    report.sub(f"{label}: ChunkingResult")
    cr = (await db.execute(select(ChunkingResult).where(ChunkingResult.document_id == doc_id))).scalars().all()
    for c in cr:
        meta = c.metadata_ or {}
        report.line(f"  status={meta.get('status')}  total_paragraphs={meta.get('total_paragraphs')}  completed={meta.get('completed_count')}  pages={meta.get('total_pages')}")
        report.line(f"  errors={meta.get('error_counts')}")
        results = c.results or {}
        for pid, pdata in results.items():
            st = pdata.get("status", "?")
            n_facts = len(pdata.get("facts", []))
            summary = (pdata.get("summary") or "")[:60]
            report.line(f"    paragraph {pid}: status={st}  facts={n_facts}  summary={summary}")
    if not cr:
        report.line("  (no ChunkingResult rows)")

    # ── ChunkingEvents ──
    report.sub(f"{label}: ChunkingEvents")
    events = (await db.execute(
        select(ChunkingEvent)
        .where(ChunkingEvent.document_id == doc_id)
        .order_by(ChunkingEvent.created_at)
    )).scalars().all()
    rows = []
    for e in events:
        data = e.event_data or {}
        msg = (data.get("message") or "")[:50]
        umsg = (data.get("user_message") or "")[:50]
        rows.append([str(e.id)[:8], e.event_type, msg, umsg])
    report.table(["id", "event_type", "message (ops)", "user_message"], rows)
    report.line(f"  Total events: {len(events)}")

    # ── HierarchicalChunks ──
    report.sub(f"{label}: HierarchicalChunks")
    chunks = (await db.execute(
        select(HierarchicalChunk)
        .where(HierarchicalChunk.document_id == doc_id)
        .order_by(HierarchicalChunk.page_number, HierarchicalChunk.paragraph_index)
    )).scalars().all()
    rows = []
    for ch in chunks:
        rows.append([str(ch.id)[:8], ch.page_number, ch.paragraph_index,
                      ch.extraction_status, ch.critique_status,
                      (ch.text or "")[:50], (ch.summary or "-")[:40]])
    report.table(["id", "page", "para", "extraction", "critique", "text_preview", "summary"], rows)
    report.line(f"  Total chunks: {len(chunks)}")

    # ── ExtractedFacts ──
    report.sub(f"{label}: ExtractedFacts")
    facts = (await db.execute(
        select(ExtractedFact).where(ExtractedFact.document_id == str(doc_id))
    )).scalars().all()
    rows = []
    for f in facts:
        rows.append([str(f.id)[:8], (f.fact_text or "")[:50], f.fact_type or "-",
                      f.page_number, f.start_offset, f.end_offset,
                      f.is_eligibility_related or "-", f.confidence or "-"])
    report.table(["id", "fact_text", "type", "page", "start", "end", "elig", "conf"], rows)
    report.line(f"  Total facts: {len(facts)}")

    # Detailed fact content
    if facts:
        report.line("\n  Extracted fact details:")
        for i, f in enumerate(facts):
            report.line(f"    [{i+1}] {(f.fact_text or '')[:]}")
            report.line(f"        type={f.fact_type} confidence={f.confidence}")
            report.line(f"        page={f.page_number} offset={f.start_offset}..{f.end_offset}")
            if f.is_eligibility_related:
                report.line(f"        eligibility_related={f.is_eligibility_related}")

    # ── PolicyParagraphs + PolicyLines ──
    # Using raw SQL to avoid the offset_match_quality varchar/float type mismatch
    report.sub(f"{label}: PolicyParagraphs + PolicyLines")
    try:
        para_rows = (await db.execute(text(
            "SELECT id, page_number, order_index, paragraph_type, text, p_tags, d_tags, j_tags "
            "FROM policy_paragraphs WHERE document_id = :did "
            "ORDER BY page_number, order_index"
        ), {"did": str(doc_id)})).fetchall()
        for pp in para_rows:
            pp_id, pg, oidx, ptype, ptxt, ptags, dtags, jtags = pp
            tags_str = f"p_tags={json.dumps(ptags)}" if ptags else "p_tags=null"
            tags_str += f" d_tags={json.dumps(dtags)}" if dtags else " d_tags=null"
            tags_str += f" j_tags={json.dumps(jtags)}" if jtags else " j_tags=null"
            report.line(f"  Paragraph idx={oidx} (page {pg}): type={ptype}  {tags_str}")
            report.line(f"    text: {(ptxt or '')[:80]}")

            line_rows = (await db.execute(text(
                "SELECT order_index, text, p_tags, d_tags, j_tags "
                "FROM policy_lines WHERE paragraph_id = :pid "
                "ORDER BY order_index"
            ), {"pid": str(pp_id)})).fetchall()
            for ln in line_rows:
                lidx, ltxt, lptags, ldtags, ljtags = ln
                lt = f"p_tags={json.dumps(lptags)}" if lptags else "p_tags=null"
                lt += f" d_tags={json.dumps(ldtags)}" if ldtags else " d_tags=null"
                report.line(f"      line {lidx}: {(ltxt or '')[:60]}  ({lt})")
            report.line(f"    ({len(line_rows)} lines)")
        if not para_rows:
            report.line("  (no PolicyParagraphs)")
    except Exception as e:
        report.line(f"  Error reading PolicyParagraphs/Lines: {e}")

    # ── EmbeddableUnits ──
    report.sub(f"{label}: EmbeddableUnits")
    try:
        eu_rows = (await db.execute(text(
            "SELECT id, generator_id, source_type, text, status, metadata "
            "FROM embeddable_units WHERE document_id = :did ORDER BY created_at"
        ), {"did": str(doc_id)})).fetchall()
        rows = []
        for r in eu_rows:
            rows.append([str(r[0])[:8], r[1] or "-", r[2], (r[3] or "")[:50], r[4]])
        report.table(["id", "gen", "source_type", "text_preview", "status"], rows)
        report.line(f"  Total embeddable_units: {len(eu_rows)}")

        if eu_rows:
            report.line("\n  Embeddable unit details:")
            for i, r in enumerate(eu_rows):
                meta = r[5] if r[5] else {}
                report.line(f"    [{i+1}] gen={r[1]} type={r[2]} status={r[4]}")
                report.line(f"         text: {(r[3] or '')[:100]}")
                if meta:
                    report.line(f"         metadata: {json.dumps(meta)[:120]}")
    except Exception as e:
        report.line(f"  Error reading embeddable_units: {e}")

    # ── DocumentTags ──
    report.sub(f"{label}: DocumentTags")
    try:
        dt_rows = (await db.execute(text(
            "SELECT p_tags, d_tags, j_tags FROM document_tags WHERE document_id = :did"
        ), {"did": str(doc_id)})).fetchall()
        for r in dt_rows:
            report.line(f"  p_tags: {json.dumps(r[0]) if r[0] else 'null'}")
            report.line(f"  d_tags: {json.dumps(r[1]) if r[1] else 'null'}")
            report.line(f"  j_tags: {json.dumps(r[2]) if r[2] else 'null'}")
        if not dt_rows:
            report.line("  (no DocumentTags row)")
    except Exception as e:
        report.line(f"  Error reading document_tags: {e}")

    # ── EmbeddingJobs ──
    report.sub(f"{label}: EmbeddingJobs")
    ej = (await db.execute(select(EmbeddingJob).where(EmbeddingJob.document_id == doc_id))).scalars().all()
    rows = []
    for e in ej:
        rows.append([str(e.id)[:8], e.status, e.generator_id or "-"])
    report.table(["id", "status", "gen"], rows)
    if ej:
        report.line(f"  Total embedding_jobs: {len(ej)}")
    else:
        report.line("  (no EmbeddingJob created)")

    # ── ProcessingErrors ──
    report.sub(f"{label}: ProcessingErrors")
    try:
        pe_rows = (await db.execute(text(
            "SELECT id, paragraph_id, error_type, error_message "
            "FROM processing_errors WHERE document_id = :did ORDER BY created_at"
        ), {"did": str(doc_id)})).fetchall()
        rows = []
        for r in pe_rows:
            rows.append([str(r[0])[:8], r[1] or "-", r[2] or "-", (r[3] or "")[:50]])
        report.table(["id", "paragraph", "category", "message"], rows)
    except Exception as e:
        report.line(f"  Error reading processing_errors: {e}")


# ── Cleanup ───────────────────────────────────────────────────────────────

async def cleanup(db, doc_id):
    from sqlalchemy import text
    tables = [
        ("embeddable_units", "document_id"),
        ("document_tags", "document_id"),
        ("policy_lexicon_candidates", "document_id"),
        ("policy_spans", "line_id IN (SELECT id FROM policy_lines WHERE document_id = :did)", True),
        ("policy_lines", "document_id"),
        ("policy_paragraphs", "document_id"),
        ("extracted_facts", "document_id"),
        ("hierarchical_chunks", "document_id"),
        ("chunk_embeddings", "document_id"),
        ("chunking_events", "document_id"),
        ("processing_errors", "document_id"),
        ("embedding_jobs", "document_id"),
        ("chunking_results", "document_id"),
        ("chunking_jobs", "document_id"),
        ("document_pages", "document_id"),
        ("documents", "id"),
    ]
    for entry in tables:
        tbl = entry[0]
        if len(entry) == 3:
            # Custom WHERE clause
            where = entry[1]
            try:
                await db.execute(text(f"DELETE FROM {tbl} WHERE {where}"), {"did": str(doc_id)})
            except Exception:
                pass
        else:
            col = entry[1]
            try:
                await db.execute(text(f"DELETE FROM {tbl} WHERE {col} = :did"), {"did": str(doc_id)})
            except Exception:
                pass
    await db.commit()
    logger.info("Cleaned up test document %s", doc_id)


# ── Main ──────────────────────────────────────────────────────────────────

async def run_path(doc_id, job_id, generator_id, label, report):
    """Process a job using the refactored worker pipeline and inspect results."""
    from app.database import AsyncSessionLocal
    from app.worker.main import process_job
    from app.worker.config import load_worker_config
    from sqlalchemy import select
    from app.models import ChunkingJob

    cfg = load_worker_config()
    logger.info("Processing %s job %s ...", label, job_id)

    # --- Phase 1: Process the job in its own session ---
    t0 = _now()
    proc_error = None
    async with AsyncSessionLocal() as db:
        job = (await db.execute(select(ChunkingJob).where(ChunkingJob.id == job_id))).scalar_one()
        try:
            await process_job(job, db, worker_cfg=cfg)
        except Exception as e:
            proc_error = e
    elapsed = (_now() - t0).total_seconds()
    logger.info("%s completed in %.2fs", label, elapsed)
    report.line(f"\n  Runtime: {elapsed:.2f}s")
    if proc_error:
        report.line(f"  EXCEPTION during processing: {type(proc_error).__name__}: {proc_error}")
        report.line(traceback.format_exc())

    # --- Phase 2: Inspect results in a fresh session ---
    async with AsyncSessionLocal() as db:
        job_r = (await db.execute(select(ChunkingJob).where(ChunkingJob.id == job_id))).scalar_one()
        report.line(f"  Job final status: {job_r.status}")
        if job_r.error_message:
            report.line(f"  Job error: {job_r.error_message}")

        await inspect(db, doc_id, report, label)


async def main():
    from app.database import AsyncSessionLocal

    report = Report()
    report.header("INTEGRATION TEST: Chunking Worker Refactor")
    report.line(f"Test document: 3 paragraphs (General, Verification, Effective Date)")
    report.line(f"Started: {_now().isoformat()}")
    report.line(f"Test markdown length: {len(TEST_MARKDOWN)} chars")

    doc_id_a = None
    doc_id_b = None

    # ══════════════════════════════════════════════════════════════════════
    # PATH A
    # ══════════════════════════════════════════════════════════════════════
    try:
        report.header("PATH A: LLM Extraction + Critique")
        async with AsyncSessionLocal() as db:
            doc_id_a = await create_test_document(db, suffix="_path_a")
            job_a = await create_chunking_job(db, doc_id_a, generator_id="A")
            job_a_id = job_a.id
        await run_path(doc_id_a, job_a_id, "A", "Path A", report)
    except Exception as e:
        report.header("PATH A: ERROR")
        report.line(f"  {type(e).__name__}: {e}")
        report.line(traceback.format_exc())

    # ══════════════════════════════════════════════════════════════════════
    # PATH B
    # ══════════════════════════════════════════════════════════════════════
    try:
        report.header("PATH B: Deterministic Policy Chunking")
        async with AsyncSessionLocal() as db:
            doc_id_b = await create_test_document(db, suffix="_path_b")
            job_b = await create_chunking_job(db, doc_id_b, generator_id="B")
            job_b_id = job_b.id
        await run_path(doc_id_b, job_b_id, "B", "Path B", report)
    except Exception as e:
        report.header("PATH B: ERROR")
        report.line(f"  {type(e).__name__}: {e}")
        report.line(traceback.format_exc())

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    report.header("SUMMARY & EXPECTATIONS vs ACTUAL")
    report.line("""
Expectations:
  Path A (LLM Extraction + Critique):
    - 3 HierarchicalChunks created (one per paragraph)
    - ExtractedFacts for each paragraph (5-10 facts expected)
    - Each fact has: fact_text, fact_type, page_number, offsets, confidence
    - ChunkingEvents emitted: job_start, paragraph_start/complete x3, chunking_complete
    - Config snapshot stored on ChunkingJob
    - EmbeddableUnits: 1 per chunk + 1 per fact
    - EmbeddingJob: queued on success
    - ChunkingResult: metadata with status, paragraph counts

  Path B (Deterministic Policy Chunking):
    - 3 PolicyParagraphs + policy_lines per paragraph
    - Tags applied (if lexicon entries exist for content)
    - HierarchicalChunks created (skipped extraction/critique status)
    - Tag propagation: line -> paragraph -> document
    - ChunkingEvents emitted: job_start, paragraph_start/complete x3, chunking_complete
    - EmbeddableUnits: 1 per policy_line
    - DocumentTags row aggregated
    - EmbeddingJob: queued on success

Behavior Notes:
    """)

    # ── Cleanup ──
    report.header("CLEANUP")
    async with AsyncSessionLocal() as db:
        if doc_id_a:
            await cleanup(db, doc_id_a)
            report.line(f"  Cleaned up Path A doc: {doc_id_a}")
        if doc_id_b:
            await cleanup(db, doc_id_b)
            report.line(f"  Cleaned up Path B doc: {doc_id_b}")

    # Print and save
    output = report.dump()
    print(output)

    with open("tests/integration_test_report.txt", "w") as f:
        f.write(output)
    print(f"\n--- Report saved to tests/integration_test_report.txt ---")


if __name__ == "__main__":
    asyncio.run(main())
