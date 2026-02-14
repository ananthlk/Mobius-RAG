"""
Chunking worker entry-point.

Thin shell: main() -> worker_loop() -> process_job() -> coordinator.run_chunking_loop().
All business logic lives in ``app.worker.{path_a, path_b, coordinator}``.
DB access is via ``app.worker.db``.  Configuration via ``app.worker.config``.
"""
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal
from app.models import Document, DocumentPage, ChunkingJob

from app.worker.config import load_worker_config, WorkerConfig
from app.worker.coordinator import run_chunking_loop
from app.worker.db import (
    enqueue_embedding_job,
    safe_commit,
    set_config_snapshot,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [WORKER] - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _utc_now_naive() -> datetime:
    """Naive UTC datetime for DB (TIMESTAMP WITHOUT TIME ZONE)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


WORKER_ID = f"worker-{os.getpid()}-{_utc_now_naive().isoformat()}"


# ---------------------------------------------------------------------------
# process_job
# ---------------------------------------------------------------------------

async def process_job(job: ChunkingJob, db: AsyncSession, *, worker_cfg: WorkerConfig | None = None):
    """Process a single chunking job."""
    if worker_cfg is None:
        worker_cfg = load_worker_config()
    job_start_time = _utc_now_naive()
    try:
        logger.info("[JOB %s] Starting for document %s (threshold: %s)", job.id, job.document_id, job.threshold)

        # --- Mark processing + config snapshot ---
        job.status = "processing"
        job.worker_id = WORKER_ID
        job.started_at = job_start_time
        await set_config_snapshot(db, job, worker_config=worker_cfg)
        await db.commit()
        logger.info("[JOB %s] Status=processing, worker=%s", job.id, WORKER_ID)

        # --- Fetch document & pages ---
        doc_uuid = UUID(str(job.document_id))
        doc_result = await db.execute(select(Document).where(Document.id == doc_uuid))
        document = doc_result.scalar_one_or_none()
        if not document:
            logger.error("[JOB %s] Document not found: %s", job.id, job.document_id)
            job.status = "failed"
            job.error_message = f"Document {job.document_id} not found"
            job.completed_at = _utc_now_naive()
            await db.commit()
            return

        pages_result = await db.execute(
            select(DocumentPage)
            .where(DocumentPage.document_id == doc_uuid)
            .order_by(DocumentPage.page_number)
        )
        pages = pages_result.scalars().all()
        if not pages:
            logger.error("[JOB %s] No pages for document %s", job.id, job.document_id)
            job.status = "failed"
            job.error_message = f"No pages found for document {job.document_id}"
            job.completed_at = _utc_now_naive()
            await db.commit()
            return

        logger.info("[JOB %s] Document: %s (%s pages)", job.id, document.filename, len(pages))

        # --- Resolve run parameters ---
        threshold = _resolve_float(job.threshold, worker_cfg.default_threshold)
        critique_enabled = _resolve_bool(job.critique_enabled, worker_cfg.default_critique_enabled)
        max_retries = _resolve_int(job.max_retries, worker_cfg.default_max_retries)
        extraction_enabled = _resolve_bool(job.extraction_enabled, worker_cfg.default_extraction_enabled)

        gen = (getattr(job, "generator_id", None) or "A").strip().upper() or "A"
        if gen == "B":
            extraction_enabled = False
            logger.info("[JOB %s] Path B: extraction_enabled=False", job.id)

        # --- Resolve prompts & LLM ---
        extraction_prompt_body, retry_extraction_prompt_body, critique_prompt_body = _resolve_prompts(job)
        llm = await _resolve_llm(job, db)
        if llm is None and extraction_enabled:
            # Path A requires an LLM; Path B does not.
            job.status = "failed"
            job.error_message = "LLM provider init failed"
            job.completed_at = _utc_now_naive()
            await db.commit()
            return

        # --- Lexicon (Path B) ---
        lexicon_snapshot = None
        if gen == "B":
            lexicon_snapshot = await _load_lexicon(job, db)

        # --- Delegate to coordinator ---
        logger.info("[JOB %s] Delegating to coordinator...", job.id)
        success = await run_chunking_loop(
            str(job.document_id),
            doc_uuid,
            str(job.id),
            pages,
            db,
            threshold=threshold,
            critique_enabled=critique_enabled,
            max_retries=max_retries,
            extraction_enabled=extraction_enabled,
            extraction_prompt_body=extraction_prompt_body,
            retry_extraction_prompt_body=retry_extraction_prompt_body,
            critique_prompt_body=critique_prompt_body,
            llm=llm,
            lexicon_snapshot=lexicon_snapshot,
            worker_cfg=worker_cfg,
        )

        await db.commit()

        # --- Finalise job ---
        job_duration = (_utc_now_naive() - job_start_time).total_seconds()
        if success:
            job.status = "completed"
            job.completed_at = _utc_now_naive()
            logger.info("[JOB %s] Completed in %.2fs", job.id, job_duration)
            try:
                enqueued = await enqueue_embedding_job(db, job.document_id, getattr(job, "generator_id", None))
                if enqueued:
                    logger.info("[JOB %s] Enqueued embedding job", job.id)
            except Exception as enq_err:
                logger.warning("[JOB %s] Failed to enqueue embedding: %s", job.id, enq_err, exc_info=True)
        else:
            job.status = "failed"
            job.error_message = "Chunking loop returned False"
            logger.warning("[JOB %s] Failed after %.2fs", job.id, job_duration)

        await db.commit()
        logger.info("[JOB %s] Final status: %s", job.id, job.status)

    except Exception as e:
        job_duration = (_utc_now_naive() - job_start_time).total_seconds()
        logger.error("[JOB %s] Error after %.2fs: %s", getattr(job, "id", None), job_duration, e, exc_info=True)
        try:
            await db.rollback()
        except Exception as rb:
            logger.error("[JOB %s] Rollback failed: %s", getattr(job, "id", None), rb, exc_info=True)
        try:
            job.status = "failed"
            job.error_message = str(e)[:2000]
            job.completed_at = _utc_now_naive()
            await db.commit()
        except Exception as commit_err:
            logger.error("[JOB %s] Failed to persist failure status: %s", getattr(job, "id", None), commit_err, exc_info=True)


# ---------------------------------------------------------------------------
# worker_loop
# ---------------------------------------------------------------------------

async def worker_loop():
    """Main worker loop — polls for pending jobs and processes them."""
    cfg = load_worker_config()
    logger.info("Worker %s starting...", WORKER_ID)

    # Startup migrations
    _run_startup_migrations_sync = [
        ("category_scores_to_columns", "app.migrations.category_scores_to_columns"),
        ("chunking_config_snapshot", "app.migrations.add_chunking_config_snapshot"),
        ("embeddable_units", "app.migrations.add_embeddable_units"),
        ("document_tags", "app.migrations.add_document_tags"),
        ("fix_offset_type", "app.migrations.fix_policy_lines_offset_type"),
    ]
    for label, mod_path in _run_startup_migrations_sync:
        try:
            import importlib
            m = importlib.import_module(mod_path)
            await m.migrate()
        except Exception as migrate_err:
            logger.warning("Startup migration (%s) skipped/failed: %s", label, migrate_err)

    poll_count = 0
    while True:
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(ChunkingJob)
                    .where(ChunkingJob.status == "pending")
                    .order_by(ChunkingJob.created_at)
                    .limit(1)
                )
                job = result.scalar_one_or_none()

                if job:
                    logger.info("Found pending job %s for document %s", job.id, job.document_id)
                    await process_job(job, db, worker_cfg=cfg)
                    poll_count = 0
                else:
                    poll_count += 1
                    if poll_count % 10 == 0:
                        logger.debug("No pending jobs (poll #%s)", poll_count)
                    await asyncio.sleep(cfg.poll_interval_seconds)
        except Exception as e:
            logger.error("Error in worker loop: %s", e, exc_info=True)
            await asyncio.sleep(cfg.error_sleep_seconds)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    """Entry point for worker process."""
    try:
        asyncio.run(worker_loop())
    except KeyboardInterrupt:
        logger.info("Worker shutting down...")
    except Exception as e:
        logger.error("Fatal error in worker: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Helpers (parameter resolution — private to this module)
# ---------------------------------------------------------------------------

def _resolve_float(raw, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _resolve_int(raw, default: int) -> int:
    if raw is None:
        return default
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return default


def _resolve_bool(raw, default: bool) -> bool:
    if raw is None:
        return default
    try:
        return str(raw).strip().lower() == "true"
    except Exception:
        return default


def _resolve_prompts(job: ChunkingJob):
    """Resolve prompt bodies from job.prompt_versions."""
    extraction_prompt_body = None
    retry_extraction_prompt_body = None
    critique_prompt_body = None
    prompt_versions = getattr(job, "prompt_versions", None) or {}
    try:
        if isinstance(prompt_versions, dict):
            from app.services.prompt_registry import get_prompt
            extraction_prompt_body = get_prompt("extraction", prompt_versions.get("extraction") or "v1")
            retry_extraction_prompt_body = get_prompt("extraction_retry", prompt_versions.get("extraction_retry") or "v1")
            critique_prompt_body = get_prompt("critique", prompt_versions.get("critique") or "v1")
    except Exception as e:
        logger.warning("[JOB %s] Prompt resolution failed, using defaults: %s", job.id, e, exc_info=True)
    return extraction_prompt_body, retry_extraction_prompt_body, critique_prompt_body


async def _resolve_llm(job: ChunkingJob, db: AsyncSession):
    """Resolve the LLM provider from job config, falling back to default."""
    llm = None
    llm_config_version = getattr(job, "llm_config_version", None)
    try:
        if llm_config_version:
            from app.services.llm_config import get_llm_config_resolved, get_llm_provider_from_config
            cfg = await get_llm_config_resolved(llm_config_version, db)
            if cfg:
                llm = get_llm_provider_from_config(cfg)
                logger.info("[JOB %s] Using LLM config: %s", job.id, llm_config_version)
    except Exception as e:
        logger.warning("[JOB %s] LLM config %r failed, falling back: %s", job.id, llm_config_version, e, exc_info=True)
    if not llm:
        try:
            from app.services.llm_provider import get_llm_provider
            llm = get_llm_provider()
        except Exception as e:
            logger.error("[JOB %s] get_llm_provider failed: %s", job.id, e, exc_info=True)
            return None
    return llm


async def _load_lexicon(job: ChunkingJob, db: AsyncSession):
    """Load lexicon snapshot for Path B."""
    try:
        from app.services.policy_lexicon_repo import load_lexicon_snapshot_db
        from app.services.policy_path_b import get_phrase_to_tag_map
        snapshot = await load_lexicon_snapshot_db(db)
        phrase_map, _refuted_map = get_phrase_to_tag_map(snapshot)
        n_phrases = len(phrase_map)
        if n_phrases == 0:
            logger.warning("[JOB %s] Path B: lexicon has 0 phrases", job.id)
        else:
            logger.info("[JOB %s] Path B: lexicon loaded (%s phrases)", job.id, n_phrases)
        return snapshot
    except Exception as lex_err:
        logger.warning("[JOB %s] Path B: lexicon load failed: %s", job.id, lex_err)
        return None
