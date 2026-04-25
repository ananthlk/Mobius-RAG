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
    recover_stale_jobs,
    safe_commit,
    set_config_snapshot,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# Logging is configured lazily from ``worker_loop`` / ``main`` so that
# merely importing ``app.worker.shutdown`` (which pulls this module via
# the package __init__ re-exports) does NOT reconfigure the root logger
# of a different entrypoint — previously this clobbered the embedding
# worker's JSON service name.
logger = logging.getLogger(__name__)


def _utc_now_naive() -> datetime:
    """Naive UTC datetime for DB (TIMESTAMP WITHOUT TIME ZONE)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


WORKER_ID = f"worker-{os.getpid()}-{_utc_now_naive().isoformat()}"


# ---------------------------------------------------------------------------
# Atomic finalize + zombie recovery
# ---------------------------------------------------------------------------


async def _finalize_job_atomic(
    job_id,
    document_id,
    *,
    success: bool,
    generator_id: str | None,
    skip_embed: bool,
    error_message: str | None,
) -> None:
    """Mark a chunking_job row completed/failed in a fresh session, with retries.

    The coordinator's primary session may be poisoned (InFailedSQLTransaction
    after a mid-coordinator commit failure) by the time we reach finalize.
    A fresh session sidesteps that. If even the fresh session fails, retry
    with exponential backoff up to 5 attempts over ~30s.

    Embedding enqueue is best-effort; a failure logs WARNING but doesn't
    block job-status finalization.
    """
    from sqlalchemy import update
    from app.models import ChunkingJob

    final_status = "completed" if success else "failed"
    backoff = 1.0
    last_err: Exception | None = None

    for attempt in range(1, 6):
        try:
            async with AsyncSessionLocal() as fresh_db:
                values: dict = {
                    "status": final_status,
                    "completed_at": _utc_now_naive(),
                }
                if error_message:
                    values["error_message"] = error_message[:2000]
                result = await fresh_db.execute(
                    update(ChunkingJob).where(ChunkingJob.id == job_id).values(**values)
                )
                # Embedding enqueue (only for successful chunking, not retag)
                if success and not skip_embed:
                    try:
                        enqueued = await enqueue_embedding_job(fresh_db, document_id, generator_id)
                        if enqueued:
                            logger.info("[JOB %s] Enqueued embedding job", job_id)
                    except Exception as enq_err:
                        logger.warning(
                            "[JOB %s] Failed to enqueue embedding (non-fatal): %s",
                            job_id, enq_err,
                        )
                await fresh_db.commit()
                if (result.rowcount or 0) > 0 or attempt > 1:
                    return
                # rowcount=0 on first try means the row didn't exist or
                # was already in target state — also a success outcome.
                logger.info("[JOB %s] finalize touched 0 rows (already finalized?)", job_id)
                return
        except Exception as exc:
            last_err = exc
            logger.warning(
                "[JOB %s] finalize attempt %d/5 failed: %s",
                job_id, attempt, exc,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 15.0)

    # All retries exhausted. Don't crash the worker — log loudly so
    # the startup recovery sweep picks it up next boot.
    logger.error(
        "[JOB %s] finalize FAILED after 5 attempts; row left at 'processing'. "
        "Startup recovery sweep will catch it next worker boot. Last error: %s",
        job_id, last_err,
    )


async def recover_finalized_zombies() -> int:
    """Find chunking_jobs stuck at 'processing' even though their work
    was actually written, and finalize them.

    A "zombie" here is a row whose:
      * status='processing' and started_at is older than 30 minutes
      * AND the document already has policy_lines OR hierarchical_chunks
        written (i.e. the coordinator legitimately completed)
      * AND no chunking_events have arrived in the last 10 minutes
        (so it's not actively being worked on)

    We mark such rows 'completed' and (for non-retag jobs) enqueue
    their embedding. Returns the number recovered.

    This is the safety net for the atomic-finalize retry loop above —
    when the retries truly run out (or the worker crashed before
    finalize started), the next worker boot picks up the cleanup.
    """
    from sqlalchemy import text

    try:
        async with AsyncSessionLocal() as db:
            # Grab candidate zombies + the work-evidence in one query so
            # we don't double-touch the row.
            r = await db.execute(text("""
                SELECT j.id::text AS job_id, j.document_id::text AS doc_id,
                       j.skip_embedding, j.generator_id,
                       (SELECT count(*) FROM policy_lines pl WHERE pl.document_id = j.document_id) AS pl_count,
                       (SELECT count(*) FROM hierarchical_chunks hc WHERE hc.document_id = j.document_id) AS hc_count
                FROM chunking_jobs j
                WHERE j.status = 'processing'
                  AND j.started_at < now() - interval '30 minutes'
                  AND NOT EXISTS (
                      SELECT 1 FROM chunking_events e
                      WHERE e.document_id = j.document_id
                        AND e.created_at > now() - interval '10 minutes'
                  )
            """))
            candidates = list(r.mappings())
            recovered = 0
            for row in candidates:
                if (row["pl_count"] or 0) == 0 and (row["hc_count"] or 0) == 0:
                    # No evidence the coordinator did anything; let the
                    # heartbeat-aware stale-recovery reset to pending
                    # instead. Don't mark as completed.
                    continue
                logger.warning(
                    "[zombie-finalize] Recovering job=%s doc=%s "
                    "(policy_lines=%s hierarchical_chunks=%s)",
                    row["job_id"], row["doc_id"], row["pl_count"], row["hc_count"],
                )
                await _finalize_job_atomic(
                    row["job_id"], row["doc_id"],
                    success=True,
                    generator_id=row["generator_id"],
                    skip_embed=(row["skip_embedding"] == "true"),
                    error_message=(
                        "Auto-recovered: coordinator wrote work to DB but "
                        "job-finalize commit never succeeded; recovered on "
                        "worker startup sweep"
                    ),
                )
                recovered += 1
            return recovered
    except Exception as exc:
        logger.warning("[zombie-finalize] sweep failed (non-fatal): %s", exc)
        return 0


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
        # Atomic + retried, in a FRESH session.
        #
        # Why fresh: ``db`` may be in InFailedSQLTransaction state from a
        # late mid-coordinator commit failure (DB failover, connection
        # pool blip). If we reuse it, the .commit() below silently leaves
        # the row at status='processing' and we get the zombie pattern
        # observed 4× across 2026-04-23/24 (FL-Care, 74973950, d9721756).
        #
        # Why retry: the same DB blip class is transient; one retry with
        # backoff usually lands. We try up to 5 times over ~30s.
        job_duration = (_utc_now_naive() - job_start_time).total_seconds()
        skip_embed = getattr(job, "skip_embedding", None) == "true"
        if success:
            logger.info("[JOB %s] Completed in %.2fs", job.id, job_duration)
        else:
            logger.warning("[JOB %s] Failed after %.2fs", job.id, job_duration)
        await _finalize_job_atomic(
            job.id, job.document_id,
            success=success,
            generator_id=getattr(job, "generator_id", None),
            skip_embed=skip_embed,
            error_message=None if success else "Chunking loop returned False",
        )
        logger.info("[JOB %s] Final status: %s", job.id, "completed" if success else "failed")

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
    """Main worker loop — polls for pending jobs and processes them.

    Graceful shutdown: installs SIGTERM/SIGINT handlers on start;
    the loop polls ``is_shutting_down()`` between iterations so
    in-flight jobs complete naturally before exit. DB rows locked
    by a half-finished session release automatically on rollback —
    the stale-recovery sweep (already running as the first step of
    each iteration) picks them up on restart.
    """
    from app.logging_setup import configure_logging
    configure_logging("mobius-rag-chunker")

    from app.worker.shutdown import (
        install_handlers, is_shutting_down, sleep_or_shutdown,
    )
    install_handlers(worker_name="chunking-worker")

    cfg = load_worker_config()
    logger.info("Worker %s starting...", WORKER_ID)

    # Startup migrations
    _run_startup_migrations_sync = [
        ("category_scores_to_columns", "app.migrations.category_scores_to_columns"),
        ("chunking_config_snapshot", "app.migrations.add_chunking_config_snapshot"),
        ("embeddable_units", "app.migrations.add_embeddable_units"),
        ("document_tags", "app.migrations.add_document_tags"),
        ("fix_offset_type", "app.migrations.fix_policy_lines_offset_type"),
        ("policy_lines_autovacuum", "app.migrations.tune_policy_lines_autovacuum"),
    ]
    for label, mod_path in _run_startup_migrations_sync:
        try:
            import importlib
            m = importlib.import_module(mod_path)
            await m.migrate()
        except Exception as migrate_err:
            logger.warning("Startup migration (%s) skipped/failed: %s", label, migrate_err)

    # Zombie-finalize sweep: catches any "processing" rows where the
    # coordinator legitimately wrote work but the finalize commit never
    # landed (DB blip, OOM, crash mid-finalize). Runs once per worker
    # boot, before the polling loop, so a fresh instance immediately
    # cleans up its predecessor's leftovers instead of waiting for
    # heartbeat-stale-recovery (which would reset to pending and
    # redo the work from scratch — wasteful when the work IS done).
    try:
        recovered_zombies = await recover_finalized_zombies()
        if recovered_zombies:
            logger.warning(
                "[zombie-finalize] Recovered %d completed-but-unmarked job(s) at startup",
                recovered_zombies,
            )
    except Exception as zexc:
        logger.warning("[zombie-finalize] startup sweep failed (non-fatal): %s", zexc)

    poll_count = 0
    while not is_shutting_down():
        try:
            # ── Periodic stale-job recovery ──────────────────────────────
            if poll_count % cfg.stale_recovery_interval_polls == 0:
                try:
                    async with AsyncSessionLocal() as recovery_db:
                        n = await recover_stale_jobs(
                            recovery_db,
                            timeout_minutes=cfg.stale_job_timeout_minutes,
                            worker_id=WORKER_ID,
                        )
                        if n:
                            logger.info("Recovered %d stale job(s)", n)
                except Exception as recovery_err:
                    logger.warning("Stale job recovery failed (non-fatal): %s", recovery_err)

            # ── Claim next pending job (atomic with FOR UPDATE SKIP LOCKED) ──
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(ChunkingJob)
                    .where(ChunkingJob.status == "pending")
                    .order_by(ChunkingJob.created_at)
                    .limit(1)
                    .with_for_update(skip_locked=True)
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
                    # Shutdown-aware sleep: returns immediately on
                    # SIGTERM so drain latency is ~100ms, not the
                    # full poll interval (typically 5-10s).
                    await sleep_or_shutdown(cfg.poll_interval_seconds)
        except Exception as e:
            logger.error("Error in worker loop: %s", e, exc_info=True)
            await sleep_or_shutdown(cfg.error_sleep_seconds)

    logger.info("Worker %s shutting down cleanly — loop exited.", WORKER_ID)


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
            # Non-fatal: extraction.py / critique.py fall through to the
            # shared llm_manager_client (chat's /internal/skill-llm) when
            # llm is None. The direct Vertex SDK is now a dev-only fallback
            # and may legitimately be absent locally.
            logger.info("[JOB %s] Direct LLM provider unavailable (%s); routing through shared LLM Manager",
                        job.id, e.__class__.__name__)
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
