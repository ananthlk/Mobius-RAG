"""
Embedding worker process - embeds hierarchical chunks and facts after chunking completes.

Polls embedding_jobs, loads chunks/facts, calls embedding API, writes chunk_embeddings and vector DB.
Queue-based parallelism: N workers each process one job at a time.
Commits after every batch so records appear incrementally.
"""
import asyncio
import logging
import os
import uuid as uuid_module
from datetime import datetime, timezone
from uuid import UUID
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal
from app.models import HierarchicalChunk, ExtractedFact, EmbeddingJob, ChunkEmbedding, ChunkingEvent
from app.services.embedding_provider import get_embedding_provider, embed_async
from app.services.vector_store import get_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [EMBED-WORKER] - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

WORKER_ID = f"embed-worker-{os.getpid()}-{datetime.now(timezone.utc).isoformat()}"

# Max concurrent embedding jobs (queue-based parallelism)
EMBEDDING_WORKER_CONCURRENCY = int(os.getenv("EMBEDDING_WORKER_CONCURRENCY", "2"))


def _utc_now_naive():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _build_text_for_chunk(chunk: HierarchicalChunk) -> str:
    """Build text to embed for a hierarchical chunk. Plan: text or summary + text if present."""
    if chunk.summary and chunk.text:
        return f"{chunk.summary}\n{chunk.text}"
    return chunk.text or ""


def _build_text_for_fact(fact: ExtractedFact) -> str:
    """Build text to embed for a fact. Plan: fact_text + optional key fields."""
    parts = [fact.fact_text or ""]
    extras = []
    if fact.who_eligible:
        extras.append(f"Who eligible: {fact.who_eligible}")
    if fact.how_verified:
        extras.append(f"How verified: {fact.how_verified}")
    if fact.limitations:
        extras.append(f"Limitations: {fact.limitations}")
    if extras:
        parts.append(" ".join(extras))
    return "\n".join(parts).strip() or ""


async def process_embedding_job(job: EmbeddingJob, db: AsyncSession) -> None:
    """Process one embedding job: load chunks/facts, embed, write chunk_embeddings + vector DB."""
    job_start = _utc_now_naive()
    try:
        job.status = "processing"
        job.worker_id = WORKER_ID
        job.started_at = job_start
        await db.commit()

        doc_uuid = UUID(str(job.document_id))
        items: list[tuple[str, str, str]] = []  # (source_type, source_id, text)

        # Load hierarchical chunks
        chunks_result = await db.execute(
            select(HierarchicalChunk)
            .where(HierarchicalChunk.document_id == doc_uuid)
            .order_by(HierarchicalChunk.page_number, HierarchicalChunk.paragraph_index)
        )
        chunks = chunks_result.scalars().all()
        for c in chunks:
            text = _build_text_for_chunk(c)
            if text:
                items.append(("hierarchical", str(c.id), text))

        # Load extracted facts
        facts_result = await db.execute(
            select(ExtractedFact).where(ExtractedFact.document_id == doc_uuid).order_by(ExtractedFact.created_at)
        )
        facts = facts_result.scalars().all()
        for f in facts:
            text = _build_text_for_fact(f)
            if text:
                items.append(("fact", str(f.id), text))

        if not items:
            logger.info("[JOB %s] No chunks or facts to embed for document %s", job.id, job.document_id)
            job.status = "completed"
            job.completed_at = _utc_now_naive()
            await db.commit()
            return

        logger.info("[JOB %s] Embedding %d items for document %s", job.id, len(items), job.document_id)

        # Get embedding provider
        config = None
        if job.embedding_config_version:
            # TODO: load from embedding config registry (similar to LLM)
            pass
        provider = get_embedding_provider(config)
        texts = [t for _, _, t in items]

        # Delete existing embeddings for this document (re-embed)
        await db.execute(delete(ChunkEmbedding).where(ChunkEmbedding.document_id == doc_uuid))
        vector_store = get_vector_store()
        vector_store.delete_by_document(str(job.document_id))
        await db.commit()
        logger.info("[JOB %s] Cleared existing embeddings, starting %d items", job.id, len(items))

        # Stream embedding_start event for live updates
        ev_start = ChunkingEvent(
            document_id=doc_uuid,
            event_type="embedding_start",
            event_data={"total_items": len(items), "message": f"Starting embedding of {len(items)} items"},
        )
        db.add(ev_start)
        await db.commit()

        # Embed in batches; commit after each batch so records appear incrementally
        batch_size = 50
        total_batches = (len(texts) + batch_size - 1) // batch_size
        total_embedded = 0
        embed_model = getattr(provider, "model", None)

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_items = items[i : i + batch_size]  # (source_type, source_id, text)
            batch_num = (i // batch_size) + 1

            logger.info("[JOB %s] Batch %d/%d: embedding %d items...", job.id, batch_num, total_batches, len(batch_texts))
            batch_emb = await embed_async(batch_texts, provider)

            ids_to_store: list[str] = []
            embeddings_to_store: list[list[float]] = []
            metadata_to_store: list[dict] = []

            for idx, (source_type, source_id, _) in enumerate(batch_items):
                if idx >= len(batch_emb):
                    break
                emb = batch_emb[idx]
                ce_id = uuid_module.uuid4()
                ce = ChunkEmbedding(
                    id=ce_id,
                    document_id=doc_uuid,
                    source_type=source_type,
                    source_id=UUID(source_id),
                    embedding=emb,
                    model=embed_model,
                )
                db.add(ce)
                ids_to_store.append(str(ce_id))
                embeddings_to_store.append(emb)
                metadata_to_store.append({
                    "document_id": str(job.document_id),
                    "source_type": source_type,
                    "source_id": source_id,
                })

            vector_store.add(ids_to_store, embeddings_to_store, metadata_to_store)
            await db.commit()

            total_embedded += len(ids_to_store)
            elapsed = (_utc_now_naive() - job_start).total_seconds()
            rate = total_embedded / elapsed if elapsed > 0 else 0
            logger.info("[JOB %s] Batch %d/%d done: %d/%d total | %.1fs | ~%.1f items/s",
                job.id, batch_num, total_batches, total_embedded, len(items), elapsed, rate)

            # Stream embedding_progress event for live updates
            ev_progress = ChunkingEvent(
                document_id=doc_uuid,
                event_type="embedding_progress",
                event_data={
                    "total_items": len(items),
                    "completed_items": total_embedded,
                    "batch_num": batch_num,
                    "total_batches": total_batches,
                    "elapsed_seconds": round(elapsed, 1),
                    "rate_per_sec": round(rate, 1),
                    "message": f"Batch {batch_num}/{total_batches}: {total_embedded}/{len(items)} items embedded",
                },
            )
            db.add(ev_progress)
            await db.commit()

        # Stream embedding_complete event
        ev_complete = ChunkingEvent(
            document_id=doc_uuid,
            event_type="embedding_complete",
            event_data={
                "total_items": total_embedded,
                "message": f"Embedding complete: {total_embedded} items stored",
            },
        )
        db.add(ev_complete)

        job.status = "completed"
        job.completed_at = _utc_now_naive()
        await db.commit()
        duration = (job.completed_at - job_start).total_seconds()
        logger.info("[JOB %s] Completed in %.2fs: %d embeddings stored", job.id, duration, total_embedded)

    except Exception as e:
        logger.error("[JOB %s] Failed: %s", job.id, e, exc_info=True)
        await db.rollback()
        try:
            err_doc_uuid = UUID(str(job.document_id))
            ev_err = ChunkingEvent(
                document_id=err_doc_uuid,
                event_type="embedding_error",
                event_data={"error": str(e)[:500], "message": f"Embedding failed: {e}"},
            )
            db.add(ev_err)
        except Exception:
            pass
        job.status = "failed"
        job.error_message = str(e)[:2000]
        job.completed_at = _utc_now_naive()
        await db.commit()


async def worker_task(worker_id: int) -> None:
    """Single worker: poll and process embedding jobs."""
    while True:
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(EmbeddingJob)
                    .where(EmbeddingJob.status == "pending")
                    .order_by(EmbeddingJob.created_at)
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
                job = result.scalar_one_or_none()
                if job:
                    logger.info("[Worker %d] Processing job %s for document %s", worker_id, job.id, job.document_id)
                    await process_embedding_job(job, db)
                else:
                    await asyncio.sleep(2)
        except Exception as e:
            logger.error("[Worker %d] Error: %s", worker_id, e, exc_info=True)
            await asyncio.sleep(5)


async def worker_loop() -> None:
    """Main loop: N workers processing embedding jobs."""
    logger.info("Embedding worker %s starting (concurrency=%d)", WORKER_ID, EMBEDDING_WORKER_CONCURRENCY)
    try:
        from app.migrations.add_embedding_tables import migrate as migrate_embedding_tables
        await migrate_embedding_tables()
    except Exception as e:
        logger.warning("Embedding tables migration skipped: %s", e)

    workers = [
        asyncio.create_task(worker_task(i))
        for i in range(EMBEDDING_WORKER_CONCURRENCY)
    ]
    await asyncio.gather(*workers)


def main():
    try:
        asyncio.run(worker_loop())
    except KeyboardInterrupt:
        logger.info("Embedding worker shutting down...")
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
