"""Durable sync from ``rag_published_embeddings`` (rag's mart) to the
two retrieval stores chat actually queries:

* **Chroma** vector index (collection ``published_rag``) — embeddings + metadata
* **chat Postgres** ``published_rag_metadata`` table — text + filterable fields

Same UUID flows through both stores per the contract documented in
mobius-chat/docs/rag_population_agent_setup.md. If the IDs drift, chat
gets vector hits whose metadata it can't hydrate and silently drops
them.

Why this lives here, not in ``publish.py``:
    * Re-runnable for backfill (a separate ``backfill_published_to_chat.py``
      can call ``sync_document_to_retrieval_stores`` for any doc).
    * Clean test boundary — publish.py's tx logic doesn't change.
    * Best-effort failure mode — chat-side outage shouldn't fail an
      otherwise-successful publish on the rag side.

Activation: env-gated. If ``CHROMA_HOST`` or ``CHAT_DATABASE_URL`` is
unset, the call is a documented no-op (returns ``skipped`` status).
This keeps the service runnable in environments that don't have the
chat-side stack provisioned (early dev, tests).
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import RagPublishedEmbedding
from app.services.metadata_canonical import (
    canonical_authority_level,
    canonical_payer,
    canonical_program,
    canonical_state,
    canonical_status,
    canonical_source_type,
)

logger = logging.getLogger(__name__)


# ── Result type ──────────────────────────────────────────────────────


@dataclass
class SyncResult:
    chunks_synced: int = 0
    chroma_status: str = "not_attempted"
    chroma_message: str | None = None
    chat_pg_status: str = "not_attempted"
    chat_pg_message: str | None = None
    duration_s: float = 0.0
    skipped_reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.chroma_status in ("ok", "skipped")
            and self.chat_pg_status in ("ok", "skipped")
        )


# ── Env helpers ──────────────────────────────────────────────────────


def _env_or_none(key: str) -> str | None:
    v = os.environ.get(key)
    if not v:
        return None
    v = v.strip()
    return v or None


# ── Public entry point ───────────────────────────────────────────────


async def sync_document_to_retrieval_stores(
    document_id: UUID,
    db: AsyncSession,
    *,
    chroma_collection: str = "published_rag",
) -> SyncResult:
    """Sync all rows for ``document_id`` from ``rag_published_embeddings``
    to Chroma + chat Postgres. Idempotent (Chroma upsert + Postgres
    ON CONFLICT). Returns a SyncResult with per-store status and counts.

    Best-effort: errors against either downstream store are logged and
    surfaced in the result but do NOT raise — the rag-side publish is
    already committed. Operator can re-run via the backfill script.
    """
    started = time.monotonic()
    result = SyncResult()

    # Step 5 of Chroma → pgvector migration: when VECTOR_STORE=pgvector,
    # chat reads from rag_published_embeddings.embedding_vec directly
    # via its own PgVectorStore client — there's no Chroma to keep in
    # sync. Gate the Chroma upsert behind the env so the codepath stays
    # intact for emergency rollback. Step 7 deletes the gated branch.
    vector_store_mode = (os.environ.get("VECTOR_STORE") or "chroma").strip().lower()
    chroma_disabled_for_pgvector = vector_store_mode == "pgvector"

    chroma_host = _env_or_none("CHROMA_HOST")
    chroma_port = int(_env_or_none("CHROMA_PORT") or "8000")
    chroma_ssl_str = (_env_or_none("CHROMA_SSL") or "0").strip().lower()
    chroma_ssl = chroma_ssl_str in ("1", "true", "yes")
    chroma_token = _env_or_none("CHROMA_AUTH_TOKEN")
    chat_db_url = _env_or_none("CHAT_DATABASE_URL")

    if chroma_disabled_for_pgvector:
        result.chroma_status = "skipped"
        result.skipped_reasons.append("VECTOR_STORE=pgvector (Chroma write gated off)")
        logger.info(
            "[publish_sync] %s: skipping Chroma upsert (VECTOR_STORE=pgvector)",
            document_id,
        )
        # Force-clear chroma_host so the upsert branch below is skipped.
        chroma_host = None
    elif not chroma_host:
        result.chroma_status = "skipped"
        result.skipped_reasons.append("CHROMA_HOST unset")
    if not chat_db_url:
        result.chat_pg_status = "skipped"
        result.skipped_reasons.append("CHAT_DATABASE_URL unset")
    if not chroma_host and not chat_db_url:
        result.duration_s = time.monotonic() - started
        logger.info("[publish_sync] %s: both stores skipped (env unset)", document_id)
        return result

    # ── Read rows from the just-published mart ──────────────────────
    rows_q = await db.execute(
        select(RagPublishedEmbedding)
        .where(RagPublishedEmbedding.document_id == document_id)
        .order_by(RagPublishedEmbedding.page_number, RagPublishedEmbedding.paragraph_index)
    )
    rows: list[RagPublishedEmbedding] = list(rows_q.scalars().all())
    if not rows:
        result.duration_s = time.monotonic() - started
        logger.info("[publish_sync] %s: no rag_published_embeddings rows; nothing to sync", document_id)
        return result

    # ── Build payloads (canonicalize metadata on the way out) ───────
    chroma_ids: list[str] = []
    chroma_embeddings: list[list[float]] = []
    chroma_documents: list[str] = []
    chroma_metadatas: list[dict] = []
    pg_rows: list[tuple] = []

    skipped_emb = 0
    for r in rows:
        text = r.text or ""
        emb = r.embedding
        # rag_published_embeddings.embedding is jsonb (array). asyncpg may
        # decode to list directly; handle string fallback.
        if isinstance(emb, str):
            import json as _json
            try:
                emb = _json.loads(emb)
            except Exception:
                logger.warning("[publish_sync] %s: row %s embedding could not be parsed as JSON, skipping", document_id, r.id)
                skipped_emb += 1
                continue
        if not isinstance(emb, list) or not emb:
            skipped_emb += 1
            continue
        try:
            embedding = [float(x) for x in emb]
        except Exception:
            skipped_emb += 1
            continue
        if len(embedding) != 1536:
            logger.warning(
                "[publish_sync] %s: row %s embedding dim=%d (expected 1536), skipping",
                document_id, r.id, len(embedding),
            )
            skipped_emb += 1
            continue

        # Canonicalize metadata. Payer/state/program/authority canonicalization
        # is the second-line defense after the upload + PATCH normalization
        # we shipped in 80b7717 — historic data may still have free-text
        # variants (e.g. "Sunshine health" lowercase). One UPDATE here
        # produces clean filters in Chroma + chat Postgres without a
        # 4-store backfill.
        payer    = canonical_payer(r.document_payer)    or ""
        state    = canonical_state(r.document_state)    or ""
        program  = canonical_program(r.document_program) or ""
        auth_lvl = canonical_authority_level(r.document_authority_level) or ""
        status   = canonical_status(r.document_status)  or "published"
        src_type = canonical_source_type(r.source_type) or "chunk"

        chroma_ids.append(str(r.id))
        chroma_embeddings.append(embedding)
        # Cap doc text size in Chroma payload — vector quality is unchanged,
        # we only need the text for display fallback. Postgres carries the
        # full text via published_rag_metadata.text.
        chroma_documents.append(text[:8000])
        chroma_metadatas.append({
            "document_id":              str(r.document_id),
            "document_payer":           payer,
            "document_state":           state,
            "document_program":         program,
            "document_authority_level": auth_lvl,
            "source_type":              src_type,
            # Critical: distinguishes approved corpus from user uploads.
            # chat's filter is {"instant_rag": {"$ne": "true"}}; setting
            # "true" here would make the chunk invisible to the corpus
            # search.
            "instant_rag":              "false",
        })

        pg_rows.append((
            str(r.id), str(r.document_id), src_type,
            str(r.source_id) if r.source_id else str(r.id),
            r.model or "gemini-embedding-001",
            r.created_at,
            text,
            r.page_number, r.paragraph_index,
            r.section_path, r.chapter_path, r.summary,
            r.document_filename, r.document_display_name,
            auth_lvl,
            r.document_effective_date, r.document_termination_date,
            payer, state, program,
            status, r.document_created_at,
            r.document_review_status, r.document_reviewed_at, r.document_reviewed_by,
            r.content_sha or "", r.updated_at,
            r.source_verification_status,
        ))

    if skipped_emb:
        logger.warning("[publish_sync] %s: %d rows skipped due to malformed embeddings", document_id, skipped_emb)

    if not chroma_ids:
        result.duration_s = time.monotonic() - started
        logger.warning("[publish_sync] %s: no syncable rows after embedding validation", document_id)
        return result

    # ── Run the two stores in a thread (sync libs) ──────────────────
    # chromadb + psycopg2 are sync; offload so we don't block the
    # FastAPI event loop.

    if chroma_host:
        try:
            await asyncio.to_thread(
                _chroma_upsert_batch,
                host=chroma_host,
                port=chroma_port,
                ssl=chroma_ssl,
                token=chroma_token,
                collection=chroma_collection,
                ids=chroma_ids,
                embeddings=chroma_embeddings,
                documents=chroma_documents,
                metadatas=chroma_metadatas,
            )
            result.chroma_status = "ok"
        except Exception as exc:
            logger.exception("[publish_sync] %s: Chroma upsert failed", document_id)
            result.chroma_status = "error"
            result.chroma_message = f"{type(exc).__name__}: {exc}"[:500]

    if chat_db_url:
        try:
            await asyncio.to_thread(
                _chat_pg_upsert_batch,
                dsn=chat_db_url,
                rows=pg_rows,
            )
            result.chat_pg_status = "ok"
        except Exception as exc:
            logger.exception("[publish_sync] %s: chat Postgres upsert failed", document_id)
            result.chat_pg_status = "error"
            result.chat_pg_message = f"{type(exc).__name__}: {exc}"[:500]

    result.chunks_synced = len(chroma_ids)
    result.duration_s = round(time.monotonic() - started, 2)
    logger.info(
        "[publish_sync] %s: synced %d chunks (chroma=%s, chat_pg=%s, %.2fs)",
        document_id, result.chunks_synced, result.chroma_status, result.chat_pg_status, result.duration_s,
    )
    return result


# ── Sync workers (run in to_thread) ──────────────────────────────────


def _chroma_upsert_batch(
    *,
    host: str, port: int, ssl: bool, token: str | None,
    collection: str,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    import chromadb

    headers: dict[str, str] = {}
    if token:
        headers["X-Chroma-Token"] = token
    client = chromadb.HttpClient(host=host, port=port, ssl=ssl, headers=headers)
    coll = client.get_collection(collection)

    BATCH = 100
    for i in range(0, len(ids), BATCH):
        s = slice(i, i + BATCH)
        coll.upsert(
            ids=ids[s],
            embeddings=embeddings[s],
            documents=documents[s],
            metadatas=metadatas[s],
        )


_INSERT_SQL = """
INSERT INTO published_rag_metadata (
    id, document_id, source_type, source_id, model,
    created_at, text, page_number, paragraph_index,
    section_path, chapter_path, summary,
    document_filename, document_display_name,
    document_authority_level,
    document_effective_date, document_termination_date,
    document_payer, document_state, document_program,
    document_status, document_created_at,
    document_review_status, document_reviewed_at, document_reviewed_by,
    content_sha, updated_at, source_verification_status
) VALUES %s
ON CONFLICT (id) DO UPDATE SET
    text = EXCLUDED.text,
    document_payer = EXCLUDED.document_payer,
    document_state = EXCLUDED.document_state,
    document_program = EXCLUDED.document_program,
    document_authority_level = EXCLUDED.document_authority_level,
    document_status = EXCLUDED.document_status,
    source_type = EXCLUDED.source_type,
    content_sha = EXCLUDED.content_sha,
    updated_at = EXCLUDED.updated_at
"""


def _chat_pg_upsert_batch(*, dsn: str, rows: list[tuple]) -> None:
    import psycopg2
    import psycopg2.extras

    # CHAT_DATABASE_URL may be a SQLAlchemy URL (postgresql+psycopg2://)
    # or a plain libpq URL. Strip the driver suffix if present.
    if dsn.startswith("postgresql+psycopg2://"):
        dsn = "postgresql://" + dsn[len("postgresql+psycopg2://"):]
    if dsn.startswith("postgresql+asyncpg://"):
        dsn = "postgresql://" + dsn[len("postgresql+asyncpg://"):]

    conn = psycopg2.connect(dsn, connect_timeout=15)
    try:
        conn.autocommit = False
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, _INSERT_SQL, rows, page_size=100)
        conn.commit()
    finally:
        conn.close()
