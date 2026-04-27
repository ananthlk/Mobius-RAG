"""One-shot cleanup of orphaned rows in Chroma.

A Chroma row is an "orphan" when its ``document_id`` metadata no longer
exists in the rag Postgres ``documents`` table. These accumulate when
``/admin/db/documents/{id}/delete-cascade`` deletes from Postgres but
fails to propagate to Chroma (or when Chroma was written by an older
codepath that's since been removed).

Production impact: chat queries that hit phantom rows return citations
to non-existent documents — the source pill renders, the click-through
404s. We're killing Chroma in Step 7 of the pgvector migration anyway
but want to clean it up first so the parity-against-Chroma sign-off
test compares against a real baseline.

Idempotent. Safe to re-run. Targets BOTH Chroma collections:
  * ``chunk_embeddings``  (worker-output mirror)
  * ``published_rag``     (publish-contract mirror)

Usage::

    DATABASE_URL=...  CHROMA_HOST=...  CHROMA_PORT=8000  \
        CHROMA_AUTH_TOKEN=...  python -m scripts.clean_chroma_orphans

Pass ``--dry-run`` to count phantoms without deleting.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

import asyncpg
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("clean_chroma_orphans")


CHROMA_HOST = os.environ.get("CHROMA_HOST", "")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000") or "8000")
CHROMA_TOKEN = os.environ.get("CHROMA_AUTH_TOKEN", "")
CHROMA_SSL = (os.environ.get("CHROMA_SSL", "0") or "0").strip().lower() in ("1", "true", "yes")

PAGE_SIZE = 5000


def _chroma_base() -> str:
    scheme = "https" if CHROMA_SSL else "http"
    return f"{scheme}://{CHROMA_HOST}:{CHROMA_PORT}/api/v2/tenants/default_tenant/databases/default_database"


def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if CHROMA_TOKEN:
        h["X-Chroma-Token"] = CHROMA_TOKEN
    return h


async def list_collections(client: httpx.AsyncClient) -> list[dict]:
    r = await client.get(f"{_chroma_base()}/collections", headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()


async def collection_doc_ids(client: httpx.AsyncClient, coll_id: str) -> set[str]:
    """Page through every row in a collection, extract distinct document_ids
    from metadata. We use offset/limit pagination — Chroma supports it on
    /get."""
    out: set[str] = set()
    offset = 0
    while True:
        body = {"limit": PAGE_SIZE, "offset": offset, "include": ["metadatas"]}
        r = await client.post(
            f"{_chroma_base()}/collections/{coll_id}/get",
            headers=_headers(), json=body, timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        metas = data.get("metadatas") or []
        if not metas:
            break
        page_doc_ids = {m.get("document_id") for m in metas if m and m.get("document_id")}
        out.update(page_doc_ids)
        if len(metas) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return out


async def alive_doc_ids(pg: asyncpg.Connection, candidate_ids: list[str]) -> set[str]:
    """Return the subset of candidate_ids that exist in documents."""
    if not candidate_ids:
        return set()
    rows = await pg.fetch(
        "SELECT id::text FROM documents WHERE id::text = ANY($1::text[])",
        candidate_ids,
    )
    return {r["id"] for r in rows}


async def delete_chroma_rows_by_doc(client: httpx.AsyncClient, coll_id: str, doc_ids: list[str]) -> int:
    """Delete all rows in a collection whose metadata.document_id is in the list."""
    if not doc_ids:
        return 0
    # Batch deletes 50 doc_ids at a time so the where filter stays sane.
    deleted = 0
    for i in range(0, len(doc_ids), 50):
        batch = doc_ids[i : i + 50]
        body = {"where": {"document_id": {"$in": batch}}}
        r = await client.post(
            f"{_chroma_base()}/collections/{coll_id}/delete",
            headers=_headers(), json=body, timeout=120,
        )
        r.raise_for_status()
        deleted += len(batch)
        log.info("  deleted batch %d-%d of %d", i, i + len(batch), len(doc_ids))
    return deleted


async def main(dry_run: bool) -> int:
    if not CHROMA_HOST:
        log.error("CHROMA_HOST not set")
        return 2
    db_url = (
        os.environ.get("DATABASE_URL")
        or os.environ.get("RAG_DATABASE_URL")
        or os.environ.get("CHAT_RAG_DATABASE_URL")
    )
    if not db_url:
        log.error("DATABASE_URL not set")
        return 2

    # Strip the SQLAlchemy ``+asyncpg`` driver suffix if present.
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

    log.info("Connecting to Postgres...")
    pg = await asyncpg.connect(db_url)

    log.info("Connecting to Chroma at %s:%d ssl=%s", CHROMA_HOST, CHROMA_PORT, CHROMA_SSL)
    async with httpx.AsyncClient() as client:
        colls = await list_collections(client)
        log.info("Found %d Chroma collections", len(colls))

        total_phantoms = 0
        for c in colls:
            name = c.get("name")
            cid = c.get("id")
            if name in ("chat_answer_cache",):
                # Cache table — never tied to documents. Skip.
                continue
            log.info("=== Collection: %s (%s) ===", name, cid)

            doc_ids = await collection_doc_ids(client, cid)
            log.info("  distinct document_ids in chroma: %d", len(doc_ids))

            alive = await alive_doc_ids(pg, sorted(doc_ids))
            phantoms = sorted(doc_ids - alive)
            log.info("  alive in postgres documents:     %d", len(alive))
            log.info("  PHANTOM (chroma yes / pg no):    %d", len(phantoms))

            if phantoms and not dry_run:
                log.info("  deleting %d phantom doc-groups from %s ...", len(phantoms), name)
                deleted = await delete_chroma_rows_by_doc(client, cid, phantoms)
                log.info("  done: removed rows for %d phantom docs", deleted)
            elif phantoms:
                log.info("  --dry-run set — skipping delete")
            total_phantoms += len(phantoms)

    await pg.close()
    log.info("== summary == phantom doc-groups across all collections: %d (dry_run=%s)", total_phantoms, dry_run)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="Count phantoms without deleting")
    args = p.parse_args()
    sys.exit(asyncio.run(main(dry_run=args.dry_run)))
