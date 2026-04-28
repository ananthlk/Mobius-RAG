"""Migration: add ``documents.expires_at`` for ephemeral chat/agent uploads.

Part of the mobius-chat ↔ mobius-rag upload consolidation
(2026-04-27). Chat-uploaded documents need a TTL so the
``/admin/cleanup_expired_documents`` cron can drop them after the
agreed retention window (default 7 days). Durable rag-UI uploads
leave ``expires_at = NULL`` and are never auto-deleted.

Properties:

* Idempotent — ``ADD COLUMN IF NOT EXISTS`` and
  ``CREATE INDEX IF NOT EXISTS``. Safe to re-run.
* Partial index — only rows with a non-NULL ``expires_at`` are
  indexed, so the durable corpus pays no index-maintenance cost.
* Prints row counts before/after so the operator can sanity-check
  no rows were perturbed.

Usage (run via Cloud Run Job — DO NOT run on prod without explicit
operator approval; the dispatch this implements is dev-only):

    python -m app.migrations.add_documents_expires_at
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg  # noqa: E402

from app.config import DATABASE_URL  # noqa: E402


async def _row_counts(conn: asyncpg.Connection) -> tuple[int, int]:
    total = await conn.fetchval("SELECT count(*) FROM documents")
    with_expiry = await conn.fetchval(
        "SELECT count(*) FROM documents WHERE expires_at IS NOT NULL"
    )
    return int(total or 0), int(with_expiry or 0)


async def migrate() -> None:
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        total_before, expiring_before = await _row_counts(conn)
        print(
            f"  Pre-migration: documents total={total_before} "
            f"with_expires_at={expiring_before}"
        )

        await conn.execute(
            "ALTER TABLE documents "
            "ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ"
        )
        print("  Ensured column: documents.expires_at")

        await conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_documents_expires_at "
            "ON documents (expires_at) WHERE expires_at IS NOT NULL"
        )
        print("  Ensured partial index: ix_documents_expires_at")

        total_after, expiring_after = await _row_counts(conn)
        print(
            f"  Post-migration: documents total={total_after} "
            f"with_expires_at={expiring_after}"
        )
    finally:
        await conn.close()


def main() -> None:
    asyncio.run(migrate())
    print("\nMigration add_documents_expires_at completed.")


if __name__ == "__main__":
    main()
