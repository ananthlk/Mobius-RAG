"""Migration: dedupe ``policy_paragraphs`` and add a unique index.

Background — the chunking worker's ``clear_policy_for_document``
helper used to swallow exceptions and return as if the cleanup
succeeded. When the cleanup actually failed (deadlock, timeout),
the worker would proceed to insert paragraphs ON TOP of the
unwiped old set, producing N× duplicate rows. Across 11 oversize
docs we saw 5×–35× duplication; the worst was "Constrained
Budgets HIV": 43 unique paragraphs stored as 1,497 rows.

This migration:

1. Deletes duplicate ``policy_paragraphs`` rows, keeping the row
   with the highest ``id`` per ``(document_id, page_number,
   order_index)`` tuple. Also cascades to dependent
   ``policy_lines`` rows.
2. Adds a UNIQUE index on the same key so future workers can no
   longer insert a duplicate at the DB level — the insert errors
   out instead of accumulating.

Idempotent: ``CREATE UNIQUE INDEX IF NOT EXISTS`` and the dedupe
DELETE is a no-op when there's nothing to dedupe.
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg  # noqa: E402

from app.config import DATABASE_URL  # noqa: E402


async def migrate() -> None:
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        before = await conn.fetchval("SELECT COUNT(*) FROM policy_paragraphs")
        print(f"  policy_paragraphs total before dedupe: {before}")

        # Use DISTINCT ON ordered by id DESC (UUIDv4 ordering is not
        # time-monotonic but for de-duplication "any one of the dupes
        # survives" is the only requirement). Postgres has no MAX()
        # for UUID, but DISTINCT ON works on any sortable type.
        keepers_cte = """
            WITH keepers AS (
                SELECT DISTINCT ON (document_id, page_number, order_index) id
                FROM policy_paragraphs
                ORDER BY document_id, page_number, order_index, id DESC
            )
        """

        # 1. Cascade delete dependent policy_lines for paragraph rows
        #    we're about to delete.
        await conn.execute(f"""
            {keepers_cte}
            DELETE FROM policy_lines
            WHERE paragraph_id IN (
                SELECT id FROM policy_paragraphs
                WHERE id NOT IN (SELECT id FROM keepers)
            )
        """)

        # 2. Delete the duplicate paragraph rows.
        result = await conn.execute(f"""
            {keepers_cte}
            DELETE FROM policy_paragraphs
            WHERE id NOT IN (SELECT id FROM keepers)
        """)
        # asyncpg returns "DELETE n" — extract the count
        deleted = int(result.split()[-1]) if result and result.split()[-1].isdigit() else 0
        after = await conn.fetchval("SELECT COUNT(*) FROM policy_paragraphs")
        print(f"  Deleted {deleted} duplicate policy_paragraphs rows")
        print(f"  policy_paragraphs total after dedupe: {after}")

        # 3. Add unique index. CONCURRENTLY would be nicer in prod but
        #    requires a separate transaction; keep it simple — at our
        #    scale (~1.5M rows post-dedupe) this completes quickly.
        await conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
              ux_policy_paragraphs_doc_page_order
            ON policy_paragraphs (document_id, page_number, order_index)
        """)
        print("  Ensured unique index: ux_policy_paragraphs_doc_page_order")

    finally:
        await conn.close()


def main() -> None:
    asyncio.run(migrate())
    print("\nMigration add_policy_paragraph_unique completed.")


if __name__ == "__main__":
    main()
