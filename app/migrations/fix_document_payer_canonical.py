"""Migration: normalize rag_published_embeddings.document_payer from canonical documents.payer.

Historical bug: an old publish path stamped document_payer from the crawl-domain
string (e.g. 'Ahca.myflorida' from DiscoveredSource.host) instead of the canonical
documents.payer ('AHCA'). Result: 1,050,893 AHCA chunks were invisible to any
retrieval filter or pool keyed on payer='AHCA'.

Fix: UPDATE document_payer from the joined documents.payer for any row where they
disagree and documents.payer is non-null/non-empty. Idempotent — re-running when
all rows already match is a no-op (WHERE condition matches zero rows).
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg  # noqa: E402

from app.config import DATABASE_URL  # noqa: E402


def _parse_db_url(url: str) -> dict:
    import re
    m = re.match(
        r"postgresql(?:\+asyncpg)?://(?P<user>[^:]+):(?P<password>[^@]+)"
        r"@(?P<host>[^:/]+):(?P<port>\d+)/(?P<database>.+)",
        url,
    )
    if not m:
        raise ValueError(f"Cannot parse DATABASE_URL: {url!r}")
    return {
        "host": m.group("host"),
        "port": int(m.group("port")),
        "user": m.group("user"),
        "password": m.group("password"),
        "database": m.group("database"),
        "ssl": False,
    }


async def migrate() -> None:
    conn = await asyncpg.connect(**_parse_db_url(DATABASE_URL))
    try:
        # Count mismatches before
        before = await conn.fetchval("""
            SELECT COUNT(*)
            FROM rag_published_embeddings rpe
            JOIN documents d ON d.id = rpe.document_id
            WHERE rpe.document_payer != COALESCE(d.payer, '')
              AND d.payer IS NOT NULL
              AND d.payer != ''
        """)
        print(f"  document_payer mismatches before fix: {before}")

        if before == 0:
            print("  Nothing to fix — all document_payer values are already canonical.")
            return

        result = await conn.execute("""
            UPDATE rag_published_embeddings rpe
            SET document_payer = d.payer
            FROM documents d
            WHERE rpe.document_id = d.id
              AND rpe.document_payer != COALESCE(d.payer, '')
              AND d.payer IS NOT NULL
              AND d.payer != ''
        """)
        updated = int(result.split()[-1]) if result and result.split()[-1].isdigit() else 0
        print(f"  Fixed {updated} rag_published_embeddings rows (document_payer → canonical)")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration fix_document_payer_canonical completed.")
