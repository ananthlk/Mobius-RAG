"""Migration: add ``search_vector`` GIN-indexed tsvector to discovered_sources.

Phase 13.5d — relevance-based registry lookup. Replaces "exact tag
match required" with BM25-style ranking over (payer, state, program,
host, path, authority, curation_notes). The chat planner can pass
imperfect topic terms (`topic="dental"`, `topic="risk adjustment"`)
and get ranked results without us pre-tagging every URL.

Implementation: GENERATED ALWAYS column. Postgres maintains the
vector automatically on every insert/update — service layer doesn't
have to remember. GIN index on the vector means @@ queries are
sub-millisecond at our scale.

Idempotent: skips if column or index already exists.
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from app.config import DATABASE_URL


# The text expression we want indexed. Includes everything a planner
# query might match against:
#   * payer / state / program — direct identifier matches
#   * host                    — "sunshinehealth" etc.
#   * path slugs              — "preauth-check", "billing-manual",
#                               "dental-plan-transition" become tokens
#                               after we replace / with space
#   * authority levels        — payer_manual / payer_policy / etc.
#   * topic_tags JSON         — the rare row that has them
#   * curation_notes          — operator-added free text
#
# regexp_replace turns '/' and '-' and '_' into spaces so URL slugs
# tokenize properly: '/providers/preauth-check/medicare-pre-auth.html'
# becomes 'providers preauth check medicare pre auth html'.
_VECTOR_EXPR = """
    to_tsvector('english',
        coalesce(payer,'') || ' ' ||
        coalesce(state,'') || ' ' ||
        coalesce(program,'') || ' ' ||
        coalesce(host,'') || ' ' ||
        coalesce(regexp_replace(path, '[/_\\-.]', ' ', 'g'), '') || ' ' ||
        coalesce(curated_authority_level, inferred_authority_level, '') || ' ' ||
        coalesce(topic_tags::text, '') || ' ' ||
        coalesce(curation_notes, '')
    )
"""


async def migrate():
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        # Add the generated column if missing.
        col_exists = await conn.fetchval("""
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='public'
              AND table_name='discovered_sources'
              AND column_name='search_vector'
        """)
        if not col_exists:
            sql = f"""
                ALTER TABLE public.discovered_sources
                ADD COLUMN search_vector tsvector
                GENERATED ALWAYS AS ({_VECTOR_EXPR.strip()}) STORED
            """
            await conn.execute(sql)
            print("  Added column discovered_sources.search_vector")
        else:
            print("  Column discovered_sources.search_vector already exists")

        # GIN index — fast @@ tsquery matching.
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ds_search_vector_gin
            ON public.discovered_sources USING gin(search_vector)
        """)
        print("  Ensured GIN index idx_ds_search_vector_gin")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_discovered_sources_search_vector completed.")
