"""Migration: add Router's 14-column redesign to rag_query_decisions.

Router's complete architecture rewrite (July 23) redesigned rag_query_decisions
from the old corpus_search_router schema (strategy_used, retrieval_grade, etc.)
to a new decision-centric schema capturing the full routing ladder, confidence
bars, per-module postures, and calibration metadata. The migration was never
applied, causing ALL decision rows (calibration and production) to fail to
persist since the rewrite.

Missing columns (14 total):
  - Routing core: depth_bucket, strategy_chosen, strategy_sequence
  - Router output: executed_ladder, shadow_ladder, confidence_bar
  - Deferred from upstream modules: gate_contour, gate_underspecified_kind,
    reformat_posture, reformat_fanout_n
  - Metrics: confidence, accuracy_estimate, cost, total_ms

All are forward-only additions (no backfill needed). Once added, new
persist_decision() calls will succeed.
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from app.config import DATABASE_URL


_ADD_COLUMNS_SQL = """
ALTER TABLE public.rag_query_decisions
ADD COLUMN depth_bucket INTEGER,
ADD COLUMN strategy_chosen TEXT,
ADD COLUMN strategy_sequence TEXT,
ADD COLUMN executed_ladder JSONB,
ADD COLUMN shadow_ladder JSONB,
ADD COLUMN confidence_bar DOUBLE PRECISION,
ADD COLUMN gate_contour TEXT,
ADD COLUMN gate_underspecified_kind TEXT,
ADD COLUMN reformat_posture TEXT,
ADD COLUMN reformat_fanout_n INTEGER,
ADD COLUMN confidence DOUBLE PRECISION,
ADD COLUMN accuracy_estimate DOUBLE PRECISION,
ADD COLUMN cost DOUBLE PRECISION,
ADD COLUMN total_ms INTEGER;
"""


async def migrate():
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        # Check which columns already exist
        existing = await conn.fetch("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema='public' AND table_name='rag_query_decisions'
            AND column_name IN (
                'depth_bucket', 'strategy_chosen', 'strategy_sequence',
                'executed_ladder', 'shadow_ladder', 'confidence_bar',
                'gate_contour', 'gate_underspecified_kind', 'reformat_posture',
                'reformat_fanout_n', 'confidence', 'accuracy_estimate', 'cost', 'total_ms'
            )
        """)
        existing_cols = {row["column_name"] for row in existing}

        if len(existing_cols) == 14:
            print("  All 14 Router columns already exist — skipping")
            return

        if existing_cols:
            print(f"  WARNING: partial migration state detected ({len(existing_cols)}/14 columns exist)")
            print(f"  Existing: {sorted(existing_cols)}")

        await conn.execute(_ADD_COLUMNS_SQL)
        print("  Added 14 Router decision columns to rag_query_decisions")
        print("    - depth_bucket, strategy_chosen, strategy_sequence")
        print("    - executed_ladder, shadow_ladder, confidence_bar")
        print("    - gate_contour, gate_underspecified_kind, reformat_posture, reformat_fanout_n")
        print("    - confidence, accuracy_estimate, cost, total_ms")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_router_decision_columns completed.")
