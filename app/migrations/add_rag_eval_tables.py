"""Migration: create ``rag_eval_runs`` + ``rag_eval_results`` tables.

Eval framework persistence — one row per batch run (`rag_eval_runs`)
and one row per query × run (`rag_eval_results`). Every result row
links back to the underlying ``rag_routing_decisions`` row so we can
reconstruct the full pipeline state offline.

Idempotent — safe to re-run.
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from app.config import DATABASE_URL


_CREATE_RUNS_SQL = """
CREATE TABLE IF NOT EXISTS public.rag_eval_runs (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ts                  TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- What was evaluated
    bank_path           TEXT NOT NULL,            -- e.g. "eval/queries.yaml"
    bank_version        TEXT,                     -- git sha or content hash
    priors_version      TEXT NOT NULL,            -- router priors snapshot
    caller_mode_filter  TEXT,                     -- "all" or specific mode
    notes               TEXT,
    config_dump         JSONB,                    -- full config snapshot

    -- Aggregates (filled at end of run)
    n_queries           INTEGER NOT NULL DEFAULT 0,
    n_correct           INTEGER DEFAULT 0,
    n_partial           INTEGER DEFAULT 0,
    n_wrong             INTEGER DEFAULT 0,
    n_unable            INTEGER DEFAULT 0,
    routing_accuracy    DOUBLE PRECISION,         -- expected.strategy == executed
    citation_hit_rate   DOUBLE PRECISION,
    median_latency_ms   INTEGER,
    p95_latency_ms      INTEGER,
    completed_at        TIMESTAMPTZ
);
"""

_CREATE_RESULTS_SQL = """
CREATE TABLE IF NOT EXISTS public.rag_eval_results (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id              UUID NOT NULL REFERENCES rag_eval_runs(id) ON DELETE CASCADE,
    ts                  TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Bank entry
    query_id            TEXT NOT NULL,            -- e.g. "q001"
    query               TEXT NOT NULL,
    expected            JSONB NOT NULL,           -- full expected dict from bank

    -- Link to the routing decision row written by the agent on this call.
    routing_decision_id UUID REFERENCES rag_routing_decisions(id),

    -- Captured response state (denormalized for fast queries)
    strategy_chosen     TEXT,
    strategy_executed   TEXT,
    confidence          TEXT,
    total_ms            INTEGER,
    n_chunks            INTEGER,
    top_rerank          DOUBLE PRECISION,
    llm_answer          TEXT,                     -- for c/d
    chunks_summary      JSONB,                    -- list of {doc_name, page, rerank, text}

    -- Deterministic checks
    routing_correct     BOOLEAN,                  -- expected.strategy == executed
    citation_hit        BOOLEAN,                  -- chunks/cites match must_cite_*
    fail_fast_correct   BOOLEAN,                  -- only meaningful when expected.strategy == 'e'

    -- LLM judge verdict
    judge_verdict       TEXT,                     -- correct | partial | wrong | unable
    judge_score         DOUBLE PRECISION,         -- 0..1
    judge_reasoning     TEXT,
    judge_model         TEXT,
    judge_ms            INTEGER
);
"""

_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_eval_runs_ts ON rag_eval_runs (ts DESC);",
    "CREATE INDEX IF NOT EXISTS idx_eval_runs_priors ON rag_eval_runs (priors_version);",
    "CREATE INDEX IF NOT EXISTS idx_eval_results_run ON rag_eval_results (run_id);",
    "CREATE INDEX IF NOT EXISTS idx_eval_results_query_id ON rag_eval_results (query_id);",
    "CREATE INDEX IF NOT EXISTS idx_eval_results_strategy ON rag_eval_results (strategy_executed);",
    "CREATE INDEX IF NOT EXISTS idx_eval_results_judge ON rag_eval_results (judge_verdict);",
    ("CREATE INDEX IF NOT EXISTS idx_eval_results_routing "
     "ON rag_eval_results (routing_decision_id) "
     "WHERE routing_decision_id IS NOT NULL;"),
]


async def migrate():
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        for table, sql in (("rag_eval_runs", _CREATE_RUNS_SQL),
                           ("rag_eval_results", _CREATE_RESULTS_SQL)):
            exists = await conn.fetchval("""
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = $1
            """, table)
            if exists:
                print(f"  Table public.{table} already exists")
            else:
                await conn.execute(sql)
                print(f"  Created table public.{table}")
        for sql in _CREATE_INDEXES_SQL:
            await conn.execute(sql)
        print(f"  Ensured {len(_CREATE_INDEXES_SQL)} indexes")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_rag_eval_tables completed.")
