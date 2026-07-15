"""Migration: create ``rag_query_decisions`` table (OBSERVE block substrate).

One row per corpus_search_agent call where grades were computed. Append-only.
Captures IMMEDIATE two-grade QA at query time — unlike rag_routing_decisions
(which takes delayed eval feedback), these grades are computed inline:

  retrieval_grade = fact_checker(must_facts, CHUNKS)   → router quality signal
  synthesis_grade = fact_checker(must_facts, ANSWER)   → synthesizer quality (prod only)
  synthesis_gap   = retrieval_grade − synthesis_grade  → synthesis loss

Credit assignment:
  - router_reward      = retrieval_grade  (bandit: reward routing/retrieval)
  - synthesizer_reward = synthesis_grade  (prod rows only; null in offline eval)

Grader:
  - fact_checker_version is set by the EVAL agent (owns the rubric).
  - All callers (eval offline, chat prod) call the same registered stage and
    log the same version → eval and prod rows join cleanly.

Corpus version:
  - corpus_version is a fingerprint of the retrieval corpus at query time.
  - Used for drift detection: same query + different corpus_version → score
    change may be corpus-driven, not model-driven.

Multi-invoke:
  - invoke_all TEXT[] is null for single-arm queries.
  - When set, both strategy arms are listed (e.g. ['a','b']).
  - Enables separate bandit reward for multi-invoke vs single-arm routing.

Idempotent — safe to re-run.
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from app.config import DATABASE_URL


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.rag_query_decisions (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ts                   TIMESTAMPTZ NOT NULL DEFAULT now(),
    agent_id             TEXT NOT NULL,
    query                TEXT NOT NULL,

    -- Routing ─────────────────────────────────────────────────────────
    strategy_used        TEXT NOT NULL,           -- a/b/c/d/e/multi
    invoke_all           TEXT[],                  -- null = single-arm
    priors_version       TEXT NOT NULL,

    -- Retrieval ────────────────────────────────────────────────────────
    n_chunks             INTEGER,
    top_rerank_score     DOUBLE PRECISION,
    corpus_version       TEXT,                    -- fingerprint for drift detection

    -- Two-grade QA ────────────────────────────────────────────────────
    -- fact_checker_version is OWNED by the EVAL agent; consumers log it as-is.
    fact_checker_version TEXT,
    retrieval_grade      DOUBLE PRECISION,        -- fact_checker(must_facts, CHUNKS)
    synthesis_grade      DOUBLE PRECISION,        -- fact_checker(must_facts, ANSWER); null if no answer
    synthesis_gap        DOUBLE PRECISION,        -- retrieval_grade − synthesis_grade; null if no synthesis

    -- Bandit reward signals ────────────────────────────────────────────
    -- router_reward = retrieval_grade (computed at read time, not stored separately)
    -- synthesizer_reward = synthesis_grade for prod rows (null for eval-only rows)
    -- is_prod distinguishes prod OBSERVE rows from eval-offline rows.
    is_prod              BOOLEAN NOT NULL DEFAULT false,

    -- Caller context ──────────────────────────────────────────────────
    caller               TEXT,
    caller_id            TEXT,
    eval_run_id          UUID                     -- set for offline eval rows; null for prod
);
"""

_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_rqd_ts ON rag_query_decisions (ts DESC);",
    "CREATE INDEX IF NOT EXISTS idx_rqd_strategy ON rag_query_decisions (strategy_used);",
    "CREATE INDEX IF NOT EXISTS idx_rqd_priors_version ON rag_query_decisions (priors_version);",
    "CREATE INDEX IF NOT EXISTS idx_rqd_fact_checker_version ON rag_query_decisions (fact_checker_version);",
    "CREATE INDEX IF NOT EXISTS idx_rqd_corpus_version ON rag_query_decisions (corpus_version);",
    "CREATE INDEX IF NOT EXISTS idx_rqd_is_prod ON rag_query_decisions (is_prod);",
    ("CREATE INDEX IF NOT EXISTS idx_rqd_eval_run_id "
     "ON rag_query_decisions (eval_run_id) "
     "WHERE eval_run_id IS NOT NULL;"),
]


async def migrate():
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        exists = await conn.fetchval("""
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'rag_query_decisions'
        """)
        if exists:
            print("  Table public.rag_query_decisions already exists — skipping create")
        else:
            await conn.execute(_CREATE_TABLE_SQL)
            print("  Created table public.rag_query_decisions")

        for sql in _CREATE_INDEXES_SQL:
            await conn.execute(sql)
        print(f"  Ensured {len(_CREATE_INDEXES_SQL)} indexes on rag_query_decisions")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_rag_query_decisions completed.")
