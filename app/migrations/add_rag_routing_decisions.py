"""Migration: create ``rag_routing_decisions`` table (Phase: router observability).

One row per call to ``corpus_search_agent``. Append-only. Captures the
full routing context (preferences received + resolved, scores, self-
assessments, withdrawal, primary + fallback) plus the immediate outcome
(confidence, n_chunks, total_ms) and reserved fields for delayed
feedback signals (eval verdict, LLM judge, user feedback, critique).

Used by:
  * the eval harness — joins ``eval_run_id`` to mark which queries ran
    in which eval batch, populates ``llm_judge_*`` and the boolean
    correctness fields after the fact
  * the RAG frontend's Routing tab — live ticker + drill-down
  * future bandit training — every column is a feature or label

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
CREATE TABLE IF NOT EXISTS public.rag_routing_decisions (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ts                       TIMESTAMPTZ NOT NULL DEFAULT now(),
    agent_id                 TEXT NOT NULL,
    query                    TEXT NOT NULL,

    -- Profile snapshot ────────────────────────────────────────────
    -- (subset of classify_query output we want to filter / group by)
    query_type               TEXT,
    query_class              TEXT,
    coverage                 DOUBLE PRECISION,
    has_d_tag                BOOLEAN,
    has_literal              BOOLEAN,
    is_exploratory           BOOLEAN,
    tag_matches              JSONB,
    literal_anchors          JSONB,
    untagged_meaningful      JSONB,

    -- Caller preferences ───────────────────────────────────────────
    caller_mode              TEXT,
    prefs_received           JSONB,
    prefs_resolved           JSONB,

    -- Routing decision ─────────────────────────────────────────────
    routing_method           TEXT NOT NULL,            -- deterministic / override / fail_fast
    scores                   JSONB,                    -- {a, b, c, d} -> float
    self_assessments         JSONB,                    -- per-strategy est_recall + reason
    withdrawn                JSONB,                    -- list of withdrawn strategy ids
    strategy_chosen          TEXT NOT NULL,            -- router's pick
    strategy_executed        TEXT NOT NULL,            -- what actually ran (may differ if not built)
    fallback_strategy        TEXT,
    priors_version           TEXT NOT NULL,
    fail_fast_reason         TEXT,

    -- Outcome ──────────────────────────────────────────────────────
    confidence               TEXT,                     -- high / medium / low
    n_chunks                 INTEGER,
    top_rerank               DOUBLE PRECISION,
    total_ms                 INTEGER,
    per_strategy_telemetry   JSONB,

    -- Delayed feedback (filled by eval / user / critique) ──────────
    eval_run_id              UUID,
    routing_correct          BOOLEAN,
    citation_hit             BOOLEAN,
    keyword_hit              BOOLEAN,
    llm_judge_verdict        TEXT,                     -- correct / partial / wrong
    llm_judge_score          DOUBLE PRECISION,         -- 0..1
    llm_judge_reasoning      TEXT,
    user_feedback            JSONB,
    critique_verdict         TEXT
);
"""

_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_rrd_ts ON rag_routing_decisions (ts DESC);",
    "CREATE INDEX IF NOT EXISTS idx_rrd_agent_id ON rag_routing_decisions (agent_id);",
    "CREATE INDEX IF NOT EXISTS idx_rrd_strategy_executed ON rag_routing_decisions (strategy_executed);",
    "CREATE INDEX IF NOT EXISTS idx_rrd_query_class ON rag_routing_decisions (query_class);",
    "CREATE INDEX IF NOT EXISTS idx_rrd_caller_mode ON rag_routing_decisions (caller_mode);",
    ("CREATE INDEX IF NOT EXISTS idx_rrd_eval_run_id "
     "ON rag_routing_decisions (eval_run_id) "
     "WHERE eval_run_id IS NOT NULL;"),
]


async def migrate():
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        exists = await conn.fetchval("""
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'rag_routing_decisions'
        """)
        if exists:
            print("  Table public.rag_routing_decisions already exists")
        else:
            await conn.execute(_CREATE_TABLE_SQL)
            print("  Created table public.rag_routing_decisions")

        for sql in _CREATE_INDEXES_SQL:
            await conn.execute(sql)
        print(f"  Ensured {len(_CREATE_INDEXES_SQL)} indexes on rag_routing_decisions")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_rag_routing_decisions completed.")
