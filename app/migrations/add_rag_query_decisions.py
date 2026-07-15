"""Migration: create ``rag_query_decisions`` (OBSERVE substrate) + ``corpus_state``.

Final shape ratified 2026-07-15 by the EVAL agent (QA/schema semantics owner)
and the Database agent (persistence/indexing owner). Supersedes the first
draft, which lacked the bandit's leaf-key hot path entirely and carried seven
low-cardinality single-column indexes on a row-per-prod-query write path.

One row per corpus_search_agent call where grades were computed. Append-only.
Captures IMMEDIATE two-grade QA at query time:

  retrieval_grade = fact_checker(must_facts, CHUNKS)   → router quality signal
  synthesis_grade = fact_checker(must_facts, ANSWER)   → synthesizer quality (prod only)
  synthesis_gap   = synthesis_grade − retrieval_grade  → neg = synthesis loss
                                                          (drop), pos = hallucination (bluff)

Credit assignment:
  - router_reward      = retrieval_grade  (computed at read time, not stored)
  - synthesizer_reward = synthesis_grade  (prod rows only; null in offline eval)

LEAF_KEY — deterministic serialization (EVAL agent spec, verbatim):
  leaf_key = the action-tree TERMINAL the query resolved to — NOT the raw
  feature vector. Canonical form:
      {action}:{arms}   where action ∈ {skim, route, union, reformulate, floor}
      arms = strategy letters, sorted alphabetically, joined by '+'
  Examples: ``route:a`` · ``union:a+b`` · ``reformulate:b`` · ``floor:f`` · ``skim:s``
  Rules: single-arm → one letter; multi-invoke union → sorted '+'-join
  (a+b == b+a, always emit a+b); reformulate → the landing strategy; fixed
  action vocabulary (no free text). This is what "mean reward per leaf"
  aggregates — discrete, countable, stable across corpus_version.
  ``feature_vector`` JSONB holds the raw linear features ({exclusivity,
  literal, corpus_depth, thematic_policy, wide_pool, inheritance, gap}) —
  introspection now, the contextual bandit's context later. Two columns,
  two consumers.

PER_CLAIM_LEDGER — compact (EVAL agent spec, verbatim):
  jsonb array, one obj per must_fact/claim:
      {"fact": "<short label>", "status": "validated"|"unvalidated"|"contradicted",
       "chunk_id": <int|null>, "support": 0.0|0.5|1.0}
  NO verbatim quote in the persisted row (display-time — UX pulls the quote
  from the chunk via chunk_id). status is the enum; support is the 0/0.5/1
  score for rollup math. Versioned under fact_checker_version — enum-set
  changes bump the version, never mutate in place.

CORPUS_VERSION — bump-at-mutation, never compute-at-query:
  ``corpus_state`` is a single-row table; every code path that mutates
  retrieval (publish / republish / delete on rag_published_embeddings,
  lexicon retag apply) increments it IN THE SAME TRANSACTION:
      UPDATE corpus_state SET corpus_version = corpus_version + 1,
             content_hint = $1, updated_at = now();
  The OBSERVE writer stamps rows from an in-process cache (30–60s TTL) —
  per-query cost ≈ zero. Do NOT derive the version from max(tagged_at)-style
  scans and do NOT invent per-writer hashes.

WRITE-PATH CONTRACT (persistence owner):
  - OBSERVE writes are fire-and-forget background tasks — never on the
    query's critical path.
  - ``id`` is CLIENT-generated (uuid4) + INSERT ... ON CONFLICT (id) DO
    NOTHING, so any retry layer is idempotent. The column default is only a
    fallback for ad-hoc inserts.
  - Writes ride a shared pool — never per-call connects.

Relationship to the eval tables — COEXIST, not supersede:
  rag_eval_results keeps per-arm/chunk-level detail (heavy, eval-only);
  this table is the per-query decision+reward row (light, both modes).
  Join: (eval_run_id, query_id) → rag_eval_results(run_id, query_id).
  Prod rows join chat via correlation_id (cross-DB — deliberately NO FK).

Idempotent — safe to re-run. If the pre-ratification draft table exists
EMPTY it is dropped and recreated; if it has rows this migration refuses
(that would need a hand-written ALTER + backfill instead).
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from app.config import DATABASE_URL


_FINAL_COLUMNS = {
    "leaf_key", "feature_vector", "strategy_scores", "per_claim_ledger",
    "query_id", "correlation_id",
}

_CREATE_TABLE_SQL = """
CREATE TABLE public.rag_query_decisions (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ts                   TIMESTAMPTZ NOT NULL DEFAULT now(),
    agent_id             TEXT NOT NULL,
    query                TEXT NOT NULL,

    -- Routing ─────────────────────────────────────────────────────────
    strategy_used        TEXT NOT NULL,           -- a/b/c/d/e/multi
    invoke_all           TEXT[],                  -- null = single-arm
    priors_version       TEXT NOT NULL,
    leaf_key             TEXT NOT NULL,           -- action-tree terminal (docstring spec)
    feature_vector       JSONB,                   -- raw linear features
    strategy_scores      JSONB,                   -- {"a": 0.42, "b": 0.61, ...}

    -- Retrieval ────────────────────────────────────────────────────────
    n_chunks             INTEGER,
    top_rerank_score     DOUBLE PRECISION,
    corpus_version       BIGINT,                  -- from corpus_state (cached read)

    -- Two-grade QA (EVAL agent owns semantics + versioning) ───────────
    fact_checker_version TEXT,
    retrieval_grade      DOUBLE PRECISION,        -- fact_checker(must_facts, CHUNKS)
    synthesis_grade      DOUBLE PRECISION,        -- fact_checker(must_facts, ANSWER)
    synthesis_gap        DOUBLE PRECISION,        -- synthesis − retrieval (neg=drop, pos=bluff)
    per_claim_ledger     JSONB,                   -- compact array (docstring spec)

    -- Mode + join keys ─────────────────────────────────────────────────
    is_prod              BOOLEAN NOT NULL DEFAULT false,
    eval_run_id          UUID REFERENCES public.rag_eval_runs(id) ON DELETE CASCADE,
    query_id             TEXT,                    -- eval rows → rag_eval_results(run_id, query_id)
    correlation_id       TEXT,                    -- prod rows → chat_turns (cross-DB, NO FK by design)

    -- Caller context ──────────────────────────────────────────────────
    caller               TEXT,
    caller_id            TEXT,

    CONSTRAINT rqd_prod_not_eval CHECK (NOT (is_prod AND eval_run_id IS NOT NULL))
);
"""

# Three purpose-built indexes + the eval partial — nothing else. The table is
# append-only; add an index the day a real query needs one, not before.
_CREATE_INDEXES_SQL = [
    # Bandit hot path: "mean reward per leaf at current corpus_version"
    # becomes an index-only scan.
    ("CREATE INDEX IF NOT EXISTS idx_rqd_leaf_corpus "
     "ON rag_query_decisions (leaf_key, corpus_version) "
     "INCLUDE (retrieval_grade);"),
    # Dashboard rollups per strategy over time windows.
    ("CREATE INDEX IF NOT EXISTS idx_rqd_strategy_ts "
     "ON rag_query_decisions (strategy_used, ts DESC);"),
    # Recent-rows scans.
    "CREATE INDEX IF NOT EXISTS idx_rqd_ts ON rag_query_decisions (ts DESC);",
    # Per-run eval reads.
    ("CREATE INDEX IF NOT EXISTS idx_rqd_eval_run_id "
     "ON rag_query_decisions (eval_run_id) "
     "WHERE eval_run_id IS NOT NULL;"),
]

_CREATE_CORPUS_STATE_SQL = """
CREATE TABLE IF NOT EXISTS public.corpus_state (
    singleton      BOOLEAN PRIMARY KEY DEFAULT true CHECK (singleton),
    corpus_version BIGINT NOT NULL DEFAULT 1,
    content_hint   TEXT,                          -- human-readable provenance (e.g. lexicon rev)
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


async def migrate():
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        exists = await conn.fetchval("""
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'rag_query_decisions'
        """)
        if exists:
            cols = {r["column_name"] for r in await conn.fetch(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name='rag_query_decisions'")}
            if _FINAL_COLUMNS <= cols:
                print("  rag_query_decisions already at final shape — skipping create")
            else:
                n = await conn.fetchval("SELECT count(*) FROM rag_query_decisions")
                if n:
                    raise SystemExit(
                        f"  REFUSING: pre-ratification rag_query_decisions has {n} rows; "
                        "write a hand-crafted ALTER + backfill instead of this migration.")
                await conn.execute("DROP TABLE rag_query_decisions;")
                await conn.execute(_CREATE_TABLE_SQL)
                print("  Recreated rag_query_decisions at final ratified shape (was empty draft)")
        else:
            await conn.execute(_CREATE_TABLE_SQL)
            print("  Created table public.rag_query_decisions")

        for sql in _CREATE_INDEXES_SQL:
            await conn.execute(sql)
        print(f"  Ensured {len(_CREATE_INDEXES_SQL)} indexes on rag_query_decisions")

        await conn.execute(_CREATE_CORPUS_STATE_SQL)
        await conn.execute(
            "INSERT INTO corpus_state (singleton) VALUES (true) "
            "ON CONFLICT DO NOTHING;")
        print("  Ensured corpus_state singleton (bump-at-mutation contract in docstring)")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_rag_query_decisions completed.")
