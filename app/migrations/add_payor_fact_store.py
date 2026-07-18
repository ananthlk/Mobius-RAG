"""Migration: schema ``facts`` — Payor Fact Store (payor_fact + fact_query_decision).

CANONICAL RESHAPE (2026-07-18). Payor's migrations 006+007 created a first
cut of this schema on dev before the persistence review landed (19 facts +
telemetry accumulating), so this migration must work BOTH ways:
  - fresh environment → creates the full canonical shape;
  - live dev         → reshapes in place, preserving all rows.
From this migration on, the DB agent owns facts.* DDL; payor owns the rows
(agreed with payor agent 2026-07-18 — their 006/007 stay as history, no new
payor-side DDL).

Spec: docs/payor-fact-store-spec.md (EVAL authored; payor builds logic/API;
DB agent owns migration/placement/indexes; RAG reads). Persistence review
folded: placement RESOLVED as shared mobius_rag DB, schema ``facts``, with
SCHEMA-SCOPED SINGLE-WRITER ownership:
  - group role ``mobius_facts_rw`` (NOLOGIN): full DML — payor's login user
    is GRANTed INTO this group at deploy time.
  - group role ``mobius_facts_ro`` (NOLOGIN): SELECT — RAG's reader joins.
  - PUBLIC gets nothing. The schema boundary IS the ownership boundary.

Index philosophy (same as rag_query_decisions): only what the query path
touches — UNIQUE(payer_key,predicate) doubles as the serve btree; GIN(j_tags)
and HNSW(embedding, m=16/ef_construction=64) both PARTIAL on
cert_status='accepted' (§2.1's gate only serves accepted rows; payor's
candidate-visibility scan tolerates a seq scan at current scale — revisit if
candidates grow); d/p-tag GINs DEFERRED (§2.2 tag_overlap runs in-memory on
the gated shortlist).

``vector(1536)`` pinned: §7.2 requires vec_sim comparability with the corpus
embedder — enforced in the type (answer-cache output_dimensionality gotcha).

VOCABULARY NOTES (live data admitted, pending spec ratification by EVAL):
  - verified_via includes 'browser' (payor's writer emits it; spec has
    rag_probe|web|human|eval_cert + §8.5's explicit_verify|scheduled|
    bandit_verify). Either the spec adopts 'browser' or payor maps it to
    'web' — until ruled, the CHECK admits it.
  - verify_outcome includes 'pending_compare' (payor's in-flight state while
    EVAL's async two_grade_compare runs) alongside §8.5's confirm|drift|none.
    Semantically useful — recommend the spec adopt it.

fact_query_decision write contract (binding, mirrors rag_query_decisions):
telemetry_id CLIENT-generated uuid4 + ON CONFLICT DO NOTHING; fire-and-forget
off the serve path; ONE shared pool. Async fills UPDATE ONLY user_feedback /
downstream_grade / verify_outcome / drift_detected / verify_trigger. Join
keys to rag_query_decisions: correlation_id (prod, cross-DB no FK) /
(eval_run_id, query_id) (eval, real FK to rag_eval_runs CASCADE).
query_embedding NULLABLE — eval/sampled rows only; sweeps recompute from
``query`` text.

Idempotent — safe to re-run.
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from app.config import DATABASE_URL


_VERIFIED_VIA_CHECK = ("('rag_probe','web','browser','human','eval_cert',"
                       "'explicit_verify','scheduled','bandit_verify')")
_VERIFY_OUTCOME_CHECK = "('confirm','drift','none','pending_compare','live_unavailable')"

_DDL = [
    "CREATE EXTENSION IF NOT EXISTS vector;",
    "CREATE SCHEMA IF NOT EXISTS facts;",

    # ── ownership group roles ──
    """
    DO $$ BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='mobius_facts_rw') THEN
            CREATE ROLE mobius_facts_rw NOLOGIN;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='mobius_facts_ro') THEN
            CREATE ROLE mobius_facts_ro NOLOGIN;
        END IF;
    END $$;
    """,

    # ── fresh-environment create (no-ops on live dev) ──
    f"""
    CREATE TABLE IF NOT EXISTS facts.payor_fact (
        fact_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        payer_key          TEXT NOT NULL,
        predicate          TEXT NOT NULL,
        record_type        TEXT NOT NULL CHECK (record_type IN ('atomic','qa')),
        value              JSONB,
        answer_text        TEXT,
        question           TEXT,
        d_tags             TEXT[] NOT NULL DEFAULT '{{}}',
        p_tags             TEXT[] NOT NULL DEFAULT '{{}}',
        j_tags             TEXT[] NOT NULL DEFAULT '{{}}',
        embedding          vector(1536),
        scope              TEXT,
        source_ref         JSONB,
        authority_level    TEXT,
        effective_date     DATE,
        valid_until        DATE,
        ttl_days           INTEGER,
        last_verified_at   TIMESTAMPTZ,
        verified_via       TEXT,
        confidence         TEXT,
        retrieval_grade    NUMERIC,
        synthesis_grade    NUMERIC,
        cert_status        TEXT NOT NULL DEFAULT 'candidate'
                           CHECK (cert_status IN ('accepted','candidate','rejected','stale')),
        cert_run_id        UUID,
        fact_checker_version TEXT,
        created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (payer_key, predicate)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS facts.fact_query_decision (
        telemetry_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
        query            TEXT NOT NULL,
        query_d_tags     TEXT[] NOT NULL DEFAULT '{}',
        query_p_tags     TEXT[] NOT NULL DEFAULT '{}',
        query_j_tags     TEXT[] NOT NULL DEFAULT '{}',
        query_embedding  vector(1536),
        payer_key        TEXT,
        gate_applied     BOOLEAN NOT NULL DEFAULT false,
        gate_excluded_n  INTEGER,
        shortlist        JSONB,
        served_fact_id   UUID,
        served_predicate TEXT,
        served_score     NUMERIC,
        hit              BOOLEAN NOT NULL DEFAULT false,
        fell_through     BOOLEAN NOT NULL DEFAULT false,
        alpha            NUMERIC,
        beta             NUMERIC,
        tau              NUMERIC,
        blend_version    TEXT,
        is_prod          BOOLEAN NOT NULL DEFAULT false,
        corpus_version   BIGINT,
        correlation_id   TEXT,
        eval_run_id      UUID,
        query_id         TEXT,
        user_feedback    TEXT,
        downstream_grade NUMERIC,
        verify_outcome   TEXT NOT NULL DEFAULT 'none',
        drift_detected   BOOLEAN NOT NULL DEFAULT false,
        verify_trigger   TEXT
    );
    """,

    # ── reshape: columns the first cut lacked ──
    "ALTER TABLE facts.fact_query_decision ADD COLUMN IF NOT EXISTS corpus_version BIGINT;",

    # verify_outcome: coalesce legacy NULLs, then enforce
    "UPDATE facts.fact_query_decision SET verify_outcome='none' WHERE verify_outcome IS NULL;",
    "ALTER TABLE facts.fact_query_decision ALTER COLUMN verify_outcome SET DEFAULT 'none';",
    "ALTER TABLE facts.fact_query_decision ALTER COLUMN verify_outcome SET NOT NULL;",

    # ── reshape: CHECK constraints (guarded; live data verified compatible) ──
    f"""
    DO $$ BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='payor_fact_verified_via_check') THEN
            ALTER TABLE facts.payor_fact ADD CONSTRAINT payor_fact_verified_via_check
                CHECK (verified_via IS NULL OR verified_via IN {_VERIFIED_VIA_CHECK});
        END IF;
        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='payor_fact_confidence_check') THEN
            ALTER TABLE facts.payor_fact ADD CONSTRAINT payor_fact_confidence_check
                CHECK (confidence IS NULL OR confidence IN ('high','medium','low'));
        END IF;
    END $$;
    """,

    # verify_outcome vocabulary evolves (pending_compare, live_unavailable, ...);
    # recreate the CHECK every run so the canonical list here is always live.
    "ALTER TABLE facts.fact_query_decision DROP CONSTRAINT IF EXISTS fqd_verify_outcome_check;",
    f"""
    ALTER TABLE facts.fact_query_decision ADD CONSTRAINT fqd_verify_outcome_check
        CHECK (verify_outcome IN {_VERIFY_OUTCOME_CHECK});
    """,

    # ── reshape: foreign keys (guarded; orphan-checked before adding) ──
    """
    DO $$ BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='fqd_served_fact_fk') THEN
            ALTER TABLE facts.fact_query_decision ADD CONSTRAINT fqd_served_fact_fk
                FOREIGN KEY (served_fact_id) REFERENCES facts.payor_fact(fact_id) ON DELETE SET NULL;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='fqd_eval_run_fk') THEN
            ALTER TABLE facts.fact_query_decision ADD CONSTRAINT fqd_eval_run_fk
                FOREIGN KEY (eval_run_id) REFERENCES public.rag_eval_runs(id) ON DELETE CASCADE;
        END IF;
    END $$;
    """,

    # ── reshape: index swap to the canonical lean set ──
    "DROP INDEX IF EXISTS facts.ix_pf_dtags;",       # deferred until measured need
    "DROP INDEX IF EXISTS facts.ix_pf_ptags;",
    "DROP INDEX IF EXISTS facts.ix_pf_jtags;",       # replaced by partial
    "DROP INDEX IF EXISTS facts.ix_pf_embedding;",   # replaced by partial
    "DROP INDEX IF EXISTS facts.ix_pf_accepted;",    # redundant with UNIQUE(payer_key,predicate) prefix
    """
    CREATE INDEX IF NOT EXISTS idx_fact_jtags_accepted
        ON facts.payor_fact USING gin (j_tags)
        WHERE cert_status = 'accepted';
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_fact_embedding_accepted
        ON facts.payor_fact USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        WHERE cert_status = 'accepted';
    """,
    "DROP INDEX IF EXISTS facts.ix_fqd_created;",
    "DROP INDEX IF EXISTS facts.ix_fqd_payer;",
    "CREATE INDEX IF NOT EXISTS idx_fqd_created ON facts.fact_query_decision (created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_fqd_payer_created ON facts.fact_query_decision (payer_key, created_at DESC);",
    """
    CREATE INDEX IF NOT EXISTS idx_fqd_served_fact
        ON facts.fact_query_decision (served_fact_id)
        WHERE served_fact_id IS NOT NULL;
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_fqd_eval_run
        ON facts.fact_query_decision (eval_run_id)
        WHERE eval_run_id IS NOT NULL;
    """,

    # ── grants: schema-scoped single writer ──
    "REVOKE ALL ON SCHEMA facts FROM PUBLIC;",
    "GRANT USAGE ON SCHEMA facts TO mobius_facts_rw, mobius_facts_ro;",
    "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA facts TO mobius_facts_rw;",
    "GRANT SELECT ON ALL TABLES IN SCHEMA facts TO mobius_facts_ro;",
    "ALTER DEFAULT PRIVILEGES IN SCHEMA facts GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO mobius_facts_rw;",
    "ALTER DEFAULT PRIVILEGES IN SCHEMA facts GRANT SELECT ON TABLES TO mobius_facts_ro;",
]


async def migrate():
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        for ddl in _DDL:
            await conn.execute(ddl)
        print(f"  Applied {len(_DDL)} DDL statements — facts schema at canonical shape "
              "(create-or-reshape), roles + grants ensured")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_payor_fact_store completed.")
