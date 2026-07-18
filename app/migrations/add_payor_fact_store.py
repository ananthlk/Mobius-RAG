"""Migration: schema ``facts`` — Payor Fact Store (payor_fact + fact_query_decision).

Spec: docs/payor-fact-store-spec.md (EVAL authored; payor builds logic/API; DB
agent owns this migration/placement/indexes; RAG reads). Persistence review
folded 2026-07-18: placement RESOLVED as shared mobius_rag DB, schema
``facts``, with SCHEMA-SCOPED SINGLE-WRITER ownership:

  - group role ``mobius_facts_rw``  (NOLOGIN): full DML on schema facts —
    payor's login user is GRANTed INTO this group at deploy time.
  - group role ``mobius_facts_ro``  (NOLOGIN): SELECT only — RAG's reader
    joins this group.
  - PUBLIC gets nothing. The schema boundary IS the ownership boundary.

Index philosophy (same as rag_query_decisions): only what the query path
touches today —
  - UNIQUE(payer_key, predicate) doubles as the serve-path btree.
  - GIN on j_tags, PARTIAL on cert_status='accepted' (the §2.1 gate only ever
    serves accepted rows).
  - HNSW on embedding (m=16, ef_construction=64 — org_docs/corpus standard),
    PARTIAL on accepted.
  - d_tags/p_tags GINs DEFERRED: §2.2 tag_overlap runs in-memory on the gated
    shortlist at Phase-1 scale. Add when a measured query needs them.

``vector(1536)`` is pinned deliberately: §7.2 requires vec_sim comparability
with the corpus embedder — enforce it in the type (answer-cache
output_dimensionality gotcha: a mismatched-dims embed writes fine and then
cosine is garbage).

fact_query_decision write contract (binding on payor's writer, mirrors
rag_query_decisions verbatim): telemetry_id CLIENT-generated uuid4 + INSERT
... ON CONFLICT DO NOTHING; fire-and-forget off the serve path; ONE shared
pool. The async outcome fill UPDATEs ONLY user_feedback / downstream_grade
(and §8's verify_outcome / drift_detected / verify_trigger from the
verify-and-recertify loop). Join keys to rag_query_decisions: correlation_id
(prod) / (eval_run_id, query_id) (eval) — same keys, joins work out of the
box; query_embedding is NULLABLE and written for eval/sampled rows only
(sweeps recompute from ``query`` text).

Idempotent — safe to re-run.
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from app.config import DATABASE_URL


_DDL = [
    "CREATE EXTENSION IF NOT EXISTS vector;",
    "CREATE SCHEMA IF NOT EXISTS facts;",

    # ── ownership roles (group roles; login users join at deploy time) ──
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

    # ── the fact record (spec §1) ──
    """
    CREATE TABLE IF NOT EXISTS facts.payor_fact (
        fact_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        -- STORE/SERVE key
        payer_key          TEXT NOT NULL,        -- canonical triple payer|state|program
        predicate          TEXT NOT NULL,        -- stable slug
        record_type        TEXT NOT NULL CHECK (record_type IN ('atomic','qa')),
        -- VALUE
        value              JSONB,
        answer_text        TEXT,
        question           TEXT,                 -- qa only
        -- QUERY SURFACE
        d_tags             TEXT[] NOT NULL DEFAULT '{}',
        p_tags             TEXT[] NOT NULL DEFAULT '{}',
        j_tags             TEXT[] NOT NULL DEFAULT '{}',  -- includes payer identity tag (the gate key)
        embedding          vector(1536),         -- MUST match corpus embedder dims
        -- SCOPE / COMPLIANCE (migration-005 semantics)
        scope              TEXT,                 -- NULL = unrestricted
        -- PROVENANCE
        source_ref         JSONB,                -- {doc_id, url, page, quote}
        authority_level    TEXT,                 -- contract_source_of_truth | payer_website | operational_suggested | ...
        -- FRESHNESS
        effective_date     DATE,
        valid_until        DATE,
        ttl_days           INTEGER,
        last_verified_at   TIMESTAMPTZ,
        verified_via       TEXT CHECK (verified_via IS NULL OR verified_via IN
                               ('rag_probe','web','human','eval_cert',
                                'explicit_verify','scheduled','bandit_verify')),  -- §8.5
        confidence         TEXT CHECK (confidence IS NULL OR confidence IN ('high','medium','low')),
        -- CERTIFICATION (EVAL owns)
        retrieval_grade    NUMERIC,
        synthesis_grade    NUMERIC,              -- atomic: NULL
        cert_status        TEXT NOT NULL DEFAULT 'candidate'
                           CHECK (cert_status IN ('accepted','candidate','rejected','stale')),
        cert_run_id        UUID,
        fact_checker_version TEXT,
        created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (payer_key, predicate)            -- doubles as the serve-path btree
    );
    """,

    # gate index: §2.1 only ever serves accepted rows
    """
    CREATE INDEX IF NOT EXISTS idx_fact_jtags_accepted
        ON facts.payor_fact USING gin (j_tags)
        WHERE cert_status = 'accepted';
    """,
    # vector shortlist, servable rows only
    """
    CREATE INDEX IF NOT EXISTS idx_fact_embedding_accepted
        ON facts.payor_fact USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        WHERE cert_status = 'accepted';
    """,

    # ── telemetry: one row per fact_query call (spec §3 + §8.5) ──
    """
    CREATE TABLE IF NOT EXISTS facts.fact_query_decision (
        telemetry_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
        query            TEXT NOT NULL,
        query_d_tags     TEXT[] NOT NULL DEFAULT '{}',
        query_p_tags     TEXT[] NOT NULL DEFAULT '{}',
        query_j_tags     TEXT[] NOT NULL DEFAULT '{}',
        query_embedding  vector(1536),          -- NULLABLE: eval/sampled rows only
        payer_key        TEXT,
        gate_applied     BOOLEAN NOT NULL DEFAULT false,
        gate_excluded_n  INTEGER,
        shortlist        JSONB,                 -- candidates + component scores, compact
        served_fact_id   UUID REFERENCES facts.payor_fact(fact_id) ON DELETE SET NULL,
        served_predicate TEXT,
        served_score     NUMERIC,
        hit              BOOLEAN NOT NULL DEFAULT false,
        fell_through     BOOLEAN NOT NULL DEFAULT false,
        alpha            NUMERIC,
        beta             NUMERIC,
        tau              NUMERIC,
        blend_version    TEXT,
        is_prod          BOOLEAN NOT NULL DEFAULT false,
        corpus_version   BIGINT,                -- from corpus_state; drift attribution on fall-throughs
        -- join keys to rag_query_decisions (same keys, joins work out of the box)
        correlation_id   TEXT,                  -- prod rows (chat turn; cross-DB, NO FK by design)
        eval_run_id      UUID REFERENCES public.rag_eval_runs(id) ON DELETE CASCADE,
        query_id         TEXT,                  -- eval rows, e.g. 'cmhc001'
        -- outcome (async fill — the ONLY UPDATEable columns, + §8 verify loop)
        user_feedback    TEXT,
        downstream_grade NUMERIC,
        verify_outcome   TEXT NOT NULL DEFAULT 'none'
                         CHECK (verify_outcome IN ('confirm','drift','none')),
        drift_detected   BOOLEAN NOT NULL DEFAULT false,
        verify_trigger   TEXT
    );
    """,

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
        for i, ddl in enumerate(_DDL, 1):
            await conn.execute(ddl)
        print(f"  Applied {len(_DDL)} DDL statements — schema facts, payor_fact, "
              "fact_query_decision, indexes, roles + grants")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_payor_fact_store completed.")
