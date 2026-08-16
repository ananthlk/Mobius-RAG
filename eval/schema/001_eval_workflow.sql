-- Eval-Workflow schema — RATIFIED by Database (Platform Architect / DB seat) 2026-08-12.
-- Source of truth: docs/rag-agents/eval-workflow-db-schema-proposal.md
-- Semantics owner: Eval-RAG. Schema/build lane: Eval-Architect.
--
-- Ratified decisions folded in:
--   1. Namespace: dedicated `eval` schema in the RAG DB (mobius-rag). Explicit isolation.
--   2. Applied as raw DDL via eval.db.execute() (no migration-file runner). Versioning is
--      SEMANTIC — a rule change bumps eval_population_rules.population_rules_version, pinned
--      to its ratification commit via spec_ref. This file is the canonical artifact.
--   3. eval_bank_run_rows indexes: (bank_run_id) + (bank_run_id, query_id, strategy) for the
--      fold. No partitioning until > 10M rows (deferred past Phase 1).
--   4. eval_published_priors is append-only / immutable audit (no UPDATE/DELETE).
--   5. Phase 0/1: priors_bootstrap.yaml stays authoritative; NO live-serving read from these
--      tables. This side is compute + audit only.
--
-- INTEGRITY SPINE: published -> computed -> bank_run -> valid_ruler FK chain. A published
-- value with no valid-ruler ancestor is physically unrepresentable.
--
-- NOT executed on write — deployment of this DDL is the irreversible step and holds for
-- Ananth's direct go (see eval/schema/apply_eval_workflow.py).

CREATE SCHEMA IF NOT EXISTS eval;

-- 1. eval_valid_rulers — the locked-ruler allowlist. Eval-RAG owns rows; adding a version = INSERT.
CREATE TABLE IF NOT EXISTS eval.eval_valid_rulers (
    ruler_id             SERIAL PRIMARY KEY,
    ruler                TEXT NOT NULL,              -- e.g. 'factcheck/gemini-2.5-pro'
    fact_checker_version TEXT NOT NULL,              -- e.g. 'fact_check_v1.2026-07-31'
    added_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    added_by             TEXT NOT NULL,
    UNIQUE (ruler, fact_checker_version)
);

-- 2. eval_population_rules — versioned, IMMUTABLE ruleset. Bumped only on population/formula
--    change (not wording). definition = machine-readable {field,population,formula} + meta_rule.
CREATE TABLE IF NOT EXISTS eval.eval_population_rules (
    population_rules_version INT PRIMARY KEY,
    definition          JSONB NOT NULL,
    spec_ref            TEXT NOT NULL,              -- spec §6b @ the git commit/tag at ratification
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 3. eval_bank_runs — one comprehensive sweep. ruler_id FK = structural locked-ruler enforcement.
CREATE TABLE IF NOT EXISTS eval.eval_bank_runs (
    bank_run_id         UUID PRIMARY KEY,
    ruler_id            INT NOT NULL REFERENCES eval.eval_valid_rulers(ruler_id),
    corpus_version      TEXT,
    query_set           TEXT NOT NULL,             -- which bank (e.g. 'cmhc_v1')
    status              TEXT NOT NULL,             -- 'running' | 'done' | 'failed'
    started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at         TIMESTAMPTZ
);

-- 4. eval_bank_run_rows — per-query observations (PriorsLabRow superset; the fold consumes these).
--    Pool features are first-class so re-bucketing is a QUERY, never a re-sweep.
CREATE TABLE IF NOT EXISTS eval.eval_bank_run_rows (
    row_id                BIGSERIAL PRIMARY KEY,
    bank_run_id           UUID NOT NULL REFERENCES eval.eval_bank_runs(bank_run_id),
    query_id              TEXT NOT NULL,
    caller_mode           TEXT NOT NULL,
    strategy              TEXT NOT NULL,
    -- grading outputs (recompute-a-cell needs ALL of these)
    recall                REAL,     -- chunk recall; NULL = grading gap (EXCLUDE), 0.0 = real zero (INCLUDE)
    recall_answer         REAL,     -- NULL on zero-output
    authority             REAL,     -- NULL on zero-output (nothing to judge)
    n_contradicted        INT,
    n_hallucinated_claims INT,
    n_facts_total         INT,
    -- pool features (enable feature-parameter bucketing as a query)
    top_score_percentile  REAL,
    distinct_content_topk INT,
    pool_size             INT,
    capacity              INT,      -- real fill_depth capacity
    fillers_ms            REAL,     -- real per-strategy latency
    UNIQUE (bank_run_id, query_id, caller_mode, strategy)
);
CREATE INDEX IF NOT EXISTS idx_eval_bank_run_rows_run
    ON eval.eval_bank_run_rows (bank_run_id);
CREATE INDEX IF NOT EXISTS idx_eval_bank_run_rows_fold
    ON eval.eval_bank_run_rows (bank_run_id, query_id, strategy);

-- 5. eval_computed_cells — a folded cell. cost_per_attempt intentionally absent (never derived).
CREATE TABLE IF NOT EXISTS eval.eval_computed_cells (
    cell_id             BIGSERIAL PRIMARY KEY,
    bank_run_id         UUID NOT NULL REFERENCES eval.eval_bank_runs(bank_run_id),
    depth_bucket        INT NOT NULL,
    strategy            TEXT NOT NULL,
    caller_mode         TEXT,     -- NULL when pooled across modes
    recall_lift         REAL,
    accuracy_estimate   REAL,
    authority           REAL,
    authority_measured_at_bucket INT,  -- authority hybrid provenance (may differ from depth_bucket)
    n                   INT NOT NULL,
    authority_n         INT,
    k0                  INT,
    latency_p50_ms      INT,
    reconciliation_ok   BOOLEAN NOT NULL,
    population_rules_version INT NOT NULL REFERENCES eval.eval_population_rules(population_rules_version),
    cell_sha256         TEXT NOT NULL,   -- canonical hash of the 4dp appliable fields
    warnings            JSONB,
    computed_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (bank_run_id, depth_bucket, strategy, caller_mode)
);

-- 6. eval_published_priors — APPEND-ONLY audit of what landed in priors_bootstrap.yaml.
--    FK chain published -> computed -> bank_run -> valid_ruler is the integrity spine.
--    Immutability (no UPDATE/DELETE) is enforced by convention + the guard trigger below.
CREATE TABLE IF NOT EXISTS eval.eval_published_priors (
    publish_id          BIGSERIAL PRIMARY KEY,
    depth_bucket        INT NOT NULL,
    strategy            TEXT NOT NULL,
    cell_id             BIGINT NOT NULL REFERENCES eval.eval_computed_cells(cell_id),
    cell_sha256         TEXT NOT NULL,           -- copied for tamper-evidence
    -- denormalized provenance (audit reads join-free; survives bank_run archival)
    ruler               TEXT NOT NULL,
    fact_checker_version TEXT NOT NULL,
    population_rules_version INT NOT NULL,
    n                   INT NOT NULL,
    published_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    published_by        TEXT NOT NULL
);

-- Append-only guard: block UPDATE/DELETE on the published audit so a live value can never be
-- silently rewritten (the ratified immutability contract, made structural rather than trusted).
CREATE OR REPLACE FUNCTION eval.reject_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'eval.eval_published_priors is append-only (attempted %)', TG_OP;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_published_priors_immutable ON eval.eval_published_priors;
CREATE TRIGGER trg_published_priors_immutable
    BEFORE UPDATE OR DELETE ON eval.eval_published_priors
    FOR EACH ROW EXECUTE FUNCTION eval.reject_mutation();
