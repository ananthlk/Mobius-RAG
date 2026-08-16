-- Eval-Workflow schema v1.1 amendment — ONE additive column.
-- Proposed to Database (DB seat) 2026-08-12; ALTER holds for Ananth's direct go.
--
-- Adds the reconciliation DELTA (the number) alongside the existing reconciliation_ok
-- (bool). Eval-RAG's §7 gate HARD-BLOCKS on the bool (exists since v1); the delta is
-- DISPLAY-ENRICHMENT — it shows the reviewer how close/far rl*ae is from
-- mean(answer_recall), but gates nothing. Written at fold time:
--     reconciliation_delta = abs(recall_lift * accuracy_estimate - mean_answer_recall)
-- Cannot be recovered post-fold (mean_answer_recall isn't otherwise stored), hence a
-- column rather than a derived value.
--
-- NO superseded_by: recompute = a NEW bank_run over the same immutable rows (Eval-RAG
-- ruled option (a) 2026-08-12) — cells stay insert-only, no in-run supersede machinery.
--
-- Nullable additive column: non-destructive, no table rewrite, reversible (DROP COLUMN).
-- Idempotent. NOT executed on write.

ALTER TABLE eval.eval_computed_cells
    ADD COLUMN IF NOT EXISTS reconciliation_delta REAL;
