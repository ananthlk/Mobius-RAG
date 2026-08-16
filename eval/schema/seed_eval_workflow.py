"""Seed the two hand-authored eval-workflow reference rows (tables 1-2).

Everything downstream (bank_runs, rows, computed_cells, published_priors) is
pipeline-derived; these two rows are the ONLY authored inputs:

  row 1  eval_valid_rulers      the locked-ruler allowlist  (§6a, Eval-RAG-owned)
  row 2  eval_population_rules   population_rules_version=1  (§6b, Eval-RAG-owned)

Both seeds CONFIRMED by Eval-RAG 2026-08-12; spec_ref pinned to commit e15187d
(docs/rag-agents/eval-workflow-tooling-spec.md, §6a+§6b as written).

Idempotent (ON CONFLICT DO NOTHING) — safe to re-run.

    python -m eval.schema.seed_eval_workflow          # seed
    python -m eval.schema.seed_eval_workflow --check   # show current rows, no writes
"""
from __future__ import annotations

import asyncio
import json
import sys

from eval.db import close_pool, execute, fetchrow

_RULER = "factcheck/gemini-2.5-pro"
_FACT_CHECKER_VERSION = "fact_check_v1.2026-07-31"
_SPEC_REF = "e15187d"

# population_rules_version=1 definition (machine-readable only; "why" lives in spec §6b).
# recall_lift and accuracy_estimate share ONE population instance (meta_rule recall-None
# exclude defines it) so recall_lift*accuracy_estimate == mean(answer_recall) holds by
# construction. authority uses a NARROWER content-bearing population -> its own authority_n,
# NOT part of the reconciliation product.
_POP_RULES_V1 = {
    "rules": [
        {"field": "recall_lift", "source": "recall", "population": "all_attempts",
         "failures": "include_as_zero", "formula": "mean(recall)"},
        {"field": "accuracy_estimate", "source": "answer_recall/recall",
         "population": "same_instance_as:recall_lift",
         "failures": "answer_recall None->0 over the SAME population",
         "formula": "clamp01(mean(answer_recall)/mean(recall))"},
        {"field": "authority", "source": "authority", "population": "content_bearing_attempts",
         "failures": "exclude_zero_output", "formula": "mean(authority)", "companion_n": "authority_n"},
        {"field": "k0", "source": "real_capacity", "population": "all_attempts",
         "formula": "round(n_weighted_mean(real_capacity))"},
        {"field": "latency_p50_ms", "source": "fillers_ms", "population": "all_attempts",
         "formula": "round(median(fillers_ms))"},
    ],
    "meta_rule": {
        "recall_none": "grading-gap -> EXCLUDE attempt from ALL populations",
        "recall_zero": "real-zero -> INCLUDE (counts as 0 in multiplied fields)",
    },
}


async def _seed() -> None:
    await execute(
        "INSERT INTO eval.eval_valid_rulers (ruler, fact_checker_version, added_by) "
        "VALUES ($1, $2, $3) ON CONFLICT (ruler, fact_checker_version) DO NOTHING",
        _RULER, _FACT_CHECKER_VERSION, "eval-rag",
    )
    await execute(
        "INSERT INTO eval.eval_population_rules (population_rules_version, definition, spec_ref) "
        "VALUES ($1, $2::jsonb, $3) ON CONFLICT (population_rules_version) DO NOTHING",
        1, json.dumps(_POP_RULES_V1), _SPEC_REF,
    )
    print("seeded (idempotent)")
    await _check()


async def _check() -> None:
    r = await fetchrow(
        "SELECT ruler_id, ruler, fact_checker_version FROM eval.eval_valid_rulers "
        "WHERE ruler=$1 AND fact_checker_version=$2", _RULER, _FACT_CHECKER_VERSION)
    print(f"  valid_ruler:     {dict(r) if r else 'MISSING'}")
    p = await fetchrow(
        "SELECT population_rules_version, spec_ref, jsonb_array_length(definition->'rules') AS n_rules "
        "FROM eval.eval_population_rules WHERE population_rules_version=1")
    print(f"  population_rules: {dict(p) if p else 'MISSING'}")
    if not (r and p):
        raise SystemExit("seed incomplete")


async def _main() -> None:
    try:
        await (_check() if "--check" in sys.argv else _seed())
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())
