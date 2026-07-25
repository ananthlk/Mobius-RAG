# Answer-engine refactor — frozen response contract

Owner: EVAL. Purpose: the refactor (shape-fill-gate loop) rewrites `corpus_search_agent`'s INTERNALS.
This is the output surface it must keep byte-compatible so no consumer breaks and the feature flag
(`RAG_ANSWER_ENGINE=legacy|shape`) is invisible downstream. If the new loop repopulates every field
below with equivalent semantics, flipping the flag is a no-op for every caller.

## The contract = `CorpusSearchAgentResponse` (app/services/corpus_search_agent.py:2789) — 12 fields

| field | must the new loop populate? | who depends on it |
|---|---|---|
| `chunks: list[CorpusChunk]` | YES | chat (answer), eval (grading), diagnostics (evidence) |
| `confidence: str` | YES | chat (≥3 reads), eval |
| `query_profile: dict` | YES | chat, diagnostics REASON leaf |
| `strategy_used: str` | YES — **heaviest chat dep (7 reads)** | chat route label, diagnostics ACT, decision row |
| `routing: dict` | YES — **bandit-critical** | decision-row writer reads `routing.{priors_version, feature_vector\|features, leaf_key parts, invoke_all, scores}`; chat; diagnostics |
| `strategies_tried: list` | YES | chat, diagnostics ACT (one sub-tree per strategy) |
| `improvement_hint: dict\|None` | YES — **becomes structured residual-gaps** | chat reframing loop |
| `telemetry: dict` | YES | chat, diagnostics per-stage timings |
| `gate: dict\|None` | YES | chat, diagnostics REASON gate leaf |
| `fail_fast: dict\|None` | YES | chat |
| `term_partition: dict` | keep (may be empty) | chat, diagnostics |
| `candidate_pool: dict` | keep (may be empty) | chat, diagnostics |

Plus `llm_answer` (served text) — chat + eval read it; must stay.

## Reading of the audit
- **The contract is WIDE and chat reads nearly all of it.** So the new loop cannot "return a cleaner shape" — it
  must project its slot-filled result back onto these exact 12 fields. Mapping:
  - slots' merged evidence → `chunks`; served text → `llm_answer`; overall fill confidence → `confidence`.
  - per-slot strategy choices → `strategies_tried` + `strategy_used` (the dominant/last filler, or the chain).
  - unfilled slots → `improvement_hint` (upgraded to structured slot-gaps — additive, back-compatible).
  - routing decision (now per-slot) → `routing` MUST still carry `priors_version`, `feature_vector`, `leaf_key`,
    `scores` or the **decision-row + bandit break silently** (this is the sharpest freeze constraint).
- **Bandit/decision-row is the hardest constraint**: `routing` feeds the training row. If per-slot routing
  changes the leaf_key/feature_vector semantics, that's a telemetry migration, not a transparent swap — flag it
  as a KNOWN contract change requiring an eval-owned decision, don't let it drift.
- Diagnostics content-tree (EVAL-owned, docs/diagnostics-card-content-tree.md) already maps to these fields; it
  stays valid as long as the fields stay populated. The ACT "one sub-tree per strategy" rule maps cleanly to
  per-slot fillers.

## Reversibility layers (recap)
1. `RAG_ANSWER_ENGINE=legacy|shape` flag, default legacy — instant flip-back, no deploy.
2. Contract frozen (this doc) — flag flip invisible to consumers.
3. Git tag `answer-engine/baseline-v0` @ cmhc `run_id` — hard fallback.
   **Checkpoint SHAs (clean, verified by Broadcaster 2026-07-20):** mobius-rag `2b46980` · mobius-chat `270ff57`
   (both on main, pushed, reachable). SHAs are immutable → no rush to tag; will tag both @ these SHAs annotated
   with the cmhc baseline `run_id` once it lands.

## STEP-1 BASELINE (pinned 2026-07-21 — the working reference for the 5-step plan)
8 nightly calibration runs, rev `00459-xzm`, cmhc bank (8e743568), judge gemini-2.5-pro, corpus AS-IS
(424 embedded-unpublished + orphans accepted per Ananth: "work with the corpus, those things should not matter").
`s`-contamination INCLUDED (8/22 queries hijacked by fact-store in every forced cell + natural, all judged wrong).

| metric | mean ± σ (n=8) | range |
|---|---|---|
| oracle_recall | **0.473 ± 0.017** | 0.451–0.508 |
| router_recall | **0.345 ± 0.007** | 0.330–0.353 |
| best_single | 0.382 ± 0.023 | 0.348–0.410 |
| a / b / c / d | 0.311 / 0.260 / 0.144 / 0.382 | c noisy (±0.055) |

Deltas ≥ ~0.02 on router are signal (2.8σ). Plan: step-2 forced-bypass (a/b/c/d/s measured clean + union-oracle) →
step-3 routing/s fix (router moves here) → step-4 fast-exit/clarify/reframe framework → step-5 re-baseline. Target router 0.65
(requires union-oracle ceiling ≥~0.72 — validate at step 2).

## Routing contract decision — RESOLVED: Option (a), EVAL-SIGNED 2026-07-22
Per-slot vs per-query routing in the `routing` dict + decision row. **DECISION: Option (a).** Keep the
per-query `routing` shape EXACTLY as-is (priors_version, feature_vector|features, leaf_key, scores, strategy);
the bandit INSERT + both decision-row writers read ONLY these, unchanged. Per-slot detail goes in a NEW
additive sub-field that no existing reader consumes (diagnostic-only). Hard constraints for the `router` module:
- `leaf_key` stays PER-QUERY (current shape) — a per-slot leaf_key is a SEPARATE future telemetry migration
  needing a fresh EVAL sign-off (that was option (b), NOT taken).
- feature_vector + scores + priors_version non-null on every non-s response; s-rows keep NULL by design.
- bandit reward + context derive SOLELY from the existing per-query keys → training row byte-identical PRE/POST in STRUCTURE.
- **EDGE VERIFIED 2026-07-22:** Router input changes in shape-first sequencing (raw_query → filled_shape context); feature_vector VALUES will differ (built from rewritten_queries, not raw query). This is NOT a telemetry migration — it's an intentional semantic input change. Structure + writer + importer locked; only context VALUES shift per flag state (legacy vs shape mode). Bandit learns consistently from whichever mode is active.
Enforced by the 3 machine-checks (one INSERT `rag_query_decisions`, one `check_facts` import, `FACT_CHECKER_VERSION`
per row). Rationale: back-compatible structure, no migration, bandit input remains valid. Option (b) (migrate bandit to per-slot
context) explicitly deferred — it breaks the current training row and would need a fresh sign-off.
