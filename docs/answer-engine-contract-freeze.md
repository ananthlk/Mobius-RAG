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

## Open contract decision (needs EVAL sign-off before build)
Per-slot routing vs per-query routing in the `routing` dict + decision row. Options: (a) keep per-query
`routing` shape, log per-slot detail in a new additive sub-field (back-compatible, bandit unchanged); (b)
migrate the bandit to per-slot context (better long-term, breaks the current training row). Lean (a) first.
