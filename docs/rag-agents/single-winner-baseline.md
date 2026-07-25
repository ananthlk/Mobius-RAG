# Single-Winner Baseline — Eval ruling (2026-07-24)

**Ruling (Eval owns the ruler/baselines).** The just-completed 22-query
observer calibration run is **kept and labeled `single-winner-baseline`**, per
Router's objection — NOT discarded. This requires **no further grading** (the
run is already graded), so it fully honors Retriever's HOLD. The A-vs-B
(stopgap vs Observer) question is retired; only the single-winner aggregate is
preserved, as the reference the blend model's lift will be claimed against.

## The baseline numbers (chunk-only rubric, real LLM-judge, 22q)

| | reported | **clean (excl. 4 rate-limited rows)** |
|---|---|---|
| arm 0 (single-rung) | 0.030 | 0.030 |
| **arm A (stopgap single-winner) — THE baseline floor** | 0.178 | **0.146** |
| arm B (Observer single-winner) — retired variant | 0.148 | 0.113 |
| **5-leg forced ORACLE (best single leg/query) — the bar** | — | **0.55** |

- Use **arm A clean = 0.146** as the single-winner integrated **floor**, NOT
  0.178. Four judge calls hit Vertex 429 and returned a spurious
  `unable_to_verify`/0.5: cmhc005, cmhc022 (arm A); cmhc006, cmhc019 (arm B).
  Excluded from the clean mean; re-grade them if the exact row is needed.
- The meaningful bar for the blend is the **single-winner oracle 0.55**, not the
  floor. The blend's entire thesis is to exceed what any perfect single-winner
  router could do by combining legs. A blend that doesn't clear 0.55 buys
  nothing a perfect single-strategy picker couldn't.

## Provenance

Post payload-token-constants fix, post SUPPLEMENT_ONLY gate, post cap-6.
Legs graded forced/chunk-only: d=0.42, b=0.20, a=0.14, c=0.07, s=0.00.

## Two caveats attached to any lift claim (measure-the-bug-not-the-change)

1. **Early-stop confound.** This single-winner baseline includes the logged
   loop defect: the continuation loop stops after leg-a returns chunks and
   never escalates to d (the strongest leg). So blend-vs-baseline lift will
   **conflate** (a) blending gains with (b) fixing the early stop.
   *Recommendation:* either fix early-stop and re-baseline single-winner, or
   attribute the two changes separately. Do not credit the blend for the
   escalation fix.
2. **Grading parity.** Baseline is synthesis-OFF (`llm_answer=""`). The blend
   MUST be graded the identical way, or both re-graded with `--synthesize`.
   Never compare blend-with-synthesis against baseline-without.

## Methodology hand-off (mine, per Router)

q-calibration and LB-propagation-through-the-capacity-transform land in Eval's
court in the unified design doc. Priors shift from P(strategy answers alone) →
each strategy's marginal contribution to a reranked set; reward from "winning
strategy answered" → "strategy's chunks survived rerank + cited." Single-winner
priors are NOT reusable as-is for the blend.
