# RAG Retrieval-Strategy Calibration — Baseline & Methodology

**Purpose.** Measure the quality/cost/reliability of the four corpus_search
strategies (a/b/c/d) so we can (1) recalibrate the router and (2) track the
*lift* from lexicon + corpus changes over time. This is the "before" the
lexicon/router work is measured against. Established 2026-06-30 (dev).

---

## 0. Adjudicator LOCK + on-demand button (2026-06-30)

- **The ruler is locked.** `rag_eval_adjudicate` (the fact_checker's judge) is pinned to a
  SINGLE model — **gemini-2.5-pro** — via `mobius-chat/app/services/model_registry.py`
  (it's the only model with that stage in `eligible_stages`; removed from
  `CORE_REASONING_STAGES`). A bandit-routed ruler would make baseline-vs-change deltas
  reflect *judge* variance, not real lift. Verify: only gemini-2.5-pro eligible for the stage.
- **On-demand:** the eval engine's **"🎯 Run Calibration"** button now forces a/b/c/d
  `skip_synthesis` and scores with the **`fact_checker`** (chunk-only recall + contradiction),
  storing the 5-axis metrics per cell in `full_response._calibration`. `GET
  /api/eval/runs/{id}/calibration_summary` aggregates → the frontend **Calibration panel**
  (5-axis + oracle). Files: `eval/calibrate.py`, `app/routers/eval.py`, `frontend/.../EvalTab.tsx`.
- **⚠️ RE-BASELINE REQUIRED:** the numbers in §3 below were measured with the UNLOCKED
  (pro/flash mix) judge. Re-run the baseline via the button once the lock is live — that
  locked run is the authoritative reference for all lift/drift comparisons.

## 1. The measurement rig (reusable)

- **Critic:** `app/services/fact_checker.py` — LLM grounding critic. Two modes:
  - *chunk-only* (`answer=None`): are the golden facts **present in the retrieved
    chunks**? Graded 0/0.5/1.0 per fact, plus a **contradicted** flag (a chunk
    asserting a *wrong* fact — retrieval error, distinct from a miss). This is the
    **retrieval** measure — answer-independent, so it doesn't penalize a strategy
    for a synthesizer that failed to use a chunk it retrieved.
  - *grounding* (`answer=...`): +grounded facts, −hallucination, **full credit for
    honest abstention** — the turn-quality / runtime-critic measure.
  - Runs on the registered `rag_eval_adjudicate` stage. `max_tokens=4096` (2048
    truncates gemini-2.5-flash → all-zero; same failure class as the eval judge).
- **Forced calibration:** force each strategy via request `mode=a|b|c|d`
  (the agent's `_is_override` now honours it verbatim — a prior bug let a forced
  arm *withdraw/cascade* to another; fixed in `corpus_search_agent.py`).
  `skip_synthesis=True` for pure-retrieval runs.
- **Matrices (this dir):** `clean_recall_matrix_20260630.json` (locked means),
  `factcheck_matrix_*` (grounding/answer-behaviour), `strategy_matrix_*v2`
  (with-synthesis + latency). Re-run these to measure lift.

## 2. The 5 axes (+ σ)

A strategy is characterised by, not one score, but:
1. **answer-rate** — will it answer vs abstain? (should *track* recall — answer iff retrieved)
2. **recall** — golden facts present in retrieved chunks
3. **accuracy** — of stated facts, how many are grounded (precision)
4. **latency** — retrieval-only wall-clock (LLM off)
5. **cost** — LLM token $ per query
6. **σ (reliability)** — variance across repeated runs (a/b deterministic ≈0; c/d non-deterministic, high)

## 3. AUTHORITATIVE LOCKED BASELINE (dev, run 4c13c9da, 88 cells, locked gemini-2.5-pro ruler, chunk-only)

| strat | answer-rate | recall | precision | contra/cell | median lat | p95 lat |
|-------|-------------|--------|-----------|-------------|------------|---------|
| **a** BM25   | 0.86 | 0.405 | 0.36 | 0.00 | **1.1s** | 24s | ← decisive default: fast, zero contra, tied-best recall |
| **b** vector | 0.55 | 0.307 | 0.34 | 0.00 | 12.2s | 28s |
| **c** LLM    | 0.52 | 0.092 | 0.40 | 0.00 | 19.0s | 29s | ← barely retrieves; deprecate |
| **d** web    | 0.86 | 0.408 | 0.31 | **0.19** | 18.9s | 22s | ← recall parity w/ a BUT slow + only strategy with contradictions (web noise) |

**Oracle recall 0.598 · best single 0.408 · routing headroom 0.19.** Precision = relevant(cited)/retrieved,
capped ~0.6 (proper @k/MRR is a follow-up). Ranking a≈d > b >> c. Only d contradicts (misleading web chunks).
This is the fixed floor for lexicon lift. Ruler locked → comparable across runs.

### (superseded) unlocked draft numbers

| strat | answer_rate | recall | accuracy | contra/cell | role |
|-------|-------------|--------|----------|-------------|------|
| **a** BM25 cascade      | 0.60 | 0.44 | 0.35 | 0.05 | precision/exact; deterministic; reliable **default** |
| **b** wide→themes→narrow| 0.75 | 0.37 | 0.35 | 0.00 | topical/conceptual |
| **c** LLM→validate      | 0.90 | 0.18 | 0.07 | 0.14 | **hallucination engine** — always answers, ~0 retrieval (≈1 chunk); deprecate for grounded use |
| **d** external/web      | 0.57 | 0.51 | 0.65 | 0.05 | highest mean BUT web/**non-deterministic** (σ untrusted: 0.31 one run, 0.65 next); slow+costly → targeted fallback only |

- **Answer-rate must track recall.** a/b calibrated (abstain on thin retrieval → honest); c/d decoupled (answer regardless) → c's answer-rate×low-recall = hallucination.
- **Per-class recall:** wide_pool → a (0.67); tight_pool → d (0.52)/a (0.42).
- **Latency (retrieval-only, `skip_synthesis`, dev):** b ~1–5s < a ~3s (**heavy tail: 30s+ on hard queries — BM25 cascade depth**) < d ~8s (web) < c ~17s (LLM generation — skip_synthesis can't strip it). c is worst on EVERY axis (recall/accuracy/latency/cost/σ) → no query where c is the right pick. a's latency tail concentrates on the unstable queries (cmhc002/008/010). Robust median/p95 = a lighter a/b-all-queries + c/d-few-samples pass; refine during daily-monitor build.
- **Cost:** _TODO — pull token $ per strategy from llm_calls (rag_strategy_* stages)._

## 4. Oracle path (router quality)

- Per-query oracle recall (always pick best arm): **0.636**
- Natural router today: **0.435** → **routing gap 0.201**
- Router picks the oracle-best arm only **7/22 (32%)**; over-defaults to **a (~9–13/22)**.
- **Current router operating latency (natural, retrieval-only):** median **5.5s**, p95 **35s**, max 35s (heavy tail from a-cascade depth / slow c-d picks). Fail-fast (e) ~0.25s. Data: `router_latency_20260630.json`.
- **Oracle vs router is a quality/latency tradeoff too:** the max-recall oracle picks d/c more, which are the slow arms — so "best recall" routing is NOT "best recall-per-second." The router's a-preference is partly a *latency* hedge.

**Router logic = class prior + per-query self-assessment.** Each query is offered
to all 4 arms; each estimates its own recall (`_estimate_internal_recall`) and
raises confidence or **withdraws**. Composite ≈ `prior(class)×need +
est_recall(query)×demand + speed + shape`. So it is NOT capped at the per-class
oracle — the self-assessment reaches toward the per-*query* oracle. Two router
tuning levers: **(a) class priors** (recalibrate from means) and **(b)
self-assessment calibration** (each arm's est_recall accuracy / withdrawal).

## 5. Go-forward (order matters)

1. **LEXICON first** — bigger lever, and it *changes the means* (so recalibrate the
   router once, after). Targets: (a) tag coverage on low-recall/abstain queries →
   recall↑; (b) purge **test docs** ("Test Run ID…") driving contradictions
   (c 0.14, a 0.05) → contradiction-rate → ~0.
2. **THEN router tweaking** — re-derive priors + tune self-assessment against the
   post-lexicon means → close the 0.20 gap (stop over-defaulting to a, stop c).
3. **σ** — repeat runs (esp. d) to get mean ± σ; a high-mean/high-σ arm can't be a default.

Re-run the rig identically after each change → the delta is the lift. Candidate
target: recall 0.44→0.60, contradictions→~0, ~80% of the routing gap closed.
