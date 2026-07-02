# Nightly Eval Runbook — bracketing a lexicon/RAG push with baseline + final calibration

**Audience:** the nightly job agent that pushes lexicon + RAG docs.
**Goal:** measure the *lift* of a nightly corpus change by running the **same** calibration
before (baseline) and after (final) the push, then diffing the two summaries.

The eval we run is the **5-axis fact-checker calibration** (22 golden queries × 5 strategies =
110 cells: forced `a/b/c/d` + `natural`=router). Each cell forces a strategy, `skip_synthesis`,
and scores retrieval with the locked LLM critic (`fact_checker.py`, chunk-only recall +
contradiction). Do **not** use `/api/eval/trigger` (that runs the legacy verdict-based `eval/run.py`,
a different metric). Use `/api/eval/calibrate/trigger`.

Base URL (dev): `https://mobius-rag-ortabkknqa-uc.a.run.app` (eval endpoints are unauthenticated on dev).

---

## Hard preconditions (verify before running anything)

1. **Adjudicator is locked** to `gemini-2.5-pro` (mobius-chat `model_registry.py`: `rag_eval_adjudicate`
   is on gemini-2.5-pro's `eligible_stages` ONLY). **Never change the judge model between the baseline
   and final run** — score deltas would then reflect judge variance, not corpus lift.
2. **One eval at a time.** The trigger is guarded by a server lock (`_RUN_LOCK`) and returns HTTP 409 if
   busy. Call `GET /api/eval/active` first; only proceed if `{"active": false}`.
3. **Corpus must be stable during a run.** Do NOT retag / write `document_tags` while a calibration is in
   flight — a moving corpus contaminates the numbers (and inflates latency via DB contention). The whole
   point of the bracket is: freeze → baseline → push → freeze → final.

---

## The bracket

### STEP 1 — Baseline (BEFORE the push)

```bash
BASE=https://mobius-rag-ortabkknqa-uc.a.run.app
# 1a. ensure nothing is running
curl -s $BASE/api/eval/active         # expect {"active": false}

# 1b. kick off calibration (returns immediately; does NOT return a run_id)
curl -s -X POST $BASE/api/eval/calibrate/trigger \
  -H 'Content-Type: application/json' \
  -d '{"notes":"nightly-baseline-<UTC-DATE>"}'
# → {"status":"started","kind":"calibration",...}

# 1c. resolve the run_id (newest in-flight run)
curl -s $BASE/api/eval/active         # → {"active":true,"run_id":"<UUID>","n_completed":N}
```

### STEP 2 — Poll to completion

Poll every ~30s. **Completion signal = `is_running:false`** (i.e. `completed_at` is set). Target is
`n_completed ≈ 110`.

```bash
RID=<UUID>
curl -s $BASE/api/eval/runs/$RID/progress
# → {"n_queries":110,"n_completed":72,"is_running":true, ...}
```

**Stall guard:** if `n_completed` does not advance for **>5 minutes** while `is_running` is still true,
the in-process background task stalled (historic Cloud Run failure mode). Fall back to the **durable
driver** (see bottom) or alert. Expected clean wall-clock: ~8–15 min depending on load.

### STEP 3 — Read + store the baseline summary

```bash
curl -s $BASE/api/eval/runs/$RID/calibration_summary
```

Returns (store the whole JSON as `baseline`):

```json
{
  "run_id": "...", "n_queries": 22,
  "strategies": {
    "a": {"answer_rate":..,"recall":..,"precision":..,"contra_per_cell":..,"median_latency_ms":..,"p95_latency_ms":..},
    "b": {...}, "c": {...}, "d": {...},
    "natural": {...}          // ← the router path
  },
  "oracle_recall": ..,        // mean of per-query max over forced a/b/c/d = corpus ceiling
  "router_recall": ..,        // = strategies.natural.recall
  "best_single_recall": ..,
  "routing_headroom": ..      // oracle_recall − router_recall (points the router leaves on the table)
}
```

### STEP 4 — Do the push (the job's real work)

Push lexicon → retag docs → update RAG docs. **Wait until retag/tagging is fully complete** (no more
writes to `document_tags`) before Step 5 — a corpus in flux contaminates the final run.

### STEP 5 — Final (AFTER the push)

Repeat Steps 1–3 with `"notes":"nightly-final-<UTC-DATE>"`. Store the summary as `final`.

### STEP 6 — Compute lift + gate

Diff `final` vs `baseline`:

| metric | source | good = | alert / rollback signal |
|---|---|---|---|
| **router_recall** | `.router_recall` | ↑ or hold | drop vs baseline |
| **oracle_recall** | `.oracle_recall` | ↑ (corpus ceiling rose) | drop = corpus regressed |
| **routing_headroom** | `.routing_headroom` | shrink or hold | grows a lot (router mis-routing) |
| **contra_per_cell** (per strategy) | `.strategies.*.contra_per_cell` | stay ~0 | rises above baseline = bad/test docs reintroduced |
| **answer_rate** | `.strategies.*.answer_rate` | ↑ with recall | recall up but answer_rate flat = tuning issue |
| latency median/p95 | `.strategies.*.*_latency_ms` | informational | large regression (but DB contention inflates — don't hard-gate) |

Primary lift number to report: **Δ router_recall** and **Δ oracle_recall**. Headroom is the router-tuning
gap; oracle is the corpus-ceiling gap that lexicon/retag is meant to raise.

**Reference baseline to beat** (locked-judge, dev run 4c13c9da / 969a5170, pre-lexicon):
`router_recall ≈ 0.408`, `oracle_recall ≈ 0.602`, `routing_headroom ≈ 0.19`, all `contra_per_cell ≈ 0`
except `d ≈ 0.19`. (After the v2 prior update, expect router_recall to climb toward oracle.)

---

## Durable fallback — if the HTTP trigger stalls

The trigger uses an in-process `asyncio` background task. The service is deployed with
`min-instances=1 --no-cpu-throttling` so it *should* survive, but if the stall guard fires, run the
calibration harness directly as an out-of-process job (the persistent-driver pattern that ran the
authoritative baselines):

```bash
# from mobius-rag/, with a Cloud SQL proxy to the dev DB on 127.0.0.1:5433
# and SKILL_KEY exported (secret: mobius-skill-llm-internal-key, project mobius-os-dev)
python -c "import asyncio; from eval.calibrate import run_calibration; \
  asyncio.run(run_calibration(endpoint='https://mobius-rag-ortabkknqa-uc.a.run.app/api/skills/v1/corpus_search_agent', \
  notes='nightly-final-<DATE>'))"
```

It writes to the same `rag_eval_results` table, so `/calibration_summary` reads it identically. The
clean long-term fix (recommended for the nightly job) is to wrap `run_calibration` in a **Cloud Run Job**
rather than the fire-and-forget HTTP task, so the nightly scheduler invokes a job with a guaranteed
lifecycle instead of depending on the API container staying warm.

---

## One-line summary for the agent

> Before push: `POST /api/eval/calibrate/trigger` → get `run_id` from `/api/eval/active` → poll
> `/runs/{id}/progress` until `is_running:false` → save `/runs/{id}/calibration_summary` as baseline.
> Do the push, wait for retag to finish. Repeat for final. Report Δ router_recall, Δ oracle_recall,
> Δ routing_headroom, and any rise in contra_per_cell. Never change the judge model mid-bracket.
