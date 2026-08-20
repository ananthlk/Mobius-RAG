# Reference-free eval — build-ready spec

**Owner:** Eval seat · **Status:** spec for build (wiring + UI; the evaluator
already exists) · **Surface:** Query (`/eval/query`) · later: Chat, live traffic.

> **Why this is the primary path, not a secondary one.** A chat query or an
> ad-hoc single query has **no golden answer** — at inference time nobody knows
> the right answer. So the *most common* evaluation is reference-free: does the
> answer stay faithful to what was retrieved? "Vs golden" only applies to a
> curated bank. Reference-free should therefore be the **default** eval mode on
> the Query surface, and the only one that works on real chat traffic.

---

## 1. What it measures (no answer key)

For a `(query, retrieved chunks, synthesized answer)` triple — **no gold facts** —
grade the answer's *faithfulness to retrieval*:

| signal | meaning |
|---|---|
| **Groundedness** | of the claims the answer makes, how many are backed by a retrieved passage |
| **Hallucination** | claims the answer asserts that no passage supports |
| **Contradiction** | claims a passage actively conflicts with (a retrieval/synthesis error) |
| **Honest abstention** | the answer declined / said "not in the sources" and asserted nothing unsupported — this is **good**, scored 1.0 |
| **Calibration** *(P1)* | stated confidence vs measured groundedness — the reliability read |

**This is faithfulness, not correctness.** A perfectly grounded answer built on a
superseded chunk is still wrong — reference-free cannot catch that (that's the
corpus/versioning job). State it on-screen: *calibrated ≠ correct.*

---

## 2. The evaluator already exists — reuse it

`app/services/fact_checker.py :: check_facts` has three modes; one **is** this:

```python
grounding_only = not must_facts and has_answer   # ← reference-free critic
```

Its own docstring: *"reference-free faithfulness critic. Enumerate the ANSWER's
own claims and check each vs the chunks (grounded / contradicted / hallucinated /
honest-abstain). This is the PROD / runtime-critic path — the compliance surface
with no answer key."* It is **already used in production** for runtime answer
grounding, so wiring it into the eval path is low-risk reuse, not net-new logic.

Call it as:
```python
verdict = await check_facts(
    query=query, must_facts=[], chunks=chunks, answer=synth_answer,
    stage="rag_eval_adjudicate",   # locks the ruler to gemini-2.5-pro
)
```

It returns a `FactCheckResult` carrying exactly what the UI needs:
- `verdicts: list[FactVerdict]` — per claim: `fact`, `grounded`, `contradicted`, `support` (0/0.5/1), `passage`, `evidence`
- `hallucinated_claims: list[str]`
- `honest_abstain: bool`
- `score: float` — honesty-weighted [0,1], abstention = 1.0
- `ledger()` — compact, versioned per-claim schema (status ∈ validated|unvalidated|contradicted, `chunk_id` = 1-based passage)
- `model`, `error`, `error_transient`

---

## 3. The fork to fill — `app/main.py`, `_run_trace_for_query`

**Line 15597 is the whole bug:**
```python
eval_result = None
if run_eval and must_facts:        # ← excludes reference-free (no must_facts)
    ...
```

### 3.1 Relax the gate
```python
if run_eval:
    ...
```

### 3.2 Branch the no-gold path
Today the synth call + grounding check sit *inside* the `must_facts` block
(around 15650–15673). **Lift the synth step out** so it runs whenever `run_eval`
is set, then branch:

- **`must_facts` present (Vs golden — unchanged):** keep the current `chunk_only`
  coverage + `full` grounding path exactly as-is.
- **`must_facts` empty (reference-free — new):**
  1. synthesize the answer (reuse the existing synth code),
  2. `check_facts(query, must_facts=[], chunks, answer=synth, stage="rag_eval_adjudicate")` → `grounding_only`,
  3. assemble `eval_result` from the returned `FactCheckResult`.

### 3.3 `eval_result` shape (reference-free)
Additive — leave the gold fields null so no existing consumer breaks:
```jsonc
{
  "mode": "reference_free",           // NEW discriminator: "reference_free" | "vs_golden"
  "coverage": null,                   // gold-only, N/A here
  "coverage_answer": null,            // gold-only, N/A here
  "groundedness": 0.0,                // NEW: n_grounded / n_claims
  "n_claims": 0, "n_grounded": 0,     // NEW
  "hallucinated_claims": [...],
  "honest_abstain": false,            // NEW
  "score": 0.0,                       // honesty-weighted
  "claims": [ {fact, grounded, contradicted, support, passage} ],  // from verdicts / ledger()
  "judge_model": "factcheck/gemini-2.5-pro",
  "ruler_ok": true,                   // existing parity guard
  "fact_checker_version": "...",
  "error": false, "error_transient": false
}
```
(`vs_golden` results keep their current shape + `"mode": "vs_golden"`.)

---

## 4. Query UI — `query.html` evalPanel

`evalPanel` currently renders coverage-vs-gold. Branch on `ev.mode`:

**`reference_free`:**
- KPI row: **Groundedness %** (`n_grounded/n_claims`) · **Hallucinated** (count, red if >0) · **Honest abstain** (badge if true) · **Score**
- Claim ledger table: claim → `grounded` (yes/no) · `contradicted` (yes/no) · `passage #` — the same table shape as the current facts table, driven by `ev.claims`
- Keep the existing **"calibrated ≠ correct"** callout (it finally has data above it)
- Keep the **ruler stamp** + quarantine banner (already built)

**`vs_golden`:** unchanged (existing render).

**Empty/degraded:** if retrieval returned 0 chunks, grounding_only will grade an
answer against no support → expect all-hallucinated or honest-abstain. Show it
honestly; pairs with the existing "retrieval degraded" banner.

---

## 5. Make it the default

Because chat / single queries have no gold, the **eval-mode control defaults to
Reference-free**, not "Vs golden". Order in the kebab: **Reference-free · Vs
golden · Off**. "Vs golden" reveals the golden box only when chosen (unchanged).
This makes the common case one click and removes the current trap where the
default ("Vs golden") silently no-ops without a golden answer.

---

## 6. Acceptance criteria

- [ ] `run_eval:true` with **no** `must_facts` returns a populated `eval` block
      (`mode: "reference_free"`), not `null`.
- [ ] `eval.groundedness`, `eval.hallucinated_claims`, `eval.honest_abstain`,
      `eval.claims[]` are present and populated from `check_facts` grounding_only.
- [ ] `coverage` / `coverage_answer` are `null` in reference-free mode.
- [ ] `judge_model` is the locked `gemini-2.5-pro` (via `stage=rag_eval_adjudicate`);
      `ruler_ok=false` quarantines the result (existing guard).
- [ ] Vs-golden path is byte-for-byte unchanged (regression check on the bank).
- [ ] Query UI renders the reference-free KPI + claim ledger; default mode is
      Reference-free.
- [ ] An honest abstention shows as **good** (score high, no hallucinations),
      not as a failure.

---

## 7. Endpoints & ownership (no new backend service)

- Reuses `POST /admin/trace-explorer/run` (the change is inside `_run_trace_for_query`).
- Reuses `check_facts` grounding_only (prod-proven).
- No new model, no schema change, no new service.
- The synth-lift refactor is the only structurally risky edit — guard it with the
  vs-golden regression check above.

---

## 8. Phasing

- **P0** — wire grounding_only into `_run_trace_for_query`; `eval_result` gains
  the reference-free fields; Query UI renders them; default to Reference-free.
- **P1** — confidence-calibration: plot stated `confidence` vs groundedness
  (reliability curve) — the "score real chat traffic to see where it belongs"
  calibration loop.
- **P2** — run reference-free over **live chat traffic** (no-gold), feeding the
  calibration/candidate-bank loop (frequent ungrounded live queries → candidates
  for a human-authored gold on the Fact-Store surface).

---

*The reference-free grade is a different scale from the gold bank (the AHCA
baseline 66.4 / 42.6). It measures faithfulness-to-retrieval, not
correctness-vs-gold — the right metric where there is no answer key, which is
almost everywhere real usage happens.*
