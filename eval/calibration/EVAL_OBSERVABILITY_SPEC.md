# Eval Observability — fingerprint, timeline, drift, lift

**Goal:** track evals *over time*, attribute deltas to *what changed* (lexicon rev,
priors, code, corpus), and separate intended **lift** from unintended **drift** — so we
never again spend hours proving a metric moved because of a bug instead of the change.

## Principle 0 — fingerprint capture lives in the eval agent, not the caller

`run_calibration` is the single funnel. **Every** invocation path goes through it:

```
UI "Run Calibration" button ─┐
persistent driver ───────────┤
nightly orchestrator ────────┼──► run_calibration()  ──►  captures FINGERPRINT + writes rag_eval_runs
cron / drift monitor ────────┤        (invocation-agnostic)
ad-hoc API ──────────────────┘
```

Fingerprint capture is intrinsic to the run, not bolted onto each caller. Consequence:
it is **impossible to store an eval result without its provenance**, regardless of where
it was triggered. This is the whole foundation — do this first.

## The fingerprint (new columns on `rag_eval_runs`)

Stamped once at run start by `run_calibration`, from AUTHORITATIVE sources — note the
source matters: the eval hits a *deployed* agent over HTTP, so agent-identity fields must
come from the agent, not the local driver.

| column | source | why |
|---|---|---|
| `priors_version` | agent response `routing.priors_version` | which router priors served |
| `agent_git_sha` | agent `/version` (or Cloud Run revision, e.g. `mobius-rag-00193-f7l`) | which CODE served — the mixed-build guard |
| `agent_revision` | serving revision name | human-readable build id |
| `lexicon_revision` | DB `max(lexicon_revision)` (document_tags / policy_lexicon_meta) | which lexicon |
| `corpus_snapshot_at` | DB `max(tagged_at)` on document_tags | corpus state at run start |
| `judge_model` | eval config (locked adjudicator) | ruler identity — deltas must not be judge noise |
| `bank_hash` | sha256 of the query-bank file | which questions (+ their must_facts) |
| `retrieval_config_hash` | agent `/version` (k, _POOL_WIDE_MAX, arms) | retrieval knobs |
| `fingerprint_stable` | computed at run end (see below) | FALSE ⇒ confounded, do not attribute |

**Agent must expose `/version`** returning `{git_sha, revision, priors_version,
retrieval_config_hash}`. Cheap, and it's what makes agent-identity authoritative instead
of guessed from the driver's local checkout.

## Principle 1 — detect mid-run build changes automatically

Every cell's response already carries `routing.priors_version`. `run_calibration` records
it per cell and, at run end, sets `fingerprint_stable = (all cells share one
priors_version AND one agent_revision)`. If a deploy lands mid-run, cells straddle two
builds → `fingerprint_stable = FALSE` → the run is flagged **confounded** and excluded
from lift attribution. This turns the mixed-build trap (which cost us hours) into a data
flag the dashboard enforces.

## Lift vs Drift — one fingerprint answers both

- **Lift** = metric moved AND fingerprint changed → attribute to the changed dim, *gated by σ*.
- **Drift** = metric moved AND fingerprint identical → alert (judge/corpus/infra changed under us).

## Prerequisite — establish σ (the noise floor) before anything else

You cannot detect drift or trust a small lift without run-to-run σ (d's web variance +
judge ±0.01–0.03 are real). **Step zero: run the current config ~5× → store per-metric
mean/σ as the reference band.** Then a lift is real only if it exceeds σ; drift is a band
breach (metric leaves mean ± 2σ with fingerprint unchanged).

## The views

1. **Timeline** — router_recall / composite / oracle over time with **version markers**
   at each fingerprint change ("lexicon 959→1050", "priors v2.0→v2.1"). See the step.
2. **A/B compare — "what changed and what didn't"**:
   - fingerprint diff at top with a **confound guard** (red banner if >1 dim changed OR
     `fingerprint_stable=false`).
   - aggregate deltas (recall / composite / oracle / per-strategy).
   - **per-query movers** with mechanism: recall Δ, retrieval-changed?, chunks in/out,
     tags gained/lost. (Productizes the per-cell trace diff we did by hand.)
3. **Lift ledger** — `intervention (lexicon N→N+1) → Δrecall, Δcomposite → confidence
   (clean A/B vs confounded)`. Ties every version bump to its attributable lift.
4. **Drift monitor** — scheduled same-config run vs the σ band; breach → alert.

## Endpoints (extend the existing eval router)

- `GET /eval/timeline` — runs + fingerprint + headline metrics, time-ordered.
- `GET /eval/compare?a=&b=` — fingerprint diff + confound flag + aggregate + per-query movers.
- `GET /eval/drift` — latest run vs σ band (mean ± 2σ), per metric, status.
- (existing) `GET /eval/runs/{id}/calibration_summary` — the per-eval deep dive, now also
  returning the fingerprint.

## Build sequence

1. **Fingerprint** — add columns to `rag_eval_runs`; add agent `/version`; capture inside
   `run_calibration` (all callers inherit it). Add the `fingerprint_stable` end-of-run check.
2. **σ baseline** — 5 same-config runs → per-metric mean/σ band row.
3. **Endpoints** — timeline, compare (+ confound guard), drift.
4. **UI** — timeline chart w/ version markers → click two points → A/B compare → drill to
   per-query trace; drift status chip. (Extends EvalTab.)
5. **Nightly** — the runbook already brackets a push with baseline+final; schedule it to
   feed the timeline + band. Drift monitor = the same run, scheduled, checked against σ.

## Mental model

> Every run carries a **fingerprint**, captured by the eval agent no matter who invoked it.
> **Lift** = metric moved *and* fingerprint changed (attribute, gated by σ). **Drift** =
> metric moved *and* fingerprint didn't (alert). The dashboard is a version-annotated
> timeline + a confound-guarded A/B diff + a σ band. `fingerprint_stable` makes the
> mixed-build trap a data flag instead of a two-hour investigation.
