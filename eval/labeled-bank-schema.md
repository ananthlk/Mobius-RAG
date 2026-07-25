# Labeled Bank Schema — weighted facts + primary gate (Eval, 2026-07-24)

The calibration bank is the oracle everything is measured against, so its
structure is load-bearing. This defines the per-fact weighting that must be
built in from fact #1 (retrofitting means re-verifying every fact). Target:
22 → 150-200 queries, candidate queries sourced from LEGACY corpus_search logs
(real users; NOT rag_query_decisions, which is self-referential), each fact
verified against its source doc at authoring time.

## Per-fact structure

`must_facts` becomes a list of typed facts (was: flat list of strings):

```yaml
- id: cmhc001
  query: What is the timely filing deadline for Sunshine Health FL Medicaid claims?
  golden_answer: |
    Participating providers must submit initial claims within 180 days from
    date of service; non-participating providers have 365 days. ...
  must_facts:
    - fact: "180 days for participating providers (initial claims)"
      tier: primary          # primary | secondary | tertiary
      source: "Sunshine Health Provider Manual 2024, §Claims Timely Filing"
    - fact: "365 days for non-participating providers (initial claims)"
      tier: primary
      source: "Sunshine Health Provider Manual 2024, §Claims Timely Filing"
    - fact: "Sunshine Health is the payer in question"
      tier: secondary
      source: "query-implied"
  bonus_facts: [...]         # unchanged — tertiary-equivalent, never gated
  forbidden_facts: [...]     # unchanged — assertion of any caps to 0
```

Three fields per fact: `fact` (the string, unchanged — what check_facts grades),
`tier` (the weight), `source` (the doc that verifies it — the non-negotiable
verification step, now recorded not just performed).

## Tiers — the reproducible test

Assign by answering ONE question about each fact, in order:
- **primary** — "Is this part of THE answer? If the response omits it, is the
  answer wrong/incomplete for what was asked?" A timely-filing query's day-counts
  are primary; the payer name is usually secondary (context, not the answer).
- **secondary** — a needed qualifier: correct-but-caveated. Missing it makes the
  answer thinner, not wrong.
- **tertiary** — nice-to-have; adds completeness, absence isn't a defect.

Tiered (not numeric 0-1) because the bank is multi-author and IS the ground
truth — a 0.6-vs-0.7 call isn't author-reproducible; "THE answer / qualifier /
nice-to-have" is. Numeric weights are a v2 refinement only if tiers prove coarse.

## Scoring — weighted_recall × primary_gate

For a graded prefix, using check_facts's per-fact ledger (which facts were
supported) mapped back to tiers:

```
weight(primary)=3, weight(secondary)=2, weight(tertiary)=1   # ordering only
weighted_recall = Σ weight(captured facts) / Σ weight(all facts)
primary_gate    = 1.0  if ALL primary facts captured
                = 0.5  (cap, calibrate) otherwise
score = weighted_recall × primary_gate
```

The **gate does the categorical work** — Ananth's "no core answer → cannot be
good" — so the exact tier numbers only order among primary-complete answers and
are low-stakes. Composes with the monotone envelope (independent transforms) and
applies to all three modes' @K curves. A query with no primary facts (rare —
pure-exploratory) has primary_gate ≡ 1.0 by vacuous truth; flag those for review.

## Bucket coverage — deliberate, not natural

depth_bucket is DERIVED at query-time from pool_metadata, so it can't be
pre-assigned. The natural distribution is skewed (current 22: 11 db2 / 9 db3 /
2 db4 / 0 db1). Author 100, run pool, read the histogram, then TARGETED-author
to fill under-represented buckets — seek tight-pool queries (specific code/payer
lookups → db1) and very-broad queries (open-ended policy → db4). Bank is
append-only + versioned so it grows without re-grading prior queries.

## Loader impact

`eval/run.py load_bank()` folds sibling keys into `expected`; it must now accept
`must_facts` as list-of-dicts and expose `[f["fact"] for f in must_facts]` to
check_facts (which wants strings), keeping the tier/source alongside for the
weighted-scoring post-process. Backward-compat: a plain-string entry = tier
`secondary`, source `unspecified` (so cmhc_v1.1 still loads while it's migrated).
