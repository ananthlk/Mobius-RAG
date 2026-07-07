# Canonical per-strategy baseline — recall, sigma, query-class split

**Purpose.** A durable reference for "is this delta real or noise" — per
strategy (a/b/c/d/natural) and per query class (conceptual vs
literal-anchor), on the `queries_cmhc.yaml` bank. Complements
`CALIBRATION_BASELINE.md` (methodology) and `EVAL_OBSERVABILITY_SPEC.md`
(fingerprint/drift design) — this doc is the actual numbers to check a new
run against.

**Status: PROVISIONAL.** Everything below is derived from n=6 calibration
runs that share `(priors_version, lexicon_revision, retrieval_config_hash)`
but NOT `agent_revision`/`agent_git_sha` (both moved across the 6 runs,
untracked by the fingerprint's stability check today — see the "gaps"
section). Treat these as the best available reference, not a finished
noise floor. **The real σ-baseline (5 back-to-back runs, zero changes of
any kind including code) is still queued and would supersede this.**

---

## 1. Per-strategy recall — mean & sigma (same-fingerprint group, n=6)

Runs: e1404528, e8b93a87, ec5d655f, 397f24b2, a125374a, f861c7be
(2026-07-02 through 2026-07-05, priors_version=v2.1.2026-07-01-canonical-blend,
lexicon_revision=1050, retrieval_config_hash=5dd00544003d).

| Strategy | mean recall | σ | range | note |
|---|---|---|---|---|
| a | 0.434 | 0.045 | [0.389, 0.510] | BM25 cascade |
| b | 0.241 | **0.007** | [0.230, 0.246] | wide→theme→narrow; near-deterministic as expected |
| c | 0.125 | 0.032 | [0.083, 0.182] | parametric-heavy, no real chunks |
| d | 0.399 | **0.048** | [0.321, 0.468] | highest σ — bimodal (strong or a flat 0.0), see §3 |
| natural (router) | 0.471 | 0.031 | [0.439, 0.518] | |

**Reading this:** `b`'s near-zero σ is a sanity check that the fingerprint
grouping is working (its retrieval path has no LLM-synthesis variance to
speak of). `d`'s σ is genuinely the highest — confirmed by tracing
individual cells, not just the aggregate (see §3). Do not read a strategy's
single-run recall against another strategy's; only compare a strategy
against its OWN mean ± 2σ band above.

**2σ bands (use these to judge "is this run's number surprising"):**

| Strategy | mean − 2σ | mean + 2σ |
|---|---|---|
| a | 0.344 | 0.524 |
| b | 0.227 | 0.255 |
| c | 0.061 | 0.189 |
| d | 0.303 | 0.495 |
| natural | 0.409 | 0.533 |

## 2. Query-class split (conceptual vs literal-anchor)

Bank composition: 18 conceptual, 3 literal-anchor, 1 wide_pool (too small
to band on its own — fold into "other" or ignore).

Same 6-run group, strategy `a` and `natural`:

| Strategy | class | first-run | last-run | trend |
|---|---|---|---|---|
| a | conceptual (n=18) | 0.407 | 0.537 | steady climb — **all of a's real gain lives here** |
| a | literal-anchor (n=3) | 0.411 | 0.411 | **flat every single run** — BM25 exact-match, as it should be |
| natural | conceptual (n=18) | 0.491 | 0.559 | climbs, tracks `a` |
| natural | literal-anchor (n=3) | 0.428 | 0.300 | **declining** — opposite direction of everything else, n=3 so fragile, but a real anomaly worth one more confirming run before acting on |

**Practical use:** if a future run shows `a`'s literal-anchor recall move
off 0.411, that's a real retrieval-mechanism change (new docs, reranker
change, pool composition) — it has never moved on its own in 6 runs. If
`natural`'s literal-anchor number keeps declining, that's the router
choosing something other than `a` for anchor-style queries and losing.

## 3. Headroom (oracle − router) — the full series, not just two points

All completed `queries_cmhc.yaml` calibrations to date (see `/api/eval/timeline`):

| run | date | router | oracle | headroom |
|---|---|---|---|---|
| 969a5170 | 07-01 | 0.408 | 0.602 | 0.194 |
| 75524dd1 | 07-01 | 0.389 | 0.522 | 0.133 |
| 41b5c5e7 | 07-01 | 0.392 | 0.590 | 0.198 |
| 2ecb72ab | 07-02 | 0.448 | 0.507 | 0.059 |
| e1404528 | 07-02 | 0.460 | 0.499 | 0.039 |
| e8b93a87 | 07-02 | 0.445 | 0.545 | 0.100 |
| ec5d655f | 07-03 | 0.454 | 0.545 | 0.091 |
| 397f24b2 | 07-05 | 0.439 | 0.560 | 0.121 |
| a125374a | 07-05 | 0.518 | 0.632 | 0.114 |
| f861c7be | 07-05 | 0.511 | 0.685 | 0.174 |
| 7dccc341 | 07-06 | 0.543 | 0.624 | 0.081 (mixed-build, only 81/110 cells clean — see [[project_eval_observability]]) |

**Headroom range across 11 runs: 0.039–0.198.** This is genuinely noisy,
not a clean trend — do not read "widening" or "narrowing" off two adjacent
points. A real trend claim needs the σ-baseline in place so headroom has
its own band, the same way §1 gives one to recall.

## 4. Known gaps in this reference (do these before trusting it further)

1. **`agent_git_sha` is blank** in every fingerprint captured so far — a
   real code deploy moved through the "same-fingerprint" group in §1
   without being flagged. Confirmed causing a real miss on 2026-07-06 (a
   mid-run deploy silently split a calibration 81/29 clean/contaminated,
   `stable:true` never caught it). Fix: populate it in `/version`.
2. **Fingerprint stability doesn't track `agent_revision`** as a dimension —
   only `priors_version`/`lexicon_revision`/`retrieval_config_hash`. Same
   root cause as #1.
3. **Real σ-baseline never run** — 5 back-to-back calibrations on a truly
   frozen build (no deploy, no retag, nothing) would replace §1's n=6
   same-fingerprint-but-not-same-code approximation with the real thing.
4. This doc is a point-in-time snapshot (2026-07-06) — re-derive §1/§3 as
   more same-fingerprint runs accumulate; don't let it go stale silently.
