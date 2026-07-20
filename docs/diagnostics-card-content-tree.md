# Diagnostics Card — content tree (single source of truth)

Owner: EVAL agent. Surface: UX (`DiagnosticsCard`). Data: `rag_query_decisions` row + `corpus_search` `full_response`.
Every node maps to a real RAG step (source-verified by RAG agent 2026-07-17). Gaps flagged inline.
Node shape = UX's `TreeNode { id, title, summary, latencyMs?, status, children?, telemetry?, strategyScores? }`.
Status: `ok` | `warn` (weak joint) | `gray` (not-built / not-triggered).

## Top-level (DiagnosticsTree)
| field | source |
|---|---|
| query | `response.query` |
| answer | `response.llm_answer` |
| route.strategy / .confidence | `response.strategy_used` / `response.confidence` |
| focusTags | `response.query_profile.tag_matches` |
| grades.retrieval/synthesis/gap | row `retrieval_grade` / `synthesis_grade` / `synthesis_gap` |
| claims | from row `per_claim_ledger`: passed=count(status='validated'), total=length |
| latencyMs | `response.telemetry.total_ms` |
| decisionId | `response.routing_decision_id` |

---

## root — "Full query trace"
summary: `routed {strategy} · {answer≤60c} · retr {rg}/synth {sg} · {passed}/{total} ✓` · latency total_ms · status ok
children: reason · act · observe · decide

## 1 · REASON  (latency = Σ classify+route ms) status ok
summary: `cleanup {tok_in}→{tok_kept} · {query_type} · scored {top_strategy} {top_score} argmax · gap {gap}`

- **gate** — Fail-fast gate. status: ok if passed / warn if fired. telemetry: `response.gate {passed, reason}` where reason ∈ phi_detected·jailbreak·self_referential·no_domain_match (first-match-wins; bypassed if include_document_ids set).
- **cleanup** — `_tokenize`: pre-extract 5 literal regexes → split rest. telemetry: from `query_profile` → {literal_anchors, tag_matches (kept), untagged_meaningful, dropped=noise}. summary `{n} tokens → {kept} kept`.
- **rewrite** — 3 variants, always computed. telemetry: `response.queries_per_strategy {hybrid, phrase_strict, vector_broad}`. summary `3 per-strategy variants`.
- **classify** — QueryProfile. telemetry: `query_profile {query_type, coverage, d_tags, j_tags, p_tags, literal_anchors, semantic_core}` + `routing.classify_flags {is_exploratory, has_service_specificity}` (LIVE, rev 00423). summary `{query_type} · coverage {cov}`.
- **scorer** — linear v1. `strategyScores` ← `routing.scores`. telemetry: `routing.score_breakdown` (per-feature contribution: base + Σ weight×feature), `routing.feature_vector` (7: exclusivity, literal, corpus_depth, thematic_policy, wide_pool, inheritance, crawlability), `routing.self_assessments` (a,b est_recall), withdrawn. summary `{top} wins {score}` · TRIGGERS why-A-won bars. Note `multi_invoke_considered` here.

## 2 · ACT  (latency = Σ strategy ms) status ok
summary: `strategy {chain} · {n_chunks} chunks · {answer_len}c answer · {confidence} · answer={answer≤50c}`
**REQUIREMENT — one ACT sub-tree PER executed strategy (Ananth 2026-07-19).** `strategies_tried` is a LIST, one entry per attempt. Render **N ACT branches** — one retrieve/rerank/assemble per `strategies_tried` entry — labelled by that entry's strategy. NEVER collapse an escalation to only the final strategy: a `b→d` escalation MUST show `ACT›b` (its corpus chunks — where the answer's evidence came from) AND `ACT›d` (0 chunks — the empty escalation). Header/summary shows the **chain** `b→d`, not just `d` (labelling it `d` alone overstates d and hides that the answer came from b). Same for `invoke_all` (a+b union = 2 branches). The escalate reason (`corpus_exhausted` etc.) annotates the transition between branches — see DECIDE/escalate; do NOT render `escalate: single attempt` when `strategy_chain` len>1.

- **retrieve** — VARIABLE sub-tree per strategy. summary `{algo} → {n} chunks · answer facts in chunk {ids}`:
  - **a · hybrid (11)**: tsquery → cascade-pool → bm25 → vector → dtag-arm → RRF(k=60) → neighbors → rerank → tag-boost → coverage-floor(1.0) → decay. telemetry per sub-step ← `strategies_tried[a].{bm25_hits, vector_hits, embed_ms, bm25_ms, vec_ms, rerank_ms, chunks_bm25_only/vector_only/both}` + `scoring_trace[chunk].{sim_raw, authority_raw, length_raw, jpd_raw, coverage_raw, coverage_present/missing, chunk_dtag_boost}`.
  - **b · wide→themes→narrow (4)**: wide(k=80) → cluster into ≤5 themes → BM25-in-theme → synth. telemetry ← `response.themes`, `theme_diagnostic`, `telemetry.strategy_b.{wide_ms, themes_ms, narrow_ms, wide_hits}`.
  - **c · reverse-RAG (4)**: LLM-generate-w/citations → locate (url→title→quote→google) → verify-verbatim → outcome-matrix (8 states). telemetry ← `response.validated_citations`.
  - **d · external (6)**: resolve-payer → sitemap → search(Vertex Grounding→DDG/CSE→plain) → rerank-hits → fetch+extract(5 URLs, 8s) → LLM-synth. telemetry ← `strategies_tried[d]` + GAP: per-tier/per-URL fetch breakdown (RAG adding).
  - **s · fact_store serve (0 retrieval, LIVE rev 00439)**: payer j-tag hard-gate → tag+vector blend → serve certified fact. NO BM25/pgvector — `n_chunks=0` BY DESIGN (a direct certified serve, not retrieval; the raw trace's "BM25 0 · pgvector 0 · 3 rounds" is the *old* card mis-reading this as a failed retrieval — see fast-exit note). Checked FIRST, before a/b/c/d scorer; hit → fast-exit, miss → fall through. telemetry ← `routing.{method:fact_store, fact_predicate, fact_score, fact_telemetry_id, fact_cert_grades}` + served `{answer_text, value, source_ref, authority_level, freshness{last_verified_at, valid_until, stale}, cert{status, grades}, score}`. summary `fact_store · {predicate} · score {fact_score} · cert {status}`. **This leaf renders the PROVENANCE + FRESHNESS card**: derivation (verified_via, grader, grades) → sources (source_ref doc/chunk/page — chunk-pointer GAP, comparator sees it, persist pending) → freshness (last_verified/valid_until/stale) → drift-watch (§8 confirm/drift, next-check). A store serve is PRE-CERTIFIED → OBSERVE reads `cert.grades` as the grade, NOT a re-grounding on empty chunks.
- **rerank** — weighted formula (NOT a model): `(0.25·sim + 0.10·auth + 0.05·len + [0/0.20]·jpd + [0/0.55]·coverage)/MAX`. thresholds high≥0.55 med≥0.35 low≥0.18. telemetry ← `scoring_trace` per-chunk signals. summary `top {top_rerank}`.
- **assemble** — page-dedup (doc,page) + content-dedup (sha/200c) + promoted-neighbors exempt (cap 10). summary `{n}/{k} · {mode}`. telemetry: chunk order.
- **synthesize** — 7-rule prompt, ≤12 chunks×1500c, JSON{answer,used_passages,confidence}, honest-abstain, 59G attribution, two-phase streaming. telemetry ← `telemetry.{llm_ms, model, used_passages, n_passages_offered}` (LIVE, rev 00423 — `used_passages` = cited passage indices → cross-link to per_claim_ledger chunk_ids). summary `answer built · {confidence}`.

## 3 · OBSERVE status ok
summary: `retrieval {rg} · synthesis {sg} · {passed}/{total} validated · 1 row`
- **retrieval_grade** — chunk-only fact_check. telemetry `{grade, basis}`. NULL in prod (no gold) → status gray in prod.
- **synthesis_grade** — grounding (prod) / coverage (eval). status warn if gap<0. telemetry `{grade, gap}`.
- **per_claim_ledger** — telemetry ← row `per_claim_ledger` [{fact, status, chunk_id, support}]. status warn if any contradicted.
- **decision_row** — telemetry ← row `{leaf_key, feature_vector, strategy_scores, priors_version, fact_checker_version, corpus_version, is_prod}`. corpus_version=1 → note "bump not wired".

## 4 · DECIDE status warn (bandit open)
summary: `gap {gap} → {single/multi} · {fast_exit} · bandit not wired`
- **multi_invoke** — v2 router, gap<0.08 AND both≥0.30. status gray if not triggered. telemetry ← `routing.multi_invoke_considered`.
- **escalate** — 4-try outer loop; abstain trigger (n_chunks>0 AND top_rerank≥0.40 AND low-conf) + inherited-authority retry; budget fast/copilot=0, thinking/research=2, default=1. telemetry ← `strategy_chain, escalated, inherited_boost`.
- **fast_exit** — query-sig (md5[:8] first 100c) seen + attempt>0. telemetry ← `fast_exit {fired, reason}`.
- **bandit** — status gray, NOT BUILT. telemetry `{reward: synthesis_grade, would_update: weights[leaf], status: loop-open}`. The dashed feedback arrow.

---

## NOT-BUILT leaves (REASON action tree) — render as gray planned nodes
`reformulate` re-query · `f` honest-floor · `m` cached-replay · `research` fanout.
(NOTE: `s` structured-read is now BUILT — it's the live `s · fact_store serve` leaf (rev 00439+). Do NOT list `s` as not-built; it's a real ACT/retrieve leaf with the provenance card. Only reformulate/f/m/research remain planned.)

## PERSISTED TRACE shape (rag_query_traces, invariant "shown→persisted", trace half CLOSED 2026-07-19)
Drill-down re-fetches a past query's full_response by correlation_id from `rag_query_traces` (PHI-scrubbed at write via /redact, fail-closed, 180d retention). **Consumers MUST tolerate TWO `query_profile` shapes:** (a) NEW allowlist (rev 00449+): `{raw_query[MASKED], query_type, coverage, tag_matches, literal_anchor_count, untagged_meaningful_count}` — leak fields `semantic_core`/`untagged_meaningful_tokens` REMOVED (they carried raw query/SSN/MRN — R2 catch); (b) LEGACY rev-00448 rows: old full shape, phi_flag=false only (classifier-cleared). Masked fields render as "a••••• l•••" (raw_query) / "[redaction unavailable]" (suppressed). phi_flag + evidence_categories (categories only) on the row. Live traces render full; reconstructed traces render the allowlist subset.

## Telemetry GAPS
CLOSED (live rev 00423): ~~classify_flags~~ ✓ · ~~synthesis used_passages/llm_ms/model/n_passages_offered~~ ✓
OPEN (RAG queued): strategy-d fetch-tier/per-URL breakdown · `caller_id` (row, NULL). Render "not captured yet" for these two only.
