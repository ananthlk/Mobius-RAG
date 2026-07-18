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
summary: `strategy {s} · {n_chunks} chunks · {answer_len}c answer · {confidence} · answer={answer≤50c}`
NOTE: if `invoke_all` set → TWO retrieve branches (a+b union). If `strategy_chain` len>1 → N attempt branches (escalation).

- **retrieve** — VARIABLE sub-tree per strategy. summary `{algo} → {n} chunks · answer facts in chunk {ids}`:
  - **a · hybrid (11)**: tsquery → cascade-pool → bm25 → vector → dtag-arm → RRF(k=60) → neighbors → rerank → tag-boost → coverage-floor(1.0) → decay. telemetry per sub-step ← `strategies_tried[a].{bm25_hits, vector_hits, embed_ms, bm25_ms, vec_ms, rerank_ms, chunks_bm25_only/vector_only/both}` + `scoring_trace[chunk].{sim_raw, authority_raw, length_raw, jpd_raw, coverage_raw, coverage_present/missing, chunk_dtag_boost}`.
  - **b · wide→themes→narrow (4)**: wide(k=80) → cluster into ≤5 themes → BM25-in-theme → synth. telemetry ← `response.themes`, `theme_diagnostic`, `telemetry.strategy_b.{wide_ms, themes_ms, narrow_ms, wide_hits}`.
  - **c · reverse-RAG (4)**: LLM-generate-w/citations → locate (url→title→quote→google) → verify-verbatim → outcome-matrix (8 states). telemetry ← `response.validated_citations`.
  - **d · external (6)**: resolve-payer → sitemap → search(Vertex Grounding→DDG/CSE→plain) → rerank-hits → fetch+extract(5 URLs, 8s) → LLM-synth. telemetry ← `strategies_tried[d]` + GAP: per-tier/per-URL fetch breakdown (RAG adding).
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
`s` structured-read · `reformulate` re-query · `f` honest-floor · `m` cached-replay · `research` fanout.

## Telemetry GAPS
CLOSED (live rev 00423): ~~classify_flags~~ ✓ · ~~synthesis used_passages/llm_ms/model/n_passages_offered~~ ✓
OPEN (RAG queued): strategy-d fetch-tier/per-URL breakdown · `caller_id` (row, NULL). Render "not captured yet" for these two only.
