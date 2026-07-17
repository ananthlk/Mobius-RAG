# GA Baseline — Joint Forensic Log

*Durable baseline run `durable-ga-baseline-00347-7mf`, rev 00347 (frozen), bank `queries_cmhc.yaml`
(22 q × 5 paths = 110 cells). Method: go **strategy by strategy (a → b → c → d → router)**, and for
**each question × strategy** record what happened, **what worked**, **what didn't**, and a
**joint-verdict** — is the strategy failing/winning for the *right* reason (behaving as designed)?
Data source: `rag_eval_results` (stored per-cell score + verdict + judge_reasoning + chunks_summary
+ must_facts — no re-probing, reproducible against the frozen fingerprint). We revisit the whole log
once all five passes are done.*

**Interim totals (95/110, will refresh):** a 0.448 · b 0.353 · c 0.188 · d 0.476 · **router 0.510** ·
oracle 0.628.

**Verdict shorthand:** ✅ right-reason win · ⚠️ win-but-fragile / lucky · 🟥 fail-for-right-reason
(honest limit) · 🐛 fail-for-wrong-reason (a bug/joint defect to fix).

---

## Strategy A — pinpoint (BM25 + vector)
*Design intent: should WIN when the answer is a specific fact living in a retrievable corpus chunk,
especially narrow-tag / literal-anchor queries. Should FAIL honestly when the tag is missing, the
chunk is buried, or the fact isn't in the corpus. A fail-for-wrong-reason = the fact IS retrievable
but a ranked/tagging/synthesis defect lost it.*

| q | score | must (req'd) | worked | didn't | joint |
|---|---|---|---|---|---|
| cmhc001 | **1.00** | Sunshine timely filing 180/365 | narrow tag → timely_filing chunk (p121) surfaced; ALL facts supported | — | ✅ right-reason win (poster child: narrow-tag specific fact, in corpus) |
| cmhc002 | 0.00 | Sunshine PA H0019 | — | **agent_timeout 90s, 0 chunks** | 🐛 INFRA — the H0019 timeout bug, NOT a strategy signal; exclude from a's real mean |
| cmhc003 | 0.33 | Aetna no PCP referral | got Aetna payer | referral-policy fact not in top-5 | 🟥 honest (payer-got/detail-missed) |
| cmhc004 | 0.67 | telehealth BH intake 90791 AV | telehealth-BH coverage | CPT 90791 + audio-video specifics | ⚠️ good partial |
| cmhc005 | 0.20 | H0015 IOP HE/HF modifier | FL Medicaid payer | **modifier absent** | 🟥 content gap (modifiers) |
| cmhc006 | 0.33 | Sunshine PA 7d/48h | Sunshine PA context | 7-day timeframe not surfaced — **but b=1.0 gets it** | 🟥→routing: fact IS retrievable, a *buries* it → escalate/boost opportunity |
| cmhc007 | 0.67 | corrected claim freq-7, ICN/DCN | claims cluster (p119–124), freq code 7 | ICN/DCN + Loop2300 REF*F8 specifics | ✅ good partial (narrow claims section) |
| cmhc008 | 0.33 | Sunshine PA psych-test 96130, 4u | Sunshine title | **only 1 chunk retrieved**; codes missed | 🟥 weak retrieval (nchunks=1) |
| cmhc009 | **0.83** | Sunshine appeal 90d, Reconsideration | appeal/dispute cluster; 90d + Reconsideration | RA-vs-claim-denial nuance | ✅ right-reason win |
| cmhc010 | 0.33 | Sunshine eligibility portal/every-visit | Sunshine implied | eligibility-process facts absent | 🟥 procedural (b) / content gap |
| cmhc011 | 0.67 | Sunshine PA inpatient psych | SMI PA + inpatient | "non-emergency" qualifier explicit | ⚠️ good partial |
| cmhc012 | 0.33 | ABA 97153/97151 | AHCA/ABA fee-schedule context | codes 97153/97151 absent | 🟥 content gap (codes) |
| cmhc013 | 0.50 | Sunshine adult dental preventive | dental listed generally | "adult" + preventive specifics | ⚠️ partial |
| cmhc014 | 0.33 | Sunshine PBM = Express Scripts | Sunshine Rx coverage | **PBM name (Express Scripts) absent** | 🟥 specific-fact gap |
| cmhc015 | 0.50 | Sunshine appeals oral/written, G&A | appeals context | oral/written + G&A dept | ⚠️ partial (procedural → b) |
| cmhc016 | **0.00** | Sunshine credentialing CAQH/NCQA | nothing | **all facts absent** (d=0.83 HAS it) | 🟥 CORPUS GAP → d territory; **router mis-routed to a** |
| cmhc017 | 0.17 | telehealth GT/95, POS 02/10 | managed-care context | modifier + POS absent | 🟥 content gap (modifiers, like cmhc005) |
| cmhc018 | 0.50 | Sunshine PA ECT, portal | Sunshine PA role | ECT-specific + portal | ⚠️ partial |
| cmhc019 | 0.50 | **Aetna** timely filing 180d | Aetna payer | **180-day deadline NOT in chunks** (vs cmhc001 Sunshine=1.0!) | 🟥 content-coverage varies by payer → **strategy-s fix** |
| cmhc020 | 0.33 | Simply PA 14d/72h | Simply title | PA timeframes absent | 🟥 shallow-payer (same pattern as cmhc006) |
| cmhc021 | 0.17 | pediatric enroll AHCA/DCF, portal | AHCA/Medicaid prereq implied | DCF ACCESS + Sunshine portal | 🟥 procedural / content gap |
| cmhc022 | 0.67 | EPSDT <21, 837P/CMS-1500 | EPSDT<21 + FL Medicaid | 837P/CMS-1500 | ⚠️ good partial |

**Strategy A — running notes:**
- **Wins are all narrow-tag specific-fact-in-corpus** (cmhc001 timely_filing 1.0, cmhc009 appeal 0.83) + strong claims-cluster partials (cmhc007, cmhc022). Joint working exactly as designed.
- **Dominant failure = "got payer (~0.33), missed the operational detail."** Two DISTINCT causes — must be separated:
  - **(a) CONTENT GAP** (fact not in corpus): modifiers (cmhc005/017), codes (cmhc012), PBM (cmhc014), credentialing (cmhc016 — d=0.83 proves web has it). Honest a-fails → fix = ingestion or d-routing, NOT an a-bug.
  - **(b) BURIED-but-retrievable** (fact IS in corpus): cmhc006 (b=1.0 gets the 7-day timeframe a misses). → a-retrieval-depth opportunity (boost/escalation), not a content gap.
- **Content-coverage unpredictability CONFIRMED at data level:** cmhc001 (Sunshine timely filing) a=1.0 vs **cmhc019 (Aetna timely filing) a=0.5** — identical question, opposite outcome, because the fact is retrievable for Sunshine but not Aetna. No query feature predicts this → the deterministic fix is **strategy s** (timely_filing field read).
- **INFRA:** cmhc002 agent_timeout (0 chunks, 0.0) — the H0019 reliability bug, not a strategy miss. a's content-quality mean *excluding* it is higher than 0.448.
- **Router leak spotted from a's view:** cmhc016 a=0.00, d=0.83, natural=0.00 → router picked a, should've picked d (biggest single leak — revisit in Router pass).
- **Joint verdict A: ✅ behaving as designed.** No a-specific defect. Failures decompose cleanly into content-gap (→ ingest/d), procedural (→ b), buried (→ boost/escalate), coverage-variance (→ strategy s), + 1 infra bug.

---

## Strategy B — assemble (thematic / multi-section)
*Design intent: should WIN on overview/procedural/multi-section policy questions where the answer is
assembled across a section. Should LOSE on literal-anchor pinpoint (over-broad). Watch: does b fire
its assembly where it should, and is it penalized only where the bank wants a pinpoint?*

| q | score | must_facts | chunks | worked | didn't | joint |
|---|---|---|---|---|---|---|
<!-- fill -->

**Strategy B — running notes:**
-

---

## Strategy C — llm_validate / reverse-RAG
*Design intent: the validate/generate path (live `c`). Historically the floor (~0.19, "hallucination
engine"). Key question: is c low because it's HONESTLY limited, or because a joint is broken? Is it
ever the oracle-winner for any query?*

| q | score | must_facts | chunks | worked | didn't | joint |
|---|---|---|---|---|---|---|
<!-- fill -->

**Strategy C — running notes:**
-

---

## Strategy D — web (external)
*Design intent: should WIN on corpus-miss + crawlable-payer queries where the web has the fact.
Should FAIL on blocked payers (Aetna) and inherited-only content not on Google. Watch: is d winning
because the corpus genuinely lacks it (right reason) or because a/b under-retrieved (masking a
corpus bug)?*

| q | score | must_facts | chunks | worked | didn't | joint |
|---|---|---|---|---|---|---|
<!-- fill -->

**Strategy D — running notes:**
-

---

## Router (natural)
*Design intent: pick the best available strategy per query. Router==oracle where it picks right;
router<oracle where it mis-routes. For each query: did the router pick the oracle-winning strategy?
If not, WHY (which feature misled it)?*

| q | router score | oracle | picked | oracle-winner | mis-route cause | joint |
|---|---|---|---|---|---|---|
<!-- fill -->

**Router — running notes:**
-

---

## Cross-strategy observations (revisit at end)
*Patterns that span strategies — bank faults, corpus gaps, tagging issues, joint defects to route to
RAG/Lexicon. Filled as they emerge.*
-
