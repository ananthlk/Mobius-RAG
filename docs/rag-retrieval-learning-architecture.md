# RAG Retrieval: Scoring, Escalation, and the Learning Loop

*One architecture, read three ways — by the **ReAct runtime** (how to drive it), by the
**bandit** (what to learn), and by the **data pipeline** (what to log). Author: Eval agent,
2026-07-14. Corrections/extensions: RAG agent, 2026-07-14. Status: canonical reference for
the router re-architecture (RAG commit 5b45257 + escalation-integrated inheritance boost,
rev 00332).*

> **Implementation status notes (RAG agent, 2026-07-14):**
> - `strategy_chain`, `escalated`, `fast_exit`, `inherited_boost` fully on response object
>   (4d3b3d0 / rev 00331). **Not yet in the decision row** — persistence is per-attempt, not
>   per-query. Fix: move `_persist_routing_decision_async` after outer loop; enrich with
>   final telemetry. Pre-production; doesn't block cert.
> - `corpus_fingerprint` not yet computed or stamped. Plan: hash of
>   `rag_published_embeddings` version counter; stamp on decision row.
> - `grounding_source` not yet computed in response. Plan: post-retrieval, classify each
>   chunk via `document_id ∈ payor_inherited_authority(payer)` (set already fetched during
>   strategy_a supplemental pass).
> - **Inherited boost per-doc cap** (b155ecb / rev 00332): over-fetch `k = n_inherited_docs × 2`
>   + top-2 per doc cap before boosting, so large inherited docs (936-chunk SMMC, 26-chunk
>   1.010) can't crowd pinpoint docs (1-chunk 59G_1020). `inherited_boost.boosted_doc_ids`
>   surfaced so EVAL can confirm 88e28899 appears. **Pending cert.**
> - **Gap-based multi-invoke** (§3 "small gap → co-plausible strategies") = **Stage 3, not
>   yet built**. Argmax-only today. Don't read §3 gap paragraph as shipped.
> - **Tool-collapse** (§3 Ananth directive, 2026-07-14): Chat collapses retrieval tools to
>   `rag(query, mode)`; RAG's router internalizes the strategy choice. Chat-side wiring in
>   progress (Chat Agent). RAG-side: mode→caller_mode already accepted; strategy `s`
>   (structured-fact) and `m` (cache) added to feature vector (§2); actual router dispatch
>   wiring **not yet built**.
> - **Strategy `s` router wiring**: Payor registry API DELIVERED (GET /structured-fields,
>   GET /populated-matrix). RAG router dispatch for `s` **not yet built** (after cert).
> - **Strategy `m` (cache)**: architecture specced (§2). Store **not yet built**.
> - **Strategy `f` honest floor** (renamed from `c` to avoid collision with live `c = llm_validate`):
>   architecture specced (§2). Explicit tier **not yet built** (today's full-miss path is a bare
>   abstain). Live `c` = LLM-validate / reverse-RAG pipeline (`corpus_search_strategy_c.py`) — see
>   §2 note below.
> - **Validation ledger** (§3 Contract A): specced in detail. Engine exists
>   (`app/services/fact_checker.py` per-claim support). Per-claim surface in response
>   **not yet built** (after cert).
> - **Exploratory/overview intent** (§2, 2026-07-14): `exploratory_intent` feature + strong
>   +b weight + `multi_domain → +b`. **Not yet built.** (See §2 below for design.)

---

## 0. The whole thing in one paragraph

Every query is scored against each retrieval strategy by a **linear function of the query's
retrieval *properties*** — not its wording. The highest score **routes**; the score's magnitude
**is the confidence**; the **gap** to the next score decides single- vs multi-strategy. Retrieval
runs, the result is **assessed**, and the system **escalates** to a *materially different* attempt
only when one exists — otherwise it **fast-exits** with the best answer so far rather than burning
a redundant pass. Every decision and its outcome are **logged**; the outcome is the **reward** that
trains the weights over time. The router is the *policy*, the escalation loop is the *ReAct layer
pulled inside RAG*, and the log is the *bandit's training set*. Same structure, three reads.

```
          ┌─────────────────────────── one query ───────────────────────────┐
 features → linear score per strategy → argmax=route, magnitude=confidence,   │
          │                              gap=single-vs-multi                   │
          │        ↓ retrieve ↓                                                │
          │   assess: grounded / abstain / low-conf                           │
          │        ↓                                                           │
          │   escalate ONLY if a materially-different attempt exists          │
          │        (new strategy OR rewritten query) → else FAST-EXIT         │
          │        ↓  (plan-scoped escalation: boost inherited-authority)     │
          │   keep-best → return                                              │
          └──────────────────────────────────────────────────────────────────┘
                    │ every decision + outcome logged (Contract C)
                    ▼
            reward = canonical answer quality  ──►  bandit updates weights (Contract B)
                                                         │
                                                         ▼  new weights
                                              back into the linear score
```

---

## 1. The flow (one query, start to finish)

1. **Featurize.** Extract the retrieval-property feature vector `x` (Section 2). These are
   properties of *how the answer is retrievable*, deliberately NOT the query's linguistic shape.
2. **Score.** For each strategy `s ∈ {a,b,c,d}`: `score_s = Σ_i w_{s,i} · x_i`. All strategies
   scored by the same function over the same features → **one common scale** (this is the
   normalization; recall, confidence, and cross-strategy comparison finally live on one axis).
3. **Decide the shape of the retrieval:**
   - **Route** = `argmax_s score_s`.
   - **Confidence** = the winning score's magnitude (deterministic, no LLM).
   - **Gap** = `top - 2nd`. Large gap → single strategy. Small gap → the top strategies are
     co-plausible → **multi-invoke and union** (e.g. a∪d), quality-gated (blind union hurts).
     *(Stage 3 — not yet built; argmax-only today.)*
   - Mode gates the budget: `fast` = top-1 only; `chat` = allow 1 escalation; `thinking` = invoke
     co-plausible strategies / allow 2+.
4. **Retrieve** with the chosen strategy(ies).
5. **Assess the outcome** — three states that drive step 6:
   - `grounded` (must-facts present, confidence honest-high),
   - `abstain` (explicitly "the passages don't contain this"),
   - `low-conf` (plausible but incomplete).
6. **Escalate or fast-exit** — the anti-thrash rule:
   - Escalate **only if a materially different attempt exists**: a *different strategy* (different
     retrieval mechanism) OR a *significantly rewritten query* (different terms/scope).
   - If the only remaining option is the same strategy on the same query → **FAST-EXIT**: return
     best-so-far + honest "this is the best available." Same (strategy, query) × same corpus =
     same chunks = same answer; retrying is pure wasted latency/tokens.
   - **Plan-scoped escalation carries the inherited-authority boost:** when an abstain/low-conf
     query is scoped to a payer that *inherits* base policy (e.g. an FL Medicaid MCO → AHCA 59G),
     the retry re-ranks the payer's inherited base docs **above CSoT, for the retry only**. This is
     safe *because* confident plan answers never reach step 6 — they resolved at step 4.
7. **Keep-best** across attempts (compare assessed quality, keep the higher) and **return**, with
   the full decision trace (Section 5).

---

## 2. The feature vector (the shared language)

The features are the vocabulary all three consumers speak. Each is a property of the query's
*retrievability*, computable pre-retrieval (except where noted).

| feature | type | what it measures | source |
|---|---|---|---|
| `tag_exclusivity` | float | selectivity of the narrowest matched d-tag (small pool = pinpointable → **a**) | lexicon tag pools |
| `literal_anchor` | bool | query carries a code/ID (HCPCS `H0019`) → **a**, kills **b** | query parse |
| `corpus_depth` | float | doc coverage for the payer/domain (deep → a/b, shallow → d) | corpus stats |
| `crawlability` | float | is the payer's web fetchable? Sunshine .8 / **Aetna 0 = blocked** / Simply 0 → the **a-vs-d gate** | discovered_sources last_fetch_status |
| `thematic_policy` | bool | d-tag in PA / appeals / credentialing → answer spread across a policy section → **b** assembles, **a** fragments | lexicon d-tags |
| `wide_pool` | bool | candidate pool > ~500 → BM25 drowns → **d** | pool size |
| `inheritance` | bool | payer inherits base policy not on Google (AHCA) → keep in corpus, not d | payor_inherited_authority |
| `multi_domain` | bool | spans ≥2 major d-categories | d-tags |
| `p_tag_request_type` | enum | what the user wants (submit / verify / compliance / lookup) | p-tags |
| `factual_vs_procedural` | float | pinpoint fact vs how-to | query parse |
| `exploratory_intent` | bool | open-ended overview/catalog question ("tell me about X", "what services does X offer", "overview of") → answer is an *assembly*, not a pinpoint | query parse (trigger phrases + absence of specific code/ID) |
| `structured_field_hit` | bool | query targets a **known structured payer field** (timely_filing_days, appeal_deadline_days, provider_phone, portal_url, …) **AND that field is populated** for the resolved payer → the answer is a *canonical field read*, not a retrieval | payor_lookup / payor registry schema + non-null check |
| `cache_hit` | bool | this question (or a payer-scoped near-duplicate) was **previously answered**, the answer is **not down-rated**, and its **corpus_fingerprint is still current** → deterministic replay | answer cache + rating store + fingerprint check |

**Bootstrap weights** (hand-reasoned, shipped; the bandit refines them):
`m` (cache): **+cache_hit (dominant, checked first)** ·
`s` (structured): **+structured_field_hit (dominant)** ·
`a`: +exclusivity +literal +corpus_depth −thematic −wide_pool ·
`b`: +thematic +corpus_depth −literal **+exploratory_intent +multi_domain** ·
`d`: +crawlability +wide_pool −inheritance −thematic −corpus_depth ·
`f` (honest floor): fires ONLY on a full miss (all other tiers failed to ground); always answers, always labeled ungrounded. *(Note: live router `c` = llm_validate / reverse-RAG, a separate existing strategy — `f` is the planned new tier.)*

> **Strategy `s` — structured-fact lookup (Ananth directive, 2026-07-14).** The payor_lookup
> registry holds *raw authoritative facts* per payer — timely-filing windows, appeal deadlines,
> phone, portal. When a query maps to one of those fields **and the field is filled for the
> resolved payer**, that read is the canonical answer — it should **dominate the route** (nothing
> beats an authoritative field on accuracy; no BM25 gamble). Two gates, both required: (1) the
> query's request-type maps to a known field; (2) the field is *populated* for this payer — an
> empty schema slot must NOT route here, it falls through to a/b/d. This is the deterministic
> resolution to the a-vs-d unpredictability EVAL proved on timely-filing (cmhc001 Sunshine vs
> cmhc019 Aetna were feature-identical yet routed oppositely, because we were *retrieving* a fact
> that is actually a *structured field*). If `structured_field_hit` fires, read the field, and use
> corpus/`a` only to attach a citation/provenance for it.
>
> **Data dependency DELIVERED (Payor Platform, 2026-07-14):** `GET /api/registry/structured-fields`
> (field schema + the same alias map payor_lookup resolves with — so router gating and skill
> behavior agree by construction) and `GET /api/registry/populated-matrix` /
> `/payors/{payor}/populated` (gate 2: `true` = directly-set-and-present; `false`/missing = fall
> through). Answer path stays `payor_lookup` (value + provenance + citation). Live completeness:
> **Aetna 10/10, Sunshine 5/10, AHCA 3/6**; `timely_filing` filled for BOTH MCOs → EVAL's cmhc001
> (Sunshine) and cmhc019 (Aetna) are the direct certification cases for strategy `s`.
> **Inherited structured defaults = deliberate LATER CUT** (v1 = directly-set only): "filled via
> inheritance" for surface facts is out-of-scope until it's designed on the doc-inheritance pattern
> — per-field inheritability policy, Ananth-ratified, and provenance that says *inherited-from-AHCA
> explicitly*. Never serve a state default silently as a payer's value (a provider misquoting a
> plan deadline is a trust failure). RAG must NOT build router logic assuming inherited structured
> fills. **RAG router-side wiring of strategy `s`: not yet built.**

> **Strategy `m` — cached-answer replay (Ananth directive, 2026-07-14).** When a question was
> previously answered and the answer was rated positively — *or simply not down-rated* — and it is
> still fresh, resurface it. A deterministic replay: the fastest, cheapest, most consistent path.
> Three gates, all required: (1) semantic match to a prior answered question ≥ threshold, scoped by
> payer/entity; (2) rating state ∈ {positive, neutral} — a down-rated answer is NEVER replayed;
> (3) `corpus_fingerprint` still current — a corpus/lexicon revision that moves the fingerprint
> **invalidates** affected entries. The cache is *rebuildable, not authoritative* — it caches the
> **product** of retrieval, and when the corpus underneath changes, stale entries are dropped or
> cheaply re-verified on next hit (this is why `corpus_fingerprint` is already in the telemetry —
> it doubles as the cache-validity key). On a valid hit `m` is checked first and short-circuits the
> router.
>
> **The determinism thesis (why this is the core, not a nicety).** Resolution runs
> deterministic-first: **`m` (replay) → `s` (field read) → `a`/`b` (retrieve) → `d` (web) → `f`
> (honest floor)**. `m` and
> `s` are *reads* (deterministic, ~1.0, instant); `a`/`b`/`d` are *retrieval* (probabilistic). Every
> approved answer becomes a cache entry and every filled field becomes a structured hit, so **the
> deterministic layer grows with use and the probabilistic fallback shrinks** — the system converges
> toward mostly-deterministic, faster and more consistent over time, and rebuilds cleanly when the
> corpus changes. Note the tiering is *emergent from the linear scores* (dominant weights on
> `cache_hit`/`structured_field_hit`), not a hardcoded if-else cascade — a new deterministic read is
> just another feature+strategy with a dominant weight.
>
> **Rating is the shared reward — two memories, one signal.** The user's thumbs-up/no-derate both
> (a) WRITES the cache — memorize the *answer* (strategy `m`), and (b) trains the bandit — memorize
> the *strategy choice* for that query shape (Contract B). Cache = exact-answer memory; bandit =
> strategy memory. Same reward feeds both. **Cache + rating store + fingerprint invalidation: not
> yet built.**

> **Strategy `f` — honest full-miss floor (Ananth directive, 2026-07-14; renamed from `c` to avoid collision with live `c = llm_validate`).** When every grounded
> tier misses (`m`/`s`/`a`/`b`/`d` all fail to find the answer), the response is NOT a bare "I
> couldn't find it." `c` surfaces a *helpful, best-effort answer* — general/directional guidance —
> that is ALWAYS and EXPLICITLY labeled **"not found in our materials."** Not cited, not
> authoritative: general help + next steps (contact the payer, provide a doc). Its contract, and
> why it's safe:
> 1. **Fires ONLY on a full miss** — it never competes with or masks a grounded answer; it is the
>    terminal tier, reached only after the ladder is exhausted.
> 2. **Never fabricates authoritative specifics** — a deadline, a code, a rate it doesn't have.
>    Where a specific fact is missing it defers explicitly ("contact the payer for the exact
>    figure"). General framing is allowed; invented specifics are not.
> 3. **Always carries the ungrounded label** — a reader can never mistake a `c` answer for a sourced
>    fact.
> `f` is the honest floor: the user always gets *something useful*, and it is always transparent
> about what it is. This also cleanly resolves the composer over-abstain problem (§ the Sunshine
> case): grounded answers (`m`/`s`/`a`/`b`/`d`) get **cited**; a true full-miss goes to `f`,
> **labeled** — so a grounded-but-partial assembly (strategy `b`) is never demoted to "no verified
> answer," and a real miss is never dressed up as grounded. The line is: *cite it, or label it —
> never blank, never bluff.* **Not yet built as an explicit tier (today's full-miss path is a bare
> abstain).**
>
> *Note on live `c` = llm_validate: the router's existing `strategy_id="c"` is the reverse-RAG /
> LLM-validate pipeline (`corpus_search_strategy_c.py`) — generate from prior, then verify each
> citation against corpus/sitemap. It is a separate live strategy and is NOT what `f` refers to.*

> **Exploratory-intent diagnosis (EVAL, 2026-07-14):** "What services does Sunshine offer?" →
> scores a=0.504, b=0.401, d=0.37 → routes a → one-facet answer (expanded benefits only, not
> the catalog). b loses by 0.10. Root cause: `thematic_policy` only fires for PA/appeals/
> credentialing d-tags; overview/catalog questions never trigger it. Fix: `exploratory_intent`
> feature gives b the signal it needs; `multi_domain → +b` catches the case where the answer
> spans multiple d-categories (services catalog always does). **Not yet built; queued after
> inheritance GA cert.**

---

## 3. Contract A — for the ReAct runtime

**PRINCIPLE (Ananth, 2026-07-14) — ONE retrieval tool, not a menu.** ReAct must NOT be handed a
menu of search tools (`corpus_search`, `payor_lookup`, `lookup_authoritative_sources`,
`google_search`, …) to choose among. Exposing them makes ReAct do strategy selection — badly — and
it routes *around* the router (observed live: "tell me more about Sunshine Health" → ReAct picked
`payor_lookup` [errored] → `lookup_authoritative_sources` [empty] → `google_search`, and NEVER
touched the corpus that holds 573 Sunshine docs). **Collapse every retrieval/search tool into a
single `rag(query, mode)` call.** The former tools become RAG-internal *strategies*: `google_search`
→ strategy **d**; `payor_lookup` / `lookup_authoritative_sources` → an internal registry/corpus
capability the router invokes; corpus a/b → internal. ReAct picks the **mode** (effort/latency
budget), never the strategy. Action tools (task creation, etc.) stay on ReAct — this collapse is
retrieval-only. This is what makes the router load-bearing: it can only "figure out the strategy" if
the caller actually delegates the choice to it.

**What ReAct passes in:** the query, the caller `mode` (fast/chat/thinking), and — if this is a
*re-ask* — the prior decision trace. **Not** a tool choice.

**What ReAct reads back (the decision trace):**
- `strategy_chain: [a, b, …]` — the ordered strategies actually invoked.
- `escalated: bool` — did the internal loop cross strategies.
- `fast_exit: {fired: bool, reason}` — did it stop because no materially-different attempt remained.
- `confidence` + `grounding` (grounded / abstain / low-conf; grounding-source doc ids).
- `answered_by` — which tier produced the text (`m`/`s`/`a`/`b`/`d`/`c`).
- **`validation_ledger`** — the per-claim breakdown (below). This is what lets ReAct *choose*.

**The validation ledger — show your work, per claim (Ananth directive, 2026-07-14).** The response
must NOT hand ReAct a trust-me blob. It decomposes the answer into claims and reports, for each,
whether it was validated and against what:
```
answered_by: "c",
answer: "…xyz…",
validation_ledger: {
  claims: [
    { claim: "x", status: "validated",   evidence: [{doc_id, snippet, tier:"a"}] },
    { claim: "y", status: "validated",   evidence: [{source_url, tier:"d"}] },
    { claim: "z", status: "unvalidated", evidence: [] }          // could not confirm
  ],
  validated: 2, total: 3
}
```
Rendered in words, this is exactly: *"`c` said xyz; we validated x and y against [these sources];
we could **not** validate z."* Every tier gets a ledger, but it matters most for `c` — it upgrades
the honest floor from a blanket "not found" to "here's general help, and of it x/y actually check
out against [evidence], z does not." **What ReAct does with it:** use the validated claims, treat
the unvalidated ones as gaps — rewrite the query to chase `z`, drop it, or present it under the
label. ReAct decides *whether and how* to use the answer from the ledger, not from a scalar.
**The LLM boundary (Ananth):** routing stays deterministic (structural features, §2); *claim
decomposition + validation of a generated answer is the legitimate LLM-critique job* — that's the
fact-checker (`app/services/fact_checker.py`, per-must-fact support), now surfaced into the response
instead of collapsed to a scalar score. It also feeds chat's Citations/Corrections tabs directly.

**The division of labor — do not duplicate the loop:**
- **RAG's internal loop owns cross-strategy retry.** By the time ReAct gets a result, RAG has
  already tried the strategies worth trying and kept the best. ReAct must **not** re-invoke the
  same query hoping a different strategy fires — RAG already did that.
- **ReAct owns query *reformulation* and *critique*.** ReAct's job is to decide whether a
  *materially rewritten* query is worth a fresh call, or whether to hand off to the critique layer,
  or to stop and present the honest best-so-far.

**The anti-thrash rule (the load-bearing contract):**
> If RAG returns `fast_exit.fired = true`, re-firing the **same query** will **not** improve the
> answer — the corpus is deterministic. ReAct must either (a) rewrite the query *materially*
> (different entities/scope/terms) and call again, or (b) stop and surface the best-so-far. It must
> **never** re-issue a not-materially-different query. This is what makes the loop terminate and
> keeps latency bounded.

The escalation loop *is* the ReAct layer, pulled inside RAG so chat's outer ReAct doesn't have to
burn a second network round-trip to get cross-strategy behavior. Chat's ReAct sits *on top* for
reformulation + critique — the next layers of the arsenal.

---

## 4. Contract B — for the bandit

There are **two decision points**, so **two policies** to learn:

| policy | when | context | action |
|---|---|---|---|
| **first-pick** | pre-retrieval | feature vector `x` | which strategy(ies) to invoke |
| **escalate-or-stop** | post-retrieval | `x` + assessed outcome | escalate (which strategy) vs fast-exit |

**Reward = the canonical answer-quality metric** (one metric, applied identically everywhere):
`reward = answer-fact-recall − hallucination-penalty`, judged on the **answer text** (not fragile
passage-alignment). Make it **mode- and cost-aware** so pure recall doesn't over-pick slow `d`.
Everything else (chunk-recall, retrieval scores, self-confidence) is a **diagnostic**, not the
reward.

**Learn a *linear contextual bandit over the features* — NOT a per-`query_class` lookup.** The old
`query_class` cell collapsed 82% of queries into "conceptual" and learned coarse noise. The linear
policy is sample-efficient and shares signal across queries. The hand-set weights (Section 2) are
the **bootstrap**; production reward moves them toward the oracle.

**The content-coverage tail is memorized, not featurized.** Whether *this specific fact* is
retrievable in *our* corpus is a content property no feature predicts (two feature-identical
queries can route oppositely). The linear model is the *skeleton*; the bandit's per-cell memory
(keyed `payer × topic × strategy`, filled from reward history) is where that unpredictable tail
gets captured. Linear features generalize; per-cell memory specializes.

**Do NOT fit weights on the eval bank.** 22 pinpoint queries can't fit the weight matrix (leave-
one-out = router-level). The bank validates *structure and direction*; **production traffic fits
the weights.** (See [feedback: optimize structure not eval number].)

---

## 5. Contract C — for the data pipeline

Every retrieval logs **one decision row**; every outcome logs **one reward row**; they join on
`decision_id`. That join *is* the bandit's training example.

**Decision row (emit at return time):**
```
decision_id, ts, query, mode, caller
features            : {tag_exclusivity, literal_anchor, corpus_depth, crawlability,
                       thematic_policy, wide_pool, inheritance, multi_domain,
                       p_tag_request_type, factual_vs_procedural}
scores              : {a, b, c, d}          # per-strategy linear scores
chosen_strategy, gap, confidence
strategy_chain      : [a, b, …]             # ordered strategies invoked
escalated           : bool
fast_exit           : {fired, reason}
inherited_boost     : bool                  # did the plan-scoped retry boost inherited docs
answered_by         : m|s|a|b|d|c           # which tier produced the text
retrieved           : [{chunk_id, document_id, authority_level, d_tags, rerank_score}]
grounding_source    : inherited | plan | mixed | none   # doc-id ∈ payor_inherited_authority?
validation_ledger   : {claims:[{claim, status, evidence[]}], validated, total}
                                             # per-claim: which held, against what, which didn't
priors_version, corpus_fingerprint          # so a decision is reproducible against its world
```
Logging the ledger per-claim (not just a scalar) turns each answer into fine-grained training
signal: *which claims* a strategy reliably validates vs leaves open becomes learnable, and the
unvalidated-claim rate per (tier, topic) is a direct corpus-gap signal.

**Reward row (emit when the outcome is known — judged, human-labeled, or downstream signal):**
```
decision_id, reward, verdict (grounded|abstain|wrong), hallucination_flag,
label_source (judge|critique|human|implicit), latency_ms, cost
```

**Grain & discipline:**
- One decision row per retrieval call; the `strategy_chain` captures the whole internal loop in one
  row (don't spread escalation across rows — it breaks the trend-on-decision rule).
- `grounding_source` is computed durably: a chunk is *inherited* if its `document_id ∈
  payor_inherited_authority(payer)` — the same view the retriever uses, so labels and retrieval
  agree by construction.
- Log **what was dropped** (multi-invoke truncation, no-retry fast-exit) — silent caps read as
  "covered everything" when they didn't.
- Stamp `corpus_fingerprint` so reward is attributed to the *world it was earned in*; a corpus
  mutation mid-stream invalidates the comparison (the 00327 lesson).

---

## 6. The flywheel — how the three close the loop

```
  ReAct drives the router (Contract A)
        │ acts: route / escalate / fast-exit
        ▼
  every decision + outcome logged (Contract C)
        │ decision_id joins action → context → reward
        ▼
  bandit learns the weights + per-cell tail (Contract B)
        │ new weights
        ▼
  the linear score the router uses  ──► back to the top
```

- **ReAct** makes the structure *usable* and *terminating* (route, escalate-when-useful, fast-exit-
  when-not).
- **Data** makes it *observable* (every joint's behavior is in the decision row) and *learnable*
  (the reward join is the training set).
- **The bandit** makes it *improve* (weights climb toward the oracle; per-cell memory captures the
  content tail features can't).

Get this structure right and observable, and the win compounds as the rest of the arsenal — the
outer ReAct reformulation, the runtime critique agent — stacks on a base that already tells the
truth about what it did and why.

---

## 7. Build & Test Roadmap

**Doctrine: certify STRUCTURE, not the number.** Each phase has a *structural* test gate — does this
joint behave as designed — with the eval bank as the diagnostic, not the target. Every certifying
run happens in a clean corpus-fingerprint window (Payor holds mutations). Owners: **RAG**
(router/strategies/ledger/telemetry), **Chat** (tool-collapse/composer/outer ReAct), **Payor**
(registry data), **Feedback** (rating store), **EVAL** (every test gate + certification).

**Phase 0 — Inheritance certification · GATE for "make us available" · NOW**
- Build (RAG): kill the canned 59G trailer + inherited-authority within-set boost (per-doc cap,
  rev 00332 — shipped, pending cert).
- Test (EVAL): clean-window run — 110 + `inheritance_aetna` v2 back-to-back, one fingerprint.
  GATES: displacement/restatement **2/2** (safety, already green) AND gap_fill **≥4/5** (unlock).
  Structural asserts: county_residence (1-chunk 59G_1020) stops starving; hospice false-red gone;
  no canned trailer. **Blocks GA — everything else waits.**

**Phase 1 — Tool-collapse → router load-bearing · highest structural leverage**
- Build (Chat): ReAct search tools → one `rag(query, mode)`; ReAct picks mode, not strategy.
  (RAG): router dispatch selects the strategy internally; internalize web (`d`) + payor-lookup.
- Test (EVAL): Sunshine cases + chat-mode suite. Structural assert: ReAct issues ONE `rag()` call,
  the router (not ReAct) selects, and the corpus is actually reached — the
  payor_lookup→google→abstain failure is gone.

**Phase 2 — Deterministic reads**
- 2a `s` (structured): Build (RAG) router dispatch + gate on `/populated` (Payor API delivered).
  Test: cmhc001 (Sunshine) + cmhc019 (Aetna) timely-filing flip coin-flip → deterministic read
  ~1.0 (report before/after); empty field falls through (never blank).
- 2b `m` (cache): Build (RAG + Feedback) cache store + rating-write + fingerprint-invalidation.
  Test: replay determinism (same Q → same A on hit); invalidation fires on fingerprint change;
  down-rated never replayed.

**Phase 3 — Honesty layer**
- 3a `f` (honest floor): Build (RAG/Chat) explicit full-miss tier + cite-or-label composer.
  Test: full-miss → labeled help (not bare abstain, no fabricated specifics); grounded-partial
  keeps citations (over-abstain fixed).
- 3b validation ledger: Build (RAG) surface `fact_checker` per-claim in the response. Test: ledger
  claims match evidence on known cases; validated/unvalidated correct; feeds Citations/Corrections.

**Phase 4 — Routing refinements**
- 4a `exploratory_intent` + `multi_domain → +b`: Test — Sunshine "what services" routes `b`,
  assembles the catalog instead of `a` pinpointing one facet.
- 4b Stage-3 gap-based multi-invoke (union): Test — union-ceiling cases (cmhc009/010/013) improve;
  blind-union guard holds (cmhc018 doesn't regress).

**Phase 5 — Learning substrate (compounds; needs 1–4 emitting fields)**
- 5a telemetry: decision row per-query (not per-attempt) + `corpus_fingerprint` + `grounding_source`
  + `validation_ledger` logged. Test: every joint present and reproducible.
- 5b bandit: linear contextual bandit consuming the canonical reward. Test: shadow-mode — learned
  weight *directions* agree with the hand-set bootstrap; no live routing change until certified.

**Sequencing:** 0 blocks all · 1 makes the router load-bearing (real chat behavior) · 2 is the
highest-value deterministic win · 3 is honesty/UX · 4 refines routing · 5 is the compounding
substrate. Within each phase: build → EVAL certifies the joint → next.
