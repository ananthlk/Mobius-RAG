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
> - **Tool-collapse** (§3 Ananth directive, 2026-07-14): internalize payor-lookup as an
>   internal registry strategy. `CorpusSearchAgentRequest` already accepts `mode`
>   (fast/chat/thinking) — the collapse is a Chat-side change (expose only `rag(query, mode)`)
>   + a RAG-side routing addition (structured payer facts strategy). **Not yet built.**
> - **Exploratory/overview intent** (§2, 2026-07-14): `exploratory_intent` feature + strong
>   +b weight + `multi_domain → +b`. Fixes Sunshine overview → a pinpoints instead of b
>   assembling. **Not yet built.** (See §2 below for design.)

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

**Bootstrap weights** (hand-reasoned, shipped; the bandit refines them):
`a`: +exclusivity +literal +corpus_depth −thematic −wide_pool ·
`b`: +thematic +corpus_depth −literal **+exploratory_intent +multi_domain** ·
`d`: +crawlability +wide_pool −inheritance −thematic −corpus_depth ·
`c`: low bias.

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
retrieved           : [{chunk_id, document_id, authority_level, d_tags, rerank_score}]
grounding_source    : inherited | plan | mixed | none   # doc-id ∈ payor_inherited_authority?
priors_version, corpus_fingerprint          # so a decision is reproducible against its world
```

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
