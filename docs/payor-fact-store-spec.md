# Payor Fact Store — Phase 1 spec

**Author/owner of spec:** Eval agent. **Builder/owner of code+DB+API:** Payor platform agent.
**Read-side:** RAG agent (strategy `s`/`m`). **Storage/telemetry provisioning:** DB agent.
Status: Phase 1 design, 2026-07-18. Supersedes the per-field RPC model (`payor_lookup {payor, field}`).

## 0. The idea in one line
A **certified fact store** — the corpus's high-authority twin — **queried, not asked field-by-field**.
Facts carry lexicon tags + a vector; a query gates on the payer tag, blends tags+vector to a shortlist,
and either **serves** (deterministic, cite-locked, fast-exit) or **falls through** to corpus retrieval.
Grows by **rows**, never by endpoints.

Diagram: query → tagger+embedder → store → **hard gate (payer tag)** → **blend (α·tag + β·vec × scope × authority × freshness)** → top-k → `≥τ ? serve : fall through`.

---

## 1. The fact record (`payor_fact`)
```
payor_fact
  fact_id            uuid pk
  -- STORE/SERVE key (exact; idempotent upsert; disambiguates the shortlist)
  payer_key          text        -- canonical triple: payer|state|program (e.g. Aetna|FL|Medicaid)
  predicate          text        -- stable slug: claims_fax | telehealth_covered | pa_required:H2019
  record_type        text        -- 'atomic' | 'qa'
  -- VALUE
  value              jsonb       -- scalar or structured
  answer_text        text        -- rendered answer (qa) / human-readable value (atomic)
  question           text        -- qa only: canonical question form
  -- QUERY SURFACE (how it's found)
  d_tags             text[]      -- domain tags (lexicon)
  p_tags             text[]
  j_tags             text[]      -- INCLUDES the payer identity tag → the gate key
  embedding          vector(1536) -- qa: of question; atomic: of predicate+value. LOCKED 1536 to match corpus.
  -- SCOPE / COMPLIANCE  (migration-005 semantics)
  scope              text        -- NULL = unrestricted; else served only when query intent matches
  -- PROVENANCE
  source_ref         jsonb       -- {doc_id, url, page, quote}
  authority_level    text        -- contract_source_of_truth | payer_website | operational_suggested | ...
  -- FRESHNESS
  effective_date     date
  valid_until        date
  ttl_days           int
  last_verified_at   timestamptz
  verified_via       text        -- rag_probe | web | human | eval_cert
  confidence         text        -- high | medium | low
  -- CERTIFICATION (Eval owns)
  retrieval_grade    numeric     -- fact present in cited source
  synthesis_grade    numeric     -- qa: answer grounded in cited facts (atomic: NULL)
  cert_status        text        -- accepted | candidate | rejected | stale
  cert_run_id        uuid
  fact_checker_version text
  created_at, updated_at timestamptz
  UNIQUE (payer_key, predicate)
```
**Two record types, one table:** `atomic` = structured facts (exact-served); `qa` = certified common Q&A (the nightly output). Same query path, different index emphasis (exact key vs vector).

**Indexes (DB agent, pruned to the query path):**
- `UNIQUE(payer_key, predicate)` — already the serve btree; no separate index.
- **GIN on `j_tags` partial `WHERE cert_status='accepted'`** — the gate index (gate only serves accepted rows).
- **HNSW on `embedding` partial `WHERE cert_status='accepted'`** (HNSW over ivfflat = corpus standard, no training step).
- `d_tags`/`p_tags` GINs **DEFERRED** — `tag_overlap` is computed in-memory on the gated shortlist (dozens–hundreds of rows/payer at Phase-1 scale); add only when a measured query needs them.
- `embedding` type pinned `vector(1536)` in DDL with a comment citing the output_dimensionality gotcha — enforced in the type, not remembered.

---

## 2. `fact_query` — the single query interface
```
POST /api/skills/v1/fact_query
request {
  query:        str            -- raw NL (for embedding + telemetry)
  d_tags,p_tags,j_tags: str[]  -- OPTIONAL: caller-precomputed (RAG passes its query_profile → no double work)
  embedding:    float[]        -- OPTIONAL: caller-precomputed
  payer_key:    str | null     -- OPTIONAL explicit payer; else inferred from j_tags
  intent_scope: str | null     -- for scope gating
  k:            int = 5
  tau:          float = <default>
}
response {
  hit:        bool                     -- top ≥ τ AND accepted AND in-scope
  served:     { record_type, predicate, answer_text, value, source_ref, authority_level,
                freshness{last_verified_at, valid_until, stale}, cert{status, grades}, score } | null
  shortlist:  [ {predicate, record_type, answer_text, score, tag_overlap, vec_sim, scope_ok, cert_status} ]
  gate:       { payer_key, applied: bool, excluded_n }
  blend:      { alpha, beta, tau, version }
  telemetry_id: uuid           -- the fact_query_decision row
}
```
If tags/embedding are absent the service computes them (self-contained); RAG SHOULD pass its profile.

### 2.1 Gate — hard, on identity only
Keep rows where `fact.j_tags ⊇ {query payer tag}` when the payer is known. **Never cross payers.**
Identity tag only — not all tags (all-tags-must-have kills recall). Payer unknown → skip gate, require higher τ (ungated serve is lower-trust).

### 2.2 Blend — soft, on topic + vector
```
tag_overlap = weighted Jaccard( query{d,p,j} , fact{d,p,j} )     -- topic match (precision on known vocab)
vec_sim     = cosine( query_embedding , fact.embedding )         -- recall on novel phrasing
base        = α·tag_overlap + β·vec_sim          (α+β=1; start 0.5/0.5)
score       = base × scope_ok × authority_mult × freshness_mult
```
- `scope_ok` ∈ {0,1} — out-of-scope → 0 (drops out, falls through as if empty).
- `authority_mult` — contract_source_of_truth 1.0, payer_website 0.85, operational_suggested 0.65.
- `freshness_mult` — 1.0 fresh; linear decay in grace window; **past `valid_until` → stale → never served** (→ nightly re-cert queue).
- **hit** iff `top.score ≥ τ` AND `top.cert_status='accepted'` AND `scope_ok`. Else `hit=false` → caller falls through to a/b/c/d.

Sibling of the existing reranker weighted formula — reuse that shape.

---

## 3. Telemetry storage — `fact_query_decision` (the "one row" analog)
Mirrors `rag_query_decisions`: **one row per fact_query call** = audit of what was served + the training row for tuning α/β/τ.
```
fact_query_decision
  telemetry_id     uuid pk           -- client uuid4; ON CONFLICT DO NOTHING (inherits rag_query_decisions write-path)
  -- JOIN KEYS to rag_query_decisions (DB §3.1 — without these, co-location buys nothing)
  correlation_id   text              -- prod join key
  eval_run_id      uuid  NULL REFERENCES rag_eval_runs(id) ON DELETE CASCADE   -- eval join key
  query_id         text  NULL        -- eval-bank id (e.g. cmhc001)
  query            text
  query_d/p/j_tags text[]
  query_embedding  vector(1536) NULL  -- NULLABLE: eval-rows-only / sampled 1-in-N; sweeps recompute from `query` (no 6KB on every prod row)
  corpus_version   int               -- stamp on fall-through rows → attribute fall-through-rate to corpus mutations
  payer_key        text
  gate_applied     bool
  gate_excluded_n  int
  shortlist        jsonb       -- candidates + component scores (tag_overlap, vec_sim, mults)
  served_fact_id   uuid | null
  served_predicate text | null
  served_score     numeric | null
  hit              bool
  fell_through     bool        -- routed to corpus instead
  alpha,beta,tau   numeric
  blend_version    text
  is_prod          bool
  -- outcome (learning signal, filled async)
  user_feedback    text | null
  downstream_grade numeric | null   -- eval two-grade on the served answer
  created_at       timestamptz
```
**Owner:** Eval (schema + analysis + α/β/τ sweep). **Provisioner:** DB agent. **Writer:** payor — fire-and-forget off the serve path, shared pool, client-uuid PK + `ON CONFLICT DO NOTHING`; UPDATEs limited to the two async outcome columns (`user_feedback`, `downstream_grade`). **Indexes (DB, lean start):** `(created_at DESC)`, `(payer_key, created_at DESC)`, partial `(served_fact_id) WHERE NOT NULL`; `blend_version` btree when the sweep runs. Joins to `rag_query_decisions` via `correlation_id` (prod) / `eval_run_id`+`query_id` (eval).

---

## 4. Certification — what makes `cert_status='accepted'` (Eval owns)
- **atomic** — `source_ref` resolves AND `retrieval_grade ≥ τ_r` (fact present in cited source). Synthesis n/a. `verified_via='human'` may accept directly.
- **qa** — answer passes **synthesis grounding** (`synthesis_grade ≥ τ_s`) against cited facts/chunks AND `retrieval_grade ≥ τ_r`. Eval's two-grade QA runs it, sets `cert_status`, stamps `cert_run_id` + `fact_checker_version`.
- **stale** — past `valid_until`/TTL → `cert_status='stale'` → not served → nightly re-cert queue.
- **candidate** — proposed, not yet certified → not served.
- **Every accepted fact is also an eval-bank gold entry** (`must_facts` from `value`/`answer_text`). One artifact, two uses.

---

## 5. Phase boundaries
- **Phase 1 (this spec):** table + `fact_query` (gate/blend/τ) + seed `atomic` facts from existing payor seeds (lookup fields, PA checks) + certify them + `fact_query_decision` telemetry + RAG `s`/`m` read + fall-through + **freshness override mechanism (§8): bypass hook + `verify_and_recertify`, triggered by explicit + ttl-proximity.** **No nightly, no bandit-explore.**
- **Phase 2:** nightly job — mine top-N common queries from `rag_query_decisions`, propose answers (fact-derived or synth), two-grade certify, upsert `qa` rows w/ TTL, register as eval-bank gold.

---

## 6. Ownership (DoD per agent)
| Agent | Owns | Phase-1 deliverable |
|---|---|---|
| **Payor platform** | `payor_fact` table logic, `fact_query` API, gate/blend impl, **bypass hook + `verify_and_recertify` (§8)**, seed loaders, nightly (P2) | store + API live on dev; seeded from existing seeds; passes eval cert; explicit+ttl bypass works |
| **Eval (me)** | certification (2-grade → `cert_status`), the **`two_grade_compare` comparator (§8.1)** + drift signal, `fact_query_decision` schema + analysis, α/β/τ sweep, eval-bank-gold tie, **bandit-explore policy + reward (§8.4, with #11)** | cert service wired; comparator wired to override; τ from held-out sweep; telemetry validated |
| **RAG** | read-side: strategy `s`/`m` call `fact_query` first, pass `query_profile` (tags+embedding), fall-through wiring, **honor `bypass_fact_store`** | `s`/`m` hit store then fall through; no double embed/tag; bypass routes to live |
| **DB** | schema migration, indexes, telemetry table provisioning, placement | migration applied to shared DB; GIN+HNSW+btree; telemetry table live |

## 7. Open decisions
1. **Placement** — ✅ **RESOLVED (DB agent, 2026-07-18):** shared `mobius_rag` DB, schema `facts`, with **schema-scoped single-writer ownership** — payor's role gets DDL/DML on schema `facts` only, RAG's role gets SELECT, nobody else writes. The schema boundary = the ownership boundary (avoids the co-owned-DDL failure). One GRANT block in the migration (DB writes it). `mobius_cache` rejected (cache = disposable; a certified store is not).
2. **Embedding model/dims** — ✅ **RESOLVED (RAG, 2026-07-18):** `vector(1536)`, provider `gemini-embedding-001` (Vertex, dev default) / `text-embedding-3-small` (OpenAI fallback). Must match — RAG threads the already-computed query embedding from `corpus_search_agent.py` into `fact_query` (no double embed). Answer-cache `output_dimensionality=1536` gotcha applies if using the OpenAI SDK path (native 1536 = no-op, but set it explicitly).
3. **α/β/τ defaults** — start 0.5/0.5; τ from a held-out sweep on the eval bank. **Rec: Eval runs the sweep post-seed.**
4. **Gate when payer unknown** — skip gate + higher τ (rec) vs refuse.
5. **Explore/verify budget** — what fraction of hits may bypass-to-verify, and the drift-risk weighting. **Rec: start with explicit + ttl-proximity only (near-zero cost); bandit sets the online budget when task #11 lands.**

---

## 8. Freshness override — bypass + verify + re-certify
The store is **exploit** (serve certified fact, fast). Bypassing to live retrieval is **explore** (spend cost to re-verify, catch drift). This is the store's *active* freshness guardrail — beyond the passive "don't serve past `valid_until`".

**Mechanism ≠ policy. Build the mechanism now; policy sources plug into it.**

### 8.1 Mechanism (Phase 1) — the bypass hook must CLOSE THE LOOP
A bypass that doesn't re-certify is just a slow path (wasted cost). The override = bypass **+ compare + update**:
```
verify_and_recertify(query, stored_fact, trigger):
  live  = run live retrieval (a/b/c/d as the query would normally route)
  agree = two_grade_compare(live.answer, stored_fact)     # Eval QA: grounding + fact-match
  if agree:  CONFIRM  → last_verified_at=now, verified_via=trigger, extend valid_until, log confirm; serve fact
  else:      DRIFT    → cert_status='stale', serve LIVE this turn, enqueue re-cert, emit fact_drift event
  → write fact_query_decision.verify_outcome + .drift_detected
```

### 8.2 Policy — who triggers the bypass (escalating, same hook)
1. **explicit** — request flag `bypass_fact_store` / `verify_freshness`, or eval/admin trigger. **Phase 1.**
2. **ttl-proximity** — fact within grace of `valid_until`, or age > `ttl_days × k`. Phase 1.5 / nightly.
3. **bandit-explore** — stochastic, weighted by drift-risk (`age_since_verified × topic_volatility`) × traffic. Activates with the contextual bandit (task #11); a pluggable trigger, no rework.

### 8.3 Budget
Bypassing costs the expensive path the store exists to avoid → verification is **budgeted**, concentrated where value-of-information is highest (volatile + high-traffic + aging facts), never uniform random. The bandit owns this allocation.

### 8.4 Reward / telemetry (Eval owns) — the loop closes to certification
`confirm` = exploit was safe; `drift` = explore paid off. Both → `fact_query_decision` (`verify_outcome`, `drift_detected`) → drive `cert_status` transitions → **become the bandit's reward**. Freshness override and certification are the same loop.

### 8.5 Schema/API additions
- `fact_query` request: `bypass_fact_store: bool`, `verify_freshness: bool`.
- `verified_via` gains: `explicit_verify | scheduled | bandit_verify`.
- New event `fact_drift { fact_id, stored_value, live_answer, detected_at, trigger }`.
- `fact_query_decision` gains: `verify_outcome ∈ {confirm, drift, none}`, `drift_detected bool`, `verify_trigger`.
