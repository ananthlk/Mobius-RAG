# Versioning & Deduplication Gate — spec

**Author/owner of spec:** Master RAG Coordinator. **Builder/owner of code:** Master RAG Coordinator (gate + back-propagation).
**Human review surface:** Fact Store. **Classification input:** Payor Platform. **Read-side:** Retriever. **DDL:** DB seat. **Miss-profile baseline:** Eval.

Status: **design, unbuilt** — 2026-08-17. Circulated for sign-off. Nothing in this spec is implemented.

Schematic (same content, visual): https://claude.ai/code/artifact/b216ac3c-6c79-44e5-b3a6-52232ab40294

---

## 0. The idea in one line

A **post-embedding gate** that decides what enters the retrieval index, what gets retired, and what a human
must adjudicate — where **retention is universal and attention is tiered**. Nothing is ever deleted; only
the tracked lane is admitted to the index, gated on completeness, and reviewed by a human.

Pipeline: `ingest → classify (Payor Platform) → extract → chunk → tag → embed → ` **`GATE`** ` → publish`

---

## 1. Where the gate sits, and why after embedding

The gate runs **after embedding, before publish**. Chunk hashes exist after chunking; vectors exist after
embedding. The gate needs hashes to make the decision and vectors to nominate candidates across thousands
of documents.

**Cost consequence (needs a ruling — see §11.1):** running after embedding means embedding documents we
will never serve. Recommendation: gate after embedding for the tracked lane, and **skip embedding entirely
for untracked documents**, which are deduplicated on `file_hash` alone.

The gate is the **only writer of `lifecycle_state`**. Everything upstream produces artifacts; the gate
alone decides which of them a query can reach.

### 1.1 Telemetry — one row per decision, written BEFORE publish

**Requirement (Ananth, 2026-08-17):** *"in the pipeline i want this telemetry produced before publish so
that i can track this too."*

The gate writes its decision record **before** any publish or index mutation. One row per document
through the gate:

| Group | Fields |
|---|---|
| identity | `document_id`, `doc_key`, `content_digest`, `prior_document_id`, `prior_digest` |
| decision | `first_version` \| `unchanged` \| `successor` \| `new_doc_key` \| `quarantined` |
| evidence | `overlap_ratio`, `chunks_total`, `chunks_changed`, `chunks_carried` |
| lane | `tracked` \| `untracked`, plus `importance` / `claimed` / `authority_level` as received |
| promotion gate | `pass` \| `fail` \| `n/a`, and which check failed |
| effect | `index_action`: `admitted` \| `none` \| `retired_predecessor`; `adjudication_emitted` + target |
| provenance | `normalization_version`, chunker `generator_id`, `chunking_config_snapshot` ref |
| timing | decision latency |

**Why before publish, not after.** If the row is written after the mutation, a failed or partial publish
leaves a decision nobody can see — precisely the failure class that produced `classified: 5259` while
changing nothing, and an import that reported success while writing zero pages. Writing the intended
action first means the record survives a failed mutation, and the two can be reconciled: a decision row
with no matching corpus change is a detectable defect rather than a silent one.

**What it makes answerable**, none of which is inferable from corpus state after the fact:

- *Is the nightly no-change path actually free?* Count `decision = unchanged AND index_action = none`.
  Pilot Case A (§12.1) is verified from this row, not from logs.
- *Is normalization drifting?* A spike in `successor` on a night with no real publications means
  boilerplate is leaking past §3 — the failure mode that would otherwise take weeks to notice.
- *Is the classifier's lane assignment sane?* Distribution of `lane` against `importance` over time.
- *How big is the human backlog, and is it draining?* `adjudication_emitted` count vs verdicts returned.
- *Did a chunker change cause a mass version event?* `generator_id` + `normalization_version` on every row.

This is the same principle already ratified for query telemetry: one row that serves the cockpit, the
trace, and the training set at once, rather than three partial records reconstructed later.

---

## 2. Three identities, three jobs

Most failure modes in this design come from collapsing these into one hash.

| Identity | Computed from | Answers | Must NOT be used for |
|---|---|---|---|
| `doc_key` | payer + jurisdiction + doc_type + rule/policy number | "Which documents are versions of each other?" | Anything byte-derived — it must survive a full rewrite |
| `content_digest` | ordered digest of normalized `chunk_sha` values | "Did this document change at all?" | Linking versions — it differs between them by definition |
| `chunk_sha` | normalized chunk text | "Which parts changed, what can we reuse?" | Detecting versions alone — chunk boundaries move when the chunker changes |

### 2.1 Why `chunk_sha` cannot detect versions on its own

Chunk boundaries are a function of chunker config, not the document. `ChunkingJob` already carries
`chunking_config_snapshot` and `generator_id` (`A`|`B`) — **two generators run today**. Changing chunk
size, overlap, or generator changes every hash in the corpus with zero content change. Under a naive
"any chunk differs → new version" rule, a chunker upgrade mass-deactivates the corpus and forces a full
re-embed.

Version identity must therefore be **chunking-independent**: `content_digest` is an ordered digest of
normalized chunk hashes, and a re-chunk with identical normalized text must produce an identical digest.
If the chunker changes, `content_digest` is recomputed for all documents in one migration and compared
against the stored value — a corpus-wide no-op, not a corpus-wide version event.

---

## 3. Normalization spec (load-bearing)

`chunk_sha` is only as good as what is hashed. Before hashing:

- collapse whitespace runs; strip trailing whitespace; normalize line endings
- normalize unicode: smart quotes, non-breaking spaces, PDF ligatures, soft hyphens
- strip page headers, footers, page numbers, "printed on" / "last updated" lines
- strip nav chrome and boilerplate (menus are class-marked `<div>`s, not `<nav>`)
- drop extraction artifacts that vary by extractor version

**Domain caution:** normalize prose but **preserve tabular structure**. In a fee schedule, column spacing
is data — naive whitespace collapsing merges columns and silently changes meaning.

**Why this is the load-bearing component:** without it, nightly boilerplate drift on an HTML-sourced page
produces a new version every night, forever — nightly deactivation and re-embed of the corpus, and a
version chain hundreds deep by year end, none of it real.

With normalization correct, the strict rule is right as stated: **any surviving chunk difference is a real
change to a policy and becomes a new version** — including a typo fix, because "what did it say on date X"
must be answerable precisely.

---

## 4. The version decision

```
on document D arriving at the gate (chunked, embedded):

  digest := ordered_hash(normalized chunk_sha[])
  prior  := latest active document with same doc_key

  if prior is NULL:
      → v1. FIRST VERSION. no lineage to resolve.

  else if digest == prior.content_digest:
      → bump prior.last_validated_at = now()
      → NO new version. NOTHING retired. D is discarded as a re-observation.
         (freshness, not recertification — mirrors Fact Store reverify semantics)

  else:
      overlap := |chunk_sha(D) ∩ chunk_sha(prior)| / |chunk_sha(D) ∪ chunk_sha(prior)|

      if overlap >= τ_high:            → SUCCESSOR. v(n+1) of this doc_key.
      else if overlap ≈ 0:             → AMBIGUOUS TAIL. resolve with vector centroid
                                          similarity + rule-number match:
                                            · same policy, rewritten → SUCCESSOR
                                            · unrelated             → NEW doc_key, not a version
      else:                            → SUCCESSOR (partial revision)

  then apply the lane rules in §5.
```

`τ_high` is unset pending the phase-4 measurement in §8 — it must be calibrated against real
version pairs, not guessed.

### 4.1 Hashes decide; vectors only nominate

High chunk overlap is decisive evidence of a successor. **Zero overlap is genuinely ambiguous** — an
unrelated document and a total rewrite look identical to a hash — and that is the only place similarity
gets a vote.

Similarity must not be the primary signal. Policies in one family are near-identical by construction: the
2023 and 2024 editions of a coverage policy may sit at 0.98 cosine because only rates moved, while two
genuinely different rules sit at 0.95 because they share definitions and appeals boilerplate. Similarity
conflates *same document, new version* with *sibling document*, and those demand opposite actions.

---

## 5. Tracked and untracked lanes

`tracked := importance ∈ {critical, standard} OR claimed = true OR tripwire fired (§5.2)`

Both lanes keep every version. **Nothing is ever deleted.** What differs is admission and scrutiny.

| Behaviour | Tracked | Untracked |
|---|---|---|
| Version history retained | yes | yes — retire, never delete |
| Raw doc + pages + chunk hashes kept | yes | yes |
| In the active vector index | yes, current version only | **never** |
| Embeddings | computed; dropped from index on retirement, recomputable | never computed |
| Promotion gate | yes — completeness vs predecessor | none, promote silently |
| Human review | `critical` changes only | never |
| Deduplication | chunk-level delta + canonical pick | `file_hash` only |
| Failure alarm | `alert_on_failure` from Payor Platform | silent |

### 5.1 The promotion gate (tracked only)

A new version is **not** promoted until it passes a completeness check against its predecessor:
page count, total text length, and chunk count within tolerance. A large unexplained shrink means a
truncated scrape (timeout, bot-wall, robots flip), not a policy rewrite.

- **pass** → promote; predecessor moves to `retired`
- **fail** → new version `quarantined`; **predecessor stays live**; alert if `critical`

Precedent for why this is required, all in-house: an import that wrote a document with **zero pages**,
and 151 stale `blocked` chunking jobs that happened not to matter. Without this gate, a truncated
re-scrape displaces a good fee schedule and nothing fires.

### 5.2 The risk this tiering creates, and the tripwire

A **missed `critical`** is treated as a nav page — no gate, no review, no alarm, because the whole point
of the untracked lane is that nothing fires. The error directions are wildly asymmetric:

- **false `critical`** → one wasted human diff. Cheap, self-limiting.
- **false `low`** → silent lineage loss on a document that mattered. No alarm, because none is wired.

Therefore **`importance` must be recall-biased** — over-claim rather than under-claim, and let review
absorb the false positives. Same principle already applied to the PHI classifier.

**Independent tripwire (cheap, no model):** if a document's text carries rule-number patterns,
`"supersedes"`, or effective-date language, treat it as at least `standard` regardless of assigned
importance. This catches the case where the classifier saw a placeholder filename or a bad URL and
shrugged.

---

## 6. Two clocks — the failure that produces confidently wrong answers

Retiring on a new hash is a statement about **our knowledge**, not about the policy.

| Stamp | Meaning | Set by |
|---|---|---|
| `retired_at` | transaction time — when we stopped believing this was current | the gate, automatically, on new `content_digest` |
| `termination_date` | valid time — when the policy actually stopped being in force | a human or a high-confidence extraction, **never the gate** |

```
on a new content_digest:
  set  retired_at       = now()     -- automatic, free, always correct
  keep termination_date = NULL      -- unknown until someone reads the document
```

**The failure case.** AHCA publishes v2 effective 2026-07-01. We crawl it 2026-08-17. If retirement writes
`termination_date = today`, v1 looks valid through 17 August, and every July date-of-service query is
answered from the superseded policy with full confidence — a six-week blind window.

`NULL` is the honest answer and it makes the gap findable. *"We don't know when this stopped applying"* is
a queryable backlog; *"it applied through August"* is a wrong answer nobody can detect.

---

## 7. The human loop — Fact Store owns the surface

RAG detects and mutates. **Fact Store is the surface a human works on.** The gate emits an adjudication
request; a person resolves it in Fact Store; the verdict flows back to RAG.

**What the gate sends:**
- the two versions and the rendered **diff** — which chunks changed, not the whole document
- candidate effective dates extracted from the text, with confidence
- any self-declared supersession found in the body (*"this policy supersedes the policy dated X"*) — the
  highest-value field for lineage, and usually extractable
- the `doc_key` the gate inferred, so a human can reject a bad linkage

**What a human returns:**
- successor / not a successor / unrelated document
- the valid-time dates — **this is what finally writes `termination_date`**
- promote, or keep the predecessor live

### 7.1 Two properties that decide whether this works

**Send decisions, not documents.** A reviewer sees *"these two differ in 3 of 180 chunks; the new one says
effective 10/1/2024 — successor?"* and answers in seconds. A queue of documents to read becomes furniture;
the 366 provenance-unknown docs are the in-house proof.

**Key the verdict on `content_digest`, not `document_id`.** Otherwise tonight's re-crawl recomputes it
away. Append-only, so a re-adjudication writes a new row and a wrong verdict stays diagnosable — the same
shape Payor Platform already uses for classifications.

### 7.2 Extraction before escalation

Dates need reading, but we already have a reader: extraction with a critique pass, prompt versioning and
retry. AHCA date language is heavily patterned. Route by confidence: high-confidence extraction with
critique agreement goes automatic; low confidence or conflicting dates goes to a human. **Humans
adjudicate disagreement, not the unambiguous 90%.**

---

## 8. Back-propagation over the existing corpus

The gate governs only what arrives next. The ~5,259 AHCA documents already in the corpus have no lineage,
and duplicates are already indexed and competing in retrieval.

| # | Phase | What it does | Risk |
|---|---|---|---|
| 1 | Backfill `chunk_sha` | Add column; compute normalized hashes over existing chunks. Deterministic, no model calls. | none — additive |
| 2 | Compute `content_digest` | Ordered digest per document. | none — additive |
| 3 | Exact-duplicate sweep | Identical digests across rows = same document at several URLs. Canonical picked by `authority_level` — authority is a property of origin. | low; retire not delete, reversible |
| 4 | **Cluster by candidate `doc_key`** | Group by payer + jurisdiction + doc_type + rule number. | **HIGH** — see §8.1 |
| 5 | Order each cluster | By extracted effective date where known; else `created_at` as a weak proxy, explicitly flagged. | medium — ingest order ≠ publication order |
| 6 | Mark active, retire the rest | Latest in chain active; rest retired with `retired_at` only. | must not write `termination_date` |
| 7 | Emit ambiguous tail | Critical clusters with unresolved ordering → Fact Store as a bounded queue. | queue size is the real output; measure before promising a date |

**Every phase runs dry first** — reports what it *would* change, mutates nothing until the diff is
reviewed. Not ceremony: this fleet has shipped a sweep that logged `classified: 5259` while changing
nothing, an import that wrote a document with zero pages, and a run that classified two unrelated
documents — all three reporting success.

### 8.1 Phase 4 is the riskiest assumption, and it is testable today

A wrong `doc_key` either orphans every version (corpus fills with unlinked v1s) or wrongly links unrelated
documents (v2 deactivates something it shouldn't). This is measurable **before any code is written**:

> Cluster the existing 5,259 documents by candidate key. Report: singleton count, clusters containing
> obvious siblings that must not be linked, and known version pairs that landed together.

That number decides whether this design is viable or needs a different key. **Recommended as the first
action after sign-off**, ahead of implementation.

---

## 9. Schema deltas

| Table | Column | Note |
|---|---|---|
| `documents` | `doc_key` | lineage key; indexed; nullable until phase 4 resolves |
| | `version_no` | ordinal within chain |
| | `content_digest` | ordered digest of chunk hashes |
| | `supersedes_id` | FK to predecessor |
| | `lifecycle_state` | `active` \| `retired` \| `quarantined` \| `shelved` |
| | `retired_at` | transaction time — never `termination_date` |
| | `last_validated_at` | bumped when a re-crawl agrees |
| `hierarchical_chunks` | `chunk_sha` | normalized chunk hash |
| | `carried_from_chunk_id` | delta reuse lineage; makes "what changed" answerable |

**Do not overload `documents.status`** — it is pipeline state (`uploaded → extracting → completed`).
Lifecycle is orthogonal. Conflate them and `completed` can no longer tell you whether a document is
servable.

**`documents.file_hash` is UNIQUE today.** A byte-identical re-scrape cannot insert — it collides. That
collision should become a `last_validated_at` bump rather than an error. **This is the cheapest useful
piece of the whole design and is shippable on its own**, independent of everything above.

---

## 10. Retrieval contract (Retriever)

The filter is **not** "exclude inactive". It is **resolve as-of a date**:

- default as-of = today → current version only
- an appeal carries a **date of service** and must reach the version valid on that date
- superseded ≠ irrelevant — you appeal a 2024 denial under 2024 rules

Untracked/`shelved` documents are excluded from retrieval in all cases (subject to §11.1).

### 10.1 Retriever's review (2026-08-17) — sign off on §10, with one technical requirement

**Sign off on the design.** The index filter on `(doc_key, lifecycle_state)` is right, and "resolve as-of a
date" rather than "exclude inactive" is the correct framing — I hit this exact gap tonight from the other
direction and had to hand-roll a stopgap for it.

**Direct connection to tonight's work, worth being explicit about:** the AHCA pilot surfaced a live
59G-4.130 collision — two real documents, 2016 and 2024, ranked 0.0003 apart in `rerank_score` (a
production coin-flip). Without `doc_key`/`lifecycle_state`/`content_digest` existing yet, I shipped a
`filler_a.py`-level tiebreak: a bounded penalty applied only when two pool candidates share a rule-number
identifier extracted from filename, gated on whether the query carries an explicit year (Crawler caught my
first cut being query-blind — flat penalty, not an actual date check — fixed and re-verified). It works,
but it's a **regex-on-raw-query-text approximation of "does this query carry a date of service"**, applied
as a post-hoc score nudge after the pool is already built from both versions.

**Once §10 ships, that tiebreak should be retired, not layered on top of the real contract.** Filtering to
the as-of version belongs in Pool's candidate query (`WHERE lifecycle_state = 'active'` or an explicit
date-range predicate against `effective_date`/`termination_date`), not as a Filler-stage score adjustment
on a pool that already contains both versions. My stopgap is doing Pool's job badly because Pool doesn't
have the data yet.

**The technical requirement this creates, for whoever builds the caller side of §10:** "an appeal carries a
date of service" needs `as_of_date` to arrive at Pool as a **structured parameter**, threaded from
Gate/Structure (or wherever the caller's date-of-service context lives), not inferred by regexing the raw
query text for a 4-digit year. My tonight's heuristic is a real, deployed, honest stopgap — not a
foundation to build the real contract on. Happy to own the Pool-side query change once `doc_key`/
`lifecycle_state`/effective-date-range exist; the caller-side "does this turn carry a date of service"
extraction is a Gate/Structure concern, not mine, and should be scoped explicitly rather than assumed.

**Also flagging, not blocking:** the composite index proposed in §11.4 is `(doc_key, lifecycle_state)` —
for the as-of-date case specifically (not just "current version"), a range predicate against
`effective_date`/`termination_date` will also need to hit an index, or a date-of-service query on a large
`doc_key` cluster does a filtered scan. Worth DB seat confirming whether `(doc_key, lifecycle_state)` alone
covers this or a third column belongs in the index.

**On §7/§11.5 (misrouted to this session):** the sign-off request that reached me asked me to review and
flip §7/§11.5 — those are Fact Store's row per this spec's own header (`Human review surface: Fact Store`)
and the §13 ledger. I'm Retriever (`Read-side: Retriever`, §10 is my row). Not signing a row that isn't
mine; flagging so the request can be routed to Fact Store's actual session rather than silently answered
by the wrong party.

**Flipping my own row: §10 signed off**, contingent on the `as_of_date` structured-parameter requirement
above being scoped to whoever owns Gate/Structure's caller-context extraction, not assumed to already
exist.

---

## 11. Open questions requiring a ruling

### 11.1 Payor Platform — exclusion vs ranking floor (**conflict**)

Contract v2 states junk and non-authoritative sources land on the ratified floor (`fyi_not_citable`)
*"so they rank last instead of disappearing."* This spec states untracked documents never enter the index.
**These conflict** — one is a ranking penalty, the other is exclusion. Joint ruling needed.

### 11.2 Payor Platform — is `importance` a property of the document or the version?

A page that is `low` today can become a policy landing page tomorrow, arriving with no history. Acceptable,
but it should be decided rather than discovered.

### 11.3 Eval — miss profile of the importance classifier

"Misses quite a few" splits into two different problems. A false *unknown* merely sizes the human queue.
A false *low* is silent and permanent under §5. Baseline both directions before we build around the gap.

### 11.4 DB seat — column contract and index strategy

Eight new fields; index on (`doc_key`, `lifecycle_state`) — every retrieval query filters on the pair.

**Raised by Retriever in §10.1, needs your ruling:** `(doc_key, lifecycle_state)` covers the
"current version" case. It may **not** cover the as-of-date case — a date-of-service query runs a range
predicate against `effective_date` / `termination_date`, and on a large `doc_key` cluster that becomes a
filtered scan. Confirm whether the pair suffices or a third column belongs in the index.

### 11.5 Fact Store — review surface shape

Diff view, verdict buttons, append-only verdict store keyed on `content_digest`.

---

## 12. Build plan and pilot

Ordered. Each step is independently useful and independently revertible.

| Step | Deliverable | Gate to proceed |
|---|---|---|
| 0 | **Phase-4 clustering measurement** (§8.1), read-only | cluster quality acceptable |
| 1 | `file_hash` collision → `last_validated_at` bump | ships alone, no dependencies |
| 2 | Normalization module + `chunk_sha` column + backfill | digest stable across a re-chunk |
| 3 | `content_digest` + lineage columns (DDL from DB seat) | §11.4 signed off |
| 4 | Gate: decision logic + lane rules, dry-run only | pilot below passes |
| 5 | Promotion gate + quarantine | — |
| 6 | Fact Store emit + verdict ingest | §11.5 signed off |
| 7 | Retriever as-of contract | §10 signed off |
| 8 | Back-propagation phases 1–7, each dry-run first | — |

### 12.1 Pilot — 1–2 documents, before any corpus-wide run

Two cases prove the mechanism end to end:

**Case A — the no-change path.** Re-ingest a document already in the corpus, unmodified.
- expect: `content_digest` identical → `last_validated_at` bumped → **no new version, nothing retired,
  no re-embed, index unchanged**
- this is the case that fires nightly, so it must be exactly free

**Case B — the real revision path.** Take a document with a known successor (a policy with two editions),
ingest the older, then the newer.
- expect: overlap above threshold → successor detected → `version_no` 2, `supersedes_id` set
- expect: predecessor `lifecycle_state = retired`, `retired_at` stamped,
  **`termination_date` still NULL**
- expect: unchanged chunks carried forward via `carried_from_chunk_id`, only changed chunks re-embedded
- expect: if `critical`, one adjudication request emitted to Fact Store carrying the diff
- expect: Retriever as-of today returns v2; as-of a date before v2's effective date returns v1

**Acceptance:** both cases produce the expected state transitions, and the Case A path performs **zero**
writes to the vector index. If Case A is not free, the nightly pipeline will thrash.

---

## 13. Sign-off ledger

| Seat | Scope of review | Status |
|---|---|---|
| Payor Platform | §11.1 exclusion vs floor · §11.2 importance grain · §5.2 recall bias | ⬜ |
| Fact Store | §7 human loop · §11.5 review surface · verdict store shape | ✅ (Eval/Fact Store, 2026-08-17) — signed w/ 2 additive refinements: (1) §7 returns TWO separate valid-time dates (successor.effective_date + predecessor.termination_date), never auto-derive one from the other, NULL if boundary unstated; (2) §11.5 verdict keyed on the DIGEST PAIR (pred+succ content_digest)+doc_key, not a lone digest. Q1: verdict store mine (classifications shape). Q2: RAG pre-renders diff, I display. Q3: priority-triaged not paginated; queue count gated on §12 step 0. |
| Retriever | §10 as-of contract · index filter on (`doc_key`, `lifecycle_state`) | ✅ signed, see §10.1 — contingent on `as_of_date` being a structured caller param, not query-text regex |
| Eval | §11.3 classifier miss profile baseline · τ_high calibration | ⬜ |
| DB seat | §9 schema deltas · §11.4 column contract + index strategy | ⬜ |
| Maintaining | §3 normalization vs coherence gate · `last_validated_at` freshness overlap · nightly-sweep interaction | ⬜ |
| Technical Review | structure + seam ownership | ⬜ |

Nothing in §12 beyond step 0 begins before the seats covering that step have signed.
