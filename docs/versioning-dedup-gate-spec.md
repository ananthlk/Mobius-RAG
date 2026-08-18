# Versioning & Deduplication Gate — spec

**Author/owner of spec:** Master RAG Coordinator. **Builder/owner of code:** Master RAG Coordinator (gate + back-propagation).
**Human review surface + classification input:** Fact Store / Payor Platform (one seat). **Read-side:** Retriever. **DDL:** DB seat. **Miss-profile baseline:** Eval.

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

### 2.2 `doc_key` — redesigned after step 0 (2026-08-17)

The original key (payer + jurisdiction + doc_type + rule number) **failed on measurement**: 2.3% coverage,
and the title fallback produced a 590-document cluster. Diagnosis below; the 590 was not a normalization
bug.

#### What step 0 actually found

**a. `display_name` is a category label, not a title, for 49.4% of AHCA documents.** 590 documents share
the literal string *"AHCA — state Medicaid managed-care contract (model/plan)"*; 574 share the policy-rule
label; 484 share the forms label. 2,701 documents share a `display_name` with at least one other. That
*is* the 590 cluster — no normalization collapsed anything, the field simply holds a classification.
`filename` is far cleaner: only 200 documents share one.

**b. A rule number in the body is a CITATION, not an identity.** Reading content lifts rule-number coverage
2.3% → 8.3%, but the additional 6% are documents that merely *cite* a rule. The `59G-4.200` cluster built
this way contains `Chatsworth_at_PGA_National_-_Redacted.pdf` and
`Subgroup_Assignments_December_2024_Revised.pdf` — unrelated documents that mention the rule. Keying on
body text would have superseded live documents against each other. **Filename/title occurrence means the
document IS the rule; body occurrence means it references one.**

**c. 83.3% of the corpus is EPISODIC and needs no key at all.** Meeting minutes, notices, quarterly LIP
reports, monthly county data, dated presentations — issued once, never revised. `January_16_2013_Minutes.pdf`
will never have a version 2. Only ~16.7% is *revisable*: policies, handbooks, fee schedules, manuals —
normative documents replaced by later editions.

This is why the naive title key over-merged in the specific way it did: **stripping period tokens is
correct for revisable documents and catastrophic for episodic ones.** For a coverage policy the year marks
*which edition*; for a quarterly report the year marks *which document*. Same operation, opposite
correctness — so the revisable/episodic call must come **before** keying, not after.

**d. `source_url` is an identity, not a lineage key.** 4,092 documents carry one and all 4,092 are
distinct — zero repeats. Keying on URL yields all singletons and no version chains. It is the right signal
for *"is this the same document re-fetched"* (the §4 `last_validated_at` path), and useless for *"is this
the successor of that."*

#### The governing principle

**No key is better than a wrong key.** A NULL `doc_key` makes a document a permanent singleton — still
retrievable, simply never superseding anything. A *wrong* `doc_key` causes false supersession, which
silently removes a valid document from retrieval and leaves no gap to find. The two failure modes are not
comparable, so a key is assigned **only on positive evidence of a revisable identity**.

#### The episodic test runs FIRST *(bug found in the §12.4 pilot)*

`Notice of Proposed Rule: 59G-4.130/Home Health Visit Services` carries the rule number **in its
filename** and is *not* the rule — it is a dated announcement about it. Under the original precedence it
took tier 1, landed on the coverage policy's `doc_key`, and at 0.019 overlap the old ladder promoted it —
**a notice would have retired the policy it announced.**

Finding §2.2b said filename occurrence means identity and body occurrence means citation. That was too
strong: *citation-vs-identity is a property of the document's nature, not of where the string appears.*
So the episodic test precedes every tier — an episodic document gets **no key even when a rule number is
present in its filename.**

#### The key, in precedence order

| Tier | Evidence required | Key | AHCA coverage |
|---|---|---|---|
| 1 | rule number in **filename or title** | `(payer, jurisdiction, rule_no)` | ~2.3% |
| 2 | `asset_type` is revisable **and** filename is distinctive | `(payer, jurisdiction, asset_type, norm(filename))` — period tokens stripped | ~14% |
| 3 | linkage asserted by a human in Fact Store | explicit pair | the tail |
| — | no tier matched | **NULL — episodic**, `version_no` 1 forever, never superseded | ~83% |

**Never a key source**, each for a measured reason:
- rule number found **only** in body text → citation (finding b)
- `display_name` matching a known category label → not a title (finding a)
- `source_url` → identity without lineage (finding d)

#### The seam this creates

Tier 2 turns on **`asset_type`** — is this document normative-and-replaceable, or a record of a moment?
That is a classification call, and classification belongs to **Fact Store / Payor Platform**, who already
return `asset_type` in the contract. I should not be re-deriving it from filename regexes on my side; the
regex used in step 0 was a measurement instrument, not a proposed implementation. **Ask added to §11.2.**

#### Consequence for the design

Version tracking covers roughly a sixth of this corpus, and that is the correct answer rather than a
shortfall — most of these documents genuinely have no successors. It also sharply reduces the human queue:
adjudication is only ever requested for revisable documents, which is hundreds, not thousands.

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

### 4.2 The ladder defaults to ASK, never to PROMOTE  *(corrected after the §12.4 pilot)*

Promotion is the **destructive** branch: it retires a live document. It therefore requires positive
evidence, and ambiguity must never fall through to it. The original ladder had `else → successor`, which
meant a 0.019-overlap pair was promoted over a live policy. Corrected:

```
overlap ≥ τ_high            → SUCCESSOR            promote, retire prior
τ_low ≤ overlap < τ_high    → AMBIGUOUS_REVISION   → Fact Store. PRIOR STAYS ACTIVE.
overlap < τ_low             → AMBIGUOUS_TAIL       → Fact Store. PRIOR STAYS ACTIVE.
```

Both ambiguous branches admit the new document and leave the predecessor live. Serving two versions
briefly is recoverable; retiring the wrong one silently is not.

### 4.3 τ CANNOT be calibrated from this corpus — measured, not assumed

The earlier figure (τ_high ≈ 0.70 from 112 pairs, median 0.722) was **contaminated and is withdrawn**.
Once episodic documents are correctly excluded from keying (§2.2), **111 of those 112 pairs disappear** —
they were notice-vs-notice pairs scoring high because announcements share boilerplate, not because they
were versions of each other.

**Exactly one genuine revisable version pair exists in the whole AHCA corpus:** 59G-4.130, 2016-11-01 vs
2024-09-01, overlap **0.698**.

One data point cannot place a threshold. Consequences:

- `τ_high` and `τ_low` stay **provisional** (0.70 / 0.35) and are *safety* parameters, not tuned ones.
- The single real pair sits at 0.698 — just under τ_high. It routes to a human, which is the correct
  outcome under §4.2 and the reason the safe default matters more than the threshold.
- Versioning is a **forward-looking capability, not a back-propagation opportunity**: there is almost
  nothing historical to link. Real pairs accumulate as the crawler re-fetches over time, and τ gets
  calibrated then. **Eval's ask (§11.3) changes accordingly** — it is not "tune τ now", it is "define the
  evidence needed before τ may be moved off its safe default."

Original note retained: `τ_high` must be calibrated against real
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

### 5.0 Publishability precondition, and what to DO about each failure

**Ruling (Ananth, 2026-08-17):** *"we should also not publish… documents with no chunks, they just add
more problems"* and *"no pages — do not publish and back propagate to remove… no chunks with pages…
failed something delete everything and retrigger or whatever needs to happen."*

A document must have **at least one chunk** to be publishable, in either lane. Zero-page and zero-chunk
documents cannot be retrieved, so they add nothing — while inflating every count and passing every "did we
ingest it" check.

#### The governing principle

**Derived artifacts are disposable; the raw document is not.** Pages, chunks and embeddings are all
rebuildable from the stored raw file. So the remedy for any partial-pipeline failure is always the same
shape: **delete the derived state and re-trigger from the last good upstream artifact** — never patch
partial state in place, because partial state is what produced the failure signature in the first place.

#### Remediation matrix

| Condition | Diagnosis | Action on ingest | Back-propagation action |
|---|---|---|---|
| 0 pages | extraction produced nothing | do not publish; `shelved` | delete derived state, re-trigger **extraction** from raw |
| pages > 0, 0 chunks | chunking never ran or never completed | do not publish; `shelved` | delete any partial chunk rows, re-enqueue **chunking** (`POST /documents/{id}/chunking/start`) |
| chunks > 0, embeddings incomplete | embedding failed mid-run | do not publish; `shelved` | delete chunks **and** embeddings, re-trigger **chunk + embed** together |
| `status` failed / `has_errors` | any upstream stage failed | do not publish; `shelved` | delete **all** derived artifacts, re-trigger from raw |
| duplicate `content_digest` | redundant copy | publish canonical only | retire non-canonical **and remove its chunks from the index** |
| raw file missing | nothing to rebuild from | do not publish; `quarantined` | **human** — cannot be fixed by retry |

#### Retry budget

Re-triggering is bounded: **2 attempts**, then `quarantined` with the failure reason on the record. A
document that fails extraction three times is not a transient failure and must stop consuming pipeline
capacity — it is a defect report, and it goes to the §7 review queue rather than into an infinite retry.
Every attempt writes a §1.1 telemetry row, so "how many are we retrying, and are they converging" is
answerable rather than inferred.

#### One distinction that matters for cleanup

A zero-chunk document has **nothing in the vector index by construction** — there is no index cleanup to
do, only corpus bookkeeping and a re-trigger. A **duplicate** is the opposite: its chunks *are* indexed and
competing in retrieval right now, so retiring it requires an actual index deletion. Conflating the two
produces either a no-op that reports success or an index left dirty.

#### Measured back-propagation workload (AHCA, §12.2)

| Condition | Docs | Action |
|---|---|---|
| 0 pages | 150 | re-trigger extraction |
| pages, 0 chunks | 211 | re-enqueue chunking |
| duplicate digest | 100 | retire + delete 14,522 indexed chunks |
| **total unpublishable** | **361** | 6.6% of the payer corpus, currently counted as ingested |

Re-triggering 361 documents is the first real test of whether these failures are transient or structural.
If most succeed on retry, this was a queue problem; if most fail again, extraction has a defect that the
zero-page count has been quietly hiding.

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

## 6. Validity in time — perpetual until superseded

**Ruling (Ananth, 2026-08-17):** *"dont trust the term date… we will assume a doc is perpetually valid
until we get another revision or the doc retires. until we read the doc there is no way to set the term
date."*

A document is **valid from its effective date onward, indefinitely**, until one of two things happens: a
newer revision supersedes it, or it is explicitly retired. We do not invent an end date at ingestion.

| Stamp | Meaning | Set by |
|---|---|---|
| `effective_date` | start of validity | the document itself |
| `termination_date` | **explicit** end of validity, stated by the authority | a human, or a high-confidence extraction, **never the gate and never ingestion** |
| `retired_at` | transaction time — when *we* stopped believing it current | the gate, automatically, on a new `content_digest` |

### 6.1 The validity window is DERIVED, not stored

```
valid_window(v) = [ v.effective_date ,
                    COALESCE( v.termination_date,              -- explicit, if ever read
                              successor(v).effective_date,     -- implicit: the next edition
                              +infinity ) )                    -- still the current edition
```

`termination_date IS NULL` is the **normal and correct** state. It means *open-ended*, not *unknown*.

**Why derived beats stored.** A derived bound cannot contradict the chain — it is computed from the
successor's start, so the two can never disagree. A stored bound can, and silently. It also degrades
honestly: with a single version the answer is "valid from X onward," which is true, rather than a
fabricated window. `termination_date` exists as an **override** for the case where the authority states
an end date explicitly (a policy that self-terminates without a replacement), which is the only situation
the derived rule cannot express.

### 6.2 Deferred, deliberately

Reading a stated end date out of document text is **not built now**. The concept and the column exist so
the model is right from the start; populating them is later work. Until then the derived rule carries the
whole load, and it is sufficient for every case except self-terminating policies.

### 6.3 The current column is UNTRUSTED — measured, not assumed

Step 0 (§12.2) found `termination_date` is `created_at + 182 days` for **5,475 of 5,494** AHCA documents,
with **five distinct values** corpus-wide. It is a refresh TTL wearing a policy date's name.

Concretely: the 59G-4.130 coverage policy effective **2016-11-01** carries `termination_date = 2027-02-15`
— the corpus asserts a nine-year-old superseded policy remains valid for another six months. 22 documents
are already past their stated termination date and still servable.

This is worse than a NULL. A missing value is a findable gap; a populated wrong value is a confident wrong
answer with nothing to detect it. **The derived values must be cleared before any as-of query depends on
this column** — §10 cannot be built on it as it stands.

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

### 10.2 Retriever's response to §17 — right catch, sign-off updated (2026-08-17)

Read §16.4 and §17 firsthand before responding. Agree with the finding, and it does change what §10 has
to mean, not just when it ships.

**You're right about what I assumed.** §10.1 above was written against a mental model of "current vs.
superseded" — one winner, one loser. §16.4's second property (multiple admitted-but-unretired versions can
coexist at one `doc_key` while verdicts are pending) means the real requirement is **selection among N
concurrently-active versions**, not a filter down to a single flagged one. `lifecycle_state = active` was
never going to be sufficient on its own — I should have derived that from §4.2's ambiguity design when I
first read it, not needed the human-loop simulation to surface it. Correcting the record rather than
leaving the narrower framing standing.

**One data point already in hand: my tonight stopgap happens to generalize.** The `filler_a.py` tiebreak
groups *all* pool candidates sharing a rule-number identifier, not just a pair — so a same-`doc_key` chain
of 5 concurrently-active versions would already get grouped together today, and the query-year gate would
already decide among however many are in the group, not just two. It's still the wrong layer (Filler-stage
score nudge, not Pool-level as-of filtering) and still text-regex rather than a structured `as_of_date`, but
the *shape* of "select among several, not pick between two" turns out to already be there by accident. Not
claiming this validates the design — flagging it because it's a real, running data point on the N-version
case, not a hypothetical.

**Agree with the build-order correction.** §10 as prerequisite rather than step 7 is right: an as-of filter
that only exists after the gate has already been writing ambiguous chains for a while means every query in
that window silently picks among concurrent versions with no real signal — worse than today, where at
least a lone `active` flag (wrong as it can be) doesn't multiply. No objection to resequencing.

**Not mine to build:** queue drain rate as a corpus-health metric and the `supersedes_id` split-cascade
traversal are both gate-side (§4.2/§8), Master RAG's scope. Noting I read them, not taking them on.

**Sign-off stands, restated precisely:** §10's retrieval contract must resolve as-of a date by **selecting
among however many versions of a `doc_key` are concurrently active** (not filtering to one pre-flagged
current version), ranked/chosen by the validity window against `effective_date`/`termination_date`. This is
a materially different (harder) implementation than what I originally signed, and I'm signing the harder
version now that it's understood, not the easier one I assumed.

---

## 11. Open questions requiring a ruling

### 11.1 Payor Platform — exclusion vs ranking floor (**conflict**)

Contract v2 states junk and non-authoritative sources land on the ratified floor (`fyi_not_citable`)
*"so they rank last instead of disappearing."* This spec states untracked documents never enter the index.
**These conflict** — one is a ranking penalty, the other is exclusion. Joint ruling needed.

### 11.2 Fact Store / Payor Platform — `importance` grain, and `asset_type` for revisability

**New ask (§2.2):** `doc_key` tier 2 depends on knowing whether a document is **revisable** (a policy or
handbook that gets replaced by a later edition) or **episodic** (minutes, a notice, a quarterly report —
issued once, never revised). 83.3% of AHCA is episodic and must never be version-linked. You already
return `asset_type`; can it carry this distinction, or should it be a separate field? I do not want to
re-derive it from filename regexes on the RAG side.

**Original question —** is `importance` a property of the document or the version?

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

### 12.2 Step 0 results — AHCA, measured 2026-08-17 (read-only)

5,494 AHCA documents, 1,168,108 chunks. Chunk hashing used `lower()` + whitespace collapse only, so
every duplicate figure below is a **lower bound** — full §3 normalization will find more.

**Waterfall — what is indexed vs what should be:**

| | Reason | Docs | Running |
|---|---|---|---|
| | AHCA documents ingested | **5,494** | |
| − | zero pages — extraction produced nothing | 150 | 5,344 |
| − | pages but zero chunks — chunking never completed | 211 | 5,133 |
| − | exact content duplicate — same normalized text, keep 1 canonical | 100 | 5,033 |
| = | **distinct indexable documents** | **5,033** | |
| − | superseded version — older edition in a rule chain *(lower bound)* | 61 | 4,972 |
| = | **should be active in index** | **4,972** | |

522 documents (9.5%) should not be servable; 18,243 chunks (1.6%) are attached to them. This waterfall is
the shape the **post-publish integrity check** should report on a schedule — see §12.3.

**Finding 1 — `doc_key` by rule number FAILS as specified.** A 59G-x.y rule number is extractable for only
**2.3%** of AHCA documents; 97.7% are unkeyed. A title-based fallback reaches 98.9% coverage but
over-merges catastrophically — one cluster of **590 documents**. §8.1 called this the riskiest assumption
and it did not survive contact. **The lineage key needs redesign before anything is built on it**, and
measuring first cost hours instead of weeks.

**Finding 2 — `termination_date` is fabricated.** See §6.3. `created_at + 182 days`, five distinct values
corpus-wide.

**Finding 3 — τ_high ≈ 0.70 is supported by data.** 112 candidate pairs: median overlap 0.722, 70 pairs
≥ 0.70, 11 at ≈0, thin middle. The ambiguous branch (§4.1) is real but small — about 10% of pairs.

**Worked case, 59G-4.130** — the collision Retriever hit in production is four documents, not two: two
rulemaking notices plus two coverage policies (effective 2016-11-01 and 2024-09-01), identical 53-chunk
counts, both carrying the same fabricated termination date. The reranker was choosing between two of four
near-identical candidates with nothing in the data to break the tie correctly.

### 12.3 Post-publish integrity checks

**Ananth:** *"these are all the integrity checks post publish."*

Distinct from the pre-publish gate. The gate decides one document at a time; these are corpus-wide
invariants that can only be checked after the fact, and they run on a schedule:

| Check | Invariant | Current AHCA |
|---|---|---|
| unpublishable present | no active document has zero chunks | 361 violations |
| duplicate content | no two active documents share a `content_digest` | 100 violations |
| chain integrity | at most one `active` version per `doc_key` | unmeasurable until key redesign |
| date sanity | no `termination_date` earlier than its `effective_date`; none derived from `created_at` | 5,475 derived |
| stale service | no active document past an explicit `termination_date` | 22 violations |
| orphan chunks | no chunk whose document is retired or shelved | to measure |

Each violation count is a number that should trend to zero and stay there. A check that has never been
green is a check that is measuring a known defect, not guarding an invariant — and should say so.

---

### 12.4 Pilot results — dry run, 12 documents, 2026-08-17

`scripts/gate_dryrun.py`, read-only. Pilot set chosen to hit every branch: the 59G-4.130 family, a
duplicate group, zero-page and zero-chunk documents, and an episodic cluster.

**It found two design bugs before any code was written — which is what the pilot was for.**

| # | Bug | Symptom in the run | Fix |
|---|---|---|---|
| 1 | episodic docs keyed by rule number | `Notice of Proposed Rule: 59G-4.130` took `doc_key = AHCA\|FL\|59G-4.130`, overlap 0.019, decision `SUCCESSOR` → **would retire the real coverage policy** | episodic test precedes all tiers (§2.2) |
| 2 | ladder defaulted to promote | anything under τ_high but over 0.01 became `successor(partial)` and promoted | default is ASK; both ambiguous branches leave the prior ACTIVE (§4.2) |

**Post-fix decisions across the 12:**

| Decision | n | Notes |
|---|---|---|
| `first_version` | 7 | includes all 4 episodic notices — correctly unkeyed, correctly not superseding anything |
| `unpublishable` | 4 | 2 zero-chunk, 2 zero-page → §5.0 remediation actions emitted |
| `ambiguous_revision` | 1 | the genuine 59G-4.130 pair at 0.698 → routed to Fact Store, prior stays active |

Third finding, from recalibrating after fix 1: **the τ measurement was contaminated** — see §4.3.

Nothing was written: no retire, no delete, no re-trigger, no index change.

---

## 13. Sign-off ledger

| Seat | Scope of review | Status |
|---|---|---|
| **Fact Store / Payor Platform** *(one seat — Ananth 2026-08-17)* | §7 human loop · §11.5 review surface · **§11.1 exclusion vs floor · §11.2 importance grain · §5.2 recall bias** | 🟡 PARTIAL — §7/§11.5 ✅ signed 2026-08-17 w/ 2 refinements: (1) §7 returns TWO separate valid-time dates, never derive one from the other; (2) verdict keyed on the DIGEST PAIR + doc_key. Verdict store theirs; RAG pre-renders diff; priority-triaged. **Still owed: §11.1 / §11.2 / §5.2** |
| Retriever | §10 as-of contract · index filter on (`doc_key`, `lifecycle_state`) | ✅ signed, see §10.1/§10.2 — updated after §17: contract is *selection among N concurrently-active versions*, not a filter to one; `as_of_date` must be a structured caller param, not query-text regex; §10 build-order moved to prerequisite, agreed |
| Eval | §11.3 classifier miss profile baseline · τ_high calibration | ⬜ |
| DB seat | §9 schema deltas · §11.4 column contract + index strategy | ⬜ |
| Maintaining | §3 normalization vs coherence gate · `last_validated_at` freshness overlap · nightly-sweep interaction | ⬜ |
| Technical Review | structure + seam ownership | ⬜ |

Nothing in §12 beyond step 0 begins before the seats covering that step have signed.

---

## 14. Fact Store / Eval sign-off — detail (2026-08-17)

Signed §7 + §11.5 (Fact Store row ✅). Read §6, §7, §7.1/7.2, §8, §11.3, §11.5, §12 firsthand. Two additive refinements — both strengthen the design, neither blocks the build.

### 14.1 §7 — return TWO valid-time dates, never auto-derive one from the other
The human returns `successor.effective_date` **and** `predecessor.termination_date` as **separate** fields. They usually coincide but not always: editions can gap (a policy lapses before its successor takes effect) or overlap (both in force during a transition). Auto-writing `predecessor.termination_date = successor.effective_date` silently manufactures the exact confidently-wrong window §6 exists to kill. If the human supplies only the successor date and the boundary is unstated, `termination_date` stays **NULL** (honest, queryable) — same discipline as §6's two clocks, applied to valid-time itself.

### 14.2 §11.5 — key the verdict on the DIGEST PAIR, not a lone digest
A verdict answers "is B a successor of A" — it is about a **pair**. A single document is adjudicated against multiple candidates, so a lone-`content_digest` key can't disambiguate which comparison a row answers. Verdict row (append-only, my side, extending the classifications-log shape):

```
(pred_content_digest, succ_content_digest, doc_key,
 verdict ∈ {successor, not_successor, unrelated},
 successor_effective_date, predecessor_termination_date,
 adjudicated_by, adjudicated_at)
```

Append-only → re-adjudication writes a new row; a wrong verdict stays diagnosable. Keyed on content_digest (not document_id) so tonight's re-crawl can't recompute the human's answer away.

### 14.3 Answers to Q1–Q3
- **Q1 (store side):** the verdict store lives on the **Fact Store** side, extending the classifications append-only pattern — one place for the human-facing record. RAG emits the adjudication request and ingests the verdict over a contract; store of record is mine.
- **Q2 (diff render):** **RAG pre-renders** the chunk diff (it falls out of your `chunk_sha` comparison — the same comparison that drove the escalation) and hands me **display-ready** content. My surface must not recompute chunking or it could show a diff that disagrees with the gate's own version decision. Contract:
  ```
  { pred_digest, succ_digest, doc_key,
    changed_chunks: [ { chunk_id, before, after } ],
    candidate_dates: [ { date, confidence, source_span } ],
    self_declared_supersession: { text, cited_date } | null }
  ```
  I own the interaction layer (diff display, verdict buttons, the two date fields); RAG owns the diff content.
- **Q3 (queue sizing):** triage by **decision-value**, not pagination. Order the ambiguous tail by (1) criticality, (2) **retrieval-impact** (a superseded doc nobody ever retrieves is low-priority — pull actual retrieval frequency), (3) confidence gap. So §8 phase-7 emit should carry a **priority score**, not just cluster membership. The pagination threshold can't be named until **§12 step-0** reports the ambiguous-tail count (that's what step-0 measures) — run it, give me the number, I size the surface (flat list if small, priority-triaged queue above ~a couple hundred). Priority-scored triage holds regardless of the number.

### 14.4 §6 two-clocks — confirmed
`retired_at` = transaction time (gate, automatic, always correct); `termination_date` = valid time (human/high-confidence extraction, **never** the gate); NULL is the honest, queryable state. Same accepted⟹grounded / abstain-not-guess discipline as the fact-store integrity model — a wrong date is undetectable, an unknown is a backlog you can clear. No objection; load-bearing and right.

### 14.5 Eval §11.3 — the miss asymmetry dictates the operating point (row left ⬜: commitment, not done)
A false *unknown* only sizes the human queue (recoverable — a human sees it). A false *low* is **silent and permanent** under §5 (low → untracked → never indexed → never reviewed → invisible). Wildly asymmetric cost. Eval ruling: baseline **both directions separately** (false-unknown rate and false-low rate are different metrics with different consequences — do not report a single accuracy), and set the importance classifier **recall-biased on "important"** (prefer false-unknown over false-low) — the same discipline as the PHI classifier: recall over precision when a miss is silent and permanent. I own the miss-profile baseline + τ_high calibration; it is gated on the importance classifier existing to measure against a labeled set. Flag me when it is runnable — I baseline both directions before §5 tiering is built around it. Nothing in §7's build (step 6) is blocked by this.

— Eval / Fact Store seat

### 14.6 Post-step-0 — Q3 now sizable, and §11.3 must cover the revisable/episodic classifier too (2026-08-17)

Read §2.2 step-0 findings. Two follow-ups.

**Endorse "no key is better than a wrong key" — it is exactly the §11.3 asymmetry, upstream.** A NULL doc_key = a recoverable singleton (retrievable, never supersedes); a WRONG doc_key = false supersession = a valid doc silently removed from retrieval with no gap to find. Not comparable — assign a key only on positive evidence of a revisable identity. This is the same "false-*low*/false-episodic is silent and permanent, false-*unknown* only sizes the queue" principle I ruled for the importance classifier. Right call.

**Q3 is now answerable, and the queue is small.** Step 0 says **83.3% of the corpus is episodic** (never revised — needs no key, never adjudicated) and only **~16.7% (~880 of 5,259) is revisable**. The adjudication queue is bounded by that revisable subset, and further by only the clusters with *unresolved* ordering (§8 phase 7) — a fraction of ~880, not 5,259. So: **my surface v1 is a priority-sorted flat list, no pagination.** Priority score on the emit still stands (criticality → retrieval-impact → confidence gap) so the reviewer works highest-value-first, but at this volume pagination is unnecessary. Revisit only if the ambiguous-with-conflict count exceeds ~200; step 0 strongly suggests it won't.

**New Eval item — §11.3 baseline must cover BOTH classifiers, not just importance.** Step 0 introduces a second gating classifier: the **revisable vs episodic** call, which "must come before keying." It carries the identical silent-miss asymmetry: a false-**episodic** on a truly revisable document → never keyed → never supersedes → a stale edition stays live in retrieval with full confidence, invisibly. So the miss-profile baseline I own now has two subjects — importance AND revisable/episodic — each baselined in **both** directions separately, each set **recall-biased toward the needs-attention class** (important / revisable). A single accuracy number on either hides the only error that matters. I'll baseline both when they're runnable against a labeled set; flag me.

— Eval / Fact Store seat


---

## 14. Readiness audit — modules and seams (2026-08-17)

Verified against the live dev schema and the two dry-run harnesses. **Nothing is built.** Every module
below exists only as prototype logic inside `scripts/gate_dryrun.py` / `scripts/gate_backprop_dryrun.py`,
which are read-only.

### 14.1 Schema — 0 of 10 present

| Object | State |
|---|---|
| `documents.doc_key` / `version_no` / `content_digest` / `supersedes_id` / `lifecycle_state` / `retired_at` / `last_validated_at` | **all MISSING** |
| `hierarchical_chunks.chunk_sha` / `carried_from_chunk_id` | **both MISSING** |
| `gate_decisions` (§1.1 telemetry) | **MISSING** — not yet specified as a table anywhere |

Blocked on the DB seat (§11.4). Note `gate_decisions` was described in §1.1 as fields but never given a
table definition — that is a gap in this spec, not just in the schema.

### 14.2 Modules

| # | Module | State | Note |
|---|---|---|---|
| 1 | normalization (§3) | prototype | prose only; **structure-preserving not built** — the fee-schedule caution is unaddressed |
| 2 | `chunk_sha` / `content_digest` | prototype | computed on the fly, nowhere to store |
| 3 | `doc_key` (§2.2) | prototype | tier 2 blocked on Fact Store `asset_type` |
| 4 | version decision (§4) | prototype | ladder corrected after pilot |
| 5 | lane rules (§5) | prototype | `importance` not yet on these rows |
| 6 | publishability + remediation (§5.0) | **decides but does not act** | emits the action; no re-trigger is wired |
| 7 | promotion gate (§5.1) | **NOT BUILT, NEVER EXERCISED** | 0 chains auto-resolved, so the branch has never run |
| 8 | telemetry (§1.1) | prints, does not persist | no table |
| 9 | delta reuse (`carried_from_chunk_id`) | measured, not wired | 59G-4.130 shows 37 carried / 8 changed |
| 10 | back-propagation (§8) | prototype, phases 1–6 | phase 7 emit not wired |
| 11 | Fact Store emit | payload shaped, not sent | §14.4 |
| 12 | verdict ingest | **NOT BUILT** | no path for an answer to come back |
| 13 | retrieval as-of (§10) | **NOT BUILT** | Retriever's; signed, unimplemented |
| 14 | ordering confidence | **BUGGY** | §14.3 |
| 15 | filename date extraction | **NOT BUILT** | required by §14.3; unowned |

### 14.3 Bug 3, found by the back-propagation run

The `Attachment II — Core Contract Provisions` cluster is a genuine **10-edition chain**: filenames carry
`2019-02-01`, `2020-02-01`, `2020-07-01`, `2020-10-01`, `2021-10-01`, `11-4-22`. Tier 2 keyed it
correctly — the first real multi-edition chain the design has found.

But the run reported `ordering_confidence: "date-ordered"`, which is **false**. All ten share
`effective_date = 2026-07-01` (the fabricated value, §6.3) — one distinct date across the whole chain.
Sorting by it is arbitrary, so the chain order is meaningless while claiming confidence.

Two fixes required:
- **Ordering confidence must detect *degenerate* ordering, not just NULL dates.** If a cluster has fewer
  distinct `effective_date` values than members, ordering is unreliable and the cluster goes to a human.
  Testing `IS NULL` was never sufficient.
- **The real dates are in the filenames.** Filename date extraction is the highest-value missing module —
  it would order this chain correctly without a human. Currently unowned.

### 14.4 Seams

| Seam | Direction | Crosses | Status |
|---|---|---|---|
| Fact Store → RAG | in | `importance`, `claimed`, `authority_level` | live, but only 86 AHCA docs carry it |
| Fact Store → RAG | in | **`asset_type` = revisable vs episodic** | **OPEN — §11.2. Blocks `doc_key` tier 2** |
| RAG → Fact Store | out | adjudication request (cluster-level) | shape drafted §14.5, not sent |
| Fact Store → RAG | in | verdict: relationship + 2 valid-time dates | shape agreed, **no ingest path** |
| RAG → Retriever | out | `doc_key`, `lifecycle_state`, as-of window | **✅ signed**, unimplemented both sides |
| RAG → DB seat | out | 9 columns + telemetry table | **⬜ unsigned — blocks everything** |
| RAG → Eval | out | τ evidence bar | ask changed after §4.3; open |
| RAG → Maintaining | out | `last_validated_at` vs coherence gate | **⬜ added late, not yet answered** |
| Crawler → RAG | in | re-fetch cadence (drives Case A) | informed, no vote |
| RAG internal | — | chunking re-trigger endpoint | **exists** — `POST /documents/{id}/chunking/start` |

### 14.5 Uncovered seams — no owner assigned

These have no seat and are not in the §13 ledger. Flagging rather than assuming:

1. **Who clears the fabricated dates?** `effective_date` (4 distinct values) and `termination_date`
   (`created_at + 182d`) are RAG's columns, but the values came from some upstream process. Remediation
   has no owner. §10 cannot ship until it is done.
2. **Who owns filename date extraction?** Needed for §14.3 ordering. Plausibly Curation, plausibly
   extraction — undecided.
3. **Who owns `display_name` remediation?** Raised with Fact Store (28.5% of the corpus is a category
   label); Q3 of that ask is exactly "yours or mine" and is unanswered.

### 14.6 Back-propagation, measured

| Step | Documents |
|---|---|
| AHCA total | 5,496 |
| − re-enqueue chunking | 213 |
| − re-trigger extraction | 150 |
| − retire duplicates | 100 *(14,522 indexed chunks deleted)* |
| keyed into clusters | 95 *(80 clusters)* |
| episodic / unkeyed | 4,938 → singleton forever |
| **multi-doc chains** | **7** |
| → auto-resolvable | **0** |
| → needs Fact Store | **7** |

Zero chains auto-resolve, entirely because of §14.3 — the ordering is degenerate, not because the content
evidence is weak. **Fixing filename date extraction is what converts this from 7 human tasks to nearly
zero**, and it is the single highest-leverage missing module.


---

## 15. Telemetry persisted, and the first real calibration (2026-08-17)

`scripts/gate_persist_run.py`. Creates `gate_decisions` and writes one row per document.
**Additive only** — no document, chunk or index was modified. The gate's decisions are *recorded*, not
applied, which is exactly the §1.1 separation: the decision row exists before (here, instead of) any
mutation, so it can be inspected before anything acts on it.

`gate_decisions` DDL is **dev-only and unratified** — the DB seat owns the column contract (§11.4) and
this table is expected to be revised to their shape. Flagged to them on creation.

### 15.1 500 documents run as incoming

| Decision | lane | n | avg overlap | carried | re-embed |
|---|---|---|---|---|---|
| `first_version` | untracked | 398 | — | — | — |
| `first_version` | tracked | 76 | — | — | — |
| `unpublishable` | — | 16 | — | — | — |
| `ambiguous_revision` | tracked | 6 | 0.636 | 7,396 | 2,289 |
| `ambiguous_order` | tracked | 2 | 0.321 | 1,645 | 2,243 |
| **`successor`** | tracked | **2** | **0.740** | **3,329** | **587** |

8 adjudications would reach Fact Store. Ordering confidence: 478 no-sibling, **8 filename-date**,
12 degenerate, 2 effective-date.

**The §14.3 fix works.** Filename date extraction produced the first automatic promotions the design has
ever made — previously 0 of 7 chains resolved. Both were verified by hand and both are correct:

```
2021-10-01  supersedes  2020-10-01     overlap 0.744   85% of embeddings reused
11-4-22     supersedes  2022-10-01     overlap 0.736   85% of embeddings reused
```

**Delta reuse is real and large:** 3,329 chunks carried against 587 re-embedded. A contract revision costs
15% of a full re-embed, not 100%.

### 15.2 τ IS document-class dependent — the finding that changes §4.3

The `Attachment II — Core Contract Provisions` chain gives nine consecutive-edition links, all ordered by
filename date, all genuine successions:

```
2019-02-01 → 2020-02-01   0.599
2020-02-01 → 2020-07-01   0.651
2020-07-01 → 2020-10-01   0.588
2020-10-01 → 2021-10-01   0.744   ← promoted
2021-10-01 → 2022-02-01   0.610
2022-02-01 → 2022-10-01   0.670
2022-10-01 → 11-4-22      0.736   ← promoted
```

Genuine consecutive editions of this contract occupy roughly **0.59–0.74**. A global `τ_high = 0.70`
slices that band almost arbitrarily: two links promote, five identical-in-kind links go to a human.

Compare the only other real pair we have — 59G-4.130, a *coverage policy*, at **0.698**.

**A single global threshold is wrong in principle.** A contract attachment revised twice a year and a
coverage policy revised once a decade have different characteristic overlap, and one number cannot serve
both. τ should be **per `asset_type`**, which is a third independent reason to need that field from Fact
Store (§11.2) — it now gates tier-2 keying *and* threshold selection.

Do **not** simply lower τ to 0.58 to capture this chain. That is fitting one document family, the same
error as the withdrawn 0.70. The correct next step is to accumulate links per asset class and let each
class carry its own threshold, with the safe default (§4.2) holding until a class has enough evidence.

### 15.3 What is still unproven

- **The promotion gate still has not run.** It is `n/a` on all 500 rows: the two successors would be its
  first invocations, and completeness-vs-predecessor has never been exercised.
- **`unchanged` never fired.** No document in the run was a re-observation of one already present, so
  pilot Case A — the nightly path that must be *exactly free* — remains unverified against real data.
  It needs a genuine re-scrape, which means Crawler, not a corpus replay.
- Latency is reported as ~0 ms because only the decision is timed; hashing and IO are not.


---

## 16. Bug 4 — the ambiguity deadlock, found by forward simulation

`scripts/gate_forward_sim.py`. **The 500-doc run in §15 was mislabelled `mode='forward'`. It was a
replay.** Verified against the corpus:

- **0 URLs have ever been fetched more than once** — a re-observation has never occurred, so `unchanged`
  was unreachable by construction.
- All 12 `Attachment II` editions were **ingested on a single day** (2026-04-29) from an archive page.
  A historical pile, not a revision stream.

Its two "promotions" were back-propagation chain reconstruction — a different question from *"a document
just arrived; does it supersede what is live?"* Correcting that is what exposed the bug.

### 16.1 The deadlock

Presenting the eight dated editions in publication order, with the original rule that only
`first_version` and `successor` advance the live pointer:

```
2019-02-01  FIRST_VERSION
2020-02-01  ambiguous_revision   ov 0.599   vs 2019-02-01
2020-07-01  ambiguous_revision   ov 0.565   vs 2019-02-01
2020-10-01  ambiguous_revision   ov 0.525   vs 2019-02-01
2021-10-01  ambiguous_revision   ov 0.531   vs 2019-02-01
2022-02-01  ambiguous_revision   ov 0.484   vs 2019-02-01
2022-10-01  ambiguous_revision   ov 0.478   vs 2019-02-01
2022-11-04  ambiguous_revision   ov 0.479   vs 2019-02-01
```

**The chain never advances, and it gets worse the longer it runs.** Because ambiguity leaves the prior
active, `live` freezes at the oldest edition; every subsequent arrival is compared against a
five-year-old ancestor, so overlap **decays monotonically** and the chain becomes progressively *less*
able to resolve. One ambiguous link permanently deadlocks a document family.

This is a direct consequence of the §4.2 safe default and could not be seen in a replay, which compares
consecutive pairs rather than against a frozen anchor.

### 16.2 The fix — admission and retirement are separate decisions

The comparison anchor must be the most recently **admitted** version, not the most recently **promoted**
one. Every branch of §4.2 admits the incoming document to the index; they differ only in whether the
*prior* is retired. Anchoring on the last promotion conflates two independent things.

```
anchor := most recent ADMITTED version at doc_key      (not: last promoted)
```

Same chain, after the fix:

```
2019-02-01  FIRST_VERSION
2020-02-01  ambiguous_revision   ov 0.599   vs 2019-02-01
2020-07-01  ambiguous_revision   ov 0.651   vs 2020-02-01
2020-10-01  ambiguous_revision   ov 0.587   vs 2020-07-01
2021-10-01  SUCCESSOR            ov 0.744   vs 2020-10-01   → retires prior
2022-02-01  ambiguous_revision   ov 0.610   vs 2021-10-01
2022-10-01  ambiguous_revision   ov 0.670   vs 2022-02-01
2022-11-04  SUCCESSOR            ov 0.736   vs 2022-10-01   → retires prior
```

Overlap now reflects the actual edition-to-edition delta, two links resolve automatically, and the chain
progresses. Re-embed cost drops with it (489 → 390 → 511 → **285** on the promoted link).

### 16.3 Case A passes

| Case | Result |
|---|---|
| re-present the active document unchanged | `UNCHANGED`, `index_action = none`, **0 chunks re-embedded** — **PASS** |
| same document + cosmetic drift (whitespace, injected "Printed on …" line) | `UNCHANGED`, 0 re-embedded — **PASS**, normalization absorbed it |

The nightly path is free, and §3 normalization does the job it was specified for. Note this is a
*synthetic* re-presentation: the corpus still contains no genuine re-fetch, so Case A remains unproven
against real crawler output. That test needs Crawler, not a replay.

### 16.4 What this says about the ambiguity design

Ambiguity must never be load-bearing on progress. Two properties now required:

1. **An unresolved adjudication cannot block the chain.** The anchor advances regardless; only retirement
   waits for a verdict.
2. **Multiple admitted-but-unretired versions can coexist** at one `doc_key` while verdicts are pending.
   That makes §10's as-of resolution more important, not less — with several unretired versions live,
   "which one answers a date-of-service query" is decided by the validity window (§6.1), not by a single
   `active` flag.


---

## 17. The human loop, simulated — and the dependency it exposes

`scripts/gate_human_loop_sim.py`. Verdicts delivered **out of order**, deliberately, to test that a late
answer still applies. Read-only.

### 17.1 What the loop proved

| Property | Evidence |
|---|---|
| A verdict keyed on the **digest pair** applies even when it arrives after the crawler has moved on | verdict 1 resolved a pair queued three editions earlier |
| **A human is the only writer of `termination_date`** | `term=2020-01-31` was the first valid-time value anything had written; the gate had left every one NULL |
| `not_successor` is **not a no-op** — it splits the lineage | 3 documents moved to a new `doc_key`; everything chained *through* the rejected link inherits the split, or later editions stay attached to the wrong ancestry |
| An automatic promotion is **reversible** | verdict 3 un-retired a document the gate had auto-retired. Recoverable *only* because retirement never deletes — had the gate dropped the row or its chunks, the verdict would be unactionable |

### 17.2 The risk this exposes — ambiguity accumulation

After the gate ran and before any human looked, the state was:

```
[ACTIVE ] v1  2019-02-01
[ACTIVE ] v2  2020-02-01
[ACTIVE ] v3  2020-07-01
[retired] v4  2020-10-01   ← auto-promoted past
[ACTIVE ] v5  2021-10-01
        4 of 5 versions simultaneously ACTIVE, 3 adjudications pending
```

**§4.2's safe default has a cost, and this is it.** Refusing to retire on ambiguity prevents silent
wrong-supersession — but it leaves the index holding four near-identical versions of one contract. That
is precisely the 59G-4.130 collision Retriever hit in production, multiplied by the length of the chain.

The safe default is only *safe* if something downstream disambiguates. That something is the validity
window (§6.1): with ordered effective dates, an as-of query resolves to exactly one version regardless of
how many carry `lifecycle_state = active`.

### 17.3 Consequence for the build order — §10 is a prerequisite, not a follow-on

**`active` is not a retrieval filter. The validity window is.** Which makes the dependency hard:

> Shipping the gate **without** §10 as-of resolution would make retrieval **worse**, not better — it
> would add concurrently-active versions to the index with nothing able to choose between them.

§12 lists the Retriever contract at step 7. That ordering is now wrong: §10 must land **with or before**
the gate's first write, not after it. Retriever signed §10 before this was known and should see it —
their sign-off assumed a filter over one active version, and the real requirement is selection among
several.

Two follow-ups this creates:

1. **Queue drain rate becomes a corpus-health metric, not just an ops number.** Every pending
   adjudication is an extra active version competing in retrieval, so a slow queue degrades answers.
   That belongs in §12.3's integrity checks.
2. **The split cascade needs a defined traversal.** The simulation moved documents by filename order as a
   stand-in; the real cascade must follow `supersedes_id`, so that exactly the documents chained through
   the rejected link move, and no others.
