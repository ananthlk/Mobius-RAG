# REQ · Ingestion classification contract (Payor Platform → Master RAG)

**From:** Fact Store / Payor Platform · **Date:** 2026-08-17 · **Status:** contract **v2** live in dev, awaiting RAG wiring · ⚠️ v1 shape in circulation earlier today is superseded — see the banner below

Ananth 2026-08-17:

> "most uploads / ingestion will go through RAG and when it does you need to establish a contract
> with them that before they do anything they need to ping you and you will give them the final
> classification.. on not classify you will share that with them.. not error, but you will show
> that in your view for user to edit"
> …
> "we need to pass the feedback to RAG and have it emit this classifier feedback.. and status"

---

## ⚠️ CONTRACT v2 — supersedes the v1 shape first sent (2026-08-17, same day)

**If you started against `decision: admit|hold|reject` and withholding chunking, stop — that was
v1 and it is wrong.** Ananth corrected it:

> "i think RAG will still chunk even in this case, just the authority of the doc will be really
> low.. and we will not monitor that doc"
> "this will not hold the RAG pipeline.. it will just allow us to claim the doc for us to
> monitor.. if a low importance job fails that is okay, but if the doc is high important we want
> to know and act"

**The contract is not a gate. It is a claim.** v1 put our judgement in the critical path of your
pipeline and treated "not worth citing" as "not worth storing". Chunking is cheap and reversible;
deciding what to *watch* is the expensive judgement, and it is ours.

### What this means for you

**You chunk everything, always. You never branch on our answer.** The integration is: call us,
store what we say, emit it. That is the whole change.

---

## The ask, in one line

**Call us at ingest; store and emit what we say. Keep chunking regardless.**

```
POST https://mobius-payor-ortabkknqa-uc.a.run.app/api/registry/ingest/classify
```

```jsonc
{ "document_id": "130cd808-…" }                      // post-ingest
{ "payor": "AHCA", "filename": "59G-4.130 …pdf",     // or pre-ingest
  "source_url": "https://ahca.myflorida.com/…", "text_sample": "…",
  "ingest_mode": "scrape",                           // "scrape" | "upload"
  "uploaded_by": "ananth",                           // upload only — see below
  "caller": "mobius-rag:import-scraped-pages" }
```

Response — **always `200`**:

```jsonc
{ "importance": "critical",                  // critical | standard | low | none
  "claimed": true,                           // do we monitor this document
  "authority_level": "contract_source_of_truth",
  "needs_human": false,
  "may_index": true,                         // see "the one exception" below
  "monitor": { "alert_on_failure": true, "reverify": true, "in_working_queue": false },
  "why": "medicaid_policy_rule from ahca.myflorida.com",
  "stages": { "source_authority": …, "junk": …, "bucket": …, "authority": …, "relevance": … },
  "contract_version": "2" }
```

| field | what it is for |
|---|---|
| `authority_level` | **store this on the document.** Junk and non-authoritative sources land on the ratified floor (`fyi_not_citable`) so they rank last instead of disappearing |
| `importance` | how much *we* care — decides whether a failure is worth acting on |
| `claimed` | whether the document enters our monitoring surface |
| `monitor.alert_on_failure` | true only for `critical`. **If a claimed document's chunk/embed job fails, tell us** — that is the point of the claim |

### Why importance is the useful output

It makes pipeline failures actionable. Right now every chunking failure looks alike — 151 stale
`blocked` jobs sat unexamined and happened not to matter (all 151 already had embeddings). That
is luck, not a system. With a claim: a **critical** document failing is an alert; a **low** one is
noise.

### The one exception — `may_index`

`may_index` is **advisory for you and binding only for us**. It is `false` only when *we*
initiated the ingestion (`initiated_by` in our own set) **and** the document is important **and**
it needs a human. That is our own AHCA corpus scan refusing to swallow something important
unreviewed — Ananth: *"we will not let a doc which requires a human intervention go through
without us looking at it when the doc is important"*.

**A user scraping a retailer is never gated by our opinion.** If you did not set
`initiated_by`, `may_index` is always `true`. You can ignore the field entirely.

### Two modes

| `ingest_mode` | meaning |
|---|---|
| `scrape` | authority comes from the canonical-domain registry for that payor |
| `upload` | **the upload IS the attestation** — pass `uploaded_by`. A person choosing to upload a document is them vouching for it; asking them to separately attest what they just handed us is the same question twice |

An upload with **no** `uploaded_by` vouches for nothing and falls back to normal origin checks —
please send the user identifier.

Drive is deliberately not modelled yet.

---

## Half 2 · Emit the verdict — the part we're asking you to add

Consuming the answer silently would repeat the failure mode that cost us most of 2026-08-17:
a step that ran, reported success, and left no trace anyone could read. Three real defects that
day each reported success — a run that classified two unrelated documents, an import that wrote a
document with **zero pages**, and a sweep that logged `classified: 5259` while changing nothing.

So please **emit the classification as a first-class event and status**, the same way extraction
and chunking already are:

### a) Persist it on the document

Store `importance`, `claimed`, `authority_level`, `why`, `needs_human` and the timestamp, so
`documents` can answer "was this classified, and what did it say" without calling us again.

### b) Emit an event

Alongside your existing ingestion events — something a human or a trace can read:

```
payor_classification · document_id=… importance=standard claimed=true
    why="provenance unestablished — uploaded document with a placeholder origin (ahca.local)"
    needs_human=true  contract_version=1
```

### c) Show it in the RAG document view

A status line on the document — importance + authority + the `why`, and a link to
`review_url` so someone looking at the document in RAG can jump straight to where it's fixable.
A document we flagged for human review should be visibly flagged, not just quietly classified.

### d) Make it queryable

"Show me every document needing human review" should be answerable from RAG's own data. That
number is the backlog, and a backlog nobody can count is a backlog nobody clears.

### e) Send us a `caller`

Pass `"caller": "mobius-rag:import-scraped-pages"` (or whichever site) in the request. We persist
it, so "is RAG actually calling us, and from where" becomes answerable — including the case that
matters most, a call site that was never wired and is therefore silently absent from the log.

---

## We persist every transaction on our side too

Ananth: *"we need to persist every transaction."* Every call to `/ingest/classify` is written to
an append-only table, whether or not it changed anything — the question asked, the answer given,
the caller, and latency.

**Append-only**: a re-classification writes a *new* row, so a document's history reads as a
history — held yesterday, admitted today after an attestation. Mutating in place would erase the
evidence that makes a wrong verdict diagnosable.

Readable at `GET /api/registry/ingest/classifications`:

| query | question it answers |
|---|---|
| `?document_id=…` | what did we say about this document, and when? |
| `?decision=hold` | what is the current backlog? |
| `?payor=AHCA` | is RAG calling us, and from which sites? |

Live sample after four real calls:

```
 totals: {'admit': 1, 'reject': 2, 'hold': 1}
  #4  hold    mobius-rag:probe                   -                       65ms
  #3  reject  mobius-rag:import-from-drive       -                       61ms
  #2  reject  mobius-rag:import-from-html        -                       61ms
  #1  admit   mobius-rag:import-scraped-pages    medicaid_policy_rule   484ms
```

The log writer never raises — a logging failure of ours must not turn a successful classification
into an error for you.

---

## Why this matters now — measured, not asserted

Run against all **5,259 AHCA documents** on 2026-08-17:

| verdict | docs | |
|---|---|---|
| authoritative | 4,893 | 93.0% |
| unknown (upload placeholder) | 366 | 7.0% |
| rejected | 0 | |

The 366 are Drive/manual uploads where ingestion stored a `gs://` blob path as the url and invented
a `<name>.local` host. Same pattern fleet-wide: `sunshine_health.local` (569), `samhsa.local` (371),
`govinfo.local` (284). They aren't bad documents — their provenance was simply never captured, and
a human attesting one flips it to `admit`. **That population is only findable because the verdict
distinguishes "we couldn't tell" from "we checked and it's not."**

---

## Worked examples — real responses from the live endpoint

| document | origin | answer | why |
|---|---|---|---|
| 59G-4.130 Coverage Policy | `ahca.myflorida.com` | **admit** | medicaid_policy_rule, `contract_source_of_truth` |
| **the same file** | `googleusercontent.com` | **reject** | search origin — content identical, provenance is not the payor |
| `Agenda_LIP_12152010.pdf` | `ahca.myflorida.com` | **reject** | filename year 2010, older than 3 years |
| `Physician_Fee_Schedule_2022.pdf` | `ahca.myflorida.com` | **reject** | age rule — still a fee schedule, just not servable |
| `59G-1.058_Eligibility.pdf` | `gs://` upload | **hold** | provenance unestablished; bucket + relevance still reported |
| **the same upload, attested** | `gs://` + human | **admit** | a named person vouched — the recovery path for the 366 |
| `misc_document.pdf` | `ahca.myflorida.com` | **hold** | good source, no bucket matched — a human decides |

Rows 1–2 are byte-identical documents. That is the property the whole stage exists for:
**authority is a property of origin, not of content.**

---

## Implementation on our side

- `mobius-payor/app/source_authority.py` — canonical-domain registry, 39 tests
- `mobius-payor/app/ingest_contract.py` — the chain verdict (pure, no I/O)
- `POST /api/registry/ingest/classify` — live in dev
- `GET/POST/DELETE /api/registry/payors/{payor}/authoritative-domains` — multiple authoritative
  domains per payor; AHCA currently has 4 (`ahca.myflorida.com`, `flrules.org`,
  `flrules.elaws.us`, `flsenate.gov`)

Visual walkthrough: see the ingestion-contract schematic shared alongside this doc.

**Nothing here changes your schema or asks you to deploy anything of ours.** Happy to pair on the
call sites, or to take a PR review — whichever is less disruptive.


---

## End-to-end acceptance — what "done" looks like

Ananth 2026-08-17: *"i want to user system to upload a doc, scrape a page and want to see those
work through their system to yours and back."*

Two user journeys, both driven from RAG's own UI, both round-tripping:

### Journey 1 — a user uploads a document
1. User uploads a PDF in RAG, tagged to a payor.
2. RAG calls us with `ingest_mode: "upload"` and `uploaded_by: <that user>`.
3. We answer: authority, importance, claim. The upload is its own attestation, so a recognisable
   policy comes back `critical` / `contract_source_of_truth`.
4. **RAG shows the verdict on the document** — importance, authority, why.
5. The call appears in our Classifications log (platform menu → Classifications), with
   `caller` naming the RAG entry point it came from.

### Journey 2 — a user scrapes a page
1. User scrapes a URL in RAG, tagged to a payor.
2. RAG calls us with `ingest_mode: "scrape"` and the real `source_url`.
3. We answer from the canonical-domain registry: a page from `ahca.myflorida.com` is
   `critical`; the same content from a search-engine mirror lands on the floor and unclaimed.
4. **RAG shows the verdict**, same as above.
5. Same row appears in our log.

**The round trip is the deliverable** — RAG UI → our contract → RAG UI, plus a durable row on our
side. Not just a 200.

Worth testing deliberately: scrape the *same* document from a canonical domain and from somewhere
else. The verdicts should differ, because authority is a property of origin, not of content. If
they come back the same, the wiring is passing the wrong `source_url`.
