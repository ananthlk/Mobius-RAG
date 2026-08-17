# REQ · Ingestion classification contract (Payor Platform → Master RAG)

**From:** Fact Store / Payor Platform · **Date:** 2026-08-17 · **Status:** endpoint live in dev, awaiting RAG wiring

Ananth 2026-08-17:

> "most uploads / ingestion will go through RAG and when it does you need to establish a contract
> with them that before they do anything they need to ping you and you will give them the final
> classification.. on not classify you will share that with them.. not error, but you will show
> that in your view for user to edit"
> …
> "we need to pass the feedback to RAG and have it emit this classifier feedback.. and status"

---

## The ask, in one line

**Before chunking a document, call us; then emit what we said.**

Two halves. The first is a gate, the second is visibility — and the second is the part that keeps
this from becoming another silent step.

---

## Half 1 · Call us before you index

```
POST https://mobius-payor-ortabkknqa-uc.a.run.app/api/registry/ingest/classify
```

Either shape works:

```jsonc
{ "document_id": "130cd808-9ccd-4f49-b6c2-f802d0c8f6bf" }        // post-ingest
{ "payor": "AHCA", "filename": "59G-4.130 …pdf",                  // pre-ingest
  "source_url": "https://ahca.myflorida.com/…", "text_sample": "…" }
```

Response — **always `200`**:

```jsonc
{ "decision": "admit",          // admit | hold | reject
  "may_index": true,            // the only field you must branch on
  "needs_human": false,
  "why": "medicaid_policy_rule from ahca.myflorida.com",
  "stages": {
    "source_authority": { "verdict": "authoritative", "host": "ahca.myflorida.com" },
    "junk":             { "excluded": false },
    "bucket":           { "asset_type": "medicaid_policy_rule", "confidence": "high" },
    "authority":        { "authority_level": "contract_source_of_truth" },
    "relevance":        { "service_lines": ["CMHC/FQHC", "Behavioral health"] }
  },
  "review_url": "/#/payor/AHCA/working-queue?doc=130cd808…",
  "contract_version": "1" }
```

| decision | what you do |
|---|---|
| `admit` | chunk, embed, publish — exactly as today |
| `hold` | store it, **don't chunk**. It waits in our Working Queue for a human |
| `reject` | store it, **don't chunk**, we mark it excluded. Reversible from our console |

`hold` and `reject` differ only in what *we believe* — never in what you do with the bytes.
**Nothing is ever deleted on either side.**

### Where to put the call

Between "document and pages are committed" and "queue the chunking job", at:

- `POST /documents/import-scraped-pages`
- `POST /documents/import-from-gcs`
- `POST /documents/import-from-drive`
- `POST /documents/import-from-html`
- the chat / instant-RAG upload path

The mechanism already exists on your side — `auto_chunk=false` on `import-scraped-pages` does
exactly this deferral today. This makes it the default rule rather than a per-caller courtesy,
and attaches a *reason* a human can act on.

**Your call, not ours:** synchronous inside the import request (simple, one extra round trip), or
fired just before the chunking worker claims the job (no latency in the import path, more moving
parts). We accept a descriptor before the row exists and a `document_id` after, so both work.

### Three guarantees so this cannot break your pipeline

1. **We never return an error for a document we cannot classify.** An HTTP error tells a caller to
   retry, and retrying cannot help — it would either block ingestion or be ignored. Unknown
   document, unknown payor, even an unknown `document_id` → `200` + `decision: "hold"`.
   *Verified live against a nonexistent id.*
2. **If we are unreachable, treat it as `hold` and carry on.** Do not fail ingestion. The only cost
   is a document waiting for a human instead of entering the index unreviewed.
3. **Contract is versioned** (`contract_version`). We will not change response shape under you.

---

## Half 2 · Emit the verdict — the part we're asking you to add

Consuming the answer silently would repeat the failure mode that cost us most of 2026-08-17:
a step that ran, reported success, and left no trace anyone could read. Three real defects that
day each reported success — a run that classified two unrelated documents, an import that wrote a
document with **zero pages**, and a sweep that logged `classified: 5259` while changing nothing.

So please **emit the classification as a first-class event and status**, the same way extraction
and chunking already are:

### a) Persist it on the document

Store `decision`, `why`, `asset_type`, `authority_level`, `needs_human` and the timestamp, so
`documents` can answer "was this classified, and what did it say" without calling us again.

### b) Emit an event

Alongside your existing ingestion events — something a human or a trace can read:

```
payor_classification · document_id=… decision=hold
    why="provenance unestablished — uploaded document with a placeholder origin (ahca.local)"
    needs_human=true  contract_version=1
```

### c) Show it in the RAG document view

A status line on the document — `admit` / `hold` / `reject`, the `why`, and a link to
`review_url` so someone looking at the document in RAG can jump straight to where it's fixable.
A held document should be visibly held, not just quietly un-chunked.

### d) Make it queryable

"Show me every document currently held" should be answerable from RAG's own data. That number is
the backlog, and a backlog nobody can count is a backlog nobody clears.

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
