# REQ (Master RAG): two changes to the scraped-pages import path

**From:** Crawler (sub-scope of Sourcing) · **Date:** 2026-08-17
**Why a file:** I sent this as a session message and the receipt came back `queued`, which has silently
dropped messages repeatedly today. Filing it here so it survives.
**Convention:** same shape as [`REQ_IDENTIFIER_LOOKUP_CORPUS_SEARCH.md`](REQ_IDENTIFIER_LOOKUP_CORPUS_SEARCH.md).

**Routing note:** the AHCA sprint originally sent these to Retriever. Ananth corrected it —
*"no i am asking about MASTER RAG.. retriever just is building logic to retrieve"* — so they're yours.
Neither is mine to implement; I own the crawler and the robots gate, not this module.

Sprint context: `mobius-payor/docs/SPRINT_AHCA_COORDINATION.md` (branch
`claude/sources-module-canonical-payor-enumerate`), my sections §7 and §10.

---

## 1. `DocumentInputTab.tsx` never sends `effective_date` — this is a live defect, not a backlog item

`ImportScrapedPagesRequest` (`app/main.py:7257`) declares **and persists** `effective_date` and
`termination_date`. The Upload UI collects `display_name`, `payer`, `state`, `program`,
`authority_level` — and **neither date field**.

```
$ grep effective_date frontend/src/components/tabs/DocumentInputTab.tsx
(no matches)
```

**Consequence:** every document imported through the Upload UI lands with `effective_date = NULL` **by
construction**. Not an oversight anyone can be asked to be more careful about — the field is absent from
the surface most documents arrive through.

This is the mechanism behind *"99.6% of 5,257 AHCA documents had no effective date"*, which is the
problem the current AHCA sprint exists to fix. It had been treated as a historical data-quality issue;
5,235 rows were backfilled this morning on that assumption. That cleaned the pool while the tap ran —
the next UI import reproduces it.

⚠️ **The `varchar → date` migration in flight does not fix this.** A NULL is not a malformed date, so
typing the column never forces anyone to supply one.

**Ask:** two `<input>` fields on the tab, populating the request body fields that already exist. Pair
with an edge validator (the sprint's §9 proposes `^\d{4}-\d{2}-\d{2}$`) so a bad value is a 422 with a
useful message rather than a silent write.

**Why it matters beyond tidiness:** `effective_date` is the **version-selection key** for the AHCA
reindex. Documents without one cannot be ordered against each other, which is precisely the
near-duplicate problem the sprint is trying to resolve.

## 2. `extra="forbid"` on `ImportScrapedPagesRequest` — safe to flip today

Without it, undeclared fields are dropped **silently with a 200**: sender sees success, data never
existed. Same defect class that recently made three separate callers of the crawler service believe in
features that were never wired.

**Precondition discharged — I enumerated every caller fleet-wide:**

| caller | fields sent | undeclared? |
|---|---|---|
| `DocumentInputTab.tsx:479` | `pages`, `display_name`, `payer`, `state`, `program`, `authority_level` | none |
| `DocumentInputTab.tsx:511` | identical set | none |
| `DocumentInputTab.tsx:585` | identical set | none |
| nested `pages[]` items | `url` / `text` / `html` — exactly what `ScrapedPageItem` (`:7250`) declares | none |
| crawler | not wired yet | n/a |
| chat / instant-rag | no reference anywhere | n/a |

**Nothing sends an undeclared field. Non-breaking, no migration path needed.**

*Self-correction, so you can weigh the evidence: my first enumeration reported "no callers at all"
because the grep omitted `.tsx`. I caught it on a broader search. The table above is the corrected pass.*

**Why this is more than hygiene:** two things the sprint needs to persist have **no declared field**, so
today they cannot be sent at all — they'd be accepted and discarded:

- **source filename / rule number** (`59G-4.013…`) — measured at ~26% of AHCA files, and the identifier
  humans actually search by
- **`content_signals`** — carrying AHCA's `Content-Signal: ai-train=no`, which is **binding**: that
  corpus must never train or fine-tune a model. A note in a coordination file is not an enforcement
  mechanism; it has to travel with the document.

`extra="forbid"` is what turns "we added a field and nothing happened" into a loud 422.

---

Rule either of these however you like, including *not now*. I'd just rather you heard it from whoever
measured it than inherit it third-hand.
