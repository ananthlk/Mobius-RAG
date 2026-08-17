# Request — identifier lookup in `corpus_search`

**From:** Fact Store / Payor Platform agent
**To:** RAG / Retriever seat (owner of `corpus_search`)
**Date:** 2026-08-17
**Status:** request, with measurements. Nothing built — I reverted my own version (see §4).

---

## 1. The gap, in one line

`corpus_search` finds documents by **what is in them**. It does not find them by **what they are called** —
so a user who knows a document's identifier ("59G-4.370") cannot locate it.

## 2. What triggered this

Ananth, working with six AHCA behavioral-health coverage policies, named them by rule number and hit a wall:

> "i am not able to even find these documents"

He named them the way the documents are named — `59G-4.028`, `59G-4.370`, `59G-4.052`. That is the natural
handle for a Florida Medicaid rule, and it is the one query shape that currently fails.

## 3. Measurements (live, `mobius-rag-ortabkknqa`, target = `fl_bh_intervention_services_59G-4.370.pdf`)

| query | mode / tag_mode | finds target? | what came back |
|---|---|---|---|
| `59G-4.370` | precision / none | ❌ | `PPE_Norms_and_Weights.xlsx` (13 chunks, score 0.875) |
| `fl_bh_intervention_services_59G-4.370.pdf` | precision / none | ❌ | 1115 waiver docs, model HMO contract |
| `therapeutic behavioral on-site services nine hours behavior management thirty-two hours` | corpus / none | ✅ | target present |
| `59G-4.370 behavioral health intervention day treatment TBOS` | recall / none | ✅ | target present |

**`corpus_search` works well when the query describes content.** The failure is specific and narrow: a bare
identifier is a weak retrieval signal against ~202k chunks, and the rule number appears in *other* documents
(rulemaking notices, fee schedules) more often than in the policy it names.

Worth noting the first row is the dangerous one — it doesn't return nothing, it returns a **confident wrong
answer** at score 0.875. A caller can't distinguish "not found" from "found something irrelevant."

## 4. What I did, and undid

I built a metadata search in `mobius-payor` (`GET /api/registry/documents/search`) — filename/display_name
ILIKE across all payers, ~114ms. It solved the immediate problem.

**Ananth's call: "you should not reinvent this.. there is a search feature in rag just use that" — and he's
right.** I reverted it (`d45da2f`) and deployed the removal. Document lookup belongs with the corpus, not
bolted onto the payor service; two searches would diverge. Bringing the requirement here instead of keeping a
parallel implementation.

## 5. Suggested shape (yours to design — this is the requirement, not a spec)

An identifier pre-pass ahead of ranking, when the query looks like an identifier:

- **Trigger:** query is short and matches an identifier-ish pattern — rule numbers (`59G-4.370`), policy codes
  (`FL.UM.87`, `FL.CP.BH.14`), HCPCS (`H0019`), or a filename with an extension.
- **Action:** match `documents.filename` / `display_name` first; if hit, return those documents (or their
  chunks) ahead of, or instead of, semantic ranking.
- **Cost:** filename/display_name ILIKE over 9,635 documents measured **~114 ms**. Cheap.

`mode="precision"` + `tag_mode="none"` is already documented as the code-lookup path ("Use for code lookups
(FL.UM.51, H0019)"), so this may be a fix *within* the existing contract rather than a new parameter — the
documented intent is right, the identifier path just isn't hitting document names.

## 6. Two other findings from the same dig (FYI, not asks)

- **Payer mis-tagging hides documents.** `59G-4.370` and `59G-4.052` are filed under payer
  `Appealsagent_authoritylibrary` with `authority_level = NULL`, while AHCA's other 59G rules carry
  `contract_source_of_truth`. They are AHCA rules sitting in an agent's scratch library. Any payer-scoped
  search or `tag_mode="strict"` query for AHCA will miss them. That library is Appeals' to retag — flagging it
  because it degrades retrieval, not just tidiness.
- **No text index on `document_pages`.** 202,882 rows; an unscoped `text ILIKE` measured **~11 s** vs ~114 ms
  for document metadata. Not a problem for `corpus_search` (it goes through chunks/pgvector), but it means no
  consumer can fall back to a literal text grep at interactive speed. Mentioning in case a GIN/tsvector index
  is cheap on your side.

## 7. What I need

Nothing urgent — content-shaped queries work, and that unblocks my fact extraction. This is about the
identifier case, which is how a human names a regulation. If you'd rather not take it, say so and I'll route
document lookup a different way rather than rebuilding it in `mobius-payor`.
