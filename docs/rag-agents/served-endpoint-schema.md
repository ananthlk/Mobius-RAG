# `/api/retriever/answer` response schema

For Chat Architecture's RAG cutover spec (2026-08-06). Answers the 5
questions asked directly; live example attached at the bottom, pulled from
a real production query against the real corpus (not fabricated).

Endpoint: `POST https://mobius-rag-ortabkknqa-uc.a.run.app/api/retriever/answer`
(no admin key needed — this is the real production route). Request body:
`{query, caller_mode, token_budget_for_retrieval, forced_strategy,
allocator_override, authority_requirement}` — all but `query` optional.

---

## 1. `chunks[]` array shape

Each element (from `contract.py`'s `build_contract`, sourced from
`CompiledCitation` in `synthesis_contracts.py`):

| field | type | meaning |
|---|---|---|
| `index` | int | 1-based, stable position — Chat/its LLM cites by this number |
| `chunk_id` | str | stable chunk identifier |
| `text` | str | the actual chunk content |
| `document_name` | str | resolved document title |
| `source_type` | str | code comment says `"internal"` \| `"external"` \| `"fact_store"`, but the live example below shows `"hierarchical"` — **flagging, not smoothing over: the documented enum and live data disagree**, see caveat below |
| `document_id` | str\|null | |
| `url` | str\|null | |
| `page_number` | int\|null | null in the live example (internal-doc chunk, no page attached) |
| `paragraph_index` | int\|null | null in the live example |
| `document_status` | str\|null | code comment says `"planned"` \| `"live"` \| null, but the live example shows `"completed"` — **same discrepancy as `source_type`**, flagging below |
| `authority` | str\|null | `"authoritative"` \| `"external"` \| `"planned"` \| null — the grounding-badge input |
| `verified` | bool | false iff a citation quote failed verification or was never checked |
| `is_neighbor` | bool | true if this chunk was pulled in as context around a matched chunk, not itself a direct match |
| `original_score` | float\|null | the filler's raw match score (NOT comparable across strategies — see caveat below) |
| `rerank_score` | float\|null | the filler's own composite ranking score, when it computed one |
| `filler_strategy` | str | **which strategy actually produced this chunk** (`a`/`b`/`c`/`d`/`s`) — this is the field to key off for "was this an s-strategy answer," not `chosen_slot` |
| `slot_id` | str | which query-slot this chunk fills |
| `slot_semantics` | str | the slot's shape, e.g. `"direct_answer"` — see §5, this is NOT a strategy identity |

**Direct quote from the code (Ananth's own note, `synthesis_contracts.py:21-23`):**
> "Field names line up with Chat's SourceRef (mobius-chat/app/skills/registry.py:60)
> so Chat needs no translation layer: document_name/source_type/url/
> document_id/page_number map directly."

Caveat already logged (Eval, 2026-07-24, not yet fixed): `original_score`
mixes incomparable scales across strategies (s's external confidence score
vs a/b's vector/BM25 scores) in the top-level `score` field's derivation —
does NOT affect the per-chunk data itself, every citation reaches Chat
unaffected, only the single diagnostic `score` number could look inflated.

**Two discrepancies found writing this doc, one fixed, one still open:**

**`source_type` — ROOT-CAUSED, FIXED, LIVE-CONFIRMED.** Live data showed
`source_type="hierarchical"` on a real b-served chunk, not the documented
`"internal"`. Root cause: `synthesis.py`'s old fallback
(`chunk.source_type or (...)`) only normalized when `source_type` was
falsy — but a/b's `candidate.source_type` is always truthy (it's the
corpus DB's own chunking/ingestion-taxonomy column, e.g.
`"hierarchical"`/`"flat"` — a completely different namespace from this
module's semantic enum), so the fallback never fired and the raw DB value
leaked straight into Chat's contract. Fixed to normalize by membership in
the known-semantic set instead of truthiness. Did **not** corrupt the
`authority` field for these chunks — `_infer_authority` already prefers
the precise `authority_level` signal over this allowlist, so grounding
badges were unaffected; only the raw exposed `source_type` value itself
was wrong. Deployed and re-verified against a fresh live query — the same
chunk that showed `"hierarchical"` earlier now correctly shows
`"internal"`. Safe to build against.

**`document_status` — found, root-caused, NOT yet fixed, needs Database
input before I guess at it.** Live value was `"completed"`, not the
documented `"planned"`/`"live"`. This one pulls straight from the corpus
DB's own document row (`document_status=row._mapping.get("document_status")`
in `pool/public_adapter.py`) with **no normalization attempt at all** —
it's almost certainly the ingestion/processing-pipeline status
(pending/processing/completed/failed), a different concept from the
policy-publication status ("is this content officially in effect yet")
the code assumes. This is more load-bearing than the source_type bug:
`document_status == "planned"` gates a real authority-suppression check
(`synthesis.py:383` — "planned content is never authoritative"), which may
never fire in this corpus if the real DB values never say `"planned"`.
I don't know the real schema well enough to guess the fix safely (is there
a separate policy-status column? is "planned" a dead code path entirely
for this corpus?) — checking with Database before touching this. **Do not
build UI/logic on Chat's side that trusts `document_status` meaning
"planned"/"live" until I confirm this is fixed** — treat it as un-normalized for now.

---

## 2. `traces` shape

Much smaller than legacy's `query_profile`/`confidence`/`themes` bundle —
most of that granularity moved into `routing_keys` (see §3) or was
genuinely retired with the old architecture:

```
"traces": {
  "gate_contour": str|null,     // Gate stage's classification of the query
  "gate_reason": str|null,      // why Gate classified it that way
  "reformat_posture": str|null, // Reformat stage's posture decision
  "narrative": str|null         // human-readable trace narrative, if generated
}
```

If you need `query_profile`-equivalent detail (query classification,
themes), that's NOT here — see §4 on `themes`/`theme_diagnostic`, likely
retired, not migrated.

---

## 3. `routing_keys` shape

This is the dense one — most of what replaces legacy's `routing`/
`candidate_pool`/`fast_exit`/`served` fields lives here:

| key | contents |
|---|---|
| `decision_id` | Router's decision identifier |
| `dispatch_path` | which allocator ran: `"greedy"` \| `"optimizer"` \| `"bayesian"` |
| `routing_ladder_per_slot` | the planned strategy sequence per slot |
| `executed_order` | `{slot_id: [strategies actually run, in order]}` — the REAL executed chain, not the plan |
| `observer_final_verdicts` | Observer's per-slot outcome verdict |
| `observer_final_reasons` | why Observer reached that verdict |
| `attempt_spans` | per-rung timing |
| `fill_depth` | `{strategy, capacity, occupancy}` per executed rung |
| `portfolio_fill` | `{strategy: {k_planned, k_delivered}}` |
| `prescreen_not_ready_deferred_slots` | strategies pushed behind a not-yet-ready alternative (readiness noise, not a performance signal) |
| `slots_filled` | count |
| `under_filled_count` | count |
| `ride_along_slots` | slots filled opportunistically alongside a required slot |
| `terminal_action` | how the routing ladder concluded |
| `routing_verdict` | `{outcome, terminal_action, helpers, confidence_bar, adjusted_bar, slots: {per-slot status/lb/terminal/required/helpers}}` — this is the closest analogue to legacy's `routing`/`confidence` bundle |
| `authority_requirement` | echoes what was requested/resolved for this call |
| `model_trace` | `[{slot_id, stage, model_id, call_id}]` — for the bandit reward path |
| `feature_context` | `{per_slot_pool_metadata: {...}}` — pool richness/depth signals per slot |

`routing_verdict` is likely your best source for whatever legacy's
`confidence` field fed — it's Router's own LB/confidence estimate, per
slot, not a single flattened number.

---

## 4. `grounding_markers` shape

```
"grounding_markers": {
  "unverified_citations": int,       // count of citations that failed/skipped verification
  "planned_status_citations": int,   // count of citations from "planned" (not yet live) docs
  "document_name_resolved": int,     // count resolved successfully
  "document_name_fallback": int      // count that fell back to a placeholder name
}
```

This does **not** map to legacy's `themes`/`theme_diagnostic` — those were
query-topic classification, which doesn't have an equivalent here as far as
I can tell from this module. If Chat needs topic/theme classification for
UI purposes, that's a gap to flag back to me, not something silently
droppable — I'd want to know it's actually used before ruling on it either
way.

---

## 5. `"s" strategy / `chosen_slot="direct_answer"` — direct answer

**Correcting my own earlier message to ReAct**: `chosen_slot="direct_answer"`
is NOT the s-strategy signal. It's the query's **slot shape** (this query
needs one direct single-fact answer, vs. a multi-part/comparison slot) —
independent of which strategy filled it.

**The real signal is `filler_strategy == "s"`** on the chunk(s), plus
`source_type == "fact_store"` on the same chunk (belt-and-suspenders — both
should agree).

**There is no separate `llm_answer`/`fact_score`/`fact_predicate` in the new
response.** The fact-store's answer content IS the chunk `text` field —
same shape as every other chunk, not a special side-channel. `original_score`
carries whatever confidence score the fact-store attached (this is the
"mixing incomparable scales" caveat from §1 — s's confidence score and a/b's
retrieval scores aren't on the same scale, so don't average/compare
`original_score` across a chunk set that mixes strategies).

If you need the fact-store's structured predicate (not just prose text),
that's a real gap — flag it back, I'd need to check whether `filler_s.py`
still carries that anywhere before this compile step, or whether it's
flattened to text-only by this point.

---

## Live example (real query, real corpus, no fabrication)

Query: `"What is the timely filing deadline for Sunshine Health FL Medicaid claims?"`,
`caller_mode="chat.default"`, `authority_requirement="citable_required"`.

Result: `dispatch_path="greedy"`, `status="ok"`, `chosen_slot="direct_answer"`,
10 chunks returned, all `filler_strategy="vector_rerank"` (strategy b — this
query didn't hit the fact store).

First chunk, exact bytes from the real response, captured POST-fix (this
is the same chunk that showed `source_type="hierarchical"` in an earlier
draft of this doc — this version is fresh, post-deploy, unedited):
```json
{
  "index": 1,
  "chunk_id": "083d75d5-41e1-4066-96b9-98e69d24ed45",
  "text": "Electronic Claim Submission Overview \nProviders are encouraged to participate in Sunshine Health's electronic claims/encounter filing program. Sunshine Health can receive ANSI XS12N 837 professional, institutional or encounter transactions. In addition, Sunshine Health can generate an ANSI X12N 835 electronic remittance advice known as an explanation of payment (EOP). For more information on electronic filing, contact Sunshine Health's Electronic Transactions (EDI) department by calling 1-800-225-2573, ext. 8025525, or send an email to EDIBA@sunshinehealth.com. Providers who bill electronically are responsible for filing claims within the same filing deadlines as providers filing paper claims...",
  "document_name": "Sunshine Provider Manual",
  "source_type": "internal",
  "document_id": "d9721756-d1b1-4cf4-845b-f44652c5fcf9",
  "url": null,
  "page_number": null,
  "paragraph_index": null,
  "document_status": "completed",
  "authority": "authoritative",
  "verified": true,
  "is_neighbor": false,
  "original_score": 0.5820004478546631,
  "rerank_score": null,
  "filler_strategy": "vector_rerank",
  "slot_id": "direct_answer",
  "slot_semantics": "direct_answer"
}
```

Note `rerank_score: null` here — not every filler computes a composite
rerank score, this one only had the raw vector-similarity `original_score`.
Note `document_status: "completed"` — this is the still-open,
NOT-yet-fixed issue described above; don't read meaning into this value
yet.

Full response, unedited, in `docs/rag-agents/served-endpoint-live-example.json`
(committed alongside this doc) if you want to walk the whole `routing_keys`/
`traces`/`grounding_markers` shape against real data rather than the
annotated tables above.
