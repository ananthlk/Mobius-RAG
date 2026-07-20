# RAG progress-emit spec (perceived latency)

Owner (contract): EVAL. Build: emit agent (transport) · RAG agent (emit points) · Chat agent (receive/render).
Goal (Ananth 2026-07-19): *"do a lot of emits in RAG just so the user thinks the process is faster and doing a lot of work."*
Today the ~30s warm query is a **silent** spinner: chat blocks on RAG's synchronous call and RAG's stage
progress goes only to Cloud Run logs (`_log_stage`, keyed by `search_id`) — never to the user.

## 1. The transport is already 90% there
- `corpus_search_agent(..., correlation_id=…)` **already receives the chat correlation_id** (agent.py:2626).
- Chat **already has** a live progress channel keyed by that cid: `chat_progress_events` table + Redis pub/sub
  + SSE `/chat/stream/:id`, with strict serial ordering per cid (`app/storage/progress.py`).
- So the only missing hop: **RAG publishes stage events on that cid → they stream to the UI through chat's existing SSE.** Zero new UI work.

**FINALIZED transport (emit agent, 2026-07-20):** label lives **chat-side**, so RAG sends only the enum +
numeric fields — **zero label text crosses the wire** (PHI §4 holds by construction, stronger than sending a label).
- RAG: `emit_progress(correlation_id|None, event, **fields)` in `app/services/progress_emit.py`. No-op if cid falsy;
  else fire-and-forget POST (~2s timeout, swallows ALL exceptions, never raises) to
  `{CHAT_INTERNAL_URL}/internal/progress/{cid}` body `{"event": event, "seq": <RAG monotonic per-cid>, **fields}`.
  `fields` = counts/enums/round only. Interpolated events (`themes`, `ranking`) send **`fields={"n": <int>}`** —
  key `n` matches chat's template placeholder (locked so RAG/chat can't drift).
- Chat: `POST /internal/progress/{cid}` maps `event` → label via a fixed dict **sourced verbatim from §2 below**,
  then calls the EXISTING `append_thinking(cid, label)` (same pipe react_loop's own emits use → zero UI/SSE work).
  204/no-op if cid isn't a known active request; never errors back to RAG.
(Alt: shared Redis channel `mobius:chat:progress:{cid}` — rejected: couples RAG to chat's store.)

## 2. Emit points → user-facing labels (I own this map; extends the diagnostics content tree)
Emit at the **agent-orchestration phase** level, not the micro `_log_stage` level. Fire EARLY and OFTEN —
the point is felt momentum, so emit on *entry* to each phase (before the slow work), plus count-updates.

| seq | event | fires when | label (templated) |
|----|-------|-----------|-------------------|
| 1 | `understanding` | on entry, after cid bound | "Understanding your question…" |
| 2 | `fact_check` | fact-store gate runs (payer j-tag only) | "Checking certified payer facts…" |
| 3 | `searching` | strategy a/b retrieval starts | "Searching the knowledge base…" |
| 4 | `themes` | strategy b clusters themes | "Found {n} themes — narrowing in…" |
| 5 | `ranking` | rerank starts | "Ranking {n} passages by relevance…" |
| 6 | `external` | strategy d escalation | "Checking external payer sources…" |
| 7 | `composing` | synthesis LLM call starts | "Composing the answer…" |
| 8 | `verifying` | grounding/observe — **NOT WIRED (grounding is async-only today; no inline point). Expected live stream = 7 events, not 8. Reserved for when/if an inline grounding path lands.** | "Verifying against sources…" |

- **fact-store fast-exit path** collapses to `understanding` → `fact_check` → `composing` (skip 3–6). That's the
  certified-serve; it *should* feel instant, so the few emits are honest, not padding.
- Escalation `b→d` fires both `searching`/`themes` **and** `external` — matches the "show both ACT branches" rule.
- Labels are **templates**: stage name + integer counts only. **Never** interpolate query text or chunk text.

## 3. Two hard invariants (non-negotiable; I gate these)
**PHI (Policy §4):** an emit label is shown to the user, so it may NOT carry raw query/chunk text. Counts and
fixed stage strings only. And the emit payload must NOT be logged raw at the RAG or chat side (log `event`+`seq`,
not a free-text label built from user input — but since labels are templated, this holds by construction).
Do **not** echo the query in any emit. This is the same class as the `[trace:classify]` raw-token leak.

**Eval-safety:** the emitter is **optional**. Eval calls `corpus_search`/agent with **no cid / no emitter** →
every emit is a no-op. Eval reads the **final JSON** fields; it must never gate on emit presence, ordering, count,
or timing. (Already true — eval passes no chat cid.) A missing/half emit stream must NEVER change the answer,
the grades, or the decision row. Emits are cosmetic; the answer path is authoritative.

## 3b. INTEGRATION DEFECT found in verification (2026-07-20) — chat-side, cross-instance drop
Verified end-to-end and it FAILED: RAG emits reach chat's endpoint on the correct turn cid (confirmed in chat
request logs — POSTs to `/internal/progress/{turn_cid}` return 204), but the labels do NOT surface in the UI,
**non-deterministically** (one injected marker appeared, an identical retry didn't).

ROOT CAUSE (chat): the `/internal/progress/{cid}` endpoint calls `append_thinking(cid, line)`, which opens with
`if correlation_id not in _progress: return` — `_progress` is a **per-process in-memory dict**, populated only on
the instance that owns the turn. Chat runs **minScale=2** (2–20 instances, load-balanced). RAG's external POST
lands on whatever instance the LB picks — usually NOT the owning one — so the cid isn't in that instance's
`_progress` and the event is **dropped at the guard, before the DB-persist/Redis-publish**. Chat's OWN thinking
emits never hit this because they run in-process on the owning instance.

FIX (chat, ~5 lines): the endpoint must write to the SHARED cross-instance channel regardless of local ownership.
Do NOT route the external push through `append_thinking`. Instead build the event dict and call
`_publish_progress_event(cid, ev)` directly (the primitive at progress.py:288 that always persists to
`chat_progress_events` + Redis-publishes). The live SSE stream reads via `get_progress_events_from_db`
(progress.py:291, DB-poll, cross-instance-safe), so a DB-persisted event surfaces on any instance.
CONTRACT GAP (emit agent): the transport contract treated an external cross-service push like a local emit;
it must specify "persist to the shared store, never gate on local in-memory turn ownership."

## 4. Ownership / build split
- **EVAL (me):** this contract — stage→label map, PHI + eval gates, integration verification (run a live query,
  confirm the 8-event stream reaches the UI in order AND a headless eval run is byte-identical in its final JSON).
- **emit agent:** transport (the `/internal/progress` hop + a tiny RAG-side `emit(event, **fields)` helper that
  no-ops when cid is None), ordering/seq, fleet-consistency with other agents' emit shapes.
- **RAG agent:** drop the 8 `emit(...)` calls at the phase entries above (the `_log_stage` sites are the anchors).
- **Chat agent:** the receive endpoint + map RAG events into the existing progress store (they already render).

## 5. Done = 
1. Live chat fact query + corpus query each stream their event sequence to the UI, in order, mid-flight.
   (Expect **7** live events max — `verifying` is unwired, async-only grounding; RAG commit 5fca401.)
2. The 30s warm corpus query now *shows motion* the whole time (no >~4s silent gap).
3. A headless eval run (no cid) produces the identical final JSON it does today — emits are invisible to eval.
4. No raw query/chunk text in any emit label or in logs.
5. **`CHAT_INTERNAL_URL` is set on RAG's live revision.** If unset, `emit_progress` fails safe (no-ops) and the
   whole stream silently never fires — same class as the unset-env-var that killed the trace writer. Verify the
   serving revision's env has it AND points at chat's reachable internal URL, not just that the code deployed.
