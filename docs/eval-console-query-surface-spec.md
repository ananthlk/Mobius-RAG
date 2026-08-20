# Eval Console — Query surface · build-ready spec

**Owner:** Eval seat · **Status:** spec for build (UX-only, no backend changes) ·
**Replaces:** the "Run a Trace" tab of `/trace-explorer` · **Shell:** collapsible
left rail + right panel (see mockup `d04f0355`).

> The Query surface is the primary use case: **run one query, watch it think,
> review the telemetry — with eval off, reference-free, or against a golden
> answer.** It is built entirely on endpoints that already exist. No backend
> change is required; if any field below is missing from a response it is a
> display fallback, never a new API.

---

## 1. Job to be done

A person types a query, chooses how the engine should run it and whether to
grade it, hits Run, and then **reads what happened** — the chunks chat received,
the per-stage telemetry, the full chunk-by-chunk trace, and (if graded) the
verdict with its supporting facts. One query, one screen, no tab-hopping to
follow it.

---

## 2. Layout (inside the right panel)

```
┌ right panel ───────────────────────────────────────────────┐
│  Query                                    [engine: healthy] │  header + live pill
│  ┌ Composer (minimal) ──────────────────────────────────┐  │
│  │ [ query text …………………………………………………… ]                 │  │
│  │ (Normal) (Vs golden)              [⋮]  [ Run query → ] │  │  summary chips + kebab
│  └────────────────────────────────────────────────────────┘ │
│  retrieval › gate › synthesis › grade      live · 3 of 4     │  stage flow
│  ┌ KPI row ─────────────────────────────────────────────┐   │  retr / synth / auth / chunks·ms
│  [ Emits | Telemetry | Detailed trace | Eval ]              │  facet tabs
│  ┌ facet body (Gate · Router · Synthesis panels, tables) ┐   │
│  └────────────────────────────────────────────────────────┘ │
│  Recent traces ▾                                            │  from /history
└─────────────────────────────────────────────────────────────┘
```

**Config lives behind the ⋮ kebab (confirmed 2026-08-20).** The page stays down
to the query and Run. The **⋮** opens a popover with all run settings; a compact
**summary-chip row** shows the current config at a glance without the controls on
the page. This kebab pattern is shared by every surface (Query, Question bank, …)
so there is one config interaction to learn.

```
[⋮] popover
  Run type       [Normal | Thinking]
  Evaluate       [Off | Reference-free | Vs golden]
  Golden answer  [ …… ]   ‹revealed inside the popover only in “Vs golden”›
  ▸ Advanced     Forced strategy [none|a|b|c|d|s] · Fan-out queries
```

---

## 3. Composer → request mapping

Endpoint: **`POST /admin/trace-explorer/run`** (sync) and
**`GET /admin/trace-explorer/run-stream?…`** (SSE, live). Auth: `X-Admin-Key`
header (shared localStorage `trace_explorer_admin_key`).

All controls below live inside the **⋮ kebab popover**, not on the page. The
composer on the page is only: the query field, the summary-chip row, ⋮, and Run.

Request body is `TraceExplorerRequest`:

| Control | Field | Values | Notes |
|---|---|---|---|
| Query | `query` | string | required; empty ⇒ Run disabled |
| Run type | `caller_mode` | `chat.default` (Normal) · `chat.thinking` (Thinking) | default `chat.default` |
| **Eval mode** | `run_eval` + `must_facts` | see §4 | the one novel control |
| Golden answer | *(→ `must_facts`)* | string, decomposed to facts | visible **only** in Golden mode |
| Advanced · forced strategy | `forced_strategy` | none·`a`·`b`·`c`·`d`·`s` | none = allocator picks |
| Advanced · fan-out | `force_fanout_queries` | string[] | optional, power-user |

`allocator_override` / `authority_requirement` are **not** on this endpoint
(they belong to the bank run); do not surface them here.

---

## 4. Eval mode — the control that defines this surface

Three states, each a different measurement contract:

| Mode | `run_eval` | golden | What is scored | Reads from `eval.*` |
|---|---|---|---|---|
| **Off** | `false` | — | nothing — telemetry only | *(no `eval` block)* |
| **Reference-free** | `true` | none | groundedness · citation · abstention · confidence-calibration | `coverage` (retriever), `hallucinated_claims`, reliability |
| **Vs golden answer** | `true` | yes | answer-correctness · retriever recall · citation | `coverage_answer` (synthesis), `coverage`, `facts[]`, `verdict`, `score` |

**Rules the UI enforces:**
- Golden field appears **only** in "Vs golden"; its content is sent as
  `must_facts` (the gold facts the answer must contain).
- "Reference-free" is the honest production-shaped grade — no gold exists, so it
  measures *faithful-to-retrieval + calibrated*, never *correct*. Label it that
  way; never show a fabricated answer-correctness number.
- If the ruler is degraded/rerouting, the Eval tab shows a **quarantine** state,
  not a score (judge ≠ locked `factcheck/gemini-2.5-pro` ⇒ don't grade).
- **Calibrated ≠ correct** — a reference-free "grounded + calibrated" result can
  still be wrong on a superseded chunk; the tab says so.

---

## 5. Live stage flow

Prefer `run-stream` (SSE) so the run reports itself; fall back to `run` (sync)
with an indeterminate flow if SSE is unavailable. Stages, in order:

`retrieval → gate → synthesis → grade`  (router · reformat · structure · slots
are sub-steps surfaced inside Detailed trace, not top-level stages).

- Each stage: pending → active → done, with elapsed.
- Plain language: "synthesis · 2.1s" — never `stage=synthesis,status=running`.
- No bare spinner; the flow *is* the progress indicator.

---

## 6. Result facets (tabs) → response mapping

The response is the existing `/run` result. Field names below are exactly what
the current page consumes, so the new UI is a re-skin of known data.

### Emits — "what chat actually received"
`d.emits[]` — the final chunks handed to chat. Per row: `doc`, `score`,
`authority_level`, `rank`. This is the ground truth of the answer's inputs.

### Telemetry — per-stage timings
`d.telemetry` — stage durations; render as a horizontal time-strip
(retrieval_ms / gate / synthesis / total). Use `tabular-nums`.

### Detailed trace — the full mechanism
`d.detailed_trace` (dt):
- **Gate** — `dt.gate` with `p_codes`, `j_codes` (procedure / jurisdiction
  codes the gate matched). Panel with code chips.
- **Router** — `dt.router.dispatch_path` (which arms ran, in order).
- **Reformat** — `dt.reformat.posture`, `dt.reformat.fanout_themes`.
- **Structure** — `dt.structure.structure_ms`; **Slots** — `dt.slots`.
- **Synthesis** — `dt.synthesis.citations[]`, plus stats `st.chunks_in` /
  `st.chunks_out` / `st.citations_trimmed_for_budget` / `st.neighbors_added` /
  `st.fusion_content_merged`.
- **Candidate table** — per-chunk: `rank`, `bm`, `composite_score`,
  `authority_sig`, `authority_level`, `sim_sig`, `length_sig`, `doc`. Sortable;
  `tabular-nums`.

### Eval — the grade (only when `run_eval`)
`d.eval`: `verdict`, `score`, `coverage` (retriever recall), `coverage_answer`
(synthesis recall), `facts[]` (each: `fact`, `support`, `in_chunk`,
`contradicted`, `passage`), `judge_model`, `synth_answer`,
`hallucinated_claims[]`, `fact_checker_version`. In Reference-free mode also
render the **reliability read** (stated confidence vs groundedness). Always show
`judge_model` — it is the trust stamp.

**KPI row** (above the tabs, always visible when a result exists):
retriever `coverage` · synthesis `coverage_answer` · authority · chunks·ms.
The **retriever − synthesis gap** is highlighted — it is the calibration lever
(the 2026-08-20 baseline showed +23.8pp).

---

## 7. States

| State | Trigger | Treatment |
|---|---|---|
| No key | admin key absent / 401 | inline "enter admin key"; Run disabled |
| Idle | fresh | composer only; empty result area with a hint |
| Running | run in flight | stage flow live; tabs show skeleton |
| Done | result returned | KPI + tabs populated |
| Error | 5xx / engine error | error card with the message + retry |
| Degraded | rerun saturating API | banner: "retrieval is degraded right now — this measures contention, not quality" |
| Quarantine | judge ≠ locked pro | Eval tab shows quarantine, not a score |

---

## 8. Recent traces

`GET /admin/trace-explorer/history?limit=30` → list under the result. Each row
deep-links (`?trace_id=`) and reloads via `GET …/result?trace_id=`. Following a
past trace is identical to a fresh one — same tabs, same fields.

---

## 9. Component inventory (build checklist)

1. Composer (query input · **summary-chip row** · **⋮ kebab popover** holding
   run-type seg, eval-mode seg, conditional golden field, advanced disclosure ·
   Run button + disabled logic). Kebab opens/closes on click + outside-click;
   choices reflect into the summary chips; golden field reveals inside the
   popover only in "Vs golden" mode.
2. Stage-flow strip (SSE-driven; sync fallback)
3. KPI row (4 tiles, gap highlight)
4. Facet tab-bar (Emits / Telemetry / Detailed / Eval)
5. Emits table · Telemetry time-strip · Candidate table (sortable)
6. Gate panel · Router panel · Reformat panel · Synthesis panel (+ stats)
7. Eval panel (verdict, facts table, synth answer, judge stamp, reliability)
8. Recent-traces list
9. Cross-cutting: degraded banner, quarantine state, error card, key gate

All styled through `--mobius-*` tokens; DM Sans / JetBrains Mono; `tabular-nums`
on every numeric column.

---

## 10. Acceptance criteria (testable)

- [ ] Run with Eval=Off → no `eval` request, no Eval tab content; telemetry
      renders.
- [ ] Eval=Reference-free → `run_eval:true`, no golden field sent; Eval shows
      groundedness/citation/abstention + reliability, **no** answer-correctness.
- [ ] Eval=Vs golden → golden field visible and required; sent as `must_facts`;
      Eval shows `coverage_answer`, `facts[]`, verdict.
- [ ] Run type toggles `caller_mode` between `chat.default`/`chat.thinking`.
- [ ] Stage flow advances retrieval→gate→synthesis→grade with elapsed.
- [ ] Every facet renders from the real `/run` response fields in §6.
- [ ] `judge_model` is shown whenever a grade is shown.
- [ ] KPI row highlights the retriever−synthesis gap.
- [ ] Degraded-engine banner appears when the API is under a rerun.
- [ ] Recent trace deep-link (`?trace_id=`) reloads a past run into the same UI.
- [ ] No literal colours/fonts; all through Mobius tokens.

---

## 11. Endpoints used (no new backend)

- `POST /admin/trace-explorer/run` — sync run (+eval)
- `GET  /admin/trace-explorer/run-stream` — SSE live variant
- `GET  /admin/trace-explorer/result?trace_id=` — reload a past trace
- `GET  /admin/trace-explorer/history?limit=` — recent traces

All gated by the existing `/admin/*` auth; page shell reveals no data.

---

## 12. Phasing

- **P0** — composer + sync `/run` + KPI + four tabs + eval Off/Ref-free/Golden.
  Ships the whole job on the sync endpoint.
- **P1** — SSE live stage flow; reliability curve in Reference-free.
- **P2** — recent-traces list + `?trace_id=` deep-link; candidate-table sort.

---

*Pending UX-architect review: composer density, the eval-mode control pattern,
and the Detailed-trace panel hierarchy. Grounded in `mobius-design/BRANDING.md`.*
