# Filler a (BM25) — Progress Tracker

**Status:** BUILT + UNIT-TESTED (v0.1) — prototype ready. **BLOCKED: Eval requires before/after calibration plan sign-off before ship.**

**Sign-off status:**
- ✅ Chat, UX, DB all clean
- ⚠️ Eval: conditional-green, requires calibration plan (in progress)
- ⚠️ TECH: answer pending

**Owner:** Filler a agent (this session).

**Sequence:** Query → Shape → Pool → **Fillers (Filler a, Step 3a)** → Router → Observer → Synthesis.

---

## Build Status

### Bug Fixed 🐛 → ✅

**Critical bug caught by Retriever code review (2026-07-23):**
- Code was reading `PoolCandidate.score` (arm-specific signals) instead of `PoolCandidate.bm25_score` (BM25 ranking).
- Silent bug: tests passed because mock data happened to have consistent `.score` values, masking the issue.
- **Fix:** Updated filler_a.py to read `.bm25_score` in all 3 locations (filter, sort, original_score passthrough).
- **Tests added:** `test_reads_bm25_score_not_generic_score()` catches this class of bug (opposite-ranking test case).
- **Result:** 15/15 tests passing (up from 14), 2 skipped (Eval pending).

### Completed ✅

- **contracts.py** — `FilledChunk`, `FilledSlot`, `FilledShape` dataclasses.
- **filler_a.py** — `fill_shape_bm25()` core algorithm.
  - Sorts PoolCandidate by score (BM25 from Pool) descending.
  - Assigns non-overlapping chunks to slots in priority order.
  - Respects slot capacity, detects under-fill.
  - Pure Python, zero DB/embed calls (gate b, single-pool principle).
- **Unit tests** — 10 tests, all passing.
  - BM25 ranking verification
  - Under-fill detection
  - Empty pool / all-neighbors edge cases
  - Capacity enforcement
  - Determinism
  - Diagnostics emit
  - Large-scale (100 candidates, 10 slots)
  - Field preservation
- **Integration tests** — 4 tests, all passing (+ 2 skipped pending Eval).
  - Realistic Pool output (multi-arm: tag_select + vector + inherited + neighbors)
  - Diagnostics passthrough
  - Field preservation (tags, document_status, content_sha, etc.)
  - Neighbor filtering (score=None excluded)
- **Test coverage:** 14/14 passing, 2 skipped (eval calibration TBD).
  - Total: ~280 lines of test code, characterization + integration.

### Blockers 🚧

- **Pool: BM25 score field** ✅ SHIPPED 2026-07-23
  - `PoolCandidate.bm25_score: float | None` is live via `ts_rank_cd(search_vec, plainto_tsquery(...), 32)`
  - All match candidates (tag_select, vector, inherited) have real float scores
  - Neighbors have `bm25_score=None` (positional adjacency only)
  - Filler a code verified working (tests passing)

- **Eval: Calibration plan approval** ⚠️ BLOCKING SHIP
  - Eval conditionally-blocked on before/after calibration (like Pool got)
  - Calibration plan drafted: `docs/rag-agents/filler-a-calibration-plan.md` (top-level, not mobius-rag-nested)
  - Routed to Eval for protocol sign-off (open Qs on baseline, metrics, thresholds, per-arm analysis)
  - Once approved: run cmhc-26 baseline vs proposed, confirm no regression

- **TECH: Parent spec confirmation** ⚠️ PENDING
  - TECH's direct answer on fillers-schematic-spec.md still unconfirmed
  - Expected to clear once Eval signs off (same dependency chain as Pool)

### Design Decisions (Resolved)

- **Non-overlapping assignment** — once a chunk is assigned to a slot, it's removed from the remaining pool. Prevents slot-crossing duplication, respects priority order.
- **Null-score filtering** — candidates with `score=None` (pure neighbors) are filtered before ranking. Neighbors typically inherit scores from nearest match; v1 doesn't score them independently.
- **Capacity as hard ceiling** — no overflow (v1); unassigned candidates are logged as overflow in diagnostics (future Router handling).

---

## Input Contract (read-only from Pool + Shape)

**PoolResult** (from Pool, Step 2):
- `candidates[]` — each with `chunk_id`, `document_id`, `text`, `score` (BM25, from Pool), `is_neighbor`, etc.
- `query` — original query string.
- `segment_ms`, `strategy_hint` — passthrough for diagnostics.

**AnswerShapeResult** (from Shape/Slots, Step 1d):
- `slots[]` — each with `slot_id`, `slot_semantics`, `capacity`, `required`, `priority`.

---

## Output Contract (FilledShape to Router, Step 4)

```python
FilledShape:
  slots[]{
    slot_id: string
    slot_semantics: enum (direct_answer | thematic_exploration | external_context)
    capacity: int
    chunks[]{
      chunk_id, document_id, text, document_status, source_type, tags
      is_neighbor, original_score (BM25), assignment_reason
    }
    occupancy, under_filled, over_filled
  }
  total_chunks_assigned: int
  filling_strategy: "bm25"
  emit: { fillers_decision, slots_filled, empty_slots, under_filled, per_slot_details[] }
```

---

## Next Steps

### Immediate (waiting)
1. **Pool response** — confirm BM25 score added to PoolCandidate.score.
2. **DB review** — TECH + DB verify query-cost side of the ts_rank_cd addition.

### Short-term (in progress)
1. ✅ **Integration tests** — Filler a + real PoolResult from Pool (passing).
2. ✅ **Calibration infrastructure built:**
   - `filler_baseline.py` — uniform top-N control (no semantic filtering)
   - `calibrate.py` — harness to run baseline vs proposed on query bank
   - Ready to invoke once orchestration clarified
3. **Awaiting:** Retriever's guidance on how to invoke Pool/Shape for cmhc-26 queries
4. **Next:** Run calibration, verify acceptance criteria, report to Eval

### Medium-term (fillers pipeline)
1. Filler a ships (v0.1) once sign-offs clear.
2. Fillers b/c/d/e/f/q/s fork as independent sessions (same pattern, different strategies).
3. All fillers union → Router (Step 4, being redesigned as an optimizer).

---

## Architecture Notes

**Single-pool principle (gate b):**
- ALL DB access lives in Pool, upstream of Fillers.
- Filler a is pure logic: sort + assign.
- Future fillers (semantic, diversity, etc.) are also pure logic over the same pool.
- This keeps performance predictable (no hidden queries) and reasoning transparent (Fillers doesn't know about DB).

**Non-overlapping assignment:**
- Slots filled in order of priority.
- Once a chunk is used, it's removed from remaining candidates for downstream slots.
- Prevents double-counting and ensures fair distribution.
- Unassigned overflow → Router (future escalation/ranking).

---

## Known Gaps / Future Work

- v1: Slot-semantic-specific filtering deferred (all slots draw from the same scored pool).
- v1: Neighbors with no score are filtered. Future: inherit score from nearest match or use a generic neighbor-rank.
- v1: No fallback synthesis for under-filled slots (Router handles that downstream).
- Eval: before/after calibration pending (same NUMBER-MOVING gate as every other module).
