"""Observer -- per-slot, per-attempt "would this slot benefit from another
turn?" (Step 4e of the answer engine, under Retriever).

Scope (Ananth via Retriever, 2026-07-23): each filler defines "good enough"
on its own bar -- Observer does NOT impose a universal confidence scale
across strategies, and does NOT make the cross-slot continuation call
(that's Router's decide_continuation() in app/services/router/
continuation.py, already built/tested). This module answers ONE slot's ONE
verdict at a time; Router aggregates N slots' verdicts + the time budget
into a single turn/done decision.

TURN SEMANTICS (Ananth, direct, 2026-07-23): "another turn" for a slot means
deploying the NEXT strategy already planned in that slot's ladder (e.g. a
filled the slot; if a's fill isn't sufficient, Router deploys d ON TOP OF
it) -- NOT re-running the current rung's own strategy again. So the bar
here is per-rung sufficiency (did THIS rung's result clear its own bar for
quality + fill), not "would repeating this exact call help" -- that framing
is what makes a uniform test possible even for a/b, which are deterministic
reranks of an already-fixed Pool (re-running the SAME rung truly can't
change anything, but that was never the question -- the next rung in the
ladder is a DIFFERENT strategy and can).

Verdict values are Router's, imported not redefined here -- Observer only
ever emits VERDICT_WOULD_BENEFIT / VERDICT_SATISFIED /
VERDICT_EXHAUSTED_ATTEMPTS. VERDICT_EXHAUSTED_BUDGET/VERDICT_ERROR are
orchestrator/infra-level states, never Observer's call (see continuation.py
module docstring). EXHAUSTED_ATTEMPTS means THIS SLOT has used up its
max_attempts_per_slot budget while still insufficient -- stop spending more
turns on it even if ladder rungs remain -- not "this one rung's signal
can't improve," which would wrongly block the next (different) rung from
ever running.

Every verdict carries a REASON string (Eval's non-negotiable ask,
2026-07-23) -- calibration exclusion rules key off both the verdict enum
and the reason text, not the enum alone.

STATUS 2026-07-23: a/b/c/s SIGNED OFF by Retriever -- not yet imported into
orchestrator.py, no live behavior change (Eval's read, confirmed by
Retriever: correct posture until Fillers+Synthesis+a calibration plan
land). Two open questions were resolved before finalizing:
(1) Sufficiency bar for a/b/s is capacity-aware (`not under_filled` ->
SATISFIED) -- Retriever's earlier "occupancy>0" phrasing was never a
ruling, just their own crude orchestrator-side stopgap talking; Ananth's
"deploy d IN ADDITION" wording is the actual, additive/capacity-aware
answer.
(2) Observer does NOT need remaining-ladder-rung visibility --
CALLER CONTRACT (Retriever's commitment): the orchestrator derives
`attempt_number`/`max_attempts` honestly from its own real ladder state
(cursor+1, len(chain)) on every `evaluate()` call. Given accurate inputs,
EXHAUSTED_ATTEMPTS here and continuation.py's independent
`"no_remaining_rungs"` drop are consistent, not competing -- both derive
from the same ladder state, just at different layers (Observer: per-rung
sufficiency; decide_continuation: cross-slot aggregation safety net).

Filler d's test (2026-07-24): filled to capacity AND no assigned chunk
scores below `_DECAY_FLOOR_RATIO` (0.6, same threshold filler_b.py already
uses) of the slot's own top-scoring chunk -- reuses an existing fleet
pattern against real BM25 scores every d-filled chunk already carries, no
new engineering. Proposed to the Web Search session 2026-07-24; Retriever
green-lit building directly against the proposal (their own bandwidth was
tied up on the speculative-search-prefetch build) rather than waiting
further, adjustable if they come back with a real objection. Does not
address the separate repeated-identical-chunk-across-URLs gap Web Search
found independently -- that's their call, as producer-owned hygiene.
(Earlier "bank-grading" attribution in Retriever's original kickoff message
did NOT check out on verification with Web Search directly -- disregarded,
not used as input to this design.)

DETERMINISM (Eval, via Retriever, 2026-07-23): a/b/s are DETERMINISTIC --
same input produces the identical result every time (a/b: pure rerank of an
already-fixed Pool; s: keyed fact-store lookup, same query -> same
hit/miss) -- so a same-rung retry has ZERO information gain for them, full
stop. c/d are NON-DETERMINISTIC (LLM sampling variance; live web fetch
variance) -- a same-rung retry of THOSE could genuinely surface something
different. Today's orchestrator never actually implements same-rung retry
for anyone (confirmed by Retriever: the routing ladder is a fixed sequence
of DISTINCT strategies, cursor only ever advances) -- so this distinction
has no behavioral effect yet. It's encoded explicitly via `_IS_DETERMINISTIC`
below anyway, per Eval's suggestion, so it's a fixed fact about each
filler's mechanism rather than something buried inside each verdict
function -- and so it's already in place the day same-rung retry (if ever
built) becomes meaningful for c/d specifically.
"""

from __future__ import annotations

from app.services.retriever.fillers.contracts import (
    ASSIGNMENT_REASON_LLM_PARTIAL_MATCH,
    ASSIGNMENT_REASON_LLM_RETRIEVED,
    ASSIGNMENT_REASON_LLM_RETRIEVED_EXTERNAL,
    FilledSlot,
)
from app.services.router.continuation import (
    VERDICT_EXHAUSTED_ATTEMPTS,
    VERDICT_SATISFIED,
    VERDICT_WOULD_BENEFIT,
)

# Same threshold filler_b.py's _rerank_vector_candidates uses for its own
# per-category decay floor -- reused here (not re-derived) for Filler d's
# sufficiency bar (_evaluate_web_search): an assigned chunk scoring below
# this fraction of the slot's own top-scoring chunk is treated as a
# decay-floor violation (likely low-relevance padding).
_DECAY_FLOOR_RATIO = 0.6

# Fixed fact about each strategy's mechanism, not a runtime decision -- see
# module docstring's DETERMINISM section. True = same input always produces
# the identical result (retrying the SAME rung has zero information gain);
# False = the strategy has real variance and a same-rung retry could
# genuinely differ (not wired as same-rung retry anywhere today, but the
# fact still belongs here, on the strategy, not inside a verdict branch).
_IS_DETERMINISTIC = {
    "a": True,   # pure rerank of an already-fixed Pool result
    "b": True,   # pure rerank of an already-fixed Pool result
    "c": False,  # LLM sampling variance
    "d": False,  # live web fetch variance
    "s": True,   # keyed fact-store lookup -- same query, same hit/miss
}


def is_deterministic(strategy_id: str) -> bool:
    """Whether `strategy_id`'s mechanism can produce a different result on
    an identical-input retry. Unknown strategies default to True (safest
    assumption: never claim retry value that hasn't been earned)."""
    return _IS_DETERMINISTIC.get(strategy_id, True)

# Which assignment_reason values mark an LLM-citation chunk (filler c) --
# the ONLY chunks where FilledChunk.quote_verified carries meaning (every
# other strategy leaves it at its None default). Imported from
# fillers/contracts.py, not hand-copied (Tech Review, 2026-07-24 -- three
# independent copies of the same load-bearing string was real drift risk;
# synthesis.py scopes its own `verified` check the same way, same fix).
_LLM_CITATION_REASONS = frozenset({
    ASSIGNMENT_REASON_LLM_RETRIEVED,
    ASSIGNMENT_REASON_LLM_RETRIEVED_EXTERNAL,
    ASSIGNMENT_REASON_LLM_PARTIAL_MATCH,
})


def evaluate(
    strategy_id: str,
    filled_slot: FilledSlot,
    *,
    attempt_number: int = 1,
    max_attempts: int = 1,
    filler_emit: dict | None = None,
) -> tuple[str, str]:
    """Per-slot, per-strategy verdict on THIS rung's sufficiency. Dispatches
    to the strategy's own "good enough" test -- see module docstring for
    turn semantics and which strategies have a real test today.

    Args:
        strategy_id: the rung's strategy letter ("a"/"b"/"c"/"d"/"s").
        filled_slot: this rung's FilledSlot output.
        attempt_number: how many turns this SLOT has already spent (1 on
            the first pass, across however many rungs it's tried so far).
            Only relevant to the WOULD_BENEFIT->EXHAUSTED_ATTEMPTS
            transition once the slot's attempt budget is used up.
        max_attempts: from ResourcePosture.max_attempts_per_slot
            (Structure's per-query posture), passed through by the
            orchestrator.
        filler_emit: the rung's FilledShape.emit dict, in case a handler
            needs more than what FilledSlot itself carries (unused today,
            accepted for forward compatibility).

    Returns:
        (verdict, reason) -- verdict is one of VERDICT_WOULD_BENEFIT /
        VERDICT_SATISFIED / VERDICT_EXHAUSTED_ATTEMPTS.
    """
    handler = _HANDLERS.get(strategy_id)
    if handler is None:
        return (
            VERDICT_EXHAUSTED_ATTEMPTS,
            f"observer_no_handler_for_strategy_{strategy_id}",
        )
    return handler(
        filled_slot,
        attempt_number=attempt_number,
        max_attempts=max_attempts,
        filler_emit=filler_emit or {},
    )


def _insufficient_verdict(attempt_number: int, max_attempts: int, reason: str) -> tuple[str, str]:
    """Shared WOULD_BENEFIT/EXHAUSTED_ATTEMPTS split once a handler decides
    its rung is NOT sufficient -- attempts remaining means "deploy the next
    rung"; attempts used up means stop, regardless of what rungs remain.
    """
    if attempt_number >= max_attempts:
        return VERDICT_EXHAUSTED_ATTEMPTS, f"{reason}_attempts_exhausted"
    return VERDICT_WOULD_BENEFIT, reason


def _evaluate_fact_store(
    filled_slot: FilledSlot, *, attempt_number: int, max_attempts: int, filler_emit: dict,
) -> tuple[str, str]:
    """s NEVER independently satisfies a slot -- a hit is real, certified
    evidence, but only ever supplementary, so the verdict is always
    WOULD_BENEFIT (or EXHAUSTED_ATTEMPTS once the slot's turn budget is
    used up), whether s hits or misses.

    BUG FIX, TWO ROUNDS (2026-07-24). Round 1 (real live evidence via
    Retriever's calibration harness, query cmhc001): the original bar was
    bare `occupancy > 0`, on the wrong assumption that "a fact-store slot
    has capacity 1, one verified fact answers it fully" -- filler_s
    actually fires on the SAME `direct_answer` slot as a/b/c/d (capacity =
    the query's whole needed breadth, e.g. 10), verified directly in
    `shape/slots.py`. First fix made this capacity-aware (`not
    under_filled`).

    Round 2 (Eval, same day): capacity-aware still isn't the right
    CRITERION, even though it happened to be behaviorally correct today --
    it makes `capacity == 1` a PROXY for "s owns a dedicated fact slot,"
    but no such slot type exists anywhere in the `AnswerSlot` contract
    (`slot_semantics` is only direct_answer/thematic_exploration/
    external_context, and filler_s only ever fires on the shared
    direct_answer one). Verified this is genuinely unreachable today, not
    just unlikely: `shape/structure.py`'s breadth formula
    (`round(_BASE_K * recall_demand / _BASELINE_RECALL_DEMAND)`) has a real
    minimum of 7 across every defined `caller_mode` (`chat.copilot`:
    round(10*0.70/0.95)=7) -- capacity can never legitimately be 1 today.
    Eval's sharper rule doesn't depend on that formula never changing:
    since s can NEVER see anything but a shared content slot in the
    current data model, it should NEVER independently emit SATISFIED, full
    stop -- not "unless capacity happens to be 1." Simpler and more robust
    than round 1's capacity check, and consistent with Router's own
    already-shipped planning-time rule: "s may never be the SOLE rung of a
    required content slot -- s supplements, never substitutes." This
    applies the same supplement-not-substitute principle at execution
    time, not a new one.

    Live evidence (cmhc001): s's hit was the bare string "68069" -- a real
    fact-store value, but not remotely an answer to the actual question;
    the resulting (loop-stopped-after-s) answer scored 0.00 on the judge
    rubric.

    REVISIT TRIGGER (Eval, 2026-07-24, forward note): "s never satisfies"
    is only correct because no dedicated fact-slot TYPE exists in the
    `AnswerSlot` contract today -- if one is ever added (a slot where a
    single certified fact genuinely IS the authoritative answer, e.g. a
    pure "what's the payer ID" lookup), this always-insufficient rule
    becomes wrong: s would loop pointlessly into a/b/c/d that add nothing.
    Don't build slot-type-aware logic for a slot type that doesn't exist
    yet (that's this decision, correctly, for v1) -- but when that day
    comes, the clean fix is keying SATISFIED off slot-type (fact slot -> s
    satisfies; shared content slot -> s supplements, this function's
    current behavior), not re-deriving another proxy.
    """
    if filled_slot.occupancy > 0:
        return _insufficient_verdict(
            attempt_number, max_attempts, "fact_store_hit_supplementary_not_sole_answer",
        )
    return _insufficient_verdict(attempt_number, max_attempts, "fact_store_miss_or_gate_failed")


def _evaluate_llm_retrieval(
    filled_slot: FilledSlot, *, attempt_number: int, max_attempts: int, filler_emit: dict,
) -> tuple[str, str]:
    """Sufficient only when the slot is filled to capacity AND every
    LLM-citation chunk's quote actually verified (`quote_verified is True`)
    -- any unverified chunk or unfilled capacity is insufficient.

    BUG FIX (2026-07-24, found investigating a Tech Review flag on a
    different, smaller issue): this used to check
    `assignment_reason == "llm_partial_match"` alone, which conflates two
    different things filler_c's `quote_verified` field already tells apart
    -- `False` (the LLM's quote was checked and did NOT match, a real
    hallucination) and `None` (the LLM gave NO quote at all, so nothing
    could be checked). A quote-less citation gets assignment_reason
    "llm_retrieved" (nothing to falsify) with `quote_verified=None` -- the
    old check silently read that as "verified", overstating confidence
    exactly the way Eval already ruled against for Synthesis's own
    `verified` flag ("nothing could be falsified" is not "confirmed" --
    `synthesis.py:588`'s `chunk.quote_verified is True`, the same fix,
    applied here now instead of a second, inconsistent definition of
    "verified" living in this module).
    """
    if filled_slot.occupancy == 0:
        return _insufficient_verdict(attempt_number, max_attempts, "llm_retrieval_empty")

    unverified = sum(
        1 for c in filled_slot.chunks
        if c.assignment_reason in _LLM_CITATION_REASONS and c.quote_verified is not True
    )
    if unverified == 0 and not filled_slot.under_filled:
        return VERDICT_SATISFIED, "all_citations_verified_filled_to_capacity"
    if unverified > 0:
        reason = f"{unverified}_of_{filled_slot.occupancy}_citations_unverified"
    else:
        reason = f"under_filled_{filled_slot.occupancy}_of_{filled_slot.capacity}_verified"
    return _insufficient_verdict(attempt_number, max_attempts, reason)


def _evaluate_bm25(
    filled_slot: FilledSlot, *, attempt_number: int, max_attempts: int, filler_emit: dict,
) -> tuple[str, str]:
    """Sufficiency = filled to capacity. Filler a is a pure, deterministic
    rerank of Pool's already-fixed candidates -- re-running THIS rung can't
    change the result, but that's not what WOULD_BENEFIT triggers (see
    module docstring): it signals Router to deploy the slot's NEXT planned
    rung on top of a's fill, which is exactly the right call when a
    under-fills or comes back empty.
    """
    if not filled_slot.under_filled:
        return VERDICT_SATISFIED, "bm25_filled_to_capacity"
    return _insufficient_verdict(
        attempt_number, max_attempts,
        f"bm25_under_filled_{filled_slot.occupancy}_of_{filled_slot.capacity}",
    )


def _evaluate_vector(
    filled_slot: FilledSlot, *, attempt_number: int, max_attempts: int, filler_emit: dict,
) -> tuple[str, str]:
    """Sufficiency = filled to capacity. Same reasoning as _evaluate_bm25 --
    Filler b is also a pure deterministic rerank of Pool's already-fixed
    vector-arm candidates (post length-floor/dedup/decay-floor), and the
    same "next rung, not same rung" turn semantics applies.
    """
    if not filled_slot.under_filled:
        return VERDICT_SATISFIED, "vector_filled_to_capacity"
    return _insufficient_verdict(
        attempt_number, max_attempts,
        f"vector_under_filled_{filled_slot.occupancy}_of_{filled_slot.capacity}",
    )


def _evaluate_web_search(
    filled_slot: FilledSlot, *, attempt_number: int, max_attempts: int, filler_emit: dict,
) -> tuple[str, str]:
    """Sufficiency = filled to capacity AND no assigned chunk falls too far
    below the slot's own top-scoring chunk. Proposed to the Web Search
    session 2026-07-24 (no reply yet as of writing; Retriever green-lit
    building it directly against this proposal, adjustable if they come
    back with a real objection): reuses filler_b's own decay-floor pattern
    (`_DECAY_FLOOR_RATIO`, same 0.6 threshold as filler_b.py's
    `_rerank_vector_candidates`) against the real BM25 scores every
    d-filled chunk already carries (`FilledChunk.original_score`, from
    `_score_bm25` -- the same ranking function Pool computes for Filler a).

    Catches the "nominally full but padded with a low-relevance chunk near
    the bottom" failure mode -- the boilerplate/junk pattern Web Search
    flagged -- without needing a dedicated boilerplate classifier (none
    exists; see Web Search's own confirmation). Does NOT address the
    separate repeated-identical-chunk-across-URLs gap Web Search found --
    that's their call to fix as producer-owned hygiene, independent of this
    sufficiency check.

    Two known limitations, confirmed by Web Search on review (2026-07-24),
    not silently absorbed:
    - **capacity=1 slots get zero signal from this check.** A relative
      floor needs at least two scores to compare; with one chunk, `floor ==
      score * 0.6 <= score` always holds, so the decay branch can never
      fire -- a capacity=1 d-filled slot is sufficiency-tested purely on
      "filled to capacity", same as if this function only checked
      `under_filled`. Not a bug (there's no second chunk to be "padding"),
      but worth knowing this check adds nothing for that slot shape.
    - **BM25 does not cleanly separate off-topic-but-vocabulary-sharing
      content from genuinely relevant content** -- a passage that happens
      to repeat query terms without answering the query can still clear
      the decay floor. Unverified risk, not a proven failure (no live case
      demonstrating it yet) -- flagged by Web Search as a real gap in what
      this signal can catch, distinct from the boilerplate/padding case it
      DOES catch.
    """
    if filled_slot.under_filled:
        return _insufficient_verdict(
            attempt_number, max_attempts,
            f"web_search_under_filled_{filled_slot.occupancy}_of_{filled_slot.capacity}",
        )

    scores = [c.original_score for c in filled_slot.chunks if c.original_score is not None]
    if not scores:
        # Shouldn't happen (_chunk_from_passage always sets original_score,
        # falling back to 1.0 on a BM25-scoring failure) -- fail open rather
        # than crash a sufficiency check over missing diagnostic data.
        return VERDICT_SATISFIED, "web_search_filled_to_capacity_no_score_data"

    floor = max(scores) * _DECAY_FLOOR_RATIO
    below_floor = sum(1 for s in scores if s < floor)
    if below_floor == 0:
        return VERDICT_SATISFIED, "web_search_filled_to_capacity_no_decay_floor_violation"
    return _insufficient_verdict(
        attempt_number, max_attempts,
        f"web_search_{below_floor}_of_{len(scores)}_chunks_below_decay_floor",
    )


_HANDLERS = {
    "a": _evaluate_bm25,
    "b": _evaluate_vector,
    "c": _evaluate_llm_retrieval,
    "d": _evaluate_web_search,
    "s": _evaluate_fact_store,
}
