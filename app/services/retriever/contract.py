"""Contract -- Step 6 of the answer engine, the frozen 12-field response
envelope (module-gates.md §6). One emitter: every code path in the
retriever pipeline (converge/reroute/empty/escalate/diverge, in the legacy
terminology) funnels through `build_contract()`, not ad hoc dict-building
at each return site.

FIELD LIST -- resolved a real discrepancy before building (two "frozen"
specs disagreed): `module-gates.md` §6 lists 12 fields and is the one every
other doc cites (`gate-emit-schema-spec.md`: "wire into the 12-field
contract's thinking_trace slot"); `retriever-build-checklist.md`'s Module
#6 lists 13 (adds `alternative_scores`/`feature_vector`, both legacy
`routing_dump`-era concepts tied to the old `corpus_search_agent.py`
architecture). Built against `module-gates.md`'s 12: `{query, chosen_slot,
score, chunks[], answer_text, thinking, traces, routing_keys,
grounding_markers, latency_ms, attempt_count, status}`.

REAL SCOPE NOTE on `answer_text`/`thinking` (Ananth's correction,
2026-07-24): the original module-gates.md spec assumed Synthesis authors
the answer text directly ("Synthesis outputs (answer, thinking,
grounding)" as this module's input) -- that assumption is now WRONG.
Synthesis compiles; Chat authors. Chat is a separate service, not part of
this pipeline's call chain in this environment. So `answer_text`/`thinking`
are genuinely optional here: `build_contract()` accepts them as caller-
supplied params (threaded back from wherever Chat's response lands), and
they're `None` when this envelope is built before Chat has run -- which is
the honest, common case for this pipeline emitting to Chat, not Chat
emitting to a user. This is NOT a stub or an oversight; it's the correct
shape given the real, corrected module boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.retriever.orchestrator import RetrieverPartialResult
from app.services.retriever.synthesis_contracts import SynthesisResult


@dataclass
class ContractEnvelope:
    """The frozen 12-field response. Field order/NULL semantics are
    byte-compat P0 (module-gates.md §6) -- do not reorder, do not add
    fields here; a genuinely new field needs a new spec revision, not a
    quiet addition."""

    query: str
    chosen_slot: str | None
    score: float | None
    chunks: list[dict] = field(default_factory=list)
    answer_text: str | None = None
    thinking: str | None = None
    traces: dict[str, Any] = field(default_factory=dict)
    routing_keys: dict[str, Any] = field(default_factory=dict)
    grounding_markers: dict[str, Any] = field(default_factory=dict)
    latency_ms: dict[str, int] = field(default_factory=dict)
    attempt_count: int = 0
    status: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "chosen_slot": self.chosen_slot,
            "score": self.score,
            "chunks": self.chunks,
            "answer_text": self.answer_text,
            "thinking": self.thinking,
            "traces": self.traces,
            "routing_keys": self.routing_keys,
            "grounding_markers": self.grounding_markers,
            "latency_ms": self.latency_ms,
            "attempt_count": self.attempt_count,
            "status": self.status,
        }


def _pick_chosen_slot(synthesis_result: SynthesisResult | None) -> tuple[str | None, float | None]:
    """The "primary" slot for the top-level chosen_slot/score fields --
    prefers a required "direct_answer"-semantics slot with real citations;
    falls back to the first slot with any citations; None if nothing
    resolved. `score` is that slot's best (max) original_score among its
    citations -- the closest real analogue to legacy's "strategy_score"
    now that scores aren't comparable across strategies (Pool/Observer's
    percentile-normalization reasoning applies here too: this is a
    best-effort single number for the envelope, not a calibrated score).

    KNOWN, SCOPED, DEFERRED (Eval, 2026-07-24): `best_score()` below takes
    `max(original_score)` across ALL citations in a slot regardless of
    which strategy produced them -- mixing s's external Payor Platform
    confidence score with a/b's vector/BM25 scores, the SAME incomparable-
    scales problem Pool's `top_score_percentile` already had and fixed
    (excluding tag_select's raw coverage count from that max). Real but
    narrow: does NOT affect `chunks[]` (every citation still reaches Chat
    unaffected) or calibration (Eval grades judge_score + per-strategy
    chunks, never this envelope field) -- only this diagnostic `score`
    number, which could report an inflated s-derived value even when a/b
    hold the real answer. Deferred behind persist/capacity work (lower
    value, not corrupting anything load-bearing) -- when picked up, reuse
    Pool's normalization decision (exclude s, and any other
    incomparable-scale strategy, from this cross-strategy max) rather than
    re-deriving a policy from scratch."""
    if not synthesis_result or not synthesis_result.slots:
        return None, None

    def best_score(slot) -> float:
        scores = [c.original_score for c in slot.citations if c.original_score is not None]
        return max(scores) if scores else 0.0

    direct_answer_slots = [
        s for s in synthesis_result.slots
        if s.slot_semantics == "direct_answer" and s.citations
    ]
    if direct_answer_slots:
        chosen = max(direct_answer_slots, key=best_score)
        return chosen.slot_id, (best_score(chosen) or None)

    any_filled = [s for s in synthesis_result.slots if s.citations]
    if any_filled:
        chosen = max(any_filled, key=best_score)
        return chosen.slot_id, (best_score(chosen) or None)

    # Nothing filled -- report the first slot's id (honest "we tried, got
    # nothing" signal) rather than a bare None, unless there are no slots.
    return synthesis_result.slots[0].slot_id, None


def _derive_status(partial_result: RetrieverPartialResult, synthesis_result: SynthesisResult | None) -> str:
    """Overall outcome, one word, for the envelope's `status` field.
    Deliberately coarse -- per-slot nuance (verdict/reason) lives in
    `routing_keys`/`traces`, not flattened into this single field."""
    if partial_result.filled_shape is None:
        # No-retrieval posture (CLARIFY/DECLINE) or no slots -- not an
        # error, a real terminal outcome Gate/Reformat already decided.
        return "no_retrieval"
    if synthesis_result is None:
        return "filled_no_synthesis"
    if not synthesis_result.citations:
        return "empty"
    any_unverified = synthesis_result.telemetry.unverified_citations > 0
    any_under_filled = any(s.under_filled for s in synthesis_result.slots)
    if any_unverified or any_under_filled:
        return "partial"
    return "ok"


def build_contract(
    partial_result: RetrieverPartialResult,
    synthesis_result: SynthesisResult | None = None,
    *,
    answer_text: str | None = None,
    thinking: str | None = None,
) -> ContractEnvelope:
    """The one emitter. Every code path threads through this -- do not
    build a response dict anywhere else in the pipeline.

    `synthesis_result` is optional because Synthesis isn't wired into
    `orchestrator.py`'s production loop yet (same pre-wiring posture as
    Observer) -- this function degrades honestly (chunks=[], grounding
    empty) rather than crash, same convention every module in this chain
    uses for "the thing upstream of me isn't real yet."
    """
    chosen_slot, score = _pick_chosen_slot(synthesis_result)

    chunks = (
        [
            {
                "index": c.index, "chunk_id": c.chunk_id, "text": c.text,
                "document_name": c.document_name, "source_type": c.source_type,
                "document_id": c.document_id, "url": c.url,
                "page_number": c.page_number, "paragraph_index": c.paragraph_index,
                "document_status": c.document_status, "authority": c.authority,
                "verified": c.verified,
                "is_neighbor": c.is_neighbor, "original_score": c.original_score,
                "slot_id": c.slot_id, "slot_semantics": c.slot_semantics,
            }
            for c in synthesis_result.citations
        ]
        if synthesis_result else []
    )

    router_decision = partial_result.router_decision
    ladder = router_decision.routing_ladder if router_decision else None
    posture = router_decision.resource_posture if router_decision else None

    # routing_verdict -- Chat's forward requirement (2026-07-24): the full
    # shape Router already locked (per_slot_status/lb/terminal + the
    # decision-level outcome/terminal_action/helpers), assembled here from
    # RoutingLadder + ResourcePosture rather than invented -- every field
    # below is a real, already-emitted Router value, not new computation.
    routing_verdict = (
        {
            "outcome": ladder.outcome,
            "terminal_action": ladder.terminal_action,
            "helpers": ladder.helpers,
            "confidence_bar": posture.confidence_bar if posture else None,
            "adjusted_bar": ladder.adjusted_confidence_bar,
            "slots": {
                sid: {
                    "status": ladder.per_slot_status.get(sid, ""),
                    "lb": ladder.per_slot_lb.get(sid),
                    "terminal": ladder.per_slot_terminal.get(sid),
                    "required": next(
                        (s.required for s in partial_result.slots.slots if s.slot_id == sid),
                        None,
                    ) if partial_result.slots else None,
                    "helpers": ladder.per_slot_helpers.get(sid, []),
                }
                for sid in ladder.per_slot
            },
        }
        if ladder else {}
    )

    # model_trace -- Chat's forward requirement for the bandit reward path.
    # Synthesis's CompiledSlot.model_trace now carries {stage, model_used,
    # llm_call_id} per slot (synthesis.py, Eval's ruling 2026-07-24: `stage`
    # in, `attempt_latency_ms` out -- that belongs in Timing's attempt_spans,
    # not here, to avoid duplicating the same timing data two ways).
    # `latency_ms` is deliberately NOT in this shape -- see attempt_spans in
    # filled_shape.emit for per-attempt timing instead.
    model_trace = (
        [
            {
                "slot_id": s.slot_id,
                "stage": s.model_trace.get("stage"),
                "model_id": s.model_trace.get("model_used"),
                "call_id": s.model_trace.get("llm_call_id"),
            }
            for s in synthesis_result.slots if s.model_trace
        ]
        if synthesis_result else []
    )

    routing_keys = {
        "decision_id": getattr(router_decision, "decision_id", None),
        "dispatch_path": getattr(router_decision, "dispatch_path", None),
        "routing_ladder_per_slot": (ladder.per_slot if ladder else {}),
        "executed_order": (
            partial_result.filled_shape.emit.get("executed_order", {})
            if partial_result.filled_shape else {}
        ),
        # --- Chat's forward requirements (2026-07-24), folded into this
        # existing dict field rather than added as new top-level dataclass
        # fields -- the envelope's 12-field COUNT is frozen (byte-compat
        # P0), but routing_keys is itself a dict and was already the
        # natural home for routing-metadata sub-keys.
        "terminal_action": ladder.terminal_action if ladder else None,
        "routing_verdict": routing_verdict,
        "authority_requirement": posture.authority_requirement if posture else None,
        "model_trace": model_trace,
        # suggested_links: NOT included here on purpose -- Filler f
        # (Sitemap)'s output routing (FilledShape field vs bypass) is still
        # an open question with DB/Retriever as of 2026-07-24. Adding a
        # key here now would either be empty in practice or fabricated;
        # revisit once that routing question resolves.
    }

    grounding_markers = (
        {
            "unverified_citations": synthesis_result.telemetry.unverified_citations,
            "planned_status_citations": synthesis_result.telemetry.planned_status_citations,
            "document_name_resolved": synthesis_result.telemetry.document_name_resolved,
            "document_name_fallback": synthesis_result.telemetry.document_name_fallback,
        }
        if synthesis_result else {}
    )

    latency_ms = {
        "gate_ms": partial_result.gate_ms,
        "reformat_ms": partial_result.reformat_ms,
        "slots_ms": partial_result.slots_ms,
        "pool_ms": partial_result.pool_ms,
        "router_ms": partial_result.router_ms,
        "fillers_ms": partial_result.fillers_ms,
        "synthesis_ms": (synthesis_result.telemetry.compile_ms if synthesis_result else 0),
        "total_ms": partial_result.total_ms,
    }

    # Total attempts across ALL slots -- sum of each slot's real executed
    # rung count, from the multi-turn loop's own telemetry (executed_order),
    # not a synthetic count. A slot the loop never touched (no_slots posture)
    # contributes 0, correctly.
    attempt_count = sum(
        len(v) for v in routing_keys["executed_order"].values()
    )

    traces = {
        "gate_contour": getattr(partial_result.gate, "contour", None),
        "gate_reason": getattr(partial_result.gate, "reason", None),
        "reformat_posture": getattr(partial_result.reformat, "posture", None),
        "narrative": partial_result.narrative,
    }

    status = _derive_status(partial_result, synthesis_result)

    return ContractEnvelope(
        query=partial_result.query,
        chosen_slot=chosen_slot,
        score=score,
        chunks=chunks,
        answer_text=answer_text,
        thinking=thinking,
        traces=traces,
        routing_keys=routing_keys,
        grounding_markers=grounding_markers,
        latency_ms=latency_ms,
        attempt_count=attempt_count,
        status=status,
    )
