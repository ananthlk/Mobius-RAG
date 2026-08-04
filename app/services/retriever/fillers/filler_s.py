"""Filler s -- Payor Platform Fact Store strategy (Step 3s of the answer engine).

Live external call: queries mobius-payor's certified Payor Fact Store
(POST {MOBIUS_PAYOR_URL}/api/skills/v1/fact_query), NOT a PoolResult consumer
like Fillers a/b. Explicit, documented exception to the parent Fillers spec's
"zero DB/embed side effects" -- same exception c/d already have.

Fires only for direct_answer-semantics slots (covers both the EXACT-posture
"direct_answer" slot and CLARIFY_REPHRASE's "best_guess" slot). A fact-store
hit answers one query's one fact -- it has no natural per-theme or
external-context analog, so thematic_exploration/external_context slots are
left untouched by this filler.

v1 ships tags-only (no `embedding` in the request), matching legacy
strategy-s's current, already-calibrated operating point exactly. Sending
`embedding` rescales the payor service's blend formula itself
(base = tag_overlap -> base = 0.5*tag_overlap + 0.5*vec_sim) and would
silently drop some currently-good serves below tau -- deliberately deferred
to a fast-follow bundled with Eval's alpha/beta/tau re-sweep and dropping
RAG's _CONCEPTUAL_MARKERS band-aid. See docs/rag-agents/filler-s-payor-module-spec.md §6.

See docs/rag-agents/fillers-schematic-spec.md and
docs/rag-agents/filler-s-payor-module-spec.md.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass

import httpx

from app.services.retriever.pool.contracts import PoolResult
from app.services.retriever.shape.slots import AnswerShapeResult
from app.services.retriever.fillers.contracts import (
    FilledChunk,
    FilledSlot,
    FilledShape,
)
from app.services.retriever.fillers.payer_context import extract_payer_slug

logger = logging.getLogger(__name__)

_FACT_QUERY_PATH = "/api/skills/v1/fact_query"
_DEFAULT_FACT_STORE_URL = "https://mobius-payor-ortabkknqa-uc.a.run.app"
_FACT_STORE_TIMEOUT_S = 15.0

# Matches legacy's hardcoded k=5 (corpus_search_agent.py:3865). Not scaled by
# resource_posture.breadth like other fillers' widths: verified directly in
# fact_store.py that `shortlist` is sorted by score BEFORE truncating to `k`
# (fact_store.py:403-404), so `k` only bounds the diagnostic shortlist's
# length -- it never changes which fact wins `shortlist[0]` (the one actually
# served). No correctness reason to vary it by posture for v1.
_DEFAULT_K = 5

# Ported verbatim from corpus_search_agent.py:3825-3831 -- no better
# conceptual-vs-factual intent signal exists anywhere in the new pipeline
# (checked directly against Gate's real Contour vocabulary, which is a
# tag/doc-coverage classification, not an intent classification).
_CONCEPTUAL_MARKERS = (
    "philosophy", "approach", "why does", "why do", "how does", "how do",
    "explain", "tell me about", "overview", "describe",
    "understanding", "background on", "rationale",
)


@dataclass
class RoutingLadder:
    """Strategy sequence per slot (from Router/two-phase design). Unused v1, same as a/b."""

    slot_id: str
    strategy_sequence: list[str]  # e.g., ["a", "b", "s"]


def _is_conceptual(raw_query: str) -> bool:
    lowered = raw_query.lower()
    return any(marker in lowered for marker in _CONCEPTUAL_MARKERS)


def _gate_passes(raw_query: str, tag_matches: list[str], slot) -> bool:
    # extract_payer_slug is the fleet-shared primitive (payer_context.py,
    # built for c/d/f/s) -- reuse it rather than re-deriving the
    # j:payor.* check inline.
    return (
        slot.slot_semantics == "direct_answer"
        and extract_payer_slug(tag_matches) is not None
        and not _is_conceptual(raw_query)
    )


def _build_request(raw_query: str, tag_matches: list[str]) -> dict:
    return {
        "query": raw_query,
        "d_tags": [t for t in tag_matches if t.startswith("d:")],
        "p_tags": [t for t in tag_matches if t.startswith("p:")],
        "j_tags": [t for t in tag_matches if t.startswith("j:")],
        "intent_scope": None,
        "k": _DEFAULT_K,
    }


def _stable_fact_id(served: dict) -> str:
    """Content-derived id for a served fact, stable across repeat calls.

    `served` carries no fact_id (the payor fact-store tracks one server-side
    as `top["fact_id"]` but never surfaces it here) -- only the per-call
    `telemetry_id`, which is a fresh uuid every request even when the same
    fact is served again. Using telemetry_id as chunk_id/document_id made
    the same query return a different chunk identity every run, breaking
    any downstream citation/cache keyed on it (caught via 3x-repeat
    determinism check, 2026-07-23). Hash the fact's own content instead --
    identical fact in, identical id out.
    """
    key = "|".join(str(served.get(k) or "") for k in (
        "payer_key", "record_type", "predicate", "answer_text",
    ))
    return f"fact_{hashlib.sha1(key.encode()).hexdigest()[:16]}"


def _chunk_from_hit(served: dict, telemetry_id: str | None, tag_matches: list[str]) -> FilledChunk:
    source_ref = served.get("source_ref") or {}
    synthetic_id = _stable_fact_id(served)
    return FilledChunk(
        chunk_id=synthetic_id,
        document_id=source_ref.get("doc_id") or synthetic_id,
        text=served.get("answer_text") or "",
        document_status=None,
        content_sha=None,
        source_type="fact_store",
        # authority_level (2026-07-27, Ananth): "s always authoritative" —
        # a certified Payor Platform fact, the highest-confidence source in
        # the fleet, same top tier as the DB's own contract_source_of_truth
        # documents. Explicit rather than relying solely on synthesis.py's
        # _AUTHORITATIVE_SOURCE_TYPES fallback (source_type="fact_store" is
        # already in that set) so s participates consistently in the same
        # authority_level-first precedence as a/b/d.
        authority_level="contract_source_of_truth",
        tags={
            "d_tags": [t for t in tag_matches if t.startswith("d:")],
            "p_tags": [t for t in tag_matches if t.startswith("p:")],
            "j_tags": [t for t in tag_matches if t.startswith("j:")],
        },
        is_neighbor=False,
        original_score=served.get("score"),
        assignment_reason="fact_store_hit",
        filler_strategy="fact_store",
        # `url` intentionally omitted -- FilledChunk has no such field yet
        # (shared, pending-DB-landing gap with Filler c/d).
    )


async def fill_shape_fact_store(
    pool_result: PoolResult,
    shape_result: AnswerShapeResult,
    raw_query: str,
    *,
    tag_matches: list[str],
    routing_ladders: list[RoutingLadder] | None = None,
    fact_store_url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> FilledShape:
    """
    Query the Payor Fact Store for the query's direct_answer slot only.

    Algorithm:
    1. For each slot in shape_result.slots, check the gate (§4 of the module
       spec): slot_semantics == "direct_answer" AND a payer tag matched AND
       the query isn't conceptual.
    2. If the gate passes for a slot, POST to the fact store (tags-only v1,
       no embedding -- see module docstring). On hit, assign ONE synthesized
       FilledChunk to that slot. On miss/error/gate-fail, leave the slot at
       occupancy 0 -- Observer's existing per-slot loop decides the next
       RoutingLadder rung, no force_s/fallthrough special-casing needed.
    3. Slots with thematic_exploration/external_context semantics are left
       untouched -- a fact-store hit has no natural per-theme analog.

    `pool_result`/`routing_ladders` are accepted for signature consistency
    with Fillers a/b but unused in v1 (same as a/b's own "unused v1" note).

    Args:
        pool_result: Output from Pool (Step 2). Unused in v1.
        shape_result: Output from Shape (Step 1).
        raw_query: The query's raw text (single call, no per-slot rewrite --
            fact-store lookups aren't theme-specific).
        tag_matches: The query's d:/p:/j: tag codes (from Gate's GateResult).
        routing_ladders: Optional per-slot strategy sequences (unused v1).
        fact_store_url: Override for MOBIUS_PAYOR_URL (mainly for tests).
        http_client: Injected httpx.AsyncClient (mainly for tests). A fresh
            client is created and closed per call if not supplied.

    Returns:
        FilledShape with only direct_answer-semantics slots potentially filled.
    """
    base_url = (
        fact_store_url
        or os.environ.get("MOBIUS_PAYOR_URL")
        or _DEFAULT_FACT_STORE_URL
    ).rstrip("/")

    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=_FACT_STORE_TIMEOUT_S)

    filled_slots: list[FilledSlot] = []
    total_assigned = 0
    per_slot_emit: list[dict] = []

    try:
        for slot in shape_result.slots:
            filled_slot = FilledSlot(
                slot_id=slot.slot_id,
                slot_semantics=slot.slot_semantics,
                capacity=slot.capacity,
                required=slot.required,
            )

            gate_passed = _gate_passes(raw_query, tag_matches, slot)
            hit = None
            telemetry_id = None
            fact_store_ms = None

            if gate_passed:
                payload = _build_request(raw_query, tag_matches)
                start = time.monotonic()
                try:
                    resp = await client.post(
                        f"{base_url}{_FACT_QUERY_PATH}", json=payload
                    )
                    fact_store_ms = int((time.monotonic() - start) * 1000)
                    if resp.status_code == 200:
                        data = resp.json()
                        hit = bool(data.get("hit"))
                        telemetry_id = data.get("telemetry_id")
                        if hit:
                            served = data.get("served") or {}
                            filled_slot.chunks = [
                                _chunk_from_hit(served, telemetry_id, tag_matches)
                            ]
                    else:
                        logger.warning(
                            "[filler_s] fact_query non-200 status=%d", resp.status_code
                        )
                        hit = False
                except Exception as exc:  # network/timeout -- clean miss, don't crash
                    fact_store_ms = int((time.monotonic() - start) * 1000)
                    logger.warning("[filler_s] fact_query error (clean miss): %s", exc)
                    hit = False

            filled_slot.occupancy = len(filled_slot.chunks)
            filled_slot.under_filled = filled_slot.occupancy < slot.capacity
            filled_slot.over_filled = False

            filled_slots.append(filled_slot)
            total_assigned += filled_slot.occupancy

            per_slot_emit.append(
                {
                    "slot_id": slot.slot_id,
                    "gate_passed": gate_passed,
                    "hit": hit,
                    "telemetry_id": telemetry_id,
                    "fact_store_ms": fact_store_ms,
                }
            )
    finally:
        if owns_client:
            await client.aclose()

    emit = {
        "fillers_decision": "fact_store_query",
        "slots_filled": len([s for s in filled_slots if s.occupancy > 0]),
        "empty_slots": len([s for s in filled_slots if s.occupancy == 0]),
        "under_filled": len([s for s in filled_slots if s.under_filled]),
        "total_chunks_assigned": total_assigned,
        "per_slot_details": per_slot_emit,
    }

    return FilledShape(
        slots=filled_slots,
        total_chunks_assigned=total_assigned,
        filling_strategy="fact_store",
        emit=emit,
    )
