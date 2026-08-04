"""Dataclass contracts for the synthesis module (Step 5 of the answer engine).

Synthesis does NOT author the answer text -- Chat does (Ananth's direct
correction, 2026-07-24, superseding this module's original "text + citations"
kickoff framing). Synthesis's real job: take Fillers' per-slot FilledChunks
(Step 3, fillers/contracts.py) + Observer's per-slot verdicts (Step 4e,
observer.py) and produce ONE compiled, reranked, deduped, neighbor-complete,
citation-ready package -- what Chat authors the answer from, Eval grades
against, and Router's bandit reward path can draw on.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CompiledCitation:
    """One chunk in the final compiled output, citation-ready.

    Field names line up with Chat's SourceRef (mobius-chat/app/skills/
    registry.py:60) so Chat needs no translation layer: document_name/
    source_type/url/document_id/page_number map directly.
    """

    index: int  # 1-based, stable position -- Chat/its LLM cite by this
    chunk_id: str
    document_name: str  # resolved at compile time -- never present upstream, see synthesis.py's _resolve_document_names
    text: str
    source_type: str  # "internal" | "external" | "fact_store" (Tech Review's catch, 2026-07-24: this comment previously omitted fact_store, a real third value synthesis.py's own _NON_CORPUS_SOURCE_TYPES/_AUTHORITATIVE_SOURCE_TYPES already handle explicitly)
    # "planned" | "live" | None -- Product-Awareness's reality gate. No
    # default on purpose (Product-Awareness's sign-off condition,
    # 2026-07-24): every construction site must explicitly decide this
    # value rather than silently inheriting a default -- None is still a
    # legitimate value (e.g. external chunks with no document row), but it
    # must be a deliberate None, not an omitted field. Synthesis passes
    # this through unchanged, never filters/transforms it -- Chat is the
    # enforcement layer (suppresses "planned" from cited_source_indices).
    document_status: str | None
    document_id: str | None = None
    url: str | None = None
    page_number: int | None = None
    paragraph_index: int | None = None
    content_sha: str | None = None
    # Chat's `SourceRef.authority` field (registry.py:60) -- the grounding
    # badge computes "grounded" when all sources are authoritative, and
    # nothing upstream carried this forward for the new Synthesis path
    # (Chat's sign-off gap, 2026-07-24). Inferred here from source_type/
    # document_status, same pattern as `verified`'s inference from
    # assignment_reason -- see synthesis.py's _infer_authority. Coarser
    # than Pool's own `PoolCandidate.authority_level` (contract_source_of_
    # truth/operational/fyi -- a genuinely more precise signal that exists
    # upstream but is dropped before FilledChunk; flagged to Retriever as a
    # separate, deeper gap, not fixed here).
    authority: str | None = None  # "authoritative" | "external" | "planned" | None
    verified: bool = True  # False iff filler c's citation quote did not verify or was never given (chunk.quote_verified is not True)
    is_neighbor: bool = False
    original_score: float | None = None
    # The filler's own final composite ranking score when it computed one
    # (see fillers/contracts.py's FilledChunk.rerank_score) -- threaded
    # through so Synthesis's own trim/rerank steps can respect the
    # filler's real ranking instead of falling back to raw original_score
    # alone (2026-07-29 fix: that fallback was unconditional before, which
    # silently discarded every non-bm25 signal a filler used to rank a
    # chunk). None for chunks from fillers that don't compute a composite.
    rerank_score: float | None = None
    # Which filler's scoring formula produced this chunk's ranking (see
    # fillers/contracts.py's FilledChunk.filler_strategy for the full
    # rationale) -- threaded through so downstream consumers (trace tooling,
    # telemetry) can tell WHICH formula scored a chunk even when the
    # production multi-rung RETAIN model merges chunks from several
    # fillers into one shape.
    filler_strategy: str | None = None
    slot_id: str = ""
    slot_semantics: str = ""


@dataclass
class SlotVerdict:
    """Observer's verdict on one slot, plus the two attribution fields Eval
    flagged as load-bearing (2026-07-24) and NOT collapsible into the bare
    verdict string: `ride_along` (a slot that succeeded because a sibling
    slot spent the turn, not on its own evidence -- selection bias Eval must
    segment out of calibration) and the verdict string itself, which must
    stay exactly what the caller passed (e.g. EXHAUSTED_BUDGET vs
    EXHAUSTED_ATTEMPTS are opposite calibration events -- clock cutoff vs a
    strategy's own limit -- and must never normalize to one "didn't finish"
    state). Optional at every field: Observer/Router's aggregation aren't
    wired into orchestrator.py yet, so callers may have nothing to report.
    """

    verdict: str = ""  # Router's VERDICT_* constant (see observer.py / continuation.py) -- passed through verbatim, never normalized
    reason: str = ""
    ride_along: bool = False


@dataclass
class CompiledSlot:
    """One slot's final compiled citations + Observer's verdict on it."""

    slot_id: str
    slot_semantics: str
    capacity: int
    required: bool
    citations: list[CompiledCitation] = field(default_factory=list)
    occupancy: int = 0
    under_filled: bool = False
    verdict: str = ""  # Router's VERDICT_* constant (see observer.py), "" if Observer wasn't run for this slot
    verdict_reason: str = ""
    ride_along: bool = False
    # Per-LLM-call attribution (model_used/llm_call_id) for whichever rung
    # filled this slot -- Eval's non-negotiable ask, 2026-07-24: "don't drop
    # attribution" (the same lesson that already bit the fillers once on
    # model_id). Empty when the filling rung made no LLM call (a/b/s) or
    # when the caller hasn't threaded it through yet.
    model_trace: dict = field(default_factory=dict)


@dataclass
class SynthesisTelemetry:
    """Diagnostics-only emit -- counts only, no raw text, no narrative
    layer (same posture as `pool`'s emit key in
    retriever-emit-telemetry-registry.md)."""

    chunks_in: int = 0
    chunks_out: int = 0
    duplicates_removed: int = 0
    neighbors_added: int = 0
    neighbors_skipped_no_anchor: int = 0
    unverified_citations: int = 0
    planned_status_citations: int = 0
    document_name_resolved: int = 0
    document_name_fallback: int = 0
    document_name_lookup_missed: int = 0  # subset of document_name_fallback where an internal chunk's document_id had no documents-table row -- an ops/data-integrity signal, not a synthesis bug (Chat's ask, 2026-07-24)
    # Real bug found 2026-07-24 (Ananth, relaying a Chat-side incident: a
    # single turn sent a ~473K-character prompt to the LLM, exhausting the
    # per-minute token quota): neighbor completion has zero token-budget
    # awareness -- it expands every eligible chunk by up to 2 paragraphs +
    # 1 page in each direction regardless of the query's real token_budget,
    # so a bounded initial retrieval can balloon past budget before ever
    # reaching Contract/Chat. `citations_trimmed_for_budget` counts how many
    # (lowest-priority: neighbors first, then lowest original_score)
    # citations compile_synthesis dropped to stay within the caller-supplied
    # token_budget -- 0 when no budget was supplied (legacy callers) or
    # nothing needed trimming.
    citations_trimmed_for_budget: int = 0
    # Fusion-level drops (blend model, retention landing 2026-07-24) --
    # DISTINCT from citations_trimmed_for_budget above: these happen
    # per-slot, inside RRF+MMR, BEFORE neighbor completion/citation
    # building ever run, using each slot's own token sub-budget (derived
    # from Router's per_slot_payload_tokens as split weights, or
    # effectively unbounded if that's not supplied -- see
    # compile_synthesis's docstring). citations_trimmed_for_budget is the
    # LATER, separate global safety-net trim that catches any overshoot
    # neighbor completion introduces afterward (it has no budget awareness
    # of its own). Both drop paths are real and can both fire on the same
    # query; keeping them as two counters, not one, is what lets the
    # reconciliation guard's identity stay honest about where a chunk
    # actually went.
    fusion_dropped_redundant: int = 0  # rejected by MMR as truly redundant (see fusion.py's MmrSelection.merged_away)
    fusion_dropped_budget: int = 0  # never evaluated by MMR -- that slot's own sub-budget ran out first (see MmrSelection.budget_cutoff_remaining)
    # rrf_fuse's OWN content-identity merge (chunk_identity.py's content_keys()
    # -- body-text-prefix/content_sha), distinct from both counters above: this
    # happens even WITHIN a single strategy group, folding chunks from two
    # separate document copies that repeat the same paragraph into one
    # canonical FusedChunk, before mmr_select or _dedup_cross_slot ever run.
    # Correct fusion behavior (don't show the LLM the same fact twice), but a
    # real drop the reconciliation guard's identity must subtract or it
    # false-positives on any corpus with duplicate-content document copies
    # (found live, 2026-07-29, Sunshine Health timely-filing query).
    fusion_content_merged: int = 0
    # Data-collection posture (Ananth, 2026-07-24): single-strategy-per-slot
    # calibration bypasses MMR's real drop entirely (see synthesis.py's
    # per-slot fusion step) so a forced strategy's true top-X reaches Chat/
    # Eval uncontaminated -- but MMR still runs once as a PROBE (real
    # slot_budget, never applied) purely to measure near-duplicate density.
    # This counts candidates the probe WOULD have rejected as redundant,
    # had real fusion been active -- log-only, never subtracted from
    # chunks_out/the reconciliation identity, since nothing was actually
    # dropped. A real data point for calibrating lambda/redundancy_threshold
    # once fusion reactivates (Retriever/Eval's ask), at near-zero cost.
    fusion_redundant_detected_not_dropped: int = 0
    # NOTE, 2026-07-24: a technical failure of the name-lookup query or
    # neighbor-expansion call (not "legitimately nothing to resolve") is
    # NOT tracked here as a soft counter. Retriever's ruling on the 3-layer
    # retry design: after synthesis.py's own local retry
    # (_MAX_TECHNICAL_RETRIES) is exhausted, the failure RE-RAISES instead
    # of degrading to an empty result -- so compile_synthesis itself raises
    # on that path and no SynthesisTelemetry is ever constructed to report
    # it in. The whole-loop retry (orchestrator-level) is the actual
    # recovery mechanism; a boolean/counter field here would always read
    # False/0 in any successfully-returned result and was removed as dead
    # telemetry.
    per_slot_verdict: dict = field(default_factory=dict)
    per_slot_ride_along: dict = field(default_factory=dict)
    compile_ms: int = 0  # total wall time across every segment below (DB's ask, 2026-07-24: confirmed this is the TOTAL, not name-lookup specifically)
    # Segment breakdown, same convention as `pool`'s `segment_ms` (see
    # retriever-emit-telemetry-registry.md) -- DB's ask, 2026-07-24: which
    # part of compile_ms is DB latency vs. pure-Python work. Synthesis makes
    # no LLM calls (it doesn't author text) and does no citation
    # verification or grounding check itself (filler c and Chat's grounding
    # badge own those respectively) -- so those aren't segments here.
    # {rerank_ms, neighbor_completion_ms, dedup_ms, name_resolution_ms,
    # citation_build_ms} -- the two DB-touching calls (neighbor_completion,
    # name_resolution) are where real latency variance lives; the rest is
    # pure-Python and should stay near-zero.
    segment_ms: dict = field(default_factory=dict)


# CoverageDiagnostic -- Router's decide_continuation() consumes this to
# rule expansion in or out (blend-model-design.md's "Coverage-gap
# diagnostic shape" section, RESOLVED between Router and Synthesizer,
# 2026-07-24, corrected 2026-07-24 after the originally-agreed
# saturated_strategies definition proved unreachable by construction --
# see fusion.py's derive_coverage_diagnostic docstring for the proof).
# Defined here, not in fusion.py, so synthesis.py can import fusion.py's
# derive_coverage_diagnostic() without a circular import (fusion.py needs
# chunk_identity.content_keys, which synthesis.py also uses; fusion.py
# must not import anything FROM synthesis.py in the other direction once
# synthesis.py starts calling INTO fusion.py's rrf_fuse()/mmr_select()).
POOL_VERDICT_GAPS_REMAIN = "gaps_remain"
POOL_VERDICT_SATURATED = "saturated"
POOL_VERDICT_BUDGET_FULL = "budget_full"


@dataclass
class CoverageDiagnostic:
    """Router's exact v1 shape. `reason` flows to telemetry VERBATIM
    (Eval's no-collapse rule -- same posture as SlotVerdict's
    verdict/reason above, never paraphrased or normalized downstream).
    `uncovered_aspect_count` is always None in v1 -- no per-fact/aspect
    granularity below chunk-level redundancy exists yet (a v2 concern).
    `per_strategy_counts` (Router's follow-up ask, 2026-07-24): structured
    per-strategy (n_selected, n_merged_away), not folded into `reason`
    prose, so Eval can measure the over-exclusion rate (1-of-6 rejected
    reads very differently from 5-of-6) and calibrate a v2 threshold."""

    slot_id: str
    pool_verdict: str  # one of the POOL_VERDICT_* constants above
    reason: str
    saturated_strategies: list[str] = field(default_factory=list)
    uncovered_aspect_count: int | None = None
    per_strategy_counts: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class SynthesisResult:
    """Output of the compile phase (Step 5)."""

    query: str
    slots: list[CompiledSlot] = field(default_factory=list)
    citations: list[CompiledCitation] = field(default_factory=list)  # flat, globally-indexed, cross-slot-deduped -- what Chat renders as SourceRef list
    telemetry: SynthesisTelemetry = field(default_factory=SynthesisTelemetry)
    # Router's coverage-gap diagnostic per slot (see CoverageDiagnostic
    # above) -- Synthesis EMITS this while compiling; the orchestrator
    # carries it into the NEXT continuation decision. Synthesis never
    # triggers filling itself -- signals in, decision at the loop (S5's
    # control-flow boundary, reaffirmed by both Router and Eval
    # independently, unchanged by this field's addition).
    coverage_diagnostics: dict[str, CoverageDiagnostic] = field(default_factory=dict)
