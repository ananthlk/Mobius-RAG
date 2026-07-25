"""Synthesis -- Step 5 of the answer engine (Shape -> Pool -> Router ->
Fillers -> Observer -> **Synthesis** -> Contract -> Timing), owned under
Retriever.

SCOPE CORRECTION (Ananth, direct, 2026-07-24): this module does NOT author
answer text -- Chat does, from what this module compiles. Synthesis's job is
purely: rerank the chunks Fillers assembled, complete neighbor context,
remove cross-slot duplicates, resolve real citations, and compile the
telemetry Chat/Eval/the bandit need. This supersedes this module's original
kickoff framing ("produce the final answer -- text + citations").

Real gaps found and worked around here, not upstream (flagged to Retriever
2026-07-24, not silently patched):
  - Neither PoolCandidate (pool/contracts.py) nor FilledChunk (fillers/
    contracts.py) carries a document display name anywhere in the pipeline
    -- only document_id/url. Chat's SourceRef (mobius-chat/app/skills/
    registry.py:60) requires document_name, so it's resolved here via a
    batched lookup against rag_published_embeddings (same columns
    corpus_search.py's neighbor-fetch already reads: document_display_name,
    document_filename), with an id/url-derived fallback label if the
    lookup misses.
  - External (url-based) chunks never carried a title either -- filler_d's
    _SearchHit/_Passage have one, but it's dropped before FilledChunk is
    built (filler_d.py's _chunk_from_passage). Fallback here humanizes the
    URL's last path segment (Chat's ask, 2026-07-24: a readable slug like
    "Prior Authorization Requirements", not a raw domain or UUID) --
    falling back further to the domain only when the path is empty.

Cross-slot dedup mirrors pool/dedup.py's two-tier pattern (chunk_id, then
content_sha falling back to a body-text prefix) -- that module only dedupes
within one pool build; a chunk can still land in two different slots, or
reappear across filler retries, by the time Fillers are done.

Neighbor completion reuses corpus_search._expand_with_neighbors (the same
helper Pool's own neighbor step calls) rather than reimplementing sibling-
fetch SQL -- "heavy reuse, not reinvention" (pool/public_adapter.py's own
convention). Only internal (document_id-bearing) chunks that don't already
have a neighbor sibling in their slot are expanded; external/fact-store
chunks have no page/paragraph anchor to expand from and are skipped (counted
in telemetry, not silently dropped).

Attribution fields Eval flagged as non-negotiable (2026-07-24, "don't drop
attribution" -- the same lesson that already bit the fillers once on
model_id): per-citation `verified` never collapses to a slot/answer-level
aggregate; per-slot `ride_along` and the verdict/reason strings (e.g.
EXHAUSTED_BUDGET vs EXHAUSTED_ATTEMPTS) pass through byte-for-byte, never
normalized; per-slot `model_trace` (stage/model_used/llm_call_id) is
threaded through when the caller supplies it (see compile_synthesis's
filler_emit_by_slot param -- not wired from orchestrator.py yet, same
pre-wiring posture as verdicts). `stage` added 2026-07-24 per Eval's
ruling: preserve-don't-drop existing attribution, not speculative
machinery -- deliberately excludes latency (Timing's attempt_spans owns
that, joined by call_id/attempt identity rather than duplicated here).

NOT handled yet, by design (Eval's forward note, 2026-07-24): retention of
superseded filler rungs' outputs (Router's observer-module-spec.md S8,
gated on this module's kickoff) isn't implemented upstream -- orchestrator.py's
_run_fillers_simple still DISCARDs a slot's prior rung on each retry, so
this module never sees a retained-but-unused chunk today. When retention
ships, whatever marks "compiled into the answer" vs "retained but not used"
must stay distinguishable -- retained-but-unused chunks must NOT count as
recall in Eval's grading. Not designed here; just don't collapse the two
into one bucket when that day comes.
"""

from __future__ import annotations

import logging
import re
import time
from urllib.parse import urlparse

from sqlalchemy import text as _sql
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.corpus_search import _expand_with_neighbors
from app.services.retriever.chunk_identity import content_keys
from app.services.retriever.fillers.contracts import (
    ASSIGNMENT_REASON_LLM_PARTIAL_MATCH,
    ASSIGNMENT_REASON_LLM_RETRIEVED,
    ASSIGNMENT_REASON_LLM_RETRIEVED_EXTERNAL,
    FilledChunk,
    FilledShape,
    FilledSlot,
)
from app.services.retriever.fusion import (
    MmrSelection,
    derive_coverage_diagnostic,
    mmr_select,
    rrf_fuse,
)
from app.services.retriever.synthesis_contracts import (
    CompiledCitation,
    CompiledSlot,
    CoverageDiagnostic,
    SlotVerdict,
    SynthesisResult,
    SynthesisTelemetry,
)

logger = logging.getLogger(__name__)

# Real bug found by Retriever, 2026-07-24, confirmed live (crashed every
# fact-store-sourced answer, not a corner case): FilledChunk's documented
# vocabulary is "internal" (document_id set, url None) vs "external" (url
# set, document_id None), but filler_s's fact_store chunks are a genuine
# THIRD category the binary doesn't account for -- chunk_id is ALWAYS a
# synthetic `fact_<hash>` string (filler_s.py's _stable_fact_id, never a
# UUID), and document_id is the fact store's own source doc_id OR that same
# synthetic string as a fallback -- neither is a real rag_published_embeddings
# row. Treating "document_id truthy, url falsy" as sufficient for "real
# internal document" sent these synthetic ids straight into a
# `CAST(:ids AS uuid[])` query, raising asyncpg.InvalidTextRepresentationError.
# Excluded from name-lookup eligibility (_resolve_document_names) and
# neighbor-completion eligibility (_complete_neighbors) below -- same
# treatment as external chunks (no real corpus row to expand from or look a
# name up against), NOT folded into either bucket.
_NON_CORPUS_SOURCE_TYPES = {"fact_store"}

# Source types that count as "authoritative" for Chat's grounding badge
# (_infer_authority). fact_store hits are a certified, pre-verified fact
# store (Payor Platform), not a lower-confidence web result -- defaulting
# them to "external" (the same bug class as above: document_id/url shape
# alone doesn't distinguish them) would understate their real authority.
# Deliberately an explicit allowlist, not "default authoritative unless
# external" -- fail-closed for any future source_type this module hasn't
# seen yet, consistent with `verified`'s and `document_status`'s posture.
_AUTHORITATIVE_SOURCE_TYPES = {"internal", "fact_store"}

# assignment_reason values that mean "this chunk came from filler c's LLM
# retrieval" -- the ONLY chunks where `quote_verified` carries meaning.
# Every other filler (a/b/s, and this module's own neighbor-completion
# chunks) leaves `quote_verified` at its FilledChunk default (None) simply
# because they never touch that field, not because "no quote was given" --
# checking `quote_verified is True` unconditionally across ALL chunks would
# misread every non-LLM chunk as unverified, which is wrong: a BM25/vector/
# fact-store match is a direct corpus/service hit, not an LLM citation with
# hallucination risk, and was never subject to quote verification in the
# first place. Imported from fillers/contracts.py, not hand-copied (Tech
# Review, 2026-07-24 -- observer.py used to mirror this same string
# independently; now both import the one source of truth).
_LLM_CITATION_REASONS = {
    ASSIGNMENT_REASON_LLM_RETRIEVED,
    ASSIGNMENT_REASON_LLM_RETRIEVED_EXTERNAL,
    ASSIGNMENT_REASON_LLM_PARTIAL_MATCH,
}

_NEIGHBOR_PARAGRAPH_WINDOW = 2
_NEIGHBOR_PAGE_WINDOW = 1

# Strategy identity for RRF fusion -- retention unions raw FilledChunks
# without adding a new strategy-id field (confirmed directly against
# orchestrator.py's real retention code, 2026-07-24: `st["retained_chunks"]
# .extend(filled_slot.chunks)` -- no tagging). So strategy is derived from
# assignment_reason, the only reliable existing signal, verified against
# each filler's actual real code (not guessed): filler_a.py's "score_rank",
# filler_b.py's "vector_rerank", filler_c.py's three llm_retrieved*
# variants, filler_d.py's "external_fetch", filler_s.py's "fact_store_hit".
# Falls back to "unknown" for anything unrecognized (e.g. a future filler
# this mapping hasn't been updated for) rather than crashing or silently
# dropping the chunk from fusion -- fail-visible, not fail-silent.
_STRATEGY_BY_ASSIGNMENT_REASON = {
    "score_rank": "a",
    "vector_rerank": "b",
    ASSIGNMENT_REASON_LLM_RETRIEVED: "c",
    ASSIGNMENT_REASON_LLM_RETRIEVED_EXTERNAL: "c",
    ASSIGNMENT_REASON_LLM_PARTIAL_MATCH: "c",
    "external_fetch": "d",
    "fact_store_hit": "s",
}
_UNKNOWN_STRATEGY = "unknown"


def _derive_strategy(chunk: FilledChunk) -> str:
    return _STRATEGY_BY_ASSIGNMENT_REASON.get(chunk.assignment_reason, _UNKNOWN_STRATEGY)


def _rerank_slot_chunks(chunks: list[FilledChunk]) -> list[FilledChunk]:
    """Final per-slot ordering: primary matches first (by score, best
    first), their neighbor context after. Slots are semantically distinct
    buckets (direct_answer/thematic_exploration/external_context per
    FilledSlot.slot_semantics) -- reranking across slots isn't meaningful
    here (Chat groups by slot; cross-strategy score comparability isn't
    established -- see pool/contracts.py's bm25_score docstring), so this
    only reorders within a slot.
    """
    return sorted(
        chunks,
        key=lambda c: (c.is_neighbor, -(c.original_score if c.original_score is not None else 0.0)),
    )


def _dedup_cross_slot(
    items: list[tuple[FilledSlot, FilledChunk]],
) -> tuple[list[tuple[FilledSlot, FilledChunk]], int]:
    """chunk_id first, then content (body-text and content_sha checked
    independently -- see chunk_identity.content_keys). First occurrence
    wins -- callers should pass items in the order they want preferred on
    a tie (here: slot order, then each slot's own reranked order).
    """
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    out: list[tuple[FilledSlot, FilledChunk]] = []
    removed = 0
    for slot, chunk in items:
        if chunk.chunk_id in seen_ids:
            removed += 1
            continue
        keys = content_keys(chunk)
        if keys and any(k in seen_content for k in keys):
            removed += 1
            continue
        seen_ids.add(chunk.chunk_id)
        seen_content.update(keys)
        out.append((slot, chunk))
    return out, removed


# Sentinel for "no per-slot budget available" (per_slot_payload_tokens not
# supplied) -- gives mmr_select a per-slot budget it will never actually
# hit, so its own redundancy-based selection still runs (dropping true
# duplicates) without artificially capping a slot before the EXISTING
# global _trim_to_token_budget pass gets a chance to enforce the caller's
# real overall budget. Not literally infinite (mmr_select's arithmetic
# would break on that) -- just far larger than any real query's token count.
_EFFECTIVELY_UNBOUNDED_TOKENS = 10**9

_MAX_TECHNICAL_RETRIES = 1  # Ananth's standing principle, 2026-07-24: force a retry on
# technical failures ("ask once and get the answer" -- try our best for
# first-pass resolution before degrading). Bounded to one retry, not
# infinite -- this is a DB call inside a single query's compile step, not
# the whole-pipeline retry loop (that's an orchestrator-level, cross-module
# concern, coordinated with Retriever separately, not reimplemented here).


async def _resolve_document_names(
    db: AsyncSession, chunk_ids: list[str]
) -> dict[str, str]:
    """Batched lookup of document_display_name/document_filename for a list
    of internal chunk ids, mirroring corpus_search.py's neighbor-fetch
    fallback chain (document_display_name -> document_filename -> generic
    label). A missing NAME for a chunk that resolved fine is not an error --
    that degrades to the caller's display fallback, tracked via
    `document_name_lookup_missed`, and must never break compilation. But a
    technical failure of the QUERY ITSELF (DB error) is different: it gets
    one local retry, and if that ALSO fails, this function RE-RAISES rather
    than degrading to an empty dict (Retriever's ruling, 2026-07-24, on the
    3-layer retry design: per-helper retry is the finest layer, but a
    failure that survives it must escalate to the orchestrator's
    whole-loop retry -- Synthesis silently returning a permanently-degraded
    result, query after query, is exactly the outcome that ruling forbids).
    """
    if not chunk_ids:
        return {}
    rows: list = []
    for attempt in range(_MAX_TECHNICAL_RETRIES + 1):
        try:
            result = await db.execute(
                _sql(
                    "SELECT id::text AS id, document_display_name, document_filename "
                    "FROM rag_published_embeddings "
                    "WHERE id = ANY(CAST(:ids AS uuid[]))"
                ),
                {"ids": chunk_ids},
            )
            rows = result.mappings().all()
            break
        except Exception as exc:
            if attempt < _MAX_TECHNICAL_RETRIES:
                logger.warning("synthesis._resolve_document_names failed, retrying once: %s", exc)
                continue
            logger.warning(
                "synthesis._resolve_document_names failed after retry, "
                "re-raising for whole-loop retry: %s", exc,
            )
            raise
    out: dict[str, str] = {}
    for row in rows:
        name = row.get("document_display_name") or row.get("document_filename")
        if name:
            out[row["id"]] = name
    return out


_URL_EXTENSION_RE = re.compile(r"\.(html?|pdf|aspx?)$", re.IGNORECASE)


def _humanize_url_path(url: str) -> str | None:
    """Last non-empty path segment, humanized: "prior-authorization-
    requirements" -> "Prior Authorization Requirements" (Chat's rendering
    convention, 2026-07-24 -- same pattern Filler's sitemap-links helper
    uses). Readable and honest, no false precision from a raw slug."""
    segments = [seg for seg in urlparse(url).path.split("/") if seg]
    if not segments:
        return None
    last = _URL_EXTENSION_RE.sub("", segments[-1])
    words = [w for w in re.split(r"[-_]+", last) if w]
    if not words:
        return None
    return " ".join(w.capitalize() for w in words)


def _infer_authority(chunk: FilledChunk, source_type: str) -> str:
    """Chat's SourceRef.authority (registry.py:60) -- the grounding badge
    computes "grounded" when every source is authoritative, and nothing
    upstream carries this forward for chunks that pass through Synthesis
    (Chat's sign-off gap, 2026-07-24). Planned content is never
    authoritative regardless of source_type -- it isn't live policy yet,
    same reasoning as the reality-gating rule itself. Coarser than Pool's
    own per-candidate `authority_level` (a genuinely more precise signal
    that exists upstream but is dropped before FilledChunk -- flagged to
    Retriever as a separate, deeper gap, not fixed here).

    fact_store included in `_AUTHORITATIVE_SOURCE_TYPES` -- a certified,
    pre-verified Payor Platform fact, not a lower-confidence web result;
    the earlier `source_type == "internal"` check would have mislabeled it
    "external" (Retriever's finding, 2026-07-24, same root cause as the
    name-lookup/neighbor-completion crash below).
    """
    if chunk.document_status == "planned":
        return "planned"
    return "authoritative" if source_type in _AUTHORITATIVE_SOURCE_TYPES else "external"


def _fallback_document_name(chunk: FilledChunk) -> str:
    if chunk.url:
        humanized = _humanize_url_path(chunk.url)
        if humanized:
            return humanized
        netloc = urlparse(chunk.url).netloc
        return netloc or chunk.url
    if chunk.document_id:
        return f"document {chunk.document_id[:8]}"
    return "source"


async def _complete_neighbors(
    db: AsyncSession,
    slot_items: list[tuple[FilledSlot, FilledChunk]],
) -> tuple[list[tuple[FilledSlot, FilledChunk]], int, int]:
    """For each slot, expand internal non-neighbor chunks that don't already
    have a sibling neighbor present in that slot. Returns (all items with
    fresh neighbors appended, neighbors_added, neighbors_skipped_no_anchor).

    External (url-based) and fact-store chunks have no document_id/
    page_number anchor to expand from -- skipped, counted, not dropped
    silently. A technical failure (the DB call itself erroring, not "no
    siblings found") gets one retry per slot; if that ALSO fails, this
    RE-RAISES immediately rather than continuing to the next slot --
    Retriever's ruling, 2026-07-24: a DB-call failure that survives its own
    retry is treated as systemic (the same broken connection would fail
    every remaining slot's expansion too), so it escalates to the
    orchestrator's whole-loop retry rather than Synthesis quietly returning
    a partially-degraded result.
    """
    added = 0
    skipped = 0
    out: list[tuple[FilledSlot, FilledChunk]] = list(slot_items)

    by_slot: dict[str, list[FilledChunk]] = {}
    slot_by_id: dict[str, FilledSlot] = {}
    for slot, chunk in slot_items:
        by_slot.setdefault(slot.slot_id, []).append(chunk)
        slot_by_id[slot.slot_id] = slot

    for slot_id, chunks in by_slot.items():
        slot = slot_by_id[slot_id]
        existing_neighbor_docs = {c.document_id for c in chunks if c.is_neighbor and c.document_id}
        # Tech Review's finding, 2026-07-24: a fresh neighbor's own
        # document_status must inherit its seed's, not default to None --
        # a hardcoded None reads as "not planned" to _infer_authority,
        # which would label a neighbor of a PLANNED document
        # "authoritative" (the seed itself correctly reads "planned", but
        # its neighbor slipped through with a more permissive label).
        status_by_doc_id = {
            c.document_id: c.document_status for c in chunks
            if c.document_id and not c.is_neighbor
        }
        seeds = []
        for c in chunks:
            if c.is_neighbor:
                continue
            if not c.document_id or c.source_type in _NON_CORPUS_SOURCE_TYPES:
                skipped += 1
                continue
            if c.document_id in existing_neighbor_docs:
                # Slot already carries neighbor context for this document.
                continue
            seeds.append({
                "id": c.chunk_id,
                "document_id": c.document_id,
                "text": c.text,
                "page_number": c.page_number,
                "paragraph_index": c.paragraph_index,
                "content_sha": c.content_sha,
                "rerank_score": c.original_score or 0.0,
            })
        if not seeds:
            continue
        combined = None
        for attempt in range(_MAX_TECHNICAL_RETRIES + 1):
            try:
                combined, meta = await _expand_with_neighbors(
                    db, seeds,
                    paragraph_window=_NEIGHBOR_PARAGRAPH_WINDOW,
                    page_window=_NEIGHBOR_PAGE_WINDOW,
                )
                break
            except Exception as exc:
                if attempt < _MAX_TECHNICAL_RETRIES:
                    logger.warning(
                        "synthesis._complete_neighbors failed slot=%s, retrying once: %s",
                        slot_id, exc,
                    )
                    continue
                logger.warning(
                    "synthesis._complete_neighbors failed slot=%s after retry, "
                    "re-raising for whole-loop retry: %s", slot_id, exc,
                )
                raise
        seed_ids = {s["id"] for s in seeds}
        for row in combined:
            if str(row.get("id")) in seed_ids:
                continue  # the seed itself, unchanged
            new_chunk = FilledChunk(
                chunk_id=str(row.get("id")),
                document_id=row.get("document_id"),
                text=row.get("text") or "",
                url=None,
                source_type="internal",
                document_status=status_by_doc_id.get(row.get("document_id")),  # inherited from the seed that pulled this neighbor in, not hardcoded None (Tech Review's finding, 2026-07-24)
                content_sha=row.get("content_sha"),
                page_number=row.get("page_number"),
                paragraph_index=row.get("paragraph_index"),
                tags={},
                is_neighbor=True,
                original_score=row.get("rerank_score"),
                assignment_reason="synthesis_neighbor_completion",
            )
            out.append((slot, new_chunk))
            added += 1

    return out, added, skipped


def _estimate_tokens(text: str | None) -> int:
    """Same chars/4 approximation Router's payload gate and the corpus p95
    measurement already use (2026-07-24) -- not a real tokenizer, but
    consistent with the rest of the fleet's existing convention rather than
    inventing a second one."""
    return len(text or "") // 4


def _trim_to_token_budget(
    citations: list[CompiledCitation],
    compiled_slots: dict[str, CompiledSlot],
    token_budget: int,
) -> int:
    """Real bug fix (2026-07-24): neighbor completion has no token-budget
    awareness and can multiply a bounded initial retrieval well past the
    caller's real budget before it ever reaches Contract/Chat -- traced to
    a live incident (a ~473K-char prompt exhausting Vertex's per-minute
    token quota). Drops lowest-priority citations (neighbors first, since
    they're supplementary context, not primary matches; then lowest
    original_score) from the END until the total estimated token count
    fits. Mutates `citations` and each CompiledSlot's citation list in
    place; re-indexes the survivors so `index` stays contiguous (Chat
    cites by index). Returns the number of citations dropped.
    """
    total = sum(_estimate_tokens(c.text) for c in citations)
    if total <= token_budget:
        return 0

    # Trim order: neighbors first (is_neighbor=True), then lowest
    # original_score among remaining -- primary matches with a real score
    # are the last thing to go. Ascending sort, NOT reversed: neighbors
    # get key-first-element 0 (< non-neighbors' 1), and within a group
    # ascending original_score puts the weakest evidence first -- both
    # exactly the "drop this first" order the loop below walks in.
    trimmable_order = sorted(
        citations,
        key=lambda c: (0 if c.is_neighbor else 1, c.original_score if c.original_score is not None else 0.0),
    )
    to_drop: set[int] = set()
    for c in trimmable_order:
        if total <= token_budget:
            break
        total -= _estimate_tokens(c.text)
        to_drop.add(id(c))

    if not to_drop:
        return 0

    survivors = [c for c in citations if id(c) not in to_drop]
    citations[:] = survivors
    for i, c in enumerate(citations, start=1):
        c.index = i
    for slot in compiled_slots.values():
        slot.citations = [c for c in slot.citations if id(c) not in to_drop]

    return len(to_drop)


async def compile_synthesis(
    query: str,
    filled_shape: FilledShape,
    *,
    db: AsyncSession,
    verdicts: dict[str, SlotVerdict] | None = None,
    filler_emit_by_slot: dict[str, dict] | None = None,
    token_budget: int | None = None,
    per_slot_payload_tokens: dict[str, int] | None = None,
    data_collection_mode: bool = False,
) -> SynthesisResult:
    """Compile Fillers' FilledShape into the final reranked, deduped,
    neighbor-complete, cited package Chat authors from and Eval grades
    against.

    Args:
        data_collection_mode: True only when the caller (orchestrator,
            reflecting Router's forced-strategy calibration posture,
            2026-07-24) forced a single strategy for this whole query.
            Real cross-strategy MMR selection is skipped -- a forced
            strategy's true top-X must reach Chat/Eval uncontaminated by
            MMR's still-uncalibrated redundancy_threshold/budget-cutoff, or
            the recall-at-K curve this phase exists to produce is corrupted
            by our own guesses. Deliberately an EXPLICIT flag, not inferred
            from "this slot's chunks happen to all share one strategy" --
            that shape also occurs organically in normal blend-mode
            operation (e.g. filler_a alone satisfies a slot, no other
            filler ever runs) and must still get real per-slot MMR
            budget-splitting there (caught by
            test_per_slot_payload_tokens_splits_budget_across_slots
            regressing when this was first implemented as shape-inferred
            instead of caller-declared).
        verdicts: slot_id -> SlotVerdict from Observer's evaluate() +
            Router's ride_along stamping (observer.py, orchestrator.py's
            _run_fillers_simple). Optional -- Observer isn't wired into
            orchestrator.py yet (per its own STATUS note); slots default to
            an empty SlotVerdict when absent rather than fabricating one.
            Verdict/reason strings are passed through byte-for-byte --
            Eval's non-negotiable ask (2026-07-24): EXHAUSTED_BUDGET vs
            EXHAUSTED_ATTEMPTS are opposite calibration events (clock
            cutoff vs a strategy's own limit) and must never collapse to
            one normalized "didn't finish" state.
        filler_emit_by_slot: slot_id -> that rung's FilledShape.emit
            per-slot detail dict (e.g. filler_c's per_slot_details entries,
            which carry model_used/llm_call_id). Optional -- today's
            orchestrator discards each rung's `emit` once it extracts the
            FilledSlot (see orchestrator.py's _run_fillers_simple
            docstring), so nothing reaches here yet; this param exists so
            wiring that through later requires no change to this function's
            signature. Eval's ask (2026-07-24): "don't drop attribution,"
            same lesson that already bit the fillers once on model_id.
        token_budget: the caller's real token budget for compiled retrieval
            content (e.g. Structure's ResourcePosture.token_budget, itself
            now caller-supplied per Chat's context-window math -- see
            structure.py's token_budget_for_retrieval). Optional -- None
            (legacy callers) skips enforcement entirely, same posture as
            every other optional param here. When supplied, neighbor
            completion's output is trimmed (lowest-priority citations
            first) to fit, since neighbor expansion has no budget awareness
            of its own -- see _trim_to_token_budget's docstring for the
            real incident this closes.
        per_slot_payload_tokens: Router's `RoutingLadder.per_slot_payload_tokens`
            (2026-07-24, blend-model-design.md §4) -- the ex-ante retrieval
            spend the allocator decided per slot (portfolio allocator: exact
            Σ k_i·tokens_i; chain allocators: MAX-over-rungs worst case,
            a rougher proxy during the pre-cutover transition period).
            Reuse this as SPLIT WEIGHTS for the synthesis-input budget
            (`slot_share = per_slot_payload_tokens[slot] / total`), NOT as
            the synthesis-input budget itself -- Router's ruling: retrieval
            budget and synthesis-input budget are distinct quantities from
            distinct sources (the latter is Chat-sourced and generally
            smaller). Optional -- None skips per-slot weighting entirely
            (today's global trim, unchanged) until the fusion rewrite
            consumes this.
    """
    t0 = time.monotonic()
    verdicts = verdicts or {}
    filler_emit_by_slot = filler_emit_by_slot or {}

    chunks_in = sum(len(slot.chunks) for slot in filled_shape.slots)

    # 1. Per-slot fusion (blend model, retention landing 2026-07-24 --
    #    blend-model-design.md). Retention means a slot's chunks may now
    #    come from several rungs/strategies, not just one winning rung, so
    #    the old single-pass _rerank_slot_chunks (cross-strategy score
    #    sort) is unsound -- a/b/d's scores are on incomparable scales
    #    (Eval's ruling, S4). Group by strategy (derived via
    #    assignment_reason -- retention doesn't tag chunks with an
    #    explicit strategy field, confirmed against orchestrator.py's real
    #    code), rank each strategy's OWN group with the same is_neighbor/
    #    score ordering _rerank_slot_chunks already used, fuse via RRF
    #    (rank-position-based, comparability-free), then select to each
    #    slot's own token sub-budget via MMR (TF-IDF redundancy check,
    #    Eval's ruling). CoverageDiagnostic captured per slot for Router's
    #    decide_continuation -- Synthesis only EMITS this, never triggers
    #    filling itself (S5's control-flow boundary, unchanged).
    t_rerank = time.monotonic()
    total_payload_tokens = sum(per_slot_payload_tokens.values()) if per_slot_payload_tokens else 0
    slot_items: list[tuple[FilledSlot, FilledChunk]] = []
    coverage_diagnostics: dict[str, CoverageDiagnostic] = {}
    fusion_dropped_redundant = 0
    fusion_dropped_budget = 0
    fusion_redundant_detected_not_dropped = 0
    for slot in filled_shape.slots:
        # No special-casing for an empty slot -- rrf_fuse({}), mmr_select([]),
        # and derive_coverage_diagnostic all handle empty input correctly
        # (the latter reads as gaps_remain, not saturated -- the vacuous-
        # truth trap this function's tests guard against), so every slot
        # gets a real CoverageDiagnostic, including ones Fillers never
        # filled at all. Router needs one per slot to make its decision;
        # a missing entry would be ambiguous ("no data" vs "not applicable").
        grouped_by_strategy: dict[str, list[FilledChunk]] = {}
        for chunk in slot.chunks:
            grouped_by_strategy.setdefault(_derive_strategy(chunk), []).append(chunk)
        for strategy_chunks in grouped_by_strategy.values():
            strategy_chunks[:] = _rerank_slot_chunks(strategy_chunks)

        fused = rrf_fuse(grouped_by_strategy)

        if token_budget is not None and per_slot_payload_tokens and total_payload_tokens > 0:
            # Router's ruling (blend-model-design.md): per_slot_payload_tokens
            # is retrieval-budget spend, reused here as SPLIT WEIGHTS for the
            # (distinct, generally smaller, Chat-sourced) synthesis-input
            # budget -- not the sub-budget value itself.
            slot_share = per_slot_payload_tokens.get(slot.slot_id, 0) / total_payload_tokens
            slot_budget = max(1, int(token_budget * slot_share))
        else:
            # No per-slot weighting available (legacy caller, or Router
            # hasn't supplied it) -- give MMR effectively unbounded room
            # per slot and let the EXISTING global _trim_to_token_budget
            # pass (unchanged, runs later) enforce the caller's overall
            # budget instead. Matches per_slot_payload_tokens's documented
            # "None skips per-slot weighting entirely" posture.
            slot_budget = _EFFECTIVELY_UNBOUNDED_TOKENS

        if data_collection_mode:
            # Data-collection posture (Ananth, 2026-07-24): Router forces a
            # single strategy per slot during recall-at-K calibration -- no
            # blend to fuse, and MMR's redundancy-drop/budget-cutoff run
            # BLIND to strategy count, so applying them here would silently
            # prune a single strategy's true top-X and corrupt the exact
            # "X delivered == X surfaced" measurement this phase exists to
            # produce (Retriever's finding, confirmed against this code).
            # Real selection bypasses MMR entirely: take `fused` in full
            # (already RRF-passthrough-ordered for one strategy -- rrf_fuse
            # never reorders a single-strategy input), deferring to the
            # existing global _trim_to_token_budget safety net (unchanged,
            # runs later) for the caller's overall budget.
            #
            # Gated on the caller's explicit declaration, NOT on
            # len(grouped_by_strategy) -- a slot organically having one
            # strategy's chunks happens in normal blend-mode operation too
            # (e.g. filler_a alone satisfies a slot) and must still get
            # real per-slot MMR budget-splitting there. Log if the
            # single-strategy assumption this mode is built on doesn't
            # actually hold -- should never happen per Retriever's
            # confirmed design, but a silent wrong-branch bug here would
            # be worse than a log line.
            if len(grouped_by_strategy) > 1:
                logger.warning(
                    "synthesis: data_collection_mode=True but slot_id=%s has %d "
                    "strategies (%s) -- expected exactly 1 under Router's forced-"
                    "strategy posture; proceeding with the data-collection bypass "
                    "anyway since the caller explicitly declared this mode",
                    slot.slot_id, len(grouped_by_strategy), sorted(grouped_by_strategy),
                )
            #
            # MMR still runs once here as a PROBE ONLY (real slot_budget,
            # so budget_cutoff_remaining is a genuine signal) -- its output
            # never reaches `slot_items`. Two uses, both diagnostic:
            #   1. `fusion_redundant_detected_not_dropped` -- log-only
            #      near-duplicate density per Retriever/Eval's ask, a real
            #      data point for calibrating lambda/threshold once fusion
            #      reactivates, at near-zero cost since the machinery
            #      already exists.
            #   2. coverage_diagnostics still reports a genuine budget_full
            #      vs gaps_remain split (did this forced strategy's own
            #      content exceed its per-slot share -- useful fill-depth
            #      signal for Router/Eval's curves even single-strategy).
            #      `merged_away` is deliberately NOT threaded through here
            #      (passed as [] below) -- `saturated` must reflect only
            #      redundancy actually APPLIED to the real output, and none
            #      is, so it correctly never fires in this mode (honest
            #      degradation, not a bug -- Retriever's read, confirmed).
            probe = mmr_select(fused, token_budget=slot_budget, estimate_tokens=_estimate_tokens)
            fusion_redundant_detected_not_dropped += len(probe.merged_away)
            diagnostic_selection = MmrSelection(
                selected=fused,
                merged_away=[],
                pairs_merged=0,
                budget_cutoff_remaining=probe.budget_cutoff_remaining,
            )
            coverage_diagnostics[slot.slot_id] = derive_coverage_diagnostic(slot.slot_id, diagnostic_selection)
            for fused_chunk in fused:
                slot_items.append((slot, fused_chunk.chunk))
        else:
            selection = mmr_select(fused, token_budget=slot_budget, estimate_tokens=_estimate_tokens)
            coverage_diagnostics[slot.slot_id] = derive_coverage_diagnostic(slot.slot_id, selection)
            fusion_dropped_redundant += len(selection.merged_away)
            fusion_dropped_budget += len(selection.budget_cutoff_remaining)

            for fused_chunk in selection.selected:
                slot_items.append((slot, fused_chunk.chunk))
    rerank_ms = int((time.monotonic() - t_rerank) * 1000)

    # 2. Neighbor completion (before cross-slot dedup, so freshly-added
    #    neighbors get a chance to collide with -- and lose to -- an
    #    already-assigned primary chunk from another slot/rung). A
    #    technical failure that survives its own local retry RAISES here
    #    (Retriever's ruling, 2026-07-24) -- deliberately uncaught, so it
    #    propagates to the orchestrator's whole-loop retry rather than
    #    compile_synthesis returning a silently-degraded result.
    t_neighbor = time.monotonic()
    slot_items, neighbors_added, neighbors_skipped = await _complete_neighbors(db, slot_items)
    neighbor_completion_ms = int((time.monotonic() - t_neighbor) * 1000)

    # 3. Cross-slot dedup, slot order preserved (first slot's copy wins).
    t_dedup = time.monotonic()
    slot_items, duplicates_removed = _dedup_cross_slot(slot_items)
    dedup_ms = int((time.monotonic() - t_dedup) * 1000)

    # 4. Resolve document names for internal chunks. Same escalation
    #    posture as step 2 -- a technical failure surviving its own retry
    #    raises rather than degrading to an empty dict. Excludes
    #    _NON_CORPUS_SOURCE_TYPES (fact_store) -- their chunk_id/document_id
    #    are synthetic, not real rag_published_embeddings rows (Retriever's
    #    crash finding, 2026-07-24: this previously sent a non-UUID string
    #    into a `CAST(:ids AS uuid[])` query).
    t_names = time.monotonic()
    internal_ids = [
        c.chunk_id for _, c in slot_items
        if c.document_id and not c.url and c.source_type not in _NON_CORPUS_SOURCE_TYPES
    ]
    resolved_names = await _resolve_document_names(db, internal_ids)
    name_resolution_ms = int((time.monotonic() - t_names) * 1000)

    # 5. Pre-register EVERY slot from FilledShape, even ones with zero
    # chunks -- iterating only slot_items (chunk-level pairs) would silently
    # drop a fully-empty/exhausted slot from the compiled output entirely.
    # That breaks Chat's ability to caveat it: a vanished slot is
    # indistinguishable from one that never existed, whereas an empty
    # CompiledSlot carrying EXHAUSTED_ATTEMPTS/WOULD_BENEFIT is exactly the
    # signal Chat needs to say "we looked and came up short" instead of
    # silently omitting that part of the answer (Ananth's QA check, 2026-07-24).
    t_citations = time.monotonic()
    citations: list[CompiledCitation] = []
    compiled_slots: dict[str, CompiledSlot] = {}
    slot_order: list[str] = []
    for slot in filled_shape.slots:
        sv = verdicts.get(slot.slot_id) or SlotVerdict()
        emit = filler_emit_by_slot.get(slot.slot_id) or {}
        compiled_slots[slot.slot_id] = CompiledSlot(
            slot_id=slot.slot_id,
            slot_semantics=slot.slot_semantics,
            capacity=slot.capacity,
            required=slot.required,
            occupancy=slot.occupancy,
            under_filled=slot.under_filled,
            verdict=sv.verdict,
            verdict_reason=sv.reason,
            ride_along=sv.ride_along,
            # `stage` added 2026-07-24 (Eval's ruling): "preserve, don't
            # drop" existing attribution, not speculative machinery -- the
            # filler already knows stage (it's the value passed to
            # generate()), same class of field as model_used/llm_call_id,
            # kept preserved so a future "does model quality diverge by
            # stage" analysis (bandit-live-path's own accepted v1
            # limitation) stays possible without a re-plumb. Deliberately
            # does NOT include latency -- that's Timing's attempt_spans
            # (t_attempt_start_ms/t_attempt_end_ms), joined by
            # call_id/attempt identity rather than duplicated here (two
            # sources of truth for one number is the anti-pattern, not the
            # fix).
            model_trace={
                k: emit[k] for k in ("stage", "model_used", "llm_call_id")
                if emit.get(k) is not None
            },
        )
        slot_order.append(slot.slot_id)

    # 6. Build compiled citations with a stable global index, grouped by slot.
    name_resolved_count = 0
    name_fallback_count = 0
    name_lookup_missed_count = 0
    unverified_count = 0
    planned_count = 0

    for slot, chunk in slot_items:
        name = resolved_names.get(chunk.chunk_id)
        if name:
            name_resolved_count += 1
        else:
            # Only a GENUINE internal chunk's missing name is an ops/data-
            # integrity signal -- fact_store (and any future
            # _NON_CORPUS_SOURCE_TYPES) chunks were never eligible for a
            # documents-table row in the first place (Retriever's polish
            # catch, 2026-07-24: the query-exclusion fix above didn't also
            # exclude this logging/counting condition, so a fact_store
            # chunk was still firing a misleading "may be missing" alarm
            # every time -- structurally never-applicable, not a miss).
            if chunk.document_id and not chunk.url and chunk.source_type not in _NON_CORPUS_SOURCE_TYPES:
                # An internal chunk Fillers assigned should have a
                # documents-table row -- a miss here is an ops/data-
                # integrity signal for DB to chase, not something to
                # silently paper over with the display fallback below
                # (Chat's ask, 2026-07-24).
                logger.warning(
                    "synthesis: document_name lookup missed for document_id=%s "
                    "chunk_id=%s -- documents-table row may be missing",
                    chunk.document_id, chunk.chunk_id,
                )
                name_lookup_missed_count += 1
            name = _fallback_document_name(chunk)
            name_fallback_count += 1

        if chunk.assignment_reason in _LLM_CITATION_REASONS:
            # Eval's ruling, 2026-07-24: fail-closed for LLM citations --
            # only a quote that actually matched counts as verified. Both
            # "given but didn't match" (False) and "no quote given at all"
            # (None) read as unverified; the old assignment_reason-only
            # check couldn't tell those two apart (both collapsed to
            # "llm_retrieved" upstream) and silently overstated confidence
            # for the no-quote case.
            verified = chunk.quote_verified is True
        else:
            verified = True
        if not verified:
            unverified_count += 1
        if chunk.document_status == "planned":
            planned_count += 1

        source_type = chunk.source_type or ("internal" if chunk.document_id else "external")
        index = len(citations) + 1
        citation = CompiledCitation(
            index=index,
            chunk_id=chunk.chunk_id,
            document_name=name,
            text=chunk.text,
            source_type=source_type,
            document_id=chunk.document_id,
            url=chunk.url,
            page_number=chunk.page_number,
            paragraph_index=chunk.paragraph_index,
            content_sha=chunk.content_sha,
            document_status=chunk.document_status,
            authority=_infer_authority(chunk, source_type),
            verified=verified,
            is_neighbor=chunk.is_neighbor,
            original_score=chunk.original_score,
            slot_id=slot.slot_id,
            slot_semantics=slot.slot_semantics,
        )
        citations.append(citation)
        compiled_slots[slot.slot_id].citations.append(citation)

    citation_build_ms = int((time.monotonic() - t_citations) * 1000)
    compile_ms = int((time.monotonic() - t0) * 1000)

    # Reconciliation guard (Eval's ask, 2026-07-24): every input chunk's
    # fate must be accounted for by exactly one counter -- kept, deduped,
    # added-as-neighbor, or fusion-dropped. `neighbors_skipped_no_anchor` is
    # NOT a drop path (those chunks stay in the pipeline unexpanded, just
    # not eligible for expansion), so it's deliberately excluded from this
    # identity. `fusion_dropped_redundant`/`fusion_dropped_budget` added
    # 2026-07-24 (blend model, retention landing) -- fusion now legitimately
    # drops chunks BEFORE neighbor completion/dedup ever run, so the
    # identity must account for that or this guard would false-positive on
    # every query. This is a logged signal, not a raised exception --
    # telemetry must never break the user's request (established fleet
    # convention) -- but a violation here means a chunk was silently
    # dropped or double-counted somewhere in compilation, exactly the
    # failure mode that's invisible without it: "chunks_out looks low" vs.
    # "counter X is the leak."
    expected_chunks_out = (
        chunks_in - fusion_dropped_redundant - fusion_dropped_budget
        - duplicates_removed + neighbors_added
    )
    if len(citations) != expected_chunks_out:
        logger.error(
            "synthesis: chunk-count reconciliation failed -- chunks_in=%d "
            "fusion_dropped_redundant=%d fusion_dropped_budget=%d "
            "duplicates_removed=%d neighbors_added=%d expected_chunks_out=%d "
            "actual_chunks_out=%d -- a chunk was silently dropped or "
            "double-counted somewhere in compilation",
            chunks_in, fusion_dropped_redundant, fusion_dropped_budget,
            duplicates_removed, neighbors_added,
            expected_chunks_out, len(citations),
        )

    # Token-budget enforcement (2026-07-24, real incident: neighbor
    # completion's unbounded expansion sent a ~473K-char prompt downstream
    # and exhausted Vertex's per-minute quota). Runs AFTER the
    # reconciliation guard above -- that guard's identity is about
    # compilation accounting (nothing silently lost/duplicated before this
    # point), not about the caller's budget, so trimming afterward keeps
    # the two concerns separate.
    t_budget = time.monotonic()
    citations_trimmed = (
        _trim_to_token_budget(citations, compiled_slots, token_budget)
        if token_budget is not None else 0
    )
    budget_enforcement_ms = int((time.monotonic() - t_budget) * 1000)

    telemetry = SynthesisTelemetry(
        chunks_in=chunks_in,
        chunks_out=len(citations),
        duplicates_removed=duplicates_removed,
        neighbors_added=neighbors_added,
        neighbors_skipped_no_anchor=neighbors_skipped,
        unverified_citations=unverified_count,
        planned_status_citations=planned_count,
        document_name_resolved=name_resolved_count,
        document_name_fallback=name_fallback_count,
        document_name_lookup_missed=name_lookup_missed_count,
        citations_trimmed_for_budget=citations_trimmed,
        fusion_dropped_redundant=fusion_dropped_redundant,
        fusion_dropped_budget=fusion_dropped_budget,
        fusion_redundant_detected_not_dropped=fusion_redundant_detected_not_dropped,
        per_slot_verdict={sid: compiled_slots[sid].verdict for sid in slot_order},
        per_slot_ride_along={sid: compiled_slots[sid].ride_along for sid in slot_order},
        compile_ms=compile_ms,
        segment_ms={
            "rerank_ms": rerank_ms,
            "neighbor_completion_ms": neighbor_completion_ms,
            "dedup_ms": dedup_ms,
            "name_resolution_ms": name_resolution_ms,
            "citation_build_ms": citation_build_ms,
            "budget_enforcement_ms": budget_enforcement_ms,
        },
    )

    return SynthesisResult(
        query=query,
        slots=[compiled_slots[sid] for sid in slot_order],
        citations=citations,
        telemetry=telemetry,
        coverage_diagnostics=coverage_diagnostics,
    )
