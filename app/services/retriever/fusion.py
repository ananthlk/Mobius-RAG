"""Reciprocal Rank Fusion (RRF) + MMR -- Eval's v1 fusion ruling for
cross-strategy blending (docs/rag-agents/blend-model-design.md S4),
2026-07-24. WIRED into compile_synthesis's real pipeline as of retention
landing (2026-07-24) -- see synthesis.py's per-slot fusion step.

Why RRF, not raw score comparison (Eval's ruling, blend-model-design.md
S4): a/b/d's scores are on different, incomparable scales/distributions --
ranking by raw score favors whichever strategy emits numerically larger
numbers, not whichever is more relevant (the same root cause as the
deferred chosen_slot/best_score bug in contract.py). RRF fuses by RANK
POSITION within each strategy's own list, comparability-free by
construction.

    score(c) = sum, over every strategy s where c appears in s's ranked
    list, of 1 / (k + rank_s(c))

rank_s is 1-based. k=60 is the standard RRF constant (Cormack et al.
2009's own empirically-chosen default, robust across corpora/query sets --
not re-derived here for lack of real data, same "seed with the literature
default, don't invent a number" posture as blend-model-design.md's own
lambda placeholder).

Identity across strategies: same two-tier key as pool/dedup.py and
synthesis.py's _dedup_cross_slot (chunk_id, then content_sha/body-text --
from chunk_identity.py, a small shared module extracted 2026-07-24
specifically so this module and synthesis.py can never drift on what
counts as "the same chunk" without either importing the other's business
logic -- synthesis.py now calls INTO this module's rrf_fuse()/
mmr_select(), so the old "import _content_keys from synthesis.py"
direction would have created a circular import). An item appearing in
multiple strategies' lists under the same identity gets its per-strategy
reciprocal-rank contributions SUMMED into one fused score -- that's the
whole point of RRF (reward cross-system consensus).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from app.services.retriever.chunk_identity import content_keys
from app.services.retriever.fillers.contracts import FilledChunk
from app.services.retriever.synthesis_contracts import (
    POOL_VERDICT_BUDGET_FULL,
    POOL_VERDICT_GAPS_REMAIN,
    POOL_VERDICT_SATURATED,
    CoverageDiagnostic,
)

DEFAULT_RRF_K = 60


@dataclass
class FusedChunk:
    """One chunk's fused result -- the RRF score and which strategies'
    ranked lists contributed to it (consensus signal: a chunk two
    strategies both surfaced is a stronger candidate than one only a
    single strategy found, which is exactly what summing reciprocal ranks
    across contributing strategies rewards)."""

    chunk: FilledChunk
    rrf_score: float = 0.0
    contributing_strategies: list[str] = field(default_factory=list)


def rrf_fuse(
    ranked_by_strategy: dict[str, list[FilledChunk]],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[FusedChunk]:
    """Fuse N strategies' own rank-ordered chunk lists into one ranking.

    Args:
        ranked_by_strategy: strategy_id -> that strategy's OWN chunks,
            already in that strategy's own rank order (best first). Rank
            position within each list is what RRF uses, not any score
            field on FilledChunk -- callers must not pre-sort by a
            cross-strategy score, since scores aren't comparable across
            strategies (the exact problem RRF exists to sidestep).
        k: RRF's damping constant. Higher k flattens the influence of rank
            position (rank 1 vs rank 50 matters less); lower k sharpens it.
            Default is the literature-standard value, not tuned against
            this fleet's own data.

    Returns:
        FusedChunk list sorted by rrf_score descending. One entry per
        unique identity (chunk_id or content-key match) -- a chunk
        appearing in multiple strategies' lists is merged into a single
        entry with a summed score and multiple contributing_strategies,
        not emitted once per strategy.
    """
    seen_by_id: dict[str, FusedChunk] = {}
    content_key_owner: dict[str, str] = {}  # content key -> canonical chunk_id
    order: list[str] = []  # first-seen order of canonical chunk_ids

    for strategy_id, chunks in ranked_by_strategy.items():
        for rank, chunk in enumerate(chunks, start=1):
            contribution = 1.0 / (k + rank)

            canonical_id = chunk.chunk_id if chunk.chunk_id in seen_by_id else None
            if canonical_id is None:
                for key in content_keys(chunk):
                    owner = content_key_owner.get(key)
                    if owner is not None:
                        canonical_id = owner
                        break

            if canonical_id is None:
                seen_by_id[chunk.chunk_id] = FusedChunk(chunk=chunk)
                for key in content_keys(chunk):
                    content_key_owner.setdefault(key, chunk.chunk_id)
                order.append(chunk.chunk_id)
                canonical_id = chunk.chunk_id

            fused = seen_by_id[canonical_id]
            fused.rrf_score += contribution
            if strategy_id not in fused.contributing_strategies:
                fused.contributing_strategies.append(strategy_id)

    fused_list = [seen_by_id[cid] for cid in order]
    fused_list.sort(key=lambda f: -f.rrf_score)
    return fused_list


# ---------------------------------------------------------------------------
# MMR selection -- realizing RRF's ranking into an actual fill-to-budget
# selection, per Eval's ruling (blend-model-design.md S4, RESOLVED
# 2026-07-24). Iteratively pick argmax(lambda*relevance - (1-lambda)*
# max_similarity(already_selected)) until the token budget is spent.
# ---------------------------------------------------------------------------

# Eval's ruling: seed relevance-leaning (0.7), NOT the 0.5 midpoint -- a
# recall-first blend where over-diversifying can drop distinct facts should
# weight relevance over diversity until real blend-output data exists to
# calibrate against. Uncalibrated placeholder, not a principled constant.
DEFAULT_MMR_LAMBDA = 0.7

# Eval's ruling: the real risk here is RECALL, not precision -- two chunks
# that share topic anchors but state DIFFERENT facts (e.g. two "Sunshine
# Health prior authorization" chunks about different procedure codes) must
# not be merged just because they're topically similar. Conservative
# threshold, biased toward KEEPING: only treat a candidate as truly
# redundant (drop it, don't select it) on very high overlap. Uncalibrated
# placeholder alongside lambda, same seed-then-measure posture.
DEFAULT_REDUNDANCY_THRESHOLD = 0.85

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _tf_idf_vectors(texts: list[str]) -> list[dict[str, float]]:
    """Minimal, dependency-free TF-IDF over one slot's candidate set.

    Eval's ruling, 2026-07-24: TF-IDF, NOT raw Jaccard -- in this corpus,
    duplicate facts share high-information RARE tokens (procedure codes
    like 96130-96139, numbers like "180 days", payer names like "Sunshine
    Health"), which TF-IDF weights and Jaccard drowns in common words.
    Embeddings would only buy the pure-paraphrase tail (disjoint
    vocabulary) -- a minority case in payer docs, per Eval's read; not
    worth the new embedding-API dependency for v1 (see this module's infra
    finding on chunk-level embeddings not existing in the pipeline today).

    No external ML library: this only ever runs over one slot's candidate
    set (tens of chunks, not a corpus), so a from-scratch vectorizer is
    cheap and avoids a new dependency for what's a small, bounded
    computation -- consistent with "zero new infra" being the whole point
    of choosing TF-IDF over embeddings in the first place.
    """
    tokenized = [_tokenize(t) for t in texts]
    n_docs = len(tokenized)
    doc_freq: Counter = Counter()
    for tokens in tokenized:
        for term in set(tokens):
            doc_freq[term] += 1
    # Smoothed idf (add-one on both numerator and denominator) so a term
    # appearing in every document doesn't collapse to ln(1) = 0 and vanish.
    idf = {term: math.log((n_docs + 1) / (count + 1)) + 1.0 for term, count in doc_freq.items()}
    vectors: list[dict[str, float]] = []
    for tokens in tokenized:
        term_freq = Counter(tokens)
        length = len(tokens) or 1
        vectors.append({
            term: (count / length) * idf.get(term, 0.0)
            for term, count in term_freq.items()
        })
    return vectors


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common_terms = set(a) & set(b)
    if not common_terms:
        return 0.0
    numerator = sum(a[term] * b[term] for term in common_terms)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return numerator / (norm_a * norm_b)


@dataclass
class MmrSelection:
    """MMR's output. `merged_away` is Eval's required instrumentation
    (2026-07-24): keeping the (dropped, kept-because-of) pairs, not just a
    count, is what lets offline calibration check "did a merged-away chunk
    carry a must_fact the survivor didn't" -- a check only possible against
    Eval's own must-fact bank, so this module surfaces the pairs and lets
    that analysis happen downstream, it doesn't compute the check itself.

    `budget_cutoff_remaining` is a DIFFERENT drop path from `merged_away`,
    tracked separately (Router's ask, 2026-07-24, for the CoverageDiagnostic
    negotiation): a candidate rejected as redundant (merged_away) means the
    pool had nothing more to offer; a candidate never even evaluated
    because the token budget ran out first (budget_cutoff_remaining) means
    the pool might still have useful content, we just didn't get to look --
    collapsing the two would make "deepening this strategy won't help"
    (saturated) indistinguishable from "we simply ran out of room"
    (budget_full), which Eval's calibration exclusion rules need to keep
    apart the same way EXHAUSTED_BUDGET/EXHAUSTED_ATTEMPTS already do
    elsewhere in this fleet.
    """

    selected: list[FusedChunk]
    merged_away: list[tuple[FusedChunk, FusedChunk]]  # (dropped_as_redundant, kept_because_of)
    pairs_merged: int  # len(merged_away) -- a real telemetry counter, distinct from budget-exhaustion drops
    budget_cutoff_remaining: list[FusedChunk] = field(default_factory=list)  # never evaluated -- the loop stopped for budget, not content-exhaustion


def mmr_select(
    fused: list[FusedChunk],
    *,
    token_budget: int,
    estimate_tokens: Callable[[str], int],
    lambda_: float = DEFAULT_MMR_LAMBDA,
    redundancy_threshold: float = DEFAULT_REDUNDANCY_THRESHOLD,
) -> MmrSelection:
    """Select a fill-to-budget subset of `fused` (RRF's already-ranked
    output) by true incremental value, not just top-N by rank.

    Args:
        fused: rrf_fuse()'s output -- already sorted by RRF score
            descending. Relevance for the MMR objective is derived from
            this rank position (best candidate = 1.0, linearly decaying),
            not a second, differently-scaled score.
        token_budget: hard cap on the sum of estimate_tokens(text) across
            selected candidates.
        estimate_tokens: injected rather than imported, so this module
            doesn't reach into synthesis.py's private _estimate_tokens --
            callers (once wired) pass that same function, keeping one
            token-estimation convention fleet-wide without a second
            reimplementation here.
        lambda_: relevance-vs-diversity weight (see DEFAULT_MMR_LAMBDA).
        redundancy_threshold: similarity above which a candidate is
            dropped as truly redundant rather than selected (see
            DEFAULT_REDUNDANCY_THRESHOLD) -- conservative, biased toward
            keeping over merging.

    Returns:
        MmrSelection -- selected candidates (budget-respecting, redundancy-
        filtered) plus the merged-away pairs for downstream instrumentation.
    """
    if not fused:
        return MmrSelection(selected=[], merged_away=[], pairs_merged=0, budget_cutoff_remaining=[])

    vectors = _tf_idf_vectors([f.chunk.text for f in fused])
    n = len(fused)
    # Relevance proxy: fused is already RRF-rank-sorted, so position alone
    # (not rrf_score's raw magnitude, which isn't on a 0-1 scale) gives a
    # simple, bounded relevance signal for the MMR objective.
    relevance = [(n - i) / n for i in range(n)]

    selected_idx: list[int] = []
    merged_away: list[tuple[FusedChunk, FusedChunk]] = []
    remaining = set(range(n))
    total_tokens = 0

    while remaining:
        best_idx = None
        best_mmr_score = None
        for i in remaining:
            max_sim = max(
                (_cosine_similarity(vectors[i], vectors[j]) for j in selected_idx),
                default=0.0,
            )
            mmr_score = lambda_ * relevance[i] - (1 - lambda_) * max_sim
            if best_mmr_score is None or mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = i

        remaining.discard(best_idx)
        candidate = fused[best_idx]

        closest_selected_idx = None
        max_sim_to_selected = 0.0
        for j in selected_idx:
            sim = _cosine_similarity(vectors[best_idx], vectors[j])
            if sim > max_sim_to_selected:
                max_sim_to_selected = sim
                closest_selected_idx = j

        if max_sim_to_selected >= redundancy_threshold:
            # Truly redundant, not just similar -- drop, don't select.
            # Distinct from a budget-exhaustion stop below: this is a
            # deliberate quality decision, not running out of room.
            merged_away.append((candidate, fused[closest_selected_idx]))
            continue

        candidate_tokens = estimate_tokens(candidate.chunk.text)
        if selected_idx and total_tokens + candidate_tokens > token_budget:
            # Budget exhausted. Same simplification _trim_to_token_budget
            # already uses elsewhere in this fleet: stop rather than search
            # for a smaller later candidate that might still fit. Always
            # keep at least one selection even if it alone exceeds budget
            # (the `selected_idx and` guard) -- never return nothing.
            # This candidate and everything still unevaluated in `remaining`
            # were never judged redundant or non-redundant -- we simply ran
            # out of room, a distinct signal from merged_away (see
            # MmrSelection's docstring).
            budget_cutoff_remaining = [candidate] + [fused[i] for i in remaining]
            return MmrSelection(
                selected=[fused[i] for i in selected_idx],
                merged_away=merged_away,
                pairs_merged=len(merged_away),
                budget_cutoff_remaining=budget_cutoff_remaining,
            )

        selected_idx.append(best_idx)
        total_tokens += candidate_tokens

    return MmrSelection(
        selected=[fused[i] for i in selected_idx],
        merged_away=merged_away,
        pairs_merged=len(merged_away),
        budget_cutoff_remaining=[],
    )


# ---------------------------------------------------------------------------
# CoverageDiagnostic -- Router's decide_continuation() consumes this to rule
# expansion in or out (blend-model-design.md's "Coverage-gap diagnostic
# shape" section, RESOLVED between Router and Synthesizer, 2026-07-24).
# Synthesis EMITS this while compiling; Router's orchestrator loop carries
# it into the NEXT continuation decision. Synthesis never triggers filling
# itself -- signals in, decision at the loop (S5's control-flow boundary,
# reaffirmed by both Router and Eval independently and unchanged here).
# Dataclass + POOL_VERDICT_* constants live in synthesis_contracts.py, not
# here (imported above) -- see that module for why.
# ---------------------------------------------------------------------------


def derive_coverage_diagnostic(slot_id: str, selection: MmrSelection) -> CoverageDiagnostic:
    """Derive Router's CoverageDiagnostic from one slot's MMR selection.

    Priority order (Router's ruling, exact): `budget_full` first -- a
    capacity cut overrides content state regardless of what the redundancy
    picture looks like, since Eval's calibration must be able to exclude it
    as "cut, not signal" the same way EXHAUSTED_BUDGET already is elsewhere.
    Then `saturated`, then `gaps_remain`.

    CORRECTED DEFINITION (2026-07-24, after the originally-agreed one
    proved unreachable): `saturated_strategies` = strategies with AT LEAST
    ONE candidate rejected as redundant (appearing anywhere in
    `merged_away`'s dropped half) -- NOT "every one of that strategy's
    candidates was rejected." The original all-rejected definition is
    mathematically unreachable whenever the pool is non-empty: the
    `kept_because_of` chunk in any merge is, by construction, always drawn
    from `selected` (mmr_select's `closest_selected_idx` only ever points
    into `selected_idx`), so whichever strategy "wins" a redundancy
    comparison always lands in `contributing_to_selected` and can never be
    saturated under the old definition -- meaning pool-level `saturated`
    (old: ALL contributing strategies fully rejected) could only fire on
    an empty pool, which already routes to `gaps_remain` separately. Router
    would never see this verdict fire in practice under the old
    definition, defeating the point of adding it.

    Under the corrected definition, a strategy counts as saturated the
    moment it demonstrates ANY internal redundancy (even if its best
    candidate still won a spot) -- a real, achievable signal that
    generalizes to "this strategy already showed a sign of repeating
    itself; deepening it is a bet against that trend, not a bet with no
    prior evidence either way." Pool-level `saturated` = every strategy
    that contributed anything to this slot showed that sign.

    Correctness note: when `budget_cutoff_remaining` is empty, mmr_select's
    own loop invariant guarantees every input candidate resolved into
    either `selected` or `merged_away` -- so the union of their
    contributing strategies is always the true, complete contributing set
    in that case, not an under-count. When `budget_cutoff_remaining` is
    non-empty, `budget_full` already wins by priority before that set is
    ever consulted.
    """
    contributing_to_selected: set[str] = set()
    n_selected_by_strategy: Counter = Counter()
    for f in selection.selected:
        contributing_to_selected.update(f.contributing_strategies)
        for s in f.contributing_strategies:
            n_selected_by_strategy[s] += 1

    contributing_to_merged_away: set[str] = set()
    n_merged_away_by_strategy: Counter = Counter()
    for dropped, _kept_because_of in selection.merged_away:
        contributing_to_merged_away.update(dropped.contributing_strategies)
        for s in dropped.contributing_strategies:
            n_merged_away_by_strategy[s] += 1

    # Corrected: showed ANY redundancy, not ALL-rejected (see docstring).
    saturated_strategies = contributing_to_merged_away
    all_contributing = contributing_to_selected | contributing_to_merged_away

    # Router's ask, 2026-07-24: structured per-strategy (n_selected,
    # n_merged_away) so Eval can measure the over-exclusion rate (1-of-6
    # rejected reads very differently from 5-of-6) rather than treat the
    # binary saturated_strategies bar as ground truth. Populated from
    # whatever was actually resolved, even in the budget_full case below --
    # partial counts are still informative.
    per_strategy_counts = {
        s: {
            "n_selected": n_selected_by_strategy.get(s, 0),
            "n_merged_away": n_merged_away_by_strategy.get(s, 0),
        }
        for s in all_contributing
    }

    if selection.budget_cutoff_remaining:
        return CoverageDiagnostic(
            slot_id=slot_id,
            pool_verdict=POOL_VERDICT_BUDGET_FULL,
            reason=(
                f"token budget exhausted with {len(selection.budget_cutoff_remaining)} "
                "candidate(s) still unevaluated -- capacity cut, not a content signal"
            ),
            saturated_strategies=[],
            per_strategy_counts=per_strategy_counts,
        )

    if all_contributing and saturated_strategies >= all_contributing:
        return CoverageDiagnostic(
            slot_id=slot_id,
            pool_verdict=POOL_VERDICT_SATURATED,
            reason=(
                f"all {len(all_contributing)} contributing strategy(ies) showed redundancy -- "
                f"{selection.pairs_merged} candidate(s) rejected, no strategy free of overlap"
            ),
            saturated_strategies=sorted(saturated_strategies),
            per_strategy_counts=per_strategy_counts,
        )

    return CoverageDiagnostic(
        slot_id=slot_id,
        pool_verdict=POOL_VERDICT_GAPS_REMAIN,
        reason=(
            f"{len(selection.selected)} chunk(s) selected; "
            f"{sorted(all_contributing - saturated_strategies)} showed no redundancy yet"
        ),
        saturated_strategies=sorted(saturated_strategies),
        per_strategy_counts=per_strategy_counts,
    )
