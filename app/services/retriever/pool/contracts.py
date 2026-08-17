"""Dataclass contracts for the pool module (Step 2 of the answer engine).

PoolResult is the output of one pool build (pool.run_pool_for_query()) --
the unioned, deduped candidate set every downstream filler reads PURELY
(no DB handle, no re-embed, no re-scan -- gate (b), the whole reason Pool
exists). See docs/rag-agents/pool-schematic-spec.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ScopeContext:
    """Which federated source an adapter is allowed to draw from.

    "federate, never leak" (retriever-meet-old-plan.md P0) -- scope rides
    the adapter, baked in at construction; Pool's orchestration core never
    branches on it directly. public=global is the only scope built today
    (pool-schematic-spec.md S1/S3.0) -- org=tenant, instant_rag=user+PHI
    are placeholders for when those adapters land, not consumed yet.
    """

    kind: str  # "public" | "org" | "instant_rag" | "cache"
    tenant_id: str | None = None
    user_id: str | None = None


@dataclass
class PoolCandidate:
    """One chunk in the pool -- a match (from a strategy) or a neighbor.

    source_type/document_status are passthrough fields required by
    Product-Awareness's reality-gating (target-structure-spec.md S10) so
    "planned" vs "live" survives pool -> shape -> synthesis -> contract.
    page_number/paragraph_index are needed by neighbor assembly
    (_fetch_sibling_chunks_batch's windowing), not just display.
    """

    chunk_id: str
    document_id: str
    text: str
    is_neighbor: bool
    source_arm: str  # "tag_select" | "vector" | "inherited" | "" for neighbors
    score: float | None
    tags: dict = field(default_factory=dict)
    document_status: str | None = None
    source_type: str | None = None
    content_sha: str | None = None
    page_number: int | None = None
    paragraph_index: int | None = None
    # Additive ranking signal for Filler a (BM25), added 2026-07-23 -- Fillers
    # can't make DB calls (gate b), so Pool supplies this instead of Filler a
    # re-querying. ts_rank_cd(search_vec, plainto_tsquery(...), 32) against
    # every match candidate regardless of which arm surfaced it -- Filler a
    # ranks across the whole pool, not just one arm. None for neighbors
    # (Retriever's explicit call): they weren't retrieved by any term match at
    # all, so a BM25 score would measure something incidental to why they're
    # in the pool -- same "None means no signal" convention `score` already
    # established for neighbors.
    bm25_score: float | None = None
    # Authority level for reranking (Filler a reranking signal). Values:
    # "contract_source_of_truth" (1.0), "operational" (0.5), "fyi" (0.2), None (0.0)
    authority_level: str | None = None
    # Coarse document-TYPE classification (2026-08-15, joint design with
    # Curation/Lexicon -- see docs/rag-agents/j-doc-type-spec.md). Set ONCE
    # at the document level (e.g. "um", "clinical_policy", "formulary",
    # "provider_manual"...) and applied to every chunk in that document --
    # deliberately coarser than document_tags.d_tags (which is per-domain,
    # phrase-derived, and can miss a chunk that doesn't locally repeat the
    # trigger phrase; see the Daraprim live case this was built to fix:
    # utilization_management.prior_authorization existed at the DOCUMENT
    # level but not on the specific clinical-criterion chunk that needed
    # it). Confirmed with Lexicon (2026-08-15): column is
    # rag_published_embeddings.document_doc_type, and -- same as
    # authority_level -- the real "not yet classified" value at THIS
    # layer is EMPTY STRING '', not NULL (publish.py COALESCEs the
    # nullable documents.doc_type -> '' before it reaches the index;
    # verified against authority_level's own live data: 7,855 published
    # docs carry document_authority_level=''). `str | None = None` here is
    # still the right Python-level default (safe for direct construction/
    # tests); any real value read off the DB is '' for unclassified, and
    # any future scoring function should treat '' the same as falsy/None
    # (`if not doc_type: ...`), exactly matching
    # _compute_authority_score's existing handling. Additive/inert (never
    # read yet) until the column and public_adapter.py's SELECT are real.
    doc_type: str | None = None
    # Same-rule-number recency tiebreak (2026-08-17, Crawler's §31 request
    # off the AHCA pilot -- two live documents for the same rule, 8 years
    # apart, ranked 0.0003 apart in rerank_score: a measured coin-flip).
    # document_effective_date is varchar ISO-or-'' at this layer (same
    # convention as authority_level/doc_type -- publish.py's COALESCE
    # pattern). document_filename carries the identifier this tiebreak
    # groups candidates BY (e.g. "59G-4.130" appears in both documents'
    # filenames) -- see filler_a.py's _extract_rule_identifier. Both
    # None-safe: a document with no effective_date or an unextractable
    # filename simply never enters a tiebreak group, no different from
    # today's behavior.
    effective_date: str | None = None
    document_filename: str | None = None
    # REAL STRUCTURAL BUG found+fixed 2026-07-23 (Retriever's live-trace
    # report): dedup_candidates() is "first-arm-wins" on chunk_id collision
    # (union order tag_select -> vector -> inherited) -- if a chunk is found
    # by BOTH tag_select and vector_search, the surviving entry keeps
    # tag_select's provenance/score, and Filler b (which filters to
    # source_arm=="vector" before ranking anything) never sees it at all,
    # even when vector_search independently found the SAME chunk with a
    # strong similarity score. Confirmed live on a real query where the
    # correct answer chunk was in the pool tagged source_arm=tag_select while
    # a direct, standalone vector_search() call found the identical chunk at
    # rank 165/1000, similarity 0.823 -- a real, silent recall loss, not a
    # one-off. Fix: compute vector_similarity for EVERY final candidate
    # regardless of which arm's provenance survived dedup -- same pattern
    # bm25_score already established (computed for every match candidate,
    # not just one arm's own results). Unlike bm25_score (which is None for
    # neighbors -- no term-match reason to be in the pool), vector similarity
    # is a meaningful signal for ANY chunk with an embedding regardless of
    # why it's in the pool, so this is populated for neighbors too, not just
    # matches. `score` stays arm-overloaded (tag_select's raw coverage count,
    # vector's cosine similarity, inherited's None) -- DO NOT read `score` as
    # a similarity value for a non-vector-arm candidate; use this field
    # instead when a genuine, comparable similarity number is needed
    # regardless of provenance.
    vector_similarity: float | None = None


@dataclass
class PoolResult:
    """Everything Pool produced for one rewritten_query (Step 2 output)."""

    query: str = ""
    candidates: list[PoolCandidate] = field(default_factory=list)
    # doc_narrow_ms (S3.1 step 2 cascade), tag_select_ms (step 4 chunk query),
    # embed_ms, vector_ms, inherited_ms, dedup_ms, neighbor_ms -- every
    # DB/retrieval segment split per TECH's clarification 2026-07-23, gate (d).
    segment_ms: dict = field(default_factory=dict)
    strategy_hint: str = ""  # which arm(s) actually contributed
    fallback_triggered: bool = False
    pool_ms: int = 0
    # Added 2026-07-23 (Filler s / Payor Platform request, Retriever-relayed
    # and independently verified): Pool's vector arm already computes this
    # via gemini-embedding-001 @ output_dimensionality=1536 -- confirmed
    # live to match the Payor Fact Store's own vector(1536) schema exactly
    # (mobius-payor/migrations/006_payor_fact_store.sql,
    # mobius-payor/app/fact_embed.py). Reuse saves Filler s a redundant
    # embed call. None when the vector arm didn't run for this query (e.g.
    # no-retrieval postures) or the embed call itself failed -- Filler s
    # falls back to its own bounded call in that case, always correct
    # either way, this is purely an optimization.
    query_embedding: list[float] | None = None
    # Added 2026-07-23 (Ananth's finding, verified against legacy + Filler A's
    # actual meta_boost code): Gate's expansion_phrases (lexicon-derived
    # terms) never carried selectivity weighting downstream -- generic terms
    # ("claims", "medicaid") diluted equally with highly-discriminating ones
    # ("timely filing", "sunshine health"). Reconstructed here from gate's
    # matched d/j/p codes + already-computed selectivity_for_tag() (no Gate
    # contract change needed -- the code<->phrase link exists at lexicon-
    # match time, just gets flattened away in GateResult). Query-level, not
    # per-candidate -- checking phrase presence in a candidate's text/tags is
    # cheap in-memory work that belongs in the consuming filler's own rerank
    # pass (confirmed against Filler A's actual filler_a.py:
    # _compute_meta_boost_score(text, tags, required_phrases, boosted_phrases),
    # which reads these as list[tuple[str, float]] via getattr(pool_result,
    # 'required_phrases', None) -- verified shape match, not just relayed).
    # DROP-bucket phrases (selectivity < 0.40) are excluded entirely, not
    # included at low weight.
    required_phrases: list[tuple[str, float]] = field(default_factory=list)
    boosted_phrases: list[tuple[str, float]] = field(default_factory=list)


class SourceAdapter(ABC):
    """Generic seam every federated source implements (pool-schematic-spec.md S3.0).

    Pool's orchestration core (union/dedup/neighbor-assembly) calls these
    four methods generically -- it never branches on which adapter it's
    talking to. public = adapter #1, the only one built today; org/
    instant-rag land later behind this exact interface, no touching
    Pool's core (Ananth's explicit plug-and-play requirement, 2026-07-23).

    ``inherited()`` is deliberately optional in EFFECT, not signature --
    AHCA inheritance is a public-corpus-specific concept; adapters with no
    equivalent return [] without violating the interface.
    """

    scope: ScopeContext

    @abstractmethod
    async def tag_select(
        self, query: str, expansion_phrases: list[str], d_codes: list[str], j_codes: list[str], p_codes: list[str], width: int
    ) -> tuple[list[PoolCandidate], dict]:
        """Takes RAW matched tag codes (unclassified) -- required/boosted/drop
        bucketing happens INSIDE the adapter, not the orchestrator, because
        tag selectivity is corpus-specific (computed against each adapter's
        own backing tag table, e.g. public's document_tags). Same ownership
        line as "each adapter owns bm25+vector+inheritance+neighbors for
        its section" (retriever-meet-old-plan.md). ``query``/``expansion_phrases``
        are needed only for the additive `bm25_score` field (Filler a,
        2026-07-23), not for the tag-coverage selection logic itself.
        ``expansion_phrases`` (Gate's lexicon-derived phrases, e.g. "timely
        filing"/"filing limit") get OR-folded into the bm25 tsquery alongside
        the raw query tokens -- restores a legacy signal (corpus_search.py's
        `_build_or_tsquery(*raw_tokens, *expansion.expansion_phrases)`) that
        never survived into GateResult's consumers (real gap found
        2026-07-23: expansion_phrases was computed by Gate, stored, and read
        by NOTHING downstream until now).

        Returns (candidates, segment_ms) -- segment_ms carries doc_narrow_ms
        + tag_select_ms as two distinct buckets (TECH's 2026-07-23 split)."""
        ...

    @abstractmethod
    async def vector_search(
        self, query: str, expansion_phrases: list[str], j_codes: list[str], width: int
    ) -> tuple[list[PoolCandidate], dict, list[float] | None]:
        """Returns (candidates, segment_ms, query_embedding) -- segment_ms
        carries embed_ms + vector_ms. query_embedding is the raw embedding
        computed for this query (None if the embed call itself failed),
        surfaced so Pool can populate PoolResult.query_embedding for reuse
        by downstream Fillers with matching embedding-space needs.
        ``expansion_phrases`` needed only for the additive `bm25_score`
        field, same as `tag_select`/`inherited`. ``j_codes`` needed for
        cross-payer exclusion (2026-07-23 correctness fix): when the query
        names a specific payer, chunks tagged with a DIFFERENT payer are
        excluded from vector's candidate set -- unlike `tag_select`, this is
        a negative exclusion, not a positive scope filter, so vector search
        keeps its wide semantic-discovery character for non-payer-specific
        content."""
        ...

    @abstractmethod
    async def inherited(
        self, query: str, expansion_phrases: list[str], payor_codes: list[str], width: int
    ) -> tuple[list[PoolCandidate], dict]:
        """Returns (candidates, segment_ms) -- segment_ms carries inherited_ms.
        No-ops ([], {}) for adapters/queries with no inheritance concept.
        ``query``/``expansion_phrases`` are needed only for the additive
        `bm25_score` field, same as `tag_select`."""
        ...

    @abstractmethod
    async def neighbors(self, candidates: list[PoolCandidate]) -> tuple[list[PoolCandidate], dict]:
        """Returns (matches + neighbors, segment_ms) -- segment_ms carries neighbor_ms."""
        ...

    @abstractmethod
    async def phrase_buckets(
        self, d_codes: list[str], j_codes: list[str], p_codes: list[str], expansion_phrases: list[str]
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """Returns (required_phrases, boosted_phrases) -- each a list of
        (phrase, selectivity) pairs, for PoolResult's same-named fields
        (added 2026-07-23, Filler A's meta_boost consumer -- verified shape
        match against their actual filler_a.py, not just relayed). Adapter-
        owned because selectivity is corpus-specific (computed against each
        adapter's own backing tag table) -- same reasoning as `tag_select`'s
        REQUIRED/BOOSTED/DROP bucketing. DROP-bucket phrases are excluded
        from both lists entirely, not included at low weight."""
        ...

    @abstractmethod
    async def attach_vector_similarity(
        self, candidates: list[PoolCandidate], query_embedding: list[float] | None
    ) -> tuple[list[PoolCandidate], dict]:
        """Returns (candidates-with-vector_similarity-populated, segment_ms).

        Fixes a real structural bug (2026-07-23, Retriever's live-trace
        report): dedup is first-arm-wins on chunk_id collision, so a chunk
        found by BOTH tag_select and vector_search only keeps tag_select's
        provenance -- any filler that filters to source_arm=="vector" never
        sees it, even when vector_search independently found the identical
        chunk with a strong similarity score. Computes similarity for EVERY
        candidate (matches AND neighbors -- unlike bm25_score, similarity is
        meaningful regardless of why a chunk is in the pool) via one batch
        query against the deduped candidate ids, reusing the query embedding
        Pool already computed once. No-ops (candidates unchanged, {}) when
        query_embedding is None (embed failed or vector arm never ran)."""
        ...
