"""Corpus search service — BM25, semantic (pgvector), and hybrid (RRF + rerank).

Owned by mobius-rag.  Exposes one entry-point:

    results, telemetry = await corpus_search(db, request)

Three modes:
  "corpus"    — hybrid: BM25 ⊕ pgvector in parallel, RRF-fused, reranked
  "precision" — BM25-only, exact phrase / code lookup
  "recall"    — vector-only, broad semantic recall

All arms query ``rag_published_embeddings`` — the dbt contract table that
is written on user Publish.  Every arm returns fully-resolved chunks (text,
doc name, page number) in a single SQL scan, eliminating the N+1 lookup
that ``/api/query`` did.

Calibration
-----------
A simplified reranker maps raw scores → [0, 1] and assigns a confidence
label (high / medium / low / abstain).  Signals used:

  score (0.30 weight)          — BM25 ts_rank or pgvector cosine similarity
  authority_level (0.15)       — contract_source_of_truth > operational > fyi
  length (0.10)                — penalise stubs < 50 chars, reward up to 500
  jpd_tag_match (0.25)         — Phase 2B: query-intent × chunk category alignment

Total weight coverage: 80 % of the full reranker_v1.yaml.  Confidence thresholds
are calibrated to the 0.80 normaliser.

RRF fusion
----------
Per-arm rerank, then Reciprocal Rank Fusion (Cormack et al. 2009, k=60).
Each output chunk carries ``retrieval_arms`` (["bm25"] / ["vector"] /
["bm25","vector"]) for telemetry and UI attribution.

Observability
-------------
Every call emits structured INFO log lines keyed by ``search_id`` so the
full retrieval pipeline can be reconstructed from logs.  Stages logged:

  corpus_search.query_intake   — request parameters
  corpus_search.bm25_arm       — per-hit raw ts_rank_cd scores
  corpus_search.vector_arm     — per-hit cosine similarities
  corpus_search.rrf_fusion     — merged scores + arm contributions
  corpus_search.rerank         — per-chunk signal breakdown
  corpus_search.confidence     — label assigned + threshold applied
  corpus_search.result         — final ranked output summary

The ``telemetry.scoring_trace`` field in the response mirrors the rerank
stage for the returned chunks so callers can inspect the scoring inline.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import re
import time
import uuid
from typing import Any

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public request / response types (contract with callers incl. chat)
# ---------------------------------------------------------------------------

class CorpusFilters(BaseModel):
    payer: str | None = None
    state: str | None = None
    program: str | None = None
    authority_level: str | None = None


class CorpusSearchRequest(BaseModel):
    query: str
    k: int = 10
    mode: str = "corpus"           # corpus | precision | recall
    filters: CorpusFilters | None = None
    include_document_ids: list[str] | None = None
    min_similarity: float | None = None

    # tag_mode controls how the lexicon-extracted j/d/p tags filter the
    # candidate pool. Exposed to the agent / ReAct loop so it can pick
    # the right precision/recall tradeoff per query intent:
    #
    #   "auto"     — STRICT first (metadata-J: documents.payer/state/program
    #                 must match), fall through to RELAXED if 0 hits.
    #                 Default for chat/agent usage. Smart fallback.
    #   "strict"   — metadata-J only. Returns empty when no AHCA-payer
    #                 doc exists for an AHCA query. Use when grounding
    #                 a specific claim about a specific authority.
    #   "relaxed"  — skip metadata-J entirely; OR across d/p body tags.
    #                 Use for exploratory / cross-payer / "what does the
    #                 literature say" questions.
    #   "none"     — no tag filter; only the candidate cap (1000 rows)
    #                 bounds the BM25 work. Use for code lookups and
    #                 generic queries where tags would hurt recall.
    tag_mode: str = "auto"

    # ── Assembly controls ────────────────────────────────────────────────
    # assembly_strategy controls how k slots are filled from the ranked pool:
    #
    #   "score"          — pure rerank_score order (default)
    #   "canonical_first"— CoT docs bubble to top within each confidence band;
    #                      remaining slots filled by score
    #   "balanced"       — reserve ceil(k × canonical_floor) slots for
    #                      tier-0/1 docs; fill rest by score
    #
    # canonical_floor is only used by the "balanced" strategy.  It sets the
    # minimum fraction of returned chunks that must come from
    # contract_source_of_truth or payer_policy sources.  Default 0.5 means
    # at least half the slots go to authoritative docs when they exist.
    assembly_strategy: str = "score"    # score | canonical_first | balanced
    canonical_floor: float = 0.5        # fraction [0, 1]; used by "balanced"

    # ── Reranker tag-coverage signal (Fix #2 — 2026-05-02) ──────────────
    # When the agent has identified REQUIRED tag phrases for the query
    # (e.g. ``["behavioral health", "prior authorization", "Sunshine Health"]``),
    # the reranker boosts chunks whose body text contains all of them
    # and penalises chunks that cover only a subset. This kills the
    # failure mode where BM25 returns chunks containing 2 of 3 required
    # topics in unrelated contexts.
    required_phrases: list[str] | None = None

    # Per-phrase weights aligned with ``required_phrases`` by index.
    # Sourced from the lexicon's selectivity per tag — rarer / more
    # discriminating phrases (e.g. "sunshine health" at sel 0.93) get
    # more weight in tag_coverage than common phrases (e.g. "prior
    # authorization" at 0.79). When a phrase has weight w, its presence
    # in body / meta haystack contributes w / sum(weights) to
    # tag_coverage. None or mismatched length → all phrases equal
    # (legacy behavior).
    required_phrase_weights: list[float] | None = None

    # Optional per-phrase TAG CODE aligned with required_phrases by
    # index. When the phrase comes from a lexicon tag (j: / d: / p:)
    # the FULL CODE goes here (e.g. ``"j:payor.sunshine_health"``).
    # Literals and free-text phrases get None. The reranker uses this
    # to short-circuit substring matching: if the phrase is a j-tag
    # and the chunk's parent document is TAGGED with that j-code,
    # presence is a hard YES — no need to substring-search body or
    # filename. j-tags are binary (a doc IS or ISN'T about Sunshine
    # Health); d-tags are softer (a doc MAY cover this topic), so the
    # binary credit only applies to j-tags.
    required_phrase_tag_codes: list[str | None] | None = None

    # ── Neighborhood expansion (ported from mobius-chat doc_assembly) ──
    # Pull ±N paragraphs (within ±M pages) of EACH assembled chunk from
    # the same document. The classic failure mode this solves: BM25
    # surfaces a chunk containing the LEAD-IN to a fact ("see the table
    # below…") and the actual numbers live in the next chunk. Without
    # neighbors the LLM never sees the answer.
    #
    # Set ``neighbor_paragraph_window=0`` to disable.
    neighbor_paragraph_window: int = 2  # ±N paragraphs per assembled chunk
    neighbor_page_window: int = 1       # ±M pages constraint


class CorpusChunk(BaseModel):
    id: str
    text: str
    document_id: str
    document_name: str
    page_number: int | None
    paragraph_index: int | None
    source_type: str
    similarity: float              # normalised [0, 1] — arm score or RRF
    rerank_score: float            # post-rerank combined score [0, 1]
    confidence_label: str          # high | medium | low | abstain
    retrieval_arms: list[str]      # ["bm25"] / ["vector"] / ["bm25","vector"]
    authority_level: str | None
    payer: str | None
    state: str | None
    jpd_tags: list[str] = []       # dominant J/P/D families matched by this chunk
    # Where in the doc this chunk lives — used by the trace UI to show
    # "is the LLM seeing enough context?" diagnostics. All optional;
    # populated when the underlying ``rag_published_embeddings`` row has
    # them set (which is the standard case for chunked-and-published docs).
    section_path: str | None = None
    chapter_path: str | None = None
    summary: str | None = None
    # Neighborhood expansion: True when this chunk was added as a
    # supporting sibling of an assembled hit, NOT as a primary retrieval
    # match. Downstream consumers can render neighbors differently
    # (smaller font, "context" badge, etc.). Defaults False so existing
    # callers see no shape change.
    is_neighbor: bool = False


class CorpusSearchResponse(BaseModel):
    chunks: list[CorpusChunk]
    telemetry: dict[str, Any]


# ---------------------------------------------------------------------------
# Authority-level weight map  (matches reranker_v1.yaml authority_level signal)
# ---------------------------------------------------------------------------
_AUTHORITY_WEIGHTS: dict[str, float] = {
    "contract_source_of_truth": 1.0,
    "operational_suggested":    0.65,
    "payer_policy":             0.50,   # published policy docs — citable source
    "fyi_not_citable":          0.20,
}
# Documents with no authority_level tag get a small default so they still
# score above zero on this signal; fully untagged docs are penalised relative
# to tagged ones but not zeroed out entirely.
_AUTHORITY_DEFAULT = 0.10

# ---------------------------------------------------------------------------
# JPD tag-match signal  (Phase 2B — 0.25 weight)
# ---------------------------------------------------------------------------
# Each entry: { category_name: [keyword_patterns, ...] }
# Patterns are matched case-insensitively against query and chunk text.
# J = Justification / eligibility   P = Prior-auth / policy gate   D = Documentation
_JPD_PATTERNS: dict[str, list[str]] = {
    # ── P — Prior Authorization / Policy ───────────────────────────────
    "prior_authorization_required": [
        "prior authorization", "prior auth", "pre-authorization",
        "pre auth", "pa required", "requires authorization",
        "authorization required", "authorization criteria",
        "medical necessity", "utilization management", "um criteria",
        "medically necessary", "clinical criteria", "level of care",
        "admission criteria", "inpatient criteria", "coverage criteria",
        "criteria for", "clinical guidelines", "covered criteria",
    ],
    "claims_authorization_submissions": [
        "claims submission", "claim form", "billing code",
        "cpt code", "hcpcs", "procedure code", "revenue code",
        "submit claim", "claim adjudication", "authorization number",
        "pa number", "claims processing", "remittance",
    ],
    # ── J — Justification / Eligibility ────────────────────────────────
    "member_eligibility_molina": [
        "member eligibility", "eligibility verification", "enrollment",
        "eligible member", "covered services", "plan benefit",
        "benefit coverage", "covered under", "who is eligible",
        "who qualifies", "eligible for", "beneficiary",
    ],
    "benefit_access_limitations": [
        "limitation", "exclusion", "not covered", "non-covered",
        "benefit limit", "annual limit", "visit limit", "frequency limit",
        "coverage limit", "service limit", "maximum benefit",
        "out-of-network", "out of network",
    ],
    "coordination_of_benefits": [
        "coordination of benefits", "cob", "dual coverage",
        "other insurance", "third party liability", "tpl",
        "primary payer", "secondary payer",
    ],
    # ── D — Documentation / Compliance ─────────────────────────────────
    "compliant_claim_requirements": [
        "documentation required", "required documentation",
        "supporting documentation", "clinical documentation",
        "medical records", "clinical notes", "progress notes",
        "treatment plan", "discharge summary", "clinical record",
        "what documentation", "documentation needed", "records required",
        "supporting evidence", "clinical evidence", "chart notes",
    ],
    "credentialing": [
        "credentialing", "credential", "provider enrollment",
        "network enrollment", "network participation", "in-network",
        "provider qualification", "licensure", "certification",
        "provider manual", "participating provider",
    ],
    "claim_submission_important": [
        "timely filing", "filing deadline", "claim deadline",
        "corrected claim", "claim adjustment", "resubmission",
    ],
    "claim_disputes": [
        "appeal", "grievance", "dispute", "reconsideration",
        "denial", "denied claim", "adverse determination",
        "fair hearing", "redetermination",
    ],
    # ── Other ───────────────────────────────────────────────────────────
    "contacting_marketing_members": [
        "contact member", "member outreach", "member communication",
        "member notification",
    ],
    "other_important": [],
}

# Human-readable short labels for the 3 JPD families
_JPD_FAMILY: dict[str, str] = {
    "prior_authorization_required":     "P",
    "claims_authorization_submissions":  "P",
    "member_eligibility_molina":        "J",
    "benefit_access_limitations":       "J",
    "coordination_of_benefits":         "J",
    "compliant_claim_requirements":     "D",
    "credentialing":                    "D",
    "claim_submission_important":       "D",
    "claim_disputes":                   "D",
    "contacting_marketing_members":     "O",
    "other_important":                  "O",
}


def _classify_jpd(text: str) -> dict[str, float]:
    """Return category → match_score for a piece of text (query or chunk).

    For short texts (queries, ≤ 15 words): any single keyword hit fires the
    category, score = 1.0 / sqrt(n_patterns) so that narrow categories score
    higher than broad ones.

    For longer texts (chunks): score = matched / total patterns, giving a
    density signal that rewards chunks richly covering the category.

    Returns only non-zero categories.
    """
    import math
    lower = text.lower()
    word_count = len(lower.split())
    is_query = word_count <= 20

    scores: dict[str, float] = {}
    for cat, patterns in _JPD_PATTERNS.items():
        if not patterns:
            continue
        hits = sum(1 for p in patterns if p in lower)
        if hits:
            if is_query:
                # Short text: reward hits inversely proportional to category breadth
                scores[cat] = min(1.0, hits / math.sqrt(len(patterns)))
            else:
                scores[cat] = min(1.0, hits / len(patterns))
    return scores


def _jpd_signal(query_cats: dict[str, float], chunk_text: str) -> tuple[float, list[str]]:
    """Compute JPD match signal [0, 1] and dominant tags for one chunk.

    Strategy: dot-product between query category vector and chunk category
    vector, normalised by total query weight.  Returns (score, [dominant_tags]).
    """
    if not query_cats:
        return 0.0, []

    chunk_cats = _classify_jpd(chunk_text)
    if not chunk_cats:
        return 0.0, []

    # Dot product over shared categories
    numerator = sum(query_cats.get(c, 0.0) * chunk_cats[c] for c in chunk_cats)
    denominator = sum(query_cats.values())
    score = min(1.0, numerator / denominator) if denominator else 0.0

    # Top-1 family tag from chunk categories
    dominant = sorted(chunk_cats, key=lambda c: -chunk_cats[c])[:2]
    tags = list({_JPD_FAMILY.get(c, "O") for c in dominant})

    return score, tags


# Confidence calibration thresholds — recalibrated for 80 %-weight reranker
_CONFIDENCE_HIGH   = 0.55
_CONFIDENCE_MEDIUM = 0.35
_CONFIDENCE_LOW    = 0.18

# Mode-default minimum confidence_label pass threshold.
# "precision" refers to the retrieval method (BM25 term-matching), not
# result quality — lower the floor to "low" so BM25 hits with missing
# authority metadata are still surfaced.  Callers can inspect
# confidence_label to impose stricter downstream filtering.
_MODE_MIN: dict[str, str] = {
    "corpus":    "low",       # keep low + medium + high
    "precision": "low",       # keep low + medium + high (method=BM25, not quality gate)
    "recall":    "abstain",   # keep everything
}

# RRF k constant (Cormack 2009)
_RRF_K = 60

# Max chars in text_preview fields inside trace logs / telemetry
_PREVIEW_LEN = 120

# ---------------------------------------------------------------------------
# Query-embedding LRU cache
# ---------------------------------------------------------------------------
# Caches the last _EMBED_CACHE_MAX (query → vector) pairs in-process.
# Avoids repeated OpenAI calls for the same query string, which is the
# dominant latency source in corpus and recall modes (~5-6 s per cold call).
# Thread/task safe: Python dict operations are GIL-protected and the async
# event loop is single-threaded, so no explicit locking is needed.

_EMBED_CACHE_MAX = 256
_embed_cache: collections.OrderedDict[str, list[float]] = collections.OrderedDict()


async def _embed_with_cache(
    query: str,
    search_id: str = "",
) -> tuple[list[float] | None, float, bool]:
    """Embed *query* with LRU in-process cache.

    Returns ``(embedding, elapsed_ms, cache_hit)``.
    On error returns ``(None, 0.0, False)`` — vector arm is simply skipped.
    """
    cache_key = query.strip().lower()

    if cache_key in _embed_cache:
        _embed_cache.move_to_end(cache_key)   # refresh LRU position
        _log_stage("embed_cache_hit", search_id, query=query[:80])
        return _embed_cache[cache_key], 0.0, True

    try:
        from app.services.embedding_provider import get_embedding_provider, embed_async
        te = time.monotonic()
        provider = get_embedding_provider()
        vecs = await embed_async([query.strip()], provider=provider)
        elapsed_ms = (time.monotonic() - te) * 1000
        embedding = vecs[0] if vecs else None

        if embedding:
            _embed_cache[cache_key] = embedding
            if len(_embed_cache) > _EMBED_CACHE_MAX:
                _embed_cache.popitem(last=False)   # evict LRU entry

        _log_stage(
            "embed",
            search_id,
            provider=type(provider).__name__,
            dim=len(embedding) if embedding else 0,
            embed_ms=elapsed_ms,
            cache_hit=False,
        )
        return embedding, elapsed_ms, False

    except Exception as exc:
        logger.error(
            "corpus_search.embed  search_id=%s  error=%r  (vector arm skipped)",
            search_id, str(exc),
        )
        return None, 0.0, False


# ---------------------------------------------------------------------------
# Structured trace logging helpers
# ---------------------------------------------------------------------------

def _preview(text: str, n: int = _PREVIEW_LEN) -> str:
    """Return first n chars of text, single-line, for log previews."""
    s = (text or "").replace("\n", " ").strip()
    return s[:n] + ("…" if len(s) > n else "")


def _log_stage(stage: str, search_id: str, **fields: Any) -> None:
    """Emit a single structured INFO line.

    Format: ``corpus_search.<stage>  search_id=<id>  k1=v1  k2=v2 …``

    All values are kept short so the line fits in a typical log viewer.
    Numeric floats are rounded to 4 dp.
    """
    parts = [f"search_id={search_id}"]
    for k, v in fields.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v!r}")
    logger.info("corpus_search.%-22s  %s", stage, "  ".join(parts))


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------

def _build_filter_clauses(
    filters: CorpusFilters | None,
    include_document_ids: list[str] | None,
    params: dict[str, Any],
) -> str:
    """Return a WHERE-compatible AND fragment and fill ``params``."""
    clauses: list[str] = []

    if filters:
        if filters.payer:
            clauses.append("document_payer = :f_payer")
            params["f_payer"] = filters.payer
        if filters.state:
            clauses.append("document_state = :f_state")
            params["f_state"] = filters.state
        if filters.program:
            clauses.append("document_program = :f_program")
            params["f_program"] = filters.program
        if filters.authority_level:
            clauses.append("document_authority_level = :f_auth")
            params["f_auth"] = filters.authority_level

    if include_document_ids:
        # Use ANY with a UUID-cast so Postgres doesn't reject the text array.
        # Table-qualify ``document_id`` because the lexicon-expansion path
        # (and the new corpus_search_agent candidate-pool path) add a
        # LEFT JOIN to document_tags, which has its own ``document_id``
        # column and makes an unqualified reference ambiguous. Same class
        # of bug as the 2026-05-01 _BM25_COLS ``id`` ambiguity fix.
        clauses.append(
            "rag_published_embeddings.document_id::text = ANY(:inc_ids)"
        )
        params["inc_ids"] = include_document_ids

    return (" AND " + " AND ".join(clauses)) if clauses else ""


def _row_to_base_dict(row: Any) -> dict[str, Any]:
    """Map a DB row to a plain dict keyed by chunk fields."""
    doc_name = (row["document_display_name"] or "").strip() or row["document_filename"] or ""
    return {
        "id": str(row["id"]),
        "text": row["text"] or "",
        "document_id": str(row["document_id"]),
        "document_name": doc_name,
        "page_number": row["page_number"],
        "paragraph_index": row["paragraph_index"],
        "source_type": row["source_type"] or "hierarchical",
        "authority_level": (row["document_authority_level"] or "").strip() or None,
        "payer": (row["document_payer"] or "").strip() or None,
        "state": (row["document_state"] or "").strip() or None,
        # Section / chapter / summary — surfaced for context-quality
        # diagnostics in the trace UI. May be empty strings in the DB;
        # normalise to None so the frontend can show "—" cleanly.
        "section_path": _none_if_empty(_safe_get(row, "section_path")),
        "chapter_path": _none_if_empty(_safe_get(row, "chapter_path")),
        "summary": _none_if_empty(_safe_get(row, "summary")),
        "content_sha": _none_if_empty(_safe_get(row, "content_sha")),
    }


def _safe_get(row, key: str):
    """``row`` may be a SQLAlchemy Row (mapping-like) or a plain dict.
    Some call paths populate the dict pre-merge without these columns."""
    try:
        return row[key]
    except (KeyError, TypeError):
        return None


def _none_if_empty(v):
    if v is None:
        return None
    s = str(v).strip()
    return s or None


# ---------------------------------------------------------------------------
# Reranker thresholds
# ---------------------------------------------------------------------------

# Hard floor on tag coverage. A chunk that is missing any REQUIRED tag phrase
# is dropped entirely — better to return nothing than to return off-topic
# context that pollutes the LLM prompt. Set to 1.0 to require ALL required
# phrases present; lower to allow partial coverage.
_TAG_COVERAGE_FLOOR = 1.0


# ---------------------------------------------------------------------------
# BM25 arm
# ---------------------------------------------------------------------------

_BM25_COLS = """
    rag_published_embeddings.id,
    rag_published_embeddings.document_id,
    rag_published_embeddings.source_type,
    rag_published_embeddings.text,
    rag_published_embeddings.page_number,
    rag_published_embeddings.paragraph_index,
    rag_published_embeddings.section_path,
    rag_published_embeddings.chapter_path,
    rag_published_embeddings.summary,
    rag_published_embeddings.content_sha,
    rag_published_embeddings.document_display_name,
    rag_published_embeddings.document_filename,
    rag_published_embeddings.document_authority_level,
    rag_published_embeddings.document_payer,
    rag_published_embeddings.document_state
"""
# Columns explicitly table-qualified because the vector arm and the
# relaxed-tag BM25 candidate stage both LEFT JOIN document_tags when
# lexicon expansion matches d/p tags. document_tags also has ``id``
# and ``document_id`` columns, so unqualified SELECT raises
# ``AmbiguousColumnError`` — observed 2026-05-01 silently zeroing out
# every vector arm whose query hit a process/domain tag.

# ---------------------------------------------------------------------------
# BM25 query normalizer
# ---------------------------------------------------------------------------
# plainto_tsquery ANDs every non-stopword token.  Natural-language questions
# like "how many days do I have to file an appeal" inject noise tokens
# ('mani', 'day') that kill the intersection even when the content terms
# match perfectly.  We strip common question lead-phrases and quantifying
# adverbs before handing the query to Postgres.

_QUESTION_LEAD = re.compile(
    r'\b('
    r'how\s+many|how\s+much|how\s+long|how\s+often|how\s+do\s+i|how\s+can\s+i|how\s+do\s+you|'
    r'how\s+is|how\s+are|how\s+does|'
    r'what\s+is\s+the|what\s+are\s+the|what\s+does|what\s+do\s+i|'
    r'when\s+do\s+i|when\s+can\s+i|when\s+is|'
    r'where\s+do\s+i|where\s+can\s+i|'
    r'who\s+can\s+i|who\s+do\s+i|'
    r'can\s+i|do\s+i|have\s+to|need\s+to|'
    r'is\s+there\s+a|are\s+there\s+any'
    r')',
    re.IGNORECASE,
)

# Noise quantifiers that add tokens but never appear in policy documents
_BM25_NOISE = frozenset({
    'many', 'much', 'often', 'several', 'various', 'certain',
    'few', 'some', 'any', 'every', 'all', 'most', 'more',
})

# English function words / PostgreSQL 'english' tsvector stopwords.
# Including these in k-of-n AND queries either causes to_tsquery errors
# or silently removes the term, breaking the k-of-n selectivity guarantee.
# We filter these from filter_tokens (used for WHERE) but NOT from score_ts
# (used for ts_rank_cd, where OR makes stopwords harmless).
_FTS_STOP = frozenset({
    'a', 'an', 'the', 'and', 'or', 'but', 'not', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'do', 'does', 'did', 'have', 'has', 'had',
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it', 'they',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'up',
    'about', 'into', 'through', 'during', 'until', 'against', 'among',
    'when', 'where', 'who', 'which', 'what', 'that', 'this', 'these', 'those',
    'can', 'will', 'just', 'should', 'would', 'could', 'use', 'used', 'using',
    'may', 'how', 'why', 'if', 'than', 'so', 'as', 'such', 'also',
})


def _normalize_bm25_query(query: str) -> str:
    """Strip question lead phrases and noise quantifiers from a query.

    Prevents plainto_tsquery from AND-ing unhelpful tokens like 'mani'
    (stem of 'many') that never appear in policy document text.

    'how many days do I have to file an appeal a denied claim with Sunshine Health'
    → 'days file appeal denied claim Sunshine Health'
    → plainto_tsquery: 'day & file & appeal & deni & claim & sunshin & health'  ✓
    """
    q = _QUESTION_LEAD.sub(' ', query)
    # Remove residual noise words
    words = [w for w in q.split() if w.lower() not in _BM25_NOISE]
    normalized = ' '.join(words).strip()
    return normalized or query  # never return empty — fall back to original


# ---------------------------------------------------------------------------
# tsquery construction helpers
# ---------------------------------------------------------------------------
# We use ``to_tsquery`` (not ``plainto_tsquery``) so we can OR-join
# expansion phrases with raw query tokens.  Input must be carefully
# sanitized: ``to_tsquery`` is a parser and treats !, &, |, (, ), :, *, '
# as operators.  We pre-tokenize on alphanumerics so no operator chars
# can survive into the query string.

def _phrase_to_tsquery_term(phrase: str) -> str:
    """Convert a phrase like 'durable medical equipment' into '(durable & medical & equipment)'.

    Returns "" for empty input or input with no alphanumeric tokens.
    Single-word phrases come back as a bare lowercase token (no parens).
    """
    tokens = re.findall(r"[a-zA-Z0-9]+", phrase or "")
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0].lower()
    return "(" + " & ".join(t.lower() for t in tokens) + ")"


def _build_or_tsquery(*phrases: str) -> str:
    """Build a ``to_tsquery`` string by OR-joining phrase groups.

    Each phrase becomes an AND-group via ``_phrase_to_tsquery_term``;
    groups are then joined with ``|``.  Returns ``""`` if every input
    phrase is empty or non-alphanumeric (caller should fall back).
    """
    parts: list[str] = []
    seen: set[str] = set()
    for p in phrases:
        term = _phrase_to_tsquery_term(p)
        if term and term not in seen:
            seen.add(term)
            parts.append(term)
    return " | ".join(parts) if parts else ""


def _build_kofn_tsquery(tokens: list[str], k: int) -> str:
    """Build a k-of-n tsquery: OR of all C(n,k) AND-combinations of tokens.

    For tokens=[t1,t2,t3,t4,t5] and k=4 this produces:
      (t1&t2&t3&t4) | (t1&t2&t3&t5) | (t1&t2&t4&t5) | (t1&t3&t4&t5) | (t2&t3&t4&t5)

    Each AND group is highly selective, so the GIN bitmap stays compact
    even for the "any k-of-n" level. Caller should cascade k from n down
    to a minimum (e.g. 3) and fall back to pure OR only if all levels fail.

    Caps combinatorial explosion: if C(n,k) > 50, returns "" (caller falls
    through to OR). Tokens are sanitized via _phrase_to_tsquery_term.
    """
    from itertools import combinations as _combinations
    clean = [_phrase_to_tsquery_term(t) for t in tokens]
    clean = [t for t in clean if t]
    n = len(clean)
    if n == 0 or k > n or k < 1:
        return ""
    combos = list(_combinations(clean, k))
    if len(combos) > 50:
        return ""
    parts = ["(" + " & ".join(combo) + ")" if len(combo) > 1 else combo[0]
             for combo in combos]
    return " | ".join(parts)


async def _bm25_arm(
    db: AsyncSession,
    query: str,
    k: int,
    filters: CorpusFilters | None,
    include_document_ids: list[str] | None,
    search_id: str = "",
    tag_mode: str = "auto",
) -> tuple[list[dict[str, Any]], str | None, dict[str, Any]]:
    """Full-text BM25 via Postgres tsvector with lexicon-driven query expansion.

    Pipeline
    --------
    1. Normalize raw query (strip question leads + noise quantifiers).
    2. Expand via ``policy_lexicon_entries`` — match strong_phrases / aliases
       and pull their full phrase bag into the expansion.
    3. Build a ``to_tsquery`` string OR-joining (raw tokens) ∪ (expansion
       phrases).  Multi-word phrases become AND-groups inside parens.
       Brand names not in the lexicon (e.g. "Express Scripts") survive as
       raw tokens because they're always part of the OR.
    4. Run against the multi-field weighted ``search_vec`` (filename A,
       summary B, paths C, text D — see migration
       ``rebuild_rag_published_fts_multifield``).

    Backward-compat fallback: if the lexicon is empty, expansion fails,
    or the resulting tsquery is empty, we still build an OR over the raw
    tokens — strictly looser recall than the previous AND-only behaviour.

    Returns ``(chunks, normalized_query_or_None, expansion_meta)`` where
    ``expansion_meta`` is suitable for embedding directly into telemetry.
    """
    # Lazy import keeps the module loadable in environments without the
    # lexicon table (e.g. unit tests against a sliced schema).
    from app.services.corpus_search_lexicon import (
        LexiconExpansion,
        expand_query_via_lexicon,
    )

    empty_meta: dict[str, Any] = {
        "matched_codes":           [],
        "expansion_phrases":       [],
        "expansion_phrases_count": 0,
        "final_tsquery":           "",
        "log":                     [],
        "domain_tags":             [],
        "jurisdiction_tags":       [],
        "process_tags":            [],
    }

    if not query.strip():
        return [], None, empty_meta

    # ── 1. Normalize ───────────────────────────────────────────────────
    raw_query = query.strip()
    bm25_query = _normalize_bm25_query(raw_query)
    normalized: str | None = bm25_query if bm25_query != raw_query else None
    if normalized and search_id:
        _log_stage("bm25_query_normalized", search_id,
                   original=raw_query[:80], normalized=bm25_query[:80])

    # ── 1.5 Code-like pattern fast-path ────────────────────────────────
    # Queries like ``FL.UM.02.00`` or ``CP.MP.71`` or ``H0019`` are
    # structured policy / HCPCS / revenue codes. The default tokenizer
    # strips the dots → ``fl | um | 02 | 00`` which matches everything.
    # Detect these and do a literal substring match on text + filename
    # first; only fall back to tsquery if the substring search returns
    # nothing (so we don't lose recall for genuinely-rare codes).
    _code_re = re.compile(r"^[A-Z]{1,4}[.\-_]?[A-Z0-9]{1,5}([.\-_][A-Z0-9]{1,5})+$|^[A-Z]\d{4,5}$")
    looks_like_code = bool(_code_re.match(raw_query.upper().strip()))
    if looks_like_code:
        if search_id:
            _log_stage("bm25_code_fastpath", search_id, query=raw_query[:80])
        # ILIKE on the indexed filename/display_name columns ONLY.
        # Skipping the text column: that's a 70k-row sequential scan
        # (no pg_trgm index) and times out at 30s. Policy IDs are
        # nearly always in the filename anyway; if they're only in
        # body text we fall through to tsquery which DOES tokenize
        # most code patterns correctly when there are no dots
        # (e.g. H0019 → ``h0019``).
        code_sql = text(f"""
            SELECT {_BM25_COLS}, 1.0 AS bm25_score
            FROM rag_published_embeddings
            WHERE (document_filename ILIKE :pat
                   OR document_display_name ILIKE :pat)
              {_build_filter_clauses(filters, include_document_ids, {})}
            LIMIT :k
        """)
        params_code = {"k": k, "pat": f"%{raw_query}%"}
        # _build_filter_clauses mutates a params dict; rebuild here
        params_code_full: dict[str, Any] = {"k": k, "pat": f"%{raw_query}%"}
        _ = _build_filter_clauses(filters, include_document_ids, params_code_full)
        # Re-execute with filters merged in (same SQL above already has them)
        try:
            result = await db.execute(code_sql, params_code_full)
            rows = result.mappings().all()
            if rows:
                out: list[dict[str, Any]] = []
                for rank0, row in enumerate(rows):
                    c = _row_to_base_dict(row)
                    c["similarity"] = 1.0
                    c["match_score"] = 1.0
                    c["_arm"] = "bm25_code"
                    out.append(c)
                if search_id:
                    _log_stage("bm25_code_fastpath_hit", search_id,
                               hits=len(out))
                # Skip lexicon expansion; return the code matches directly.
                return out, normalized, {
                    "matched_codes": [],
                    "expansion_phrases": [],
                    "expansion_phrases_count": 0,
                    "final_tsquery": "",
                    "log": [f"code_fastpath: matched {len(out)} via ILIKE"],
                    "domain_tags": [],
                    "jurisdiction_tags": [],
                    "process_tags": [],
                }
            # else: no hits via fast path — fall through to normal tsquery
            if search_id:
                _log_stage("bm25_code_fastpath_miss", search_id,
                           note="no ILIKE hits — falling back to tsquery")
        except Exception as exc:
            logger.warning("bm25 code fast-path error (falling back): %s", exc)

    # ── 2. Lexicon expansion (best-effort; never raises) ───────────────
    try:
        expansion: LexiconExpansion = await expand_query_via_lexicon(db, bm25_query)
    except Exception as exc:
        logger.warning("corpus_search bm25: lexicon expansion failed (continuing with raw tokens): %s", exc)
        expansion = LexiconExpansion()

    # ── 3. Build tsqueries ────────────────────────────────────────────
    # score_ts: OR of raw tokens + expansion phrases — used for ts_rank_cd
    #           scoring (broad, gives partial-match credit)
    # filter_tokens: capped list of raw tokens used to build the WHERE
    #           filter via k-of-n cascade (selective, keeps GIN bitmap small)
    raw_tokens = re.findall(r"[a-zA-Z0-9]+", bm25_query)
    score_ts = _build_or_tsquery(*raw_tokens, *expansion.expansion_phrases)

    if not score_ts:
        # All-junk query (no alphanumerics at all).  Nothing to do.
        return [], normalized, empty_meta

    # Build filter_tokens: strip PostgreSQL English stopwords first, THEN cap.
    # Stopwords in to_tsquery AND-clauses either raise errors or get silently
    # dropped, collapsing k-of-n into a smaller AND and triggering the slow
    # OR fallback. Filtering before capping also ensures content terms past
    # position 7 (e.g. when a question preamble fills the first 7 slots)
    # are still considered. Cap at 7 — C(7,3)=35 stays under the 50-combo guard.
    filter_tokens = [t for t in raw_tokens if t.lower() not in _FTS_STOP and len(t) > 1][:7]

    expansion_meta: dict[str, Any] = {
        "matched_codes":           list(expansion.matched_codes),
        "expansion_phrases":       list(expansion.expansion_phrases),
        "expansion_phrases_count": len(expansion.expansion_phrases),
        "final_tsquery":           score_ts,
        "log":                     list(expansion.log),
        "domain_tags":             list(expansion.domain_tags),
        "jurisdiction_tags":       list(expansion.jurisdiction_tags),
        "process_tags":            list(expansion.process_tags),
    }

    if search_id:
        _log_stage(
            "bm25_lexicon_expansion",
            search_id,
            matched_codes=expansion.matched_codes,
            expansion_count=len(expansion.expansion_phrases),
            final_tsquery=score_ts[:200],
        )

    params: dict[str, Any] = {"k": k, "query": score_ts, "filter_query": score_ts}
    filter_sql = _build_filter_clauses(filters, include_document_ids, params)

    # ── 3.5. Strategy A — Namespace filter via document_tags overlap ──
    # The lexicon emits tag codes like ``j:payor.sunshine_health``,
    # ``d:health_care_services.behavioral_health``,
    # ``p:utilization_management.prior_authorization``.
    # The ``document_tags`` table stores those same codes WITHOUT the
    # ``j:``/``d:``/``p:`` prefix as KEYS in a JSONB OBJECT (not array)
    # like ``{"payor.sunshine_health": 5, "state.florida": 1}``.
    #
    # So we strip the prefix and use ``jsonb_exists()`` (the function
    # form of the ``?`` operator, safer with asyncpg's ``$N`` parameter
    # binding) to test for key presence. Tags within a category OR
    # together; categories OR across each other — this is permissive,
    # not strict (a doc only needs to share ONE tag in any category).
    #
    # Strategy C — Two-stage rank with hard candidate cap. Even if the
    # tag filter doesn't fire (lexicon classified nothing), the CTE
    # caps candidates so ``ts_rank_cd`` never scores more than ``CAP``
    # rows. This is the safety floor: BM25 latency is bounded
    # regardless of corpus size or query expansion shape.
    def _strip_prefix(tag: str, prefix: str) -> str | None:
        return tag[len(prefix):] if tag.startswith(prefix) else None

    j_keys = [k for k in (_strip_prefix(t, "j:") for t in expansion.jurisdiction_tags) if k]
    d_keys = [k for k in (_strip_prefix(t, "d:") for t in expansion.domain_tags) if k]
    p_keys = [k for k in (_strip_prefix(t, "p:") for t in expansion.process_tags) if k]
    has_any_tag = bool(j_keys or d_keys or p_keys)

    # Strategy refinement (2026-04-30): TWO-STAGE filter using
    # documents.payer/state/program METADATA (authoritative) instead
    # of document_tags.j_tags (body-derived, noisy). When the lexicon
    # extracts ``j:regulatory_authority.ahca`` from the query, we
    # filter docs whose ``payer`` column actually IS AHCA — not docs
    # that merely mention AHCA in their text. Same for state/program.
    #
    #   Stage 1 (strict, metadata-J): ``documents.payer/state/program``
    #     match the lexicon-extracted j-tag values. Authoritative.
    #   Stage 2 (relaxed-DP fallback): drop j requirement, narrow by
    #     d/p body-tag overlap. Catches queries that name a jurisdiction
    #     we don't have docs for.
    tag_join_sql = ""
    tag_filter_strict = ""
    tag_filter_relaxed = ""

    # Map j-tag suffix → SQL clause on documents columns.
    # ``rag_published_embeddings`` has document_payer/state/program
    # columns inlined from the publish, so we match those directly.
    # For payor/regulatory_authority we additionally check dt.j_tags
    # (JSONB) via a LEFT JOIN to document_tags, because some docs are
    # published with a generic payer (e.g. "Ahca.myflorida") even though
    # their j_tags contain the correct payor key (e.g. "payor.aetna").
    # The OR catches both cases without extra round-trips.
    _needs_dt_join_for_strict = False

    def _j_to_metadata_clauses(j_keys_list: list[str]) -> tuple[list[str], dict]:
        nonlocal _needs_dt_join_for_strict
        clauses: list[str] = []
        local_params: dict = {}
        i = 0
        for jk in j_keys_list:
            # jk is like "payor.sunshine_health", "regulatory_authority.ahca",
            # "state.fl", "program.medicaid"
            if "." not in jk:
                continue
            cat, val = jk.split(".", 1)
            val_human = val.replace("_", " ")
            pname = f"_meta_{i}"; i += 1
            if cat == "state":
                # state stored as 'FL' (2-char code). Match upper case.
                local_params[pname] = val.upper()[:2]
                clauses.append(f"document_state = :{pname}")
            elif cat == "program":
                local_params[pname] = f"%{val_human}%"
                clauses.append(f"document_program ILIKE :{pname}")
            elif cat in ("payor", "regulatory_authority"):
                # Check both the inlined document_payer column AND the j_tags
                # JSONB on document_tags — some docs authored by a regulatory
                # body (e.g. AHCA) carry the true payor only in j_tags.
                local_params[pname] = f"%{val_human}%"
                jtag_pname = f"_jtag_{i}"
                local_params[jtag_pname] = jk  # e.g. "payor.aetna"
                clauses.append(
                    f"(document_payer ILIKE :{pname} OR jsonb_exists(dt.j_tags, :{jtag_pname}))"
                )
                _needs_dt_join_for_strict = True
        return clauses, local_params

    if has_any_tag:
        # Build STRICT (metadata-J) clauses
        strict_clauses, strict_params = _j_to_metadata_clauses(j_keys)
        if strict_clauses:
            params.update(strict_params)
            tag_filter_strict = " AND (" + " OR ".join(strict_clauses) + ")"

        # Build RELAXED (d/p body-tag) clauses, requires LEFT JOIN
        # to document_tags
        if d_keys or p_keys or _needs_dt_join_for_strict:
            tag_join_sql = " LEFT JOIN document_tags dt ON dt.document_id = rag_published_embeddings.document_id "
            idx = 0
            relaxed_ors: list[str] = []
            for k in d_keys:
                pname = f"_dt_r_{idx}"; idx += 1
                relaxed_ors.append(f"jsonb_exists(dt.d_tags, :{pname})")
                params[pname] = k
            for k in p_keys:
                pname = f"_pt_r_{idx}"; idx += 1
                relaxed_ors.append(f"jsonb_exists(dt.p_tags, :{pname})")
                params[pname] = k
            if relaxed_ors:
                tag_filter_relaxed = " AND (" + " OR ".join(relaxed_ors) + ")"

        if search_id:
            _log_stage(
                "bm25_tag_filter",
                search_id,
                mode="metadata_two_stage",
                j_keys=j_keys, d_keys=d_keys, p_keys=p_keys,
                strict_clauses=len(strict_clauses),
            )
    else:
        if search_id:
            _log_stage(
                "bm25_no_tag_cap",
                search_id,
                note="no j/d/p tags from lexicon — relying on candidate cap only",
            )

    # tag_mode controls which filter we run and whether to fall back.
    #   auto    → strict first, relaxed on zero hits  (back-compat default)
    #   strict  → strict only, return empty if zero  (no auto-fallback)
    #   relaxed → relaxed only, skip strict
    #   none    → no tag filter at all (only candidate cap applies)
    tm = (tag_mode or "auto").lower().strip()
    if tm == "none":
        tag_filter_sql = ""
        tag_filter_relaxed = ""    # disable fallback
        tag_filter_strict = ""
    elif tm == "relaxed":
        tag_filter_sql = tag_filter_relaxed
        tag_filter_strict = ""     # disable retry
        tag_filter_relaxed = ""
    elif tm == "strict":
        tag_filter_sql = tag_filter_strict
        tag_filter_relaxed = ""    # disable fallback
    else:  # auto (default)
        tag_filter_sql = tag_filter_strict
        # tag_filter_relaxed kept as the auto-fallback target

    # ── 4. Execute against multi-field GIN-indexed search_vec ──────────
    # Two-stage CTE pattern: cheap candidate prefilter (uses GIN on
    # search_vec + optional tag overlap), then expensive ``ts_rank_cd``
    # only on the bounded candidate set. Without this, ts_rank_cd was
    # observed at 36s on a 1962-doc corpus (2026-04-29 baseline).
    #
    # WHERE uses a selective k-of-n tsquery (cascade: all-N → any-(N-1)
    # → any-(N-2) → floor-3 → full-OR fallback). Each AND-group in the
    # OR keeps the GIN bitmap compact. Score always uses the full OR
    # query for partial-match credit.
    gin_path = True
    candidate_cap = 1000
    params["_cap"] = candidate_cap
    def _build_main_sql(tfilter: str) -> Any:
        # Determinism fix (2026-05-03): ORDER BY bm25_score DESC, id ASC
        # in BOTH the candidate CTE and the outer SELECT.
        #
        # Without ORDER BY in the CTE, ``LIMIT :_cap`` truncated in
        # whatever physical order Postgres returned rows — heap layout
        # / parallel scan order, NOT stable across executions on the
        # same data. For wide candidate sets (>1000 chunks matching
        # the tsvector) this dropped different chunks each call, which
        # then produced different reranker top-K and different agent
        # answers across runs of the same query (cmhc006/a returned
        # 4 unique top docs across 5 runs in the N=5 variance matrix).
        # The downstream variance was being attributed to the strategy,
        # but it was a Postgres tie-breaking artefact.
        #
        # Adding ORDER BY ts_rank_cd DESC keeps the top-K BM25 matches
        # by score (the principled set), and the secondary ``id ASC``
        # tie-breaks deterministically when scores tie. Outer SELECT
        # repeats the order so the returned rank is deterministic too.
        return text(f"""
            WITH candidates AS (
                SELECT rag_published_embeddings.id,
                       ts_rank_cd(search_vec, to_tsquery('english', :query), 32) AS _ts
                FROM rag_published_embeddings
                {tag_join_sql}
                WHERE search_vec @@ to_tsquery('english', :filter_query)
                  {filter_sql}
                  {tfilter}
                ORDER BY _ts DESC, rag_published_embeddings.id ASC
                LIMIT :_cap
            )
            SELECT {_BM25_COLS},
                ts_rank_cd(search_vec, to_tsquery('english', :query), 32) AS bm25_score
            FROM rag_published_embeddings
            WHERE id IN (SELECT id FROM candidates)
            ORDER BY bm25_score DESC, id ASC
            LIMIT :k
        """)

    # Detect code-like anchor tokens (HCPCS: H0015, T1000; CPT-ish: G0001).
    # When present, try (anchor & score_ts_OR) first — a single specific
    # AND that keeps the GIN bitmap tiny (code appears in <50 docs typically).
    # This runs before the k-of-n cascade and short-circuits on a hit.
    _code_re_anchor = re.compile(r'^[A-Z]\d{3,5}$|^[A-Z]{2}\d{3,4}$')
    anchor_tokens = [t.lower() for t in filter_tokens if _code_re_anchor.match(t.upper())]

    async def _run_with_kofn_cascade(tfilter: str) -> list:
        """Try anchor-AND, then k-of-n from all-N down to floor, then full OR."""
        # ── anchor fast-path ───────────────────────────────────────────────
        if anchor_tokens:
            anchor_and = " & ".join(anchor_tokens)
            rest_or = _build_or_tsquery(*[t for t in filter_tokens if t.lower() not in set(anchor_tokens)])
            anchor_q = f"({anchor_and}) & ({rest_or})" if rest_or else anchor_and
            params["filter_query"] = anchor_q
            if search_id:
                _log_stage("bm25_kofn_filter", search_id,
                           k="anchor", anchor=anchor_and, filter_query=anchor_q[:120])
            result = await db.execute(_build_main_sql(tfilter), params)
            rows = result.mappings().all()
            if rows:
                return list(rows)

        # ── k-of-n cascade ─────────────────────────────────────────────────
        n = len(filter_tokens)
        min_k = max(3, (n + 1) // 2)  # at least half the tokens, floor 3
        levels = list(range(n, min_k - 1, -1))
        for k_level in levels:
            kofn_ts = _build_kofn_tsquery(filter_tokens, k_level)
            if not kofn_ts:
                continue
            params["filter_query"] = kofn_ts
            if search_id:
                _log_stage("bm25_kofn_filter", search_id,
                           k=k_level, n=n, filter_query=kofn_ts[:120])
            result = await db.execute(_build_main_sql(tfilter), params)
            rows = result.mappings().all()
            if rows:
                return list(rows)
        # Full OR fallback — always uses score_ts (broad)
        params["filter_query"] = score_ts
        if search_id:
            _log_stage("bm25_kofn_filter", search_id,
                       k="or_fallback", n=n, filter_query=score_ts[:120])
        result = await db.execute(_build_main_sql(tfilter), params)
        return list(result.mappings().all())

    try:
        # Stage 1: strict tag filter with k-of-n cascade
        rows_check = await _run_with_kofn_cascade(tag_filter_strict)
        # If zero hits AND we have a relaxed fallback different from strict,
        # retry with the relaxed filter.
        if not rows_check and tag_filter_relaxed and tag_filter_relaxed != tag_filter_strict:
            if search_id:
                _log_stage(
                    "bm25_tag_filter_relaxed",
                    search_id,
                    note="strict (hard-J) returned 0; retrying with relaxed (d/p only)",
                )
            rows_check = await _run_with_kofn_cascade(tag_filter_relaxed)
        # Package back via a tiny adapter so the downstream code that
        # expects ``result.mappings().all()`` keeps working without restructuring.
        class _R:
            def __init__(self, rows): self._rows = rows
            def mappings(self):
                class _M:
                    def __init__(self, r): self._r = r
                    def all(self): return self._r
                return _M(self._rows)
        result = _R(rows_check)
    except Exception as exc:
        if "search_vec" in str(exc):
            # Multi-field migration not yet applied — fall back to inline
            # to_tsvector over text only.  Identical correctness, slower.
            gin_path = False
            logger.warning("corpus_search bm25: search_vec column missing, using inline tsvector: %s", exc)
            sql = text(f"""
                WITH candidates AS (
                    SELECT rag_published_embeddings.id,
                           ts_rank_cd(
                               to_tsvector('english', coalesce(text, '')),
                               to_tsquery('english', :query), 32
                           ) AS _ts
                    FROM rag_published_embeddings
                    {tag_join_sql}
                    WHERE to_tsvector('english', coalesce(text, ''))
                          @@ to_tsquery('english', :query)
                      {filter_sql}
                      {tag_filter_sql}
                    ORDER BY _ts DESC, rag_published_embeddings.id ASC
                    LIMIT :_cap
                )
                SELECT {_BM25_COLS},
                    ts_rank_cd(
                        to_tsvector('english', coalesce(text, '')),
                        to_tsquery('english', :query), 32
                    ) AS bm25_score
                FROM rag_published_embeddings
                WHERE id IN (SELECT id FROM candidates)
                ORDER BY bm25_score DESC, id ASC
                LIMIT :k
            """)
            result = await db.execute(sql, params)
        else:
            raise

    rows = result.mappings().all()
    out: list[dict[str, Any]] = []
    for rank0, row in enumerate(rows):
        c = _row_to_base_dict(row)
        raw_score = float(row["bm25_score"] or 0.0)
        # ts_rank_cd returns [0, 1] with log normalisation (flag 32).
        c["similarity"] = min(1.0, raw_score)
        c["match_score"] = c["similarity"]
        c["_arm"] = "bm25"
        out.append(c)
        _log_stage(
            "bm25_arm",
            search_id,
            rank=rank0 + 1,
            chunk_id=c["id"],
            doc=c["document_name"][:50],
            page=c["page_number"],
            authority=c["authority_level"],
            ts_rank=raw_score,
            gin_path=gin_path,
            preview=_preview(c["text"]),
        )

    if search_id:
        _log_stage("bm25_arm_summary", search_id, hits=len(out), gin_path=gin_path)
    return out, normalized, expansion_meta


# ---------------------------------------------------------------------------
# Vector arm
# ---------------------------------------------------------------------------

async def _vector_arm(
    db: AsyncSession,
    query_embedding: list[float],
    k: int,
    filters: CorpusFilters | None,
    include_document_ids: list[str] | None,
    search_id: str = "",
    expansion: "LexiconExpansion | None" = None,
    tag_mode: str = "auto",
    min_similarity: float | None = None,
    over_fetch_factor: int = 1,
) -> list[dict[str, Any]]:
    """pgvector ANN over rag_published_embeddings.embedding_vec (HNSW cosine).

    Honors the same ``tag_mode`` semantics as the BM25 arm so the
    chosen jurisdiction filter applies symmetrically across both
    search arms in hybrid retrieval.

    ``min_similarity`` (0..1, post-filter on cosine similarity) drops
    chunks below the threshold AFTER the SQL ORDER BY/LIMIT. Useful
    when the corpus has a large boilerplate cluster that ties at one
    similarity value: the rerank later may dramatically change scores,
    so we only want to bring forward candidates that have a reasonable
    initial similarity. Defaults to ``None`` (no filter).

    ``over_fetch_factor`` multiplies ``k`` for the SQL LIMIT before
    threshold filtering. With over_fetch_factor=8 and k=80 we ask
    Postgres for 640 candidates, drop the boilerplate-tied ones below
    the threshold, then keep up to k for downstream rerank. Cheap
    insurance against HNSW tie-crowding.
    """
    sql_limit = max(1, k) * max(1, int(over_fetch_factor))
    params: dict[str, Any] = {"k": sql_limit}
    filter_sql = _build_filter_clauses(filters, include_document_ids, params)
    # pgvector text form: '[f1,f2,...]'
    params["query_vec"] = "[" + ",".join(repr(float(x)) for x in query_embedding) + "]"

    # Build metadata-J / d-p filters from the lexicon expansion, mirroring
    # the BM25 arm. Without this, vector search would surface AHCA-mention
    # docs (e.g., Sunshine Provider Manual) for an "AHCA rules" query
    # when the user explicitly wants AHCA-authored content.
    tag_filter_strict = ""
    tag_filter_relaxed = ""
    tag_join_sql = ""
    if expansion is not None and (tag_mode or "auto").lower() != "none":
        def _strip_prefix(tag: str, prefix: str) -> str | None:
            return tag[len(prefix):] if tag.startswith(prefix) else None
        j_keys = [k for k in (_strip_prefix(t, "j:") for t in expansion.jurisdiction_tags) if k]
        d_keys = [k for k in (_strip_prefix(t, "d:") for t in expansion.domain_tags) if k]
        p_keys = [k for k in (_strip_prefix(t, "p:") for t in expansion.process_tags) if k]
        idx = 0
        # Strict — metadata-J on rag_published_embeddings inlined columns
        strict_clauses: list[str] = []
        for jk in j_keys:
            if "." not in jk:
                continue
            cat, val = jk.split(".", 1)
            val_human = val.replace("_", " ")
            pname = f"_v_meta_{idx}"; idx += 1
            if cat == "state":
                params[pname] = val.upper()[:2]
                strict_clauses.append(f"document_state = :{pname}")
            elif cat == "program":
                params[pname] = f"%{val_human}%"
                strict_clauses.append(f"document_program ILIKE :{pname}")
            elif cat in ("payor", "regulatory_authority"):
                params[pname] = f"%{val_human}%"
                strict_clauses.append(f"document_payer ILIKE :{pname}")
        if strict_clauses:
            tag_filter_strict = " AND (" + " OR ".join(strict_clauses) + ")"
        # Relaxed — d/p body tags (require JOIN)
        if d_keys or p_keys:
            tag_join_sql = " LEFT JOIN document_tags dt ON dt.document_id = rag_published_embeddings.document_id "
            relaxed_ors: list[str] = []
            for k_ in d_keys:
                pname = f"_v_dt_r_{idx}"; idx += 1
                relaxed_ors.append(f"jsonb_exists(dt.d_tags, :{pname})")
                params[pname] = k_
            for k_ in p_keys:
                pname = f"_v_pt_r_{idx}"; idx += 1
                relaxed_ors.append(f"jsonb_exists(dt.p_tags, :{pname})")
                params[pname] = k_
            if relaxed_ors:
                tag_filter_relaxed = " AND (" + " OR ".join(relaxed_ors) + ")"

    tm = (tag_mode or "auto").lower().strip()
    if tm == "none":
        tag_filter_sql = ""
        tag_filter_relaxed = ""
    elif tm == "relaxed":
        tag_filter_sql = tag_filter_relaxed
        tag_filter_relaxed = ""    # disable retry
    elif tm == "strict":
        tag_filter_sql = tag_filter_strict
        tag_filter_relaxed = ""
    else:  # auto
        tag_filter_sql = tag_filter_strict

    def _build_vec_sql(tfilter: str) -> Any:
        return text(f"""
            SELECT {_BM25_COLS},
                1 - (embedding_vec <=> CAST(:query_vec AS vector)) AS similarity
            FROM rag_published_embeddings
            {tag_join_sql}
            WHERE embedding_vec IS NOT NULL
              {filter_sql}
              {tfilter}
            ORDER BY embedding_vec <=> CAST(:query_vec AS vector)
            LIMIT :k
        """)

    try:
        result = await db.execute(_build_vec_sql(tag_filter_sql), params)
        rows_check = result.mappings().all()
        # Auto fallback: if strict returned 0 and we have a relaxed
        # fallback, retry with that filter.
        if not rows_check and tag_filter_relaxed and tag_filter_relaxed != tag_filter_sql:
            if search_id:
                _log_stage(
                    "vector_tag_filter_relaxed",
                    search_id,
                    note="strict returned 0; retrying with relaxed (d/p only)",
                )
            result = await db.execute(_build_vec_sql(tag_filter_relaxed), params)
            rows_check = result.mappings().all()

        class _R:
            def __init__(self, rows): self._rows = rows
            def mappings(self):
                class _M:
                    def __init__(self, r): self._r = r
                    def all(self): return self._r
                return _M(self._rows)
        result = _R(rows_check)
    except Exception as exc:
        logger.error("corpus_search vector arm failed: %s", exc, exc_info=True)
        return []

    rows = result.mappings().all()
    out: list[dict[str, Any]] = []
    n_below_threshold = 0
    for rank0, row in enumerate(rows):
        cosine_sim = max(0.0, min(1.0, float(row["similarity"] or 0.0)))
        if min_similarity is not None and cosine_sim < min_similarity:
            n_below_threshold += 1
            continue
        c = _row_to_base_dict(row)
        c["similarity"] = cosine_sim
        c["match_score"] = cosine_sim
        c["_arm"] = "vector"
        out.append(c)
        if len(out) >= k:
            break
        _log_stage(
            "vector_arm",
            search_id,
            rank=rank0 + 1,
            chunk_id=c["id"],
            doc=c["document_name"][:50],
            page=c["page_number"],
            authority=c["authority_level"],
            cosine=cosine_sim,
            preview=_preview(c["text"]),
        )

    if search_id:
        _log_stage(
            "vector_arm_summary",
            search_id,
            hits=len(out),
            scanned=len(rows),
            below_threshold=n_below_threshold,
            threshold=min_similarity,
            sql_limit=sql_limit,
        )
    return out


# ---------------------------------------------------------------------------
# RRF fusion  (ported from mobius-chat/app/services/retriever_hybrid.py)
# ---------------------------------------------------------------------------

def _rrf_merge(
    arms: dict[str, list[dict[str, Any]]],
    k: int = _RRF_K,
    search_id: str = "",
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion.  Returns chunks ordered by RRF score (desc)."""
    fused: dict[str, dict[str, Any]] = {}
    for arm_name, ranked in arms.items():
        for rank0, chunk in enumerate(ranked):
            cid = chunk.get("id") or ""
            if not cid:
                continue
            rank1 = rank0 + 1
            contribution = 1.0 / (k + rank1)

            if cid not in fused:
                fused[cid] = dict(chunk)
                fused[cid]["retrieval_arms"] = [arm_name]
                fused[cid]["arm_ranks"] = {arm_name: rank1}
                fused[cid]["arm_scores"] = {arm_name: float(chunk.get("similarity", 0.0))}
                fused[cid]["rrf_score"] = contribution
            else:
                # Fill missing fields; never overwrite existing non-null text/metadata
                for key, val in chunk.items():
                    if key in ("retrieval_arms", "arm_ranks", "arm_scores", "rrf_score"):
                        continue
                    if fused[cid].get(key) in (None, "", []) and val not in (None, "", []):
                        fused[cid][key] = val
                if arm_name not in fused[cid]["retrieval_arms"]:
                    fused[cid]["retrieval_arms"].append(arm_name)
                fused[cid]["arm_ranks"][arm_name] = rank1
                fused[cid]["arm_scores"][arm_name] = float(chunk.get("similarity", 0.0))
                fused[cid]["rrf_score"] += contribution

    def _sort_key(c: dict[str, Any]) -> tuple[float, int]:
        best_rank = min(c.get("arm_ranks", {}).values() or [999])
        return (-float(c.get("rrf_score", 0.0)), best_rank)

    out = sorted(fused.values(), key=_sort_key)
    # Promote rrf_score → similarity for uniform downstream handling
    for rank0, c in enumerate(out):
        c["similarity"] = float(c.get("rrf_score", 0.0))
        if search_id:
            _log_stage(
                "rrf_fusion",
                search_id,
                rrf_rank=rank0 + 1,
                chunk_id=c["id"],
                doc=c["document_name"][:50],
                arms="+".join(c.get("retrieval_arms") or []),
                arm_ranks=c.get("arm_ranks"),
                arm_scores={a: round(s, 4) for a, s in (c.get("arm_scores") or {}).items()},
                rrf_score=c["similarity"],
            )
    return out


# ---------------------------------------------------------------------------
# Reranker  (simplified — score + authority + length signals)
# ---------------------------------------------------------------------------

def _authority_score(level: str | None) -> float:
    if not level:
        return _AUTHORITY_DEFAULT
    return _AUTHORITY_WEIGHTS.get((level or "").strip().lower(), _AUTHORITY_DEFAULT)


def _length_score(t: str) -> float:
    """Reward texts up to 500 chars; penalise stubs below 50."""
    n = len(t or "")
    if n < 50:
        return 0.0
    return min(1.0, (n - 50) / 450)


def _best_arm_sim(c: dict[str, Any]) -> float:
    """Return the most discriminating similarity signal for a candidate chunk.

    Problem: in corpus (RRF) mode ``c["similarity"]`` is the RRF score
    (≈ 0.015), which is too compressed to separate relevant from irrelevant
    results — all candidates cluster within ±0.001.  Using the raw per-arm
    scores gives far better discrimination.

    Rescaling:
      • BM25 (ts_rank_cd, flag 32) is already [0, 1] — used as-is.
      • Cosine similarity is [0, 1] but in practice stays above ≈ 0.5 for
        nearest-neighbour results.  Rescaling to (cosine − 0.5) × 2 maps
        the useful [0.5, 1.0] range onto [0, 1] so the 0.30-weight signal
        spans the full range rather than being squeezed into [0, 0.8].
    """
    arm_scores: dict[str, float] = c.get("arm_scores") or {}
    if not arm_scores:
        # Single-arm bypass path — similarity already holds the raw arm score
        return float(c.get("similarity") or 0.0)

    best = 0.0
    for arm, score in arm_scores.items():
        if arm == "vector":
            rescaled = max(0.0, (float(score) - 0.5) * 2.0)
        else:                        # bm25 — ts_rank_cd already [0, 1]
            rescaled = float(score)
        best = max(best, rescaled)
    return best


# ---------------------------------------------------------------------------
# Rerank haystack helpers — used by both tag_coverage and jpd signals
# ---------------------------------------------------------------------------
#
# The reranker's body-only haystack threw away crucial doc-level context.
# A chunk in ``FL_SunshineHealth_Caid_PAReq...`` discussing "H0019" in a
# table doesn't repeat the words "Sunshine Health" or "prior
# authorization" in its body — those concepts live in the doc title,
# section header, and inherited tags. Without doc-level context the
# reranker scored the chunk at 0.32 (and the floor dropped it to 0).
#
# Two enrichments fix this:
#
#   1. ``_body_haystack(c)``  — chunk body PLUS any pre-fetched neighbor
#      paragraphs (``c["_neighbor_text"]``). The neighbor pass happens
#      ONCE per corpus_search call, batched, before rerank. Neighbors
#      give us the section header / table label / paragraph above and
#      below — which is where the topical context lives.
#
#   2. ``_meta_haystack(c)``  — doc filename + display name + payer +
#      state + section_path + chapter_path + summary. Captures all the
#      doc-level metadata as a separate evidence type, used by the
#      ``meta_boost`` signal in the formula.
#
# Both haystacks are normalised (lowercased + whitespace-collapsed) so
# substring matching stays simple and case-insensitive.

def _normalise_for_haystack(s: str | None) -> str:
    if not s:
        return ""
    return " ".join(str(s).lower().split())


def _body_haystack(c: dict[str, Any]) -> str:
    """Body text + any pre-fetched neighbor paragraph bodies."""
    parts: list[str] = [_normalise_for_haystack(c.get("text"))]
    nbr = _normalise_for_haystack(c.get("_neighbor_text"))
    if nbr:
        parts.append(nbr)
    return " | ".join(p for p in parts if p)


def _meta_haystack(c: dict[str, Any]) -> str:
    """Doc-level metadata that semantically belongs to this chunk
    even when the body doesn't repeat it. Used by ``meta_boost``.

    Filename gets a dual treatment: raw + an underscore-replaced
    version, because filenames like ``FL_SunshineHealth_Caid_PAReq``
    pack multiple words into a single token — splitting on
    ``_/-`` lets ``"sunshine health"`` substring-match.

    Inherited d/j/p tag leaves are also folded in (when
    ``_attach_inherited_doc_tags`` ran upstream): a tag like
    ``utilization_management.prior_authorization`` contributes its
    leaf ``"prior authorization"`` to the haystack so a chunk
    inherits the topical concepts assigned at ingest time.
    """
    parts: list[str] = []
    for key in (
        "document_name", "document_filename", "document_display_name",
        "payer", "state", "section_path", "chapter_path", "summary",
    ):
        v = c.get(key)
        if not v:
            continue
        parts.append(_normalise_for_haystack(v))
        # Filename / paths often pack multi-word concepts into one
        # token via separators. Add a separator-split version so
        # substring lookups work.
        if key in ("document_filename", "section_path", "chapter_path"):
            split = _normalise_for_haystack(
                str(v).replace("_", " ").replace("-", " ").replace("/", " ").replace(".", " ")
            )
            if split:
                parts.append(split)
    # Inherited document tags — leaf names with underscores expanded
    # to spaces so substring matching works.
    for tag_key in ("_doc_d_tags", "_doc_j_tags", "_doc_p_tags"):
        for tag in (c.get(tag_key) or []):
            # Leaf is the bit after the last "."; expand underscores.
            leaf = str(tag).split(".")[-1].replace("_", " ").strip().lower()
            if leaf:
                parts.append(leaf)
            # Also fold the FULL dotted path expanded — captures parent
            # category words too (e.g. "utilization management" from
            # "utilization_management.prior_authorization").
            full = str(tag).replace(".", " ").replace("_", " ").strip().lower()
            if full and full != leaf:
                parts.append(full)
    return " | ".join(p for p in parts if p)


def _rerank(
    chunks: list[dict[str, Any]],
    search_id: str = "",
    query: str = "",
    required_phrases: list[str] | None = None,
    required_phrase_weights: list[float] | None = None,
    required_phrase_tag_codes: list[str | None] | None = None,
) -> list[dict[str, Any]]:
    """Apply weighted signals and sort by rerank_score (desc).

    Weights (reranker_v1.2 — 2026-05-03 hayack-expansion):
      sim (0.25) + authority (0.10) + length (0.05) + jpd (0.20)
        + tag_coverage (0.40) + meta_boost (0.15)   = up to 1.15

    Two changes vs v1.1:

    1. ``tag_coverage`` and ``jpd`` now run against an ENRICHED
       body haystack — the chunk's body PLUS any pre-fetched
       neighbor paragraphs (``c["_neighbor_text"]`` set by the
       upstream sibling-fetch pass). This fixes the failure mode
       where a chunk in a topically-tagged doc has a sparse body
       (e.g. just a row in a code table) and the body-only
       reranker scored it near zero even though its parent
       document is exactly on-topic. The neighbor paragraphs
       carry the section header / table label that gives the
       chunk its semantic context.

    2. New ``meta_boost`` signal — fraction of ``required_phrases``
       found in the chunk's DOC-LEVEL metadata (filename,
       display name, payer, state, section_path, chapter_path,
       summary). This is a SEPARATE signal from ``tag_coverage``
       so a chunk that lives in the right document but whose body
       doesn't substantively talk about the topic still gets some
       credit (instead of being silently dropped by the body-only
       coverage floor).

    When ``required_phrases`` is empty/None (e.g. literal-only
    queries or PRECISION_DOMINANT cases), both ``tag_coverage`` and
    ``meta_boost`` fall out of the mix.
    """
    if not chunks:
        return chunks

    # Pre-classify query JPD intent once (shared across all chunks)
    query_cats = _classify_jpd(query) if query else {}
    has_jpd = bool(query_cats)

    has_tag_cov = bool(required_phrases)
    required_lower = [p.lower() for p in (required_phrases or []) if p]
    has_meta = has_tag_cov   # meta_boost only meaningful when phrases set

    # Selectivity-weighted phrase coverage. The classifier already
    # computed an IDF-style selectivity per tag (rarer / more
    # discriminating tags get higher selectivity, e.g. payor names at
    # ~0.93 vs broad domains at ~0.79). Pass-through here so a chunk
    # that hits a rare term ("sunshine health") gets more credit than
    # a chunk that hits a common term ("prior authorization") — fixes
    # the failure mode where tangential AHCA quarterly reports outscore
    # the actual Sunshine Provider Manual on a Sunshine PA query
    # because both contain the common term but only one contains the
    # rare one.
    if (
        required_phrase_weights
        and len(required_phrase_weights) == len(required_lower)
        and any(w > 0 for w in required_phrase_weights)
    ):
        phrase_weights = [max(0.0, float(w)) for w in required_phrase_weights]
    else:
        phrase_weights = [1.0] * len(required_lower)
    total_weight = sum(phrase_weights) or 1.0

    # Per-phrase tag codes (j:/d:/p: full codes) aligned by index. j-tags
    # get BINARY doc-level credit: if the chunk's parent doc carries the
    # j-tag (per ``_doc_j_tags`` attached upstream), the phrase counts
    # as present even if the body/meta haystack doesn't substring-match
    # the leaf word. This reflects how j-tags work — payor / state /
    # jurisdiction is a yes/no domain membership for the doc, not a
    # topical theme that has to be repeated in every paragraph. d-tags
    # are softer (a doc MAY cover this topic); we keep substring matching
    # for d/p so a chunk in a multi-topic doc only gets credit for the
    # topics actually discussed in its body or section header.
    if (
        required_phrase_tag_codes
        and len(required_phrase_tag_codes) == len(required_lower)
    ):
        phrase_tag_codes: list[str | None] = list(required_phrase_tag_codes)
    else:
        phrase_tag_codes = [None] * len(required_lower)

    # Reranker v1.3 (2026-05-03): unified selectivity-weighted
    # coverage. The earlier v1.2 split body and meta into two separate
    # signals (tag_coverage 0.40 + meta_boost 0.15) which double-counted
    # phrases hitting BOTH body and meta. With selectivity weights
    # added on top this caused regressions (cmhc001 dropped from
    # CORRECT to HONEST_ABSTAIN because the right chunk had the rare
    # phrase only in meta, not in body, while a sibling chunk had the
    # phrase in BOTH and outscored it). Now: a single weighted
    # ``coverage`` signal where each phrase counts once, scored by
    # selectivity, presence checked across body OR meta.
    W_SIM = 0.25
    W_AUTH = 0.10
    W_LEN = 0.05
    W_JPD = 0.20 if has_jpd else 0.0
    W_COV = 0.55 if has_tag_cov else 0.0   # was tag_coverage 0.40 + meta_boost 0.15
    MAX_WEIGHT = W_SIM + W_AUTH + W_LEN + W_JPD + W_COV

    for c in chunks:
        sim   = _best_arm_sim(c)
        auth  = _authority_score(c.get("authority_level"))
        body  = (c.get("text") or "")
        lsig  = _length_score(body)

        # Build enriched haystacks once per chunk.
        body_hay = _body_haystack(c)   # body + neighbor paragraphs
        meta_hay = _meta_haystack(c) if has_meta else ""

        # JPD signal now reads the enriched body haystack so chunks
        # whose section header is in a sibling paragraph get credit.
        jpd, jpd_tags = _jpd_signal(query_cats, body_hay) if has_jpd else (0.0, [])

        # Unified weighted coverage — body OR meta counts as evidence
        # for each phrase. Each phrase counts ONCE, weighted by its
        # selectivity. No double-count when a phrase appears in both
        # body and meta. This makes a chunk in a payer-tagged doc
        # whose body doesn't repeat the payer name score equally with
        # a chunk that DOES repeat the payer name — the doc filename /
        # payer field already tells us who the chunk belongs to.
        if has_tag_cov:
            # Binary j-tag credit: a phrase whose tag_code is a j-code
            # counts as present when the chunk's parent doc carries that
            # j-code (e.g. j:payor.sunshine_health). Only when the doc
            # ISN'T tagged do we fall back to substring matching against
            # the body/meta haystack to see if the chunk's section at
            # least belongs to that j-domain.
            doc_j_tags = c.get("_doc_j_tags") or {}
            # _doc_j_tags is a JSONB object keyed by code-without-prefix
            # (e.g. {"payor.sunshine_health": {...}}). Treat dict-keys or
            # list entries uniformly.
            if isinstance(doc_j_tags, dict):
                doc_j_codes = set(doc_j_tags.keys())
            elif isinstance(doc_j_tags, (list, tuple, set)):
                doc_j_codes = set(doc_j_tags)
            else:
                doc_j_codes = set()

            body_present: list[str] = []
            meta_present: list[str] = []
            jtag_present: list[str] = []   # binary-credit hits, for telemetry
            combined_present: set[str] = set()
            for i, p in enumerate(required_lower):
                code = phrase_tag_codes[i] if i < len(phrase_tag_codes) else None
                # j-tag binary credit
                if code and code.startswith("j:"):
                    j_code_body = code.split(":", 1)[1]
                    if j_code_body in doc_j_codes:
                        jtag_present.append(p)
                        combined_present.add(p)
                        continue   # don't double-count via substring
                # Substring fallback (body OR meta)
                in_body = p in body_hay
                in_meta = bool(meta_hay) and p in meta_hay
                if in_body:
                    body_present.append(p)
                if in_meta:
                    meta_present.append(p)
                if in_body or in_meta:
                    combined_present.add(p)
            present = list(combined_present)
            missing = [p for p in required_lower if p not in combined_present]
            cov = sum(
                phrase_weights[i] for i, p in enumerate(required_lower)
                if p in combined_present
            ) / total_weight
            combined_present_set = combined_present
        else:
            cov = 0.0
            present = []
            missing = []
            body_present = []
            meta_present = []
            jtag_present = []
            combined_present_set = set()

        # Legacy variables for the per-signal breakdown / floor (kept
        # for telemetry). The floor uses the same combined coverage as
        # the score now, so they're equal.
        tag_cov = cov
        meta_boost = (
            sum(phrase_weights[i] for i, p in enumerate(required_lower)
                if p in meta_present) / total_weight
            if has_tag_cov and meta_present else 0.0
        )
        combined_cov = cov
        combined_missing = missing
        # Mutate the chunk so the downstream floor (in _assemble) uses
        # the union, not body-only.
        c["_combined_coverage"] = combined_cov
        c["_combined_missing"] = combined_missing

        raw   = (W_SIM * sim
                 + W_AUTH * auth
                 + W_LEN * lsig
                 + W_JPD * jpd
                 + W_COV * cov)
        score = raw / MAX_WEIGHT if MAX_WEIGHT > 0 else raw
        c["rerank_score"] = score
        c["_jpd_tags"] = jpd_tags
        # Stash per-signal breakdown for trace / telemetry
        c["_rerank_signals"] = {
            "sim_raw":              round(sim, 4),
            "sim_weighted":         round(W_SIM * sim, 4),
            "authority_raw":        round(auth, 4),
            "authority_weighted":   round(W_AUTH * auth, 4),
            "length_raw":           round(lsig, 4),
            "length_weighted":      round(W_LEN * lsig, 4),
            "jpd_raw":              round(jpd, 4),
            "jpd_weighted":         round(W_JPD * jpd, 4),
            "jpd_tags":             jpd_tags,
            "neighbor_chars":       len(c.get("_neighbor_text") or ""),
            "coverage_raw":         round(cov, 4),
            "coverage_weighted":    round(W_COV * cov, 4),
            "coverage_present":     present,
            "coverage_missing":     missing,
            "coverage_body_only":   body_present,
            "coverage_meta_only":   [p for p in meta_present if p not in body_present],
            "coverage_jtag_binary": jtag_present,
            "total_raw":            round(raw, 4),
            "rerank_score":         round(score, 4),
            "max_weight":           round(MAX_WEIGHT, 4),
        }
        if search_id:
            _log_stage(
                "rerank_signal",
                search_id,
                chunk_id=c["id"],
                doc=c["document_name"][:50],
                arm=c.get("_arm") or "+".join(c.get("retrieval_arms") or []),
                sim=sim,
                auth=auth,
                length=lsig,
                jpd=round(jpd, 4),
                jpd_tags=jpd_tags,
                raw=raw,
                rerank_score=score,
                authority_level=c.get("authority_level"),
            )

    # ── Tag-coverage hard floor (Fix #5 — 2026-05-02) ─────────────────
    # When the query has REQUIRED phrases, drop any chunk whose body
    # text doesn't contain ALL of them. The soft tag_coverage signal
    # earlier already penalised partial-coverage chunks via the score,
    # but they could still appear when nothing better exists. Better to
    # return nothing than off-topic. This makes the strategy abstain
    # cleanly so the router falls to a fallback.
    pre_coverage_count = len(chunks)
    dropped_coverage_ids: list[str] = []
    if has_tag_cov:
        chunks_after_cov: list[dict[str, Any]] = []
        for c in chunks:
            # Use COMBINED coverage (body OR meta) for the floor.
            # A chunk in the right document whose body is sparse
            # (e.g. just a row in a code table) shouldn't be dropped
            # as long as the doc-level metadata covers what's missing
            # from the body. Body-only ``tag_coverage_raw`` and
            # meta-only ``meta_boost_raw`` still feed the SCORE
            # separately as different evidence types.
            cov = c.get("_combined_coverage")
            if cov is None:
                cov = (c.get("_rerank_signals") or {}).get("tag_coverage_raw", 1.0)
            # Promoted topic-block neighbors (e.g. table rows whose body
            # is just numbers) are exempt from the coverage floor — they
            # exist to give the synthesis LLM the full surrounding
            # context for a high-coverage seed. Their parent seed has
            # full coverage; dropping them throws away the answer table.
            is_promoted_neighbor = (
                c.get("_promoted_from_seed") is not None
                or "bm25_inherited" in (c.get("retrieval_arms") or [])
            )
            if cov < _TAG_COVERAGE_FLOOR and not is_promoted_neighbor:
                dropped_coverage_ids.append(c["id"])
                if search_id:
                    _log_stage(
                        "rerank_coverage_drop",
                        search_id,
                        chunk_id=c["id"],
                        doc=c.get("document_name", "")[:50],
                        combined_coverage=cov,
                        body_coverage=(c.get("_rerank_signals") or {}).get("tag_coverage_raw"),
                        meta_boost=(c.get("_rerank_signals") or {}).get("meta_boost_raw"),
                        floor=_TAG_COVERAGE_FLOOR,
                        missing=c.get("_combined_missing"),
                    )
                continue
            chunks_after_cov.append(c)
        chunks = chunks_after_cov
        if search_id:
            _log_stage(
                "rerank_coverage_summary",
                search_id,
                pre=pre_coverage_count,
                post=len(chunks),
                dropped=len(dropped_coverage_ids),
                floor=_TAG_COVERAGE_FLOOR,
            )

    # Per-category decay: drop chunks where rerank_score < 0.6 × best in category
    cat_best: dict[str, float] = {}
    for c in chunks:
        arm = (c.get("_arm") or "vector")
        cat = f"{arm}_{c.get('source_type', 'hierarchical')}"
        cat_best[cat] = max(cat_best.get(cat, 0.0), c["rerank_score"])

    filtered: list[dict[str, Any]] = []
    dropped_decay: list[str] = []
    for c in chunks:
        arm = (c.get("_arm") or "vector")
        cat = f"{arm}_{c.get('source_type', 'hierarchical')}"
        best = cat_best.get(cat, 0.0)
        if best > 0 and c["rerank_score"] < 0.6 * best:
            dropped_decay.append(c["id"])
            if search_id:
                _log_stage(
                    "rerank_decay_drop",
                    search_id,
                    chunk_id=c["id"],
                    doc=c["document_name"][:50],
                    cat=cat,
                    score=c["rerank_score"],
                    cat_best=best,
                    threshold=round(0.6 * best, 4),
                )
            continue
        filtered.append(c)

    filtered.sort(key=lambda c: -c["rerank_score"])

    if search_id:
        _log_stage(
            "rerank_summary",
            search_id,
            input=len(chunks),
            after_decay=len(filtered),
            dropped_decay=len(dropped_decay),
        )
    return filtered


# ---------------------------------------------------------------------------
# Confidence calibration
# ---------------------------------------------------------------------------

_LABEL_ORDER = ["abstain", "low", "medium", "high"]


def _confidence_label(rerank_score: float) -> str:
    if rerank_score >= _CONFIDENCE_HIGH:
        return "high"
    if rerank_score >= _CONFIDENCE_MEDIUM:
        return "medium"
    if rerank_score >= _CONFIDENCE_LOW:
        return "low"
    return "abstain"


def _passes_threshold(label: str, min_label: str) -> bool:
    return _LABEL_ORDER.index(label) >= _LABEL_ORDER.index(min_label)


# ---------------------------------------------------------------------------
# Assembly  — canonical-ratio-aware slot filling
# ---------------------------------------------------------------------------
#
# Authority tiers (descending authority):
#   0 — contract_source_of_truth   (signed contract, provider manual CoT)
#   1 — payer_policy / operational  (published UM policy, clinical criteria)
#   2 — fyi_not_citable             (informational reference)
#   3 — untagged / None             (unknown provenance)
#
# "canonical ratio" = (tier-0 + tier-1) / total  — fraction of results that
# come from citable authoritative sources.
# "strict canonical ratio" = tier-0 / total      — fraction from CoT sources.

_AUTHORITY_TIER: dict[str | None, int] = {
    "contract_source_of_truth": 0,
    "payer_policy":              1,
    "operational_suggested":     1,
    "fyi_not_citable":           2,
}
_AUTHORITY_TIER_DEFAULT = 3


def _authority_tier(level: str | None) -> int:
    return _AUTHORITY_TIER.get((level or "").strip().lower(), _AUTHORITY_TIER_DEFAULT)


def _assemble(
    candidates: list[dict[str, Any]],
    k: int,
    strategy: str,
    canonical_floor: float,
    seen_pages: set[tuple[str, int | None]],
    min_label: str,
    search_id: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select k chunks from the reranked pool using *strategy*.

    Returns ``(selected_chunks, assembly_meta)`` where ``assembly_meta``
    carries tier breakdown and ratio metrics for telemetry.

    The *seen_pages* set is updated in-place so the caller's dedup state
    is shared.

    Strategies
    ----------
    score           Pure rerank_score descending.  Same as before.
    canonical_first Sort tier-0 before tier-1 before tier-2 before tier-3
                    within each confidence band (HIGH > MED > LOW > ABSTAIN),
                    then by score.  Canonical docs surface even if their raw
                    score is slightly below a lower-tier competitor.
    balanced        Fill ``ceil(k × canonical_floor)`` slots from tier-0/1
                    docs first (sorted by score), then fill remaining slots
                    from whatever is left (sorted by score).  This guarantees
                    a minimum canonical presence when available.
    """
    import math

    # Split into authoritative (tier 0+1) and other pools
    authoritative = [c for c in candidates if _authority_tier(c.get("authority_level")) <= 1]
    other         = [c for c in candidates if _authority_tier(c.get("authority_level")) >  1]

    # Sort each pool by rerank_score desc
    authoritative.sort(key=lambda c: -c.get("rerank_score", 0))
    other.sort(key=lambda c: -c.get("rerank_score", 0))

    if strategy == "canonical_first":
        # Within each confidence band, tier-0 before tier-1 before others
        def _cf_key(c: dict[str, Any]) -> tuple[int, int, float]:
            label_rank = _LABEL_ORDER.index(_confidence_label(c.get("rerank_score", 0)))
            tier       = _authority_tier(c.get("authority_level"))
            return (-label_rank, tier, -c.get("rerank_score", 0))
        ordered = sorted(candidates, key=_cf_key)

    elif strategy == "balanced":
        canonical_slots = math.ceil(k * max(0.0, min(1.0, canonical_floor)))
        other_slots     = max(0, k - canonical_slots)
        # Fill canonical slots first, then remaining from other pool
        ordered = authoritative[:canonical_slots * 3] + other[:other_slots * 3]
        # Sort each half by score so best within each tier surfaces first
        ca_part = sorted(authoritative[:canonical_slots * 3],
                         key=lambda c: -c.get("rerank_score", 0))
        ot_part = sorted(other[:other_slots * 3],
                         key=lambda c: -c.get("rerank_score", 0))
        ordered = ca_part + ot_part

    else:  # "score"
        ordered = candidates   # already rerank_score-sorted from _rerank()

    # Walk ordered list, apply confidence threshold + page dedup, fill k slots
    _THRESHOLD_PASS = lambda c: _passes_threshold(
        _confidence_label(c.get("rerank_score", 0)), min_label
    )

    # Content-sha dedupe (Fix #3 — 2026-05-02). Two filename copies of
    # the same doc (e.g. ``Provider_Manual.pdf`` and ``Sunshine
    # Provider Manual``) produce identical chunk text but distinct
    # document_ids. The page-key dedupe above wouldn't catch them. We
    # also dedupe by the first 200 chars of body text as a fallback
    # for chunks where content_sha isn't populated.
    seen_content: set[str] = set()

    def _content_key(c: dict[str, Any]) -> str:
        sha = (c.get("content_sha") or "").strip()
        if sha:
            return f"sha:{sha}"
        # Fallback — first 200 normalised chars.
        body = (c.get("text") or "").lower()
        body = " ".join(body.split())[:200]
        return f"body:{body}"

    selected: list[dict[str, Any]] = []
    promoted_extra: list[dict[str, Any]] = []
    PROMOTED_EXTRA_CAP = 10  # max additional promoted-neighbor chunks beyond k
    for c in ordered:
        # Promoted topic-block neighbors (e.g. table-row paragraphs that
        # share a page with their seed) are exempt from BOTH the
        # page-dedup AND the k-slot cap. Their whole purpose is to fill
        # in the surrounding context that lives in OTHER paragraphs of
        # the same page (cmhc001: the 180/365/90 table sits in a
        # neighbor of the BM25 winner; without these exemptions
        # synthesis never sees the table). Content-dedup still applies.
        is_promoted = (
            c.get("_promoted_from_seed") is not None
            or "bm25_inherited" in (c.get("retrieval_arms") or [])
        )
        if not _THRESHOLD_PASS(c):
            continue
        ckey = _content_key(c)
        if ckey in seen_content:
            continue
        page_key = (c["document_id"], c.get("page_number"))

        if is_promoted:
            # Promoted: doesn't count toward k, exempt from page-dedup,
            # capped separately.
            if len(promoted_extra) >= PROMOTED_EXTRA_CAP:
                continue
            seen_content.add(ckey)
            promoted_extra.append(c)
            continue

        if len(selected) >= k:
            # Once primary k is full, only promoted can still be added.
            continue
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)
        seen_content.add(ckey)
        selected.append(c)
    # Append promoted chunks AFTER the primary k so they show up as
    # additional context in the response without displacing the seed
    # citations.
    selected.extend(promoted_extra)

    # ── Compute assembly metadata ─────────────────────────────────────────
    tier_counts: dict[str, int] = {
        "contract_source_of_truth": 0,
        "payer_policy":             0,
        "fyi_not_citable":          0,
        "untagged":                 0,
    }
    for c in selected:
        level = (c.get("authority_level") or "").strip().lower()
        if level == "contract_source_of_truth":
            tier_counts["contract_source_of_truth"] += 1
        elif level in ("payer_policy", "operational_suggested"):
            tier_counts["payer_policy"] += 1
        elif level == "fyi_not_citable":
            tier_counts["fyi_not_citable"] += 1
        else:
            tier_counts["untagged"] += 1

    total = len(selected)
    canonical_count     = tier_counts["contract_source_of_truth"]
    authoritative_count = tier_counts["payer_policy"]
    canonical_ratio      = (canonical_count + authoritative_count) / total if total else 0.0
    strict_ratio         = canonical_count / total if total else 0.0

    assembly_meta: dict[str, Any] = {
        "strategy":            strategy,
        "canonical_floor":     canonical_floor if strategy == "balanced" else None,
        "canonical_ratio":     round(canonical_ratio, 3),     # CoT + payer_policy / total
        "strict_canonical_ratio": round(strict_ratio, 3),     # CoT only / total
        "tier_breakdown":      tier_counts,
        "total_selected":      total,
    }

    if search_id:
        _log_stage(
            "assembly",
            search_id,
            strategy=strategy,
            canonical_ratio=canonical_ratio,
            strict_ratio=strict_ratio,
            tier_breakdown=tier_counts,
            returned=total,
        )

    return selected, assembly_meta


# ---------------------------------------------------------------------------
# Neighborhood expansion — port of mobius-chat's doc_assembly.assemble_with_neighbors
# ---------------------------------------------------------------------------
#
# Why: a winning chunk often contains the lead-in to a fact ("see the table
# below…") but the actual values live in the next chunk. Without neighbor
# expansion the LLM gets the preamble and hallucinates the answer. With
# expansion we pull ±N paragraphs (within ±M pages) from the same document
# as supporting context. Marked ``is_neighbor=True`` so downstream can
# render them differently from primary hits.
#
# This is one batched UNNEST query for ALL seeds, not N round-trips. With
# Cloud SQL latency the per-call overhead is the dominant cost so batching
# matters.

# Caps to keep the citation list bounded even if the corpus has very long
# adjacent chunks or many seeds in a single document.
#
# Per-doc cap was bumped from 8 → 20 after observing single-doc queries
# (e.g. all hits from "Sunshine Provider Manual") where the cap was
# absorbed entirely by seeds, leaving no room for neighbors. Twenty
# chunks per doc still fits comfortably in a typical LLM prompt and
# gives the neighborhood enough oxygen to surface adjacent tables /
# subsections.
_NEIGHBOR_TOTAL_CAP   = 50    # absolute ceiling on chunks after expansion
_NEIGHBOR_PER_DOC_CAP = 20    # max chunks kept from any one document

# Big sentinel so chunks without a page_number get effectively no page
# constraint (still bounded by paragraph_index window).
_NEIGHBOR_NO_PAGE_HI = 10_000_000


async def _fetch_sibling_chunks_batch(
    db: AsyncSession,
    seeds: list[dict[str, Any]],
    *,
    paragraph_window: int = 2,
    page_window: int = 1,
) -> list[dict[str, Any]]:
    """Fetch ±paragraph_window paragraphs (within ±page_window pages) from
    the same document for each seed. Single round-trip via UNNEST. Excludes
    the seeds themselves.

    Returns a list of chunk-shaped dicts with ``is_neighbor=True`` set so
    the caller can rank/cap them distinctly from primary hits.
    """
    if not seeds:
        return []

    doc_ids: list[str] = []
    para_lo: list[int] = []
    para_hi: list[int] = []
    page_lo: list[int] = []
    page_hi: list[int] = []
    excludes: list[str] = []
    for s in seeds:
        doc_id = s.get("document_id")
        if not doc_id:
            continue
        pi = s.get("paragraph_index")
        pi_int = int(pi) if pi is not None else 0
        doc_ids.append(str(doc_id))
        para_lo.append(max(0, pi_int - paragraph_window))
        para_hi.append(pi_int + paragraph_window)
        page = s.get("page_number")
        if isinstance(page, int):
            page_lo.append(max(0, page - page_window))
            page_hi.append(page + page_window)
        else:
            page_lo.append(0)
            page_hi.append(_NEIGHBOR_NO_PAGE_HI)
        cid = s.get("id")
        excludes.append(str(cid) if cid is not None else "")

    if not doc_ids:
        return []

    # Note: ``:param::text[]`` cast syntax on bound parameters confuses
    # asyncpg's parser (it sees the colon as a placeholder boundary).
    # Use ``CAST(:param AS text[])`` instead — semantically identical
    # but SQLAlchemy handles the bind-rewrite correctly.
    sql = (
        "SELECT DISTINCT ON (m.id) "
        "       m.id::text                         AS id, "
        "       m.document_id::text                AS document_id, "
        "       m.text                             AS text, "
        "       m.page_number                      AS page_number, "
        "       m.paragraph_index                  AS paragraph_index, "
        "       m.section_path                     AS section_path, "
        "       m.chapter_path                     AS chapter_path, "
        "       m.summary                          AS summary, "
        "       m.content_sha                      AS content_sha, "
        "       m.document_display_name            AS document_display_name, "
        "       m.document_filename                AS document_filename, "
        "       m.document_authority_level         AS document_authority_level, "
        "       m.document_payer                   AS document_payer, "
        "       m.document_state                   AS document_state "
        "FROM rag_published_embeddings m "
        "JOIN ( "
        "   SELECT UNNEST(CAST(:doc_ids  AS text[])) AS doc_id, "
        "          UNNEST(CAST(:para_lo  AS int[]))  AS lo, "
        "          UNNEST(CAST(:para_hi  AS int[]))  AS hi, "
        "          UNNEST(CAST(:page_lo  AS int[]))  AS plo, "
        "          UNNEST(CAST(:page_hi  AS int[]))  AS phi, "
        "          UNNEST(CAST(:excludes AS text[])) AS exclude_id "
        ") r "
        # Cast r.doc_id → uuid so Postgres can use the (document_id, paragraph_index,
        # page_number) btree index. Casting m.document_id::text instead would force
        # a full sequential scan on 882K rows per seed chunk (15s baseline).
        "  ON m.document_id          = r.doc_id::uuid "
        " AND m.paragraph_index       BETWEEN r.lo  AND r.hi "
        " AND m.page_number           BETWEEN r.plo AND r.phi "
        " AND m.id::text              <> COALESCE(NULLIF(r.exclude_id, ''), "
        "                                          '00000000-0000-0000-0000-000000000000') "
        "ORDER BY m.id, m.page_number, m.paragraph_index "
        "LIMIT 500"
    )

    try:
        result = await db.execute(text(sql), {
            "doc_ids":  doc_ids,
            "para_lo":  para_lo,
            "para_hi":  para_hi,
            "page_lo":  page_lo,
            "page_hi":  page_hi,
            "excludes": excludes,
        })
        rows = result.mappings().all()
    except Exception as exc:
        logger.warning("neighbor fetch failed: %s", exc)
        return []

    out: list[dict[str, Any]] = []
    for row in rows:
        out.append({
            "id":                 row["id"],
            "document_id":        row["document_id"],
            "text":               row.get("text") or "",
            "page_number":        row.get("page_number"),
            "paragraph_index":    row.get("paragraph_index"),
            "section_path":       _none_if_empty(row.get("section_path")),
            "chapter_path":       _none_if_empty(row.get("chapter_path")),
            "summary":            _none_if_empty(row.get("summary")),
            "content_sha":        _none_if_empty(row.get("content_sha")),
            "document_name":     (row.get("document_display_name")
                                  or row.get("document_filename")
                                  or "document"),
            "source_type":        "hierarchical",
            "similarity":         0.0,    # neighbors don't have an arm score
            # Inherit a fraction of seed's score later; default 0 here.
            "rerank_score":       0.0,
            "confidence_label":   "low",
            "retrieval_arms":     ["neighbor"],
            "authority_level":    _none_if_empty(row.get("document_authority_level")),
            "payer":              _none_if_empty(row.get("document_payer")),
            "state":              _none_if_empty(row.get("document_state")),
            "jpd_tags":           [],
            "is_neighbor":        True,
        })
    return out


def _apply_neighbor_caps(
    seeds: list[dict[str, Any]],
    neighbors: list[dict[str, Any]],
    *,
    total_cap: int = _NEIGHBOR_TOTAL_CAP,
    per_doc_cap: int = _NEIGHBOR_PER_DOC_CAP,
) -> list[dict[str, Any]]:
    """Combine seeds and neighbors with caps. Seeds always come first
    (preserving their order); neighbors are appended sorted by
    inherited score desc, capped per-document and overall.
    """
    if not seeds and not neighbors:
        return []

    def _score(c: dict[str, Any]) -> float:
        try:
            return float(c.get("rerank_score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    nbrs = sorted(neighbors, key=_score, reverse=True)
    per_doc: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    # Seed pass — preserve original order, count toward per-doc cap.
    for c in seeds:
        if len(out) >= total_cap:
            break
        doc_key = str(c.get("document_id") or c.get("document_name") or "_unknown")
        per_doc[doc_key] = per_doc.get(doc_key, 0) + 1
        out.append(c)
    # Neighbor pass.
    for c in nbrs:
        if len(out) >= total_cap:
            break
        doc_key = str(c.get("document_id") or c.get("document_name") or "_unknown")
        if per_doc.get(doc_key, 0) >= per_doc_cap:
            continue
        per_doc[doc_key] = per_doc.get(doc_key, 0) + 1
        out.append(c)
    return out


async def _attach_inherited_doc_tags(
    db: AsyncSession,
    candidates: list[dict[str, Any]],
) -> None:
    """One batched lookup of d/j/p tags from ``document_tags`` for every
    candidate's parent doc, attached as ``_doc_d_tags`` / ``_doc_j_tags``
    / ``_doc_p_tags`` lists on each candidate dict.

    Why: the reranker's ``meta_boost`` signal needs to know what
    topical tags the parent document carries, so a chunk in (e.g.)
    a ``d:utilization_management.prior_authorization``-tagged doc can
    get credit for the "prior authorization" concept even when its
    body doesn't repeat the words. Without this query each chunk's
    meta_haystack only sees filename + payer + section_path — fine for
    the doc title but blind to the topical d-tags assigned at ingest.
    """
    if not candidates:
        return
    doc_ids = list({str(c.get("document_id") or "") for c in candidates if c.get("document_id")})
    if not doc_ids:
        return
    try:
        rows = await db.execute(
            text(
                "SELECT document_id::text AS doc_id, d_tags, j_tags, p_tags "
                "FROM document_tags "
                "WHERE document_id::text = ANY(CAST(:ids AS text[]))"
            ),
            {"ids": doc_ids},
        )
        tag_rows = rows.mappings().all()
    except Exception as exc:
        logger.debug("inherited tag fetch failed: %s", exc)
        return

    by_doc: dict[str, dict[str, Any]] = {}
    for r in tag_rows:
        by_doc[r["doc_id"]] = {
            "d_tags": _coerce_jsonb_to_list(r.get("d_tags")),
            "j_tags": _coerce_jsonb_to_list(r.get("j_tags")),
            "p_tags": _coerce_jsonb_to_list(r.get("p_tags")),
        }
    for c in candidates:
        info = by_doc.get(str(c.get("document_id") or ""))
        if info:
            c["_doc_d_tags"] = info["d_tags"]
            c["_doc_j_tags"] = info["j_tags"]
            c["_doc_p_tags"] = info["p_tags"]


def _coerce_jsonb_to_list(v: Any) -> list[str]:
    """document_tags.{d,j,p}_tags is JSONB. May be a list of strings,
    or an object whose keys are tag codes (legacy shape). Normalise
    to a flat list of dotted code strings."""
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if x]
    if isinstance(v, dict):
        return [str(k) for k in v.keys() if k]
    return []


async def _enrich_candidates_with_neighbor_text(
    db: AsyncSession,
    candidates: list[dict[str, Any]],
    *,
    paragraph_window: int = 1,
    page_window: int = 1,
    max_neighbor_chars: int = 1500,
    search_id: str = "",
) -> None:
    """Mutate each candidate dict in-place by attaching a
    ``_neighbor_text`` string built from the bodies of its ±N
    paragraph siblings (within ±M pages of the same document).

    This runs BEFORE rerank so the body-haystack signals
    (``tag_coverage``, ``jpd``) see the chunk's surrounding
    context — section headers, table labels, the paragraph above
    that explains what the chunk's data row is about. Without this
    a chunk with a sparse body (e.g. just a row of HCPCS codes)
    looks "off-topic" to the reranker even when its parent
    document is exactly on-topic.

    One batched DB call for ALL candidates. Cap each candidate's
    neighbor text at ``max_neighbor_chars`` so the rerank stays
    cheap (substring scans).
    """
    if not candidates or paragraph_window <= 0:
        return

    raw = await _fetch_sibling_chunks_batch(
        db, candidates,
        paragraph_window=paragraph_window,
        page_window=page_window,
    )
    if not raw:
        if search_id:
            _log_stage(
                "rerank_neighbor_enrich",
                search_id,
                candidates=len(candidates),
                fetched=0,
                kept=0,
            )
        return

    # Group neighbors by (document_id, page) → list of bodies, keyed
    # so we can find each candidate's siblings quickly.
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for n in raw:
        by_doc.setdefault(str(n.get("document_id") or ""), []).append(n)

    enriched_count = 0
    for c in candidates:
        doc_id = str(c.get("document_id") or "")
        page = c.get("page_number")
        para = c.get("paragraph_index")
        if not doc_id or page is None or para is None:
            continue
        siblings = by_doc.get(doc_id) or []
        # Pick siblings within the requested windows. We don't dedupe
        # against the candidate itself (sibling fetch already excluded
        # the seed by id) but we still skip exact-text duplicates so
        # form-letter pages that share text don't pad the haystack.
        seen_text: set[str] = set()
        bodies: list[str] = []
        for n in siblings:
            np = n.get("page_number")
            ni = n.get("paragraph_index")
            if np is None or ni is None:
                continue
            if abs(int(ni) - int(para)) > paragraph_window:
                continue
            if abs(int(np) - int(page)) > page_window:
                continue
            text = (n.get("text") or "").strip()
            if not text:
                continue
            key = " ".join(text.lower().split())[:200]
            if key in seen_text:
                continue
            seen_text.add(key)
            bodies.append(text)
            if sum(len(b) for b in bodies) >= max_neighbor_chars:
                break
        if bodies:
            joined = " || ".join(bodies)[:max_neighbor_chars]
            c["_neighbor_text"] = joined
            enriched_count += 1

    if search_id:
        _log_stage(
            "rerank_neighbor_enrich",
            search_id,
            candidates=len(candidates),
            fetched=len(raw),
            enriched=enriched_count,
        )


async def _promote_high_sim_neighbors_to_candidates(
    db: AsyncSession,
    candidates: list[dict[str, Any]],
    *,
    sim_threshold: float = 0.7,
    paragraph_window: int = 3,
    page_window: int = 0,
    max_per_seed: int = 5,
    inherit_decay: float = 0.7,   # unused now (kept for signature compat)
    search_id: str = "",
) -> int:
    """Merge same-page same-topic-block sibling text INTO each high-sim
    seed's ``text`` field, so the synthesis LLM sees one rich passage
    instead of fragments scattered across separate citations.

    Why this exists: BM25 retrieves chunks whose body lexically matches
    the query. But the answer often lives in a sibling paragraph
    WITHIN THE SAME TOPIC BLOCK whose body has none of the query's
    words — the canonical case is a table where the column headers
    (matching the query) are in one chunk and the answer numbers (180
    days, 365 days) are in a sibling chunk.

    Earlier iterations of this function added the siblings as separate
    rerank candidates. That worked at the retrieval layer (the table
    chunk reached top-K) but FAILED at synthesis: the LLM saw the
    table as a separate ``[citation]`` from its intro and didn't
    piece them together. Per user feedback ("these are context files,
    should be treated as a single k"), we now MERGE the topic-block
    siblings' text into the seed's text field with a ``[…context…]``
    separator, in document order. One citation, one k slot, one rich
    passage that lets the LLM follow the table from intro to footer.

    cmhc001 (timely filing): seed p.121 pi.4 ("Electronic Claim
    Submission") gets pi.1 ("Providers must submit claims in a timely
    manner…"), pi.2 ("Initial Claim*"), pi.3 (180/365/90 table)
    appended in order. Synthesis sees one chunk that flows: intro →
    column headers → row labels → numbers → footnote.

    Returns the number of seeds whose text was extended.
    """
    if not candidates:
        return 0
    # Pick seeds whose own BM25 score is high enough to trust as a topic
    # anchor. Merging context for low-sim seeds wastes LLM tokens.
    seeds: list[dict[str, Any]] = []
    for c in candidates:
        sc = (c.get("arm_scores") or {}).get("bm25", 0.0)
        if not sc:
            sc = float(c.get("similarity") or 0.0)
        if sc >= sim_threshold:
            seeds.append(c)
    if not seeds:
        return 0

    raw = await _fetch_sibling_chunks_batch(
        db, seeds,
        paragraph_window=paragraph_window,
        page_window=page_window,
    )
    if not raw:
        return 0

    # Group siblings by (doc_id, page_number); we only merge chunks that
    # share the seed's page (page_window=0 default).
    # Dedup against:
    #   1. Same paragraph_index seen multiple times (multi-ingest corpus)
    #   2. Page-header chunks (just the page number / SH_xxxx / Manual title
    #      — pure noise that bloats the prompt)
    import re as _re
    _HEADER_PAT = _re.compile(r"^\s*\d{1,4}\s*(SH_\d+)?\s*(?:©\s*\d{4})?", _re.IGNORECASE)
    def _is_page_header(text: str) -> bool:
        t = (text or "").strip()
        if len(t) < 80 and ("Provider Manual" in t or "SH_" in t):
            return True
        return False

    by_doc_page: dict[tuple[str, int], list[dict[str, Any]]] = {}
    seen_pi: dict[tuple[str, int], set[int]] = {}
    for n in raw:
        ndoc = str(n.get("document_id") or "")
        npage = n.get("page_number")
        npi   = n.get("paragraph_index")
        if not ndoc or npage is None or npi is None:
            continue
        key = (ndoc, int(npage))
        npi_int = int(npi)
        # Dedup by paragraph_index — multi-ingest corpus has the same
        # (doc, page, pi) repeated 2-3× with different chunk ids.
        seen = seen_pi.setdefault(key, set())
        if npi_int in seen:
            continue
        if _is_page_header(n.get("text") or ""):
            continue
        seen.add(npi_int)
        by_doc_page.setdefault(key, []).append(n)

    extended = 0
    for s in seeds:
        sdoc = str(s.get("document_id") or "")
        spage = s.get("page_number")
        spara = s.get("paragraph_index") or 0
        if not sdoc or spage is None:
            continue
        siblings = by_doc_page.get((sdoc, int(spage))) or []
        if not siblings:
            continue
        # Sort siblings by paragraph_index — preserve document reading order.
        siblings = sorted(siblings, key=lambda n: int(n.get("paragraph_index") or 0))
        # Pick up to max_per_seed siblings within paragraph_window of seed.
        picked: list[dict[str, Any]] = []
        for n in siblings:
            npi = int(n.get("paragraph_index") or 0)
            if abs(npi - spara) > paragraph_window:
                continue
            ntext = (n.get("text") or "").strip()
            if not ntext:
                continue
            picked.append(n)
            if len(picked) >= max_per_seed:
                break
        if not picked:
            continue

        # Build the merged text in document order: prepend siblings whose
        # paragraph_index < seed.paragraph_index, append those after.
        before = [p for p in picked if int(p.get("paragraph_index") or 0) < spara]
        after  = [p for p in picked if int(p.get("paragraph_index") or 0) > spara]
        seed_text = (s.get("text") or "").strip()
        merged_parts: list[str] = []
        for p in before:
            merged_parts.append((p.get("text") or "").strip())
        merged_parts.append(seed_text)
        for p in after:
            merged_parts.append((p.get("text") or "").strip())
        # Drop empties and join with a visible separator so the LLM can see
        # this is one block but composed of contiguous paragraphs.
        merged_text = "\n\n".join(p for p in merged_parts if p)
        s["text"] = merged_text
        s["_topic_block_merged"] = {
            "n_siblings": len(picked),
            "before": len(before),
            "after":  len(after),
        }
        extended += 1

    if search_id:
        _log_stage(
            "promote_neighbors",
            search_id,
            seeds=len(seeds),
            siblings_fetched=len(raw),
            extended=extended,
            sim_threshold=sim_threshold,
            mode="merge_into_seed_text",
        )
    return extended


async def _expand_with_neighbors(
    db: AsyncSession,
    seeds: list[dict[str, Any]],
    *,
    paragraph_window: int = 2,
    page_window: int = 1,
    search_id: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch siblings, dedupe vs seeds, inherit a fractional score from the
    nearest seed in the same doc, apply caps. Returns (chunks, meta).

    Returns the original seeds passed in if expansion is disabled by
    callers (zero windows) or the DB call fails.
    """
    if not seeds or paragraph_window <= 0:
        return seeds, {"requested": False, "fetched": 0, "kept": 0}

    raw = await _fetch_sibling_chunks_batch(
        db, seeds,
        paragraph_window=paragraph_window,
        page_window=page_window,
    )

    # Dedupe vs seeds AND content-wise. Two failure modes to fight:
    #   1. The same chunk id appears in seeds (rare; SQL ``exclude_id``
    #      should already prevent this but be defensive).
    #   2. The corpus has multiple ingests of the same document, each
    #      with distinct chunk ids but identical content_sha / body
    #      text. Without content-level dedup the LLM sees the same
    #      paragraph 3-5 times.
    seed_ids = {str(s.get("id")) for s in seeds if s.get("id")}

    seen_content: set[str] = set()
    # Pre-seed with seed content so neighbors don't duplicate seed bodies.
    for s in seeds:
        sha = (s.get("content_sha") or "").strip()
        if sha:
            seen_content.add(f"sha:{sha}")
        else:
            body = " ".join((s.get("text") or "").lower().split())[:200]
            if body:
                seen_content.add(f"body:{body}")

    fresh: list[dict[str, Any]] = []
    for n in raw:
        if str(n.get("id")) in seed_ids:
            continue
        sha = (n.get("content_sha") or "").strip()
        if sha:
            ckey = f"sha:{sha}"
        else:
            body = " ".join((n.get("text") or "").lower().split())[:200]
            ckey = f"body:{body}"
        if not ckey or ckey == "body:":
            continue
        if ckey in seen_content:
            continue
        seen_content.add(ckey)
        fresh.append(n)

    # Inherit a fractional score from the closest seed in the SAME document.
    # We use 0.5 × seed.rerank_score to keep neighbors below their seeds at
    # rerank-sort time — they're context, not primary citations.
    seed_score_by_doc: dict[str, float] = {}
    for s in seeds:
        doc = str(s.get("document_id") or "")
        sc = float(s.get("rerank_score") or 0.0)
        if doc:
            seed_score_by_doc[doc] = max(seed_score_by_doc.get(doc, 0.0), sc)
    for n in fresh:
        doc = str(n.get("document_id") or "")
        n["rerank_score"] = 0.5 * seed_score_by_doc.get(doc, 0.0)
        n["confidence_label"] = "low"

    combined = _apply_neighbor_caps(seeds, fresh)

    if search_id:
        _log_stage(
            "neighbor_expand",
            search_id,
            paragraph_window=paragraph_window,
            page_window=page_window,
            seeds=len(seeds),
            fetched=len(raw),
            after_dedupe=len(fresh),
            combined=len(combined),
        )

    return combined, {
        "requested":     True,
        "fetched":       len(raw),
        "kept":          len(combined) - len(seeds),
        "para_window":   paragraph_window,
        "page_window":   page_window,
    }


# ---------------------------------------------------------------------------
# Persistence helper
# ---------------------------------------------------------------------------

async def _persist_search_event(
    telemetry: dict[str, Any],
    caller: str = "api",
    caller_id: str | None = None,
) -> None:
    """Fire-and-forget: write search telemetry to search_events.

    Persists both the legacy timing/trace JSONB columns and the Phase B
    lexicon-aware columns (matched_codes, expansion_phrases, final_tsquery,
    bm25_hits, vector_hits, total_chunks, filters, domain/jurisdiction/
    process tags) consumed by the ``/admin/search_events`` feed.

    Opens its OWN session because the caller's session is being torn down
    by the FastAPI dependency the moment ``corpus_search`` returns —
    sharing it produces ``cannot perform operation: another operation is
    in progress`` on asyncpg.

    Errors are swallowed — persistence must never block or fail a search.
    """
    from app.database import AsyncSessionLocal
    try:
        import json as _json
        bm25_expansion: dict[str, Any] = telemetry.get("bm25_expansion") or {}
        arm_hits: dict[str, Any] = telemetry.get("arm_hits") or {}
        raw_query: str = telemetry.get("query", "") or ""
        normalized_query: str | None = telemetry.get("bm25_normalized_query")
        filters_obj = telemetry.get("filters")

        async with AsyncSessionLocal() as db:
            await db.execute(
                text("""
                    INSERT INTO search_events
                        (search_id, caller, caller_id,
                         query, raw_query, bm25_normalized_query, normalized_query,
                         mode, k, returned, total_chunks,
                         total_ms, embed_ms, bm25_ms, vec_ms, rerank_ms,
                         arm_hits, arm_results, scoring_trace, assembly,
                         matched_codes, expansion_phrases, final_tsquery,
                         bm25_hits, vector_hits,
                         filters, domain_tags, jurisdiction_tags, process_tags)
                    VALUES
                        (:search_id, :caller, :caller_id,
                         :query, :raw_query, :bm25_normalized_query, :normalized_query,
                         :mode, :k, :returned, :total_chunks,
                         :total_ms, :embed_ms, :bm25_ms, :vec_ms, :rerank_ms,
                         CAST(:arm_hits AS jsonb), CAST(:arm_results AS jsonb),
                         CAST(:scoring_trace AS jsonb), CAST(:assembly AS jsonb),
                         :matched_codes, :expansion_phrases, :final_tsquery,
                         :bm25_hits, :vector_hits,
                         CAST(:filters AS jsonb),
                         :domain_tags, :jurisdiction_tags, :process_tags)
                """),
                {
                    "search_id":             telemetry.get("search_id", ""),
                    "caller":                caller,
                    "caller_id":             caller_id,
                    "query":                 raw_query,
                    "raw_query":             raw_query,
                    "bm25_normalized_query": normalized_query,
                    "normalized_query":      normalized_query,
                    "mode":                  telemetry.get("mode", "corpus"),
                    "k":                     telemetry.get("k", 10),
                    "returned":              telemetry.get("returned", 0),
                    "total_chunks":          telemetry.get("returned", 0),
                    "total_ms":              telemetry.get("total_ms"),
                    "embed_ms":              telemetry.get("embed_ms"),
                    "bm25_ms":               telemetry.get("bm25_ms"),
                    "vec_ms":                telemetry.get("vec_ms"),
                    "rerank_ms":             telemetry.get("rerank_ms"),
                    "arm_hits":              _json.dumps(arm_hits),
                    "arm_results":           _json.dumps(telemetry.get("arm_results")),
                    "scoring_trace":         _json.dumps(telemetry.get("scoring_trace")),
                    "assembly":              _json.dumps(telemetry.get("assembly")),
                    "matched_codes":         list(bm25_expansion.get("matched_codes") or []),
                    "expansion_phrases":     list(bm25_expansion.get("expansion_phrases") or []),
                    "final_tsquery":         bm25_expansion.get("final_tsquery"),
                    "bm25_hits":             int(arm_hits.get("bm25") or 0),
                    "vector_hits":           int(arm_hits.get("vector") or 0),
                    "filters":               _json.dumps(filters_obj) if filters_obj else None,
                    "domain_tags":           list(bm25_expansion.get("domain_tags") or []),
                    "jurisdiction_tags":     list(bm25_expansion.get("jurisdiction_tags") or []),
                    "process_tags":          list(bm25_expansion.get("process_tags") or []),
                },
            )
            await db.commit()
    except Exception as exc:
        logger.warning("search_events write failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def corpus_search(
    db: AsyncSession,
    request: CorpusSearchRequest,
    caller: str = "api",
    caller_id: str | None = None,
) -> CorpusSearchResponse:
    """Run BM25 / vector / hybrid search and return ranked, labelled chunks."""
    search_id = uuid.uuid4().hex[:12]   # short ID correlates all log lines for this call

    if not (request.query or "").strip():
        return CorpusSearchResponse(
            chunks=[],
            telemetry={"mode": request.mode, "k": request.k, "error": "empty query",
                       "search_id": search_id},
        )

    mode = request.mode or "corpus"
    k    = max(1, min(100, request.k))
    t0   = time.monotonic()

    # ── 0. Log query intake ───────────────────────────────────────────────
    _log_stage(
        "query_intake",
        search_id,
        query=request.query[:120],
        mode=mode,
        k=k,
        filters=request.filters.model_dump() if request.filters else None,
        doc_ids=len(request.include_document_ids or []),
        min_sim=request.min_similarity,
    )

    # ── 1 + 2. Retrieve arms — strategy depends on mode ──────────────────
    #
    # corpus mode:    BM25 (DB) + embed (OpenAI API) run in PARALLEL via
    #                 asyncio.create_task, then vector arm runs after embed.
    #                 BM25 and embed share no resources so concurrent is safe.
    #                 Saves ~bm25_ms off the critical path (typically 200-400 ms).
    #                 Embedding is LRU-cached so repeat queries pay 0 ms.
    #
    # precision mode: BM25 only — no embed needed.
    # recall mode:    embed → vector only — no BM25.
    #
    query_embedding: list[float] | None = None
    bm25_chunks: list[dict[str, Any]] = []
    vec_chunks:  list[dict[str, Any]] = []
    embed_ms = bm25_ms = vec_ms = 0.0
    bm25_normalized_query: str | None = None   # set when normalizer changed the query
    bm25_expansion: dict[str, Any] = {
        "matched_codes":           [],
        "expansion_phrases":       [],
        "expansion_phrases_count": 0,
        "final_tsquery":           "",
        "log":                     [],
        "domain_tags":             [],
        "jurisdiction_tags":       [],
        "process_tags":            [],
    }

    if mode == "corpus":
        # Schedule BM25 as a background task so it overlaps with the embed call.
        # asyncio tasks interleave at await-points; DB and OpenAI use separate I/O.
        tb = time.monotonic()
        bm25_task: asyncio.Task[
            tuple[list[dict[str, Any]], str | None, dict[str, Any]]
        ] = asyncio.create_task(
            _bm25_arm(
                db, request.query, k * 2,
                request.filters, request.include_document_ids,
                search_id=search_id,
                tag_mode=request.tag_mode,
            )
        )
        # Embed runs in foreground while BM25 is in flight.
        query_embedding, embed_ms, _cache_hit = await _embed_with_cache(
            request.query, search_id
        )
        # BM25 is usually done by now; await completes immediately if so.
        bm25_chunks, bm25_normalized_query, bm25_expansion = await bm25_task
        bm25_ms = (time.monotonic() - tb) * 1000 - embed_ms  # net DB time

        if query_embedding:
            # Reuse the lexicon expansion from BM25 so vector arm applies
            # the same metadata-J filter (no second lexicon call).
            from app.services.corpus_search_lexicon import LexiconExpansion as _LE
            _exp_for_vec = _LE(
                matched_codes=bm25_expansion.get("matched_codes") or [],
                expansion_phrases=bm25_expansion.get("expansion_phrases") or [],
                domain_tags=bm25_expansion.get("domain_tags") or [],
                jurisdiction_tags=bm25_expansion.get("jurisdiction_tags") or [],
                process_tags=bm25_expansion.get("process_tags") or [],
                log=[],
            )
            tv = time.monotonic()
            vec_chunks = await _vector_arm(
                db, query_embedding, k * 2,
                request.filters, request.include_document_ids,
                search_id=search_id,
                expansion=_exp_for_vec,
                tag_mode=request.tag_mode,
            )
            vec_ms = (time.monotonic() - tv) * 1000

    elif mode == "precision":
        tb = time.monotonic()
        bm25_chunks, bm25_normalized_query, bm25_expansion = await _bm25_arm(
            db, request.query, k * 2,
            request.filters, request.include_document_ids,
            search_id=search_id,
            tag_mode=request.tag_mode,
        )
        bm25_ms = (time.monotonic() - tb) * 1000

    else:  # recall
        query_embedding, embed_ms, _cache_hit = await _embed_with_cache(
            request.query, search_id
        )
        if query_embedding:
            # Recall mode: still classify the query via lexicon so we
            # can apply the metadata-J filter to the vector arm.
            from app.services.corpus_search_lexicon import (
                expand_query_via_lexicon as _expand,
            )
            try:
                _exp_for_vec = await _expand(db, request.query)
            except Exception:
                _exp_for_vec = None
            tv = time.monotonic()
            # Recall mode: instead of a flat k * N candidate count, fetch
            # MORE candidates from SQL (over_fetch_factor=8) and post-
            # filter by a similarity threshold. Many corpora contain
            # large clusters of duplicate-text chunks (multi-page form
            # PDFs with repeating headers) that all tie at one specific
            # similarity score and crowd the HNSW result list. With a
            # threshold, those tied boilerplate chunks are dropped if
            # they're below the floor, and the higher-similarity unique
            # content underneath the cluster gets reached. The rerank
            # below may dramatically shift scores, so we want to bring
            # forward EVERYTHING that has a reasonable initial sim, not
            # just the first k that happens to come out of HNSW.
            vec_chunks = await _vector_arm(
                db, query_embedding, k * 2,
                request.filters, request.include_document_ids,
                search_id=search_id,
                expansion=_exp_for_vec,
                tag_mode=request.tag_mode,
                min_similarity=request.min_similarity,
                over_fetch_factor=8,
            )
            vec_ms = (time.monotonic() - tv) * 1000

    # ── 3. Fuse ───────────────────────────────────────────────────────────
    tr = time.monotonic()
    if mode == "corpus":
        candidates = _rrf_merge(
            {"bm25": bm25_chunks, "vector": vec_chunks},
            search_id=search_id,
        )
    elif mode == "precision":
        for c in bm25_chunks:
            c.setdefault("retrieval_arms", ["bm25"])
        candidates = bm25_chunks
        _log_stage("fusion_bypass", search_id, reason="precision→bm25_only",
                   candidates=len(candidates))
    else:  # recall
        for c in vec_chunks:
            c.setdefault("retrieval_arms", ["vector"])
        candidates = vec_chunks
        _log_stage("fusion_bypass", search_id, reason="recall→vector_only",
                   candidates=len(candidates))

    # ── 3b. Defensive content dedup ───────────────────────────────────────
    # Multi-page form-letter PDFs (DSH/LIP quarterly reports etc.) put
    # the same agency-header text on every page. The chunker stores
    # each one separately, so we end up with hundreds of identical-text
    # chunks that share an embedding (Vertex returns the same vector
    # for the same input). At query time these all tie at the same
    # similarity score, crowd the HNSW result list, and push the
    # actually-relevant higher-similarity content chunks off the top-N.
    #
    # NOTE: ``content_sha`` is NOT a hash of the chunk body — it's a
    # per-chunk identifier (different value for every chunk even when
    # the text is identical). So dedup MUST be done on the actual text.
    # We hash a normalised version of the body so case / whitespace
    # variants collapse together. Empty bodies are kept (degenerate
    # case; let rerank decide).
    pre_dedup = len(candidates)
    if candidates:
        seen_keys: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for c in candidates:
            body = " ".join((c.get("text") or "").lower().split())
            if not body:
                deduped.append(c)
                continue
            # Use first 400 chars of the normalised body as the dedup
            # key. Long enough to distinguish real content variants;
            # short enough that minor end-of-chunk differences (page
            # numbers, footers) don't break dedup of header chunks.
            key = body[:400]
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(c)
        candidates = deduped
    if pre_dedup != len(candidates):
        _log_stage(
            "content_dedup",
            search_id,
            pre=pre_dedup,
            post=len(candidates),
            dropped=pre_dedup - len(candidates),
        )

    # ── 3c. Pre-rerank enrichment ─────────────────────────────────────
    # Two batched lookups before rerank:
    #   1. Sibling paragraphs (±1) → body-haystack expansion so
    #      tag_coverage / jpd see the chunk's surrounding context
    #      (section header, table label, paragraph above/below).
    #   2. Inherited d/j/p tags from document_tags → meta_haystack
    #      expansion so a chunk in a topically-tagged doc gets credit
    #      for those concepts even if its body doesn't repeat them.
    if request.required_phrases:
        await _enrich_candidates_with_neighbor_text(
            db, candidates,
            paragraph_window=1,
            page_window=1,
            search_id=search_id,
        )
        await _attach_inherited_doc_tags(db, candidates)
        # Promote topic-block neighbors of high-sim seeds to first-class
        # rerank candidates with inherited BM25 sim. Fixes the "table
        # answer is in a sibling chunk that BM25 missed" failure mode
        # (e.g. cmhc001 timely-filing 180-day table). Only triggers for
        # seeds with sim ≥ 0.7 to avoid polluting the candidate set
        # with neighbors of weak BM25 hits.
        await _promote_high_sim_neighbors_to_candidates(
            db, candidates,
            sim_threshold=0.7,
            paragraph_window=3,
            page_window=0,    # same page only — keeps the topic block tight
            max_per_seed=5,
            inherit_decay=0.7,
            search_id=search_id,
        )

    # ── 4. Rerank ─────────────────────────────────────────────────────────
    reranked = _rerank(
        candidates,
        search_id=search_id,
        query=request.query,
        required_phrases=request.required_phrases,
        required_phrase_weights=request.required_phrase_weights,
        required_phrase_tag_codes=request.required_phrase_tag_codes,
    )
    rerank_ms = (time.monotonic() - tr) * 1000

    # ── 5. Assemble — canonical-ratio-aware slot filling ─────────────────
    min_label = _MODE_MIN.get(mode, "low")
    if request.min_similarity is not None:
        min_label = _confidence_label(request.min_similarity)

    seen_pages: set[tuple[str, int | None]] = set()
    assembled, assembly_meta = _assemble(
        candidates       = reranked[:k * 3],    # give assemble a generous pool
        k                = k,
        strategy         = request.assembly_strategy or "score",
        canonical_floor  = max(0.0, min(1.0, request.canonical_floor)),
        seen_pages       = seen_pages,
        min_label        = min_label,
        search_id        = search_id,
    )

    # ── 5b. Neighborhood expansion ────────────────────────────────────────
    # Pull ±N paragraphs from the same document for each assembled hit so
    # the LLM sees the surrounding context (the table after "see below…",
    # the next paragraph that completes the answer, etc.). This is the
    # ported sibling-fetch logic from mobius-chat/doc_assembly. Set
    # ``neighbor_paragraph_window=0`` on the request to disable.
    if (
        request.neighbor_paragraph_window
        and request.neighbor_paragraph_window > 0
        and assembled
    ):
        assembled, neighbor_meta = await _expand_with_neighbors(
            db, assembled,
            paragraph_window=request.neighbor_paragraph_window,
            page_window=max(0, int(request.neighbor_page_window)),
            search_id=search_id,
        )
        assembly_meta["neighbor_expansion"] = neighbor_meta
    else:
        assembly_meta["neighbor_expansion"] = {"requested": False}

    # ── 6. Build output + scoring trace ──────────────────────────────────
    chunks_out: list[CorpusChunk] = []
    scoring_trace: list[dict[str, Any]] = []

    for rank, c in enumerate(assembled, 1):
        score = float(c.get("rerank_score") or 0.0)
        label = _confidence_label(score)

        chunk = CorpusChunk(
            id=c["id"],
            text=c["text"],
            document_id=c["document_id"],
            document_name=c["document_name"],
            page_number=c.get("page_number"),
            paragraph_index=c.get("paragraph_index"),
            source_type=c.get("source_type") or "hierarchical",
            similarity=round(float(c.get("similarity") or 0.0), 4),
            rerank_score=round(score, 4),
            confidence_label=label,
            retrieval_arms=c.get("retrieval_arms") or ["unknown"],
            authority_level=c.get("authority_level"),
            payer=c.get("payer"),
            state=c.get("state"),
            jpd_tags=c.get("_jpd_tags") or [],
            section_path=c.get("section_path"),
            chapter_path=c.get("chapter_path"),
            summary=c.get("summary"),
            is_neighbor=bool(c.get("is_neighbor")),
        )
        chunks_out.append(chunk)

        # Scoring trace for this chunk
        trace_entry: dict[str, Any] = {
            "rank":             rank,
            "chunk_id":         chunk.id,
            "document_name":    chunk.document_name,
            "document_id":      chunk.document_id,
            "page_number":      chunk.page_number,
            "paragraph_index":  chunk.paragraph_index,
            "retrieval_arms":   chunk.retrieval_arms,
            "authority_level":  chunk.authority_level,
            "authority_tier":   _authority_tier(chunk.authority_level),
            "confidence_label": label,
            "text_preview":     _preview(chunk.text),
        }
        if "arm_scores" in c:
            trace_entry["arm_scores"] = {
                a: round(s, 4) for a, s in c["arm_scores"].items()
            }
            trace_entry["arm_ranks"] = c.get("arm_ranks")
            trace_entry["rrf_score"] = round(
                float(c.get("rrf_score") or c.get("similarity") or 0.0), 4
            )
        else:
            arm = (c.get("retrieval_arms") or ["unknown"])[0]
            trace_entry["arm_scores"] = {arm: round(float(c.get("similarity") or 0.0), 4)}
        if "_rerank_signals" in c:
            trace_entry["rerank_signals"] = c["_rerank_signals"]
        scoring_trace.append(trace_entry)

        _log_stage(
            "result",
            search_id,
            rank=rank,
            chunk_id=chunk.id,
            doc=chunk.document_name[:50],
            page=chunk.page_number,
            arms="+".join(chunk.retrieval_arms),
            rerank_score=score,
            confidence_label=label,
            authority=chunk.authority_level,
            authority_tier=_authority_tier(chunk.authority_level),
            payer=chunk.payer,
        )

    total_ms = round((time.monotonic() - t0) * 1000, 1)

    _log_stage(
        "summary",
        search_id,
        mode=mode,
        k=k,
        bm25_hits=len(bm25_chunks),
        vector_hits=len(vec_chunks),
        candidates=len(reranked),
        returned=len(chunks_out),
        canonical_ratio=assembly_meta["canonical_ratio"],
        strict_canonical_ratio=assembly_meta["strict_canonical_ratio"],
        embed_ms=embed_ms,
        bm25_ms=bm25_ms,
        vec_ms=vec_ms,
        rerank_ms=rerank_ms,
        total_ms=total_ms,
    )

    # ── Raw arm results for frontend trace display ───────────────────────
    # Expose the per-arm hits BEFORE fusion so the UI can render a full
    # pipeline trace: what BM25 found, what vector found, how they fused.
    # Text is omitted here (available on the returned chunks); this is
    # metadata only — doc name, page, raw score, authority.
    def _arm_summary(chunks: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id":       c["id"],
                "document_name":  c["document_name"][:60],
                "document_id":    c["document_id"],
                "page_number":    c["page_number"],
                "authority_level": c.get("authority_level"),
                "authority_tier": _authority_tier(c.get("authority_level")),
                "payer":          c.get("payer"),
                score_key:        round(float(c.get("similarity") or 0.0), 4),
                "text_preview":   _preview(c.get("text") or ""),
            }
            for c in chunks
        ]

    telemetry: dict[str, Any] = {
        "search_id":             search_id,
        "query":                 request.query,
        "bm25_normalized_query": bm25_normalized_query,   # None when no normalization occurred
        "mode":                  mode,
        "k":                     k,
        "embed_ms":              round(embed_ms, 1),
        "bm25_ms":               round(bm25_ms, 1),
        "vec_ms":                round(vec_ms, 1),
        "rerank_ms":             round(rerank_ms, 1),
        "total_ms":              total_ms,
        "arm_hits": {
            "bm25":   len(bm25_chunks),
            "vector": len(vec_chunks),
        },
        # Per-arm raw results BEFORE fusion — for frontend trace panels
        "arm_results": {
            "bm25":   _arm_summary(bm25_chunks, "ts_rank"),
            "vector": _arm_summary(vec_chunks,  "cosine"),
        },
        "candidates":            len(reranked),
        "returned":              len(chunks_out),
        "min_label_applied":     min_label,
        "reranker":              "score+authority+length (phase1)",
        "assembly":              assembly_meta,
        "scoring_trace":         scoring_trace,
        # Lexicon-driven BM25 query expansion — surfaced for chat-side
        # SearchTracePanel so reviewers can see what lexicon entries fired
        # and what tsquery was finally executed.
        "bm25_expansion":        bm25_expansion,
        # Request filters surfaced for persistence (Phase B search_events.filters)
        "filters":               request.filters.model_dump() if request.filters else None,
    }

    # Persist pipeline trace (fire-and-forget — never blocks the response).
    # Uses its OWN session inside _persist_search_event because the caller's
    # ``db`` is being torn down by the FastAPI dependency.
    asyncio.create_task(
        _persist_search_event(telemetry, caller=caller, caller_id=caller_id)
    )

    return CorpusSearchResponse(chunks=chunks_out, telemetry=telemetry)
