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
        # Use ANY with a UUID-cast so Postgres doesn't reject the text array
        clauses.append("document_id::text = ANY(:inc_ids)")
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
    }


# ---------------------------------------------------------------------------
# BM25 arm
# ---------------------------------------------------------------------------

_BM25_COLS = """
    id, document_id, source_type, text,
    page_number, paragraph_index,
    document_display_name, document_filename,
    document_authority_level, document_payer, document_state
"""

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


async def _bm25_arm(
    db: AsyncSession,
    query: str,
    k: int,
    filters: CorpusFilters | None,
    include_document_ids: list[str] | None,
    search_id: str = "",
) -> tuple[list[dict[str, Any]], str | None]:
    """Full-text BM25 via Postgres tsvector.

    Uses the ``search_vec`` GENERATED column + GIN index when available
    (after ``add_rag_published_fts`` migration).  Falls back to inline
    ``to_tsvector`` (sequential scan) if the column doesn't exist yet —
    result is identical, just slower on large tables.
    """
    if not query.strip():
        return [], None

    # Normalize query before handing to plainto_tsquery.
    # Natural-language questions contain noise tokens ('mani', 'day', 'file')
    # that AND-kill results even when the content terms all match.
    raw_query = query.strip()
    bm25_query = _normalize_bm25_query(raw_query)
    normalized: str | None = bm25_query if bm25_query != raw_query else None
    if normalized and search_id:
        _log_stage("bm25_query_normalized", search_id,
                   original=raw_query[:80], normalized=bm25_query[:80])

    params: dict[str, Any] = {"k": k, "query": bm25_query}
    filter_sql = _build_filter_clauses(filters, include_document_ids, params)

    # Try GIN-indexed path first
    gin_path = True
    try:
        sql = text(f"""
            SELECT {_BM25_COLS},
                ts_rank_cd(search_vec, plainto_tsquery('english', :query), 32) AS bm25_score
            FROM rag_published_embeddings
            WHERE search_vec @@ plainto_tsquery('english', :query)
              {filter_sql}
            ORDER BY bm25_score DESC
            LIMIT :k
        """)
        result = await db.execute(sql, params)
    except Exception as exc:
        if "search_vec" in str(exc):
            # Migration not yet applied — fall back to inline tsvector
            gin_path = False
            logger.warning("corpus_search bm25: search_vec column missing, using inline tsvector: %s", exc)
            sql = text(f"""
                SELECT {_BM25_COLS},
                    ts_rank_cd(
                        to_tsvector('english', coalesce(text, '')),
                        plainto_tsquery('english', :query), 32
                    ) AS bm25_score
                FROM rag_published_embeddings
                WHERE to_tsvector('english', coalesce(text, ''))
                      @@ plainto_tsquery('english', :query)
                  {filter_sql}
                ORDER BY bm25_score DESC
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
    return out, normalized


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
) -> list[dict[str, Any]]:
    """pgvector ANN over rag_published_embeddings.embedding_vec (HNSW cosine)."""
    params: dict[str, Any] = {"k": k}
    filter_sql = _build_filter_clauses(filters, include_document_ids, params)
    # pgvector text form: '[f1,f2,...]'
    params["query_vec"] = "[" + ",".join(repr(float(x)) for x in query_embedding) + "]"

    sql = text(f"""
        SELECT {_BM25_COLS},
            1 - (embedding_vec <=> CAST(:query_vec AS vector)) AS similarity
        FROM rag_published_embeddings
        WHERE embedding_vec IS NOT NULL
          {filter_sql}
        ORDER BY embedding_vec <=> CAST(:query_vec AS vector)
        LIMIT :k
    """)
    try:
        result = await db.execute(sql, params)
    except Exception as exc:
        logger.error("corpus_search vector arm failed: %s", exc, exc_info=True)
        return []

    rows = result.mappings().all()
    out: list[dict[str, Any]] = []
    for rank0, row in enumerate(rows):
        c = _row_to_base_dict(row)
        cosine_sim = max(0.0, min(1.0, float(row["similarity"] or 0.0)))
        c["similarity"] = cosine_sim
        c["match_score"] = cosine_sim
        c["_arm"] = "vector"
        out.append(c)
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
        _log_stage("vector_arm_summary", search_id, hits=len(out))
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


def _rerank(
    chunks: list[dict[str, Any]],
    search_id: str = "",
    query: str = "",
) -> list[dict[str, Any]]:
    """Apply weighted signals and sort by rerank_score (desc).

    Weights from reranker_v1.yaml (Phase 2B — full 80 % coverage):
      score (0.30) + authority_level (0.15) + length (0.10) + jpd_tag_match (0.25) = 0.80

    We rescale to [0, 1] by dividing by 0.80 (MAX_WEIGHT) so confidence
    thresholds remain interpretable.

    The ``score`` signal uses ``_best_arm_sim`` which rescales the raw arm
    scores (BM25 ts_rank or cosine similarity) rather than the compressed
    RRF score, giving far better discrimination between candidates.

    The ``jpd_tag_match`` signal classifies the query into J/P/D intent
    categories and scores each chunk by keyword overlap.  When the query has
    no JPD intent the signal contributes 0; the other signals fill the gap and
    MAX_WEIGHT falls back to 0.55 (Phase 1 behaviour).
    """
    if not chunks:
        return chunks

    # Pre-classify query JPD intent once (shared across all chunks)
    query_cats = _classify_jpd(query) if query else {}
    has_jpd = bool(query_cats)

    MAX_WEIGHT = 0.30 + 0.15 + 0.10 + (0.25 if has_jpd else 0.0)

    for c in chunks:
        sim   = _best_arm_sim(c)
        auth  = _authority_score(c.get("authority_level"))
        lsig  = _length_score(c.get("text") or "")
        jpd, jpd_tags = _jpd_signal(query_cats, c.get("text") or "") if has_jpd else (0.0, [])

        raw   = 0.30 * sim + 0.15 * auth + 0.10 * lsig + 0.25 * jpd
        score = raw / MAX_WEIGHT if MAX_WEIGHT > 0 else raw
        c["rerank_score"] = score
        c["_jpd_tags"] = jpd_tags
        # Stash per-signal breakdown for trace / telemetry
        c["_rerank_signals"] = {
            "sim_raw":            round(sim, 4),
            "sim_weighted":       round(0.30 * sim, 4),
            "authority_raw":      round(auth, 4),
            "authority_weighted": round(0.15 * auth, 4),
            "length_raw":         round(lsig, 4),
            "length_weighted":    round(0.10 * lsig, 4),
            "jpd_raw":            round(jpd, 4),
            "jpd_weighted":       round(0.25 * jpd, 4),
            "jpd_tags":           jpd_tags,
            "total_raw":          round(raw, 4),
            "rerank_score":       round(score, 4),
            "max_weight":         MAX_WEIGHT,
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

    selected: list[dict[str, Any]] = []
    for c in ordered:
        if len(selected) >= k:
            break
        if not _THRESHOLD_PASS(c):
            continue
        page_key = (c["document_id"], c.get("page_number"))
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)
        selected.append(c)

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
# Persistence helper
# ---------------------------------------------------------------------------

async def _persist_search_event(
    db: AsyncSession,
    telemetry: dict[str, Any],
    caller: str = "api",
) -> None:
    """Fire-and-forget: write search telemetry to search_events.

    Errors are swallowed — persistence must never block or fail a search.
    """
    try:
        import json as _json
        await db.execute(
            text("""
                INSERT INTO search_events
                    (search_id, caller, query, bm25_normalized_query,
                     mode, k, returned,
                     total_ms, embed_ms, bm25_ms, vec_ms, rerank_ms,
                     arm_hits, arm_results, scoring_trace, assembly)
                VALUES
                    (:search_id, :caller, :query, :bm25_normalized_query,
                     :mode, :k, :returned,
                     :total_ms, :embed_ms, :bm25_ms, :vec_ms, :rerank_ms,
                     :arm_hits, :arm_results, :scoring_trace, :assembly)
            """),
            {
                "search_id":             telemetry.get("search_id", ""),
                "caller":                caller,
                "query":                 telemetry.get("query", ""),
                "bm25_normalized_query": telemetry.get("bm25_normalized_query"),
                "mode":                  telemetry.get("mode", "corpus"),
                "k":                     telemetry.get("k", 10),
                "returned":              telemetry.get("returned", 0),
                "total_ms":              telemetry.get("total_ms"),
                "embed_ms":              telemetry.get("embed_ms"),
                "bm25_ms":               telemetry.get("bm25_ms"),
                "vec_ms":                telemetry.get("vec_ms"),
                "rerank_ms":             telemetry.get("rerank_ms"),
                "arm_hits":              _json.dumps(telemetry.get("arm_hits")),
                "arm_results":           _json.dumps(telemetry.get("arm_results")),
                "scoring_trace":         _json.dumps(telemetry.get("scoring_trace")),
                "assembly":              _json.dumps(telemetry.get("assembly")),
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

    if mode == "corpus":
        # Schedule BM25 as a background task so it overlaps with the embed call.
        # asyncio tasks interleave at await-points; DB and OpenAI use separate I/O.
        tb = time.monotonic()
        bm25_task: asyncio.Task[tuple[list[dict[str, Any]], str | None]] = asyncio.create_task(
            _bm25_arm(
                db, request.query, k * 2,
                request.filters, request.include_document_ids,
                search_id=search_id,
            )
        )
        # Embed runs in foreground while BM25 is in flight.
        query_embedding, embed_ms, _cache_hit = await _embed_with_cache(
            request.query, search_id
        )
        # BM25 is usually done by now; await completes immediately if so.
        bm25_chunks, bm25_normalized_query = await bm25_task
        bm25_ms = (time.monotonic() - tb) * 1000 - embed_ms  # net DB time

        if query_embedding:
            tv = time.monotonic()
            vec_chunks = await _vector_arm(
                db, query_embedding, k * 2,
                request.filters, request.include_document_ids,
                search_id=search_id,
            )
            vec_ms = (time.monotonic() - tv) * 1000

    elif mode == "precision":
        tb = time.monotonic()
        bm25_chunks, bm25_normalized_query = await _bm25_arm(
            db, request.query, k * 2,
            request.filters, request.include_document_ids,
            search_id=search_id,
        )
        bm25_ms = (time.monotonic() - tb) * 1000

    else:  # recall
        query_embedding, embed_ms, _cache_hit = await _embed_with_cache(
            request.query, search_id
        )
        if query_embedding:
            tv = time.monotonic()
            vec_chunks = await _vector_arm(
                db, query_embedding, k * 2,
                request.filters, request.include_document_ids,
                search_id=search_id,
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

    # ── 4. Rerank ─────────────────────────────────────────────────────────
    reranked = _rerank(candidates, search_id=search_id, query=request.query)
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
    }

    # Persist pipeline trace (fire-and-forget — never blocks the response)
    asyncio.create_task(_persist_search_event(db, telemetry, caller=caller))

    return CorpusSearchResponse(chunks=chunks_out, telemetry=telemetry)
