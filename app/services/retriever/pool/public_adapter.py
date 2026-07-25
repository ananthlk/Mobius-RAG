"""PUBLIC SourceAdapter -- the only adapter built in v1 (pool-schematic-spec.md S1/S3.0).

Heavy reuse, not reinvention: partition/cascade/inheritance/neighbor logic
all come straight from the legacy module, imported and called as-is. Only
the chunk-level tag-coverage selection (S3.1 step 3-4) and the
strategy-union/dedup story around it are genuinely new -- that's Pool's
actual job (target-structure-spec.md S1: no shared pool builder exists
today).
"""

from __future__ import annotations

import time

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.corpus_search import _embed_with_cache, _expand_with_neighbors
from app.services.corpus_search_agent import (
    _SELECTIVITY_BOOST,
    _SELECTIVITY_REQUIRED,
    _inherited_authority_doc_ids,
    build_candidate_pool,
    selectivity_for_tag,
    TermAssignment,
    TermPartition,
)
from app.services.retriever.pool.contracts import PoolCandidate, ScopeContext, SourceAdapter

_CHUNK_COLS = """
    id, document_id, text, chunk_d_tags, chunk_p_tags, chunk_j_tags,
    document_status, source_type, content_sha, page_number, paragraph_index, document_authority_level
"""

# Additive ranking signal for Filler a (BM25), 2026-07-23 -- Fillers can't
# make DB calls (gate b), so Pool computes this once per match candidate
# regardless of which arm surfaced it, not just for a "bm25 arm" that no
# longer exists in Pool's own retrieval logic (S1: BM25 dropped from
# strategy 1 by choice). plainto_tsquery, NOT to_tsquery -- to_tsquery is a
# strict parser that throws on real user text (!, &, |, (, ), :, *, ' are
# all special); plainto_tsquery is what legacy's actual production BM25
# path uses for exactly this reason (corpus_search.py).
_BM25_SCORE_EXPR = "ts_rank_cd(search_vec, plainto_tsquery('english', :query), 32) AS bm25_score"


def _row_to_candidate(row, *, source_arm: str, score: float | None, is_neighbor: bool = False) -> PoolCandidate:
    tags = {}
    for col in ("chunk_d_tags", "chunk_p_tags", "chunk_j_tags"):
        v = row._mapping.get(col)
        if v:
            tags.update(v)
    bm25 = row._mapping.get("bm25_score")
    return PoolCandidate(
        chunk_id=str(row._mapping["id"]),
        document_id=str(row._mapping["document_id"]),
        text=row._mapping["text"] or "",
        is_neighbor=is_neighbor,
        source_arm=source_arm,
        score=score,
        tags=tags,
        document_status=row._mapping.get("document_status"),
        source_type=row._mapping.get("source_type"),
        content_sha=row._mapping.get("content_sha"),
        page_number=row._mapping.get("page_number"),
        paragraph_index=row._mapping.get("paragraph_index"),
        bm25_score=float(bm25) if bm25 is not None else None,
        authority_level=row._mapping.get("document_authority_level"),
    )


def _split_by_kind(codes: list[str]) -> dict[str, list[str]]:
    """"j:payor.aetna" -> kind="j", bare code="payor.aetna" -- same split
    _doc_ids_with_tag() uses, so chunk_*_tags keys match document_tags'
    keying convention (bare codes, no kind prefix)."""
    out: dict[str, list[str]] = {"d": [], "p": [], "j": []}
    for full_code in codes:
        if ":" not in full_code:
            continue
        kind, bare = full_code.split(":", 1)
        if kind in out:
            out[kind].append(bare)
    return out


class PublicSourceAdapter(SourceAdapter):
    """rag_published_embeddings (1.94M rows), global scope, adapter #1."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.scope = ScopeContext(kind="public")

    async def _partition_tags_only(self, d_codes: list[str], j_codes: list[str], p_codes: list[str]) -> TermPartition:
        """Tag-only TermPartition -- S4.1's v1 gap: GateResult carries no
        literal_anchors/untagged_meaningful_tokens (Gate never computes
        them), so this covers only the tag-classification third of
        partition_terms(), reusing selectivity_for_tag()/thresholds as-is."""
        required: list[TermAssignment] = []
        boosted: list[TermAssignment] = []
        dropped: list[TermAssignment] = []
        for code in [*d_codes, *j_codes, *p_codes]:
            sel = await selectivity_for_tag(self.db, code)
            if sel >= _SELECTIVITY_REQUIRED:
                required.append(TermAssignment(term=code, kind="tag", full_code=code, selectivity=sel, bucket="REQUIRED"))
            elif sel >= _SELECTIVITY_BOOST:
                boosted.append(TermAssignment(term=code, kind="tag", full_code=code, selectivity=sel, bucket="BOOSTED"))
            else:
                dropped.append(TermAssignment(term=code, kind="tag", full_code=code, selectivity=sel, bucket="DROP"))
        return TermPartition(required=required, boosted=boosted, dropped=dropped)

    async def tag_select(
        self, query: str, d_codes_in: list[str], j_codes_in: list[str], p_codes_in: list[str], width: int
    ) -> tuple[list[PoolCandidate], dict]:
        segment_ms: dict = {}
        partition = await self._partition_tags_only(d_codes_in, j_codes_in, p_codes_in)
        required_codes = [t.full_code for t in partition.required]
        boosted_codes = [t.full_code for t in partition.boosted]

        t0 = time.monotonic()
        pool = await build_candidate_pool(self.db, partition)
        segment_ms["doc_narrow_ms"] = int((time.monotonic() - t0) * 1000)

        if not pool.document_ids:
            segment_ms["tag_select_ms"] = 0
            return [], segment_ms

        by_kind = _split_by_kind(required_codes + boosted_codes)
        d_bare, p_bare, j_bare = by_kind["d"], by_kind["p"], by_kind["j"]

        # Coverage-maximizing selection within the narrowed doc set --
        # S3.1 step 3/4. chunk_*_tags ?| :codes is the predicate GIN (i)
        # covers (gate resolved 2026-07-23) -- bounded to pool.document_ids
        # regardless, never a blind scan of the 1.94M table.
        # NULLIF(..., 'null'::jsonb) matters: verified live (2026-07-23) that
        # chunk_{d,p,j}_tags stores an actual JSON-null LITERAL on ~55k rows
        # (not SQL NULL) -- jsonb_object_keys() throws "cannot call
        # jsonb_object_keys on a scalar" on that value, and plain COALESCE
        # doesn't catch it (COALESCE only fires on true SQL NULL).
        t1 = time.monotonic()
        rows = (await self.db.execute(
            sql_text(f"""
                SELECT {_CHUNK_COLS},
                    (
                        (SELECT count(*) FROM jsonb_object_keys(COALESCE(NULLIF(chunk_d_tags, 'null'::jsonb), '{{}}'::jsonb)) k WHERE k = ANY(:d_codes)) +
                        (SELECT count(*) FROM jsonb_object_keys(COALESCE(NULLIF(chunk_p_tags, 'null'::jsonb), '{{}}'::jsonb)) k WHERE k = ANY(:p_codes)) +
                        (SELECT count(*) FROM jsonb_object_keys(COALESCE(NULLIF(chunk_j_tags, 'null'::jsonb), '{{}}'::jsonb)) k WHERE k = ANY(:j_codes))
                    ) AS coverage,
                    {_BM25_SCORE_EXPR}
                FROM rag_published_embeddings
                WHERE document_id = ANY(:doc_ids)
                    AND (
                        (:d_codes = '{{}}' OR chunk_d_tags ?| :d_codes)
                        OR (:p_codes = '{{}}' OR chunk_p_tags ?| :p_codes)
                        OR (:j_codes = '{{}}' OR chunk_j_tags ?| :j_codes)
                    )
                ORDER BY coverage DESC
                LIMIT :width
            """),
            {
                "doc_ids": pool.document_ids,
                "d_codes": d_bare,
                "p_codes": p_bare,
                "j_codes": j_bare,
                "width": width,
                "query": query,
            },
        )).all()
        segment_ms["tag_select_ms"] = int((time.monotonic() - t1) * 1000)

        candidates = [
            _row_to_candidate(r, source_arm="tag_select", score=float(r._mapping["coverage"]))
            for r in rows
        ]
        return candidates, segment_ms

    async def vector_search(
        self, query: str, width: int
    ) -> tuple[list[PoolCandidate], dict, list[float] | None]:
        segment_ms: dict = {}
        embedding, embed_ms, _cache_hit = await _embed_with_cache(query)
        segment_ms["embed_ms"] = embed_ms
        if not embedding:
            segment_ms["vector_ms"] = 0
            return [], segment_ms, None

        query_vec = "[" + ",".join(repr(float(x)) for x in embedding) + "]"
        t0 = time.monotonic()
        rows = (await self.db.execute(
            sql_text(f"""
                SELECT {_CHUNK_COLS},
                    1 - (embedding_vec <=> CAST(:query_vec AS vector)) AS similarity,
                    {_BM25_SCORE_EXPR}
                FROM rag_published_embeddings
                WHERE embedding_vec IS NOT NULL
                ORDER BY embedding_vec <=> CAST(:query_vec AS vector)
                LIMIT :width
            """),
            {"query_vec": query_vec, "width": width, "query": query},
        )).all()
        segment_ms["vector_ms"] = int((time.monotonic() - t0) * 1000)

        candidates = [
            _row_to_candidate(r, source_arm="vector", score=float(r._mapping["similarity"]))
            for r in rows
        ]
        return candidates, segment_ms, embedding

    async def inherited(self, query: str, payor_codes: list[str], width: int) -> tuple[list[PoolCandidate], dict]:
        """AHCA-authority augmentation -- fires only on an actual j:payor.*
        match (S1 correction: the "no payor tag -> AHCA" behavior is
        already handled by tag_select's own cascade substituting L3_AHCA_D/
        L4_AHCA, NOT by this method -- this method is specifically the
        augment-a-plan-scoped-pool-with-inherited-docs case, per
        _augment_pool_with_inheritance()'s own docstring: "Only applied to
        plan-scoped (L1/L2) pools.")"""
        segment_ms = {"inherited_ms": 0}
        j_payor = [c for c in payor_codes if c.startswith("j:payor.")]
        if not j_payor:
            return [], segment_ms

        t0 = time.monotonic()
        doc_ids = await _inherited_authority_doc_ids(self.db, j_payor)
        if not doc_ids:
            segment_ms["inherited_ms"] = int((time.monotonic() - t0) * 1000)
            return [], segment_ms

        rows = (await self.db.execute(
            sql_text(f"""
                SELECT {_CHUNK_COLS}, {_BM25_SCORE_EXPR}
                FROM rag_published_embeddings
                WHERE document_id = ANY(:doc_ids)
                LIMIT :width
            """),
            {"doc_ids": doc_ids, "width": width, "query": query},
        )).all()
        segment_ms["inherited_ms"] = int((time.monotonic() - t0) * 1000)

        candidates = [_row_to_candidate(r, source_arm="inherited", score=None) for r in rows]
        return candidates, segment_ms

    async def neighbors(self, candidates: list[PoolCandidate]) -> tuple[list[PoolCandidate], dict]:
        if not candidates:
            return [], {"neighbor_ms": 0}
        seeds = [
            {
                "id": c.chunk_id,
                "document_id": c.document_id,
                "text": c.text,
                "content_sha": c.content_sha,
                "page_number": c.page_number,
                "paragraph_index": c.paragraph_index,
            }
            for c in candidates
        ]
        t0 = time.monotonic()
        assembled, _neighbor_meta = await _expand_with_neighbors(self.db, seeds, paragraph_window=2, page_window=1)
        neighbor_ms = int((time.monotonic() - t0) * 1000)

        seed_ids = {c.chunk_id for c in candidates}
        out = list(candidates)
        for d in assembled:
            cid = str(d.get("id") or "")
            if cid in seed_ids:
                continue
            out.append(PoolCandidate(
                chunk_id=cid,
                document_id=str(d.get("document_id") or ""),
                text=d.get("text") or "",
                is_neighbor=True,
                source_arm="",
                score=None,
                content_sha=d.get("content_sha"),
                page_number=d.get("page_number"),
                paragraph_index=d.get("paragraph_index"),
            ))
        return out, {"neighbor_ms": neighbor_ms}
