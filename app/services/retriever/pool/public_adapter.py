"""PUBLIC SourceAdapter -- the only adapter built in v1 (pool-schematic-spec.md S1/S3.0).

Heavy reuse, not reinvention: partition/cascade/inheritance/neighbor logic
all come straight from the legacy module, imported and called as-is. Only
the chunk-level tag-coverage selection (S3.1 step 3-4) and the
strategy-union/dedup story around it are genuinely new -- that's Pool's
actual job (target-structure-spec.md S1: no shared pool builder exists
today).
"""

from __future__ import annotations

import dataclasses
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
from app.services.corpus_search_lexicon import _load_lexicon_snapshot
from app.services.retriever.pool.contracts import PoolCandidate, ScopeContext, SourceAdapter

_CHUNK_COLS = """
    id, document_id, text, chunk_d_tags, chunk_p_tags, chunk_j_tags,
    document_status, source_type, content_sha, page_number, paragraph_index, document_authority_level
"""

# Statement timeout for Pool's candidate-fetch queries (Ananth's live catch,
# 2026-08-04): the coverage-scoring query in particular does a per-row
# correlated jsonb_object_keys() count via a Bitmap Heap Scan that can touch
# tens of thousands of rows -- under a cold buffer cache or concurrent load,
# confirmed live to take 6-9+ MINUTES with no bound at all, appearing
# indistinguishable from a genuine hang (no error, no DB lock a monitoring
# query could see -- just a legitimately slow scan with nothing capping it).
# Every other external call in this codebase already has a timeout (Vertex
# Grounding, the LLM proxy, HTTP fetches) -- this was the one gap.
# Raised 20s->60s (2026-08-04, same day): 20s turned out too tight under
# real live conditions -- the small Cloud SQL instance (db-custom-2-7680,
# 2 vCPU/7.68GB RAM for a 1.94M-row table) is cache-thrashing after a full
# day of sustained sweep load (20+ deploys, dozens of bank runs, real
# production traffic sharing the same instance), confirmed via EXPLAIN
# ANALYZE showing genuine disk I/O (96% cache misses on a representative
# query), not a bad query plan. Every query in a live re-run consistently
# hit 20s and failed -- a capacity problem, not a hang, so raised the bound
# rather than keep fighting it with a tight timeout. Still finite: a query
# that can't finish in 60s under DB capacity that recovers (e.g. off-peak,
# a properly-sized instance) is still a real signal worth surfacing, not
# something to raise indefinitely.
_POOL_QUERY_TIMEOUT_MS = 60_000


async def _set_pool_query_timeout(db) -> None:
    """SET LOCAL only applies within the current transaction (same pattern
    already established by the HNSW ef_search tuning below) -- must be
    re-issued before each candidate query, not just once per session."""
    await db.execute(sql_text(f"SET LOCAL statement_timeout = {_POOL_QUERY_TIMEOUT_MS}"))

# Additive ranking signal for Filler a (BM25), 2026-07-23 -- Fillers can't
# make DB calls (gate b), so Pool computes this once per match candidate
# regardless of which arm surfaced it, not just for a "bm25 arm" that no
# longer exists in Pool's own retrieval logic (S1: BM25 dropped from
# strategy 1 by choice). plainto_tsquery, NOT to_tsquery -- to_tsquery is a
# strict parser that throws on real user text (!, &, |, (, ), :, *, ' are
# all special); plainto_tsquery is what legacy's actual production BM25
# path uses for exactly this reason (corpus_search.py).
#
# REAL BUG found+fixed 2026-07-23 (Payor-Policy's live-trace report): a bare
# plainto_tsquery('english', :query) AND-joins EVERY content word in the raw
# question (verified: an 8-content-word query produced 'time' & 'file' &
# 'deadlin' & 'sunshin' & 'health' & 'fl' & 'medicaid' & 'claim') -- legacy's
# own production use of plainto_tsquery is as a WHERE-clause FILTER on an
# already-narrowed BM25 candidate set (only chunks matching ALL terms are
# selected in the first place), where an AND-query is exactly right. Pool's
# use is different: ranking an ARBITRARY candidate set (chosen by tag/vector/
# inheritance, never filtered by full-text match at all), where an AND-query
# is nearly always false -- confirmed live: 0/581 real candidates scored
# nonzero across a real query. Fix: convert plainto_tsquery's AND-tsquery
# into an OR-tsquery (reuses Postgres's own stemming/stopword handling via
# plainto_tsquery, just swaps the boolean operator) -- verified live this
# produces real graduated scores (e.g. 0.444 on a chunk that previously read
# a flat 0.0).
#
# EXPANSION-PHRASE REGRESSION found+fixed 2026-07-23 (Ananth, verified against
# legacy): corpus_search.py's real production BM25 (~line 943) OR-joined the
# raw query tokens with Gate's lexicon-derived expansion_phrases:
# `_build_or_tsquery(*raw_tokens, *expansion.expansion_phrases)` -- each
# phrase becomes its own AND-group (multi-word phrase coherence preserved),
# groups OR'd together. GateResult.expansion_phrases (shape/gate.py:319) is
# computed and stored but was read by NOTHING downstream -- confirmed via
# grep, zero hits in reformat.py/pool.py/any filler. Restored here: each
# phrase is turned into its own plainto_tsquery (correct per-phrase stemming,
# naturally an AND-group since plainto_tsquery ANDs a short phrase's words),
# OR-joined via string_agg, then OR-joined again with the raw-query part.
# Postgres's own tsquery operator precedence (& binds tighter than |) means
# no explicit parens are needed for this to parse correctly -- verified live
# against a real 33-phrase expansion set including OCR-noise entries (e.g.
# "cafi orida") with no SQL errors.
_BM25_SCORE_EXPR = (
    "ts_rank_cd(search_vec, "
    "to_tsquery('english', "
    "  replace(plainto_tsquery('english', :query)::text, ' & ', ' | ')"
    "  || CASE WHEN cardinality(CAST(:expansion_phrases AS text[])) > 0 THEN"
    "       ' | ' || (SELECT string_agg(plainto_tsquery('english', p)::text, ' | ')"
    "                 FROM unnest(CAST(:expansion_phrases AS text[])) AS p"
    "                 WHERE plainto_tsquery('english', p)::text != '')"
    "     ELSE '' END"
    "), 32) AS bm25_score"
)

# TRIED AND REJECTED 2026-07-30 (Ananth's own verification standard applied
# to my own proposal, not just others'): "exclude chunks with no d-tag AND
# no p-tag" from vector_search, on the theory that junk always lacks both.
# Tag-emptiness alone is NOT a safe junk proxy once you're past the length
# floor -- verified live: of the ~101k chunks that are untagged AND >=50
# chars (i.e. NOT already caught by filler_b's existing length floor),
# roughly a third sampled were genuinely substantive prose that simply
# never got tagged (a real Affinity-program comparison paragraph, a real
# "Direct Ownership Interest" regulatory definition, real grant-guidance
# text) -- not junk, just a Lexicon coverage gap of the same shape found
# repeatedly elsewhere this session. The corpus-wide count backs this up:
# ~1,006,244 rows (>50% of the corpus!) would have been excluded, of which
# 90% were already short/junk (<50 chars, redundant with the length floor)
# but ~101k were not. Shipping the naive tag-emptiness filter would have
# silently cut real content on a false premise. Left here as a documented
# dead end so nobody re-tries the same naive version -- the real fix (per
# Filler b's own second, independent signal) needs corpus-wide EXACT-TEXT
# DUPLICATE FREQUENCY (junk repeats verbatim across dozens-to-hundreds of
# thousands of unrelated documents; real prose essentially never does),
# which requires a precomputed/indexed frequency signal, not a per-query
# WHERE clause -- a Curation/DB-side task, not solved here.


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
        self, query: str, expansion_phrases: list[str], d_codes_in: list[str], j_codes_in: list[str], p_codes_in: list[str], width: int
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
        # `, id ASC` tiebreak (Ananth's live catch, 2026-08-04): `coverage`
        # is a small integer (count of matched tag codes) -- ties are common,
        # not rare, and ORDER BY coverage DESC alone left them to whatever
        # order Postgres happened to scan rows in, which is NOT guaranteed
        # stable across executions/plans. Confirmed live: the SAME query
        # against the SAME pool produced a DIFFERENT top-ranked chunk across
        # two bank runs (two chunks tied on bm25_score too, further downstream
        # in fill_shape_bm25's composite rerank -- Python's sort is stable, so
        # the real non-determinism traces back to Pool's own candidate order
        # here, not the filler's rerank). Legacy corpus_search.py already hit
        # and fixed this exact class of bug (its own "Determinism fix
        # (2026-05-03): ORDER BY bm25_score DESC, id ASC" comment) -- this
        # port never carried that fix over into the new Pool module.
        # NULLIF(..., 'null'::jsonb) matters: verified live (2026-07-23) that
        # chunk_{d,p,j}_tags stores an actual JSON-null LITERAL on ~55k rows
        # (not SQL NULL) -- jsonb_object_keys() throws "cannot call
        # jsonb_object_keys on a scalar" on that value, and plain COALESCE
        # doesn't catch it (COALESCE only fires on true SQL NULL).
        t1 = time.monotonic()
        await _set_pool_query_timeout(self.db)
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
                ORDER BY coverage DESC, id ASC
                LIMIT :width
            """),
            {
                "doc_ids": pool.document_ids,
                "d_codes": d_bare,
                "p_codes": p_bare,
                "j_codes": j_bare,
                "width": width,
                "query": query,
                "expansion_phrases": expansion_phrases,
            },
        )).all()
        segment_ms["tag_select_ms"] = int((time.monotonic() - t1) * 1000)

        candidates = [
            _row_to_candidate(r, source_arm="tag_select", score=float(r._mapping["coverage"]))
            for r in rows
        ]
        return candidates, segment_ms

    async def vector_search(
        self, query: str, expansion_phrases: list[str], j_codes: list[str], width: int
    ) -> tuple[list[PoolCandidate], dict, list[float] | None]:
        segment_ms: dict = {}
        embedding, embed_ms, _cache_hit = await _embed_with_cache(query)
        segment_ms["embed_ms"] = embed_ms
        if not embedding:
            segment_ms["vector_ms"] = 0
            return [], segment_ms, None

        query_vec = "[" + ",".join(repr(float(x)) for x in embedding) + "]"
        t0 = time.monotonic()
        # REAL BUG found+fixed 2026-07-23 (Filler b's live-trace report):
        # app/database.py sets hnsw.ef_search=100 as a connection-level
        # default (tuned for legacy's k=80 wide-phase search). Pool's own
        # width can run up to breadth*100 (=1000 for a typical breadth=10)
        # -- 10x what ef_search=100 explores during HNSW graph traversal.
        # Verified live on the cmhc001 repro: the TRUE best match
        # (sim=0.8892, the actual answer chunk) was completely ABSENT from
        # the top-1000 results at ef_search=100 -- HNSW doesn't just return
        # fewer good matches when under-provisioned for a wide LIMIT, it
        # backfills with genuinely worse ones while silently missing true
        # nearest neighbors it never explored far enough to find. Raising
        # ef_search to 500/1000 surfaced that same chunk at rank 4/3.
        # SET LOCAL doesn't accept bind parameters (Postgres syntax
        # restriction, confirmed live) -- safe to interpolate directly since
        # `width` is an internally-computed int, never user text. Capped at
        # 1000 as a defensive ceiling on query cost, not because Postgres
        # itself rejects higher values (it doesn't, confirmed live).
        ef_search = min(max(width, 100), 1000)
        await self.db.execute(sql_text(f"SET LOCAL hnsw.ef_search = {ef_search}"))
        await _set_pool_query_timeout(self.db)
        # REAL CORRECTNESS BUG found+fixed 2026-07-23 (Payor-Policy's
        # live-trace report, verified directly): vector_search() had ZERO
        # payer/jurisdiction filtering -- a bare similarity search over the
        # entire 1.94M-row corpus, unlike tag_select which scopes via
        # build_candidate_pool(). Confirmed live: a Sunshine-Health-specific
        # query ("How do I submit a corrected claim to Sunshine Health
        # Florida?", gate correctly matches j:payor.sunshine_health) surfaced
        # 10 candidates in the top-1000 tagged ONLY with payor.aetna or
        # payor.molina_healthcare (no Sunshine/Centene co-tag), scoring
        # 0.77-0.81 similarity -- genuinely different payers' policies shown
        # as if relevant to a payer-specific question. This is a correctness
        # issue (wrong-payer content as fact), not just a recall/precision
        # one, so it gets a NEGATIVE exclusion rather than a positive
        # AND-filter: chunks with NO payor tag at all (generic/AHCA-authority
        # content) still pass through untouched -- vector's wide-net
        # semantic-discovery purpose is preserved -- but chunks tagged with
        # a DIFFERENT, non-matched payor.* code are excluded outright. Only
        # activates when the query actually names a specific payer
        # (j_payors non-empty); a query with no payer tag gets no exclusion.
        j_payors = [c.split(":", 1)[1] for c in j_codes if c.startswith("j:payor.")]
        rows = (await self.db.execute(
            sql_text(f"""
                SELECT {_CHUNK_COLS},
                    1 - (embedding_vec <=> CAST(:query_vec AS vector)) AS similarity,
                    {_BM25_SCORE_EXPR}
                FROM rag_published_embeddings
                WHERE embedding_vec IS NOT NULL
                    AND (
                        CAST(:j_payors AS text[]) = '{{}}'
                        OR NOT EXISTS (
                            SELECT 1 FROM jsonb_object_keys(COALESCE(NULLIF(chunk_j_tags, 'null'::jsonb), '{{}}'::jsonb)) k
                            WHERE k LIKE 'payor.%' AND k != ALL(CAST(:j_payors AS text[]))
                        )
                    )
                -- NO tiebreak here, unlike the coverage/inherited queries
                -- (Ananth's live catch, 2026-08-04 -- REVERTED same day):
                -- a `, id ASC` was added earlier today for determinism, same
                -- reasoning as those two queries, but this one is DIFFERENT
                -- -- pgvector's HNSW index can only accelerate
                -- `ORDER BY embedding <=> query_vec LIMIT k` when that
                -- expression is the ONLY sort key. Adding id ASC silently
                -- disabled the ANN index entirely: confirmed via EXPLAIN,
                -- plan flipped from "Index Scan using
                -- rag_published_embeddings_vec_hnsw" to a full
                -- "Parallel Seq Scan" across all 1.94M rows computing exact
                -- distances -- this was the actual cause of today's
                -- widespread bank-eval timeouts (mistakenly chased as DB
                -- capacity/contention for hours before isolating it here).
                -- Cosine-distance ties are rare enough in practice that
                -- losing tie-break determinism on JUST this query is an
                -- acceptable trade against silently losing the index on
                -- every vector search in production.
                ORDER BY embedding_vec <=> CAST(:query_vec AS vector)
                LIMIT :width
            """),
            {
                "query_vec": query_vec, "width": width, "query": query,
                "expansion_phrases": expansion_phrases, "j_payors": j_payors,
            },
        )).all()
        segment_ms["vector_ms"] = int((time.monotonic() - t0) * 1000)

        candidates = [
            _row_to_candidate(r, source_arm="vector", score=float(r._mapping["similarity"]))
            for r in rows
        ]
        return candidates, segment_ms, embedding

    async def inherited(
        self, query: str, expansion_phrases: list[str], payor_codes: list[str], width: int
    ) -> tuple[list[PoolCandidate], dict]:
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

        # ORDER BY bm25_score, added 2026-07-23 (Payor-Policy's live-trace
        # report): previously no ORDER BY at all -- an arbitrary DB-scan-order
        # slice of up to `width` chunks across every AHCA-inherited document
        # for the payor, regardless of topical fit to the actual query (real
        # example: a Sunshine-Health timely-filing query pulling "Early
        # Intervention Services Coverage Policy" chunks just because that
        # inherited doc happened to be scanned early). bm25_score was already
        # computed here but unused for ordering -- now that it's a real,
        # meaningful signal (see _BM25_SCORE_EXPR fix above), prioritizing by
        # it means the AHCA-inherited chunks that actually relate to the
        # query surface first when `width` truncates the set.
        await _set_pool_query_timeout(self.db)
        rows = (await self.db.execute(
            sql_text(f"""
                SELECT {_CHUNK_COLS}, {_BM25_SCORE_EXPR}
                FROM rag_published_embeddings
                WHERE document_id = ANY(:doc_ids)
                ORDER BY bm25_score DESC, id ASC
                LIMIT :width
            """),
            {"doc_ids": doc_ids, "width": width, "query": query, "expansion_phrases": expansion_phrases},
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

    async def phrase_buckets(
        self, d_codes: list[str], j_codes: list[str], p_codes: list[str], expansion_phrases: list[str]
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """Reconstructs Gate's expansion_phrases -> REQUIRED/BOOSTED
        selectivity buckets. The phrase<->code link exists at lexicon-match
        time (expand_query_via_lexicon appends a matched entry's own phrases
        alongside its full_code) but is flattened away by the time it lands
        on GateResult.expansion_phrases -- rebuilt here via the same lexicon
        snapshot Gate reads (_load_lexicon_snapshot, cached, public table),
        paired with each matched code's already-computed selectivity_for_tag()
        score. Verified live 2026-07-23 against a real query: reproduces the
        exact REQUIRED (timely_filing/sunshine_health, sel>=0.65) vs DROP
        (claims/medicaid, sel<0.40) split that was diluting bm25 ranking.
        """
        snapshot = await _load_lexicon_snapshot(self.db)
        phrases_by_code = {e["full_code"]: e["phrases"] for e in snapshot}
        expansion_set = set(expansion_phrases)

        required: dict[str, float] = {}
        boosted: dict[str, float] = {}
        for code in [*d_codes, *j_codes, *p_codes]:
            sel = await selectivity_for_tag(self.db, code)
            code_phrases = [p for p in phrases_by_code.get(code, []) if p in expansion_set]
            if not code_phrases:
                continue
            if sel >= _SELECTIVITY_REQUIRED:
                for p in code_phrases:
                    required[p] = max(required.get(p, 0.0), sel)
            elif sel >= _SELECTIVITY_BOOST:
                for p in code_phrases:
                    boosted[p] = max(boosted.get(p, 0.0), sel)
            # sel < _SELECTIVITY_BOOST (DROP): excluded entirely, not kept at low weight.
        return list(required.items()), list(boosted.items())

    async def attach_vector_similarity(
        self, candidates: list[PoolCandidate], query_embedding: list[float] | None
    ) -> tuple[list[PoolCandidate], dict]:
        """Real structural bug fix (2026-07-23, Retriever's live-trace report):
        dedup's first-arm-wins on chunk_id collision means a chunk found by
        BOTH tag_select and vector_search only keeps tag_select's provenance
        -- Filler b (which filters to source_arm=="vector") never sees it.
        Computes similarity for EVERY candidate here (matches AND neighbors),
        one batch query against the deduped id set, reusing the query
        embedding Pool already computed -- same pattern bm25_score already
        established for making a signal available regardless of which arm's
        provenance survived the union."""
        if not query_embedding or not candidates:
            return candidates, {"vector_similarity_ms": 0}

        query_vec = "[" + ",".join(repr(float(x)) for x in query_embedding) + "]"
        ids = [c.chunk_id for c in candidates]
        t0 = time.monotonic()
        rows = (await self.db.execute(
            sql_text("""
                SELECT id, 1 - (embedding_vec <=> CAST(:query_vec AS vector)) AS similarity
                FROM rag_published_embeddings
                WHERE id = ANY(CAST(:ids AS uuid[])) AND embedding_vec IS NOT NULL
            """),
            {"query_vec": query_vec, "ids": ids},
        )).all()
        segment_ms = {"vector_similarity_ms": int((time.monotonic() - t0) * 1000)}

        similarity_by_id = {str(r._mapping["id"]): float(r._mapping["similarity"]) for r in rows}
        updated = [
            dataclasses.replace(c, vector_similarity=similarity_by_id[c.chunk_id])
            if c.chunk_id in similarity_by_id else c
            for c in candidates
        ]
        return updated, segment_ms
