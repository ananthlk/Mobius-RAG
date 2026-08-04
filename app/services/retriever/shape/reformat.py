"""Reformat — Step 1b of the shape module (Gate → Reformat → Structure).

Translates a GateResult (already classified) into what Pool actually
searches: one strict query (PRECISE), a bounded/ranked fan-out (FAN_OUT,
vector-clustered themes + one deliberately-uncorpus-derived catch-all),
suggested clarifying questions (CLARIFY / CLARIFY_REPHRASE), a signal to
defer to Router's external strategies (RELY_ON_EXTERNAL), or a decline
(DECLINE). See docs/rag-agents/shape-reformat-schematic-spec.md for the
full design record, including what was tried and rejected.

Does not touch Gate's classification logic. Does not do Pool's actual
corpus search.
"""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.retriever.shape.contracts import (
    MAX_FANOUT_THEMES,
    Contour,
    FanoutTheme,
    GateResult,
    ReformatPosture,
    ReformatResult,
)

# Named themes before the catch-all fills the last slot. Eval's proposed
# starting point (queries_reformat_postures.yaml §6) — not re-derived here.
_TARGET_THEME_COUNT = 3

# Hybrid score weights: score = W_PREVALENCE*prevalence_norm + W_LEXICON_PROXIMITY*lexicon_proximity.
# Eval's proposed starting values, same source — empirical tuning is Eval's job, not guessed here.
_W_PREVALENCE = 0.4
_W_LEXICON_PROXIMITY = 0.6

# Vertex's hard API limit, confirmed live 2026-07-23 via the actual error:
# "400 250 instance(s) is allowed per prediction. Actual: 294" at 294. Not a
# guess — this is the real ceiling for text-embedding-004.
_MAX_INSTANCES_PER_EMBED_CALL = 250

_clustering_model = None  # lazy singleton, see _embed_for_clustering()


def _embed_for_clustering(texts: list[str]) -> list[list[float]]:
    """Deliberately NOT the shared app.services.embedding_provider path.

    Two independent reasons:
    1. Model choice: the shared EMBEDDING_PROVIDER/EMBEDDING_MODEL config
       (gemini-embedding-001, 1536-dim) is tuned for corpus embeddings that
       must interoperate with rag_published_embeddings, and
       gemini-embedding-001 takes exactly 1 input per API call. Reformat's
       clustering vectors only ever get compared against each other and
       never touch the corpus index — no dimension/model compatibility
       requirement, free to pick text-embedding-004 instead.
    2. Batch size: the shared `_vertex_embed()` hard-codes `batch_size = 5`
       for any non-gemini model — NOT a real API limit, just an overly
       conservative default (confirmed live 2026-07-23: 81 texts at
       batch_size=5 took 5.28s / 17 calls; ONE call of all 81 took 1.37s —
       a real API limit test at 294 texts found the true ceiling is 250
       instances/call, not 5). Going through the shared abstraction would
       inherit that hard-coded 5 with no way to override it, so this calls
       Vertex directly instead — isolated, doesn't touch or risk the shared
       corpus-ingestion embedding path (embedding_worker.py etc.), which
       may have its own real reasons for batch_size=5 (rate limits, request
       size) that don't apply here and weren't investigated.

    Chunks at the real 250-instance ceiling, not an arbitrary smaller
    number — for the largest fanout domain seen live (health_care_services,
    293 codes), this is 2 calls, not 60.
    """
    global _clustering_model
    if _clustering_model is None:
        import vertexai
        from vertexai.language_models import TextEmbeddingModel

        from app.config import CLUSTERING_EMBEDDING_MODEL, VERTEX_LOCATION, VERTEX_PROJECT_ID

        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        _clustering_model = TextEmbeddingModel.from_pretrained(CLUSTERING_EMBEDDING_MODEL)

    from vertexai.language_models import TextEmbeddingInput

    out: list[list[float]] = []
    for i in range(0, len(texts), _MAX_INSTANCES_PER_EMBED_CALL):
        batch = texts[i : i + _MAX_INSTANCES_PER_EMBED_CALL]
        inputs = [TextEmbeddingInput(t, task_type="RETRIEVAL_DOCUMENT") for t in batch]
        resp = _clustering_model.get_embeddings(inputs)
        out.extend(list(r.values) for r in resp)
    return out


async def _embed_for_clustering_async(texts: list[str]) -> list[list[float]]:
    return await asyncio.to_thread(_embed_for_clustering, texts)


async def run_reformat(db: AsyncSession, gate: GateResult) -> ReformatResult:
    t0 = time.monotonic()
    result = await _dispatch(db, gate)
    result.reformat_ms = int((time.monotonic() - t0) * 1000)
    return result


async def _dispatch(db: AsyncSession, gate: GateResult) -> ReformatResult:
    if gate.contour == Contour.EXACT:
        # REVERTED 2026-07-29 (Ananth's direct call, live-trace evidence):
        # auto-firing _decompose_query_corpus() on every EXACT-contour query
        # (2026-07-29 earlier today) surfaced too much noise once tested
        # broadly across the 22-query bank, not just the hand-picked cases
        # that motivated it -- e.g. cmhc003 ("does Aetna require a referral
        # for OUTPATIENT behavioral health therapy") decomposed into
        # "hospital admission" (inpatient -- the OPPOSITE facet), "prior
        # auth", and "manual updates", none of which are real sub-facts of
        # the question. The co-occurrence signal finds tags that share a
        # document with the base topic, which is not the same as finding
        # genuine compound sub-facts -- confirmed unreliable at broad scale.
        # _decompose_query_corpus() and its FanoutTheme/Pool-threading
        # plumbing are left in place (still reachable via the
        # force_fanout_queries debug override in orchestrator.py, and the
        # trace-explorer bank runner's auto_fanout_corpus flag) for
        # continued manual testing, just no longer auto-invoked here.
        # Real lever going forward per this same investigation: BM25's
        # cover-density ranking (ts_rank_cd) rewarding repeated matches of
        # non-discriminating required terms (e.g. payer name) over a single
        # clear match of the real answer -- see filler_a.py's rerank weight
        # rebalance (2026-07-29) for the fix in progress on that front.
        return ReformatResult(
            query=gate.query,
            posture=ReformatPosture.PRECISE,
            rewritten_queries=[gate.normalized or gate.query],
            reason="EXACT contour — pass through unchanged, minimal work",
        )

    if gate.contour == Contour.UNDERSPECIFIED:
        if gate.underspecified_kind == "explore_siblings":
            return await _fan_out(db, gate)
        t0 = time.monotonic()
        questions = await _suggest_clarify_questions(db, gate)
        clarify_ms = int((time.monotonic() - t0) * 1000)
        return ReformatResult(
            query=gate.query,
            posture=ReformatPosture.CLARIFY,
            clarify_questions=questions,
            reason=(
                f"UNDERSPECIFIED/{gate.underspecified_kind} — no enumerable siblings, "
                "suggesting clarification instead of a blind ask"
            ),
            segment_ms={"clarify_lookup_ms": clarify_ms},
        )

    if gate.contour in (Contour.VICINITY, Contour.CORPUS_GAP):
        return ReformatResult(
            query=gate.query,
            posture=ReformatPosture.RELY_ON_EXTERNAL,
            external_reason=gate.contour.value,
            reason=f"{gate.contour.value} — internal coverage insufficient, defer to Router c/d",
        )

    if gate.contour == Contour.OUT_OF_SCOPE:
        return ReformatResult(
            query=gate.query,
            posture=ReformatPosture.DECLINE,
            decline_reason="out_of_scope",
            reason="OUT_OF_SCOPE — hard boundary, no fallback attempted",
        )

    # Contour.UNCLEAR — TENTATIVE default, not yet confirmed by Ananth. See
    # schematic spec §2/§6 and eval bank reformat013. Do not treat this branch
    # as locked; flagged explicitly rather than silently assumed correct.
    return ReformatResult(
        query=gate.query,
        posture=ReformatPosture.CLARIFY_REPHRASE,
        clarify_questions=["I didn't quite understand that — could you rephrase your question?"],
        reason="UNCLEAR — tentative CLARIFY_REPHRASE default, NOT yet confirmed by Ananth",
    )


# ---------------------------------------------------------------------------
# FAN_OUT — vector-clustered themes + catch-all
# ---------------------------------------------------------------------------


async def _fan_out(db: AsyncSession, gate: GateResult) -> ReformatResult:
    codes = gate.fanout_codes
    if not codes:
        # Defensive: Gate said explore_siblings but handed nothing to explore.
        # Shouldn't happen (Gate only sets this kind when fanout_codes is
        # populated) — fall back to CLARIFY rather than crash or fan out on
        # nothing.
        return ReformatResult(
            query=gate.query,
            posture=ReformatPosture.CLARIFY,
            clarify_questions=["Could you be more specific about what you're asking?"],
            reason="explore_siblings with empty fanout_codes — defensive fallback, not expected",
        )

    segment_ms: dict[str, int] = {}

    t0 = time.monotonic()
    phrase_by_code = await _fetch_lexicon_phrases(db, codes)
    segment_ms["lexicon_fetch_ms"] = int((time.monotonic() - t0) * 1000)

    valid_codes = [c for c in codes if phrase_by_code.get(c)]
    if not valid_codes:
        return ReformatResult(
            query=gate.query,
            posture=ReformatPosture.CLARIFY,
            clarify_questions=["Could you be more specific about what you're asking?"],
            reason="explore_siblings — no lexicon phrase text for any sibling code, can't cluster",
            segment_ms=segment_ms,
        )

    t0 = time.monotonic()
    texts_to_embed = [phrase_by_code[c] for c in valid_codes] + [gate.normalized or gate.query]
    vectors = await _embed_for_clustering_async(texts_to_embed)
    segment_ms["embed_ms"] = int((time.monotonic() - t0) * 1000)
    code_vecs = np.array(vectors[:-1], dtype=np.float64)
    query_vec = np.array(vectors[-1], dtype=np.float64)

    t0 = time.monotonic()
    n_clusters = min(_TARGET_THEME_COUNT, len(valid_codes))
    cluster_of = _agglomerative_cluster(code_vecs, n_clusters)
    segment_ms["clustering_ms"] = int((time.monotonic() - t0) * 1000)

    # ONE query for all candidate codes' doc-id sets, not one per theme —
    # see _prevalence_doc_ids_by_code's docstring for why (seq-scan finding).
    t0 = time.monotonic()
    doc_ids_by_code = await _prevalence_doc_ids_by_code(db, valid_codes)
    segment_ms["prevalence_ms"] = int((time.monotonic() - t0) * 1000)

    themes: list[FanoutTheme] = []
    for cluster_id in sorted(set(cluster_of)):
        member_idxs = [i for i, c in enumerate(cluster_of) if c == cluster_id]
        member_codes = [valid_codes[i] for i in member_idxs]
        prevalence = _union_prevalence(doc_ids_by_code, member_codes)
        centroid = code_vecs[member_idxs].mean(axis=0)
        proximity = _cosine(query_vec, centroid)
        # Label with whichever member is closest to the cluster centroid —
        # most representative of the group, not just whichever code sorted
        # first (that was the v1 approach; centroid-closest is barely more
        # code but meaningfully more honest about what the cluster actually is).
        closest_local = int(np.argmax([_cosine(code_vecs[i], centroid) for i in member_idxs]))
        label = _label_for_cluster([member_codes[closest_local]], phrase_by_code)
        themes.append(
            FanoutTheme(
                theme_label=label,
                member_codes=member_codes,
                prevalence_docs=prevalence,
                lexicon_proximity=proximity,
            )
        )

    max_prev = max((t.prevalence_docs for t in themes), default=0) or 1
    for t in themes:
        prevalence_norm = t.prevalence_docs / max_prev
        t.score = _W_PREVALENCE * prevalence_norm + _W_LEXICON_PROXIMITY * t.lexicon_proximity

    themes.sort(key=lambda t: -t.score)
    selected = themes[: min(_TARGET_THEME_COUNT, MAX_FANOUT_THEMES - 1)]

    catchall = FanoutTheme(
        theme_label="other considerations",
        member_codes=[],
        is_catchall=True,
    )
    all_themes = selected + [catchall]

    rewritten = [_build_theme_query(gate, t) for t in selected]
    rewritten.append(_build_catchall_query(gate, selected))

    return ReformatResult(
        query=gate.query,
        posture=ReformatPosture.FAN_OUT,
        rewritten_queries=rewritten,
        fanout_themes=all_themes,
        reason=(
            f"explore_siblings — {len(valid_codes)} candidate siblings clustered into "
            f"{len(selected)} theme(s) + 1 catch-all"
        ),
        segment_ms=segment_ms,
    )


async def _fetch_lexicon_phrases(db: AsyncSession, codes: list[str]) -> dict[str, str]:
    """Live lexicon lookup — never guess phrase text. Matches the
    verify-before-trust discipline Gate's build established."""
    if not codes:
        return {}
    rows = (
        await db.execute(
            text("SELECT code, spec FROM policy_lexicon_entries WHERE kind='d' AND code = ANY(:codes)"),
            {"codes": codes},
        )
    ).mappings()
    out: dict[str, str] = {}
    for row in rows:
        raw_spec = row["spec"]
        spec = json.loads(raw_spec) if isinstance(raw_spec, str) else (raw_spec or {})
        desc = spec.get("description") or ""
        phrases = spec.get("strong_phrases") or []
        text_blob = f"{desc}. {', '.join(phrases)}".strip(". ")
        if text_blob:
            out[row["code"]] = text_blob
    return out


async def _fetch_lexicon_short_labels(db: AsyncSession, codes: list[str]) -> dict[str, str]:
    """Short, literal query-friendly label per d-code -- deliberately NOT
    `_fetch_lexicon_phrases` above, which concatenates the FULL description +
    every strong_phrase into one paragraph-sized blob meant for
    embedding-based clustering (that's fine for _fan_out's vector similarity
    use, wrong for a literal BM25 sub-query string). Confirmed live
    (2026-07-29): dumping the whole blob into a sub-query diluted BM25 enough
    that recall stayed flat despite the right themes being selected.

    Two label-quality bugs found + fixed live after that (2026-07-29,
    Ananth's catch on cmhc022): the ORIGINAL "shortest strong_phrase" rule
    picked whatever string was fewest characters with zero regard for
    whether it read as English -- health_care_services.dental's shortest
    phrase is literally "gum" (from its ['dental', ..., 'gum', 'toothpaste']
    list), behavioral_health's is the acronym "bh". The underlying CODE
    selection (selectivity-weighted co-occurrence) was correct -- dental /
    behavioral health / healthy_start are genuinely EPSDT-adjacent Florida
    Medicaid child-health domains -- only the label rendering made it look
    like nonsense. Fixed by preferring the shortest MULTI-WORD phrase (reads
    as real English, e.g. "dental care" / "mental health") and only falling
    back to a single word if no multi-word phrase exists for that code.
    Second bug: a code with NO strong_phrases at all (e.g. d:disputes) fell
    back to `description.split('.')[0]` -- for a run-on description with no
    early period, that grabbed the ENTIRE description verbatim as a
    "label" (confirmed live on cmhc007). Capped at 60 chars now."""
    if not codes:
        return {}
    rows = (
        await db.execute(
            text("SELECT code, spec FROM policy_lexicon_entries WHERE kind='d' AND code = ANY(:codes)"),
            {"codes": codes},
        )
    ).mappings()
    out: dict[str, str] = {}
    for row in rows:
        raw_spec = row["spec"]
        spec = json.loads(raw_spec) if isinstance(raw_spec, str) else (raw_spec or {})
        phrases = [p for p in (spec.get("strong_phrases") or []) if p]
        multiword = [p for p in phrases if " " in p.strip()]
        if multiword:
            out[row["code"]] = min(multiword, key=len)
        elif phrases:
            out[row["code"]] = min(phrases, key=len)
        else:
            desc = (spec.get("description") or "").split(".")[0].strip()
            if desc:
                out[row["code"]] = desc[:60].rsplit(" ", 1)[0] if len(desc) > 60 else desc
    return out


_DECOMPOSE_TOP_N = 3


async def _decompose_query_corpus(
    db: AsyncSession, gate: GateResult,
) -> tuple[list[str], list["FanoutTheme"]] | None:
    """Corpus-grounded compound-fact decomposition for EXACT-contour queries
    (2026-07-29, Ananth's alternative to an LLM-dimension-guessing approach):
    instead of asking an LLM what facets matter, find tags that ACTUALLY
    CO-OCCUR with Gate's matched codes on the same real chunks --
    deterministic, one DB round trip, no LLM call, grounded in what's
    actually in the corpus.

    Motivating case: "What is the timely filing deadline for Sunshine
    Health FL Medicaid claims?" -- EXACT contour passes this through
    unchanged, but the golden answer's bonus facts (appeals/reconsideration,
    EOB, coordination of benefits) live on DIFFERENT chunks than the base
    timely-filing chunks, so a single PRECISE query never retrieves them.

    Two real bugs found and fixed live before this worked (kept here so the
    fix isn't silently re-broken):
    1. ANDing ALL of Gate's matched d/j codes together to find the
       intersection returns ZERO chunks in practice (confirmed: 6 codes
       simultaneously required -> 0 rows) -- not every code Gate matched is
       co-tagged on the very same chunk, even when the whole DOCUMENT is
       clearly about all of them. Fixed by using only the single most
       SELECTIVE d-code and j-code (via the same selectivity_for_tag()
       Pool already uses for phrase_buckets) as the intersection.
    2. Ranking co-occurring tags by raw count surfaces corpus-wide noise --
       confirmed live: "provider.general"/"health_care_services.dental"
       dominated raw counts purely because they're common EVERYWHERE
       (2.5-3.3% of the whole 1.9M-chunk corpus). Fixed by weighting
       (co-occurrence fraction within the intersection) * (the tag's own
       corpus-wide selectivity), not raw count alone.
    3. `_fetch_lexicon_phrases`'s full description+strong_phrases blob,
       reused naively for sub-query text, diluted BM25 enough to keep
       recall flat despite selecting the right themes. Fixed by
       `_fetch_lexicon_short_labels` above (shortest strong_phrase only).

    Validated 2026-07-29 against the full 22-query eval bank via the
    debug force_fanout_queries override: avg recall 0.407 -> 0.441,
    ZERO per-query regressions (4 improved, 18 flat). Known limitation,
    NOT fixed here (separate lexicon-tagging gap, flagged out of scope):
    a fact that's ONLY body-text with no corresponding tag at all (e.g.
    participating/non-participating provider status) won't surface this
    way regardless of decomposition quality.

    Returns None (caller keeps plain PRECISE) if there are too few
    co-tagged chunks to trust the signal, or no co-occurring tags found.
    """
    from collections import Counter
    import asyncio
    from app.services.corpus_search_agent import selectivity_for_tag

    if not gate.d_codes:
        return None

    # Latency fix (2026-07-29, Ananth's catch): these were sequential await
    # loops -- each selectivity_for_tag() is up to 2 DB round trips when
    # uncached, so a cold cache turned "look up ~20 codes' selectivity"
    # into ~20-40 serialized round trips, measured live contributing
    # multiple seconds to reformat_ms alone. Gathered concurrently instead
    # -- independent lookups, no shared state.
    d_codes_list = list(gate.d_codes)
    j_codes_list = list(gate.j_codes or [])
    d_sel_vals, j_sel_vals = await asyncio.gather(
        asyncio.gather(*[selectivity_for_tag(db, c) for c in d_codes_list]),
        asyncio.gather(*[selectivity_for_tag(db, c) for c in j_codes_list]),
    )
    d_sel = dict(zip(d_codes_list, d_sel_vals))
    j_sel = dict(zip(j_codes_list, j_sel_vals))
    best_d = max(d_sel, key=d_sel.get).split(":", 1)[1]
    best_j = max(j_sel, key=j_sel.get).split(":", 1)[1] if j_sel else None

    params = {"d": best_d}
    where = "chunk_d_tags ? :d"
    if best_j:
        where += " AND chunk_j_tags ? :j"
        params["j"] = best_j
    rows = (await db.execute(
        text(f"SELECT chunk_d_tags FROM rag_published_embeddings WHERE {where}"), params,
    )).fetchall()
    if len(rows) < 3:
        return None

    co = Counter()
    for r in rows:
        for k in (r.chunk_d_tags or {}).keys():
            if k != best_d:
                co[k] += 1
    if not co:
        return None

    candidate_tags = [t for t, _n in co.most_common(15)]  # cap the selectivity-lookup fanout
    candidate_sels = await asyncio.gather(*[
        selectivity_for_tag(db, f"d:{tag}") for tag in candidate_tags
    ])
    weighted = [
        (tag, (co[tag] / len(rows)) * sel)
        for tag, sel in zip(candidate_tags, candidate_sels)
    ]
    weighted.sort(key=lambda pair: pair[1], reverse=True)
    top_tags = [tag for tag, score in weighted[:_DECOMPOSE_TOP_N] if score > 0]
    if not top_tags:
        return None

    score_by_tag = dict(weighted)
    label_by_code = await _fetch_lexicon_short_labels(db, top_tags)
    base_query = gate.normalized or gate.query
    sub_queries = [
        f"{base_query} — specifically regarding {label_by_code.get(tag, tag.replace('.', ' ').replace('_', ' '))}"
        for tag in top_tags
    ]
    themes = [
        FanoutTheme(
            theme_label=label_by_code.get(tag, tag),
            # UNION with the base topic's own d_codes, not a bare
            # replacement -- Pool.run_pool_fanout uses member_codes to
            # override d_codes per-slot (2026-07-29 fix), and this
            # mechanism's sub-queries are explicitly "the base topic, AND
            # also this co-occurring theme" (e.g. timely filing + appeals),
            # not a wholly different topic the way explore_siblings' theme
            # codes are. Losing the base d_codes here would let this slot's
            # tag_select/inherited/phrase_buckets arms drift entirely off
            # the base topic.
            member_codes=[*gate.d_codes, f"d:{tag}"],
            score=score_by_tag.get(tag, 0.0),
        )
        for tag in top_tags
    ]
    return sub_queries, themes


async def _prevalence_doc_ids_by_code(db: AsyncSession, codes: list[str]) -> dict[str, set]:
    """Per-code sets of matching document_ids, ONE round trip, ONE query.

    REPLACED 2026-07-23 (Retriever's live re-verification caught this):
    the original per-THEME approach OR-chained up to 32 `d_tags ? :code`
    predicates into one WHERE clause per call (3 calls per FAN_OUT). Real
    problem, confirmed via EXPLAIN (ANALYZE, BUFFERS) — 3735-4557ms of
    GENUINE server-side Seq Scan per call (not a proxy artifact, verified
    against a 179ms raw ping): the GIN index (`ix_document_tags_d_tags_gin`)
    isn't used once an OR-chain crosses this many predicates, same
    seq-scan-fallback condition gate.py documents at its own ~12-code cap,
    just reliably triggered here since FAN_OUT clusters run 17-32 codes.

    Fix: UNION ALL of single-key `d_tags ? :code` subqueries — each branch
    is a single predicate, cheap and GIN-indexed (Bitmap Index Scan,
    confirmed live: ~8ms per branch), batched into ONE round trip via
    UNION ALL rather than one round trip per code. Measured live for 80
    codes: ~1.06s total (vs ~11s+ for the old 3-call OR-chain approach) —
    fetches document_id (not just COUNT) so the caller can compute an exact
    per-theme UNION via Python set union, not sum-of-per-code-counts (which
    would double-count any document tagged with 2+ codes from the same
    theme — a real correctness difference, not just a style choice).
    """
    if not codes:
        return {}
    parts = []
    params: dict[str, str] = {}
    for i, code in enumerate(codes):
        pname = f"c{i}"
        params[pname] = code
        parts.append(f"SELECT :{pname} AS code, document_id FROM document_tags WHERE d_tags ? :{pname}")
    sql = text(" UNION ALL ".join(parts))
    rows = (await db.execute(sql, params)).mappings().all()
    out: dict[str, set] = {c: set() for c in codes}
    for row in rows:
        out[row["code"]].add(row["document_id"])
    return out


def _union_prevalence(doc_ids_by_code: dict[str, set], member_codes: list[str]) -> int:
    """Exact distinct-document count across a theme's member codes."""
    union: set = set()
    for c in member_codes:
        union |= doc_ids_by_code.get(c, set())
    return len(union)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def _agglomerative_cluster(vectors: np.ndarray, n_clusters: int, seed: int = 0) -> list[int]:
    """Spherical k-means (cosine similarity, unit-normalized vectors), pure
    numpy — sklearn/scipy aren't installed in this environment (checked
    live before writing this), so this is hand-rolled rather than adding a
    new dependency without review.

    REPLACED average-linkage agglomerative clustering here 2026-07-23 after
    a live run on real "eligibility" data (80 siblings) collapsed 78 of 80
    into one cluster, leaving two near-meaningless singletons — a known
    failure mode of average-linkage on high-dimensional text embeddings
    that sit in a fairly uniform similarity neighborhood (same-domain
    lexicon phrases share vocabulary, so most pairwise similarities are
    high and undifferentiated, and average-linkage chains everything into
    one blob once it starts absorbing nearby points). k-means partitions
    directly for balance instead of greedily merging nearest pairs.

    k-means++-style init (spread out initial centroids rather than random)
    + a fixed small iteration cap — this is a bounded clustering problem
    (n <= ~300, k <= 3), not a case that needs convergence tolerance
    tuning.
    """
    n = len(vectors)
    if n <= n_clusters:
        return list(range(n))

    norms = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(seed)

    # k-means++ init: first centroid random, subsequent ones weighted by
    # squared distance to the nearest already-chosen centroid (spreads
    # centroids apart instead of risking two random picks landing in the
    # same blob).
    centroid_idxs = [int(rng.integers(0, n))]
    for _ in range(n_clusters - 1):
        chosen = norms[centroid_idxs]
        sims = norms @ chosen.T  # (n, len(chosen))
        nearest_sim = sims.max(axis=1)
        dist = np.clip(1.0 - nearest_sim, 0.0, None)
        probs = dist**2
        total = probs.sum()
        probs = probs / total if total > 1e-9 else np.ones(n) / n
        centroid_idxs.append(int(rng.choice(n, p=probs)))
    centroids = norms[centroid_idxs].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(25):
        sims = norms @ centroids.T
        new_labels = np.argmax(sims, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for k in range(n_clusters):
            members = norms[labels == k]
            if len(members) == 0:
                # empty cluster — reseed on the point farthest from its own centroid
                sims_to_own = sims[np.arange(n), labels]
                worst = int(np.argmin(sims_to_own))
                centroids[k] = norms[worst]
                labels[worst] = k
                continue
            mean = members.mean(axis=0)
            mnorm = np.linalg.norm(mean)
            centroids[k] = mean / mnorm if mnorm > 1e-9 else mean

    return labels.tolist()


def _label_for_cluster(member_codes: list[str], phrase_by_code: dict[str, str]) -> str:
    """Caller passes a single-element list — the cluster's centroid-closest
    member (see _fan_out) — so this just extracts and trims that one
    member's lexicon description. Not a guessed/hardcoded name."""
    if not member_codes:
        return "other considerations"
    first = phrase_by_code.get(member_codes[0], member_codes[0])
    return first.split(".")[0].strip() or member_codes[0]


def _build_theme_query(gate: GateResult, theme: FanoutTheme) -> str:
    """Template-based construction, v1 — not LLM-rewritten. Root description
    + theme label + jurisdiction context if present. Documented as a known
    simplification, not a hidden one."""
    root_desc = gate.normalized or gate.query
    j_suffix = f", for {' / '.join(gate.j_codes)}" if gate.j_codes else ""
    return f"{root_desc} — {theme.theme_label}{j_suffix}"


def _build_catchall_query(gate: GateResult, selected: list[FanoutTheme]) -> str:
    """Deliberately NOT corpus/lexicon-derived (Ananth, 2026-07-23) — an open
    'what else' question, phrased to route to Router strategy c/d (LLM
    synthesis + Vertex-grounded search) rather than internal Pool search."""
    covered = ", ".join(t.theme_label for t in selected) or "the topics already covered"
    root_desc = gate.normalized or gate.query
    return f"Beyond {covered}, what other aspects of {root_desc} exist?"


async def _suggest_clarify_questions(db: AsyncSession, gate: GateResult) -> list[str]:
    """v1: query real co-occurring tags rather than guess. missing_jurisdiction
    → top J-codes seen alongside the matched D/P codes. missing_domain → top
    D-codes seen alongside the matched J/P codes. Falls back to a generic
    prompt if nothing co-occurs (thin data, not a bug)."""
    if gate.underspecified_kind == "missing_jurisdiction" and (gate.d_codes or gate.p_codes):
        anchor_codes = gate.d_codes or gate.p_codes
        anchor_kind = "d" if gate.d_codes else "p"
        rows = await _top_cooccurring(db, anchor_kind, anchor_codes, "j_tags", limit=3)
        if rows:
            options = " / ".join(rows)
            return [f"Which jurisdiction did you mean — {options}?"]
        return ["Which state or jurisdiction did you mean?"]

    if gate.underspecified_kind == "missing_domain" and gate.j_codes:
        rows = await _top_cooccurring(db, "j", gate.j_codes, "d_tags", limit=3)
        if rows:
            options = " / ".join(rows)
            return [f"Could you clarify the topic — did you mean {options}?"]
        return ["Could you clarify what topic this relates to?"]

    return ["Could you clarify what topic this relates to?"]


async def _top_cooccurring(
    db: AsyncSession, anchor_kind: str, anchor_codes: list[str], target_col: str, limit: int
) -> list[str]:
    """Real DB query — top codes in target_col co-occurring with any anchor
    code in {anchor_kind}_tags. Live, not guessed."""
    if not anchor_codes:
        return []
    params: dict[str, str | int] = {}
    exprs = []
    for i, code in enumerate(anchor_codes):
        pname = f"a{i}"
        params[pname] = code
        exprs.append(f"{anchor_kind}_tags ? :{pname}")
    where = " OR ".join(exprs)
    params["lim"] = limit
    sql = text(
        f"SELECT jsonb_object_keys({target_col}) AS code, COUNT(*) AS n "
        f"FROM document_tags WHERE ({where}) AND {target_col} IS NOT NULL "
        f"GROUP BY code ORDER BY n DESC LIMIT :lim"
    )
    rows = (await db.execute(sql, params)).mappings().all()
    return [r["code"] for r in rows]
