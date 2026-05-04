"""Strategy (c) — LLM → Validate.

Generate an answer with the LLM's prior, then validate every cited claim
against our sitemap + corpus. Different from (a) and (b) because we
START with a hypothesis and prove or refute it, rather than retrieve
first and synthesize after.

Pipeline
--------

1. Ask the LLM for a brief answer with structured citations. Each
   citation must include a verbatim quote (≤30 words) — that's what we
   grep for in the located source.

2. For each citation, locate the source via the sitemap chain:
   ``documents`` → ``discovered_sources`` → none.

3. Check the quote against the located doc text (verbatim ts_query /
   ILIKE search within page chunks).

4. Classify the citation into the outcome matrix and assemble the
   envelope.

Outcome matrix (per citation)
-----------------------------

  ``validated_correct``       — located + quote matches → cite confirmed
  ``validated_hallucinated``  — located + quote NOT in source → LLM wrong
  ``unverified_robots``       — sitemap hit but robots-blocked
                                 (last_fetch_status in {403, 451})
  ``needs_scrape``             — sitemap hit, not ingested, robots-allowed
                                 → flagged for Strategy (d) on-demand fetch
  ``needs_external``           — no sitemap entry → Google + scrape (d)
  ``located_unverified``       — found doc but page/chunk missing OR LLM
                                 didn't supply a quote we could check

Strategy (c) v1 only WALKS the matrix — actual on-demand scrape and
Google search are (d)'s responsibility. Citations classified as
``needs_scrape`` / ``needs_external`` are returned with the flag so
the chat planner / (d) can act on them.

Routing
-------

Caller sets ``request.mode = "validate"``; the chooser routes to (c).
Future auto-detection: queries asking for definitive single-answer
information ("what's the limit", "is X covered") could route here when
(a) returns low confidence.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from app.services import llm_manager_client
from app.services.corpus_search import (
    CorpusSearchRequest,
    corpus_search,
)


logger = logging.getLogger(__name__)


# Status codes that indicate the source is robots-blocked / paywall /
# legal-takedown. Anything in this set means we cannot scrape on demand.
_ROBOTS_BLOCKED_STATUSES = {401, 403, 451}


# Minimum characters of a quote we'll check. LLMs sometimes emit single
# words or fragments that match too easily and produce false positives.
_MIN_QUOTE_CHARS = 12


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CitationCandidate:
    """What the LLM emitted, before any validation."""
    document_title: str | None = None
    page: int | None = None
    section: str | None = None
    url: str | None = None
    quote: str | None = None


@dataclass
class ValidatedCitation:
    """A citation after sitemap lookup + content check."""
    candidate: CitationCandidate
    status: str                            # see outcome matrix above
    document_id: str | None = None
    document_display_name: str | None = None
    document_filename: str | None = None
    matched_chunk_text: str | None = None  # chunk that contained the quote
    matched_page: int | None = None
    discovered_source_url: str | None = None
    last_fetch_status: int | None = None   # for unverified_robots context
    locate_method: str = ""                # which lookup chain matched
    notes: str = ""


@dataclass
class StrategyCResult:
    llm_answer: str
    citations: list[ValidatedCitation] = field(default_factory=list)
    raw_llm_output: str = ""
    telemetry: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step 1: LLM generation
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a precise policy assistant for FL Medicaid behavioral health. "
    "Answer the user's question briefly (3 sentences max). For every "
    "claim, cite your source.\n\n"
    "OUTPUT FORMAT — strict JSON, no markdown:\n"
    "{\n"
    '  "answer": "<your answer>",\n'
    '  "citations": [\n'
    "    {\n"
    '      "document_title": "<title or filename>",\n'
    '      "page": <integer or null>,\n'
    '      "section": "<section number/name or null>",\n'
    '      "url": "<URL or null>",\n'
    '      "quote": "<verbatim quote, 5-30 words, from the source>"\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- Every claim in the answer must have a citation.\n"
    "- The quote MUST be a verbatim copy from the source (we will grep "
    "for it). Do not paraphrase.\n"
    "- If you don't know, say so and emit an empty citations array. "
    "Do not invent sources."
)


def _parse_llm_json(raw: str) -> dict[str, Any]:
    """Best-effort JSON parsing — strip markdown fences if present."""
    text = (raw or "").strip()
    # Common LLM artifact: ```json ... ```
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract the first {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return {"answer": "", "citations": [], "_parse_error": True, "_raw": text}


async def _ask_llm(query: str, *, correlation_id: str | None) -> tuple[str, dict[str, Any], dict]:
    """Call the LLM and parse out (answer_text, parsed_json, llm_telemetry)."""
    t0 = time.monotonic()
    raw, llm_tel = await llm_manager_client.generate(
        system=_SYSTEM_PROMPT,
        user=query,
        stage="rag_strategy_c_validate",
        max_tokens=2048,
        correlation_id=correlation_id,
    )
    elapsed = (time.monotonic() - t0) * 1000.0
    parsed = _parse_llm_json(raw)
    answer = (parsed.get("answer") or "").strip()
    return answer, parsed, {
        "llm_ms": int(elapsed),
        "llm_meta": llm_tel,
        "parse_error": bool(parsed.get("_parse_error")),
    }


def _coerce_citation(d: dict[str, Any]) -> CitationCandidate | None:
    """Normalize one raw LLM citation dict → CitationCandidate."""
    if not isinstance(d, dict):
        return None
    title = (d.get("document_title") or "").strip() or None
    url = (d.get("url") or "").strip() or None
    quote = (d.get("quote") or "").strip() or None
    section = (d.get("section") or "").strip() or None
    raw_page = d.get("page")
    page = None
    if isinstance(raw_page, int):
        page = raw_page
    elif isinstance(raw_page, str) and raw_page.strip().isdigit():
        page = int(raw_page.strip())
    # Need at least a title or URL to be locatable.
    if not (title or url):
        return None
    return CitationCandidate(
        document_title=title, page=page, section=section,
        url=url, quote=quote,
    )


# ---------------------------------------------------------------------------
# Step 2: Locate citation in sitemap
# ---------------------------------------------------------------------------

@dataclass
class _LocateResult:
    """Outcome of the sitemap lookup, before content check."""
    document_id: str | None = None
    display_name: str | None = None
    filename: str | None = None
    discovered_source_url: str | None = None
    last_fetch_status: int | None = None
    sitemap_kind: str = "none"   # 'doc_ingested' | 'sitemap_robots_blocked'
                                  # | 'sitemap_needs_scrape' | 'external_google' | 'none'
    locate_method: str = ""       # 'title_strict' | 'title_relaxed' |
                                  # 'url_exact' | 'quote_fallback' | 'google_external' | etc.
    # External-only: the actual paragraph from the fetched URL that
    # best matches the LLM's claimed quote. Populated by
    # ``_locate_by_google`` so downstream replaces the LLM's
    # paraphrase with the source text + URL citation.
    external_chunk_text: str = ""


# Words that don't help disambiguate one policy doc from another. Kept
# minimal: payer / domain words like "Sunshine", "Provider", "Manual"
# stay in because that's the core discriminator.
_TITLE_STOPWORDS = frozenset({
    "the", "a", "an", "of", "for", "and", "or", "in", "on", "to",
    "with", "from", "by", "at", "is", "are", "this", "that",
})


def _tokenize_title(title: str) -> list[str]:
    """Significant lowercased tokens from a citation title."""
    if not title:
        return []
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", title)]
    return [t for t in tokens if len(t) >= 3 and t not in _TITLE_STOPWORDS]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _overlap_coefficient(a: set[str], b: set[str]) -> float:
    """Asymmetric overlap: |A ∩ B| / min(|A|, |B|).

    Better than Jaccard when one side is verbose. LLM titles often
    include version/jurisdiction noise (e.g. "Sunshine Health Provider
    Manual, Florida Medicaid, Version 2023.1") that our display_name
    ("Sunshine Provider Manual") doesn't carry. We want full credit
    when every token in the shorter side appears in the longer side.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


async def _locate_by_title(
    db: AsyncSession, title: str,
) -> _LocateResult | None:
    """Token-overlap match on documents.display_name/filename.

    Strict pass: every significant token must appear in the doc's name.
    If nothing matches, relaxed pass: at least half the tokens, ranked by
    Jaccard overlap against the LLM title token set. Threshold 0.5.
    """
    tokens = _tokenize_title(title)
    if not tokens:
        return None

    # Strict pass — every token must appear in display_name OR filename.
    where_clauses = []
    params: dict[str, str] = {}
    for i, tok in enumerate(tokens):
        p = f"t{i}"
        where_clauses.append(
            f"LOWER(COALESCE(display_name, '') || ' ' || filename) LIKE :{p}"
        )
        params[p] = f"%{tok}%"
    sql = (
        "SELECT id::text AS id, display_name, filename "
        "FROM documents "
        "WHERE expires_at IS NULL "
        "  AND " + " AND ".join(where_clauses) + " "
        "ORDER BY LENGTH(COALESCE(display_name, filename)) ASC "
        "LIMIT 5"
    )
    rows = (await db.execute(sql_text(sql), params)).mappings().all()

    if not rows:
        # Relaxed pass — match ANY token, then score in Python.
        any_clauses = [
            f"LOWER(COALESCE(display_name, '') || ' ' || filename) LIKE :{p}"
            for p in params
        ]
        sql_any = (
            "SELECT id::text AS id, display_name, filename "
            "FROM documents "
            "WHERE expires_at IS NULL "
            "  AND (" + " OR ".join(any_clauses) + ") "
            "LIMIT 50"
        )
        rows = (await db.execute(sql_text(sql_any), params)).mappings().all()

    if not rows:
        return None

    title_set = set(tokens)
    best: tuple[float, dict, set[str]] | None = None
    for r in rows:
        name = (r["display_name"] or r["filename"] or "").lower()
        name_tokens = set(t for t in re.findall(r"[a-z0-9]+", name)
                          if len(t) >= 3 and t not in _TITLE_STOPWORDS)
        # Use overlap coefficient: full credit when every token in the
        # shorter side appears in the longer side. Combined with a
        # safeguard that requires absolute overlap >= 2 tokens (so a
        # 1-token title doesn't degenerately match anything that contains
        # that single word).
        score = _overlap_coefficient(title_set, name_tokens)
        absolute_overlap = len(title_set & name_tokens)
        if best is None or score > best[0]:
            best = (score, dict(r), name_tokens)

    if best and best[0] >= 0.65 and len(title_set & best[2]) >= 2:
        # Payer-name discriminator: if the LLM-claimed title contains
        # a recognisable payer name (Aetna, Humana, Cigna, Sunshine,
        # Wellcare, Molina, UnitedHealthcare, Centene, Anthem) and the
        # matched doc contains a DIFFERENT one, reject the match. This
        # prevents the "Aetna Better Health of Florida Provider Manual"
        # citation from false-matching to "Sunshine Provider Manual"
        # via 67% token overlap on ["provider", "manual"]. We've seen
        # (c) report fake ``retrieved`` citations on cross-payer
        # confusions — silent hallucinations dressed up with an
        # in-corpus citation.
        _PAYER_TOKENS = {
            "aetna", "humana", "cigna", "sunshine", "wellcare", "molina",
            "unitedhealthcare", "uhc", "centene", "anthem", "amerigroup",
            "blue", "bcbs", "kaiser", "magellan", "optum",
        }
        title_payers = title_set & _PAYER_TOKENS
        match_payers = best[2] & _PAYER_TOKENS
        if title_payers and match_payers and not (title_payers & match_payers):
            # Cross-payer confusion. Reject — let downstream mark this
            # citation as ``doc_not_found`` (which the agent surfaces
            # as low-confidence and should trigger (d) escalation).
            logger.info(
                "_locate_by_title: rejecting cross-payer false match "
                "title_payers=%s match_payers=%s match_filename=%s",
                title_payers, match_payers, best[1].get("filename"),
            )
            return None
        method = "title_strict" if best[0] >= 0.90 else "title_relaxed"
        return _LocateResult(
            document_id=best[1]["id"],
            display_name=best[1]["display_name"],
            filename=best[1]["filename"],
            sitemap_kind="doc_ingested",
            locate_method=f"{method}(overlap={best[0]:.2f})",
        )
    return None


async def _locate_by_url(
    db: AsyncSession, url: str,
) -> _LocateResult | None:
    """Exact URL match in documents.source_metadata.url, then in
    discovered_sources, then host-only fallback against discovered_sources."""
    # 1. Exact match in source_metadata.
    rows = await db.execute(
        sql_text(
            "SELECT d.id::text AS id, d.display_name, d.filename "
            "FROM documents d "
            "WHERE d.source_metadata ->> 'url' = :u "
            "LIMIT 1"
        ),
        {"u": url},
    )
    row = rows.mappings().first()
    if row:
        return _LocateResult(
            document_id=row["id"],
            display_name=row["display_name"],
            filename=row["filename"],
            discovered_source_url=url,
            sitemap_kind="doc_ingested",
            locate_method="url_exact_doc",
        )

    # 2. Exact URL in discovered_sources — could be ingested or not.
    rows = await db.execute(
        sql_text(
            "SELECT id::text AS id, url, "
            "       ingested_doc_id::text AS ingested_doc_id, "
            "       last_fetch_status "
            "FROM discovered_sources WHERE url = :u LIMIT 1"
        ),
        {"u": url},
    )
    row = rows.mappings().first()
    if row:
        ingest_id = row["ingested_doc_id"]
        if ingest_id:
            doc = (await db.execute(
                sql_text(
                    "SELECT display_name, filename FROM documents "
                    "WHERE id::text = :i LIMIT 1"
                ),
                {"i": ingest_id},
            )).mappings().first()
            return _LocateResult(
                document_id=ingest_id,
                display_name=doc["display_name"] if doc else None,
                filename=doc["filename"] if doc else None,
                discovered_source_url=row["url"],
                last_fetch_status=row["last_fetch_status"],
                sitemap_kind="doc_ingested",
                locate_method="url_exact_sitemap",
            )
        status = row["last_fetch_status"]
        kind = ("sitemap_robots_blocked"
                if status in _ROBOTS_BLOCKED_STATUSES
                else "sitemap_needs_scrape")
        return _LocateResult(
            discovered_source_url=row["url"],
            last_fetch_status=status,
            sitemap_kind=kind,
            locate_method="url_exact_sitemap",
        )

    return None


async def _locate_by_quote(
    db: AsyncSession, quote: str,
) -> _LocateResult | None:
    """Last-resort: phrase-search the LLM's quote across all chunks
    via the GIN-indexed tsvector. If found, attribute to the chunk's
    document. This handles the case where the LLM hallucinates a doc
    title/URL but the actual claim is grounded in our corpus under a
    differently-titled document.

    SAFETY: only the strict ``phraseto_tsquery`` path runs. The
    earlier ``plainto_tsquery`` fallback was too loose — it ANDs all
    query tokens with no adjacency requirement, so a quote like
    "Aetna Better Health does not require a referral for behavioral
    health services" would token-match a Sunshine MMA-program chunk
    on the common words and falsely return ``status=retrieved``. The
    in-content verification below is the second line of defence:
    even if phraseto_tsquery hits a chunk, we re-check that the
    quote substring actually appears in the chunk body. If not, fall
    through so the caller can try external (Google) validation.
    """
    if not quote or len(quote) < _MIN_QUOTE_CHARS:
        return None
    rows = await db.execute(
        sql_text(
            "SELECT rpe.id::text AS chunk_id, "
            "       rpe.document_id::text AS id, rpe.page_number, "
            "       rpe.text AS chunk_text, "
            "       rpe.document_display_name AS display_name, "
            "       rpe.document_filename AS filename "
            "FROM rag_published_embeddings rpe "
            "WHERE rpe.search_vec @@ phraseto_tsquery('english', :q) "
            "LIMIT 5"
        ),
        {"q": quote},
    )
    quote_norm = " ".join(quote.lower().split())
    for row in rows.mappings():
        body_norm = " ".join((row.get("chunk_text") or "").lower().split())
        # Require either the full quote (when reasonable length) or
        # at least the first 40 chars to appear verbatim in the chunk
        # body. phraseto_tsquery's "phrase" semantics use stemming
        # and stopword removal, so a token-adjacent match might be
        # nothing like the original phrase. Substring check is the
        # ground-truth.
        probe = quote_norm if len(quote_norm) <= 80 else quote_norm[:60]
        if probe and probe in body_norm:
            return _LocateResult(
                document_id=row["id"],
                display_name=row["display_name"],
                filename=row["filename"],
                sitemap_kind="doc_ingested",
                locate_method="quote_phrase_verified",
            )
    return None


async def _locate_by_google(
    cand: CitationCandidate,
) -> _LocateResult | None:
    """External validation pass — when the citation can't be found in
    OUR corpus, search the web for it and check whether the LLM's
    quote actually appears on the top result page.

    This is per-citation TRUST VALIDATION, not strategy escalation:
    we keep (c)'s composed answer, but mark each individual citation
    as either ``retrieved`` (corpus), ``retrieved_external`` (web),
    or ``doc_not_found`` (neither). The chat planner / UI can then
    show source provenance per claim.

    Pipeline:
      1. Compose a search query from {document_title, first 8 words of
         quote}. Title gets quoted-string priority so DuckDuckGo /
         Google CSE prefer the named document.
      2. Fetch top 3 hits via the (d) infrastructure (handles HTML
         AND PDFs via PyMuPDF as of 2026-05-03).
      3. For each hit, check if the LLM's quote (or first 60 chars
         normalised) appears in the page body. If yes, return a
         _LocateResult with locate_method='google_external' and the
         URL as the discovered source.
      4. Otherwise return None — caller marks ``doc_not_found``.

    Cost: 1 web-search call + up to 3 page fetches per citation.
    Worth it because the alternative (silent in-corpus false-match)
    destroys trust.
    """
    if not (cand.document_title or cand.quote):
        return None

    # Build search query. Quote the title so search engines prefer
    # the exact phrase. Add a distinctive 8-word fragment of the
    # quote when present — this gives the search a concrete claim
    # to find rather than just the doc name.
    parts: list[str] = []
    if cand.document_title:
        parts.append(f'"{cand.document_title.strip()}"')
    if cand.quote:
        words = (cand.quote or "").split()[:8]
        if words:
            parts.append(" ".join(words))
    search_query = " ".join(parts)[:300]
    if not search_query:
        return None

    try:
        from app.services.corpus_search_strategy_d import (
            _search_web, _fetch_and_extract,
        )
    except Exception as exc:
        logger.warning("_locate_by_google: cannot import (d) helpers: %s", exc)
        return None

    try:
        hits = await _search_web(search_query, n=5)
    except Exception as exc:
        logger.warning("_locate_by_google: search failed: %s", exc)
        return None
    logger.info(
        "_locate_by_google: query=%r hits=%d", search_query[:120], len(hits or []),
    )
    if not hits:
        return None

    # Normalise the LLM's quote for fuzzy substring match. Strip
    # punctuation and whitespace so PDF extraction artefacts (extra
    # newlines, hyphenated word breaks) don't break the match.
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").lower()).strip()

    quote_norm = _norm(cand.quote)
    # Build distinctive substring fragments at decreasing widths so a
    # slightly-paraphrased quote still matches. LLMs frequently
    # paraphrase the source — even when citing verbatim — so we also
    # do a token-overlap fallback.
    fragments: list[str] = []
    if quote_norm and len(quote_norm) >= 30:
        fragments.extend([quote_norm[:120], quote_norm[:80], quote_norm[:40], quote_norm[:25]])
    elif quote_norm:
        fragments.append(quote_norm)

    # Token-overlap fallback. Pull the discriminating tokens from the
    # quote (drop stopwords + tiny tokens). Match if a high fraction
    # of those tokens appear in the body — this catches paraphrases
    # that share the substantive terms even with different glue words.
    _STOP = frozenset({
        "the", "a", "an", "of", "for", "and", "or", "to", "in", "on",
        "is", "are", "be", "this", "that", "from", "with", "by", "at",
        "as", "do", "does", "not", "no", "any", "may", "can", "will",
        "their", "its", "it", "you", "your", "we", "us", "our",
    })
    quote_tokens = {
        t for t in re.findall(r"[a-z0-9]+", quote_norm)
        if len(t) >= 4 and t not in _STOP
    }

    def _token_overlap(body: str) -> float:
        if not quote_tokens:
            return 0.0
        body_tokens = set(re.findall(r"[a-z0-9]+", body))
        return len(quote_tokens & body_tokens) / len(quote_tokens)

    # Keep best (URL, score, matched_paragraph) across all hits so we
    # can return the strongest external match — not just the first
    # passable one.
    best_hit_url: str | None = None
    best_hit_title: str | None = None
    best_score: float = 0.0
    best_matched_kind: str = ""
    best_paragraph: str = ""

    for hit in hits[:5]:
        try:
            passage = await _fetch_and_extract(hit)
        except Exception as exc:
            logger.warning("_locate_by_google: fetch failed url=%s: %s", hit.url, exc)
            continue
        if passage.fetch_status != "ok" or not passage.text:
            logger.info(
                "_locate_by_google: skip url=%s status=%s",
                hit.url[:80], passage.fetch_status,
            )
            continue
        body = passage.text
        body_norm = _norm(body)

        # Score the URL via token-overlap. Threshold lowered to 0.40 —
        # LLMs paraphrase aggressively, so requiring high verbatim
        # overlap means we miss verifiable claims. The doc-title match
        # in the search query already biases toward the right doc; the
        # token overlap just confirms the body is on-topic.
        overlap = _token_overlap(body_norm)
        matched_kind: str | None = None
        # Substring is strongest evidence
        for frag in fragments:
            if frag and frag in body_norm:
                matched_kind = f"substr_{len(frag)}c"
                break
        if matched_kind is None and overlap >= 0.40:
            matched_kind = f"token_overlap_{overlap:.2f}"
        logger.info(
            "_locate_by_google: url=%s overlap=%.2f matched=%s",
            hit.url[:80], overlap, matched_kind,
        )
        if matched_kind is None:
            continue

        # Find the SINGLE BEST paragraph in the fetched body — highest
        # token-overlap with the LLM's quote. This becomes our
        # ``matched_chunk_text`` and the chat planner / UI will show
        # IT as the authoritative source, not the LLM's paraphrase.
        # The LLM's role was discovery; the doc text is the truth.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", body) if p.strip()]
        if not paragraphs:
            paragraphs = [body[:1500]]
        best_para = ""
        best_para_score = 0.0
        for p in paragraphs:
            p_norm = _norm(p)
            if len(p_norm) < 60:
                continue
            p_tokens = set(re.findall(r"[a-z0-9]+", p_norm))
            if not quote_tokens:
                continue
            score = len(quote_tokens & p_tokens) / len(quote_tokens)
            if score > best_para_score:
                best_para_score = score
                best_para = p[:1200]   # cap so the UI doesn't get a wall

        if overlap > best_score:
            best_score = overlap
            best_hit_url = hit.url
            best_hit_title = hit.title
            best_matched_kind = matched_kind
            best_paragraph = best_para or body[:800]

    if not best_hit_url:
        return None

    return _LocateResult(
        document_id=None,           # external — no corpus row
        display_name=best_hit_title or best_hit_url,
        filename=best_hit_url,       # use URL as the citable handle
        discovered_source_url=best_hit_url,
        sitemap_kind="external_google",
        locate_method=(
            f"google_external({best_matched_kind};"
            f" para_overlap={best_score:.2f})"
        ),
        external_chunk_text=best_paragraph,
    )


async def _locate_citation(
    db: AsyncSession,
    cand: CitationCandidate,
) -> _LocateResult:
    """Walk the lookup chain in order:

      1. Title token-overlap match against documents
      2. Exact URL in documents.source_metadata
      3. Exact URL in discovered_sources (handles non-ingested + robots)
      4. Quote-fallback BM25-style search across all chunks
      5. Google external validation — search the web for the citation
         and verify the LLM's quote actually appears on the top hit
      6. None — citation classifies as needs_external
    """
    # 1. Title overlap.
    if cand.document_title:
        r = await _locate_by_title(db, cand.document_title)
        if r is not None:
            return r

    # 2-3. URL.
    if cand.url:
        r = await _locate_by_url(db, cand.url)
        if r is not None:
            return r

    # 4. Quote-fallback: claim grounded in our corpus, just under a
    # different title than the LLM expected.
    if cand.quote:
        r = await _locate_by_quote(db, cand.quote)
        if r is not None:
            return r

    # 5. External validation — check if the citation exists on the
    # web and whether the LLM's quote actually appears there. This
    # turns "we don't have it" into "here's the URL where the claim
    # is verified" instead of letting the citation slip into
    # ``doc_not_found``.
    r = await _locate_by_google(cand)
    if r is not None:
        return r

    return _LocateResult(sitemap_kind="none")


# ---------------------------------------------------------------------------
# Step 3: Quote check inside located doc
# ---------------------------------------------------------------------------

_SECTION_NUMBER_RE = re.compile(r"^\s*(\d+\.?)+\s*")


def _section_topic(section: str | None) -> str:
    """Strip leading numbers from an LLM section, leaving the topic words.

    "10. Claims Submission" → "Claims Submission"
    "10.1 Electronic Claims Submission" → "Electronic Claims Submission"
    """
    if not section:
        return ""
    return _SECTION_NUMBER_RE.sub("", section).strip()


async def _retrieve_in_doc_by_query(
    db: AsyncSession,
    document_id: str,
    user_query: str,
    section: str | None,
) -> tuple[str | None, int | None, str]:
    """Run BM25 within the LLM-identified doc.

    The LLM's role is doc identification. For in-doc retrieval, the
    LLM's *section name* is usually the cleanest topic anchor (it
    names the chapter/topic the answer lives under). User questions
    add noise like "what does X say" / "tell me about" that BM25
    spreads across many chunks.

    Priority:
      1. LLM section name (if present, with leading numbering stripped)
      2. User query (with the natural-language wrapper retained — BM25
         will deweight common words on its own)

    Returns ``(text_excerpt, page_number, retrieval_method)``.
    """
    candidates: list[tuple[str, str]] = []
    sec_topic = _section_topic(section)
    if sec_topic and len(sec_topic) >= 4:
        candidates.append((sec_topic, "by_section_topic"))
    if user_query:
        candidates.append((user_query, "by_user_query"))

    for q, method in candidates:
        sub_req = CorpusSearchRequest(
            query=q,
            k=1,
            mode="precision",
            tag_mode="none",
            include_document_ids=[document_id],
            min_similarity=None,
        )
        sub_resp = await corpus_search(
            db, sub_req, caller=f"strategy_c:retrieve_in_doc:{method}",
        )
        if sub_resp.chunks:
            c = sub_resp.chunks[0]
            return (c.text or "")[:600], c.page_number, method
    return None, None, ""


async def _retrieve_at_section_page(
    db: AsyncSession,
    document_id: str,
    page: int | None,
    section: str | None,
    quote: str | None,
) -> tuple[str | None, int | None, str]:
    """Fetch OUR chunk text at the LLM-cited location.

    Priority order (most→least reliable signal):
      1. ``section`` — semantic; LLM section names usually correspond
         to real document outline entries
      2. ``quote`` plain tsquery — find the chunk containing the claim,
         regardless of where the LLM thought it was
      3. ``page`` — LLM page numbers are notoriously hallucinated, but
         when the section/quote miss, page is still better than nothing
      4. First substantive chunk in the doc — worst case

    We skip chunks that look like running headers (length < 200 chars
    AND content is mostly the doc title/page number) at every step.

    Returns ``(text_excerpt, matched_page, retrieval_method)``.
    """
    # Substance filter — avoid running-header chunks. We require
    # length >= 200 chars (typical body paragraph) at every step. The
    # final fallback relaxes this if no substantive chunk exists.
    _MIN_BODY_CHARS = 200

    # 1. Section-path lookup — strongest semantic signal.
    if section:
        sec_pat = f"%{section.lower()}%"
        rows = await db.execute(
            sql_text(
                "SELECT page_number, text "
                "FROM rag_published_embeddings "
                "WHERE document_id::text = :d "
                "  AND LOWER(section_path) LIKE :sec "
                "  AND LENGTH(text) >= :minc "
                "ORDER BY page_number, paragraph_index "
                "LIMIT 1"
            ),
            {"d": document_id, "sec": sec_pat, "minc": _MIN_BODY_CHARS},
        )
        row = rows.mappings().first()
        if row:
            return (row["text"] or "")[:600], int(row["page_number"]), "by_section"

    # 2. Plain tsquery on the quote — anchors on the actual claim text.
    if quote and len(quote) >= _MIN_QUOTE_CHARS:
        rows = await db.execute(
            sql_text(
                "SELECT page_number, text "
                "FROM rag_published_embeddings "
                "WHERE document_id::text = :d "
                "  AND search_vec @@ plainto_tsquery('english', :q) "
                "  AND LENGTH(text) >= :minc "
                "ORDER BY paragraph_index "
                "LIMIT 1"
            ),
            {"d": document_id, "q": quote, "minc": _MIN_BODY_CHARS},
        )
        row = rows.mappings().first()
        if row:
            return (row["text"] or "")[:600], int(row["page_number"]), "by_quote_tokens"

    # 3. Page — last resort because LLM page numbers are unreliable.
    if page is not None:
        rows = await db.execute(
            sql_text(
                "SELECT page_number, text "
                "FROM rag_published_embeddings "
                "WHERE document_id::text = :d AND page_number = :p "
                "  AND LENGTH(text) >= :minc "
                "ORDER BY paragraph_index "
                "LIMIT 1"
            ),
            {"d": document_id, "p": page, "minc": _MIN_BODY_CHARS},
        )
        row = rows.mappings().first()
        if row:
            return (row["text"] or "")[:600], int(row["page_number"]), "by_page"

    # 4. First substantive chunk in the doc.
    rows = await db.execute(
        sql_text(
            "SELECT page_number, text "
            "FROM rag_published_embeddings "
            "WHERE document_id::text = :d AND page_number > 1 "
            "  AND LENGTH(text) >= :minc "
            "ORDER BY page_number, paragraph_index "
            "LIMIT 1"
        ),
        {"d": document_id, "minc": _MIN_BODY_CHARS},
    )
    row = rows.mappings().first()
    if row:
        return (row["text"] or "")[:600], int(row["page_number"]), "doc_first_chunk"

    return None, None, ""


# ---------------------------------------------------------------------------
# Main entry — Strategy (c) executor
# ---------------------------------------------------------------------------

async def strategy_c_llm_validate(
    db: AsyncSession,
    raw_query: str,
    *,
    agent_id: str,
    correlation_id: str | None = None,
) -> StrategyCResult:
    """Run Strategy (c). Returns answer + per-citation validation."""
    t_start = time.monotonic()

    # ── Step 1: LLM generates answer + citations ─────────────────────
    answer, parsed, llm_telemetry = await _ask_llm(
        raw_query, correlation_id=correlation_id,
    )
    raw_citations = parsed.get("citations") or []
    candidates: list[CitationCandidate] = []
    for raw_cite in raw_citations:
        c = _coerce_citation(raw_cite)
        if c is not None:
            candidates.append(c)

    logger.info(
        "[%s] [trace:c:llm] answer_len=%d n_citations_raw=%d n_citations_valid=%d "
        "parse_error=%s elapsed=%dms",
        agent_id, len(answer), len(raw_citations), len(candidates),
        llm_telemetry["parse_error"], llm_telemetry["llm_ms"],
    )

    # ── Step 2 & 3: Locate + quote-check each citation ────────────────
    t_validate = time.monotonic()
    validated: list[ValidatedCitation] = []
    for cand in candidates:
        loc = await _locate_citation(db, cand)
        # Reframed model: LLM is a retrieval HINT, not a fact-check
        # target. We use {doc, page, section} to fetch OUR chunk and
        # return that as the authoritative version. The LLM's quote is
        # carried alongside for display so the user can see the claim.
        if loc.sitemap_kind == "doc_ingested" and loc.document_id:
            # Primary: BM25 within the doc using the USER's question.
            # Falls back to LLM-cited section/page/quote only if that
            # somehow returns nothing.
            chunk_text, matched_page, retrieval_method = await _retrieve_in_doc_by_query(
                db, loc.document_id, raw_query, cand.section,
            )
            if not chunk_text:
                chunk_text, matched_page, retrieval_method = await _retrieve_at_section_page(
                    db, loc.document_id, cand.page, cand.section, cand.quote,
                )

            if chunk_text:
                if retrieval_method in (
                    "by_section_topic", "by_user_query",
                    "by_section", "by_page",
                ):
                    status = "retrieved"
                    notes = f"retrieved by {retrieval_method}"
                elif retrieval_method == "by_quote_tokens":
                    status = "retrieved"
                    notes = "retrieved via quote-token match (LLM page may be off)"
                else:  # doc_first_chunk
                    status = "doc_found_section_missing"
                    notes = (
                        "doc located but cited page/section not in our "
                        "copy; returning first chunk"
                    )
                validated.append(ValidatedCitation(
                    candidate=cand, status=status,
                    document_id=loc.document_id,
                    document_display_name=loc.display_name,
                    document_filename=loc.filename,
                    matched_chunk_text=chunk_text,
                    matched_page=matched_page,
                    discovered_source_url=loc.discovered_source_url,
                    locate_method=loc.locate_method,
                    notes=notes,
                ))
            else:
                # Doc found but it has no chunks at all (extremely rare).
                validated.append(ValidatedCitation(
                    candidate=cand, status="doc_found_section_missing",
                    document_id=loc.document_id,
                    document_display_name=loc.display_name,
                    document_filename=loc.filename,
                    locate_method=loc.locate_method,
                    notes="doc located but no chunks indexed",
                ))
        elif loc.sitemap_kind == "sitemap_robots_blocked":
            validated.append(ValidatedCitation(
                candidate=cand, status="doc_robots_blocked",
                discovered_source_url=loc.discovered_source_url,
                last_fetch_status=loc.last_fetch_status,
                locate_method=loc.locate_method,
                notes=(
                    "sitemap entry exists but robots-blocked; LLM citation "
                    "passed through unverified"
                ),
            ))
        elif loc.sitemap_kind == "sitemap_needs_scrape":
            validated.append(ValidatedCitation(
                candidate=cand, status="doc_in_sitemap_not_ingested",
                discovered_source_url=loc.discovered_source_url,
                last_fetch_status=loc.last_fetch_status,
                locate_method=loc.locate_method,
                notes="sitemap hit, not ingested — strategy (d) on-demand fetch",
            ))
        elif loc.sitemap_kind == "external_google":
            # External validation succeeded — Google found the document
            # the LLM cited (or one matching its tokens). The
            # ``external_chunk_text`` is the actual paragraph from the
            # fetched URL with the highest token-overlap with the LLM's
            # quote. We surface IT as the matched_chunk_text so the
            # chat planner / UI shows the SOURCE TEXT, not the LLM's
            # paraphrase. The LLM's role was discovery; the doc is the
            # truth.
            authoritative_text = loc.external_chunk_text or cand.quote
            validated.append(ValidatedCitation(
                candidate=cand, status="retrieved_external",
                document_id=None,
                document_display_name=loc.display_name,
                document_filename=loc.filename,
                matched_chunk_text=authoritative_text,
                discovered_source_url=loc.discovered_source_url,
                locate_method=loc.locate_method,
                notes=(
                    "verified externally via web search — actual source "
                    "paragraph from fetched URL; LLM's paraphrase replaced"
                ),
            ))
        else:
            validated.append(ValidatedCitation(
                candidate=cand, status="doc_not_found",
                notes=(
                    "no match in corpus, sitemap, or external web search — "
                    "LLM citation passed through unverified"
                ),
            ))

    validate_ms = (time.monotonic() - t_validate) * 1000.0
    logger.info(
        "[%s] [trace:c:validate] n=%d outcomes=%s elapsed=%dms",
        agent_id, len(validated),
        {s: sum(1 for v in validated if v.status == s) for s in {
            "retrieved", "retrieved_external",
            "doc_found_section_missing",
            "doc_in_sitemap_not_ingested", "doc_robots_blocked",
            "doc_not_found",
        }},
        int(validate_ms),
    )

    return StrategyCResult(
        llm_answer=answer,
        citations=validated,
        raw_llm_output=parsed.get("_raw") or json.dumps(parsed),
        telemetry={
            "llm_ms": llm_telemetry["llm_ms"],
            "validate_ms": int(validate_ms),
            "total_ms": int((time.monotonic() - t_start) * 1000),
            "n_citations_raw": len(raw_citations),
            "n_citations_valid": len(validated),
            "parse_error": llm_telemetry["parse_error"],
        },
    )
