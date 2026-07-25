"""Filler d — Web Search (Step 3d of the answer engine).

Live external call: Vertex AI Grounding-with-Google-Search (primary) +
DuckDuckGo (fallback), fetch + extract the top hits, return the extracted
passages as facts — no LLM narrative synthesis. Explicit, documented
exception to the parent Fillers spec's "zero DB/embed side effects", same
class of exception as Fillers c/f/s.

Ported (not imported) from ``corpus_search_strategy_d.py`` per the fleet's
port-don't-import directive (2026-07-23) — read as reference, re-verified
fresh, not depended on live. See ``docs/rag-agents/filler-d-web-kickoff.md``
for the full design record and every cross-agent resolution this module is
built against; only the deltas from that doc are summarized here:

- **`_synthesize()` dropped entirely** (Eval-confirmed safe: `confidence_label`
  has zero references anywhere in `app/services/retriever/`). This module
  never calls an LLM — search + fetch + extract only.
- **Payer context is a parameter, not a live call.** `payer_context.py`
  (shared, built by the Sitemap filler) is imported for `PayerContext` typing
  only; `resolve_payer_context()` itself is NOT called here — the orchestrator
  resolves it once post-Gate and threads the same object to both Router's
  crawl-gate and this filler's execution, per the agreed design (avoids
  paying the ~3s Payor Platform timeout twice per query).
- **Query-enrichment simplified vs. legacy.** Legacy's `build_authoritative_query`
  had a `partition` (`TermPartition`) fast path from `corpus_search_agent.py`'s
  pre-route step — that object doesn't exist in the new pipeline. This module
  only ports the tag-fallback path, and further simplifies it: legacy's
  no-domain branch called `selectivity_for_tag()` (a DB lookup in the legacy
  god-file) to decide whether an unanchored d-tag phrase was safe to quote on
  the open web. This module skips that DB call entirely and only adds a d-tag
  exact-term when a verified `site_domain` is already present to anchor it —
  a strict, safe subset of the original behavior (the original's own
  reasoning: "with a domain, use it regardless — the site restriction already
  anchors it"), not a fabrication. Documented here, not silently dropped.
- **`url` field — RESOLVED 2026-07-23.** DB landed `FilledChunk.url` (real
  contract field now, not a workaround). External chunks follow the
  documented convention: `url` populated, `document_id` `None` (see
  `_chunk_from_passage`). `emit.fillers_d.passages[].url` is kept anyway as
  free-form diagnostic telemetry, not a replacement for the real field.
- **Real BM25 relevance scoring, added 2026-07-23 (Ananth's steer): "run
  bm25 on the retrieved chunks... see how much of the reranking you can
  inherit from BM25."** After fetch+extract, `_score_bm25()` scores every
  passage's FULL body text against the query using the exact same ranking
  function Pool computes for Filler a (`ts_rank_cd(..., plainto_tsquery(...),
  32)`) — genuinely the same BM25 scale, not a look-alike. Reorders passages
  by real relevance before capacity-truncating (previously: raw fetch
  order), and feeds `FilledChunk.original_score` a real number instead of a
  flat `1.0`. Distinct from `_rerank_hits` (which reorders raw search hits
  by title/snippet BEFORE fetch, to prioritize what gets fetched at all) —
  this reranks AFTER fetch, on real body text, to decide what gets kept.
- **`_lookup_sitemap_candidates()` is out of scope** — that's the Sitemap
  filler's own job now, not ported here.

See docs/rag-agents/fillers-schematic-spec.md and
docs/rag-agents/filler-d-web-kickoff.md.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.retriever.pool.contracts import PoolResult
from app.services.retriever.shape.slots import AnswerShapeResult
from app.services.retriever.fillers.contracts import (
    FilledChunk,
    FilledSlot,
    FilledShape,
)
from app.services.retriever.fillers.payer_context import PayerContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — same values as legacy, ported verbatim (no calibration
# reason to change them; this module doesn't recalibrate, it re-implements).
# ---------------------------------------------------------------------------

_FETCH_TIMEOUT_S = 8.0
_MAX_PASSAGE_CHARS = 2000
_VERTEX_GROUNDING_MODEL = "gemini-2.5-flash"

# Widened funnel (2026-07-23, per Ananth: "get 15 or 20 to fill 5" instead
# of pulling exactly N-for-N). Verified live what each backend's real
# ceiling is before picking these: Vertex Grounding tops out around 5-7
# real grounding_chunks for a typical query regardless of the requested n
# (it's the model's own internal limit, not something raising n controls);
# the DDG/mobius-skills fallback has its own hard cap of 10
# (``min(10, ...)`` in ``_search_web``). To actually reach ~15-20 raw
# candidates -- not just relax an early-exit that was already capping below
# either backend's real ceiling -- both backends now run CONCURRENTLY and
# get merged (see ``fill_shape_external``), not DDG-only-as-last-resort.
# BM25 scoring (``_score_bm25``) is what actually filters this wider pool
# down to the best content for each slot -- a bigger funnel only helps if
# something downstream can tell good from bad, which it now can.
_DEFAULT_N_SEARCH = 20
_MAX_FETCH = 15

# Length floor (2026-07-24, per Eval's ask following the quality-check
# finding): same threshold as Filler b's own _MIN_CHUNK_LENGTH, reused not
# reinvented -- catches genuinely tiny/degenerate extractions (a failed
# scrape, an empty page). Does NOT solve the PDF-TOC-not-substance finding
# (a long-but-empty-of-the-actual-answer chunk passes this fine, same word
# count as a real answer) -- that's a different, structural failure mode
# already logged separately (observer-bayesian-confidence-spec.md's third
# known limitation). Two different failure modes wearing similar symptoms,
# per Eval's own framing when this was scoped.
_MIN_PASSAGE_LENGTH = 50

# Same bar as legacy's _WEB_QUOTE_SELECTIVITY_MIN reasoning, but only reached
# via the domain-anchored branch now (see module docstring) — kept as a
# named constant in case a future revision restores the unanchored path.
_GENERIC_D_TAG_LEAVES = frozenset({"general", "info", "information", "misc", "other"})


@dataclass
class RoutingLadder:
    """Strategy sequence per slot (from Router/two-phase design). Unused v1, same as a/b/s."""

    slot_id: str
    strategy_sequence: list[str]


@dataclass
class _SearchHit:
    title: str
    snippet: str
    url: str


@dataclass
class _Passage:
    """A scraped + extracted passage from one search result."""

    url: str
    title: str
    snippet: str
    text: str
    fetch_status: str  # "ok" | "timeout" | f"http_{code}" | "extract_failed:..." | f"error:..."
    fetch_ms: int


# ---------------------------------------------------------------------------
# Query enrichment (simplified port — see module docstring)
# ---------------------------------------------------------------------------


def _bare_leaf(full_code: str) -> str:
    """The tag's leaf segment with any ``x:`` kind-prefix stripped — e.g.
    ``"d:claims.general"`` -> ``"general"``, ``"d:general"`` -> ``"general"``.
    Shared by the generic-leaf exclusion checks below and ``_tag_to_phrase``
    (which additionally underscore/lowercase-normalizes it).
    """
    leaf = full_code.split(".")[-1]
    if len(leaf) > 2 and leaf[1] == ":":
        leaf = leaf[2:]
    return leaf


def _most_specific_d_tag(tag_matches: list[str] | None) -> str | None:
    """Return the most specific ``d:`` tag (most dots, then longest code),
    excluding generic filler leaves. Ported from ``corpus_search_strategy_d.py``'s
    ``_most_specific_d_tag``.

    SECOND BUG FOUND DURING PORTING (2026-07-23, same class as
    ``_tag_to_phrase``'s, caught by a failing test on ``_boost_phrases``
    which shares this pattern): legacy's generic-leaf check was
    ``t.split(".")[-1] not in _GENERIC_D_TAG_LEAVES`` — for a MULTI-segment
    code the split correctly isolates the bare leaf, but for a
    single-segment code like ``"d:general"`` (no dot), ``split(".")[-1]``
    returns the WHOLE string with the "d:" prefix still attached, which
    never matches the (bare-word) exclusion set — so a single-segment
    generic tag like ``"d:general"`` silently slipped through as a
    "specific" candidate. Fixed here via ``_bare_leaf``, same fix as
    ``_tag_to_phrase``.
    """
    d_tags = [t for t in (tag_matches or []) if t.startswith("d:")]
    if not d_tags:
        return None
    candidates = [t for t in d_tags if _bare_leaf(t) not in _GENERIC_D_TAG_LEAVES]
    if not candidates:
        return None
    return max(candidates, key=lambda t: (t.count("."), len(t)))


def _tag_to_phrase(full_code: str) -> str:
    """``d:utilization_management.prior_authorization`` -> "prior authorization".
    Works on any ``d:``/``p:``/``j:`` tag, not just ``d:`` — generalized
    (2026-07-23) so the same helper serves both the query-embedded exact
    term (d-tags only, see ``build_authoritative_query``) and the broader
    reranking-only boost phrases (all three kinds, see ``_boost_phrases``).

    BUG FOUND DURING PORTING (2026-07-23, caught by a failing test, not
    inherited silently): legacy's version was ``leaf = full_code.split(".")[-1]``
    with no explicit prefix strip — for a MULTI-segment code the split
    incidentally drops the "d:" prefix (it's part of the first segment, not
    the last), but for a single-segment code like ``"d:eligibility"`` (no
    dot at all), ``split(".")[-1]`` returns the WHOLE original string,
    prefix still attached — so legacy would search-quote the literal
    string ``"d:eligibility"``, colon included. Fixed here by stripping any
    ``"x:"`` kind-prefix explicitly instead of relying on the split to do
    it incidentally.
    """
    return _bare_leaf(full_code).replace("_", " ").strip().lower()


def build_authoritative_query(
    raw_query: str, tag_matches: list[str] | None, site_domain: str | None,
) -> tuple[str, str | None, list[str]]:
    """Return (query, site_domain, exact_terms). The raw query text passes
    through unchanged; site/exact_terms are applied as real search-engine
    operators by ``_embed_search_operators``, not string concatenation.

    Simplified vs. legacy (see module docstring): only adds a d-tag exact
    term when ``site_domain`` is present to anchor it. No literal-code
    (``partition``) fast path — that object doesn't exist in the new
    pipeline; no unanchored-quote selectivity DB lookup.

    Deliberately narrow — this is what actually constrains the live search,
    and an over-specified query can silently zero out results. Broader tag
    signal (p:/j: tags, and d-tags without a domain to anchor them) is used
    for reranking instead, not embedded here — see ``_boost_phrases`` +
    ``_rerank_hits``, per Ananth's steer (2026-07-23): safer to rerank on
    more signal than to risk killing recall by cramming it into the query.
    """
    exact_terms: list[str] = []
    if site_domain:
        candidate_tag = _most_specific_d_tag(tag_matches)
        if candidate_tag:
            exact_terms.append(_tag_to_phrase(candidate_tag))
    return raw_query, site_domain, exact_terms


def _boost_phrases(tag_matches: list[str] | None) -> list[str]:
    """Every matched d:/p:/j: tag, turned into a rerank-only boost phrase —
    broader than ``build_authoritative_query``'s single domain-anchored
    exact term on purpose (2026-07-23, per Ananth: use the fuller tag
    signal in reranking rather than risk over-constraining the live query).
    Generic leaves excluded (same bar as the d-tag-only path); deduped,
    order-preserving.
    """
    phrases: list[str] = []
    seen: set[str] = set()
    for t in (tag_matches or []):
        if not (t.startswith("d:") or t.startswith("p:") or t.startswith("j:")):
            continue
        if _bare_leaf(t) in _GENERIC_D_TAG_LEAVES:
            continue
        phrase = _tag_to_phrase(t)
        if phrase and phrase not in seen:
            seen.add(phrase)
            phrases.append(phrase)
    return phrases


def _dedup_hits(*hit_lists: list[_SearchHit]) -> list[_SearchHit]:
    """Merge multiple hit lists (e.g. constrained + unconstrained Vertex
    calls, plus DDG) into one, deduped by URL, preserving first-seen order
    across the lists in the priority order they're passed.
    """
    seen: set[str] = set()
    out: list[_SearchHit] = []
    for hits in hit_lists:
        for h in hits:
            if h.url not in seen:
                seen.add(h.url)
                out.append(h)
    return out


def _dedup_passages_by_text(passages: list[_Passage]) -> list[_Passage]:
    """Drop passages whose extracted text exactly matches an earlier one.

    ``_dedup_hits`` only catches duplicate URLs -- different URLs (mirror
    pages, syndicated content, the same PDF hosted at two places) can
    return near-identical or identical text that survives URL-level dedup
    untouched (real gap, flagged during Observer's stop-signal design
    review 2026-07-24, fixed here since it's this filler's own hygiene,
    not Observer's problem to work around).

    Exact-text match only, not fuzzy similarity -- no calibration data
    exists yet for a similarity threshold, and exact match already covers
    the concrete case this was raised for (mirror/syndicated pages
    returning byte-identical extracted text). First-seen wins, same
    convention as ``_dedup_hits`` -- doesn't matter for quality (identical
    text scores identically on BM25 regardless of which URL it came from),
    only for which URL/chunk_id ends up representing it.
    """
    seen: set[str] = set()
    out: list[_Passage] = []
    for p in passages:
        key = p.text.strip()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _rerank_hits(
    hits: list[_SearchHit], site_domain: str | None, exact_terms: list[str],
    boost_terms: list[str] | None = None,
) -> list[_SearchHit]:
    """Promote hits matching the known payer domain, the query-embedded
    exact_terms, and/or the broader boost_terms (tag signal NOT embedded in
    the query itself — see ``_boost_phrases``). Stable sort — ties keep the
    engine's original order. Ported from ``corpus_search_strategy_d.py``'s
    ``_rerank_hits``, extended with ``boost_terms`` (2026-07-23).
    """
    boost_terms = boost_terms or []
    if not site_domain and not exact_terms and not boost_terms:
        return hits

    def _score(h: _SearchHit) -> int:
        s = 0
        if site_domain and site_domain.lower() in h.url.lower():
            s += 2
        text = f"{h.title} {h.snippet}".lower()
        s += sum(1 for term in exact_terms if term.lower() in text)
        # Dedup against exact_terms so a term counted once above (it's
        # already guaranteed present for vertex/DDG hits that matched the
        # embedded query) doesn't also double-count here.
        s += sum(1 for term in boost_terms if term not in exact_terms and term.lower() in text)
        return s

    return sorted(hits, key=_score, reverse=True)


def _embed_search_operators(
    raw_query: str, site_domain: str | None, exact_terms: list[str],
) -> str:
    """Append real search-engine operators (quoted phrases, ``site:``) to
    the query text. Ported verbatim from ``corpus_search_strategy_d.py``'s
    ``_embed_search_operators``.
    """
    q = raw_query
    for term in exact_terms:
        q = f'{q} "{term}"'
    if site_domain:
        q = f"{q} site:{site_domain}"
    return q


# ---------------------------------------------------------------------------
# Search — Vertex AI Grounding with Google Search (primary), DuckDuckGo (fallback)
# ---------------------------------------------------------------------------


async def _search_web_vertex(
    query: str, *, n: int = 5, site: str | None = None, exact: list[str] | None = None,
) -> list[_SearchHit]:
    """Vertex AI's "Grounding with Google Search" — NOT the closed Google
    Custom Search JSON API (see ``project_google_cse_closed_new_customers``
    memory). Ported from ``corpus_search_strategy_d.py``'s
    ``_search_web_vertex``, re-verified: ``site``/``exact`` are embedded as
    real search operators in the prompt text (grounding has no structured
    field for them); ``grounding_chunks`` exposes no per-source snippet, so
    hits come back with ``snippet=""`` — real page text is filled in by
    ``_fetch_and_extract``.
    """
    prompt = query
    for term in (exact or []):
        prompt = f'{prompt} "{term}"'
    if site:
        prompt = f"{prompt} site:{site}"

    try:
        from app.config import VERTEX_PROJECT_ID, VERTEX_LOCATION
        from google import genai
        from google.genai.types import GenerateContentConfig, GoogleSearch, Tool

        if not VERTEX_PROJECT_ID:
            logger.warning("[filler_d] VERTEX_PROJECT_ID not set; skipping vertex grounding search")
            return []

        client = genai.Client(vertexai=True, project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        tool = Tool(google_search=GoogleSearch())
        response = await client.aio.models.generate_content(
            model=_VERTEX_GROUNDING_MODEL,
            contents=prompt,
            config=GenerateContentConfig(tools=[tool]),
        )
        cand = response.candidates[0] if response.candidates else None
        gm = getattr(cand, "grounding_metadata", None) if cand else None
        chunks = (gm.grounding_chunks or []) if gm else []

        out: list[_SearchHit] = []
        seen: set[str] = set()
        for c in chunks:
            web = getattr(c, "web", None)
            if not web or not web.uri or web.uri in seen:
                continue
            seen.add(web.uri)
            out.append(_SearchHit(title=web.title or "", snippet="", url=web.uri))
            if len(out) >= n:
                break
        return out
    except Exception as exc:
        logger.warning("[filler_d] vertex grounding search failed: %s", exc)
        return []


async def _search_web(
    query: str, *, n: int = 5, site: str | None = None, exact: str | None = None,
) -> list[_SearchHit]:
    """Call the shared google-search skill (falls back to DuckDuckGo
    internally, same infra Fillers c/f share). Ported from
    ``corpus_search_strategy_d.py``'s ``_search_web``.
    """
    import os

    base = os.environ.get("CHAT_SKILLS_GOOGLE_SEARCH_URL", "").strip()
    if not base:
        logger.warning("[filler_d] CHAT_SKILLS_GOOGLE_SEARCH_URL not set; cannot run fallback web search")
        return []

    sep = "&" if "?" in base else "?"
    url = (
        base.rstrip("/") + sep
        + "q=" + urllib.parse.quote(query)
        + f"&num={min(10, max(1, n))}"
    )
    if site:
        url += "&site=" + urllib.parse.quote(site)
    if exact:
        url += "&exact=" + urllib.parse.quote(exact)

    def _do_request() -> list[_SearchHit]:
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read().decode()
            payload = json.loads(data)
            items = payload if isinstance(payload, list) else (
                payload.get("items") or payload.get("results") or []
            ) if isinstance(payload, dict) else []
            out: list[_SearchHit] = []
            for r in (items or [])[:n]:
                if not isinstance(r, dict):
                    continue
                u = (r.get("url") or r.get("link") or "").strip()
                if not u:
                    continue
                out.append(_SearchHit(
                    title=(r.get("title") or "").strip(),
                    snippet=(r.get("snippet") or r.get("description") or "").strip(),
                    url=u,
                ))
            return out
        except Exception as exc:
            logger.warning("[filler_d] fallback web search failed: %s", exc)
            return []

    return await asyncio.to_thread(_do_request)


# ---------------------------------------------------------------------------
# Fetch + extract
# ---------------------------------------------------------------------------


async def _fetch_and_extract(hit: _SearchHit) -> _Passage:
    """Fetch one URL with timeout, extract main text (HTML sections or PDF
    page text). Ported from ``corpus_search_strategy_d.py``'s
    ``_fetch_and_extract``, re-verified fresh (real HTTP call, real
    extraction, not assumed correct from reading the original).
    """
    t_start = time.monotonic()
    try:
        async with httpx.AsyncClient(
            timeout=_FETCH_TIMEOUT_S,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/avif,image/webp,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
        ) as client:
            resp = await client.get(hit.url)
            elapsed = int((time.monotonic() - t_start) * 1000)
            if resp.status_code >= 400:
                return _Passage(
                    url=hit.url, title=hit.title, snippet=hit.snippet,
                    text="", fetch_status=f"http_{resp.status_code}", fetch_ms=elapsed,
                )
            content_type = (resp.headers.get("content-type") or "").lower()
            body_bytes = resp.content
            html = resp.text
    except Exception as exc:
        return _Passage(
            url=hit.url, title=hit.title, snippet=hit.snippet,
            text="", fetch_status=f"error:{type(exc).__name__}",
            fetch_ms=int((time.monotonic() - t_start) * 1000),
        )

    is_pdf = (
        "application/pdf" in content_type
        or hit.url.lower().endswith(".pdf")
        or body_bytes[:4] == b"%PDF"
    )
    if is_pdf:
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(stream=body_bytes, filetype="pdf")
            page_texts: list[str] = []
            char_total = 0
            for page in doc:
                pt = (page.get_text() or "").strip()
                if not pt:
                    continue
                page_texts.append(pt)
                char_total += len(pt)
                if char_total >= _MAX_PASSAGE_CHARS:
                    break
            doc.close()
            text = "\n\n".join(page_texts)[:_MAX_PASSAGE_CHARS]
            elapsed = int((time.monotonic() - t_start) * 1000)
            if not text:
                return _Passage(
                    url=hit.url, title=hit.title, snippet=hit.snippet,
                    text=hit.snippet, fetch_status="pdf_empty", fetch_ms=elapsed,
                )
            return _Passage(
                url=hit.url, title=hit.title, snippet=hit.snippet,
                text=text, fetch_status="ok", fetch_ms=elapsed,
            )
        except Exception as exc:
            return _Passage(
                url=hit.url, title=hit.title, snippet=hit.snippet,
                text=hit.snippet, fetch_status=f"pdf_extract_failed:{type(exc).__name__}",
                fetch_ms=int((time.monotonic() - t_start) * 1000),
            )

    try:
        from app.services.html_extractor import extract_sections

        sections = extract_sections(html, source_url=hit.url)
        bodies: list[str] = []
        for s in sections:
            body = (s.get("text") or "").strip()
            if body:
                bodies.append(body)
        bodies.sort(key=lambda b: -len(b))
        text = ""
        for b in bodies:
            if len(text) + len(b) + 2 > _MAX_PASSAGE_CHARS:
                remaining = _MAX_PASSAGE_CHARS - len(text)
                if remaining > 200:
                    text += "\n\n" + b[:remaining]
                break
            text = (text + "\n\n" + b) if text else b
        if not text:
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()[:_MAX_PASSAGE_CHARS]
        elapsed = int((time.monotonic() - t_start) * 1000)
        return _Passage(
            url=hit.url, title=hit.title, snippet=hit.snippet,
            text=text, fetch_status="ok", fetch_ms=elapsed,
        )
    except Exception as exc:
        return _Passage(
            url=hit.url, title=hit.title, snippet=hit.snippet,
            text=hit.snippet, fetch_status=f"extract_failed:{type(exc).__name__}",
            fetch_ms=int((time.monotonic() - t_start) * 1000),
        )


# ---------------------------------------------------------------------------
# Passage -> FilledChunk (facts, not narrative — see kickoff doc)
# ---------------------------------------------------------------------------


def _stable_chunk_id(url: str) -> str:
    """Synthetic id — no real DB row backs a live-fetched passage. Shared
    scheme with the Sitemap/LLM-retrieval fillers for their equivalent
    no-real-id cases (kickoff doc, round 2).
    """
    return "ext:" + hashlib.sha1(url.encode()).hexdigest()[:16]


# plainto_tsquery ANDs every non-stopword token together -- a natural-
# language question ("What is the timely filing deadline for Sunshine
# Health...") carries lead-phrase/quantifier noise ("what", "is", "the",
# "many") that has zero chance of appearing verbatim in a real fetched
# passage, so the AND kills the WHOLE match even when every real content
# word is present. Ported from ``corpus_search.py``'s ``_normalize_bm25_query``
# / ``_QUESTION_LEAD`` / ``_BM25_NOISE`` (already the fleet's battle-tested
# fix for this exact failure mode) — verified live: without this, a real
# Vertex-fetched passage that genuinely discusses Sunshine Health's timely
# filing deadline scored a hard 0.0 against the raw question.
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

_BM25_NOISE = frozenset({
    'many', 'much', 'often', 'several', 'various', 'certain',
    'few', 'some', 'any', 'every', 'all', 'most', 'more',
})


def _normalize_bm25_query(query: str) -> str:
    """Strip question lead phrases and noise quantifiers before handing the
    query to ``plainto_tsquery``. Never returns empty — falls back to the
    original query text if stripping would leave nothing.
    """
    q = _QUESTION_LEAD.sub(' ', query)
    words = [w for w in q.split() if w.lower() not in _BM25_NOISE]
    normalized = ' '.join(words).strip()
    return normalized or query


async def _score_bm25(
    db: AsyncSession, raw_query: str, passages: list[_Passage],
) -> dict[str, float]:
    """Real BM25 relevance score per passage, against its FULL extracted
    body text (not just title/snippet) — inherited from the exact same
    ranking function Pool computes for Filler a (2026-07-23, per Ananth's
    steer: "run bm25 on the retrieved chunks... see how much of the
    reranking you can inherit from BM25").

    Pool's own formula (``pool/public_adapter.py``'s ``_BM25_SCORE_EXPR``):
    ``ts_rank_cd(search_vec, plainto_tsquery('english', :query), 32)``.
    Same ranking function and normalization flag (32) here, but the query
    side is deliberately NOT a straight ``plainto_tsquery`` (AND semantics)
    — verified live, not theoretical, that this breaks on real fetched
    passages: a genuine, correct "timely filing: 180 days" passage scored a
    hard 0.0 against the raw question, because ``plainto_tsquery`` ANDs
    every stemmed term together and the passage simply didn't repeat every
    single query word verbatim (no "Sunshine"/"Medicaid"/"Florida" on a
    page that IS the timely-filing table). Pool's arm-a doesn't hit this
    because it's ranking within an already-tag-filtered corpus; ranking
    heterogeneous free-form web text needs OR semantics instead — same
    reasoning ``corpus_search.py`` already uses its own OR-tsquery variant
    for ranking (comment: "OR makes stopwords harmless"). Built here via
    ``to_tsquery('english', replace(plainto_tsquery(...)::text, ' & ', ' | '))``
    — reuses ``plainto_tsquery``'s safe stemming/tokenizing of raw user
    text (never throws on stray punctuation, unlike a hand-built
    ``to_tsquery`` string) and just swaps its AND operators for OR before
    re-parsing. Verified live: a genuinely relevant fetched passage now
    scores 0.23 vs. 0.17 for a same-payer-but-off-topic one — real,
    correctly-discriminating signal, not another silent zero.

    ``raw_query`` is also normalized first (``_normalize_bm25_query``) —
    strips lead-phrase/quantifier noise before either query variant sees
    it, same fleet-established fix ``corpus_search.py`` already uses.

    One batched query (UNION ALL over ≤``_MAX_FETCH`` rows, not N round
    trips). Fails closed on any DB error — returns an empty dict, callers
    fall back to fetch order with a neutral score rather than crashing the
    whole filler over a ranking nicety.
    """
    if not passages:
        return {}
    normalized_query = _normalize_bm25_query(raw_query)
    parts: list[str] = []
    params: dict[str, Any] = {"query": normalized_query}
    for i, p in enumerate(passages):
        parts.append(
            f"SELECT :url_{i} AS url, "
            f"ts_rank_cd(to_tsvector('english', :text_{i}), "
            f"to_tsquery('english', replace(plainto_tsquery('english', :query)::text, ' & ', ' | ')), "
            f"32) AS score"
        )
        params[f"url_{i}"] = p.url
        params[f"text_{i}"] = p.text
    sql = " UNION ALL ".join(parts)
    try:
        from sqlalchemy import text as sql_text

        rows = (await db.execute(sql_text(sql), params)).mappings().all()
        return {r["url"]: float(r["score"]) for r in rows}
    except Exception as exc:
        logger.warning("[filler_d] bm25 scoring failed (fetch-order fallback): %s", exc)
        return {}


def _chunk_from_passage(p: _Passage, bm25_score: float | None) -> FilledChunk:
    """RESOLVED 2026-07-23 — DB landed `FilledChunk.url`. External chunks
    now follow the documented convention (`contracts.py`'s `FilledChunk`
    docstring): `url` populated, `document_id` **None** (no real doc row
    backs a live-fetched passage — `chunk_id` alone, synthetic and stable,
    is the identity; `document_id` is intentionally not a second synthetic
    id pretending to be a real one). Previously this stashed a synthetic id
    in `document_id` and omitted `url` entirely as a workaround — no longer
    needed, removed.
    """
    return FilledChunk(
        chunk_id=_stable_chunk_id(p.url),
        document_id=None,
        text=p.text,
        url=p.url,
        source_type="external",
        document_status=None,
        content_sha=None,
        page_number=None,
        paragraph_index=None,
        tags={},
        is_neighbor=False,
        # Real BM25 score against the full fetched text (see _score_bm25) —
        # same ranking function Filler a's original_score uses, genuinely
        # comparable via Observer's percentile-within-pool normalization
        # (kickoff doc). Falls back to a flat 1.0 only if BM25 scoring
        # itself failed (DB error) — every chunk still gets SOME score
        # rather than silently going unranked.
        original_score=bm25_score if bm25_score is not None else 1.0,
        assignment_reason="external_fetch",
    )


# ---------------------------------------------------------------------------
# Speculative pre-fetch (search-only) — Ananth green-lit 2026-07-23, v1 of
# the idea documented in filler-d-web-tracker.md's "Latency investigation"
# section. Extracted out of fill_shape_external so the orchestrator can fire
# JUST this piece concurrently with Pool's build (same shape as
# payer_context's existing concurrent resolution) and cache the result —
# fetch+synthesize stay sequential, paid only if Router actually picks "d".
# Orchestrator wiring (where/when this gets called, threading the cached
# result into the eventual fill_shape_external call) is Retriever's, not
# built here.
# ---------------------------------------------------------------------------


@dataclass
class PrescreenedSearch:
    """Search-only result — no fetch, no BM25, no DB. Cacheable and reusable:
    pass into ``fill_shape_external``'s ``prescreened`` param to skip
    re-running search entirely (the whole point — the ~5-10s Vertex leg
    already sunk during Pool's build, per real measurements in the tracker).
    """

    hits: list[_SearchHit]
    search_backend: str
    site_domain: str | None
    exact_terms: list[str]
    boost_terms: list[str]
    search_ms: int
    n_vertex_hits: int
    n_vertex_unconstrained_hits: int
    n_ddg_hits: int


def should_prescreen_search(authority_requirement: str) -> bool:
    """Whether firing d's search speculatively is worth it for this query.

    Gated on Router's real, already-built caller-declared authority signal
    (``allocation.py``'s ``AUTHORITY_CITABLE_REQUIRED`` gate, verified live
    2026-07-23) — for ``citable_required`` queries assigned to a REQUIRED
    slot, "d" is already ineligible (``strategy_authority_eligible``), so
    pre-fetching its search would be pure waste.

    Deliberately a CONSERVATIVE simplification of the real rule, not a
    re-derivation of it: the actual gate (``strategy_authority_eligible``)
    is per-slot — "d" stays eligible even under ``citable_required`` for
    OPTIONAL/external_context slots (web context is still useful context,
    just not citable evidence). This function only checks the query-level
    ``authority_requirement``, not per-slot ``required``-ness, because that
    per-slot detail isn't available yet at the point (concurrent with Pool,
    before Router runs) where this decision has to be made. Consequence:
    this can occasionally skip a legitimate optional-slot opportunity
    (a missed optimization — Filler d still works fine without a
    prescreened cache, just pays the search cost fresh) but never fires
    when it definitely shouldn't. Correctness-safe, not optimization-complete.
    """
    return authority_requirement != "citable_required"


async def prescreen_search(
    raw_query: str,
    *,
    tag_matches: list[str] | None = None,
    payer_context: PayerContext | None = None,
    n_search: int = _DEFAULT_N_SEARCH,
    agent_id: str = "filler_d",
) -> PrescreenedSearch:
    """Run the search-only phase: build the authoritative query, fire the
    widened/diversified concurrent search fan-out (Vertex constrained +
    Vertex unconstrained + DDG, per the 2026-07-23 diversification change),
    merge/dedup, and rerank. No fetch, no BM25, no DB call.

    This is exactly what ``fill_shape_external`` used to do inline before
    the search-only extraction (2026-07-23) — factored out so it can be
    called speculatively, concurrently with Pool, from the orchestrator.
    """
    site_domain = payer_context.site_domain if payer_context else None
    query_text, site_domain, exact_terms = build_authoritative_query(
        raw_query, tag_matches, site_domain,
    )

    # Widened, diversified search funnel (2026-07-23, per Ananth): run
    # multiple search calls CONCURRENTLY rather than one narrow call or a
    # sequential fallback, then merge+dedup — "get 15-20 to fill 5" (a
    # single call tops out well below capacity/slot count once fetch
    # failures + BM25 quality-filtering thin it out) and "2 parallel calls,
    # one with the must-haves and one without... we will/may end up with
    # different chunks" (an over-constrained query can silently miss good
    # results that don't happen to repeat the exact anchor phrase).
    #
    # - Vertex, constrained (site_domain/exact_terms embedded) — highest
    #   precision when we have a verified payer domain to anchor to.
    # - Vertex, UNCONSTRAINED (raw query only) — only run as a second,
    #   genuinely different call when there's something to vary; with no
    #   domain/exact_terms the two calls would be identical, so skip the
    #   duplicate work.
    # - DDG/mobius-skills — now a real concurrent contributor to the pool,
    #   not a last-resort-only-if-Vertex-returned-nothing fallback (still
    #   useful even when Vertex succeeds, since it's a different index).
    t_search = time.monotonic()
    is_constrained = bool(site_domain or exact_terms)
    ddg_query = (
        _embed_search_operators(query_text, site_domain, exact_terms)
        if is_constrained else query_text
    )
    search_tasks = [
        _search_web_vertex(query_text, n=n_search, site=site_domain, exact=exact_terms),
        _search_web(
            ddg_query, n=n_search, site=site_domain,
            exact=" ".join(exact_terms) if exact_terms else None,
        ),
    ]
    if is_constrained:
        search_tasks.append(_search_web_vertex(query_text, n=n_search, site=None, exact=[]))

    search_results = await asyncio.gather(*search_tasks)
    vertex_hits, ddg_hits = search_results[0], search_results[1]
    vertex_unconstrained_hits = search_results[2] if is_constrained else []

    web_hits = _dedup_hits(vertex_hits, vertex_unconstrained_hits, ddg_hits)
    # Last-resort fallback: constrained calls returned nothing (a wrong/
    # stale site_domain can zero out both Vertex-constrained and DDG's
    # hard site: filter) — retry once, fully unconstrained, rather than
    # returning empty when a plain query likely would have worked.
    if not web_hits and is_constrained:
        web_hits = await _search_web(query_text, n=n_search)

    search_backend = "+".join(filter(None, [
        "vertex" if vertex_hits else "",
        "vertex_unconstrained" if vertex_unconstrained_hits else "",
        "ddg" if ddg_hits else "",
    ])) or ("unconstrained_fallback" if web_hits else "none")
    search_ms = int((time.monotonic() - t_search) * 1000)

    boost_terms = _boost_phrases(tag_matches)
    web_hits = _rerank_hits(web_hits, site_domain, exact_terms, boost_terms)
    logger.info(
        "[%s] [trace:d:search] backend=%s query_len=%d site=%s exact=%s boost=%s "
        "n_vertex=%d n_vertex_unconstrained=%d n_ddg=%d n_merged=%d elapsed=%dms",
        agent_id, search_backend, len(query_text), site_domain, exact_terms, boost_terms,
        len(vertex_hits), len(vertex_unconstrained_hits), len(ddg_hits), len(web_hits), search_ms,
    )

    return PrescreenedSearch(
        hits=web_hits,
        search_backend=search_backend,
        site_domain=site_domain,
        exact_terms=exact_terms,
        boost_terms=boost_terms,
        search_ms=search_ms,
        n_vertex_hits=len(vertex_hits),
        n_vertex_unconstrained_hits=len(vertex_unconstrained_hits),
        n_ddg_hits=len(ddg_hits),
    )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


async def fill_shape_external(
    pool_result: PoolResult,
    shape_result: AnswerShapeResult,
    raw_query: str,
    *,
    db: AsyncSession,
    tag_matches: list[str] | None = None,
    payer_context: PayerContext | None = None,
    n_search: int = _DEFAULT_N_SEARCH,
    n_fetch: int = _MAX_FETCH,
    agent_id: str = "filler_d",
    routing_ladders: list[RoutingLadder] | None = None,
    prescreened: PrescreenedSearch | None = None,
) -> FilledShape:
    """Run one search+fetch+extract pass, assign the resulting facts to
    every slot in ``shape_result.slots`` (in the real orchestrator this is
    always exactly one slot — see ``orchestrator._run_fillers_simple``'s
    single-slot-per-call convention — but this loops generically, same
    pattern as Fillers a/b, so it isn't coupled to that caller).

    ``db``/``pool_result`` are accepted for signature consistency with the
    other live-external fillers (c/s) but unused in v1 — no DB call happens
    here at all (payer resolution already happened upstream; see module
    docstring).

    Args:
        pool_result: Output from Pool (Step 2). Unused in v1.
        shape_result: Output from Shape (Step 1) / one slot per the
            orchestrator's convention.
        raw_query: The query's raw text.
        db: Unused in v1 (kept for signature consistency, may be needed if
            a future revision restores DB-backed query enrichment).
        tag_matches: The query's d:/p:/j: tag codes (from Gate's
            GateResult), used only for the d-tag exact-term enrichment.
        payer_context: Already-resolved payer context from the orchestrator
            (site_domain/display_name/crawlable) — NOT re-resolved here.
            None when Gate found no payer tag, or the orchestrator skipped
            resolution (real_time cache miss) — fails open, same as
            Router's own crawl-gate semantics.
        n_search: Max search hits requested. Ignored if ``prescreened`` is given.
        n_fetch: Max URLs fetched+extracted.
        agent_id: For logging/tracing only.
        routing_ladders: Optional per-slot strategy sequences (unused v1,
            same as a/b/s).
        prescreened: Already-run search result (``prescreen_search()``,
            fired speculatively by the orchestrator concurrently with
            Pool's build) — skips re-running search entirely when provided.
            None (default) runs search here, same as before this param
            existed — fully backward compatible.

    Returns:
        FilledShape with slots assigned from real fetched-passage facts.
    """
    t_start = time.monotonic()

    search = prescreened or await prescreen_search(
        raw_query, tag_matches=tag_matches, payer_context=payer_context,
        n_search=n_search, agent_id=agent_id,
    )
    web_hits = search.hits
    search_backend = search.search_backend
    search_ms = search.search_ms
    vertex_hits_count = search.n_vertex_hits
    vertex_unconstrained_hits_count = search.n_vertex_unconstrained_hits
    ddg_hits_count = search.n_ddg_hits

    passages: list[_Passage] = []
    fetch_ms = 0
    if web_hits:
        t_fetch = time.monotonic()
        passages = list(await asyncio.gather(
            *[_fetch_and_extract(h) for h in web_hits[:n_fetch]],
            return_exceptions=False,
        ))
        fetch_ms = int((time.monotonic() - t_fetch) * 1000)

    n_ok = sum(1 for p in passages if p.fetch_status == "ok" and p.text.strip())
    logger.info(
        "[%s] [trace:d:fetch] n_attempted=%d n_ok=%d elapsed=%dms statuses=%s",
        agent_id, len(passages), n_ok, fetch_ms, [p.fetch_status for p in passages],
    )

    usable_passages = [p for p in passages if p.fetch_status == "ok" and p.text.strip()]
    n_below_length_floor = sum(1 for p in usable_passages if len(p.text.strip()) < _MIN_PASSAGE_LENGTH)
    usable_passages = [p for p in usable_passages if len(p.text.strip()) >= _MIN_PASSAGE_LENGTH]
    n_before_text_dedup = len(usable_passages)
    usable_passages = _dedup_passages_by_text(usable_passages)
    n_text_duplicates_dropped = n_before_text_dedup - len(usable_passages)
    if n_below_length_floor or n_text_duplicates_dropped:
        logger.info(
            "[%s] [trace:d:quality] below_length_floor=%d text_duplicates_dropped=%d",
            agent_id, n_below_length_floor, n_text_duplicates_dropped,
        )

    # Real BM25 against the full fetched text, inherited from Filler a's
    # exact ranking function (see _score_bm25) -- reorders by genuine
    # relevance instead of trusting fetch order, and feeds original_score.
    t_bm25 = time.monotonic()
    bm25_scores = await _score_bm25(db, raw_query, usable_passages)
    bm25_ms = int((time.monotonic() - t_bm25) * 1000)
    usable_passages.sort(key=lambda p: bm25_scores.get(p.url, 0.0), reverse=True)
    logger.info(
        "[%s] [trace:d:bm25] n_scored=%d elapsed=%dms scores=%s",
        agent_id, len(bm25_scores), bm25_ms, {p.url: bm25_scores.get(p.url) for p in usable_passages},
    )

    usable_chunks = [
        _chunk_from_passage(p, bm25_scores.get(p.url)) for p in usable_passages
    ]

    filled_slots: list[FilledSlot] = []
    total_assigned = 0
    remaining = usable_chunks.copy()

    for slot in shape_result.slots:
        filled_slot = FilledSlot(
            slot_id=slot.slot_id,
            slot_semantics=slot.slot_semantics,
            capacity=slot.capacity,
            required=slot.required,
        )
        assigned = remaining[: slot.capacity]
        remaining = remaining[slot.capacity :]

        filled_slot.chunks = assigned
        filled_slot.occupancy = len(assigned)
        filled_slot.under_filled = filled_slot.occupancy < slot.capacity
        filled_slot.over_filled = False

        filled_slots.append(filled_slot)
        total_assigned += filled_slot.occupancy

    emit = {
        "fillers_decision": "web_search_fetch_extract",
        "search_backend": search_backend,
        "slots_filled": len([s for s in filled_slots if s.occupancy > 0]),
        "empty_slots": len([s for s in filled_slots if s.occupancy == 0]),
        "under_filled": len([s for s in filled_slots if s.under_filled]),
        "total_chunks_assigned": total_assigned,
        "search_ms": search_ms,
        "fetch_ms": fetch_ms,
        "bm25_ms": bm25_ms,
        "total_ms": int((time.monotonic() - t_start) * 1000),
        "n_hits": len(web_hits),
        "n_vertex_hits": vertex_hits_count,
        "n_vertex_unconstrained_hits": vertex_unconstrained_hits_count,
        "n_ddg_hits": ddg_hits_count,
        "prescreened": prescreened is not None,
        "n_fetched": len(passages),
        "n_ok": n_ok,
        "n_below_length_floor": n_below_length_floor,
        "n_text_duplicates_dropped": n_text_duplicates_dropped,
        # Diagnostic detail per attempted URL (incl. ones that failed to
        # fetch, unlike FilledChunk which only carries successful ones) --
        # not a substitute for FilledChunk.url (real field now).
        "passages": [
            {
                "url": p.url, "title": p.title, "fetch_status": p.fetch_status, "fetch_ms": p.fetch_ms,
                "bm25_score": bm25_scores.get(p.url),
            }
            for p in passages
        ],
        "per_slot_details": [
            {"slot_id": s.slot_id, "slot_semantics": s.slot_semantics, "occupancy": s.occupancy, "capacity": s.capacity}
            for s in filled_slots
        ],
    }

    return FilledShape(
        slots=filled_slots,
        total_chunks_assigned=total_assigned,
        filling_strategy="web_search",
        emit=emit,
    )
