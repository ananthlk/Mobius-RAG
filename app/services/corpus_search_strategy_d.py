"""Strategy (d) — External First.

Highest-recall, lowest-precision tier in the retrieval portfolio. Used
when the corpus has no signal AND the LLM's parametric prior is
unreliable (recent events, hyper-specific external authorities, query
class with broad public sources).

Pipeline
--------

1. **Search** — call the shared mobius-skills Google search service
   (CHAT_SKILLS_GOOGLE_SEARCH_URL). Falls back to DuckDuckGo internally.
   Returns up to N {title, snippet, url} hits.

2. **Fetch + extract** — async-fetch the top K URLs in parallel, parse
   with the existing ``html_extractor`` to get clean section text.
   Pages that 4xx/5xx or timeout are skipped (best-effort).

3. **Synthesize** — pass the user query + extracted passages to the
   LLM, ask for a short answer with which-passage-supports-which-claim
   citations. We DON'T try to verify against our corpus here — that
   was (c)'s job. (d) is recall-of-last-resort; the user is told the
   sources are external.

4. **Return** — chunks carry the scraped passage text + source URL +
   ``source_type="external"`` so downstream (chat planner, UI) can
   render them with appropriate "external — verify" framing.

Cost / latency: search ~1-3s, parallel fetch ~3-8s (longest URL
dominates), LLM ~3-6s. End-to-end ~7-15s. (d)'s static priors should
reflect this.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services import llm_manager_client


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Per-URL fetch timeout. Sites that hang past this are dropped.
_FETCH_TIMEOUT_S = 8.0

# How many URLs to fetch + extract from search results. More = more
# recall but more latency. 5 covers typical 403/404 attrition.
_MAX_FETCH = 5

# Max chars of extracted passage we keep per source. Bounds LLM input.
_MAX_PASSAGE_CHARS = 2000


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

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
    text: str            # extracted body text, truncated to _MAX_PASSAGE_CHARS
    fetch_status: str    # "ok" | "timeout" | "http_error" | "extract_failed" | "skipped"
    fetch_ms: int


@dataclass
class StrategyDResult:
    llm_answer: str
    passages: list[_Passage] = field(default_factory=list)
    raw_llm_output: str = ""
    telemetry: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Query enrichment — bias the search toward authoritative payer sources
# and the right document type, instead of sending the raw user question
# verbatim. Evidence: 2026-07-07 calibration review found (d) never wins
# a single narrow/code-specific query (n=8, zero exceptions) but does
# win cleanly on standardized administrative facts IF the search actually
# surfaces the payer's own provider-facing page instead of generic SEO
# content. Only payers with a VERIFIED domain (seen in the payor registry
# seeds) get a site: restriction — guessing a domain would silently zero
# out results, which is worse than not restricting at all.
# ---------------------------------------------------------------------------

# Payer domain + display-name resolution is fully data-driven (see
# _resolve_payer_context below) — deliberately NOT a hand-maintained
# allowlist. A static dict needs a manual edit for every new payer, and
# (found 2026-07-07) can silently assert something the data doesn't
# support: an earlier hardcoded entry for "aetna" claimed
# aetnabetterhealth.com as verified, but discovered_sources has ZERO
# fetchable rows for Aetna — that's exactly the "guessed domain" risk
# this file otherwise refuses to take, just asserted instead of guessed.
# Resolving against the live crawl means any payer the curator actually
# covers becomes usable automatically, and a payer with real coverage
# gaps (like Aetna) correctly gets no restriction until the crawl catches
# up — no code change needed either way.

# d-tag prefix -> URL-path keyword. discovered_sources.topic_tags is NULL
# across the board (checked 2026-07-07), so we match on URL path instead.
_D_TAG_URL_KEYWORDS: list[tuple[str, list[str]]] = [
    ("claims.timely_filing", ["timely", "filing"]),
    ("utilization_management.prior_authorization", ["preauth", "prior-auth", "authorization"]),
    ("utilization_management", ["preauth", "prior-auth", "authorization"]),
    ("pharmacy", ["pharmacy"]),
    ("eligibility", ["eligib"]),
    ("disputes", ["appeal", "grievance"]),
    ("claims", ["claims"]),
]


def _extract_payer_slug(tag_matches: list[str] | None) -> str | None:
    for t in (tag_matches or []):
        if t.startswith("j:payor."):
            return t.split("j:payor.", 1)[1]
    return None


_GENERIC_D_TAG_LEAVES = {"general", "info", "information", "misc", "other"}


def _most_specific_d_tag(tag_matches: list[str] | None) -> str | None:
    """Return the MOST SPECIFIC matched d-tag, full code (e.g.
    ``d:utilization_management.prior_authorization``) — kept as the full
    code, not just the leaf phrase, so the caller can look up its corpus
    selectivity via ``selectivity_for_tag`` before deciding whether it's
    discriminating enough to force as a must-have. Filler leaves
    ("general", "info") are excluded first — a tag like ``claims.general``
    and ``claims.timely_filing`` tie on dot-count, and "general" alone is
    useless as a must-have phrase regardless of its corpus frequency.
    Ties among the remaining candidates break on longest code (more
    specific overall).
    """
    d_tags = [t for t in (tag_matches or []) if t.startswith("d:")]
    if not d_tags:
        return None
    candidates = [t for t in d_tags if t.split(".")[-1] not in _GENERIC_D_TAG_LEAVES]
    if not candidates:
        # BUG FIX (2026-07-07, cmhc005 live regression): the old fallback
        # was re-admitting the generic leaves this filter just excluded —
        # exactly when it matters most (a query whose only d-tag is e.g.
        # ``claims.general``). No candidates surviving the filter means
        # there IS no good must-have candidate here — return None.
        return None
    return max(candidates, key=lambda t: (t.count("."), len(t)))


def _tag_to_phrase(full_code: str) -> str:
    """``d:utilization_management.prior_authorization`` -> "prior authorization"."""
    leaf = full_code.split(".")[-1]
    return leaf.replace("_", " ").strip().lower()


_MIN_PAYER_ROWS = 3  # a single stray crawl row can't manufacture a "verified" domain

_PAYOR_PLATFORM_BASE = os.environ.get(
    "MOBIUS_PAYOR_URL", "https://mobius-payor-ortabkknqa-uc.a.run.app",
).rstrip("/")
_PAYOR_PLATFORM_TIMEOUT_S = 3.0


async def _resolve_payer_context(
    db: AsyncSession, slug: str | None,
) -> tuple[str | None, str | None]:
    """Resolve (site_domain, payer display name) for a ``j:payor.<slug>``
    tag. Two layers, in order:

    1. **Payor Platform registry** (2026-07-07 collaboration) — the
       authoritative source. It combines a live robots.txt check with our
       own crawl evidence into a tri-state ``crawlable`` verdict:
       ``true`` -> safe to ``site:``-restrict, ``false`` -> explicitly
       DON'T (robots disallows machines even where our own crawler found
       some pages — confirmed for Aetna: 0 fetchable rows AND a robots
       Disallow, so this is a real, not just under-crawled, gap), ``null``
       (no metafact yet) -> fall through to layer 2.
    2. **Our own curator crawl** (discovered_sources) — degrade
       gracefully when the registry is unreachable or has no opinion yet.
       Guesses a display name from the slug (title-case —
       "simply_healthcare" -> "Simply Healthcare"), matches
       case-insensitively against discovered_sources.payer, then takes
       the dominant host among that payer's fetchable rows. Requires
       >= _MIN_PAYER_ROWS matching rows AND >= _MIN_PAYER_ROWS of those
       on the single dominant host, so a handful of stray/mixed rows
       can't produce a false-confidence domain.

    Either layer returning nothing usable is the same "we don't know"
    outcome — (None, None), same as an unlisted payer always had. Never
    invents a domain either way. Pushing our own crawl evidence back to
    the registry (the POST half of the collaboration) belongs to the
    nightly curator pipeline, not this per-request read path — a live
    query shouldn't pay for or depend on a write.
    """
    if not slug:
        return None, None
    display_name = slug.replace("_", " ").title()

    try:
        import httpx
        async with httpx.AsyncClient(timeout=_PAYOR_PLATFORM_TIMEOUT_S) as client:
            resp = await client.get(
                f"{_PAYOR_PLATFORM_BASE}/api/registry/payors/"
                f"{urllib.parse.quote(display_name)}/web-domain"
            )
        if resp.status_code == 200:
            data = resp.json()
            crawlable = data.get("crawlable")
            host = data.get("host")
            if crawlable is True and host:
                return host, display_name
            if crawlable is False:
                return None, None
            # crawlable is null (no metafact yet) -> fall through below.
    except Exception as exc:
        logger.warning("(d) payor platform web-domain lookup failed: %s", exc)

    try:
        from sqlalchemy import text as sql_text
        rows = (await db.execute(sql_text(
            """
            SELECT payer, url FROM discovered_sources
            WHERE lower(payer) = lower(:name) AND last_fetch_status = 200
            """
        ), {"name": display_name})).mappings().all()
    except Exception as exc:
        logger.warning("(d) payer context resolution failed: %s", exc)
        return None, None
    if len(rows) < _MIN_PAYER_ROWS:
        return None, None
    hosts = Counter(
        (urllib.parse.urlparse(r["url"]).netloc or "").removeprefix("www.")
        for r in rows
    )
    if not hosts:
        return None, None
    dominant_host, count = hosts.most_common(1)[0]
    if not dominant_host or count < _MIN_PAYER_ROWS:
        return None, None
    return dominant_host, rows[0]["payer"]


# 2026-07-07: NOT the same bar as corpus_search_agent's
# _SELECTIVITY_REQUIRED (0.65) — that threshold answers "is this tag
# selective enough to be a safe REQUIRED filter for OUR OWN corpus
# retrieval," a different, lower-stakes question. This one answers "is
# this phrase rare enough that quoting it on the PUBLIC WEB, unanchored
# to any domain, won't drift onto generic/competitor content" — checked
# against the full distribution of 1,612 distinct d-tags across 9,130
# docs: median selectivity is 0.997 (the vast majority of tags are
# ordinary/specific), with a sharp knee around the 10th percentile
# (~0.94) before a long common/generic tail. The two known-bad unanchored
# cases (utilization_management.prior_authorization=0.819,
# health_care_services.therapy_services=0.760) sit well below the 5th
# percentile (0.892); the one known-good case
# (claims.timely_filing=0.994) sits at the median — an ordinary tag, not
# an outlier. 0.95 draws the line just above that knee: only the
# rarest ~8-10% of tags in the whole corpus clear it.
_WEB_QUOTE_SELECTIVITY_MIN = 0.95


async def build_authoritative_query(
    db: AsyncSession, raw_query: str, tag_matches: list[str] | None, domain: str | None,
    partition: Any = None,
) -> tuple[str, str | None, list[str]]:
    """Return (query, site_domain_or_None, exact_terms — a list, possibly
    empty). The raw query text is passed through UNCHANGED here —
    site/exactTerms are resolved separately and applied as real
    search-engine syntax by ``_embed_search_operators``, not string
    concatenation. ``domain`` comes from ``_resolve_payer_context``
    (data-driven), never invented here.

    ``partition`` is the ``TermPartition`` the router already computes
    once per query via ``partition_terms`` (corpus_search_agent.py) —
    reusing it instead of re-deriving "what's important in this query"
    from scratch. It already separates:
      - literal anchors (regex-extracted codes like "H0015") — always
        REQUIRED, selectivity=1.0 by construction. These are precise
        identifiers, the single best kind of must-have term, and used
        unconditionally regardless of domain.
      - REQUIRED lexicon tags — cleared partition_terms's own bar
        (0.65), which answers "safe to require for OUR corpus
        retrieval." That's a different, lower-stakes question than "safe
        to quote unanchored on the public web" (see
        _WEB_QUOTE_SELECTIVITY_MIN's comment) — so a REQUIRED tag only
        becomes a must-have here without a domain if it ALSO clears the
        stricter web bar. With a domain, use it regardless — the site
        restriction already anchors it.
    Multiple terms can combine (e.g. a literal anchor AND a REQUIRED tag
    phrase, each independently quoted) — real search engines AND multiple
    quoted segments together.

    Falls back to the single-most-specific-tag path (2026-07-07 original
    design) when no partition was computed upstream (e.g. pre-route pool
    build failed) — same selectivity bar, just scoped to one candidate
    instead of the full per-query partition.
    """
    exact_terms: list[str] = []
    if partition is not None:
        exact_terms.extend(
            t.term for t in partition.required if t.kind == "literal"
        )
        for t in partition.required:
            if t.kind != "tag" or not t.full_code:
                continue
            if domain or t.selectivity >= _WEB_QUOTE_SELECTIVITY_MIN:
                exact_terms.append(_tag_to_phrase(t.full_code))
    else:
        candidate_tag = _most_specific_d_tag(tag_matches)
        if candidate_tag:
            if domain:
                exact_terms.append(_tag_to_phrase(candidate_tag))
            else:
                from app.services.corpus_search_agent import selectivity_for_tag
                sel = await selectivity_for_tag(db, candidate_tag)
                if sel >= _WEB_QUOTE_SELECTIVITY_MIN:
                    exact_terms.append(_tag_to_phrase(candidate_tag))
    seen: set[str] = set()
    exact_terms = [t for t in exact_terms if not (t in seen or seen.add(t))]
    return raw_query, domain, exact_terms


def _rerank_hits(
    hits: list[_SearchHit], site_domain: str | None, exact_terms: list[str],
) -> list[_SearchHit]:
    """Second layer of defense on top of search-engine operators. ``site:``
    and quoted phrases steer the engine's own ranking, but generic
    aggregator/SEO-farm results (verified 2026-07-07: medicalbillingrcm.com,
    studyquiz.blog, payerlookup.com) can still slip in below the real
    payer pages, especially when no verified domain exists yet to
    ``site:``-restrict to. Score independently of what the engine
    returned first: +2 for matching the known payer domain, +1 for EACH
    exact_terms phrase actually appearing in title/snippet (0-N terms can
    apply now that must-haves are a list, not a single phrase). Stable
    sort — ties keep the engine's original order, this only promotes
    verified matches, never invents relevance for a hit that has neither
    signal.
    """
    if not site_domain and not exact_terms:
        return hits

    def _score(h: _SearchHit) -> int:
        s = 0
        if site_domain and site_domain.lower() in h.url.lower():
            s += 2
        text = f"{h.title} {h.snippet}".lower()
        s += sum(1 for term in exact_terms if term.lower() in text)
        return s

    return sorted(hits, key=_score, reverse=True)


def _embed_search_operators(
    raw_query: str, site_domain: str | None, exact_terms: list[str],
) -> str:
    """Append real search-engine operators to the query text. Verified
    2026-07-07 against live DuckDuckGo results: unquoted "timely filing"
    drifted onto other payers' identical generic content (genmeditech,
    happybilling, muni.health), quoted '"timely filing"' pulled Sunshine
    Health's own pages back to the top. ``site:`` restricted results to
    the payer's domain with zero off-domain hits. Both engines parse
    these as operators, not literal keywords — this is what makes it
    safe where the earlier qualifier-concatenation regression wasn't:
    no new words are being added for the engine to match against, just
    a scope restriction on the words already there. Multiple exact_terms
    (e.g. a literal anchor code AND a REQUIRED tag phrase) each get their
    own quoted segment — real search engines AND them together.
    """
    q = raw_query
    for term in exact_terms:
        q = f'{q} "{term}"'
    if site_domain:
        q = f"{q} site:{site_domain}"
    return q


# ---------------------------------------------------------------------------
# Sitemap check — before blind Google search, check discovered_sources
# (the curator's own sitemap/BFS crawl, 11k+ rows) for a URL we already
# know exists for this payer + topic. Preference goes to rows that are
# NOT yet ingested but ARE fetchable (last_fetch_status=200) — that's
# the genuine gap-fill case: a real page our own crawl found, that the
# corpus hasn't absorbed yet. Also includes already-ingested URLs,
# since (d) reads the WHOLE page text rather than chunk-level retrieval,
# so it can succeed even where corpus chunking/rerank missed the
# relevant section.
# ---------------------------------------------------------------------------

async def _lookup_sitemap_candidates(
    db: AsyncSession, tag_matches: list[str] | None, display_name: str | None, *, limit: int = 3,
) -> list[str]:
    """``display_name`` comes from ``_resolve_payer_context`` — the caller
    resolves it once and shares it with the search-operator path, rather
    than each doing its own separate (and potentially inconsistent)
    payer-name lookup."""
    if not display_name:
        return []
    d_tags = [t for t in (tag_matches or []) if t.startswith("d:")]
    keywords: list[str] = []
    for prefix, kws in _D_TAG_URL_KEYWORDS:
        if any(dt == f"d:{prefix}" or dt.startswith(f"d:{prefix}.") for dt in d_tags):
            keywords = kws
            break
    if not keywords:
        return []
    try:
        from sqlalchemy import text as sql_text
        like_clauses = " OR ".join(f"url ILIKE :kw{i}" for i in range(len(keywords)))
        params = {f"kw{i}": f"%{kw}%" for i, kw in enumerate(keywords)}
        params["payer"] = display_name
        params["limit"] = limit
        rows = (await db.execute(sql_text(
            f"""
            SELECT url FROM discovered_sources
            WHERE payer = :payer
              AND last_fetch_status = 200
              AND curation_status NOT IN ('noise', 'stale')
              AND ({like_clauses})
            ORDER BY ingested DESC, last_seen_at DESC
            LIMIT :limit
            """
        ), params)).mappings().all()
        return [r["url"] for r in rows]
    except Exception as exc:
        logger.warning("(d) sitemap lookup failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Step 1: Search — Vertex AI Grounding with Google Search (primary),
# mobius-skills google-search / DuckDuckGo (fallback)
# ---------------------------------------------------------------------------

_VERTEX_GROUNDING_MODEL = "gemini-2.5-flash"


async def _search_web_vertex(
    query: str, *, n: int = 5,
    site: str | None = None, exact: list[str] | None = None,
) -> list[_SearchHit]:
    """Primary search backend for (d), 2026-07-07 — Vertex AI's "Grounding
    with Google Search" Gemini feature, not the old Custom Search JSON API
    (closed to new customers — see project_google_cse_closed_new_customers
    memory; this is a different Vertex AI product path and hit no such
    restriction). Uses infrastructure already wired for LLM synthesis: no
    new GCP API to enable (aiplatform.googleapis.com already on), no new
    package (google-genai is already a transitive dep of
    google-cloud-aiplatform). Native async client (``client.aio``) — no
    thread involved, unlike the urllib-based DDG path below.

    ``site``/``exact`` are embedded as real search operators in the prompt
    text (the same site:/quoted-phrase syntax as _embed_search_operators),
    not a structured API field — grounding has none. Verified live
    (2026-07-07): unlike a hard site: filter, a wrong/guessed domain
    degrades gracefully — the model still finds and grounds on the REAL
    correct source rather than returning nothing, which is a materially
    safer failure mode than DuckDuckGo's site: given our domain
    resolution is itself a best-effort guess.

    grounding_chunks only exposes {domain, title=domain, uri} — no
    per-source snippet — so hits come back with snippet="" (same as
    sitemap-derived hits below); real page text is filled in by
    _fetch_and_extract same as any other hit. Returned URIs are Google's
    own redirect-proxy links; verified they resolve to the real page
    server-side, so no changes needed downstream.
    """
    prompt = query
    for term in (exact or []):
        prompt = f'{prompt} "{term}"'
    if site:
        prompt = f"{prompt} site:{site}"

    try:
        from google import genai
        from google.genai.types import GenerateContentConfig, GoogleSearch, Tool

        project = os.environ.get("VERTEX_PROJECT_ID", "").strip()
        location = os.environ.get("VERTEX_LOCATION", "us-central1").strip()
        if not project:
            logger.warning("(d) VERTEX_PROJECT_ID not set; skipping vertex grounding search")
            return []

        client = genai.Client(vertexai=True, project=project, location=location)
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
        logger.warning("(d) vertex grounding search failed: %s", exc)
        return []


async def _search_web(
    query: str, *, n: int = 5,
    site: str | None = None, exact: str | None = None,
) -> list[_SearchHit]:
    """Call the shared google-search skill. Returns up to n hits.

    ``site``/``exact`` are real Google CSE fields (siteSearch/exactTerms),
    forwarded as separate query params — NOT concatenated into ``query``.
    """
    base = os.environ.get("CHAT_SKILLS_GOOGLE_SEARCH_URL", "").strip()
    if not base:
        logger.warning(
            "CHAT_SKILLS_GOOGLE_SEARCH_URL not set; (d) cannot run web search"
        )
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
            req = urllib.request.Request(
                url, headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read().decode()
            payload = json.loads(data)
            if isinstance(payload, list):
                items = payload
            elif isinstance(payload, dict):
                items = payload.get("items") or payload.get("results") or []
            else:
                items = []
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
            logger.warning("(d) web search failed: %s", exc)
            return []

    # urllib is sync; run in thread to avoid blocking the loop.
    return await asyncio.to_thread(_do_request)


# ---------------------------------------------------------------------------
# Step 2: Fetch + extract
# ---------------------------------------------------------------------------

async def _fetch_and_extract(hit: _SearchHit) -> _Passage:
    """Fetch one URL with timeout, extract main text. Returns a _Passage
    with status tracking so the caller can see what happened.
    """
    t_start = time.monotonic()
    try:
        import httpx  # async HTTP
        async with httpx.AsyncClient(
            timeout=_FETCH_TIMEOUT_S,
            follow_redirects=True,
            headers={
                # Browser-like UA reduces bot-detection 403s on policy
                # sites. We're not crawling at scale and respect the
                # implicit terms (read-only, single fetch per search).
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
                    text="", fetch_status=f"http_{resp.status_code}",
                    fetch_ms=elapsed,
                )
            content_type = (resp.headers.get("content-type") or "").lower()
            body_bytes = resp.content
            html = resp.text
    except asyncio.TimeoutError:
        return _Passage(
            url=hit.url, title=hit.title, snippet=hit.snippet,
            text="", fetch_status="timeout",
            fetch_ms=int((time.monotonic() - t_start) * 1000),
        )
    except Exception as exc:
        logger.warning("(d) fetch failed url=%s: %s", hit.url, exc)
        return _Passage(
            url=hit.url, title=hit.title, snippet=hit.snippet,
            text="", fetch_status=f"error:{type(exc).__name__}",
            fetch_ms=int((time.monotonic() - t_start) * 1000),
        )

    # PDF branch — content_type=application/pdf or URL ends in .pdf or
    # body starts with the PDF magic bytes ``%PDF``. Use PyMuPDF
    # (already a dependency for ingest extraction) to pull plain text
    # page-by-page. This fixes the failure mode where DuckDuckGo's top
    # hit was a real authoritative PDF (e.g. Aetna Better Health FL
    # Provider Training) but the HTML extractor returned binary garbage
    # like ``%PDF-1.6 %����…`` and the synthesis LLM had nothing to read.
    is_pdf = (
        "application/pdf" in content_type
        or hit.url.lower().endswith(".pdf")
        or body_bytes[:4] == b"%PDF"
    )
    if is_pdf:
        try:
            import fitz   # PyMuPDF
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
                    text=hit.snippet,
                    fetch_status="pdf_empty",
                    fetch_ms=elapsed,
                )
            # Use the canonical "ok" status so downstream filters that
            # gate on exactly fetch_status == "ok" surface this passage.
            # Track the source separately if needed via the URL ext.
            return _Passage(
                url=hit.url, title=hit.title, snippet=hit.snippet,
                text=text, fetch_status="ok", fetch_ms=elapsed,
            )
        except Exception as exc:
            logger.warning("(d) pdf extract failed url=%s: %s", hit.url, exc)
            return _Passage(
                url=hit.url, title=hit.title, snippet=hit.snippet,
                text=hit.snippet,
                fetch_status=f"pdf_extract_failed:{type(exc).__name__}",
                fetch_ms=int((time.monotonic() - t_start) * 1000),
            )

    # Extract main text using existing extractor.
    try:
        from app.services.html_extractor import extract_sections
        sections = extract_sections(html, source_url=hit.url)
        # Concatenate section bodies, prefer the longest few.
        bodies: list[str] = []
        for s in sections:
            body = (s.get("text") or "").strip()
            if body:
                bodies.append(body)
        # Take up to _MAX_PASSAGE_CHARS, prefer longer sections first.
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
            # Fallback: strip HTML tags crudely.
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()[:_MAX_PASSAGE_CHARS]
        elapsed = int((time.monotonic() - t_start) * 1000)
        return _Passage(
            url=hit.url, title=hit.title, snippet=hit.snippet,
            text=text, fetch_status="ok", fetch_ms=elapsed,
        )
    except Exception as exc:
        logger.warning("(d) extract failed url=%s: %s", hit.url, exc)
        return _Passage(
            url=hit.url, title=hit.title, snippet=hit.snippet,
            text=hit.snippet,    # fall back to search-result snippet
            fetch_status=f"extract_failed:{type(exc).__name__}",
            fetch_ms=int((time.monotonic() - t_start) * 1000),
        )


# ---------------------------------------------------------------------------
# Step 3: LLM synthesis
# ---------------------------------------------------------------------------

_SYNTHESIS_SYSTEM = (
    "You are a research assistant. The user has asked a question and we "
    "have fetched several web pages that may contain the answer. Your "
    "job is to write a brief answer (3 sentences max) using ONLY the "
    "passages provided. Cite each claim by passage number [1], [2], etc.\n\n"
    "OUTPUT FORMAT — strict JSON:\n"
    "{\n"
    '  "answer": "<your answer with [N] citations>",\n'
    '  "used_passages": [<integers — which passages you actually cited>],\n'
    '  "confidence": "high" | "medium" | "low"\n'
    "}\n\n"
    "Rules:\n"
    "- Use only the passages; do not draw on other knowledge.\n"
    "- If the passages don't answer the question, say so and emit "
    'confidence "low".\n'
    "- Quote exact phrasing from passages when possible."
)


async def _synthesize(
    query: str,
    passages: list[_Passage],
    *,
    correlation_id: str | None = None,
) -> tuple[str, list[int], str, dict[str, Any]]:
    """Ask the LLM to synthesize an answer from passages. Returns
    (answer, used_passage_indices, confidence_label, telemetry).
    """
    # Build user prompt with numbered passages.
    usable = [p for p in passages if p.text.strip()]
    if not usable:
        return (
            "No external sources could be fetched or extracted to answer "
            "this question.",
            [],
            "low",
            {"llm_ms": 0, "no_passages": True},
        )
    parts = [f"Question: {query}", ""]
    for i, p in enumerate(usable, start=1):
        parts.append(
            f"[{i}] Title: {p.title}\nURL: {p.url}\n\n{p.text}"
        )
        parts.append("")
    user_prompt = "\n".join(parts)

    t0 = time.monotonic()
    raw, llm_meta = await llm_manager_client.generate(
        system=_SYNTHESIS_SYSTEM,
        user=user_prompt,
        stage="rag_strategy_d_external",
        max_tokens=1024,
        correlation_id=correlation_id,
    )
    elapsed = int((time.monotonic() - t0) * 1000)

    text = (raw or "").strip()
    # Strip markdown fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = json.loads(m.group()) if m else {}

    answer = (parsed.get("answer") or "").strip() or text
    used = parsed.get("used_passages") or []
    if not isinstance(used, list):
        used = []
    confidence = (parsed.get("confidence") or "low").lower()
    if confidence not in ("high", "medium", "low"):
        confidence = "low"

    return answer, [int(i) for i in used if isinstance(i, int)], confidence, {
        "llm_ms": elapsed,
        "llm_meta": llm_meta,
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

async def strategy_d_external(
    db: AsyncSession,
    raw_query: str,
    *,
    agent_id: str,
    correlation_id: str | None = None,
    n_search: int = 5,
    n_fetch: int = _MAX_FETCH,
    tag_matches: list[str] | None = None,
    partition: Any = None,
) -> StrategyDResult:
    """Run Strategy (d). Returns answer + scraped passages.

    ``partition`` is the ``TermPartition`` the caller already computed via
    ``partition_terms`` for routing purposes (corpus_search_agent.py) —
    passed through so must-have terms reuse the router's own per-query
    importance ranking (literal anchors + REQUIRED tags) instead of a
    separate, narrower derivation. Optional — falls back gracefully to a
    single-tag heuristic when not provided (see build_authoritative_query).
    """
    t_start = time.monotonic()

    # Step 1: Search. Enrich the raw query with a document-type qualifier
    # (provider manual / PA requirements / etc, derived from d-tags) and,
    # if we have a VERIFIED official domain for the named payer, try a
    # site-restricted + exactTerms-constrained search first (real CSE API
    # fields, not string concatenation — see build_authoritative_query).
    # Fall back to the plain, unrestricted query if that comes back empty
    # — never let a guessed/wrong domain or an over-strict exactTerms
    # silently zero out results.
    payer_slug = _extract_payer_slug(tag_matches)
    site_domain, payer_display_name = await _resolve_payer_context(db, payer_slug)
    query_text, site_domain, exact_terms = await build_authoritative_query(
        db, raw_query, tag_matches, site_domain, partition,
    )
    t_search = time.monotonic()

    # Check our own sitemap crawl first — a URL we already know exists
    # for this payer+topic beats hoping Google resurfaces it. Shares the
    # same resolved payer_display_name as the search-operator path above
    # (one resolution, not two potentially-inconsistent lookups).
    sitemap_urls = await _lookup_sitemap_candidates(db, tag_matches, payer_display_name)
    hits: list[_SearchHit] = [
        _SearchHit(title="", snippet="", url=u) for u in sitemap_urls
    ]
    if sitemap_urls:
        logger.info(
            "[%s] [trace:d:sitemap] n_candidates=%d urls=%s",
            agent_id, len(sitemap_urls), sitemap_urls,
        )

    # Primary: Vertex AI grounding (2026-07-07 — see _search_web_vertex).
    # Fall back to the mobius-skills google-search / DuckDuckGo path only
    # if Vertex errors or returns nothing — never let one backend being
    # briefly unavailable stall (d) entirely.
    web_hits: list[_SearchHit] = await _search_web_vertex(
        query_text, n=n_search, site=site_domain, exact=exact_terms,
    )
    if web_hits:
        logger.info(
            "[%s] [trace:d:search] vertex grounding hit query=%r site=%s exact=%s n_hits=%d",
            agent_id, query_text, site_domain, exact_terms, len(web_hits),
        )
    else:
        logger.info("[%s] [trace:d:search] vertex grounding empty/failed, falling back to DDG", agent_id)
        if site_domain or exact_terms:
            operator_query = _embed_search_operators(query_text, site_domain, exact_terms)
            # Structured site=/exact= params are also passed for the Google
            # CSE path (currently unusable on this project — see
            # project_google_cse_closed_new_customers memory), but the
            # operators embedded in operator_query are what actually take
            # effect on the live DuckDuckGo fallback.
            web_hits = await _search_web(
                operator_query, n=n_search, site=site_domain,
                exact=" ".join(exact_terms) if exact_terms else None,
            )
            if web_hits:
                logger.info(
                    "[%s] [trace:d:search] constrained hit query=%r site=%s exact=%s n_hits=%d",
                    agent_id, operator_query, site_domain, exact_terms, len(web_hits),
                )
        if not web_hits:
            web_hits = await _search_web(query_text, n=n_search)

    web_hits = _rerank_hits(web_hits, site_domain, exact_terms)

    # Sitemap candidates first (highest trust), then web hits, deduped by URL.
    seen_urls = {h.url for h in hits}
    hits.extend(h for h in web_hits if h.url not in seen_urls)
    search_ms = int((time.monotonic() - t_search) * 1000)
    logger.info(
        "[%s] [trace:d:search] query=%r site=%s exact=%s n_sitemap=%d n_web=%d total=%d elapsed=%dms",
        agent_id, query_text, site_domain, exact_terms, len(sitemap_urls), len(web_hits), len(hits), search_ms,
    )

    if not hits:
        return StrategyDResult(
            llm_answer=(
                "Web search returned no results (or the search service "
                "is not configured). Cannot run external research."
            ),
            passages=[],
            telemetry={
                "search_ms": search_ms,
                "fetch_ms": 0,
                "llm_ms": 0,
                "total_ms": int((time.monotonic() - t_start) * 1000),
                "n_hits": 0,
            },
        )

    # Step 2: Parallel fetch + extract.
    t_fetch = time.monotonic()
    passages = await asyncio.gather(
        *[_fetch_and_extract(h) for h in hits[:n_fetch]],
        return_exceptions=False,
    )
    fetch_ms = int((time.monotonic() - t_fetch) * 1000)
    n_ok = sum(1 for p in passages if p.fetch_status == "ok")
    logger.info(
        "[%s] [trace:d:fetch] n_attempted=%d n_ok=%d elapsed=%dms statuses=%s",
        agent_id, len(passages), n_ok, fetch_ms,
        [p.fetch_status for p in passages],
    )

    # Step 3: LLM synthesis over passages.
    answer, used_idx, conf, llm_tel = await _synthesize(
        raw_query, list(passages), correlation_id=correlation_id,
    )
    logger.info(
        "[%s] [trace:d:synthesize] used_passages=%s conf=%s llm_ms=%d",
        agent_id, used_idx, conf, llm_tel.get("llm_ms"),
    )

    return StrategyDResult(
        llm_answer=answer,
        passages=list(passages),
        telemetry={
            "search_ms": search_ms,
            "fetch_ms": fetch_ms,
            "llm_ms": llm_tel.get("llm_ms", 0),
            "total_ms": int((time.monotonic() - t_start) * 1000),
            "n_hits": len(hits),
            "n_fetched": len(passages),
            "n_ok": n_ok,
            "used_passage_indices": used_idx,
            "synthesis_confidence": conf,
        },
    )
