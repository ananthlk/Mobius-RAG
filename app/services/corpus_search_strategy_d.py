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
# Step 1: Search via mobius-skills google-search
# ---------------------------------------------------------------------------

async def _search_web(query: str, *, n: int = 5) -> list[_SearchHit]:
    """Call the shared google-search skill. Returns up to n hits."""
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
) -> StrategyDResult:
    """Run Strategy (d). Returns answer + scraped passages."""
    t_start = time.monotonic()

    # Step 1: Search.
    t_search = time.monotonic()
    hits = await _search_web(raw_query, n=n_search)
    search_ms = int((time.monotonic() - t_search) * 1000)
    logger.info(
        "[%s] [trace:d:search] n_hits=%d elapsed=%dms",
        agent_id, len(hits), search_ms,
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
