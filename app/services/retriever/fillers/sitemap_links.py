"""Sitemap-links lookup -- suggested-link enrichment for the answer engine.

NOT a strategy-lettered filler, and never was one -- correction from
Retriever (2026-07-23, see docs/rag-agents/filler-f-sitemap-kickoff.md):
Router's real strategy `"f"` is a separate, unbuilt, scored bandit arm
("external fallback, similar to d") that has its own seeded priors in
`router-build-spec.md`. This module was mistakenly labeled with that same
letter early on; it was never that strategy. This module's output has no
score, no slot occupancy, and no citation role -- it's an inert
`suggested_links` field riding through the normal Fillers -> Router ->
Synthesis -> Contract pipeline unchanged (Chat's ruling, Option A), not a
retrieval attempt competing for a slot.

Ported (not imported) from corpus_search_strategy_d.py's
_lookup_sitemap_candidates() -- verified fresh against that source, not
copy-pasted. Returns candidate URLs already known from the curator's own
sitemap/BFS crawl (`discovered_sources`), no fetch, no LLM call.

Output shape (`SuggestedLink`) and its exact placement in the
Retriever -> Contract chain are per Chat's ruling (Option A: new top-level
field on FilledShape, passed through unchanged) -- still pending DB's
sign-off on the field name/shape in contracts.py. This module is
deliberately NOT wired into any orchestrator dispatch yet; the trigger
condition (when it runs relative to Pool/slots) is still in flux with the
Observer redesign, per Retriever.
"""

from __future__ import annotations

import logging
import urllib.parse
from dataclasses import dataclass

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class SuggestedLink:
    """One link-only result -- no fetched content, ever. Distinct from
    FilledChunk on purpose: this is never scored, never counted toward slot
    occupancy, and never enters the grounding-badge calculation (Chat's
    ruling, 2026-07-23)."""

    url: str
    title: str | None = None


# d-tag prefix -> URL-path keyword. discovered_sources.topic_tags is NULL
# across the board (verified against legacy's own comment, still true as of
# this port), so matching happens on URL path instead.
_D_TAG_URL_KEYWORDS: list[tuple[str, list[str]]] = [
    ("claims.timely_filing", ["timely", "filing"]),
    ("utilization_management.prior_authorization", ["preauth", "prior-auth", "authorization"]),
    ("utilization_management", ["preauth", "prior-auth", "authorization"]),
    ("pharmacy", ["pharmacy"]),
    ("eligibility", ["eligib"]),
    ("disputes", ["appeal", "grievance"]),
    ("claims", ["claims"]),
]


def _derive_title_from_url(url: str) -> str | None:
    """Path-slug -> scannable title, e.g.
    "/providers/prior-authorization-requirements" -> "Prior Authorization
    Requirements". Per Chat's ruling: discovered_sources carries no title of
    any kind, so this is the only non-null title source available for v1.
    Not real content -- a display-only best-effort label."""
    path = urllib.parse.urlparse(url).path
    segments = [s for s in path.split("/") if s]
    if not segments:
        return None
    slug = segments[-1]
    # Strip a file extension if present (e.g. "policy.pdf" -> "policy").
    slug = slug.rsplit(".", 1)[0] if "." in slug else slug
    words = [w for w in slug.replace("-", " ").replace("_", " ").split(" ") if w]
    if not words:
        return None
    return " ".join(w.capitalize() for w in words)


def _keywords_for_tags(tag_matches: list[str] | None) -> list[str]:
    d_tags = [t for t in (tag_matches or []) if t.startswith("d:")]
    for prefix, kws in _D_TAG_URL_KEYWORDS:
        if any(dt == f"d:{prefix}" or dt.startswith(f"d:{prefix}.") for dt in d_tags):
            return kws
    return []


async def lookup_sitemap_links(
    db: AsyncSession,
    tag_matches: list[str] | None,
    payer_display_name: str | None,
    *,
    limit: int = 3,
) -> list[SuggestedLink]:
    """Check discovered_sources for URLs already known to exist for this
    payer + topic, before any live web search is attempted elsewhere.

    Gate conditions (preserved from legacy, don't drop): no resolved payer
    display name -> []. No matching d-tag prefix -> [] (nothing to filter
    the URL-keyword match on).
    """
    if not payer_display_name:
        return []
    keywords = _keywords_for_tags(tag_matches)
    if not keywords:
        return []
    try:
        like_clauses = " OR ".join(f"url ILIKE :kw{i}" for i in range(len(keywords)))
        params = {f"kw{i}": f"%{kw}%" for i, kw in enumerate(keywords)}
        params["payer"] = payer_display_name
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
    except Exception as exc:
        logger.warning("filler f: sitemap lookup failed: %s", exc)
        return []
    return [
        SuggestedLink(url=r["url"], title=_derive_title_from_url(r["url"]))
        for r in rows
    ]
