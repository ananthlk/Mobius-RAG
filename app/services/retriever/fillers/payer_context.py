"""Shared payer-context resolution for the live-external fillers (c/d/s) and
the sitemap-links suggested-link lookup (see sitemap_links.py -- not a
strategy-lettered filler, Retriever corrected the earlier "Filler f" label).

Ported (not imported) from ``corpus_search_strategy_d.py``'s
``_extract_payer_slug``/``_resolve_payer_context`` — verified fresh against
that source rather than copy-pasted, per the fleet's port-don't-import
directive. Built once as shared infra because multiple consumers
independently need the same (payer slug -> site_domain, display_name,
crawlable) resolution and nothing upstream in the new retriever chain
resolves it (confirmed via grep across app/services/retriever/ before this
file existed). Known real consumers as of 2026-07-23: Filler s
(`extract_payer_slug` only), Filler d (planned, blocked on their own `url`
field), sitemap_links (this module's sibling).

``PayerContext.crawlable`` exists for Router's crawl-gate
(`RoutingContext.payer_crawlable`) -- see the dataclass docstring for the
tri-state semantics Router specifically needs. ``.source`` is a derived
property, not stored state (Router's own withdrawal, 2026-07-23, once they
saw it's fully recoverable from crawlable/site_domain -- see the property's
docstring).

See docs/rag-agents/filler-f-sitemap-kickoff.md for the coordination trail
(filename kept stable for other sessions' existing links despite the
identity correction).
"""

from __future__ import annotations

import logging
import os
import urllib.parse
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class PayerContext:
    """Resolved payer context. ``crawlable`` is a genuine tri-state, kept
    distinct from "unknown" -- Router's crawl-gate (RoutingContext.payer_crawlable)
    needs to tell an explicit robots-disallow (False, disqualifies strategy d)
    apart from no-registry-opinion-yet (None, fails open). Collapsing both into
    one "we don't know" value (as the discovered_sources-fallback-only view
    of this data always has, and as this function's own site_domain/display_name
    fields still do for simpler consumers) would silently break that gate.

    ``slug`` added per Router's ask (relayed via Filler d, 2026-07-23) so the
    result is self-contained for threading/telemetry without the caller
    separately tracking the input."""

    slug: str | None
    display_name: str | None
    site_domain: str | None
    crawlable: bool | None  # True/False from a live registry verdict; None = no registry opinion either way (unreachable, no metafact, or resolved via the discovered_sources fallback only)

    @property
    def source(self) -> str:
        """"metafact" (registry gave an explicit verdict) | "crawl_history"
        (resolved via the discovered_sources fallback) | "none" (nothing
        resolved). Originally requested as a stored field so Eval could
        weight a registry-confirmed verdict differently from a
        crawl-history-inferred one -- Router withdrew that once they saw
        this function's actual behavior makes it fully derivable and
        therefore not worth storing as separate, driftable state:
        ``crawlable`` is only ever non-None on an explicit registry verdict
        (this function never falls through to the discovered_sources layer
        once the registry has answered), so ``crawlable is not None``
        implies "metafact" unambiguously; the discovered_sources layer has
        no robots signal at all and can only ever land here with
        ``crawlable=None``, so a resolved ``site_domain`` with no
        ``crawlable`` verdict implies "crawl_history"."""
        if self.crawlable is not None:
            return "metafact"
        if self.site_domain is not None:
            return "crawl_history"
        return "none"

# A single stray discovered_sources row can't manufacture a "verified" domain.
_MIN_PAYER_ROWS = 3

_PAYOR_PLATFORM_BASE = os.environ.get(
    "MOBIUS_PAYOR_URL", "https://mobius-payor-ortabkknqa-uc.a.run.app",
).rstrip("/")
_PAYOR_PLATFORM_TIMEOUT_S = 3.0


def extract_payer_slug(tag_matches: list[str] | None) -> str | None:
    """Pull the ``j:payor.<slug>`` tag out of a query's tag matches, if any."""
    for t in (tag_matches or []):
        if t.startswith("j:payor."):
            return t.split("j:payor.", 1)[1]
    return None


async def resolve_payer_context(
    db: AsyncSession, slug: str | None,
) -> PayerContext:
    """Resolve site_domain / display_name / crawlable for a payer slug.

    Two layers, in order, same as the legacy implementation:

    1. Payor Platform registry (authoritative) — a live HTTP call returning a
       tri-state ``crawlable`` verdict. ``true`` + a host -> safe to
       site-restrict. ``false`` -> explicitly don't (robots disallows), but
       ``display_name`` is still returned (it's derived from the slug, not
       from crawlability -- unlike legacy/the tuple-returning predecessor of
       this function, which discarded it too; no known consumer relied on
       that discard, and Router's crawl-gate wants ``crawlable`` and
       ``display_name`` as independent signals).
       ``null``/unreachable -> fall through to layer 2, ``crawlable`` stays
       ``None`` (no registry opinion).
    2. Our own curator crawl (``discovered_sources``) — degrade gracefully.
       Matches case-insensitively against ``discovered_sources.payer``,
       requires >= ``_MIN_PAYER_ROWS`` rows on a single dominant host so a
       handful of stray/mixed rows can't produce a false-confidence domain.
       This layer has no robots-based signal at all -- ``crawlable`` stays
       ``None`` even when a fallback host is found.

    ``crawlable`` is only ever ``True``/``False`` when the registry itself
    gave an explicit verdict; every other path (no slug, registry
    unreachable, fallback-only resolution, fallback below the row
    threshold) is ``None`` -- "no opinion," not "confirmed not crawlable."
    Callers that need to fail-open on unknown crawlability (Router's
    crawl-gate) must treat only ``crawlable is False`` as a disqualifier.
    """
    if not slug:
        return PayerContext(slug=slug, site_domain=None, display_name=None, crawlable=None)
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
                return PayerContext(slug=slug, site_domain=host, display_name=display_name, crawlable=True)
            if crawlable is False:
                return PayerContext(slug=slug, site_domain=None, display_name=display_name, crawlable=False)
            # crawlable is null (no metafact yet) -> fall through below.
    except Exception as exc:
        logger.warning("payer context: payor platform web-domain lookup failed: %s", exc)

    try:
        rows = (await db.execute(sql_text(
            """
            SELECT payer, url FROM discovered_sources
            WHERE lower(payer) = lower(:name) AND last_fetch_status = 200
            """
        ), {"name": display_name})).mappings().all()
    except Exception as exc:
        logger.warning("payer context: discovered_sources fallback failed: %s", exc)
        return PayerContext(slug=slug, site_domain=None, display_name=None, crawlable=None)
    if len(rows) < _MIN_PAYER_ROWS:
        return PayerContext(slug=slug, site_domain=None, display_name=None, crawlable=None)
    hosts = Counter(
        (urllib.parse.urlparse(r["url"]).netloc or "").removeprefix("www.")
        for r in rows
    )
    if not hosts:
        return PayerContext(slug=slug, site_domain=None, display_name=None, crawlable=None)
    dominant_host, count = hosts.most_common(1)[0]
    if not dominant_host or count < _MIN_PAYER_ROWS:
        return PayerContext(slug=slug, site_domain=None, display_name=None, crawlable=None)
    return PayerContext(slug=slug, site_domain=dominant_host, display_name=rows[0]["payer"], crawlable=None)
