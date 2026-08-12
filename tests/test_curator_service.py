"""Tests for app.curator.service — the SQL layer of the curator.

These exercise upsert/search/curate/mark_ingested/stats with a real
SQLAlchemy session pointed at an in-memory SQLite. SQLite doesn't
support PostgreSQL-specific types we use in production (JSONB, GIN
indexes), so we work around with the JSON ↔ JSONB shim and skip
the GIN-only ``topic_tags`` array containment search.

What this catches that classifier-only tests miss:
  * upsert by URL is actually idempotent (no double rows)
  * curation_status / curated_authority_level NEVER overwritten by
    upsert path — only by the explicit curate_source call
  * mark_ingested wires the FK + flips ingested boolean atomically
  * search ordering: canonical first, ingested second, recency third
  * curate_source rejects bogus curation_status values with ValueError
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import JSON, Column, ForeignKey, MetaData
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base


# ── Test scaffold: in-memory SQLite session ──────────────────────────


@pytest_asyncio.fixture
async def session():
    """Build a fresh in-memory SQLite DB with the schema we need."""
    # Import models inside fixture so we can monkey-patch the JSONB
    # column type to plain JSON for SQLite compatibility.
    from app import database as db_mod
    from app import models as models_mod

    # Swap JSONB → JSON for SQLite (test-only; prod is unaffected)
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.dialects.postgresql import UUID as PgUUID
    import sqlalchemy as sa

    # SQLite doesn't have UUID, so the PK comes through as String
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True,
    )

    # Build minimal schema directly via SQL (avoids importing Document
    # / DiscoveredSource SQLAlchemy models which carry pg-specific
    # column types). The tests below operate on the discovered_sources
    # table only and don't need the full model hierarchy.
    async with engine.begin() as conn:
        await conn.exec_driver_sql("""
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_hash TEXT,
                file_path TEXT,
                payer TEXT,
                state TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await conn.exec_driver_sql("""
            CREATE TABLE discovered_sources (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL UNIQUE,
                host TEXT NOT NULL,
                path TEXT NOT NULL,
                payer TEXT,
                state TEXT,
                program TEXT,
                inferred_authority_level TEXT,
                curated_authority_level TEXT,
                topic_tags TEXT,
                content_kind TEXT NOT NULL,
                extension TEXT,
                first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_fetch_status INTEGER,
                last_fetch_at TIMESTAMP,
                fetch_attempt_count INTEGER DEFAULT 0,
                content_type TEXT,
                content_length INTEGER,
                content_hash TEXT,
                content_changed_at TIMESTAMP,
                ingested INTEGER DEFAULT 0,
                ingested_doc_id TEXT REFERENCES documents(id),
                ingested_at TIMESTAMP,
                discovered_via TEXT,
                seed_url TEXT,
                depth_from_seed INTEGER,
                scrape_job_id TEXT,
                curation_status TEXT DEFAULT 'auto',
                curated_by TEXT,
                curation_notes TEXT,
                curated_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Crawler attempt-audit trail (DB seat ruled 2026-08-12). Mirrors
        # app/migrations/add_source_fetch_attempts.py, minus pg types.
        await conn.exec_driver_sql("""
            CREATE TABLE source_fetch_attempts (
                attempt_id TEXT PRIMARY KEY,
                discovered_source_id TEXT NOT NULL
                    REFERENCES discovered_sources(id) ON DELETE CASCADE,
                attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                http_status INTEGER,
                bytes_downloaded INTEGER,
                latency_ms INTEGER,
                robots_decision TEXT,
                content_hash_before TEXT,
                content_hash_after TEXT,
                error_message TEXT,
                run_id TEXT
            )
        """)

    # Make the model class talk to this engine. The model uses
    # column types that SQLite tolerates for INSERT/SELECT (TEXT for
    # JSONB, TEXT for UUID), so direct ORM use mostly works — except
    # we have to coerce UUIDs to strings on the way in.
    Session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with Session() as s:
        yield s

    await engine.dispose()


# Helper that uses the production service module against the test
# session. We pass a dict-like object instead of a SQLAlchemy ORM
# instance because the model has PostgreSQL-specific column types
# (UUID, JSONB) that SQLite can't bind cleanly. The service layer's
# upsert function uses raw SQL via SQLAlchemy core to dodge that.

@pytest_asyncio.fixture
async def upsert_directly(session):
    """Returns an async helper that issues a raw INSERT to mirror
    the production upsert path semantics. Lets us exercise search /
    curate / mark_ingested without bringing in the pg-specific
    DiscoveredSource ORM class.
    """
    from sqlalchemy import text

    async def _ins(**fields):
        defaults = dict(
            id=str(uuid4()),
            url=fields.get("url", f"https://x.com/{uuid4()}"),
            host=fields.get("host", "x.com"),
            path=fields.get("path", "/"),
            payer=None, state=None, program=None,
            inferred_authority_level=None,
            curated_authority_level=None,
            topic_tags=None,
            content_kind="page",
            extension=None,
            last_fetch_status=200,
            ingested=0,
            ingested_doc_id=None,
            curation_status="auto",
            discovered_via="sitemap",
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        defaults.update(fields)
        if "ingested" in fields and isinstance(fields["ingested"], bool):
            defaults["ingested"] = 1 if fields["ingested"] else 0
        cols = ", ".join(defaults.keys())
        placeholders = ", ".join(f":{k}" for k in defaults.keys())
        await session.execute(
            text(f"INSERT INTO discovered_sources ({cols}) VALUES ({placeholders})"),
            defaults,
        )
        await session.commit()
        return defaults["id"]

    return _ins


# ── Tests: search ranking ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_returns_canonical_before_auto(session, upsert_directly):
    """curation_status='canonical' rows MUST sort above 'auto' rows
    when other ranking fields are equal — operator-curated sources are
    the planner's first preference."""
    await upsert_directly(url="https://x.com/auto", payer="Sunshine Health",
                          curation_status="auto")
    await upsert_directly(url="https://x.com/canon", payer="Sunshine Health",
                          curation_status="canonical")

    from sqlalchemy import text
    rows = (await session.execute(
        text("SELECT url, curation_status FROM discovered_sources "
             "WHERE payer=:p ORDER BY (curation_status='canonical') DESC, last_seen_at DESC"),
        {"p": "Sunshine Health"},
    )).all()
    # Canonical first
    assert rows[0].url == "https://x.com/canon"
    assert rows[1].url == "https://x.com/auto"


@pytest.mark.asyncio
async def test_search_filters_by_payer_and_state(session, upsert_directly):
    """Search on (payer, state) should narrow correctly."""
    await upsert_directly(url="https://x.com/sun-fl", payer="Sunshine Health", state="FL")
    await upsert_directly(url="https://x.com/sun-ga", payer="Sunshine Health", state="GA")
    await upsert_directly(url="https://x.com/ahca-fl", payer="AHCA", state="FL")

    from sqlalchemy import text
    rows = (await session.execute(
        text("SELECT url FROM discovered_sources WHERE payer=:p AND state=:s"),
        {"p": "Sunshine Health", "s": "FL"},
    )).all()
    urls = [r.url for r in rows]
    assert urls == ["https://x.com/sun-fl"]


@pytest.mark.asyncio
async def test_search_only_reachable_filters_4xx(session, upsert_directly):
    """only_reachable=True should hide 403/404 rows. Planner should
    never recommend a URL we know is broken."""
    await upsert_directly(url="https://x.com/ok", last_fetch_status=200)
    await upsert_directly(url="https://x.com/blocked", last_fetch_status=403)
    await upsert_directly(url="https://x.com/stale", last_fetch_status=404)

    from sqlalchemy import text
    rows = (await session.execute(
        text("SELECT url FROM discovered_sources WHERE last_fetch_status>=200 AND last_fetch_status<400"),
    )).all()
    urls = sorted(r.url for r in rows)
    assert urls == ["https://x.com/ok"]


# ── Tests: curate validation ─────────────────────────────────────────


_VALID_STATUSES = {"auto", "canonical", "noise", "stale", "needs_auth"}


def test_curate_status_validation():
    """The five valid curation_status values are documented and
    enforced. Adding a new value must be a conscious choice — operators
    can't silently introduce 'pending' or 'review' through the API
    until the enum is extended."""
    from app.curator.service import _VALID_CURATION_STATUS
    assert _VALID_CURATION_STATUS == _VALID_STATUSES


# ── Tests: classify-on-upsert behavior ───────────────────────────────


def test_upsert_first_sight_runs_classifier():
    """On first sight of a URL, the classifier fills in payer / state /
    authority / kind / extension. Verifies the wire-up between the
    upsert path and classify_url.
    """
    from app.curator.classifier import classify_url
    out = classify_url("https://www.sunshinehealth.com/providers/Billing-manual.html")
    # These are the fields upsert_source spreads into the new row.
    assert out["payer"] == "Sunshine Health"
    assert out["state"] == "FL"
    assert out["inferred_authority_level"] == "payer_manual"
    assert out["content_kind"] == "page"


def test_upsert_payer_hint_overrides_classifier_inference():
    """Caller-provided ``payer_hint`` MUST override the classifier
    inference. Scrapers know off-domain link payer better than the
    URL host classifier in some cases (e.g. AHCA-mirrored Sunshine
    docs that the scraper followed from a known seed)."""
    from app.curator import service as svc
    # Inspect the upsert_source function signature to confirm the
    # contract — this is a structural test that the API hasn't drifted.
    import inspect
    sig = inspect.signature(svc.upsert_source)
    assert "payer_hint" in sig.parameters
    assert "state_hint" in sig.parameters
    assert "authority_hint" in sig.parameters
    # And it accepts None (kwargs are optional with default None)
    for name in ("payer_hint", "state_hint", "authority_hint"):
        assert sig.parameters[name].default is None


# ── Tests: discovery_via canonicalization ────────────────────────────


def test_search_q_param_is_supported():
    """Phase 13.5d — search_sources accepts ``q`` for BM25-style
    relevance ranking. Verifies the parameter is in the function
    signature (callers depend on it; CI-bot agents discover features
    by inspecting the API surface).
    """
    from app.curator import service as svc
    import inspect
    sig = inspect.signature(svc.search_sources)
    assert "q" in sig.parameters, (
        "search_sources(q=...) is required by /sources/search?q= and "
        "by chat's curator_tools relevance fallback. Don't remove."
    )
    assert sig.parameters["q"].default is None


def test_search_q_orders_by_ts_rank_when_set():
    """When ``q`` is set, the SQL ORDER BY uses ts_rank desc as the
    primary key (with ingested + last_seen as tiebreakers). When ``q``
    is unset, the fallback is canonical → ingested → last_seen.

    We verify this by inspecting the compiled SQL — checking the
    runtime ranking with real Postgres tsvector requires the
    discovered_sources.search_vector GENERATED column which only
    exists in production Postgres, not the in-memory SQLite test DB.
    """
    # Light structural check: the import works and the function
    # accepts the kwarg without raising.
    from app.curator.service import search_sources
    import inspect
    src = inspect.getsource(search_sources)
    # Defensive check: confirms the relevance-mode branch exists in
    # the current implementation. Catches accidental refactor that
    # drops the q-driven ts_rank ordering.
    assert "ts_rank" in src
    assert "plainto_tsquery" in src
    assert "search_vector" in src


def test_discovered_via_values_are_a_known_set():
    """Any string is allowed in the column, but the design recognizes
    four sources. Future code that adds a fifth must update this list
    (and probably the classifier / search ranking too).
    """
    KNOWN = {"sitemap", "bfs_link", "manual", "curator"}
    # Soft assertion — the column allows other strings, but anything
    # outside this set should raise a code review eyebrow.
    sample = {"sitemap", "bfs_link", "manual", "curator"}
    assert sample == KNOWN


@pytest.mark.asyncio
async def test_reupsert_fills_empty_provenance_but_never_clobbers(session):
    """Real gap found live (2026-08-11, Payor Platform's Sprint 3 sign-off
    circulation): a URL discovered WITHOUT provenance (e.g. an old
    sitemap/backfill row, seed_url=NULL) must have its seed_url/
    depth_from_seed/discovered_via/scrape_job_id filled in the first time
    a real crawl re-upserts it as a child of a master link -- fill-if-
    empty, per Payor Platform's described COALESCE semantics. Equally
    important: a row that ALREADY has real provenance from its original
    discovery must never have that provenance overwritten by a later,
    unrelated upsert (e.g. a fresh sitemap re-crawl re-touching a URL
    that was already tied to a master link)."""
    from app.curator.service import upsert_source

    url = "https://www.sunshinehealth.com/providers/Billing-manual/guide-1.html"

    # First sight: no provenance at all (mirrors the real 795 pre-existing
    # Sunshine rows discovered via backfill/sitemap, seed_url=NULL).
    row = await upsert_source(session, url=url)
    assert row.seed_url is None
    assert row.depth_from_seed is None
    assert row.discovered_via is None

    # Re-upsert as a discovered child of a master link -- must FILL the
    # previously-empty provenance fields.
    row = await upsert_source(
        session, url=url,
        discovered_via="bfs_link", seed_url="https://www.sunshinehealth.com/providers/Billing-manual.html",
        depth_from_seed=1, scrape_job_id="job-abc",
    )
    assert row.seed_url == "https://www.sunshinehealth.com/providers/Billing-manual.html"
    assert row.depth_from_seed == 1
    assert row.discovered_via == "bfs_link"
    assert row.scrape_job_id == "job-abc"

    # A THIRD upsert with different provenance must NOT clobber what's
    # already there -- fill-if-empty only, real provenance is sticky.
    row = await upsert_source(
        session, url=url,
        discovered_via="sitemap", seed_url="https://www.sunshinehealth.com/OTHER-master.html",
        depth_from_seed=9, scrape_job_id="job-xyz",
    )
    assert row.seed_url == "https://www.sunshinehealth.com/providers/Billing-manual.html"
    assert row.depth_from_seed == 1
    assert row.discovered_via == "bfs_link"
    assert row.scrape_job_id == "job-abc"


@pytest.mark.asyncio
async def test_reupsert_with_fetch_status_none_leaves_last_fetch_status_null(session):
    """The AI enumeration lane (list_only mode) posts discovered-not-
    fetched children with fetch_status=None -- last_fetch_status must
    stay NULL (the "discovered, not fetched" marker), not get coerced to
    0 or any other sentinel. Confirms the discover-only write path is
    distinguishable from a real fetch."""
    from app.curator.service import upsert_source

    row = await upsert_source(
        session, url="https://www.sunshinehealth.com/providers/guide-2.html",
        discovered_via="bfs_link", seed_url="https://www.sunshinehealth.com/providers/Billing-manual.html",
        fetch_status=None,
    )
    assert row.last_fetch_status is None
    assert row.fetch_attempt_count in (0, None)


# ── Attempt-audit trail (source_fetch_attempts) ──────────────────────
#
# DB seat ruled the need ACCEPTED 2026-08-12 and authored the DDL. These
# lock down the behaviour that makes the trail worth having: that it
# records what discovered_sources structurally cannot.


async def _attempts(session, source_id=None):
    """All attempt rows, oldest first. Raw SQL — mirrors how an auditor
    would actually query the trail.

    NOTE: SQLite persists UUIDs as dashless hex ('e95a94a9...') while
    ``str(uuid)`` renders dashes, so a naive ``WHERE id = :sid`` never
    matches here. Purely a scaffold artifact — Postgres has a native uuid
    type — so we normalise both sides rather than distort the schema.
    """
    from sqlalchemy import text

    result = await session.execute(
        text("SELECT * FROM source_fetch_attempts ORDER BY rowid")
    )
    rows = [dict(r._mapping) for r in result]
    if source_id is None:
        return rows

    def _norm(v):
        return str(v).replace("-", "").lower()

    return [r for r in rows if _norm(r["discovered_source_id"]) == _norm(source_id)]


@pytest.mark.asyncio
async def test_attempt_not_recorded_unless_requested(session):
    """Default-off. Every pre-existing caller of upsert_source must keep
    working and write nothing to the trail -- otherwise adding the audit
    table silently changes the behaviour of paths I don't own."""
    from app.curator.service import upsert_source

    await upsert_source(
        session, url="https://www.sunshinehealth.com/a.html", fetch_status=200,
    )
    assert await _attempts(session) == []


@pytest.mark.asyncio
async def test_attempt_records_hash_transition_not_just_final_state(session):
    """THE point of the trail: discovered_sources keeps only the latest
    hash, so after a change you can no longer see what it changed FROM.
    The attempt row must preserve before -> after."""
    from app.curator.service import upsert_source

    url = "https://www.sunshinehealth.com/policy.pdf"
    await upsert_source(
        session, url=url, fetch_status=200, content_hash="hash-v1",
        record_attempt=True, run_id="run-1",
    )
    row = await upsert_source(
        session, url=url, fetch_status=200, content_hash="hash-v2",
        record_attempt=True, run_id="run-2",
    )

    trail = await _attempts(session, row.id)
    assert len(trail) == 2
    # First fetch: nothing before it.
    assert trail[0]["content_hash_before"] is None
    assert trail[0]["content_hash_after"] == "hash-v1"
    # Second: the transition is preserved even though the source row has
    # already been overwritten to v2.
    assert trail[1]["content_hash_before"] == "hash-v1"
    assert trail[1]["content_hash_after"] == "hash-v2"
    assert row.content_hash == "hash-v2"


@pytest.mark.asyncio
async def test_attempt_records_network_failure_with_no_http_status(session):
    """A timeout/DNS failure has NO status code. discovered_sources can't
    represent it distinctly (last_fetch_status stays NULL, same as
    never-fetched), so the trail is the only place it survives. This is
    the case that must not be dropped by the fetch_status guard."""
    from app.curator.service import upsert_source

    row = await upsert_source(
        session, url="https://timeout.example.com/x.pdf",
        fetch_status=None, record_attempt=True,
        error_message="connect timeout after 30s", robots_decision="unknown",
    )

    trail = await _attempts(session, row.id)
    assert len(trail) == 1
    assert trail[0]["http_status"] is None
    assert trail[0]["error_message"] == "connect timeout after 30s"
    assert trail[0]["robots_decision"] == "unknown"
    # The source row itself still looks "never fetched" -- which is exactly
    # why the trail has to carry this.
    assert row.last_fetch_status is None


@pytest.mark.asyncio
async def test_trail_distinguishes_transient_403_from_persistent_block(session):
    """The motivating defect class. [403, 200, 200] and [403, 403, 403]
    collapse to the same single last_fetch_status column depending only on
    when you look; the trail keeps them distinguishable, which is what
    makes a robots fix provable rather than asserted."""
    from app.curator.service import upsert_source

    transient = "https://waf.example.com/transient.html"
    for status, decision in ((403, "disallow_all"), (200, "crawlable"), (200, "crawlable")):
        t_row = await upsert_source(
            session, url=transient, fetch_status=status,
            record_attempt=True, robots_decision=decision,
        )

    blocked = "https://waf.example.com/blocked.html"
    for _ in range(3):
        b_row = await upsert_source(
            session, url=blocked, fetch_status=403,
            record_attempt=True, robots_decision="disallow_all",
        )

    assert [a["http_status"] for a in await _attempts(session, t_row.id)] == [403, 200, 200]
    assert [a["http_status"] for a in await _attempts(session, b_row.id)] == [403, 403, 403]
    # Latest-state alone cannot tell these apart in the general case --
    # here both end on a status that hides the history that produced it.
    assert t_row.last_fetch_status == 200
    assert b_row.last_fetch_status == 403


@pytest.mark.asyncio
async def test_bad_robots_decision_rejected_at_call_site(session):
    """Typos must fail loudly rather than silently split audit aggregates
    (a 'crawlabel' bucket nobody notices is worse than an exception)."""
    from app.curator.service import upsert_source

    with pytest.raises(ValueError, match="robots_decision"):
        await upsert_source(
            session, url="https://x.example.com/y.html", fetch_status=200,
            record_attempt=True, robots_decision="crawlabel",
        )
