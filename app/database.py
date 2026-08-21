"""Async engine and session factory for mobius-rag.

Components sharing one event loop share one connection pool: FastAPI
(get_db), chunking worker, embedding worker, and any code using
``AsyncSessionLocal``. One session = one connection from the pool;
sessions close after each request/job so connections return to the
pool.

2026-08-21: per-event-loop engines.

  asyncpg binds every connection to the event loop that created it.
  The chunking worker runs two lanes (batch + instant) in two threads,
  each with its own ``asyncio.run()`` loop, but both drew from this
  single module-level pool. Whichever lane checked out a connection
  first became the pool's de-facto loop owner; the other lane's first
  checkout awaited a future owned by a foreign loop and hung forever —
  after the TCP connect but before any statement, so the backend sat
  ``idle`` in ``ClientRead`` with an empty query and nothing crashed,
  nothing logged, and the supervisor never restarted it. Observed as
  1 of 4 chunking instances consuming the queue while 3 sat silent.

  ``AsyncSessionLocal()`` now resolves a sessionmaker for the *running*
  loop. Single-loop processes (the API, the embedding worker) keep the
  original module-level engine and are unaffected. Additional loops get
  their own smaller pool.

2026-04-21 hardening:

* Bumped pool to 5+10 so a running chunking worker can't starve the
  API of connections mid-ingest (prior pool_size=1, max_overflow=2 was
  causing TimeoutError on /documents during 100-page chunking runs).
* Wired server-side ``statement_timeout`` + ``idle_in_transaction_
  session_timeout`` via asyncpg ``server_settings``. Orphan backends
  from SIGKILLed workers now release their transaction locks within
  a bounded window instead of hanging the queue for hours.
* Any dialect quirks tolerated: the pg-only server_settings are only
  attached when the URL is an asyncpg URL.
"""
import asyncio
import logging
import threading
import weakref

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

logger = logging.getLogger(__name__)

from app.config import (
    DATABASE_URL,
    DB_IDLE_IN_TXN_TIMEOUT_MS,
    DB_MAX_OVERFLOW,
    DB_POOL_SIZE,
    DB_STATEMENT_TIMEOUT_MS,
)

# asyncpg-specific connect args: connection timeout + server-side
# per-session timeouts applied to every backend in the pool.
_connect_args: dict = {}
if "asyncpg" in DATABASE_URL:
    _connect_args = {
        "timeout": 15,
        # server_settings translates into SET <key> = <value> on each
        # new connection so the timeouts apply for the connection's
        # lifetime, not just a single statement.
        "server_settings": {
            "statement_timeout": str(DB_STATEMENT_TIMEOUT_MS),
            "idle_in_transaction_session_timeout": str(DB_IDLE_IN_TXN_TIMEOUT_MS),
            # Name the connection so pg_stat_activity shows which
            # service owns the backend — makes ops debugging the
            # "who's holding the lock?" question trivial.
            "application_name": "mobius-rag",
            # pgvector HNSW recall: default ef_search=40 is below our
            # wide-phase k=80, causing non-deterministic misses on sparse
            # table chunks (e.g. timely-filing deadline table near the
            # recall boundary). 100 gives a 25% buffer over k=80 with
            # ~10% latency impact vs the 5x cost of ef_search=200.
            # Set at connection init so it applies to every query on
            # this connection without touching transaction state.
            "hnsw.ef_search": "100",
        },
    }

def _make_engine(pool_size: int, max_overflow: int):
    return create_async_engine(
        DATABASE_URL,
        echo=False,
        connect_args=_connect_args,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
        pool_recycle=300,
    )


engine = _make_engine(DB_POOL_SIZE, DB_MAX_OVERFLOW)
_default_sessionmaker = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Secondary loops (the chunking worker's second lane) get a deliberately
# small pool: they are a single serial consumer, not a request fan-out,
# and every extra pool multiplies against Cloud Run instance count
# against a shared max_connections budget.
_SECONDARY_POOL_SIZE = 2
_SECONDARY_MAX_OVERFLOW = 3

_owner_loop_ref: "weakref.ref | None" = None
_loop_sessionmakers: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_loop_engines: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_loop_lock = threading.Lock()


def _sessionmaker_for_running_loop():
    """Return the sessionmaker whose pool belongs to the running loop.

    The first loop to ask claims the module-level ``engine``; any other
    loop gets its own engine so asyncpg never hands a connection created
    on loop A to code awaiting on loop B.
    """
    global _owner_loop_ref
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop (sync context / engine used directly) — the
        # caller is not about to await on a foreign loop.
        return _default_sessionmaker

    with _loop_lock:
        owner = _owner_loop_ref() if _owner_loop_ref is not None else None
        if owner is None:
            _owner_loop_ref = weakref.ref(loop)
            return _default_sessionmaker
        if owner is loop:
            return _default_sessionmaker

        maker = _loop_sessionmakers.get(loop)
        if maker is None:
            lane_engine = _make_engine(
                _SECONDARY_POOL_SIZE, _SECONDARY_MAX_OVERFLOW
            )
            maker = async_sessionmaker(
                lane_engine, class_=AsyncSession, expire_on_commit=False
            )
            _loop_engines[loop] = lane_engine
            _loop_sessionmakers[loop] = maker
            logger.warning(
                "[db] second event loop detected (%r) — created a dedicated "
                "pool (size=%d overflow=%d). Sharing one asyncpg pool across "
                "loops deadlocks; see module docstring.",
                loop, _SECONDARY_POOL_SIZE, _SECONDARY_MAX_OVERFLOW,
            )
        return maker


def AsyncSessionLocal() -> AsyncSession:
    """Session bound to the running loop's pool. Call, don't subclass."""
    return _sessionmaker_for_running_loop()()

Base = declarative_base()


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


# ── Org-docs DB (mobius_org_docs) ────────────────────────────────────
# Separate engine for the per-org namespace DB. None when ORG_DOCS_DATABASE_URL
# is unset — callers check before using (endpoint returns 503 if None).
from app.config import ORG_DOCS_DATABASE_URL as _ORG_DOCS_URL

OrgDocsSessionLocal: async_sessionmaker | None = None

if _ORG_DOCS_URL:
    _org_connect_args: dict = {}
    if "asyncpg" in _ORG_DOCS_URL:
        _org_connect_args = {
            "timeout": 15,
            "server_settings": {
                "statement_timeout": str(DB_STATEMENT_TIMEOUT_MS),
                "idle_in_transaction_session_timeout": str(DB_IDLE_IN_TXN_TIMEOUT_MS),
                "application_name": "mobius-rag-org-docs",
                "hnsw.ef_search": "100",
            },
        }
    _org_engine = create_async_engine(
        _ORG_DOCS_URL,
        echo=False,
        connect_args=_org_connect_args,
        pool_size=3,
        max_overflow=5,
        pool_pre_ping=True,
        pool_recycle=300,
    )
    OrgDocsSessionLocal = async_sessionmaker(_org_engine, class_=AsyncSession, expire_on_commit=False)
