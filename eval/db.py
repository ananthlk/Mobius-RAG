"""Shared asyncpg connection pool for the eval scripts.

Why a pool (2026-07-07): ``eval/run.py`` used to open a brand-new asyncpg
connection per persistence call — ~111 over one calibration run (1
insert_run + 110 insert_result). That churn against the local
cloud-sql-proxy (127.0.0.1:5433) is one of the triggers for the proxy's
degraded state where Postgres handshakes hang ~60s while TCP still
connects fine (restarting the proxy clears it), and it cost us partial
data loss in long 30–90 min runs. One lazily-created pool per process
(min 1 / max 5, acquire per statement) reuses a handful of connections
instead, and replaces the temporary 3-retry/2s-backoff patch that
briefly lived in ``eval/run.py::_conn()``.

Deliberately standalone rather than a dependency on mobius-db-agent (the
intended centralised DB-access layer): its PoolManager is sync
SQLAlchemy behind an MCP server, and these eval scripts must stay
runnable as plain CLI tools against a local proxy. If db-agent grows an
async pool API, this module is the single seam to swap.

Usage::

    from eval.db import execute, fetchrow, close_pool

    await execute("INSERT ...", ...)        # retries dead connections
    row = await fetchrow("SELECT ...")

    # CLI entrypoints only — long-lived processes (the eval router inside
    # the RAG API) just leave the pool up:
    await close_pool()

IDEMPOTENCY CONTRACT: ``execute``/``fetchrow`` retry on connection death,
which means a statement may run twice if the first attempt committed but
the ack was lost. Only pass statements that are safe to repeat: SELECTs,
keyed UPDATEs, and INSERTs with a client-generated id + ``ON CONFLICT
(id) DO NOTHING`` (see eval/run.py insert_run/insert_result). Raw
``(await get_pool()).execute(...)`` remains available for anything that
must not retry.
"""
from __future__ import annotations

import asyncio
import logging
import os

import asyncpg

logger = logging.getLogger("eval.db")

_POOL_MIN = 1
_POOL_MAX = 5
# Fail fast instead of inheriting asyncpg's 60s default: the degraded-proxy
# failure mode is precisely a handshake that hangs ~60s, and a quick failure
# + retry (or a loud error telling you to restart the proxy) beats stalling
# a calibration run one minute per statement.
_CONNECT_TIMEOUT_S = 30.0
_CREATE_RETRIES = 2
_CREATE_BACKOFF_S = 2.0
# Per-statement retry (added 2026-07-07, same night as the pool): the original
# assumption that pooled connections don't need statement-level retry broke in
# production — two calibration runs died (28/110 and 0/110 rows) with
# ``ConnectionDoesNotExistError: connection was closed in the middle of
# operation`` on connections freshly handed out by the pool. The degraded
# proxy kills LIVE connections, not just handshakes, so pool-creation retry
# alone can't save a run. Escalation ladder per attempt: expire the pool's
# idle connections (1st failure) → rebuild the whole pool (2nd+ failure).
_STMT_RETRIES = 3
_STMT_BACKOFF_S = (1.0, 2.0, 4.0)
# What "the connection under you died" looks like. PostgresConnectionError
# covers ConnectionDoesNotExistError / ConnectionFailureError; InterfaceError
# covers asyncpg's "connection is closed"; ConnectionError/TimeoutError cover
# the OS-level flavors surfaced through the proxy.
_RETRYABLE_ERRORS = (
    asyncpg.PostgresConnectionError,
    asyncpg.InterfaceError,
    ConnectionError,
    TimeoutError,
)

_pool: asyncpg.Pool | None = None
_pool_loop: asyncio.AbstractEventLoop | None = None


def _resolve_url() -> str:
    # DB URL selection must survive three environments (payor-agent diagnosis,
    # 2026-07-03). The raw ``DATABASE_URL`` env commonly points at the DIRECT
    # Cloud SQL IP (34.135.72.145:5432), FIREWALLED from local dev → asyncpg
    # connect times out (Errno 60) and background writers die silently. But we
    # can't just switch to ``app.config.DATABASE_URL`` either — in some
    # checkouts it ALSO resolves to the direct IP (env lacks
    # CHAT_RAG_DATABASE_URL). So: honour an explicit local/proxy override
    # first (the persistent driver sets 127.0.0.1:5433), else the app's
    # resolved URL (correct in-cloud where the direct IP is reachable). Local
    # dev needs the cloud-sql-proxy on 5433.
    from app.config import DATABASE_URL as _APP_DB_URL

    env = os.environ.get("DATABASE_URL", "")
    url = env if ("127.0.0.1" in env or "localhost" in env) else _APP_DB_URL
    return url.replace("postgresql+asyncpg://", "postgresql://").replace("+asyncpg", "")


async def get_pool() -> asyncpg.Pool:
    """Return the process-wide pool, creating it lazily on first use.

    Loop-aware: a pool is bound to the event loop it was created on, so if
    the caller is on a different loop (a second ``asyncio.run()`` in the
    same process, or a test loop), the stale pool is dropped and a fresh
    one is created for the current loop.
    """
    global _pool, _pool_loop
    loop = asyncio.get_running_loop()
    if _pool is not None and _pool_loop is loop:
        return _pool
    if _pool is not None:
        logger.info("event loop changed — discarding pool from previous loop")
        _pool = None

    last_exc: Exception | None = None
    for attempt in range(_CREATE_RETRIES + 1):
        try:
            _pool = await asyncpg.create_pool(
                _resolve_url(),
                min_size=_POOL_MIN,
                max_size=_POOL_MAX,
                # Recycle idle connections after 5 min — a long judge stall
                # shouldn't leave a stale proxy connection to trip on later.
                max_inactive_connection_lifetime=300.0,
                timeout=_CONNECT_TIMEOUT_S,
            )
            _pool_loop = loop
            return _pool
        except Exception as exc:
            last_exc = exc
            if attempt == _CREATE_RETRIES:
                break
            logger.warning(
                "pool creation failed (attempt %d/%d), retrying in %.0fs: %s",
                attempt + 1, _CREATE_RETRIES + 1, _CREATE_BACKOFF_S, exc,
            )
            await asyncio.sleep(_CREATE_BACKOFF_S)
    assert last_exc is not None
    raise last_exc


async def _retrying(method: str, query: str, *args):
    last_exc: Exception | None = None
    for attempt in range(_STMT_RETRIES + 1):
        pool = await get_pool()
        try:
            return await getattr(pool, method)(query, *args)
        except _RETRYABLE_ERRORS as exc:
            last_exc = exc
            if attempt == _STMT_RETRIES:
                break
            logger.warning(
                "%s failed on a dead connection (attempt %d/%d), retrying in %.0fs: %s",
                method, attempt + 1, _STMT_RETRIES + 1, _STMT_BACKOFF_S[attempt], exc,
            )
            try:
                # Flush idle connections that likely share the dead one's fate.
                pool.expire_connections()
            except Exception:
                pass
            if attempt >= 1:
                # Second strike: assume the pool itself is poisoned (proxy
                # bounce) — tear it down; the next get_pool() rebuilds it and
                # brings pool-creation retry into play.
                await close_pool()
            await asyncio.sleep(_STMT_BACKOFF_S[attempt])
    assert last_exc is not None
    raise last_exc


async def execute(query: str, *args):
    """``pool.execute`` with dead-connection retry. Statement must be
    idempotent — see the module docstring's contract."""
    return await _retrying("execute", query, *args)


async def fetchrow(query: str, *args):
    """``pool.fetchrow`` with dead-connection retry. Statement must be
    idempotent — see the module docstring's contract."""
    return await _retrying("fetchrow", query, *args)


_CLOSE_TIMEOUT_S = 10.0


async def close_pool() -> None:
    """Close the pool. Call from CLI entrypoints before the loop exits so
    asyncio.run() doesn't tear down a loop with live connections. The next
    get_pool() call recreates it, so this is always safe.

    Bounded-graceful: ``asyncpg.Pool.close()`` waits INDEFINITELY for every
    connection to be released and to close gracefully — against a degraded
    proxy (or with any acquire leaked by a cancelled task) it never returns,
    which left calibrate.py hanging at teardown as a zombie process
    (observed 2026-07-15). Give graceful close a bounded window, then
    ``terminate()`` (immediate socket close) so the process always exits."""
    global _pool, _pool_loop
    pool, _pool, _pool_loop = _pool, None, None
    if pool is None:
        return
    try:
        await asyncio.wait_for(pool.close(), timeout=_CLOSE_TIMEOUT_S)
    except (TimeoutError, asyncio.TimeoutError):
        logger.warning(
            "pool graceful close exceeded %.0fs (degraded proxy or leaked "
            "acquire) — terminating connections", _CLOSE_TIMEOUT_S,
        )
        pool.terminate()
