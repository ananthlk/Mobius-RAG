"""Single async engine and session factory for mobius-rag.

All components share the same connection pool: FastAPI (get_db),
chunking worker, embedding worker, and any code using
``AsyncSessionLocal``. One session = one connection from the pool;
sessions close after each request/job so connections return to the
pool.

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
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

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
        },
    }

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args=_connect_args,
    pool_size=DB_POOL_SIZE,
    max_overflow=DB_MAX_OVERFLOW,
    pool_pre_ping=True,
    pool_recycle=300,
)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
