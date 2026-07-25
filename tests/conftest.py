"""Pytest fixtures for Mobius RAG tests."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

import app.database as _dbmod

# Swap the shared module-level engine for a NullPool one for the whole test
# session. `app.database.engine`'s AsyncAdaptedQueuePool is a process-wide
# singleton, but each `TestClient(app)` instance runs requests through its
# own anyio portal thread with its own event loop, and pytest-asyncio tests
# run on yet another (session-scoped) loop. A pooled asyncpg connection
# opened on one of those loops and later checked out on a different one
# makes SQLAlchemy's pool_pre_ping (or teardown) raise "Future ... attached
# to a different loop" (observed full-suite-only in test_api.py). NullPool
# opens a fresh physical connection per checkout and never reuses one
# across loops, so the mismatch can't happen. Done here (not in
# app/database.py) since it's purely a multi-loop test artifact, not a
# production concern. Must run before any test module does
# `from app.database import AsyncSessionLocal`, since that binds a local
# name to whatever object exists at import time.
_dbmod.engine = create_async_engine(
    _dbmod.DATABASE_URL,
    echo=False,
    connect_args=_dbmod._connect_args,
    poolclass=NullPool,
)
_dbmod.AsyncSessionLocal = async_sessionmaker(
    _dbmod.engine, class_=AsyncSession, expire_on_commit=False
)

from app.main import app  # noqa: E402  (must follow the engine patch above)


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client."""
    return TestClient(app)
