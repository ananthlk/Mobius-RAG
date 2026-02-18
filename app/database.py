"""
Single async engine and session factory for mobius-rag.

All components share the same connection pool: FastAPI (get_db), chunking worker,
embedding worker, and any code using AsyncSessionLocal. One session = one connection
from the pool; sessions are closed after each request/job so connections return to the pool.
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.config import DATABASE_URL

# Connection timeout (seconds) so Cloud Run doesn't hang waiting for DB
_connect_args = {"timeout": 15} if "asyncpg" in DATABASE_URL else {}
# Cap connections per instance to avoid exhausting shared DB (e.g. db-f1-micro ~25 conns)
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args=_connect_args,
    pool_size=1,
    max_overflow=2,
    pool_pre_ping=True,
    pool_recycle=300,
)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
