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
    pool_size=2,
    max_overflow=3,
)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
