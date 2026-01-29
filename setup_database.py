#!/usr/bin/env python3
"""Create database and initialize tables."""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from app.database import Base
from app.config import DATABASE_URL


async def setup_database():
    # Parse DATABASE_URL to get connection info for 'postgres' database
    # Format: postgresql+asyncpg://user@host:port/dbname
    url_parts = DATABASE_URL.replace("postgresql+asyncpg://", "").split("/")
    if len(url_parts) < 2:
        print("Error: Invalid DATABASE_URL format")
        return
    
    auth_part = url_parts[0]
    target_db = url_parts[1]
    
    # Create connection string for 'postgres' database
    postgres_url = DATABASE_URL.replace(f"/{target_db}", "/postgres")
    
    print(f"Connecting to Postgres to create database '{target_db}'...")
    
    try:
        # Connect to 'postgres' database
        engine = create_async_engine(postgres_url, echo=False)
        
        async with engine.begin() as conn:
            # Check if database exists
            result = await conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
                {"dbname": target_db}
            )
            exists = result.scalar_one_or_none()
            
            if exists:
                print(f"Database '{target_db}' already exists.")
            else:
                # Create database (note: CREATE DATABASE cannot be run in a transaction)
                await conn.commit()
                await conn.execute(text(f'CREATE DATABASE "{target_db}"'))
                print(f"Database '{target_db}' created successfully!")
        
        await engine.dispose()
        
        # Now initialize tables in the new database
        print(f"\nInitializing tables in '{target_db}'...")
        engine = create_async_engine(DATABASE_URL, echo=True)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()
        print("Database tables created successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(setup_database())
