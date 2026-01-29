#!/usr/bin/env python3
"""Create the mobius_rag database."""
import asyncio
import asyncpg
from app.config import DATABASE_URL

async def create_database():
    # Extract connection info from DATABASE_URL
    # Format: postgresql+asyncpg://user@host:port/dbname
    # We need to connect to 'postgres' database first to create our database
    
    # Parse the URL
    url_parts = DATABASE_URL.replace("postgresql+asyncpg://", "").split("/")
    if len(url_parts) < 2:
        print("Error: Invalid DATABASE_URL format")
        return
    
    auth_part = url_parts[0]  # user@host:port
    target_db = url_parts[1]  # mobius_rag
    
    # Parse auth part
    if "@" in auth_part:
        user, host_port = auth_part.split("@")
        if ":" in host_port:
            host, port = host_port.split(":")
        else:
            host = host_port
            port = "5432"
    else:
        user = auth_part
        host = "localhost"
        port = "5432"
    
    print(f"Connecting to Postgres as {user}@{host}:{port}...")
    
    try:
        # Connect to 'postgres' database to create our database
        conn = await asyncpg.connect(
            user=user,
            host=host,
            port=int(port),
            database='postgres'  # Connect to default database
        )
        
        # Check if database exists
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", target_db
        )
        
        if exists:
            print(f"Database '{target_db}' already exists!")
        else:
            # Create database
            await conn.execute(f'CREATE DATABASE "{target_db}"')
            print(f"Database '{target_db}' created successfully!")
        
        await conn.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(create_database())
