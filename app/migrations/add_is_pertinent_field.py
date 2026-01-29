#!/usr/bin/env python3
"""Migration: Add is_pertinent_to_claims_or_members field to extracted_facts table."""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from app.config import DATABASE_URL


async def migrate():
    """Add is_pertinent_to_claims_or_members column to extracted_facts table."""
    # Parse DATABASE_URL
    # Format: postgresql+asyncpg://user:password@host:port/database
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    
    # Connect using asyncpg directly
    conn = await asyncpg.connect(url)
    
    try:
        # Check if column already exists
        column_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.columns 
                WHERE table_name = 'extracted_facts' 
                AND column_name = 'is_pertinent_to_claims_or_members'
            )
        """)
        
        if column_exists:
            print("✓ Column 'is_pertinent_to_claims_or_members' already exists in 'extracted_facts' table")
        else:
            # Add the column
            await conn.execute("""
                ALTER TABLE extracted_facts 
                ADD COLUMN is_pertinent_to_claims_or_members VARCHAR(10) NULL
            """)
            print("✓ Added column 'is_pertinent_to_claims_or_members' to 'extracted_facts' table")
        
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
