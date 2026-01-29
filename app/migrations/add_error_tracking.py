"""
Migration script to add error tracking to the database.

Run this script to:
1. Create the processing_errors table
2. Add error tracking columns to the documents table

Usage:
    python -m app.migrations.add_error_tracking
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from app.config import DATABASE_URL


async def run_migration():
    """Run the migration to add error tracking."""
    # Parse DATABASE_URL
    # Format: postgresql+asyncpg://user:password@host:port/database
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    
    # Connect using asyncpg directly
    conn = await asyncpg.connect(url)
    
    try:
        # Create processing_errors table
        print("Creating processing_errors table...")
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = 'processing_errors'
            )
        """)
        
        if not table_exists:
            await conn.execute("""
                CREATE TABLE processing_errors (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    paragraph_id VARCHAR(100),
                    error_type VARCHAR(50) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    error_message TEXT NOT NULL,
                    error_details JSONB,
                    stage VARCHAR(50) NOT NULL,
                    resolved VARCHAR(10) NOT NULL DEFAULT 'false',
                    resolution VARCHAR(20),
                    resolved_by VARCHAR(255),
                    resolved_at TIMESTAMP,
                    resolution_notes TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            print("✓ Created processing_errors table")
        else:
            print("✓ processing_errors table already exists")
        
        # Create indexes
        print("Creating indexes...")
        indexes = [
            ("idx_processing_errors_document_id", "processing_errors(document_id)"),
            ("idx_processing_errors_resolved", "processing_errors(resolved)"),
            ("idx_processing_errors_severity", "processing_errors(severity)")
        ]
        
        for idx_name, idx_def in indexes:
            idx_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes 
                    WHERE indexname = $1
                )
            """, idx_name)
            
            if not idx_exists:
                await conn.execute(f"CREATE INDEX {idx_name} ON {idx_def}")
                print(f"✓ Created index {idx_name}")
            else:
                print(f"✓ Index {idx_name} already exists")
        
        # Add error tracking columns to documents table
        print("Adding error tracking columns to documents table...")
        
        columns_to_add = [
            ("has_errors", "VARCHAR(10) NOT NULL DEFAULT 'false'"),
            ("error_count", "INTEGER NOT NULL DEFAULT 0"),
            ("critical_error_count", "INTEGER NOT NULL DEFAULT 0"),
            ("review_status", "VARCHAR(20) NOT NULL DEFAULT 'pending'")
        ]
        
        for col_name, col_def in columns_to_add:
            col_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'documents' AND column_name = $1
                )
            """, col_name)
            
            if not col_exists:
                await conn.execute(f"ALTER TABLE documents ADD COLUMN {col_name} {col_def}")
                print(f"✓ Added column {col_name} to documents table")
            else:
                print(f"✓ Column {col_name} already exists in documents table")
        
        print("Migration completed successfully!")
        
    except Exception as e:
        print(f"Migration failed: {e}")
        raise
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(run_migration())
