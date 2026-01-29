-- Migration: Add error tracking to the database
-- Run this SQL script directly in your PostgreSQL database

-- Create processing_errors table
CREATE TABLE IF NOT EXISTS processing_errors (
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
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_processing_errors_document_id 
    ON processing_errors(document_id);

CREATE INDEX IF NOT EXISTS idx_processing_errors_resolved 
    ON processing_errors(resolved);

CREATE INDEX IF NOT EXISTS idx_processing_errors_severity 
    ON processing_errors(severity);

-- Add error tracking columns to documents table
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'documents' AND column_name = 'has_errors'
    ) THEN
        ALTER TABLE documents ADD COLUMN has_errors VARCHAR(10) NOT NULL DEFAULT 'false';
    END IF;
END $$;

DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'documents' AND column_name = 'error_count'
    ) THEN
        ALTER TABLE documents ADD COLUMN error_count INTEGER NOT NULL DEFAULT 0;
    END IF;
END $$;

DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'documents' AND column_name = 'critical_error_count'
    ) THEN
        ALTER TABLE documents ADD COLUMN critical_error_count INTEGER NOT NULL DEFAULT 0;
    END IF;
END $$;

DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'documents' AND column_name = 'review_status'
    ) THEN
        ALTER TABLE documents ADD COLUMN review_status VARCHAR(20) NOT NULL DEFAULT 'pending';
    END IF;
END $$;
