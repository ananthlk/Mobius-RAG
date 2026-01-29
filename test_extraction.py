#!/usr/bin/env python3
"""
Test script to verify PDF text extraction works.
Run this after installing pymupdf and setting up the database.
"""
import asyncio
import sys
from app.services.extract_text import extract_text_from_gcs

async def test_extraction(gcs_path: str):
    """Test text extraction from a GCS path."""
    print(f"Testing extraction from: {gcs_path}")
    print("-" * 60)
    
    try:
        pages = await extract_text_from_gcs(gcs_path)
        
        print(f"\n✓ Successfully extracted {len(pages)} pages\n")
        
        # Statistics
        successful = sum(1 for p in pages if p["extraction_status"] == "success")
        failed = sum(1 for p in pages if p["extraction_status"] == "failed")
        empty = sum(1 for p in pages if p["extraction_status"] == "empty")
        
        print(f"Summary:")
        print(f"  Total pages: {len(pages)}")
        print(f"  Successful: {successful}")
        print(f"  Empty: {empty}")
        print(f"  Failed: {failed}")
        
        # Show problematic pages
        problematic = [p for p in pages if p["extraction_status"] != "success"]
        if problematic:
            print(f"\n⚠️  Problematic pages ({len(problematic)}):")
            for page in problematic:
                print(f"\n  Page {page['page_number']}:")
                print(f"    Status: {page['extraction_status']}")
                print(f"    Text length: {page['text_length']}")
                if page.get('extraction_error'):
                    print(f"    Error: {page['extraction_error']}")
        else:
            print("\n✓ All pages extracted successfully!")
        
        # Show sample text from first successful page
        successful_pages = [p for p in pages if p["extraction_status"] == "success"]
        if successful_pages:
            first_page = successful_pages[0]
            sample_text = first_page["text"][:200] if first_page["text"] else ""
            print(f"\nSample text from page {first_page['page_number']} (first 200 chars):")
            print(f"  {sample_text}...")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_extraction.py <gcs_path>")
        print("Example: python test_extraction.py gs://mobius-rag-uploads-mobiusos/test.pdf")
        sys.exit(1)
    
    gcs_path = sys.argv[1]
    asyncio.run(test_extraction(gcs_path))
