#!/usr/bin/env python3
"""Delete document via the running API server."""
import requests
import json
import sys

API_BASE = "http://localhost:8000"

def list_documents():
    """List all documents."""
    try:
        response = requests.get(f"{API_BASE}/documents")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error listing documents: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return None

def delete_document(document_id):
    """Delete a document by ID."""
    try:
        response = requests.delete(f"{API_BASE}/documents/{document_id}")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error deleting document: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error deleting document: {e}")
        return None

def main():
    pattern = "01-05-26-MFL-Medicaid-Provider-Handbook"
    if len(sys.argv) > 1:
        pattern = sys.argv[1]
    
    print(f"Listing documents matching: {pattern}")
    data = list_documents()
    
    if not data:
        print("Could not list documents. Is the server running?")
        return
    
    documents = data.get("documents", [])
    matching = [d for d in documents if pattern.lower() in d.get("filename", "").lower()]
    
    if not matching:
        print(f"No documents found matching: {pattern}")
        print(f"\nAll documents in database:")
        for doc in documents:
            print(f"  - {doc.get('id')}: {doc.get('filename')}")
        return
    
    print(f"\nFound {len(matching)} matching document(s):")
    for doc in matching:
        print(f"  - ID: {doc.get('id')}, Filename: {doc.get('filename')}")
    
    print(f"\nDeleting {len(matching)} document(s)...")
    for doc in matching:
        doc_id = doc.get("id")
        result = delete_document(doc_id)
        if result:
            print(f"✅ Deleted: {doc.get('filename')}")
        else:
            print(f"❌ Failed to delete: {doc.get('filename')}")

if __name__ == "__main__":
    main()
