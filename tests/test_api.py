"""API tests for Mobius RAG."""
import uuid

import pytest
from fastapi.testclient import TestClient


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"


def test_chunking_stop(client: TestClient):
    doc_id = str(uuid.uuid4())
    r = client.post(f"/documents/{doc_id}/chunking/stop")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"


def test_chunking_results_no_row(client: TestClient):
    """GET chunking/results with valid UUID but no persisted row returns 200 + empty metadata/results."""
    doc_id = str(uuid.uuid4())
    r = client.get(f"/documents/{doc_id}/chunking/results")
    assert r.status_code == 200
    data = r.json()
    assert data["document_id"] == doc_id
    assert data["metadata"] == {}
    assert data["results"] == {}


def test_chunking_results_invalid_uuid(client: TestClient):
    r = client.get("/documents/not-a-uuid/chunking/results")
    assert r.status_code == 400
