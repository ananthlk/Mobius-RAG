"""Real-DB integration tests for passenger_tables_loader.py, same convention
as test_integration_production_shapes.py: live DATABASE_URL, no mocks, marked
`integration` rather than skipped. The pure resolution logic (dedup, fail-
open, breadcrumb-vs-fallback priority) is already covered by
test_passenger_tables.py's 27 cases against fakes; this file's only job is to
prove the two real SQL queries against the live `document_tables` schema
actually return what the pure resolver expects.

Run with:
    .venv/bin/python -m pytest app/services/retriever/test_passenger_tables_loader.py -v
"""

from types import SimpleNamespace

import pytest

from app.database import AsyncSessionLocal
from app.services.retriever.passenger_tables_loader import (
    _fetch_tables_by_id,
    _fetch_tables_by_page,
    load_passenger_tables,
)

pytestmark = pytest.mark.integration


def _citation(chunk_id, text, document_id=None, page_number=None):
    return SimpleNamespace(chunk_id=chunk_id, text=text, document_id=document_id, page_number=page_number)


class TestFetchTablesById:
    @pytest.mark.asyncio
    async def test_resolves_a_real_breadcrumb_table_id(self):
        async with AsyncSessionLocal() as db:
            out = await _fetch_tables_by_id(db, ["bba1d1f4-b158-4c0f-9b90-88a6bd72bf46"])
        assert "bba1d1f4-b158-4c0f-9b90-88a6bd72bf46" in out
        row = out["bba1d1f4-b158-4c0f-9b90-88a6bd72bf46"]
        assert row["document_id"] == "762576d4-825f-4e3d-a867-17c0c656778b"
        assert row["page_number"] == 3

    @pytest.mark.asyncio
    async def test_unknown_id_absent_not_none_valued(self):
        async with AsyncSessionLocal() as db:
            out = await _fetch_tables_by_id(db, ["00000000-0000-0000-0000-000000000000"])
        assert out == {}

    @pytest.mark.asyncio
    async def test_empty_input_short_circuits(self):
        async with AsyncSessionLocal() as db:
            out = await _fetch_tables_by_id(db, [])
        assert out == {}


class TestFetchTablesByPage:
    @pytest.mark.asyncio
    async def test_page_with_two_tables_returns_both(self):
        async with AsyncSessionLocal() as db:
            out = await _fetch_tables_by_page(db, [("506e5412-82d5-4014-9e84-2fc1173a59a4", 36)])
        rows = out.get(("506e5412-82d5-4014-9e84-2fc1173a59a4", 36), [])
        assert {r["id"] for r in rows} == {
            "fd554cdb-3386-4490-b74b-6b5e5a1407d9",
            "7920d66f-eec7-4e04-b341-094f71d642c2",
        }

    @pytest.mark.asyncio
    async def test_page_with_no_table_absent(self):
        async with AsyncSessionLocal() as db:
            out = await _fetch_tables_by_page(db, [("506e5412-82d5-4014-9e84-2fc1173a59a4", 999999)])
        assert out.get(("506e5412-82d5-4014-9e84-2fc1173a59a4", 999999), []) == []


class TestLoadPassengerTablesEndToEnd:
    @pytest.mark.asyncio
    async def test_no_breadcrumb_numeric_citation_on_live_two_table_page_attaches_both(self):
        citations = [
            _citation(
                "chunk-1",
                "12  34",  # numeric-only, no breadcrumb -- forces the page-proximity path
                document_id="506e5412-82d5-4014-9e84-2fc1173a59a4",
                page_number=36,
            )
        ]
        async with AsyncSessionLocal() as db:
            result = await load_passenger_tables(db, citations)

        assert {t.table_id for t in result} == {
            "fd554cdb-3386-4490-b74b-6b5e5a1407d9",
            "7920d66f-eec7-4e04-b341-094f71d642c2",
        }
        assert all(t.matched_via == "page_proximity" for t in result)

    @pytest.mark.asyncio
    async def test_breadcrumb_citation_resolves_via_id_path_not_fallback(self):
        table_id = "bba1d1f4-b158-4c0f-9b90-88a6bd72bf46"
        citations = [
            _citation(
                "chunk-1",
                f"Rate detail. [Table: whatever caption · →document_tables:{table_id}]",
                document_id="762576d4-825f-4e3d-a867-17c0c656778b",
                page_number=3,
            )
        ]
        async with AsyncSessionLocal() as db:
            result = await load_passenger_tables(db, citations)

        assert len(result) == 1
        assert result[0].table_id == table_id
        assert result[0].matched_via == "breadcrumb"

    @pytest.mark.asyncio
    async def test_many_numeric_citations_one_table_page_dedup_to_one_passenger_table(self):
        # The exact hazard the go-ahead named: many cells off one page must
        # collapse to one PassengerTable against the REAL query, not just a fake.
        citations = [
            _citation(f"chunk-{i}", "059404", document_id="762576d4-825f-4e3d-a867-17c0c656778b", page_number=3)
            for i in range(15)
        ]
        async with AsyncSessionLocal() as db:
            result = await load_passenger_tables(db, citations)

        assert len(result) == 1
        assert len(result[0].cited_by_chunk_ids) == 15

    @pytest.mark.asyncio
    async def test_prose_citation_no_table_on_page_returns_empty_fail_open(self):
        citations = [
            _citation(
                "chunk-1",
                "Ordinary prose with no numbers and no breadcrumb.",
                document_id="762576d4-825f-4e3d-a867-17c0c656778b",
                page_number=999999,
            )
        ]
        async with AsyncSessionLocal() as db:
            result = await load_passenger_tables(db, citations)
        assert result == []
