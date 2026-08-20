"""Gate 3 (Mobius/docs/TABLE_CAPTURE_PROGRAM.md): one query, end to end --
retrieve a chunk, resolve its table, attach it, and the result carries a
table this query's prose alone does not. Real DB, real document_tables rows,
same convention as test_integration_production_shapes.py.

Run with:
    .venv/bin/python -m pytest app/services/retriever/test_synthesis_passenger_tables.py -v
"""

import pytest

from app.services.retriever import synthesis
from app.services.retriever.fillers.contracts import FilledChunk, FilledShape, FilledSlot
from app.database import AsyncSessionLocal

pytestmark = pytest.mark.integration


async def _fake_resolve_names(db, chunk_ids):
    return {}


async def _fake_neighbors(db, slot_items):
    return slot_items, 0, 0


class TestPassengerTablesWiredIntoCompileSynthesis:
    @pytest.mark.asyncio
    async def test_flag_off_by_default_no_passenger_tables_attached(self, monkeypatch):
        monkeypatch.setattr(synthesis, "_resolve_document_names", _fake_resolve_names)
        monkeypatch.setattr(synthesis, "_complete_neighbors", _fake_neighbors)

        chunk = FilledChunk(
            chunk_id="c1", document_id="762576d4-825f-4e3d-a867-17c0c656778b",
            text="059404", source_type="internal", page_number=3, original_score=0.9,
        )
        slot = FilledSlot(slot_id="s1", slot_semantics="direct_answer", capacity=1,
                           required=True, chunks=[chunk], occupancy=1)
        shape = FilledShape(slots=[slot], total_chunks_assigned=1)

        async with AsyncSessionLocal() as db:
            result = await synthesis.compile_synthesis("what is the rate", shape, db=db)

        assert result.passenger_tables == []  # PASSENGER_TABLE_RETRIEVAL is off in this env by default

    @pytest.mark.asyncio
    async def test_flag_on_numeric_chunk_on_live_table_page_attaches_real_table(self, monkeypatch):
        monkeypatch.setattr(synthesis, "_resolve_document_names", _fake_resolve_names)
        monkeypatch.setattr(synthesis, "_complete_neighbors", _fake_neighbors)
        monkeypatch.setattr(synthesis, "PASSENGER_TABLE_RETRIEVAL", True)

        # Retrieved chunk: numeric-only, no breadcrumb, on a real table-bearing
        # page (document_tables id bba1d1f4-..., doc 762576d4-..., page 3).
        chunk = FilledChunk(
            chunk_id="c1", document_id="762576d4-825f-4e3d-a867-17c0c656778b",
            text="059404", source_type="internal", page_number=3, original_score=0.9,
        )
        slot = FilledSlot(slot_id="s1", slot_semantics="direct_answer", capacity=1,
                           required=True, chunks=[chunk], occupancy=1)
        shape = FilledShape(slots=[slot], total_chunks_assigned=1)

        async with AsyncSessionLocal() as db:
            result = await synthesis.compile_synthesis("what is the rate", shape, db=db)

        # Gate 3's own wording: the attached table carries a value that exists
        # only inside the table, not in the retrieved chunk's own prose.
        assert len(result.passenger_tables) == 1
        table = result.passenger_tables[0]
        assert table.table_id == "bba1d1f4-b158-4c0f-9b90-88a6bd72bf46"
        assert table.matched_via == "page_proximity"
        assert table.payload.get("grid"), "attached table must carry real grid content, not an empty stub"
        assert chunk.text not in str(table.payload["grid"])  # the retrieved cell's own text != the table's full content

    @pytest.mark.asyncio
    async def test_flag_on_breadcrumb_chunk_resolves_and_dedups_against_fallback(self, monkeypatch):
        monkeypatch.setattr(synthesis, "_resolve_document_names", _fake_resolve_names)
        monkeypatch.setattr(synthesis, "_complete_neighbors", _fake_neighbors)
        monkeypatch.setattr(synthesis, "PASSENGER_TABLE_RETRIEVAL", True)

        table_id = "bba1d1f4-b158-4c0f-9b90-88a6bd72bf46"
        breadcrumb_chunk = FilledChunk(
            chunk_id="c1", document_id="762576d4-825f-4e3d-a867-17c0c656778b",
            text=f"Reimbursement detail. [Table: rates · →document_tables:{table_id}]",
            source_type="internal", page_number=3, original_score=0.9,
        )
        numeric_chunk = FilledChunk(
            chunk_id="c2", document_id="762576d4-825f-4e3d-a867-17c0c656778b",
            text="059404", source_type="internal", page_number=3, original_score=0.8,
        )
        slot = FilledSlot(slot_id="s1", slot_semantics="direct_answer", capacity=2,
                           required=True, chunks=[breadcrumb_chunk, numeric_chunk], occupancy=2)
        shape = FilledShape(slots=[slot], total_chunks_assigned=2)

        async with AsyncSessionLocal() as db:
            result = await synthesis.compile_synthesis("what is the rate", shape, db=db)

        # Same table reached two ways (breadcrumb on c1, page-proximity on
        # c2) must dedup to ONE PassengerTable, cited by both chunks.
        assert len(result.passenger_tables) == 1
        assert result.passenger_tables[0].table_id == table_id
        assert result.passenger_tables[0].matched_via == "breadcrumb"
        assert set(result.passenger_tables[0].cited_by_chunk_ids) == {"c1", "c2"}
