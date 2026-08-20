"""Unit tests for app/services/table_capture — no DB, no corpus, no network.

Fitz is mocked so the pure function is testable in isolation (Stage-1 gate). Verifies:
detection→grid, structure-based header split, consistency gate, excision→breadcrumb,
and the non-negotiable fail-open contract.
"""
import uuid

from app.services.table_capture import capture_page_tables, _split_header, _is_table


# --- fitz page test doubles ------------------------------------------------
class _FakeTable:
    def __init__(self, rows, bbox):
        self._rows, self.bbox = rows, bbox
    def extract(self):
        return self._rows


class _FakeFinder:
    def __init__(self, tables):
        self.tables = tables


class _FakePage:
    def __init__(self, tables, blocks):
        self._tables, self._blocks = tables, blocks
    def find_tables(self, **kwargs):
        return _FakeFinder(self._tables)
    def get_text(self, kind="text", **kwargs):
        return self._blocks if kind == "blocks" else "page text"


# --- pure helpers ----------------------------------------------------------
def test_split_header_merges_super_header():
    grid = [["Facility Info", "", "Method"],   # sparse merged super-header
            ["ID", "Name", "Rate"],            # real column labels
            ["001", "Abbey", "$10"],
            ["002", "Bay", "$20"]]
    hdr, data, ncols = _split_header(grid)
    assert ncols == 3
    assert data[0][0] == "001"          # data starts at the first non-header row
    assert "ID" in hdr[0]               # column label captured, not the super-header alone


def test_is_table_rewards_consistent_body():
    assert _is_table([["a", "b"], ["c", "d"], ["e", "f"]], 2)
    assert not _is_table([["only one row"]], 2)


# --- capture + excise ------------------------------------------------------
def test_capture_excises_cells_and_leaves_breadcrumb():
    rows = [["ID", "Name", "Rate"], ["001", "Abbey", "$10"], ["002", "Bay", "$20"]]
    table = _FakeTable(rows, bbox=(10, 50, 200, 120))
    blocks = [
        (10, 10, 200, 30, "Fee Schedule 2024", 0, 0),                        # caption above
        (10, 55, 200, 115, "ID Name Rate 001 Abbey $10 002 Bay $20", 1, 0),  # the table region
        (10, 130, 200, 150, "See notes below.", 2, 0),                       # prose after
    ]
    clean, tables = capture_page_tables(_FakePage([table], blocks), "raw", page_number=7)

    assert len(tables) == 1
    t = tables[0]
    # shape matches the live document_tables schema
    assert t["page_number"] == 7 and t["table_index"] == 0 and t["n_cols"] == 3
    assert t["strategy"] in ("lines", "text")
    assert t["grid"]["rows"][0][0] == "001" and t["grid"]["header"]        # grid jsonb carries header+rows
    assert t["caption"] == "Fee Schedule 2024"
    assert t["anchor"] == {"section": "Fee Schedule 2024", "page": 7, "bbox": t["bbox"]}
    uuid.UUID(t["id"])                                                     # a real uuid, links the breadcrumb
    # excision: the table cells are gone from the page text, replaced by the uuid breadcrumb
    assert f"→document_tables:{t['id']}" in clean and t["breadcrumb"] in clean
    assert "$10" not in clean and "Abbey" not in clean
    assert "See notes below." in clean                                    # non-table prose preserved


def test_no_tables_passes_text_through():
    clean, tables = capture_page_tables(_FakePage([], []), "original text", 1)
    assert clean == "original text" and tables == []


def test_fail_open_on_raise():
    """A raise anywhere inside MUST degrade to (raw_text unmodified, [])."""
    class _Boom:
        def find_tables(self, **kwargs):
            raise RuntimeError("detector exploded")
        def get_text(self, *a, **k):
            return "x"
    clean, tables = capture_page_tables(_Boom(), "ORIGINAL", 1)
    assert clean == "ORIGINAL" and tables == []
