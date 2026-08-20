"""Persistence contract for captured tables — the two silent-failure traps.

These are unit tests over the row mapping and the guard conditions. The
behaviours that need the live schema (id honoured over the DB default, reingest
replacing rather than accumulating) were verified directly against migration 050
in a rolled-back transaction; asserted here against a fake connection so they
stay covered without a database.
"""
import json, sys, uuid, asyncio
import pytest
sys.path.insert(0, ".")
from app.services.table_persist import _params, _COLS, persist_page_tables, persist_document_tables


class _Res:
    def __init__(self, n): self.rowcount = n


class _Savepoint:
    """Stands in for db.begin_nested(). Real code needs a savepoint per insert
    because Postgres aborts the whole transaction on an integrity error."""
    def __init__(self, conn): self.conn = conn
    async def __aenter__(self): return self
    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.conn.rolled_back += 1
        return False        # propagate, so persist_page_tables counts the failure


class FakeConn:
    """Stands in for the AsyncSession; can be told to fail on a given row id."""
    def __init__(self, removed=0, fail_on=()):
        self.removed, self.fail_on, self.inserts, self.deletes = removed, set(fail_on), [], []
        self.rolled_back = 0

    def begin_nested(self):
        return _Savepoint(self)
    async def execute(self, sql, params=None):
        if "DELETE" in str(sql):
            self.deletes.append(params)
            return _Res(self.removed)
        if params and params.get("id") in self.fail_on:
            raise ValueError("bad row")
        self.inserts.append(params)
        return _Res(1)


def _t(tid=None, cap="Rates"):
    return {"id": tid or str(uuid.uuid4()), "page_number": 1,
            "grid": {"header": ["Code"], "rows": [["99213"]]},
            "anchor": {"page": 1}, "coverage": ["99213"], "caption": cap,
            "strategy": "lines", "n_rows": 1, "n_cols": 1, "is_clean": True,
            "bbox": (1, 2, 3, 4), "breadcrumb": "[Table: Rates]"}


def test_row_maps_exactly_the_applied_columns():
    r = _params(_t(), "doc-1")
    assert set(r) == set(_COLS)
    assert r["document_id"] == "doc-1"
    # bbox and breadcrumb must NOT be persisted: bbox lives inside anchor, the
    # breadcrumb lives in the page text.
    assert "bbox" not in _COLS and "breadcrumb" not in _COLS


def test_jsonb_columns_are_serialised_and_coverage_is_a_list():
    r = _params(_t(), "doc-1")
    assert isinstance(r["grid"], str)                       # jsonb
    assert json.loads(r["grid"])["header"] == ["Code"]
    assert r["coverage"] == ["99213"]                       # text[], not jsonb


def test_trap1_client_minted_id_is_always_supplied():
    """If the id were omitted the DB default would fire and every breadcrumb
    would point at a row that does not exist -- silently, since both capture and
    Retriever fail open."""
    tid = str(uuid.uuid4())
    r = _params(_t(tid), "doc-1")
    assert r["id"] == tid


def test_trap2_delete_precedes_insert_so_reingest_replaces():
    c = FakeConn(removed=3)
    s = asyncio.run(persist_page_tables(c, "doc-1", 1, [_t(), _t()]))
    assert len(c.deletes) == 1, "must clear the page before writing"
    assert s == {"written": 2, "removed": 3, "failed": 0}


def test_empty_list_still_clears_the_page():
    """A page that genuinely lost its tables must not keep the old rows."""
    c = FakeConn(removed=2)
    s = asyncio.run(persist_page_tables(c, "doc-1", 1, []))
    assert len(c.deletes) == 1 and s["removed"] == 2 and s["written"] == 0


def test_one_bad_table_does_not_cost_the_page_its_others():
    bad = _t(cap="bad")
    c = FakeConn(fail_on=[bad["id"]])
    s = asyncio.run(persist_page_tables(c, "doc-1", 1, [_t(), bad]))
    assert s["written"] == 1 and s["failed"] == 1     # logged, never raised


def test_page_without_tables_key_is_left_alone():
    """capture was OFF for that run -- clearing would delete real captured data
    merely because the flag was off."""
    c = FakeConn(removed=5)
    s = asyncio.run(persist_document_tables(c, "doc-1", [{"page_number": 1}]))
    assert c.deletes == [] and s["pages"] == 0


def test_page_with_empty_tables_list_is_processed():
    c = FakeConn()
    s = asyncio.run(persist_document_tables(c, "doc-1", [{"page_number": 1, "tables": []}]))
    assert len(c.deletes) == 1 and s["pages"] == 1


def test_a_failed_insert_rolls_back_only_its_savepoint():
    """Postgres aborts the entire transaction on an integrity error, so without a
    savepoint per insert a single bad table loses the good ones AND poisons the
    caller's session -- verified against the live schema before this was added."""
    bad = _t(cap="bad")
    c = FakeConn(fail_on=[bad["id"]])
    s = asyncio.run(persist_page_tables(c, "doc-1", 1, [_t(), bad, _t()]))
    assert s["written"] == 2, "tables after the failure must still be written"
    assert s["failed"] == 1
    assert c.rolled_back == 1, "exactly one savepoint rolled back, not the transaction"
