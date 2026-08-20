"""Re-chunking must actually re-chunk.

Found 2026-08-19: after table capture rewrote the page text of 5 documents, a
full re-chunk created ZERO new rows and left 4,063 chunks from May in place,
still holding pre-excision text. `persist_chunk` is keyed on
(document, page, paragraph_index) and returned the existing row untouched, so a
document whose pages changed kept its old chunks forever.

Two separate defects, both needed fixing:
  1. text was never refreshed when it differed
  2. rows past the new end of a shortened page were never removed
"""
import sys, asyncio, types
import pytest
sys.path.insert(0, ".")


class _Chunk:
    def __init__(self, text, idx=0, page=1):
        self.id = f"c{page}_{idx}"
        self.text, self.text_length = text, len(text)
        self.page_number, self.paragraph_index = page, idx
        self.section_path = None
        self.start_offset_in_page = 0
        self.extraction_status = self.critique_status = "passed"
        self.summary = "stale summary of old text"


class _Res:
    def __init__(self, v): self._v = v
    def scalar_one_or_none(self): return self._v
    def scalars(self): return self
    def all(self): return self._v


class _DB:
    """Minimal AsyncSession stand-in; records deletes."""
    def __init__(self, existing=None, ids=None):
        self.existing, self.ids, self.added, self.deletes = existing, ids or [], [], 0
        self._first = True
    async def execute(self, stmt):
        txt = str(stmt).split("\n")[0].upper()
        if txt.startswith("DELETE"):
            self.deletes += 1
            return _Res(None)
        if self.ids and "ID" in txt:
            return _Res(self.ids)
        return _Res(self.existing)
    def add(self, o): self.added.append(o)
    async def flush(self): pass


def test_existing_chunk_text_is_refreshed_when_page_changed():
    from app.worker.db import persist_chunk
    old = _Chunk("-")                       # a May chunk: an orphaned table cell
    db = _DB(existing=old)
    got = asyncio.run(persist_chunk(db, "doc", 1, 0, "Prior authorization required."))
    assert got.text == "Prior authorization required.", "stale text must be replaced"
    assert got.text_length == len("Prior authorization required.")


def test_refresh_reopens_enrichment_because_the_content_changed():
    """A 'passed' verdict describes text that no longer exists."""
    from app.worker.db import persist_chunk
    old = _Chunk("-")
    db = _DB(existing=old)
    got = asyncio.run(persist_chunk(db, "doc", 1, 0, "New content.",
                                    extraction_status="pending", critique_status="pending"))
    assert got.extraction_status == "pending"
    assert got.critique_status == "pending"
    assert got.summary is None, "a summary of deleted text must not survive"


def test_unchanged_text_is_left_alone():
    from app.worker.db import persist_chunk
    same = _Chunk("Prior authorization required.")
    same.extraction_status = "passed"
    db = _DB(existing=same)
    got = asyncio.run(persist_chunk(db, "doc", 1, 0, "Prior authorization required."))
    assert got.extraction_status == "passed", "no needless re-enrichment"


def test_missing_chunk_is_created():
    from app.worker.db import persist_chunk
    db = _DB(existing=None)
    asyncio.run(persist_chunk(db, "doc", 1, 0, "text"))
    assert len(db.added) == 1


def test_prune_removes_rows_past_the_new_end():
    """Excision SHORTENS pages -- 40 paragraphs can become 3."""
    from app.worker.db import prune_page_chunks
    db = _DB(ids=["a", "b", "c"])
    n = asyncio.run(prune_page_chunks(db, "doc", 1, keep_count=3))
    assert n == 3
    assert db.deletes == 2, "facts cleared, then chunks"


def test_prune_is_a_noop_when_nothing_is_stale():
    from app.worker.db import prune_page_chunks
    db = _DB(ids=[])
    assert asyncio.run(prune_page_chunks(db, "doc", 1, keep_count=5)) == 0
    assert db.deletes == 0
