"""Every Pool-serving filler must thread page_number onto FilledChunk.

This is a source-level guard, not a behavioural one, and deliberately so: the
bug it prevents is an OMISSION. Five of six fillers simply never passed
PoolCandidate.page_number along, the field defaulted to None, and page-proximity
passenger-table attachment -- which joins on (document_id, page_number) --
silently resolved nothing. Because that path fails open, no test, log or metric
anywhere reported the miss; it took reading a live trace to see it.

A behavioural test per filler would need each one's full candidate/slot
scaffolding and would still only cover the fillers someone remembered to write
one for. Reading the construction site catches the next filler too.
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

FILLERS_DIR = Path("app/services/retriever/fillers")

# Fillers that serve chunks sourced from Pool (internal corpus), where the page
# is known and page-proximity depends on it. filler_s is excluded: it serves
# certified facts from the Payor fact store, which are not page-scoped.
POOL_FILLERS = ["filler_a", "filler_b", "filler_baseline", "filler_c", "filler_d"]


def _filled_chunk_blocks(path: Path) -> list[str]:
    """Return the source of each FilledChunk(...) call, brace-matched."""
    src = path.read_text()
    blocks = []
    for m in re.finditer(r"FilledChunk\(", src):
        depth, start = 0, m.start()
        for i in range(start, len(src)):
            if src[i] == "(":
                depth += 1
            elif src[i] == ")":
                depth -= 1
                if depth == 0:
                    blocks.append(src[start:i])
                    break
    return blocks


@pytest.mark.parametrize("name", POOL_FILLERS)
def test_pool_filler_threads_page_number(name):
    blocks = _filled_chunk_blocks(FILLERS_DIR / f"{name}.py")
    assert blocks, f"{name}: no FilledChunk(...) call found -- test is stale"
    for i, b in enumerate(blocks):
        assert "page_number=" in b, (
            f"{name}: FilledChunk call #{i + 1} does not set page_number. "
            "Page-proximity table attachment joins on (document_id, page_number); "
            "a null page resolves no table and fails open, so nothing will report it."
        )
