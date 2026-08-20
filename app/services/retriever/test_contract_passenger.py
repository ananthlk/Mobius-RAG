"""Contract must carry page_number through, and must carry passenger tables out.

Both defects were found from one live trace (2026-08-19):
  * five of six fillers never threaded PoolCandidate.page_number onto
    FilledChunk, so every chunk reached the envelope with page_number=None --
    and page-proximity table attachment joins on (document_id, page_number),
    so it silently resolved nothing while failing open.
  * passenger tables resolved onto SynthesisResult and were dropped by
    build_contract(), which was frozen at 12 fields.
"""
import sys
sys.path.insert(0, ".")
from dataclasses import dataclass, field

from app.services.retriever.contract import ContractEnvelope, _passenger_tables


@dataclass
class _PT:
    table_id: str
    caption: str | None
    payload: dict
    cited_by_chunk_ids: tuple
    matched_via: str


@dataclass
class _Synth:
    passenger_tables: list = field(default_factory=list)


def test_envelope_is_appended_not_reordered():
    """Byte-compat P0: the 12 original keys keep their exact positions."""
    keys = list(ContractEnvelope(query="q", chosen_slot=None, score=None).to_dict())
    assert keys[:12] == [
        "query", "chosen_slot", "score", "chunks", "answer_text", "thinking",
        "traces", "routing_keys", "grounding_markers", "latency_ms",
        "attempt_count", "status",
    ]
    assert keys[12] == "passenger_tables" and len(keys) == 13


def test_defaults_to_empty_list_not_none():
    """A consumer that predates the field sees [], never a null or a KeyError."""
    e = ContractEnvelope(query="q", chosen_slot=None, score=None)
    assert e.to_dict()["passenger_tables"] == []


def test_tables_are_projected_as_plain_json():
    """Chat must not need to import a retriever-internal type to read this."""
    s = _Synth([_PT("t1", "Rates", {"grid": {"header": ["Code"]}}, ("c1", "c2"), "page_proximity")])
    out = _passenger_tables(s)
    assert len(out) == 1
    t = out[0]
    assert isinstance(t, dict)
    assert t["table_id"] == "t1"
    assert t["caption"] == "Rates"
    assert t["cited_by_chunk_ids"] == ["c1", "c2"]      # tuple -> list, JSON-safe
    assert t["matched_via"] == "page_proximity"


def test_matched_via_survives_so_the_two_paths_stay_distinguishable():
    s = _Synth([
        _PT("a", None, {}, ("c1",), "breadcrumb"),
        _PT("b", None, {}, ("c2",), "page_proximity"),
    ])
    assert [t["matched_via"] for t in _passenger_tables(s)] == ["breadcrumb", "page_proximity"]


def test_no_synthesis_degrades_to_empty():
    assert _passenger_tables(None) == []


def test_malformed_table_does_not_cost_the_caller_the_answer():
    class Boom:
        @property
        def table_id(self): raise RuntimeError("bad table")
    s = _Synth([Boom(), _PT("ok", None, {}, (), "breadcrumb")])
    out = _passenger_tables(s)
    assert [t["table_id"] for t in out] == ["ok"]      # additive, never load-bearing
