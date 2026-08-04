"""Tests for literal procedure-code expansion (app/services/retriever/shape/code_expand.py).

Two layers: pure unit tests on `expand_query()` with a manually-populated
in-memory dict (no DB), and DB-integration tests confirming `ensure_loaded()`
+ `run_gate()` end-to-end resolves real HCPCS-code queries that previously
declined (OUT_OF_SCOPE/UNCLEAR/dead-end UNDERSPECIFIED).
"""

from __future__ import annotations

import pytest

from app.database import AsyncSessionLocal
from app.services.retriever.shape import code_expand
from app.services.retriever.shape.contracts import Contour
from app.services.retriever.shape.gate import run_gate


class TestExpandQueryPureUnit:
    """Directly manipulate the module-level dicts — no DB, deterministic."""

    def setup_method(self):
        code_expand._CODE_TO_DESC.clear()
        code_expand._DESC_TO_CODE.clear()
        code_expand._CODE_TO_DESC["H0019"] = "Behavioral health residential treatment"
        code_expand._DESC_TO_CODE["behavioral health residential treatment"] = "H0019"
        code_expand._LOADED = True

    def teardown_method(self):
        code_expand._CODE_TO_DESC.clear()
        code_expand._DESC_TO_CODE.clear()
        code_expand._LOADED = False

    def test_code_in_query_appends_description(self):
        out = code_expand.expand_query("What do you know about H0019?")
        assert "What do you know about H0019?" in out  # original text preserved
        assert "Behavioral health residential treatment" in out  # decoded text appended

    def test_description_in_query_appends_code(self):
        out = code_expand.expand_query("Tell me about behavioral health residential treatment")
        assert "H0019" in out

    def test_no_match_returns_query_unchanged(self):
        q = "What is the timely filing deadline?"
        assert code_expand.expand_query(q) == q

    def test_empty_query_returns_unchanged(self):
        assert code_expand.expand_query("") == ""

    def test_word_boundary_no_false_positive_on_partial_token(self):
        # "H00199" is NOT the code "H0019" — must not match on a substring.
        out = code_expand.expand_query("Reference number H00199 in the file")
        assert "Behavioral health residential treatment" not in out

    def test_no_op_when_nothing_loaded(self):
        code_expand._CODE_TO_DESC.clear()
        code_expand._DESC_TO_CODE.clear()
        q = "What about H0019?"
        assert code_expand.expand_query(q) == q  # safe no-op, never raises


@pytest.mark.asyncio
class TestCodeExpandIntegration:
    """DB-backed: confirms the real fix for the gap Shape-Reformat found —
    bare/embedded HCPCS codes previously carried zero lexicon signal."""

    async def test_h0019_bare_code_no_longer_declines(self):
        async with AsyncSessionLocal() as db:
            r = await run_gate(db, "H0019")
            # Previously UNCLEAR (fails _is_malformed's word-count check on a
            # single non-word token). Now resolves real D-codes from the
            # decoded description, landing on a real (non-decline) contour.
            assert r.contour not in (Contour.UNCLEAR, Contour.OUT_OF_SCOPE)
            assert r.d_codes  # real domain signal now, not empty

    async def test_h0019_in_sentence_resolves_exact(self):
        async with AsyncSessionLocal() as db:
            r = await run_gate(db, "Does Sunshine Health cover H0019 for Medicaid patients?")
            assert r.contour == Contour.EXACT
            assert r.d_codes  # was [] before this fix

    async def test_original_query_text_preserved_for_display(self):
        # GateResult.query/.normalized must NOT contain decoded jargon —
        # only the lexicon-matching input is expanded, per Retriever's ruling.
        async with AsyncSessionLocal() as db:
            q = "What do you know about H0019?"
            r = await run_gate(db, q)
            assert r.query == q
            assert "Behavioral" not in r.normalized and "residential" not in r.normalized
