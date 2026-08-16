"""Tests for POST /eval/grade-claim — the verify_claim judge (§16 handshake).

check_facts (the LLM judge) is monkeypatched so these assert the ENDPOINT's
verdict-mapping + fail-closed + locked-ruler behavior, not the model.
"""
import pytest
from fastapi.testclient import TestClient

import app.services.fact_checker as fc_mod
from app.services.fact_checker import FactCheckResult, FactVerdict


def _install_fake(monkeypatch, *, result=None, raises=None, capture=None):
    """Replace check_facts with a fake that records kwargs and returns `result`."""
    async def _fake(**kwargs):
        if capture is not None:
            capture.update(kwargs)
        if raises is not None:
            raise raises
        return result
    monkeypatch.setattr(fc_mod, "check_facts", _fake)


def test_agree_maps_from_supported_uncontradicted(client: TestClient, monkeypatch):
    res = FactCheckResult(
        verdicts=[FactVerdict(fact="c", support=1.0, grounded=True,
                              contradicted=False, evidence="within sixty (60) calendar days")],
        coverage=1.0)
    _install_fake(monkeypatch, result=res)
    r = client.post("/api/eval/grade-claim",
                    json={"claim": "appeal deadline is 60 days", "source_text": "…sixty (60) calendar days…", "page": 80})
    assert r.status_code == 200
    d = r.json()
    assert d["verdict"] == "agree"
    assert d["quote"] == "within sixty (60) calendar days"
    assert d["page"] == 80
    assert d["status"] == "ok"


def test_contradict_maps_from_contradicted(client: TestClient, monkeypatch):
    res = FactCheckResult(
        verdicts=[FactVerdict(fact="c", support=0.0, contradicted=True,
                              evidence="within thirty (30) calendar days")],
        coverage=0.0)
    _install_fake(monkeypatch, result=res)
    r = client.post("/api/eval/grade-claim", json={"claim": "60 days", "source_text": "…thirty (30) days…"})
    assert r.json()["verdict"] == "contradict"
    assert r.json()["quote"] == "within thirty (30) calendar days"


def test_low_coverage_from_thin_support(client: TestClient, monkeypatch):
    res = FactCheckResult(verdicts=[FactVerdict(fact="c", support=0.0, contradicted=False, evidence=None)],
                          coverage=0.0)
    _install_fake(monkeypatch, result=res)
    r = client.post("/api/eval/grade-claim", json={"claim": "x", "source_text": "unrelated page text"})
    assert r.json()["verdict"] == "low_coverage"
    assert r.json()["quote"] == ""


def test_transient_judge_failure_is_loud_low_coverage_never_agree(client: TestClient, monkeypatch):
    res = FactCheckResult(error=True, error_transient=True)
    _install_fake(monkeypatch, result=res)
    r = client.post("/api/eval/grade-claim", json={"claim": "x", "source_text": "some text"})
    d = r.json()
    assert d["verdict"] == "low_coverage"       # a judge failure must NEVER certify
    assert d["status"] == "error"
    assert d["error_transient"] is True


def test_judge_exception_is_low_coverage(client: TestClient, monkeypatch):
    _install_fake(monkeypatch, raises=RuntimeError("boom"))
    r = client.post("/api/eval/grade-claim", json={"claim": "x", "source_text": "some text"})
    assert r.status_code == 200
    assert r.json()["verdict"] == "low_coverage"
    assert r.json()["status"] == "error"


def test_missing_claim_is_422(client: TestClient):
    r = client.post("/api/eval/grade-claim", json={"source_text": "text"})
    assert r.status_code == 422


def test_missing_source_text_is_low_coverage_not_agree(client: TestClient):
    # No source to grade against → cannot support the claim. Loud, never agree.
    r = client.post("/api/eval/grade-claim", json={"claim": "x"})
    assert r.status_code == 200
    assert r.json()["verdict"] == "low_coverage"
    assert r.json()["status"] == "no_source"


def test_LOCKED_ruler_stage_is_pinned(client: TestClient, monkeypatch):
    """The load-bearing guarantee: grade-claim MUST call check_facts on the
    locked adjudication stage, never the default bandit-routed ruler."""
    cap: dict = {}
    res = FactCheckResult(verdicts=[FactVerdict(fact="c", support=1.0, evidence="q")], coverage=1.0)
    _install_fake(monkeypatch, result=res, capture=cap)
    client.post("/api/eval/grade-claim", json={"claim": "x", "source_text": "text"})
    assert cap.get("stage") == "rag_eval_adjudicate"
    assert cap.get("must_facts") == ["x"]        # chunk_only mode: claim as the must_fact
    assert cap.get("chunks") == [{"text": "text"}]
