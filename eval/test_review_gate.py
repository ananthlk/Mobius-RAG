"""Branch coverage for the §7 approvability predicate (eval/review_gate.py).

Pure-function tests — no DB. Each builds a fetch-shaped cell dict and asserts the
verdict. Run: python -m pytest eval/test_review_gate.py -q
"""
from eval.review_gate import evaluate_cell, evaluate_run, N_FLOOR_OVERRIDE, N_FLOOR_CLEAN


def _cell(**over):
    """A clean, healthy, first-publish cell; override fields per test."""
    base = {
        "cell_id": 1,
        "bucket": "depth_3",
        "strategy": "b",
        "caller_mode": "default",
        "sha256": "deadbeef",
        "reconciliation_ok": True,
        "reconciliation_delta": 0.0,
        "n": 65,
        "authority_n": 64,
        "appliable": {"recall_lift": 0.7185, "accuracy_estimate": 0.5789,
                      "authority": 0.9804, "k0": 9},
        "latency_p50_ms": 1200,
        "warnings": [],
        "fact_checker_version": "fact_check_v1.2026-07-31",
        "population_rules_version": 1,
        "currently_published": None,
    }
    base.update(over)
    return base


def test_clean_first_publish():
    v = evaluate_cell(_cell())
    assert v["verdict"] == "clean"
    assert v["approvable"] is True
    assert v["sha"] == "deadbeef"
    assert v["hard_block"] is None
    assert v["gates"] == []
    assert v["sha_diff"]["is_first_publish"] is True


def test_reconciliation_false_is_hard_block():
    v = evaluate_cell(_cell(reconciliation_ok=False))
    assert v["verdict"] == "hard_block"
    assert v["approvable"] is False
    assert v["sha"] is None                       # cannot sign a non-approvable cell
    assert v["hard_block"]["code"] == "reconciliation_failed"


def test_missing_n_is_hard_block():
    v = evaluate_cell(_cell(n=None))
    assert v["approvable"] is False
    assert v["hard_block"]["code"] == "n_missing"


def test_thin_n_below_override_floor_needs_reason():
    v = evaluate_cell(_cell(n=N_FLOOR_OVERRIDE - 1))
    assert v["verdict"] == "needs_override"
    assert v["approvable"] is True                 # approvable, but gated
    g = [g for g in v["gates"] if g["code"] == "thin_n_override"][0]
    assert g["severity"] == "override"
    assert g["requires_reason"] is True


def test_thin_n_between_floors_needs_ack():
    v = evaluate_cell(_cell(n=(N_FLOOR_OVERRIDE + N_FLOOR_CLEAN) // 2))
    assert v["verdict"] == "needs_ack"
    g = [g for g in v["gates"] if g["code"] == "thin_n_ack"][0]
    assert g["requires_reason"] is False


def test_n_at_clean_floor_is_clean():
    v = evaluate_cell(_cell(n=N_FLOOR_CLEAN))
    assert v["verdict"] == "clean"


def test_population_warning_becomes_ack():
    v = evaluate_cell(_cell(warnings=["accuracy_estimate clamped (efficiency>1)"]))
    assert v["verdict"] == "needs_ack"
    assert any(g["code"] == "population_warning" for g in v["gates"])


def test_override_dominates_ack_in_verdict():
    # thin-n override + a warning ack -> the stronger (override) sets the verdict.
    v = evaluate_cell(_cell(n=5, warnings=["some warning"]))
    assert v["verdict"] == "needs_override"


def test_sha_diff_reports_changed_fields():
    pub = {
        "sha256": "oldsha",
        "appliable": {"recall_lift": 0.7000, "accuracy_estimate": 0.5789,
                      "authority": 0.9804, "k0": 9},
        "n": 60,
        "published_at": "2026-08-10T00:00:00",
        "fact_checker_version": "fact_check_v1.2026-07-31",
    }
    v = evaluate_cell(_cell(currently_published=pub))
    d = v["sha_diff"]
    assert d["is_first_publish"] is False
    assert d["sha_changed"] is True
    assert "recall_lift" in d["changed_fields"]
    assert d["changed_fields"]["recall_lift"]["delta"] == round(0.7185 - 0.7000, 4)
    assert "accuracy_estimate" not in d["changed_fields"]   # unchanged
    assert v["verdict"] == "clean"


def test_no_op_republish_flagged():
    pub = {
        "sha256": "deadbeef",                     # SAME sha as the cell
        "appliable": {"recall_lift": 0.7185, "accuracy_estimate": 0.5789,
                      "authority": 0.9804, "k0": 9},
        "n": 65,
        "published_at": "2026-08-10T00:00:00",
        "fact_checker_version": "fact_check_v1.2026-07-31",
    }
    v = evaluate_cell(_cell(currently_published=pub))
    assert v["sha_diff"]["no_op"] is True
    assert any(g["code"] == "no_op_republish" for g in v["gates"])


def test_ruler_version_mismatch_is_ack_not_block():
    pub = {
        "sha256": "oldsha",
        "appliable": {"recall_lift": 0.70, "accuracy_estimate": 0.58,
                      "authority": 0.98, "k0": 9},
        "n": 60,
        "published_at": "2026-08-10T00:00:00",
        "fact_checker_version": "fact_check_v1.2026-06-01",   # older ruler
    }
    v = evaluate_cell(_cell(currently_published=pub))
    assert v["approvable"] is True
    assert any(g["code"] == "ruler_version_mismatch" for g in v["gates"])
    assert v["verdict"] == "needs_ack"


def test_reconciliation_delta_none_still_functions():
    # Pre-v1.1: delta column absent -> None. Hard block still works off the bool.
    v = evaluate_cell(_cell(reconciliation_delta=None))
    assert v["reconciliation"]["delta"] is None
    assert v["verdict"] == "clean"                # bool True -> not blocked


def test_run_rollup_counts():
    cells = [
        _cell(cell_id=1),                                        # clean
        _cell(cell_id=2, reconciliation_ok=False, strategy="a"),  # hard_block
        _cell(cell_id=3, n=10, strategy="c"),                     # needs_override
        _cell(cell_id=4, warnings=["w"], strategy="d"),           # needs_ack
    ]
    r = evaluate_run(cells)
    s = r["summary"]
    assert s["total"] == 4
    assert s["clean"] == 1
    assert s["hard_block"] == 1
    assert s["needs_override"] == 1
    assert s["needs_ack"] == 1
    assert s["blocked_slots"] == ["depth_3/a/default"]
