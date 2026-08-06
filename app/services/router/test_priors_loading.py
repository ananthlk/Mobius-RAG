"""Priors file-loading tests — the swappability contract.

The contract under test (Ananth/Retriever requirement):
  Editing the priors YAML changes Router's behavior WITHOUT a code change or
  redeploy. Hardcoded defaults are a last-resort fallback only.
"""

import os
import textwrap

import pytest

from app.services.router.allocation import allocate_strategies, AnswerSlot
from app.services.router.priors import (
    default_priors_path,
    get_priors_version,
    load_priors,
    lookup_priors,
    lookup_priors_qclass_fallback,
)


POOL_DEPTH_2 = {"top_score_percentile": 0.60, "pool_size": 400}


def _write_yaml(path, s_lift_depth2: float, a_lift_depth2: float = 0.543,
                s_n: int = 8):
    path.write_text(textwrap.dedent(f"""
        seed_priors:
          depth_2:
            s:
              recall_lift: {s_lift_depth2}
              n: {s_n}
              k0: 10
              latency_p50_ms: 100
              cost_per_attempt: 0
              accuracy_estimate: 0.5
            a:
              recall_lift: {a_lift_depth2}
              latency_p50_ms: 500
              cost_per_attempt: 1
              accuracy_estimate: 0.5
        fallback_qclass_priors:
          a:
            tight_pool: 0.5
    """))


def _slot():
    return AnswerSlot(slot_id="slot_0", slot_semantics="direct_answer", capacity=5,
                      rewritten_query="q", required=True, priority=0)


def _posture():
    return {
        "speed_budget": "interactive",
        "confidence_bar": 0.85,
        "caller_mode": "chat.default",  # adjusted bar = .7225
        "max_attempts_per_slot": 6,
        # tests simulate PAYOR queries so tag-gated 's' stays eligible
        "gate_j_codes": ["payor.sunshine_health"],
    }


@pytest.mark.live_priors
class TestFileIsPrimarySource:
    """DELIBERATE change-detectors on the LIVE eval file — the one test class
    that must be re-derived whenever Eval folds new data into the priors.
    The 2026-07-23 Beta-update fold was RETRACTED same day (Eval's
    spot-check proved the recall grading inflated — token-presence proxy).
    Current pins (2026-08-05): depth_3's a/b/c/d are a REAL, ruler-confirmed
    fold (n=65, k0=9) — the first cell to graduate past seed. Everything
    else (s at every depth; a/b/c/d elsewhere) is still pure n=8 seed."""

    def test_loads_real_eval_file_by_default(self):
        bundle = load_priors(force_reload=True)
        assert bundle.source == "file"
        # Values match eval/priors_bootstrap.yaml, NOT any hardcoded snapshot
        prof = lookup_priors(2, "a", bundle)
        assert prof.recall_lift == pytest.approx(0.543)
        assert prof.n == 8  # pure seed pseudo-count (fold retracted)
        assert prof.latency_p50_ms == 500
        # cost_per_attempt zeroed across ALL strategies (Ananth, 2026-07-26):
        # these values were never derived from real calibration -- same
        # seed-data status as accuracy_estimate/recall_lift -- so cost no
        # longer participates as an unvalidated tie-breaker in the LB-
        # maximizer's argmax (allocation.py/optimizer.py: required slots
        # break ties on (lb, confidence, -cost, -latency)).
        assert prof.cost == pytest.approx(0.0)

    def test_version_string_carries_file_identity(self):
        bundle = load_priors(force_reload=True)
        assert get_priors_version(bundle).startswith("file:priors_bootstrap.yaml@")

    def test_qclass_fallback_read_from_file(self):
        bundle = load_priors(force_reload=True)
        prof = lookup_priors_qclass_fallback("tight_pool", "a", bundle)
        assert prof.recall_lift == pytest.approx(0.500)  # from fallback_qclass_priors
        prof_unknown = lookup_priors_qclass_fallback("no_such_class", "a", bundle)
        assert prof_unknown.recall_lift == pytest.approx(0.543)  # depth-2 fallback

    def test_depth_3_abcd_are_real_fold_everything_else_seed(self):
        """RE-DERIVED 2026-08-05 (per this class's own docstring): depth_3's
        a/b/c/d cells are now a REAL fold — ruler confirmed (judge_model=
        factcheck/gemini-2.5-pro pulled from persisted job traces, 264
        gradings uniform), n=65 (22-query x 3-mode sweep, less the 4
        cmhc013 x chat.copilot rows that are genuinely bucket_2 pool_size,
        not bucket_3 — excluding them was itself a real, twice-repeated
        population-contamination catch on this thread). Everything else
        (s at every depth; a/b/c/d at depths 0/1/2/4) remains pure n=8
        seed — no other cell has a comparable real observation yet."""
        bundle = load_priors(force_reload=True)
        for sid in ("a", "b", "c", "d"):
            prof = lookup_priors(3, sid, bundle)
            assert prof.n == 65, f"{sid} depth_3 n"
            assert prof.k0 == 9, f"{sid} depth_3 k0"
        assert lookup_priors(3, "s", bundle).n == 8  # s untouched by this fold
        for depth in (0, 1, 2, 4):
            for sid in ("a", "b", "c", "d", "s"):
                assert lookup_priors(depth, sid, bundle).n == 8, f"{sid} depth_{depth} n"


class TestSwapWithoutCodeChange:
    def test_editing_yaml_changes_allocation(self, tmp_path, monkeypatch):
        """THE core swappability test: same code, two YAML files, different ladders.

        RE-DERIVED 2026-08-05 (best-LB-first + SUPPLEMENT_ONLY): the ONLY
        two cells in this file are s/a, and SUPPLEMENT_ONLY excludes s from
        the sole/FIRST rung whenever a (non-supplement) is viable —
        REGARDLESS of s's LB, by design (a categorical rule, not a
        confidence comparison: s is a fact-store lookup, never sole
        evidence). So the chain composition [a,s] is now IDENTICAL in both
        scenarios; the observable behavior change moves to STATUS/LB:
        A: s lift .5 @ n=8 (lb .249) → chain doesn't clear, UNDER_CONFIDENT.
        B: s lift .95 @ n=200 (lb .918) → same [a,s] chain now clears the
        bar comfortably (composed LB .941), CLEARED.
        Still proves n-per-cell flows from the file into the enforced LB —
        same file-swap contract, just observed through status/LB now that
        SUPPLEMENT_ONLY structurally caps what chain COMPOSITION can show."""
        yaml_a = tmp_path / "priors_a.yaml"
        yaml_b = tmp_path / "priors_b.yaml"
        _write_yaml(yaml_a, s_lift_depth2=0.500, s_n=8)
        _write_yaml(yaml_b, s_lift_depth2=0.950, s_n=200)

        monkeypatch.setenv("ROUTER_PRIORS_PATH", str(yaml_a))
        ladder_a = allocate_strategies([_slot()], {"slot_0": POOL_DEPTH_2}, _posture())
        assert ladder_a.per_slot["slot_0"] == ["a", "s"]
        assert ladder_a.per_slot_status["slot_0"] == "UNDER_CONFIDENT"

        monkeypatch.setenv("ROUTER_PRIORS_PATH", str(yaml_b))
        ladder_b = allocate_strategies([_slot()], {"slot_0": POOL_DEPTH_2}, _posture())
        # behavior changed, zero code change — same chain shape, but now clears
        assert ladder_b.per_slot["slot_0"] == ["a", "s"]
        assert ladder_b.per_slot_status["slot_0"] == "CLEARED"
        assert ladder_b.per_slot_lb["slot_0"] > ladder_a.per_slot_lb["slot_0"]

    def test_rewriting_same_file_invalidates_cache_via_mtime(self, tmp_path, monkeypatch):
        """Eval's loop rewrites the file in place → Router picks it up (mtime cache)."""
        yaml_path = tmp_path / "priors.yaml"
        _write_yaml(yaml_path, s_lift_depth2=0.500)
        monkeypatch.setenv("ROUTER_PRIORS_PATH", str(yaml_path))

        v1 = load_priors()
        assert lookup_priors(2, "s", v1).recall_lift == pytest.approx(0.500)

        _write_yaml(yaml_path, s_lift_depth2=0.950)
        os.utime(yaml_path, ns=(1_000_000_000_000, 2_000_000_000_000_000_000))  # force mtime change

        v2 = load_priors()
        assert lookup_priors(2, "s", v2).recall_lift == pytest.approx(0.950)
        assert v1.version != v2.version

    def test_env_var_overrides_default_path(self, tmp_path, monkeypatch):
        yaml_path = tmp_path / "elsewhere.yaml"
        _write_yaml(yaml_path, s_lift_depth2=0.42)
        monkeypatch.setenv("ROUTER_PRIORS_PATH", str(yaml_path))
        assert default_priors_path() == yaml_path
        assert lookup_priors(2, "s", load_priors()).recall_lift == pytest.approx(0.42)


class TestHardcodedIsLastResortOnly:
    def test_missing_file_falls_back_with_hardcoded_source(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ROUTER_PRIORS_PATH", str(tmp_path / "does_not_exist.yaml"))
        bundle = load_priors()
        assert bundle.source == "hardcoded"
        assert bundle.version == "hardcoded-fallback-2026-07-23"
        # Still functional: allocation works on the fallback
        ladder = allocate_strategies([_slot()], {"slot_0": POOL_DEPTH_2}, _posture())
        assert len(ladder.per_slot["slot_0"]) >= 1

    def test_malformed_yaml_falls_back(self, tmp_path, monkeypatch):
        bad = tmp_path / "bad.yaml"
        bad.write_text("seed_priors: [this, is, not, a, mapping]")
        monkeypatch.setenv("ROUTER_PRIORS_PATH", str(bad))
        bundle = load_priors()
        assert bundle.source == "hardcoded"

    def test_non_mapping_yaml_falls_back(self, tmp_path, monkeypatch):
        bad = tmp_path / "bad2.yaml"
        bad.write_text("- just\n- a\n- list\n")
        monkeypatch.setenv("ROUTER_PRIORS_PATH", str(bad))
        assert load_priors().source == "hardcoded"


class TestSanitization:
    def test_negative_lift_clamped_and_garbage_latency_defaulted(self, tmp_path, monkeypatch):
        yaml_path = tmp_path / "weird.yaml"
        yaml_path.write_text(textwrap.dedent("""
            seed_priors:
              depth_2:
                a:
                  recall_lift: -0.5
                  latency_p50_ms: garbage
                  cost_per_attempt: -3
                  accuracy_estimate: 1.7
                s:
                  recall_lift: 0.6
                  latency_p50_ms: 100
                  cost_per_attempt: 0
                  accuracy_estimate: 0.5
        """))
        monkeypatch.setenv("ROUTER_PRIORS_PATH", str(yaml_path))
        bundle = load_priors()
        assert bundle.source == "file"
        a = lookup_priors(2, "a", bundle)
        assert a.recall_lift == 0.0        # clamped from -0.5
        assert a.latency_p50_ms == 1000    # defaulted from garbage
        assert a.cost == 0.0               # clamped from -3
        assert a.accuracy_estimate == 1.0  # clamped from 1.7
        # allocation must skip the zero-lift 'a', use 's' only — no crash
        ladder = allocate_strategies(
            [_slot()], {"slot_0": POOL_DEPTH_2},
            {"speed_budget": "interactive", "confidence_bar": 0.5,
             "caller_mode": "batch", "max_attempts_per_slot": 6,
             "gate_j_codes": ["payor.sunshine_health"]},
        )
        assert ladder.per_slot["slot_0"] == ["s"]

    def test_unrecognized_depth_keys_skipped(self, tmp_path, monkeypatch):
        yaml_path = tmp_path / "depths.yaml"
        yaml_path.write_text(textwrap.dedent("""
            seed_priors:
              depth_banana:
                a: {recall_lift: 0.9, latency_p50_ms: 500, cost_per_attempt: 1, accuracy_estimate: 0.5}
              depth_2:
                a: {recall_lift: 0.4, latency_p50_ms: 500, cost_per_attempt: 1, accuracy_estimate: 0.5}
        """))
        monkeypatch.setenv("ROUTER_PRIORS_PATH", str(yaml_path))
        bundle = load_priors()
        assert bundle.source == "file"
        assert lookup_priors(2, "a", bundle).recall_lift == pytest.approx(0.4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestEmpiricalCellValidation:
    """Eval's fail-closed rule: explicit n ⇒ explicit k0 REQUIRED. A
    measurement claim without its capacity is malformed — reject the cell
    (never silently rebase); seeds (no n key) keep harmless defaults."""

    def test_n_without_k0_rejected(self, tmp_path, caplog):
        import logging
        import textwrap
        f = tmp_path / "p.yaml"
        f.write_text(textwrap.dedent("""
            seed_priors:
              depth_2:
                a: {recall_lift: 0.7, latency_p50_ms: 500, cost_per_attempt: 1,
                    accuracy_estimate: 0.5, n: 30}
                b: {recall_lift: 0.5, latency_p50_ms: 1500, cost_per_attempt: 1,
                    accuracy_estimate: 0.5, n: 30, k0: 10}
                s: {recall_lift: 0.5, latency_p50_ms: 100, cost_per_attempt: 0,
                    accuracy_estimate: 0.5}
        """))
        with caplog.at_level(logging.WARNING):
            bundle = load_priors(path=f, force_reload=True)
        assert 2 not in bundle.by_depth.get("a", {})   # rejected
        assert bundle.by_depth["b"][2].n == 30          # valid triple kept
        assert bundle.by_depth["s"][2].n == 8           # seed: defaults fine
        assert any("REJECTING" in r.getMessage() and "n without k0" in r.getMessage()
                   for r in caplog.records)
