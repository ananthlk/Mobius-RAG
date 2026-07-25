"""Prefix-grading: build per-strategy recall@K CURVES (Eval, blend-pullback
data-collection, 2026-07-24). For each (query, strategy), grade the top-K
prefix of that strategy's ranked chunks with the real rubric judge at
K in {1,3,5,10}. The rubric score on a top-K prefix == fraction of must_facts
present in the top-K chunks == recall@K. Aggregate by depth_bucket → the
recall curve whose PLATEAU is what the return-to-blend decision needs.

No LLM index self-reporting (rejected as hallucination-prone) — we grade
actual prefixes. Deterministic strategies (a/b/s) have a fixed ranking so
one pass is the curve; c/d are a single sample (repeat later for variance).

Reuses already-retrieved chunks in forced_filler_bank_run.json — no re-run.
Usage: .venv/bin/python scripts/prefix_grade.py --legs a,d --ks 1,3,5,10
"""
from __future__ import annotations
import argparse, asyncio, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
import yaml  # noqa: E402
from judge import adjudicate  # noqa: E402
try:
    from app.services.router.priors import compute_depth_bucket  # type: ignore
except Exception:  # noqa: BLE001
    compute_depth_bucket = None

FORCED = ROOT / "eval" / "artifacts" / "forced_filler_bank_run.json"
BANK = ROOT / "eval" / "queries_cmhc.yaml"
OUT = ROOT / "eval" / "artifacts" / "recall_curves.json"


def depth_of(pool_metadata):
    if compute_depth_bucket:
        try:
            return compute_depth_bucket(pool_metadata or {})
        except Exception:  # noqa: BLE001
            pass
    # fallback: empty metadata → bucket 4 (matches documented default)
    ps = (pool_metadata or {}).get("pool_size") or 0
    tsp = (pool_metadata or {}).get("top_score_percentile") or 0
    if not ps:
        return 4
    if tsp >= 0.8 and ps < 200:
        return 2
    if ps >= 400:
        return 3
    return 2


JUDGE_MODELS_SEEN: set = set()


async def grade(query_text, q, chunks):
    for attempt in range(4):
        try:
            _, score, _, model, _ = await adjudicate(
                query_text, q, {"chunks": chunks, "llm_answer": "", "confidence": None})
            # Lock verification: capture the ACTUAL judge model on every call so
            # the ruler's identity is provable from data, not just asserted in
            # model_registry. rag_eval_adjudicate is locked to gemini-2.5-pro;
            # if anything but pro appears here, the lock has drifted.
            JUDGE_MODELS_SEEN.add(model or "unknown")
            return score
        except Exception:  # noqa: BLE001
            if attempt == 3:
                return None
            await asyncio.sleep(4 * (attempt + 1))


EXPECTED_JUDGE = "gemini-2.5-pro"


def assert_locked_ruler():
    """FAIL-CLOSED: refuse to grade unless routed through the LLM Manager proxy
    (locked gemini-2.5-pro). Never silently dev-fall-back to an unlocked model
    and produce numbers on the wrong ruler. Load-bearing eval-integrity guard
    (Eval, 2026-07-24) — see adjudicator-calibration-spec.md §1b item 0."""
    import os
    if not os.getenv("CHAT_INTERNAL_LLM_URL") or not os.getenv("MOBIUS_SKILL_LLM_INTERNAL_KEY"):
        raise SystemExit(
            "REFUSING TO GRADE: CHAT_INTERNAL_LLM_URL / MOBIUS_SKILL_LLM_INTERNAL_KEY "
            "unset → judge would dev-fall-back to an UNLOCKED model (not gemini-2.5-pro). "
            "Run on GCP / set the proxy env so grading uses the locked ruler. "
            "Fail-closed by design — do not grade on the wrong ruler.")


def assert_judge_model_locked():
    """Belt-and-suspenders: after grading starts, verify the ACTUAL model was
    the locked pro. Aborts the batch immediately if a dev-fallback / non-pro
    model slipped through, rather than wasting the run and quarantining later."""
    bad = [m for m in JUDGE_MODELS_SEEN if EXPECTED_JUDGE not in (m or "") or "unknown" in (m or "")]
    if bad:
        raise SystemExit(
            f"REFUSING TO CONTINUE: judge model(s) {sorted(JUDGE_MODELS_SEEN)} "
            f"!= locked {EXPECTED_JUDGE}. The ruler is wrong — aborting the batch.")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", default="a,d")
    ap.add_argument("--ks", default="1,3,5,10")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="ONLY for the local 22-query methodology run on the KNOWN-unlocked "
                         "dev-fallback ruler; NEVER for authoritative GCP numbers.")
    args = ap.parse_args()
    if not args.allow_fallback:
        assert_locked_ruler()
    legs = args.legs.split(",")
    ks = [int(x) for x in args.ks.split(",")]

    forced = json.loads(FORCED.read_text())["results"]
    bank = {q["id"]: q for q in yaml.safe_load(BANK.read_text())["queries"]}
    out = json.loads(OUT.read_text()) if OUT.exists() else {}

    for r in forced:
        qid = r["id"]; q = bank.get(qid)
        if not q:
            continue
        db = depth_of(r.get("pool_metadata"))
        rec = out.setdefault(qid, {"depth_bucket": db, "legs": {}})
        rec["depth_bucket"] = db
        for leg in legs:
            ps = r["per_strategy"].get(leg, {})
            chunks = ps.get("chunks") or []
            curve = rec["legs"].setdefault(leg, {})
            for k in ks:
                if str(k) in curve:
                    continue
                prefix = chunks[:k]
                score = 0.0 if not prefix else await grade(r["query"], q, prefix)
                curve[str(k)] = score
                print(f"{qid} db{db} {leg} @{k}: {score}")
        # Fail-closed after the first real grade: abort if the ruler is wrong.
        if not args.allow_fallback and JUDGE_MODELS_SEEN:
            assert_judge_model_locked()
        OUT.write_text(json.dumps(out, indent=2))

    # aggregate curves by (depth_bucket, leg) — with MONOTONE ENVELOPE.
    # recall@K is non-decreasing in K by definition (top-K ⊇ top-K'), but the
    # LLM judge grades each prefix independently and its non-determinism can
    # produce dips (e.g. @1=0.25 then @3=0.00). A fact present in top-1 is
    # present in top-3, so we take the running max over K per (query,leg)
    # before aggregating. The stored recall_curves.json keeps RAW scores;
    # this envelope is a read-time correction.
    print("\n=== recall@K curves by (depth_bucket, leg), monotone-enveloped ===")
    agg = {}
    for qid, rec in out.items():
        db = rec["depth_bucket"]
        for leg, curve in rec["legs"].items():
            running = 0.0
            for k in sorted(curve, key=int):
                s = curve[k]
                if s is None:
                    continue
                running = max(running, s)  # monotone envelope
                agg.setdefault((db, leg), {}).setdefault(k, []).append(running)
    for (db, leg) in sorted(agg):
        cells = agg[(db, leg)]
        pts = " ".join(f"@{k}={sum(v)/len(v):.2f}(n{len(v)})" for k, v in sorted(cells.items(), key=lambda x: int(x[0])))
        print(f"  db{db} {leg}: {pts}")
    # Lock stamp: record the actual judge model(s) that graded this artifact.
    out["__judge_lock__"] = {"models_seen": sorted(JUDGE_MODELS_SEEN),
                             "expected": "rubric/gemini-2.5-pro"}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nJUDGE MODELS SEEN: {sorted(JUDGE_MODELS_SEEN)} (expected rubric/gemini-2.5-pro)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
