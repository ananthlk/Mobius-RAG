"""Unified 3-mode prefix-grading harness (Eval, blend-pullback, 2026-07-24).

Builds recall@K curves in ALL THREE adjudicator modes over the same prefixes,
reusing fact_checker.check_facts() — which already routes the three modes by
argument combo (verified 2026-07-24):
  (a) chunk-recall     = check_facts(must_facts, chunks, answer=None)  → .coverage
  (b) answer-complete  = check_facts(must_facts, chunks, answer=SYNTH) → .coverage
  (c) groundedness     = check_facts([],         chunks, answer=SYNTH) → .score   (reference-free)

For each (query, strategy, K∈{1,3,5,10}): grade the top-K prefix in all three
modes. (a)@K = recall@K; (b)@K = answer-completeness@K; (a)−(b) gap = the
synthesis-loss curve; (c) is the prod-deployable proxy, calibrated here against
(a)/(b) where golden facts exist.

FAIL-CLOSED: refuses to run unless routed through the locked gemini-2.5-pro
proxy (never dev-fall-back to grade authoritative numbers). Monotone envelope +
weighted/primary-gate applied at read time (raw stored).

Usage: .venv/bin/python scripts/prefix_grade_3mode.py --legs a,d --ks 1,3,5,10
"""
from __future__ import annotations
import argparse, asyncio, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
import yaml  # noqa: E402
from app.services.fact_checker import check_facts  # noqa: E402
from app.services import llm_manager_client  # noqa: E402
try:
    from app.services.router.priors import compute_depth_bucket  # type: ignore
except Exception:  # noqa: BLE001
    compute_depth_bucket = None

FORCED = ROOT / "eval" / "artifacts" / "forced_filler_bank_run.json"
BANK = ROOT / "eval" / "queries_cmhc.yaml"
OUT = ROOT / "eval" / "artifacts" / "recall_curves_3mode.json"
EXPECTED_JUDGE = "gemini-2.5-pro"
JUDGE_MODELS_SEEN: set = set()

SYNTH_SYSTEM = (
    "You are a claims/payer support assistant. Answer the question using ONLY "
    "the provided source passages — state the codes, day-counts, and yes/no "
    "determinations they support. If the passages don't answer it, say so. "
    "Do not invent facts."
)


def assert_locked_ruler(allow_fallback: bool):
    if allow_fallback:
        return
    if not os.getenv("CHAT_INTERNAL_LLM_URL") or not os.getenv("MOBIUS_SKILL_LLM_INTERNAL_KEY"):
        raise SystemExit(
            "REFUSING TO GRADE: proxy env unset → judge/synth would dev-fall-back to an "
            "UNLOCKED model. Deploy / set CHAT_INTERNAL_LLM_URL so grading uses locked "
            "gemini-2.5-pro. Fail-closed by design.")


def depth_of(pm):
    if compute_depth_bucket:
        try:
            return compute_depth_bucket(pm or {})
        except Exception:  # noqa: BLE001
            pass
    return 4 if not (pm or {}).get("pool_size") else 2


async def synthesize(query, chunks):
    body = "\n\n".join(f"[{i+1}] {c.get('text','')}" for i, c in enumerate(chunks[:12]))
    raw, meta = await llm_manager_client.generate(
        system=SYNTH_SYSTEM, user=f"Question: {query}\n\nPassages:\n{body}\n\nAnswer:",
        # 3000 not 1024: 1024 truncated answers mid-sentence (verified live —
        # cut off before stating all must_facts), systematically depressing
        # mode-b answer-completeness. Payer answers need room for multiple
        # day-counts/codes. (Eval, 2026-07-24, Retriever finding 2.)
        stage="rag_eval_adjudicate", max_tokens=3000)
    JUDGE_MODELS_SEEN.add((meta or {}).get("model") or "unknown")
    return raw.strip()


async def _cf(query, must_facts, chunks, answer):
    for attempt in range(4):
        try:
            # Route grading through the LOCKED adjudicate stage (pro-only), NOT
            # check_facts's default rag_fact_check stage (bandit-routes pro/flash).
            # Same model (pro) + same check_facts prompt = identical grade; the
            # stage is just the routing key. (Eval, 2026-07-24, Retriever finding 1.)
            r = await check_facts(query=query, must_facts=must_facts, chunks=chunks,
                                  answer=answer, stage="rag_eval_adjudicate")
            JUDGE_MODELS_SEEN.add(r.model or "unknown")
            if r.error and not r.error_transient:
                return None
            return r
        except Exception:  # noqa: BLE001
            if attempt == 3:
                return None
            await asyncio.sleep(4 * (attempt + 1))


async def grade_prefix(query, must_facts, chunks_k):
    """Return {a, b, c} for one prefix. a=coverage(chunks), b=coverage(answer),
    c=groundedness score(answer, no golden)."""
    ra = await _cf(query, must_facts, chunks_k, None)
    a = ra.coverage if ra else None
    if not chunks_k:
        return {"a": 0.0, "b": 0.0, "c": 0.0}
    synth = await synthesize(query, chunks_k)
    rb = await _cf(query, must_facts, chunks_k, synth)
    rc = await _cf(query, [], chunks_k, synth)  # grounding_only → reference-free
    return {"a": a, "b": (rb.coverage if rb else None), "c": (rc.score if rc else None)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", default="a,d")
    ap.add_argument("--ks", default="1,3,5,10")
    ap.add_argument("--allow-fallback", action="store_true")
    args = ap.parse_args()
    assert_locked_ruler(args.allow_fallback)
    legs = args.legs.split(","); ks = [int(x) for x in args.ks.split(",")]

    forced = json.loads(FORCED.read_text())["results"]
    bank = {q["id"]: q for q in yaml.safe_load(BANK.read_text())["queries"]}
    out = json.loads(OUT.read_text()) if OUT.exists() else {}

    for r in forced:
        qid = r["id"]; q = bank.get(qid)
        if not q:
            continue
        mf = q.get("expected", {}).get("must_facts") or q.get("must_facts") or []
        db = depth_of(r.get("pool_metadata"))
        rec = out.setdefault(qid, {"depth_bucket": db, "legs": {}}); rec["depth_bucket"] = db
        for leg in legs:
            chunks = (r["per_strategy"].get(leg) or {}).get("chunks") or []
            curve = rec["legs"].setdefault(leg, {})
            for k in ks:
                if str(k) in curve:
                    continue
                curve[str(k)] = await grade_prefix(r["query"], mf, chunks[:k])
                print(f"{qid} db{db} {leg} @{k}: {curve[str(k)]}")
        if not args.allow_fallback and JUDGE_MODELS_SEEN:
            bad = [m for m in JUDGE_MODELS_SEEN if EXPECTED_JUDGE not in (m or "") or "unknown" in (m or "")]
            if bad:
                raise SystemExit(f"REFUSING: judge model(s) {sorted(JUDGE_MODELS_SEEN)} != {EXPECTED_JUDGE}")
        OUT.write_text(json.dumps(out, indent=2))

    out["__judge_lock__"] = {"models_seen": sorted(JUDGE_MODELS_SEEN), "expected": f"rubric/{EXPECTED_JUDGE}"}
    OUT.write_text(json.dumps(out, indent=2))
    # aggregate: per (depth, leg, mode), monotone-enveloped
    print("\n=== 3-mode recall@K by (depth, leg) ===")
    agg = {}
    for qid, rec in out.items():
        if qid.startswith("__"):
            continue
        db = rec["depth_bucket"]
        for leg, curve in rec["legs"].items():
            run = {"a": 0.0, "b": 0.0, "c": 0.0}
            for k in sorted(curve, key=int):
                for m in ("a", "b", "c"):
                    s = curve[k].get(m)
                    if s is None:
                        continue
                    run[m] = max(run[m], s)
                    agg.setdefault((db, leg, m), {}).setdefault(k, []).append(run[m])
    for (db, leg, m) in sorted(agg):
        cells = agg[(db, leg, m)]
        pts = " ".join(f"@{k}={sum(v)/len(v):.2f}" for k, v in sorted(cells.items(), key=lambda x: int(x[0])))
        print(f"  db{db} {leg} mode-{m}: {pts}")
    print(f"\nJUDGE MODELS: {sorted(JUDGE_MODELS_SEEN)}\nwrote {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
