"""One-off VERBOSE inspector for a single query — shows EVERYTHING the 3-mode
grader sees and produces, not just the summary curve. Reuses the exact
synth + check_facts calls prefix_grade_3mode.py uses.

Runs locally → dev-fallback LLM (flash), so this is a MECHANICS view (what
gets scored, what the LLM writes, per-fact hit/miss) — NOT the authoritative
locked-2.5-pro number. Labeled as such.

Usage: .venv/bin/python scripts/inspect_one_query.py [qid] [legs] [k]
"""
import asyncio, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.services.fact_checker import check_facts
from app.services import llm_manager_client

QID = sys.argv[1] if len(sys.argv) > 1 else "cmhc001"
LEGS = (sys.argv[2] if len(sys.argv) > 2 else "a,b,c,d,s").split(",")
K = int(sys.argv[3]) if len(sys.argv) > 3 else 10
# Locked-pro ruler for BOTH synth and grading (Finding 1 fix): route
# check_facts through rag_eval_adjudicate (pro-locked), not the unlocked
# rag_fact_check stage.
LOCKED_STAGE = "rag_eval_adjudicate"
# Live artifact the web dashboard polls. Rewritten (atomically) after EVERY K so
# the page can render progress without the run finishing first.
OUT_JSON = Path(os.environ.get("KCURVE_JSON", "/tmp/kcurve_results.json"))


def write_state(state):
    tmp = OUT_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(OUT_JSON)

SYNTH_SYSTEM = (
    "You are a claims/payer support assistant. Answer the question using ONLY "
    "the provided source passages — state the codes, day-counts, and yes/no "
    "determinations they support. If the passages don't answer it, say so. "
    "Do not invent facts."
)
# Draft → critique → final (Ananth, 2026-07-24): prod Chat runs ReAct + a
# critique loop; our single-shot synth doesn't, so mode-b was both a lower
# bound AND structurally different (no self-correction — hence the
# hallucinations leg a/d produced). This critique step mirrors Chat's
# composer: catch unsupported claims + missed facts, then revise.
CRITIQUE_SYSTEM = (
    "You are a strict groundedness reviewer. Your ONLY job is to find claims in "
    "the DRAFT that the PASSAGES do NOT support — hallucinations to REMOVE.\n"
    "Go claim by claim. For EACH factual claim in the draft (each day-count, code, "
    "condition, program rule), quote the passage number that backs it. If you "
    "cannot point to a specific passage that states it, mark it UNSUPPORTED.\n"
    "Do NOT suggest adding anything. Do NOT use your own knowledge of payer rules "
    "— a fact you 'know' but that is not written in a passage is UNSUPPORTED, not "
    "missing. Output only:\n"
    "UNSUPPORTED: <list each unsupported claim, or 'none'>"
)


def line(c="─"):
    print(c * 78)


async def _gen(system, user):
    raw, meta = await llm_manager_client.generate(
        system=system, user=user, stage="rag_eval_adjudicate", max_tokens=2048)
    return raw.strip(), (meta or {}).get("model") or "unknown"


async def synth(query, chunks):
    """Draft → FACT-CHECK → final (Ananth: "the critique is really a fact
    check"). The critique is NOT an ad-hoc review prompt (that trusts the
    draft's own citations and misses hallucinations) — it's the SAME rigorous
    grounding critic we grade with (check_facts grounding_only), which
    verifies each claim against the actual passage text and returns the
    hallucinated list. The final removes exactly those, adds nothing."""
    # NO cap here: caller already sliced to top-K. The synth must see EXACTLY
    # the K chunks it's handed (at K=1, one chunk) — never re-widen the view.
    body = "\n\n".join(f"[{i+1}] {c.get('text','')}" for i, c in enumerate(chunks))
    draft, model = await _gen(SYNTH_SYSTEM, f"Question: {query}\n\nPassages:\n{body}\n\nAnswer:")
    crit = await check_facts(query=query, must_facts=[], chunks=chunks, answer=draft, stage=LOCKED_STAGE)
    halluc = list(crit.hallucinated_claims or [])
    if not halluc:
        return draft, draft, ["(none — draft already fully grounded)"], model
    halluc_txt = "\n".join(f"- {h}" for h in halluc)
    final, _ = await _gen(
        SYNTH_SYSTEM,
        f"Question: {query}\n\nPassages:\n{body}\n\nDraft answer:\n{draft}\n\n"
        f"A grounding check verified these specific claims are NOT supported by "
        f"ANY passage:\n{halluc_txt}\n\nRewrite the answer: DELETE each of those "
        f"claims exactly. Do NOT add anything. Keep everything else. Output the "
        f"corrected answer only:")
    return final, draft, halluc, model


def dump_result(tag, r):
    if r is None:
        print(f"  {tag}: <judge call failed>")
        return
    print(f"  {tag}:  coverage={r.coverage:.3f}  score={r.score:.3f}  "
          f"honest_abstain={r.honest_abstain}  model={r.model}")
    for v in r.verdicts:
        mark = "✓" if v.support >= 1.0 else ("~" if v.support >= 0.5 else "✗")
        print(f"     [{mark} {v.support:.1f}] {getattr(v, 'fact', getattr(v, 'claim', '?'))}")
    if r.hallucinated_claims:
        print(f"     HALLUCINATED (answer claims no passage backs):")
        for h in r.hallucinated_claims:
            print(f"       ! {h}")
    if r.reasoning:
        print(f"     reasoning: {r.reasoning[:400]}")


async def main():
    forced = json.load(open(ROOT / "eval/artifacts/forced_filler_bank_run.json"))
    q = next((r for r in forced["results"] if r["id"] == QID), None)
    if not q:
        raise SystemExit(f"query {QID} not found")

    print("\n" + "=" * 78)
    print(f"  QUERY {QID}  (K={K})   synth + judge both on LOCKED gemini-2.5-pro")
    print(f"                          (rag_eval_adjudicate stage; verify model= lines below)")
    print("=" * 78)
    print(f"\nQUESTION:\n  {q['query']}")
    print(f"\nGOLDEN ANSWER:\n  {q['golden_answer'].strip()}")
    print(f"\nMUST_FACTS (what recall is graded against):")
    for f in q["must_facts"]:
        print(f"  • {f}")

    KS = [1, 3, 5, 10]

    def cscore(rc):
        if rc is None or rc.error:
            return "excl"   # parse-failure / judge error — excluded, not 0
        return f"{rc.score:.2f}"

    # ---- live artifact skeleton (dashboard polls this) ----
    state = {
        "qid": QID,
        "question": q["query"],
        "golden_answer": q["golden_answer"].strip(),
        "must_facts": q["must_facts"],
        "judge": "rag_eval_adjudicate → locked gemini-2.5-pro",
        "status": "running",
        "started_at": time.time(),
        "updated_at": time.time(),
        "legs": [],
    }
    leg_label = {"a": "our vector index", "b": "internal arm b", "c": "internal arm c",
                 "d": "web (Google)", "s": "payor fact-store"}
    write_state(state)

    for leg in LEGS:
        ps = q["per_strategy"].get(leg)
        line("═")
        print(f"LEG '{leg}'  (occupancy={ps.get('occupancy') if ps else '?'})")
        line("═")
        leg_rec = {"leg": leg, "label": leg_label.get(leg, leg),
                   "occupancy": (ps.get("occupancy") if ps else 0),
                   "chunks_available": len(ps.get("chunks") or []) if ps else 0,
                   "ks": [], "empty": not (ps and ps.get("chunks"))}
        state["legs"].append(leg_rec)
        state["updated_at"] = time.time(); write_state(state)
        if leg_rec["empty"]:
            print("  (no chunks — leg returned nothing)")
            continue
        all_chunks = ps["chunks"]
        # Effective K values: a leg with <10 chunks caps out; dedupe so we
        # don't re-grade an identical prefix.
        eff_ks = sorted({min(k, len(all_chunks)) for k in KS})
        leg_rec["eff_ks"] = eff_ks
        write_state(state)

        print(f"  chunks available: {len(all_chunks)}   grading K ∈ {eff_ks}")
        print("  (at each K, synth + all 3 judges see ONLY the top-K chunks)\n")
        for k in eff_ks:
            chunks_k = all_chunks[:k]  # ONLY top-k — synth + all 3 judges see nothing deeper
            print(f"  ┌─ K={k}  (synth sees chunks 1..{k}) ─────────────────────────────")
            sys.stdout.flush()
            final, draft, critique, smodel = await synth(q["query"], chunks_k)
            print(f"  │ ① DRAFT ({smodel}):")
            for ln in draft.splitlines()[:10]:
                print(f"  │    {ln}")
            print(f"  │ ② FACT-CHECK (grounding critic → claims to REMOVE):")
            for h in critique:
                print(f"  │    ✗ {h}")
            print(f"  │ ③ FINAL (scored answer):")
            for ln in final.splitlines()[:10]:
                print(f"  │    {ln}")
            sys.stdout.flush()
            ra = await check_facts(query=q["query"], must_facts=q["must_facts"], chunks=chunks_k, answer=None, stage=LOCKED_STAGE)
            # b·draft: score the RAW pre-critique draft. This is the honest
            # "what synthesis produced". Comparing it to b·final exposes how much
            # the (table-blind) critique DELETED — the "critique damage" delta.
            rb_draft = await check_facts(query=q["query"], must_facts=q["must_facts"], chunks=chunks_k, answer=draft, stage=LOCKED_STAGE)
            rb = await check_facts(query=q["query"], must_facts=q["must_facts"], chunks=chunks_k, answer=final, stage=LOCKED_STAGE)
            rc = await check_facts(query=q["query"], must_facts=[], chunks=chunks_k, answer=final, stage=LOCKED_STAGE)
            print(f"  │ ── per-fact grade ──")
            facts = []
            for v in (ra.verdicts if ra else []):
                mark = "✓" if v.support >= 1.0 else ("~" if v.support >= 0.5 else "✗")
                in_answer = bool(rb and any(getattr(bv, 'fact', '') == getattr(v, 'fact', '') and bv.support >= 1.0 for bv in rb.verdicts))
                fact = getattr(v, 'fact', '?')
                print(f"  │    chunk[{mark}] answer[{'in-answer' if in_answer else 'DROPPED':>9}]  {fact}")
                facts.append({"fact": fact, "in_chunk": v.support, "in_answer": in_answer})
            jmodel = (ra.model if ra else '?')
            print(f"  ├─ K={k} SCORES │ (a)chunk-recall={ra.coverage:.2f}  "
                  f"(b)answer-cov={rb.coverage:.2f}  (b)honesty={rb.score:.2f}  "
                  f"(c)ground={cscore(rc)}   judge={jmodel}")
            print(f"  └────────────────────────────────────────────────────────────\n")
            sys.stdout.flush()
            leg_rec["ks"].append({
                "k": k,
                "draft": draft,
                "pruned": [] if (critique and critique[0].startswith("(none")) else list(critique),
                "final": final,
                "synth_model": smodel,
                "facts": facts,
                "mode_a_recall": round(ra.coverage, 3) if ra else None,
                "mode_b_draft": round(rb_draft.coverage, 3) if rb_draft else None,
                "mode_b_answercov": round(rb.coverage, 3) if rb else None,
                "critique_damage": (round((rb_draft.coverage or 0) - (rb.coverage or 0), 3)
                                    if (rb_draft and rb) else None),
                "mode_b_honesty": round(rb.score, 3) if rb else None,
                "mode_c_ground": None if (rc is None or rc.error) else round(rc.score, 3),
                "mode_c_excluded": bool(rc is None or rc.error),
                "judge": jmodel,
            })
            state["updated_at"] = time.time(); write_state(state)
    state["status"] = "done"; state["updated_at"] = time.time(); write_state(state)
    print(f"\nwrote live artifact → {OUT_JSON}")
    print()


asyncio.run(main())
