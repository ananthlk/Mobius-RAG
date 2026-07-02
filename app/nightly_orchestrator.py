"""Nightly pipeline orchestrator — server-side port of scripts/run_nightly_pipeline.sh.

Runs the whole corpus+lexicon loop in a daemon thread and exposes live per-step
state for the Repository-tab UI to poll. Faithful to the validated bash driver:
each step hits the SAME tested endpoints (RAG admin over localhost, lexicon svc
over its URL, eval via the calibrate trigger) so behaviour matches the CLI run.

Sequence (bracketed by the 5-axis calibration eval):
  infra_up → baseline_eval → publish(QA→RAG,gated) → retag → chunk → embed
  → gate → freeze → final_eval → push(gated) → infra_down → lift

Serialization invariants (from eval/calibration/NIGHTLY_EVAL_RUNBOOK.md):
  * evals and corpus writes never overlap; workers idle + queues drain before final
  * DB resize never overlaps an eval (infra steps live outside the bracket)
  * judge model stays locked across the bracket

Infra (DB resize + worker scaling) is delegated to nightly_infra.py; if that is
not configured the infra steps report 'skipped' and the rest proceeds (gentle).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── Live state (single concurrent run; mirrors the retag/build-eu job dicts) ──
_STEP_DEFS = [
    ("infra_up",      "Scale up infra"),
    ("baseline_eval", "Baseline eval"),
    ("publish",       "Publish lexicon QA→RAG"),
    ("retag",         "Retag in place"),
    ("chunk",         "Chunk new docs"),
    ("embed",         "Embed → publish"),
    ("gate",          "Integrity gate"),
    ("freeze",        "Freeze corpus"),
    ("final_eval",    "Final eval"),
    ("push",          "Push to chat"),
    ("infra_down",    "Revert infra"),
    ("lift",          "Lift report"),
]

_NIGHTLY: dict = {
    "running": False,
    "stop": False,
    "started_at": None,
    "finished_at": None,
    "current": None,
    "error": None,
    "opts": {},
    "steps": [],
    "eval_run_id": None,   # the calibration run currently in flight (for cell-level progress)
    "baseline": None,      # calibration_summary JSON
    "final": None,
    "lift": None,          # {metric: {baseline, final, delta}}
    "gate": None,          # {passed, published, documents_total, frac, stale_tags}
    "push_done": False,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_LOG_CAP = 60   # keep the last N log lines per step (debug trail, bounded)


def _init_steps() -> None:
    _NIGHTLY["steps"] = [
        {"key": k, "label": lbl, "status": "pending", "detail": "",
         "started_at": None, "ended_at": None, "log": []}
        for k, lbl in _STEP_DEFS
    ]


def _step(key: str, status: str, detail: str | None = None) -> None:
    """Update a step. Every status transition or changed detail is appended to
    the step's `log` (timestamped, deduped, bounded) so the UI can expand each
    step and see its full progression for visibility + debugging."""
    hhmmss = datetime.now(timezone.utc).strftime("%H:%M:%S")
    for s in _NIGHTLY["steps"]:
        if s["key"] == key:
            prev_status, prev_detail = s["status"], s["detail"]
            s["status"] = status
            if detail is not None:
                s["detail"] = detail
            if status == "running" and not s["started_at"]:
                s["started_at"] = _now()
            if status in ("done", "skipped", "failed") and not s["ended_at"]:
                s["ended_at"] = _now()
            # log line on any status change or new detail (dedup consecutive)
            line = f"[{hhmmss}] {status}" + (f" · {detail}" if detail else "")
            if status != prev_status or (detail is not None and detail != prev_detail):
                if not s["log"] or s["log"][-1].split("] ", 1)[-1] != line.split("] ", 1)[-1]:
                    s["log"].append(line)
                    if len(s["log"]) > _LOG_CAP:
                        s["log"] = s["log"][-_LOG_CAP:]
            break
    if status == "running":
        _NIGHTLY["current"] = key


def _log(key: str, msg: str) -> None:
    """Append an ad-hoc log line to a step without changing its status/detail."""
    hhmmss = datetime.now(timezone.utc).strftime("%H:%M:%S")
    for s in _NIGHTLY["steps"]:
        if s["key"] == key:
            s["log"].append(f"[{hhmmss}] {msg}")
            if len(s["log"]) > _LOG_CAP:
                s["log"] = s["log"][-_LOG_CAP:]
            break


def _stopping() -> bool:
    return bool(_NIGHTLY.get("stop"))


# ── HTTP helpers (RAG = localhost self-calls; admin is open on dev) ───────────
def _base() -> str:
    return f"http://localhost:{os.getenv('PORT', '8080')}"


def _http(method: str, url: str, body: dict | None = None,
          headers: dict | None = None, timeout: int = 120) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode()
    return json.loads(raw) if raw else {}


def _rag_get(path: str, timeout: int = 120) -> dict:
    return _http("GET", _base() + path, timeout=timeout)


def _rag_post(path: str, body: dict | None = None, timeout: int = 120) -> dict:
    return _http("POST", _base() + path, body=body if body is not None else {}, timeout=timeout)


# ── Eval bracket (trigger → poll with stall guard → summary) ─────────────────
def _run_eval(notes: str, bank: str) -> dict | None:
    """Run one calibration to completion; return its calibration_summary (or None)."""
    active = _rag_get("/api/eval/active", timeout=30)
    if active.get("active"):
        _step(_NIGHTLY["current"], "failed", "another eval already active")
        return None
    _rag_post("/api/eval/calibrate/trigger", {"notes": notes, "bank_path": bank}, timeout=30)
    time.sleep(5)
    rid = None
    for _ in range(6):
        rid = (_rag_get("/api/eval/active", timeout=30) or {}).get("run_id")
        if rid:
            break
        time.sleep(5)
    if not rid:
        _log(_NIGHTLY["current"], "could not resolve run_id after trigger")
        return None
    _NIGHTLY["eval_run_id"] = rid
    _log(_NIGHTLY["current"], f"run_id={rid} · bank={bank}")
    last, stall = -1, 0
    while not _stopping():
        pr = _rag_get(f"/api/eval/runs/{rid}/progress", timeout=30)
        nc = pr.get("n_completed")
        nq = pr.get("n_queries")
        running = pr.get("is_running")
        _step(_NIGHTLY["current"], "running", f"{nc}/{nq} cells")
        if running is False:
            break
        if nc == last:
            stall += 1
        else:
            stall, last = 0, nc
        if stall >= 10:   # >5 min no progress → in-process task died
            _step(_NIGHTLY["current"], "failed", f"stalled at {nc} (needs durable driver)")
            _NIGHTLY["eval_run_id"] = None
            return None
        time.sleep(30)
    summary = _rag_get(f"/api/eval/runs/{rid}/calibration_summary", timeout=30)
    _NIGHTLY["eval_run_id"] = None
    return summary


def _compute_lift(base: dict | None, fin: dict | None) -> dict | None:
    if not base or not fin:
        return None
    out = {}
    for k in ("router_recall", "oracle_recall", "best_single_recall", "routing_headroom"):
        bv, fv = base.get(k), fin.get(k)
        d = round(fv - bv, 3) if isinstance(bv, (int, float)) and isinstance(fv, (int, float)) else None
        out[k] = {"baseline": bv, "final": fv, "delta": d}
    return out


# ── Lexicon service (publish QA→RAG, push RAG→Chat) ──────────────────────────
def _lex_token() -> str | None:
    """Mint a dev platform token for the lexicon service (dev only)."""
    chat = os.getenv("CHAT_BASE_URL")
    if not chat:
        return None
    try:
        r = _http("POST", f"{chat}/chat/admin/mint-dev-token", body={}, timeout=30)
        return r.get("access_token")
    except Exception as exc:
        logger.warning("[nightly] mint token failed: %s", exc)
        return None


def _lex_call(path: str, body: dict) -> dict | None:
    from app.config import LEXICON_MAINTENANCE_URL
    if not LEXICON_MAINTENANCE_URL:
        return None
    tok = _lex_token()
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    return _http("POST", LEXICON_MAINTENANCE_URL.rstrip("/") + path, body=body,
                 headers=headers, timeout=120)


# ── Infra (delegated; graceful no-op if not configured) ──────────────────────
def _infra(direction: str) -> str:
    try:
        from app import nightly_infra
    except Exception:
        return "skipped: infra control not available"
    try:
        return nightly_infra.scale(direction, stopping=_stopping)
    except Exception as exc:
        logger.warning("[nightly] infra %s failed: %s", direction, exc)
        return f"error: {exc}"


# ── Small polling helper for the RAG background sub-jobs ──────────────────────
def _poll_job(status_path: str, budget_min: int) -> dict:
    t, last = 0, {}
    while t < budget_min * 60 and not _stopping():
        st = _rag_get(status_path, timeout=30)
        last = st
        if st.get("running") in (False, None):
            break
        _step(_NIGHTLY["current"], "running", f"{st.get('done')}/{st.get('total')}")
        time.sleep(15)
        t += 15
    return last


# ── The orchestration thread ─────────────────────────────────────────────────
def _run_nightly(opts: dict) -> None:
    include_eval = opts.get("include_eval", True)
    dry_run = opts.get("dry_run", False)
    skip_infra = opts.get("skip_infra", False)
    embed_budget = int(opts.get("embed_budget_min", 60))
    quiesce_budget = int(opts.get("quiesce_budget_min", 20))
    bank = opts.get("eval_bank", "eval/queries_cmhc.yaml")
    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")

    try:
        # 0 — infra up
        _step("infra_up", "running")
        if skip_infra:
            _step("infra_up", "skipped", "skip_infra")
        else:
            _step("infra_up", "done", _infra("up"))
        if _stopping():
            raise RuntimeError("stopped")

        # A — baseline eval
        if include_eval and not dry_run:
            _step("baseline_eval", "running")
            _NIGHTLY["baseline"] = _run_eval(f"nightly-baseline-{date_tag}", bank)
            _step("baseline_eval", "done" if _NIGHTLY["baseline"] else "failed",
                  f"router {_NIGHTLY['baseline'].get('router_recall')}" if _NIGHTLY["baseline"] else "no summary")
        else:
            _step("baseline_eval", "skipped", "eval off" if not include_eval else "dry run")

        # 1 — publish lexicon QA→RAG (tag sanity gate lives in the lexicon svc dry-run)
        _step("publish", "running")
        if dry_run:
            _step("publish", "skipped", "dry run")
        else:
            dry = _lex_call("/policy/lexicon/publish", {"dry_run": True})
            if dry is None:
                _step("publish", "skipped", "lexicon svc not configured")
            else:
                qa = dry.get("qa_revision"); rag = dry.get("rag_revision_before")
                if qa is not None and rag is not None and int(qa) <= int(rag):
                    _step("publish", "skipped", f"already current (rev {rag})")
                else:
                    _lex_call("/policy/lexicon/publish", {"dry_run": False})
                    _step("publish", "done", f"rev {rag}→{qa}")
        if _stopping():
            raise RuntimeError("stopped")

        # 2 — retag in place
        _step("retag", "running")
        if dry_run:
            _step("retag", "skipped", "dry run")
        else:
            _rag_post("/admin/retag-in-place", {"only_stale": True}, timeout=30)
            st = _poll_job("/admin/retag-in-place/status", 90)
            _step("retag", "done", f"{st.get('done')}/{st.get('total')} · {st.get('errors',0)} err")
        if _stopping():
            raise RuntimeError("stopped")

        # 3 — chunk new docs (build-eu + remediate; deterministic Path-B)
        _step("chunk", "running")
        if dry_run:
            _step("chunk", "skipped", "dry run")
        else:
            _rag_post("/admin/build-eu-from-lines", {}, timeout=30)
            _poll_job("/admin/build-eu-from-lines/status", 60)
            _rag_post("/admin/integrity/remediate", {}, timeout=30)
            _poll_job("/admin/integrity/remediate/status", 15)
            _step("chunk", "done", "build-eu + remediate")
        if _stopping():
            raise RuntimeError("stopped")

        # 4 — embed → publish (time-budgeted drain; reset zombie processing first)
        _step("embed", "running")
        if dry_run:
            _step("embed", "skipped", "dry run")
        else:
            _rag_post("/admin/db/execute", {"sql":
                "UPDATE embedding_jobs SET status='pending', started_at=NULL "
                "WHERE status='processing' AND now()-started_at > interval '10 min'"}, timeout=60)
            t = 0
            while t < embed_budget * 60 and not _stopping():
                pend = (((_rag_post("/admin/db/execute",
                        {"sql": "SELECT count(*) AS n FROM embedding_jobs WHERE status='pending'"},
                        timeout=60).get("records") or [{}])[0]).get("n"))
                pub = _rag_get("/admin/integrity/report", timeout=60).get("published")
                _step("embed", "running", f"pending {pend} · published {pub}")
                if str(pend) == "0":
                    break
                time.sleep(60); t += 60
            _rag_post("/admin/publish_unpublished?limit=500", {}, timeout=120)
            _step("embed", "done", f"published {pub}")
        if _stopping():
            raise RuntimeError("stopped")

        # 5 — integrity gate
        _step("gate", "running")
        rep = _rag_get("/admin/integrity/report", timeout=60)
        gaps = rep.get("gaps", {})
        pub = int(rep.get("published") or 0)
        tot = int(rep.get("documents_total") or 1)
        reing = int(gaps.get("need_reingest") or 0)
        stale = int(gaps.get("stale_tags") or 0)
        frac = pub / max(tot - reing, 1)
        passed = (frac >= 0.97 and stale == 0)
        _NIGHTLY["gate"] = {"passed": passed, "published": pub, "documents_total": tot,
                            "frac": round(frac, 4), "stale_tags": stale}
        _step("gate", "done", f"{'PASS' if passed else 'FAIL'} frac={frac:.3f} stale={stale}")

        # FREEZE — drain (workers up) then idle, before the final eval
        _step("freeze", "running")
        if include_eval and not dry_run:
            _rag_post("/admin/db/execute", {"sql":
                "UPDATE embedding_jobs SET status='pending', started_at=NULL "
                "WHERE status='processing' AND now()-started_at > interval '10 min'"}, timeout=60)
            qt = 0
            while qt < quiesce_budget * 60 and not _stopping():
                q = (((_rag_post("/admin/db/execute", {"sql":
                    "SELECT (SELECT count(*) FROM embedding_jobs WHERE status='pending') "
                    "+ (SELECT count(*) FROM chunking_jobs WHERE status='pending') AS n"},
                    timeout=60).get("records") or [{}])[0]).get("n"))
                _step("freeze", "running", f"pending backlog {q}")
                if str(q) == "0":
                    break
                time.sleep(30); qt += 30
            if not skip_infra:
                _infra("freeze")   # idle workers so nothing writes during the eval
            _step("freeze", "done", "corpus frozen")
        else:
            _step("freeze", "skipped", "eval off" if not include_eval else "dry run")

        # B — final eval
        if include_eval and not dry_run:
            _step("final_eval", "running")
            _NIGHTLY["final"] = _run_eval(f"nightly-final-{date_tag}", bank)
            _step("final_eval", "done" if _NIGHTLY["final"] else "failed",
                  f"router {_NIGHTLY['final'].get('router_recall')}" if _NIGHTLY["final"] else "no summary")
        else:
            _step("final_eval", "skipped", "eval off" if not include_eval else "dry run")

        # 6 — push lexicon+tags RAG→Chat (only if gate passed)
        _step("push", "running")
        if dry_run:
            _step("push", "skipped", "dry run")
        elif not passed:
            _step("push", "skipped", "gate failed — not pushing partial corpus")
        else:
            res = _lex_call("/policy/lexicon/push-to-chat", {"dry_run": False})
            _NIGHTLY["push_done"] = res is not None
            _step("push", "done" if res is not None else "skipped",
                  "pushed" if res is not None else "lexicon svc not configured")

        # 0' — revert infra
        _step("infra_down", "running")
        if skip_infra:
            _step("infra_down", "skipped", "skip_infra")
        else:
            _step("infra_down", "done", _infra("down"))

        # L — lift
        _step("lift", "running")
        _NIGHTLY["lift"] = _compute_lift(_NIGHTLY.get("baseline"), _NIGHTLY.get("final"))
        if _NIGHTLY["lift"]:
            dr = _NIGHTLY["lift"].get("router_recall", {}).get("delta")
            _step("lift", "done", f"Δrouter {dr:+.3f}" if isinstance(dr, (int, float)) else "computed")
        else:
            _step("lift", "skipped", "no bracket to diff")

    except Exception as exc:
        _NIGHTLY["error"] = str(exc)
        cur = _NIGHTLY.get("current")
        if cur:
            _step(cur, "failed", str(exc))
        logger.exception("[nightly] run failed: %s", exc)
    finally:
        _NIGHTLY["running"] = False
        _NIGHTLY["finished_at"] = _now()
        _NIGHTLY["current"] = None


# ── Public API (called by the thin main.py routes) ───────────────────────────
def start(opts: dict) -> dict:
    if _NIGHTLY.get("running"):
        return {"status": "already_running", **status()}
    _NIGHTLY.update({
        "running": True, "stop": False, "started_at": _now(), "finished_at": None,
        "current": None, "error": None, "opts": opts,
        "eval_run_id": None, "baseline": None, "final": None, "lift": None,
        "gate": None, "push_done": False,
    })
    _init_steps()
    threading.Thread(target=_run_nightly, args=(opts,), daemon=True).start()
    return status()


def stop() -> dict:
    _NIGHTLY["stop"] = True
    return {"status": "stopping", **status()}


def status() -> dict:
    return {k: _NIGHTLY[k] for k in (
        "running", "stop", "started_at", "finished_at", "current", "error",
        "opts", "steps", "eval_run_id", "baseline", "final", "lift", "gate", "push_done")}
