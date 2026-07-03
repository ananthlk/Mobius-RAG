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
    "run_id": None,
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
    # Persist live state continuously so the tracker survives an instance recycle
    # (in-memory _NIGHTLY is lost on restart; the DB copy is what get_run falls
    # back to). Throttled so tight-ish poll loops don't hammer the DB.
    _persist_throttled()


import time as _time
_LAST_PERSIST = [0.0]


def _persist_throttled(min_gap_s: float = 4.0) -> None:
    now = _time.monotonic()
    if now - _LAST_PERSIST[0] >= min_gap_s:
        _LAST_PERSIST[0] = now
        _persist()


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


def _try(fn, default=None):
    """Run fn(); on any exception, log it and return `default` instead of
    propagating. Used to keep a transient HTTP/DB blip or a slow best-effort
    call (e.g. the giant-doc publish sweep) from aborting the whole run."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        try:
            _log(_NIGHTLY.get("current", ""), f"non-fatal: {str(exc)[:140]}")
        except Exception:
            pass
        return default


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


def _admin_headers() -> dict:
    """Self-calls to /admin/* pass through the admin-auth middleware, so the
    orchestrator authenticates to its own service with ADMIN_API_KEY."""
    try:
        from app.config import ADMIN_API_KEY
        return {"X-Admin-Key": ADMIN_API_KEY} if ADMIN_API_KEY else {}
    except Exception:
        return {}


def _rag_get(path: str, timeout: int = 120) -> dict:
    return _http("GET", _base() + path, headers=_admin_headers(), timeout=timeout)


def _rag_post(path: str, body: dict | None = None, timeout: int = 120) -> dict:
    return _http("POST", _base() + path, body=body if body is not None else {},
                 headers=_admin_headers(), timeout=timeout)


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
    """Mint a platform token for the lexicon service. Derives the chat base from
    CHAT_BASE_URL or (dev) CHAT_INTERNAL_LLM_URL — same source as /dev/mint-token."""
    from app.config import CHAT_INTERNAL_LLM_URL
    base = os.getenv("CHAT_BASE_URL") or (CHAT_INTERNAL_LLM_URL or "").split("/internal")[0]
    if not base:
        return None
    try:
        r = _http("POST", base.rstrip("/") + "/chat/admin/mint-dev-token", body={}, timeout=30)
        return r.get("access_token")
    except Exception as exc:
        logger.warning("[nightly] mint token failed: %s", exc)
        return None


def _lex_call(path: str, body: dict) -> dict | None:
    """Call the lexicon service. Returns the JSON, None if not configured, or
    {"_error": ...} on failure — NEVER raises, so a LEX hiccup can't abort a run
    that already did expensive work (e.g. the baseline eval)."""
    from app.config import LEXICON_MAINTENANCE_URL
    if not LEXICON_MAINTENANCE_URL:
        return None
    tok = _lex_token()
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    try:
        return _http("POST", LEXICON_MAINTENANCE_URL.rstrip("/") + path, body=body,
                     headers=headers, timeout=120)
    except Exception as exc:
        return {"_error": str(exc)}


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

        # A — baseline eval (resume: reuse the prior result, don't re-run ~32 min)
        if _done("baseline_eval"):
            _log("baseline_eval", "resumed — reused prior result")
        elif include_eval and not dry_run:
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
            elif dry.get("_error"):
                # LEX unreachable/401 — don't abort; corpus steps proceed on the
                # prior revision (spec §7: Step-1 failure may allow Steps 2-7).
                _step("publish", "failed", f"lexicon: {str(dry['_error'])[:80]} (continuing on prior rev)")
            else:
                qa = dry.get("qa_revision"); rag = dry.get("rag_revision_before")
                if qa is not None and rag is not None and int(qa) <= int(rag):
                    _step("publish", "skipped", f"already current (rev {rag})")
                else:
                    res = _lex_call("/policy/lexicon/publish", {"dry_run": False})
                    if res and res.get("_error"):
                        _step("publish", "failed", f"lexicon: {str(res['_error'])[:80]} (continuing on prior rev)")
                    else:
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
            _try(lambda: _rag_post("/admin/db/execute", {"sql":
                "UPDATE embedding_jobs SET status='pending', started_at=NULL "
                "WHERE status='processing' AND now()-started_at > interval '10 min'"}, timeout=60))
            t = 0
            pub = None
            while t < embed_budget * 60 and not _stopping():
                # Transient DB/report blips must not abort the whole run — a
                # single failed poll just retries next tick.
                rec = _try(lambda: _rag_post("/admin/db/execute",
                        {"sql": "SELECT count(*) AS n FROM embedding_jobs WHERE status='pending'"},
                        timeout=60)) or {}
                pend = ((rec.get("records") or [{}])[0]).get("n")
                pub = (_try(lambda: _rag_get("/admin/integrity/report", timeout=60)) or {}).get("published", pub)
                _step("embed", "running", f"pending {pend} · published {pub}")
                if str(pend) == "0":
                    break
                time.sleep(60); t += 60
            # Best-effort backstop publish for NON-giant docs. Giants publish via
            # the worker's background auto-publish (they can take >60s each, which
            # would blow this request's timeout) — so a failure/timeout here is
            # NON-FATAL and must never abort the run.
            _try(lambda: _rag_post("/admin/publish_unpublished?limit=500", {}, timeout=120))
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
            if res is None:
                _step("push", "skipped", "lexicon svc not configured")
            elif res.get("_error"):
                _step("push", "failed", f"lexicon: {str(res['_error'])[:80]}")
            else:
                _NIGHTLY["push_done"] = True
                _step("push", "done", "pushed")

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
        _persist()   # durable snapshot for run history


# ── Public API (called by the thin main.py routes) ───────────────────────────
def start(opts: dict) -> dict:
    import uuid
    if _NIGHTLY.get("running"):
        return {"status": "already_running", **status()}
    # Resume: carry over a prior run's COMPLETED steps + baseline so we don't
    # re-run expensive work (esp. the ~32-min baseline eval). Steps that weren't
    # done are reset to pending and re-executed from where the run failed.
    prior_baseline = None
    prior_steps_by_key: dict = {}
    resume_from = opts.get("resume_from")
    if resume_from:
        prior = get_run(resume_from)
        if isinstance(prior, dict) and prior.get("steps"):
            prior_baseline = prior.get("baseline")
            prior_steps_by_key = {s["key"]: s for s in prior["steps"]}
    _NIGHTLY.update({
        "run_id": str(uuid.uuid4()),
        "running": True, "stop": False, "started_at": _now(), "finished_at": None,
        "current": None, "error": None, "opts": opts,
        "eval_run_id": None, "baseline": prior_baseline, "final": None, "lift": None,
        "gate": None, "push_done": False,
    })
    if prior_steps_by_key:
        _NIGHTLY["steps"] = []
        for k, lbl in _STEP_DEFS:
            p = prior_steps_by_key.get(k)
            if p and p.get("status") in ("done", "skipped"):
                _NIGHTLY["steps"].append({**p, "label": lbl})   # keep completed
            else:
                _NIGHTLY["steps"].append({"key": k, "label": lbl, "status": "pending",
                                          "detail": "", "started_at": None, "ended_at": None, "log": []})
    else:
        _init_steps()
    _persist()   # record the run as 'running' so it appears in history immediately
    threading.Thread(target=_run_nightly, args=(opts,), daemon=True).start()
    return status()


def _done(key: str) -> bool:
    """True if a step is already complete (used to skip it on a resumed run)."""
    return any(s["key"] == key and s["status"] in ("done", "skipped") for s in _NIGHTLY["steps"])


def stop() -> dict:
    _NIGHTLY["stop"] = True
    return {"status": "stopping", **status()}


def status() -> dict:
    return {k: _NIGHTLY[k] for k in (
        "run_id", "running", "stop", "started_at", "finished_at", "current", "error",
        "opts", "steps", "eval_run_id", "baseline", "final", "lift", "gate", "push_done")}


# ── Run history (durable across the frequent single-instance restarts) ────────
def _dsn() -> str:
    from app.config import DATABASE_URL
    return (DATABASE_URL
            .replace("postgresql+asyncpg://", "postgresql://")
            .replace("postgresql+psycopg2://", "postgresql://"))


def _ensure_table() -> None:
    import psycopg2
    conn = psycopg2.connect(_dsn(), connect_timeout=15)
    try:
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS nightly_runs ("
            "id text PRIMARY KEY, started_at timestamptz, finished_at timestamptz, "
            "status text, dry_run boolean, include_eval boolean, "
            "router_delta double precision, state jsonb, created_at timestamptz DEFAULT now())")
        conn.commit()
    finally:
        conn.close()


def _persist() -> None:
    """Snapshot the current run into nightly_runs (upsert by run_id)."""
    import json as _j
    import psycopg2
    rid = _NIGHTLY.get("run_id")
    if not rid:
        return
    try:
        _ensure_table()
        run_status = ("running" if _NIGHTLY.get("running")
                      else "failed" if _NIGHTLY.get("error")
                      else "stopped" if _NIGHTLY.get("stop") else "done")
        rd = ((_NIGHTLY.get("lift") or {}).get("router_recall") or {}).get("delta")
        conn = psycopg2.connect(_dsn(), connect_timeout=15)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO nightly_runs (id, started_at, finished_at, status, dry_run, "
            "include_eval, router_delta, state) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT (id) DO UPDATE SET finished_at=EXCLUDED.finished_at, "
            "status=EXCLUDED.status, router_delta=EXCLUDED.router_delta, state=EXCLUDED.state",
            (rid, _NIGHTLY.get("started_at"), _NIGHTLY.get("finished_at"), run_status,
             bool(_NIGHTLY["opts"].get("dry_run")), bool(_NIGHTLY["opts"].get("include_eval")),
             rd, _j.dumps(status())))
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("[nightly] persist failed: %s", exc)


def list_runs(limit: int = 30) -> dict:
    import psycopg2
    try:
        _ensure_table()
        conn = psycopg2.connect(_dsn(), connect_timeout=15)
        cur = conn.cursor()
        cur.execute(
            "SELECT id, started_at, finished_at, status, dry_run, include_eval, router_delta "
            "FROM nightly_runs ORDER BY started_at DESC NULLS LAST LIMIT %s", (limit,))
        live = _NIGHTLY.get("run_id") if _NIGHTLY.get("running") else None
        runs = []
        for r in cur.fetchall():
            # A 'running' row that isn't the live in-memory run is stale (killed) —
            # report it as 'failed' so the history never shows a dead run as running.
            st = r[3]
            if st == "running" and r[0] != live:
                st = "failed"
            runs.append({"id": r[0],
                         "started_at": r[1].isoformat() if r[1] else None,
                         "finished_at": r[2].isoformat() if r[2] else None,
                         "status": st, "dry_run": r[4], "include_eval": r[5], "router_delta": r[6]})
        conn.close()
        return {"runs": runs}
    except Exception as exc:
        return {"runs": [], "error": str(exc)}


def get_run(run_id: str) -> dict:
    # The live current run is freshest in memory; past runs come from the table.
    if _NIGHTLY.get("run_id") == run_id and _NIGHTLY.get("running"):
        return status()
    import psycopg2
    try:
        conn = psycopg2.connect(_dsn(), connect_timeout=15)
        cur = conn.cursor()
        cur.execute("SELECT state FROM nightly_runs WHERE id=%s", (run_id,))
        r = cur.fetchone()
        conn.close()
        if not r:
            return {"error": "not found"}
        state = r[0] if isinstance(r[0], dict) else {}
        # A run that isn't the live in-memory one CANNOT be running (single
        # instance, one run at a time). Killed runs persist a stale running=true;
        # force it false so the UI never shows a dead run as live.
        state["running"] = False
        return state
    except Exception as exc:
        return {"error": str(exc)}
