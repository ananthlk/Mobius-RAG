"""Auto-infra for the nightly orchestrator (#17).

`infra_up` at run start resizes the shared Cloud SQL instance up (the critical
anti-thrash lever — embed on 2 vCPU starves the DB) and scales the workers up;
`infra_down` reverts. Called by nightly_orchestrator._infra().

Uses the RAG runtime service account via `google.auth` to call the Cloud SQL
Admin + Cloud Run Admin v2 REST APIs directly — NO new deps. IAM required on the
runtime SA (mobius-platform-dev):
  * roles/cloudsql.admin      (patch instance tier)   ← required
  * roles/run.admin           (scale worker services) ← optional; worker scaling
                                is best-effort, so a missing grant just leaves
                                workers at their deploy floor.

Everything is best-effort and never raises to the caller (the orchestrator's
_infra wraps it and treats failures as non-fatal — a run proceeds on whatever
infra is currently up).
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request

from app.config import (
    VERTEX_PROJECT_ID as _PROJ_DEFAULT,
    VERTEX_LOCATION as _REGION_DEFAULT,
)
import os

logger = logging.getLogger(__name__)

PROJECT = os.getenv("GCP_PROJECT") or _PROJ_DEFAULT or "mobius-os-dev"
REGION = os.getenv("GCP_REGION") or _REGION_DEFAULT or "us-central1"
SQL_INSTANCE = os.getenv("NIGHTLY_SQL_INSTANCE", "mobius-platform-dev-db")
DB_TIER_BIG = os.getenv("NIGHTLY_DB_TIER_BIG", "db-custom-8-32768")
DB_TIER_SMALL = os.getenv("NIGHTLY_DB_TIER_SMALL", "db-custom-2-7680")
# These workers are SELF-POLLING supervisors (each instance runs one poll loop
# claiming jobs via FOR UPDATE SKIP LOCKED). Cloud Run autoscaling never fires
# from HTTP load, so parallelism == instance count and we must pin min==max==N.
# We manage the EMBEDDING worker only (1 poller by deploy default → the giant-doc
# bottleneck); the chunking worker ships at min=max=12 and we leave it there
# (reducing it would regress the instant-rag queue SLA). Embed is capped at 4 to
# stay under the Vertex embedding-API 429 ceiling seen at 6.
WORKER_SCALE = {
    "mobius-rag-embedding-worker": int(os.getenv("NIGHTLY_EMBED_WORKERS", "4")),
}
WORKER_FLOOR = int(os.getenv("NIGHTLY_WORKER_FLOOR", "1"))   # min after revert (1 keeps the queue draining)


def _token() -> str:
    import google.auth
    from google.auth.transport.requests import Request
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(Request())
    return creds.token


def _api(method: str, url: str, token: str, body: dict | None = None, timeout: int = 60) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode()
    return json.loads(raw) if raw else {}


def _sql_set_tier(tier: str, token: str, stopping=None) -> str:
    """Patch the instance tier and poll the operation to DONE (resize restarts
    the instance ~1-2 min; the caller must not do DB work until it's back)."""
    base = f"https://sqladmin.googleapis.com/sql/v1beta4/projects/{PROJECT}/instances/{SQL_INSTANCE}"
    op = _api("PATCH", base, token, body={"settings": {"tier": tier}})
    opname = op.get("name")
    if not opname:
        return f"DB patch→{tier} (no op)"
    opurl = f"https://sqladmin.googleapis.com/sql/v1beta4/projects/{PROJECT}/operations/{opname}"
    for _ in range(72):   # ~6 min cap
        if stopping and stopping():
            break
        try:
            st = _api("GET", opurl, token, timeout=30)
        except Exception:
            time.sleep(5); continue
        if st.get("status") == "DONE":
            return f"DB→{tier} done"
        time.sleep(5)
    return f"DB→{tier} (still applying)"


def _scale_worker(svc: str, minc: int, maxc: int, token: str) -> None:
    """Pin a worker to min==max==N instances (N parallel pollers) via Cloud Run
    Admin v2. The scaling knob lives on the REVISION TEMPLATE, not the service
    root — updateMask must be ``template.scaling.*`` and the body must nest under
    ``template``. (Patching top-level ``scaling`` silently no-ops for
    request-scaled services — that was the original bug that left embed at 1.)
    The patch mints a new revision with only scaling changed."""
    url = (f"https://run.googleapis.com/v2/projects/{PROJECT}/locations/{REGION}"
           f"/services/{svc}"
           "?updateMask=template.scaling.minInstanceCount,template.scaling.maxInstanceCount")
    _api("PATCH", url, token,
         body={"template": {"scaling": {"minInstanceCount": minc, "maxInstanceCount": maxc}}})


def _scale_workers(scale: dict, token: str) -> str:
    out = []
    for svc, n in scale.items():
        try:
            _scale_worker(svc, n if isinstance(n, int) else n[0],
                          n if isinstance(n, int) else n[1], token)
            out.append(f"{svc.split('-')[-1]}={n}")
        except Exception as exc:
            out.append(f"{svc.split('-')[-1]}=err({str(exc)[:30]})")
    return "workers " + ", ".join(out)


def scale(direction: str, stopping=None) -> str:
    """direction: 'up' | 'freeze' | 'down'. Returns a human summary. Best-effort."""
    token = _token()
    if direction == "up":
        db = _sql_set_tier(DB_TIER_BIG, token, stopping)
        wk = _scale_workers(WORKER_SCALE, token)
        return f"{db}; {wk}"
    if direction == "freeze":
        # idle only the workers (stop writes before the final eval); leave DB big
        return _scale_workers({s: WORKER_FLOOR for s in WORKER_SCALE}, token)
    if direction == "down":
        wk = _scale_workers({s: WORKER_FLOOR for s in WORKER_SCALE}, token)
        db = _sql_set_tier(DB_TIER_SMALL, token, stopping)
        return f"{db}; {wk}"
    return f"unknown direction: {direction}"
