#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Nightly corpus + lexicon pipeline — ON-DEMAND driver.
#
# Strings the whole loop (spec: docs/nightly-pipeline-spec.md), bracketed by the
# 5-axis calibration eval (runbook: eval/calibration/NIGHTLY_EVAL_RUNBOOK.md):
#
#   0.  upsize the shared Cloud SQL instance + scale workers up
#   A.  BASELINE eval  (before any corpus write — corpus frozen)
#   1.  publish lexicon QA→RAG   (behind an automated tag sanity gate)
#   2.  retag existing docs in place (AC matcher, fast)
#   3.  chunk+tag new docs        (build-eu-from-lines + Path-B remediate; no LLM)
#   4.  embed → publish           (time-budgeted drain)
#   5.  straggler cleanup + integrity report + gate
#   --- FREEZE: idle workers, confirm queues drained (no more document_tags writes)
#   B.  FINAL eval    (after writes settle — corpus frozen)
#   6.  push lexicon+tags RAG→Chat  (only if the integrity gate passed)
#   0'. downsize DB (AFTER the final eval — resize restarts the instance)
#   L.  LIFT report   (Δ router_recall / Δ oracle_recall / Δ headroom / contra rise)
#
# WHY the bracket is shaped this way (hard constraints from the runbook):
#   • The eval hits RAG (corpus_search_agent) → it measures *corpus* lift.
#   • Corpus MUST be stable during a run: no retag / document_tags writes, and no
#     embed/publish (which changes rag_published_embeddings the eval retrieves).
#     → evals and corpus-writes are strictly serialized; workers idled before FINAL.
#   • DB resize restarts the shared instance → never overlap an eval → both evals
#     run on the (already-stable) big tier; downsize happens AFTER the final eval.
#   • The judge model is locked (gemini-2.5-pro) and must not change mid-bracket.
#
# Idempotent: safe to re-run; every step keys on current state, not "ran tonight".
#
# Usage:
#   ./run_nightly_pipeline.sh                 # full run (upsize→work→eval→downsize)
#   DRY_RUN=1 ./run_nightly_pipeline.sh       # print what it would do, no mutations
#   SKIP_RESIZE=1 ./run_nightly_pipeline.sh   # keep DB tier as-is (gentle; no restart)
#   RUN_EVAL=0 ./run_nightly_pipeline.sh      # skip the eval bracket
#   EMBED_BUDGET_MIN=90 ./run_nightly_pipeline.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
PROJECT="${PROJECT:-mobius-os-dev}"
REGION="${REGION:-us-central1}"
DB_INSTANCE="${DB_INSTANCE:-mobius-platform-dev-db}"
DB_TIER_BIG="${DB_TIER_BIG:-db-custom-8-32768}"
DB_TIER_SMALL="${DB_TIER_SMALL:-db-custom-2-7680}"   # revert target

CHAT="${CHAT:-https://mobius-chat-ortabkknqa-uc.a.run.app}"
RAG="${RAG:-https://mobius-rag-ortabkknqa-uc.a.run.app}"
LEX="${LEX:-https://mobius-lexicon-maintenance-ortabkknqa-uc.a.run.app}"

CHUNK_WORKERS="${CHUNK_WORKERS:-4}"
EMBED_WORKERS="${EMBED_WORKERS:-6}"
EMBED_BUDGET_MIN="${EMBED_BUDGET_MIN:-120}"          # max minutes to wait for embed drain
POLL_S="${POLL_S:-20}"

# Tag sanity gate thresholds (Step 1)
GATE_MIN_RATIO="${GATE_MIN_RATIO:-0.8}"              # qa.entries >= ratio * rag.entries
GATE_MAX_RATIO="${GATE_MAX_RATIO:-2.0}"              # qa.entries <= ratio * rag.entries

# Eval bracket
RUN_EVAL="${RUN_EVAL:-1}"                            # 0 = skip both evals
# The 5-axis calibration_summary (recall/precision/contradiction) only populates
# for a bank whose queries carry `must_facts`. The default eval/queries.yaml is the
# legacy keyword bank (answer_keywords, NO must_facts) → summary comes back all
# zeros. Point at the CMHC golden-facts bank. Smoke: eval/queries_cmhc_smoke.yaml.
EVAL_BANK="${EVAL_BANK:-eval/queries_cmhc.yaml}"
QUIESCE_BUDGET_MIN="${QUIESCE_BUDGET_MIN:-20}"       # wait for queues to drain before FINAL eval

DRY_RUN="${DRY_RUN:-0}"
SKIP_RESIZE="${SKIP_RESIZE:-0}"

DATE_TAG="$(date -u +%Y%m%d-%H%M)"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
die() { echo "[$(date -u +%H:%M:%S)] FATAL: $*" >&2; exit 1; }
run() { if [[ "$DRY_RUN" == "1" ]]; then echo "  DRYRUN> $*"; else eval "$@"; fi; }

# ── Auth: mint a platform token (dev). In prod, swap for a real service token. ──
mint_token() {
  curl -s -X POST -H 'Content-Type: application/json' -d '{}' \
    "$CHAT/chat/admin/mint-dev-token" \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('access_token',''))" 2>/dev/null
}
TOK="$(mint_token)"; [[ -n "$TOK" ]] || die "could not mint token"

# ── Small HTTP helpers ───────────────────────────────────────────────────────
# GET/POST returning raw body. The dev token expires (~1h) but a full bracket
# (baseline eval + corpus + final eval) runs LONGER than that — so re-mint TOK
# transparently on any auth failure and retry once. Without this, every api()
# call past the 1h mark 401s and reads back empty (silently breaking the gate,
# push, and drain-queue reads mid-run).
api() { local m="$1" url="$2" data="${3:-}"; local a=(-s --max-time 120 -X "$m");
        [[ -n "$data" ]] && a+=(-H 'Content-Type: application/json' -d "$data")
        local body; body="$(curl "${a[@]}" -H "Authorization: Bearer $TOK" "$url")"
        if [[ "$body" == *'"unauthorized"'* || "$body" == *'invalid or missing'* ]]; then
          TOK="$(mint_token)"
          body="$(curl "${a[@]}" -H "Authorization: Bearer $TOK" "$url")"
        fi
        printf '%s' "$body"; }
# extract a json field from a body (stdin)
jget() { python3 -c "import sys,json; d=json.load(sys.stdin); print(d$1)" 2>/dev/null; }

# poll a status endpoint until running==false (or timeout minutes)
poll_until_done() {  # $1=status_url  $2=timeout_min
  local url="$1" tmax="$2" t=0
  while (( t < tmax*60 )); do
    local body; body="$(api GET "$url")"
    local running; running="$(echo "$body" | jget "['running']")"
    local done tot; done="$(echo "$body" | jget "['done']")"; tot="$(echo "$body" | jget "['total']")"
    log "    …$url done=$done/$tot running=$running"
    [[ "$running" == "False" || "$running" == "false" ]] && return 0
    sleep "$POLL_S"; t=$((t+POLL_S))
  done
  return 1
}

# ── Eval helpers (5-axis calibration bracket; endpoints are UNAUTH on dev) ─────
# Runs ONE calibration end-to-end. Echoes the calibration_summary JSON to stdout;
# all human logs go to stderr so stdout stays clean JSON for capture.
eval_run() {  # $1=notes  → stdout: summary JSON | rc 1=busy/failed  2=stalled
  local notes="$1"
  # precondition: nothing already running (server lock returns 409 otherwise)
  local active; active="$(curl -s --max-time 30 "$RAG/api/eval/active" | jget "['active']")"
  if [[ "$active" == "True" || "$active" == "true" ]]; then
    log "  eval[$notes]: another calibration is active — aborting bracket" >&2; return 1
  fi
  # trigger (returns immediately, no run_id) — bank MUST carry must_facts (see EVAL_BANK)
  curl -s --max-time 30 -X POST "$RAG/api/eval/calibrate/trigger" \
    -H 'Content-Type: application/json' \
    -d "{\"notes\":\"$notes\",\"bank_path\":\"$EVAL_BANK\"}" >/dev/null
  sleep 5
  # resolve the run_id (newest in-flight run)
  local rid="" i
  for i in 1 2 3 4 5 6; do
    rid="$(curl -s --max-time 30 "$RAG/api/eval/active" | jget "['run_id']")"
    [[ -n "$rid" && "$rid" != "None" ]] && break
    sleep 5
  done
  [[ -n "$rid" && "$rid" != "None" ]] || { log "  eval[$notes]: could not resolve run_id" >&2; return 1; }
  log "  eval[$notes]: run_id=$rid — polling to completion…" >&2
  # poll progress with a >5min stall guard
  local last=-1 stall=0
  while :; do
    local pr; pr="$(curl -s --max-time 30 "$RAG/api/eval/runs/$rid/progress")"
    local running; running="$(echo "$pr" | jget "['is_running']")"
    local nc; nc="$(echo "$pr" | jget "['n_completed']")"
    local nq; nq="$(echo "$pr" | jget "['n_queries']")"
    log "    eval[$notes]: ${nc:-?}/${nq:-?} running=${running:-?}" >&2
    [[ "$running" == "False" || "$running" == "false" ]] && break
    if [[ "$nc" == "$last" ]]; then stall=$((stall+1)); else stall=0; last="$nc"; fi
    if (( stall >= 10 )); then   # 10 × 30s = 5 min with no progress
      log "  eval[$notes]: STALLED >5min at $nc — use durable driver (runbook §fallback)" >&2
      return 2
    fi
    sleep 30
  done
  curl -s --max-time 30 "$RAG/api/eval/runs/$rid/calibration_summary"
}

# Diff two calibration_summary JSON files → human lift report (to stdout)
eval_lift() {  # $1=baseline_json  $2=final_json
  python3 - "$1" "$2" <<'PY'
import sys, json
def load(p):
    try:
        with open(p) as fh: return json.load(fh)
    except Exception: return {}
b, f = load(sys.argv[1]), load(sys.argv[2])
if not b or not f:
    print("  lift: missing baseline or final summary — cannot diff"); raise SystemExit
def num(d,k):
    v=d.get(k); return v if isinstance(v,(int,float)) else None
print(f"  {'metric':22} {'baseline':>9} {'final':>9} {'Δ':>9}")
for k in ("router_recall","oracle_recall","best_single_recall","routing_headroom"):
    bv,fv=num(b,k),num(f,k)
    if bv is None or fv is None:
        print(f"  {k:22} {str(bv):>9} {str(fv):>9} {'n/a':>9}")
    else:
        print(f"  {k:22} {bv:9.3f} {fv:9.3f} {fv-bv:+9.3f}")
bs=b.get("strategies",{}) or {}; fs=f.get("strategies",{}) or {}
print("  contra_per_cell (baseline → final):")
for s in ("a","b","c","d","natural"):
    bc=(bs.get(s,{}) or {}).get("contra_per_cell"); fc=(fs.get(s,{}) or {}).get("contra_per_cell")
    if bc is None and fc is None: continue
    print(f"    {s:8} {str(bc):>6} → {str(fc):>6}")
PY
}

BASELINE_JSON=""; FINAL_JSON=""
if [[ "$RUN_EVAL" == "1" ]]; then BASELINE_JSON="$(mktemp)"; FINAL_JSON="$(mktemp)"; fi

# ══ STEP 0 — upsize DB + scale workers ═══════════════════════════════════════
log "STEP 0  infra: DB tier + worker scale"
CUR_TIER="$(gcloud sql instances describe "$DB_INSTANCE" --project="$PROJECT" --format='value(settings.tier)' 2>/dev/null)"
log "  current DB tier: ${CUR_TIER:-unknown}"
if [[ "$SKIP_RESIZE" != "1" && "$CUR_TIER" != "$DB_TIER_BIG" ]]; then
  log "  upsizing $DB_INSTANCE → $DB_TIER_BIG (restarts the instance!)"
  run "gcloud sql instances patch '$DB_INSTANCE' --project='$PROJECT' --tier='$DB_TIER_BIG' --quiet"
fi
run "gcloud run services update mobius-rag --project='$PROJECT' --region='$REGION' --min-instances=1 --max-instances=1 --no-cpu-throttling --quiet >/dev/null"
run "gcloud run services update mobius-rag-chunking-worker --project='$PROJECT' --region='$REGION' --min-instances='$CHUNK_WORKERS' --max-instances='$CHUNK_WORKERS' --quiet >/dev/null"
run "gcloud run services update mobius-rag-embedding-worker --project='$PROJECT' --region='$REGION' --min-instances='$EMBED_WORKERS' --max-instances='$EMBED_WORKERS' --quiet >/dev/null"

# ══ EVAL A — BASELINE (before any corpus write; corpus is frozen) ════════════
if [[ "$RUN_EVAL" == "1" && "$DRY_RUN" != "1" ]]; then
  log "EVAL A  baseline calibration (before push)"
  eval_run "nightly-baseline-$DATE_TAG" > "$BASELINE_JSON"
  rc=$?
  [[ $rc -eq 0 ]] || log "  baseline eval did not complete (rc=$rc) — lift report will be partial"
elif [[ "$RUN_EVAL" == "1" ]]; then
  log "EVAL A  (DRY_RUN) would run baseline calibration here"
fi

# ══ STEP 1 — publish lexicon QA→RAG (automated tag sanity gate) ══════════════
log "STEP 1  publish lexicon QA→RAG (with sanity gate)"
DRY="$(api POST "$LEX/policy/lexicon/publish" '{"dry_run": true}')"
QA_REV="$(echo "$DRY" | jget "['qa_revision']")"; RAG_REV="$(echo "$DRY" | jget "['rag_revision_before']")"
QA_ENT="$(echo "$DRY" | jget "['qa_entries']")";  RAG_ENT="$(echo "$DRY" | jget "['rag_entries_before']")"
log "  qa rev/entries=$QA_REV/$QA_ENT  rag=$RAG_REV/$RAG_ENT"
if [[ -z "$QA_REV" || -z "$RAG_REV" ]]; then die "sync/dry-run failed"; fi
GATE="$(python3 - "$QA_REV" "$RAG_REV" "$QA_ENT" "$RAG_ENT" "$GATE_MIN_RATIO" "$GATE_MAX_RATIO" <<'PY'
import sys
qr,rr,qe,re,lo,hi = int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),float(sys.argv[5]),float(sys.argv[6])
if qr <= rr: print("SKIP:already current"); raise SystemExit
if re>0 and qe < lo*re: print(f"FAIL:mass_deletion {qe}<{lo}*{re}"); raise SystemExit
if re>0 and qe > hi*re: print(f"FAIL:explosion {qe}>{hi}*{re}"); raise SystemExit
print("PASS")
PY
)"
log "  gate: $GATE"
case "$GATE" in
  PASS*)  log "  publishing QA→RAG…"; run "api POST '$LEX/policy/lexicon/publish' '{\"dry_run\": false}' >/dev/null"; sleep 15 ;;
  SKIP*)  log "  lexicon already current in RAG; skipping publish" ;;
  FAIL*)  log "  !! tag sanity gate FAILED ($GATE) — skipping lexicon publish, continuing on prior revision" ;;
  *)      die "gate check errored" ;;
esac

# ══ STEP 2 — retag existing docs (in-place) ══════════════════════════════════
log "STEP 2  retag-in-place (only_stale)"
run "api POST '$RAG/admin/retag-in-place' '{\"only_stale\": true}' >/dev/null"
[[ "$DRY_RUN" == "1" ]] || poll_until_done "$RAG/admin/retag-in-place/status" 90 || log "  (retag still running at budget; continuing)"

# ══ STEP 3 — chunk+tag new docs (deterministic) ══════════════════════════════
log "STEP 3  build-eu-from-lines + remediate (Path-B chunk, no LLM)"
run "api POST '$RAG/admin/build-eu-from-lines' '{}' >/dev/null"
[[ "$DRY_RUN" == "1" ]] || poll_until_done "$RAG/admin/build-eu-from-lines/status" 60 || log "  (build-eu still running; continuing)"
run "api POST '$RAG/admin/integrity/remediate' '{}' >/dev/null"
[[ "$DRY_RUN" == "1" ]] || poll_until_done "$RAG/admin/integrity/remediate/status" 15 || true

# ══ STEP 4 — embed → publish (time-budgeted drain) ═══════════════════════════
log "STEP 4  wait for embed drain (budget ${EMBED_BUDGET_MIN}min)"
if [[ "$DRY_RUN" != "1" ]]; then
  # Reset zombie 'processing' jobs (dead workers leave rows stuck forever) so the
  # drain can actually complete instead of idling on the un-completable.
  api POST "$RAG/admin/db/execute" "{\"sql\":\"UPDATE embedding_jobs SET status='pending', started_at=NULL WHERE status='processing' AND now()-started_at > interval '10 min'\"}" >/dev/null || true
  t=0
  while (( t < EMBED_BUDGET_MIN*60 )); do
    # PENDING is the real backlog; PROCESSING can wedge on dead workers. Break when
    # nothing is left to pick up, even if a few processing rows are stuck.
    pend="$(api POST "$RAG/admin/db/execute" "{\"sql\":\"SELECT count(*) AS n FROM embedding_jobs WHERE status='pending'\"}" | jget "['records'][0]['n']")"
    proc="$(api POST "$RAG/admin/db/execute" "{\"sql\":\"SELECT count(*) AS n FROM embedding_jobs WHERE status='processing'\"}" | jget "['records'][0]['n']")"
    pub="$(api GET "$RAG/admin/integrity/report" | jget "['published']")"
    log "    embed pending=${pend:-?} processing=${proc:-?} published=${pub:-?}"
    [[ "$pend" == "0" ]] && { log "  embed backlog drained (pending=0)"; break; }
    sleep 60; t=$((t+60))
  done
  api POST "$RAG/admin/publish_unpublished?limit=500" >/dev/null || true   # sweep straggler publishes
fi

# ══ STEP 5 — straggler cleanup + integrity gate ══════════════════════════════
log "STEP 5  integrity report + gate"
REP="$(api GET "$RAG/admin/integrity/report")"
echo "$REP" | python3 -m json.tool 2>/dev/null | sed 's/^/    /'
PUB="$(echo "$REP" | jget "['published']")"; TOTAL="$(echo "$REP" | jget "['documents_total']")"
REING="$(echo "$REP" | jget "['gaps']['need_reingest']")"; STALE="$(echo "$REP" | jget "['gaps']['stale_tags']")"
# publish fraction excluding the permanent no-pages (reingest) set
OK="$(python3 - "$PUB" "$TOTAL" "$REING" "$STALE" <<'PY'
import sys
pub,tot,reing,stale=[int(x) for x in sys.argv[1:5]]
frac = pub / max(tot-reing,1)
print("PASS" if (frac >= 0.97 and stale == 0) else f"FAIL frac={frac:.3f} stale={stale}")
PY
)"
log "  gate (fraction excl. re-ingest): $OK"

# ══ FREEZE — drain (workers UP) THEN idle, before the FINAL eval ═════════════
# The final eval must see a stable corpus: no in-flight chunk (writes tags) or
# embed/publish (writes rag_published_embeddings). ORDER MATTERS: drain the
# backlog with workers STILL RUNNING (pull-based workers scaled to 0 can't drain
# anything), then idle them to stop further writes, then a short settle.
if [[ "$RUN_EVAL" == "1" && "$DRY_RUN" != "1" ]]; then
  log "FREEZE  drain backlog (workers up) then idle (budget ${QUIESCE_BUDGET_MIN}min)"
  api POST "$RAG/admin/db/execute" "{\"sql\":\"UPDATE embedding_jobs SET status='pending', started_at=NULL WHERE status='processing' AND now()-started_at > interval '10 min'\"}" >/dev/null || true
  qt=0
  while (( qt < QUIESCE_BUDGET_MIN*60 )); do
    q="$(api POST "$RAG/admin/db/execute" "{\"sql\":\"SELECT (SELECT count(*) FROM embedding_jobs WHERE status='pending') + (SELECT count(*) FROM chunking_jobs WHERE status='pending') AS n\"}" | jget "['records'][0]['n']")"
    log "    pending backlog=${q:-?}"
    [[ "$q" == "0" ]] && { log "  backlog drained"; break; }
    sleep 30; qt=$((qt+30))
  done
  # now freeze: idle workers so nothing writes during the eval, brief settle for in-flight
  gcloud run services update mobius-rag-chunking-worker  --project="$PROJECT" --region="$REGION" --min-instances=0 --quiet >/dev/null 2>&1 || true
  gcloud run services update mobius-rag-embedding-worker --project="$PROJECT" --region="$REGION" --min-instances=0 --quiet >/dev/null 2>&1 || true
  log "  workers idled; settling 30s"; sleep 30
fi

# ══ EVAL B — FINAL (after writes settle; corpus frozen) ══════════════════════
if [[ "$RUN_EVAL" == "1" && "$DRY_RUN" != "1" ]]; then
  log "EVAL B  final calibration (after push)"
  eval_run "nightly-final-$DATE_TAG" > "$FINAL_JSON"
  rc=$?
  [[ $rc -eq 0 ]] || log "  final eval did not complete (rc=$rc) — lift report will be partial"
elif [[ "$RUN_EVAL" == "1" ]]; then
  log "EVAL B  (DRY_RUN) would run final calibration here"
fi

# ══ STEP 6 — push lexicon+tags RAG→Chat (only if gate passed) ════════════════
if [[ "$OK" == PASS* ]]; then
  log "STEP 6  push lexicon RAG→Chat"
  run "api POST '$LEX/policy/lexicon/push-to-chat' '{\"dry_run\": false}' >/dev/null"
else
  log "STEP 6  SKIPPED — RAG integrity below gate; not pushing a partial corpus to chat"
fi

# ══ STEP 0' — downsize DB (AFTER the final eval; workers already idled) ═══════
log "STEP 0' revert infra"
if [[ "$RUN_EVAL" != "1" || "$DRY_RUN" == "1" ]]; then
  run "gcloud run services update mobius-rag-chunking-worker --project='$PROJECT' --region='$REGION' --min-instances=0 --quiet >/dev/null"
  run "gcloud run services update mobius-rag-embedding-worker --project='$PROJECT' --region='$REGION' --min-instances=0 --quiet >/dev/null"
fi
if [[ "$SKIP_RESIZE" != "1" ]]; then
  log "  downsizing $DB_INSTANCE → $DB_TIER_SMALL (restarts the instance)"
  run "gcloud sql instances patch '$DB_INSTANCE' --project='$PROJECT' --tier='$DB_TIER_SMALL' --quiet"
fi

# ══ LIFT — Δ router_recall / Δ oracle_recall / headroom / contra ═════════════
if [[ "$RUN_EVAL" == "1" && "$DRY_RUN" != "1" ]]; then
  log "LIFT  final vs baseline"
  eval_lift "$BASELINE_JSON" "$FINAL_JSON"
  log "  (baseline=$BASELINE_JSON  final=$FINAL_JSON)"
fi

log "DONE. Push to chat: $([[ "$OK" == PASS* ]] && echo yes || echo SKIPPED). Re-ingest backlog: ${REING:-?} docs."
