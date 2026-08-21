#!/usr/bin/env bash
# Deploy Mobius RAG to Cloud Run in mobius-os-dev.
#
# What this script does (all idempotent):
#   1. Build rag Docker image with Cloud Build → Artifact Registry
#   2. Deploy mobius-rag API (uvicorn app.main:app)
#   3. Deploy mobius-rag-chunking-worker (uvicorn app.worker_server_chunking:app)
#   4. Deploy mobius-rag-embedding-worker (uvicorn app.worker_server_embedding:app)
#
# Prerequisites (one-time, already done as of 2026-04-23):
#   * Secret Manager contains mobius-skill-llm-internal-key (shared with chat)
#   * Secret Manager contains rag-admin-api-key
#   * Chat service is patched to MOBIUS_SKILL_LLM_INTERNAL_KEY=<secret>
#   * Runtime SA (mobius-platform-dev) has secretmanager.secretAccessor role
#
# Usage:
#   ./deploy/deploy_cloudrun_dev.sh            # tag with git sha
#   TAG=v7-smoke ./deploy/deploy_cloudrun_dev.sh
set -euo pipefail

PROJECT_ID="mobius-os-dev"
REGION="us-central1"
CLOUD_SQL_INSTANCE="mobius-platform-dev-db"
CLOUD_SQL_CONNECTION="${PROJECT_ID}:${REGION}:${CLOUD_SQL_INSTANCE}"

# Chat URL is the proxy endpoint for rag's LLM calls. Point at
# /internal/skill-llm — chat gates on MOBIUS_SKILL_LLM_INTERNAL_KEY.
CHAT_INTERNAL_LLM_URL="https://mobius-chat-ortabkknqa-uc.a.run.app/internal/skill-llm"

# lexicon-maintenance service — inline candidate cleanup after extraction.
LEXICON_MAINTENANCE_URL="https://mobius-lexicon-maintenance-ortabkknqa-uc.a.run.app"

# DB connection via Cloud SQL Unix socket (Cloud Run connector). Chat
# uses `postgres@` (no password) with the auth-proxy; rag historically
# uses `mobius_app` with a password. Keep the latter for parity with
# staging.
DB_USER="postgres"
DB_PASS_SECRET="db-password"  # Secret Manager secret that has the rag DB password
# Read the password so we can inline it (gcloud run deploy doesn't
# support --set-secrets for URL-embedded passwords).
DB_PASS=$(gcloud secrets versions access latest --secret="$DB_PASS_SECRET" --project="$PROJECT_ID" 2>/dev/null || echo "")
if [[ -z "$DB_PASS" ]]; then
  echo "ERROR: could not read secret $DB_PASS_SECRET. Either create it or edit this script."
  exit 1
fi
# urlencode the '$' in the password manually; we know the password shape.
# URL-encode the password for safe embedding in connection strings
# (the dev password contains a ``$`` that would otherwise break
# shell expansion on the other side of the wire).
DB_PASS_ENC=$(python3 -c "import urllib.parse, sys; print(urllib.parse.quote(sys.argv[1]))" "$DB_PASS")
DB_URL="postgresql+asyncpg://${DB_USER}:${DB_PASS_ENC}@/mobius_rag?host=%2Fcloudsql%2F${PROJECT_ID}%3A${REGION}%3A${CLOUD_SQL_INSTANCE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo "manual-$(date +%s)")}"
REPO="us-central1-docker.pkg.dev/${PROJECT_ID}/mobius-rag"
IMAGE="${REPO}/rag:${TAG}"

echo "=============================================================="
echo "Deploy Mobius RAG → ${PROJECT_ID} (tag=${TAG})"
echo "=============================================================="

# 1. Ensure Artifact Registry repo exists (idempotent)
gcloud artifacts repositories describe mobius-rag \
  --project="$PROJECT_ID" --location="$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create mobius-rag \
    --project="$PROJECT_ID" --location="$REGION" \
    --repository-format=docker --description="Mobius RAG container images" --quiet

# 2a. Frontend: rebuild dist before docker build. The Dockerfile expects
#     ``frontend/dist`` to be present (no node toolchain inside the
#     image to keep size down). Operators were forgetting this step
#     and shipping with stale UI bundles — the 'wizard not visible'
#     debug session on 2026-04-27 was exactly this.
if [[ -d frontend ]] && [[ -f frontend/package.json ]]; then
  echo "--- building frontend dist ---"
  # Vite reads VITE_* env vars at build time and bakes them into the
  # bundle. Without these the URL panel's Submit button posts to a
  # same-origin /scrape (which is the rag service, not the scraper),
  # producing "Method Not Allowed". Default to the dev URLs; operators
  # can override per-call by exporting these before invoking this script.
  : "${VITE_SCRAPER_API_BASE:=https://mobius-web-scraper-ortabkknqa-uc.a.run.app}"
  : "${VITE_API_BASE:=}"   # empty → same-origin (rag-self), correct for the deployed shell
  # Lexicon Maintenance lives on its own Cloud Run service; the header link
  # is the natural path from the RAG dashboard into tag curation.
  : "${VITE_LEXICON_URL:=https://mobius-lexicon-maintenance-ortabkknqa-uc.a.run.app}"
  # DEV ONLY: lets the app self-mint a platform token when a long session
  # outlives the launcher token (avoids /admin/* 401s). Points at a SAME-ORIGIN
  # RAG proxy (/dev/mint-token) that server-side calls chat's mint-dev-token —
  # avoids browser CORS. NEVER set in prod; the proxy is also ENV-gated off there.
  : "${VITE_DEV_MINT_URL:=/dev/mint-token}"
  export VITE_SCRAPER_API_BASE VITE_API_BASE VITE_LEXICON_URL VITE_DEV_MINT_URL
  echo "--- frontend env: VITE_SCRAPER_API_BASE=$VITE_SCRAPER_API_BASE  VITE_API_BASE=${VITE_API_BASE:-<same-origin>}  VITE_LEXICON_URL=$VITE_LEXICON_URL ---"
  (cd frontend && npm run build) || {
    echo "ERROR: frontend build failed; aborting deploy" >&2
    exit 1
  }
  echo "--- frontend dist hash: $(ls frontend/dist/assets/index-*.js 2>/dev/null | head -1) ---"
fi

# 2b. Build image via Cloud Build (uses mobius-rag/.gcloudignore to avoid
#     pushing 1GB of pycache/node_modules).
echo "--- building $IMAGE ---"
gcloud builds submit --project="$PROJECT_ID" --tag="$IMAGE" .

# Common env + flag shape shared by all three services
# NOTE: LLM_PROVIDER removed (Phase 1 gate rejects it as stale).
#
# Chroma + chat Postgres are wired so the publish endpoint can sync
# directly into chat's retrieval stores (see
# app/services/publish_sync.py). Without these, publish still
# succeeds but downstream chat retrieval never sees the doc.
# Contract reference: mobius-chat/docs/rag_population_agent_setup.md.
CHAT_DB_URL_FOR_RAG="postgresql+psycopg2://postgres:${DB_PASS_ENC}@/mobius_chat?host=%2Fcloudsql%2F${PROJECT_ID}%3A${REGION}%3A${CLOUD_SQL_INSTANCE}"
ORG_DOCS_DB_URL="postgresql+asyncpg://postgres:${DB_PASS_ENC}@/mobius_org_docs?host=%2Fcloudsql%2F${PROJECT_ID}%3A${REGION}%3A${CLOUD_SQL_INSTANCE}"

COMMON_ENV=(
  "ENV=staging"
  # DEV ONLY: enables /dev/mint-token so the SPA can self-mint a platform token
  # (long sessions outlive the launcher token → /admin/* 401 → UI blanks). This is
  # the dev deploy script; prod/staging deploys must NOT set this.
  "ALLOW_DEV_MINT=1"
  "DATABASE_URL=${DB_URL}"
  "GCS_BUCKET=mobius-rag-uploads-dev"
  "VERTEX_PROJECT_ID=${PROJECT_ID}"
  "VERTEX_LOCATION=${REGION}"
  "VERTEX_MODEL=gemini-2.5-flash"
  "EMBEDDING_PROVIDER=vertex"
  "CHAT_INTERNAL_LLM_URL=${CHAT_INTERNAL_LLM_URL}"
  "LEXICON_MAINTENANCE_URL=${LEXICON_MAINTENANCE_URL}"
  # Vector store — pgvector is the prod backend post-cutover (2026-04-27).
  # Without this, vector_store.get_vector_store() falls back to Chroma when
  # CHROMA_HOST is set, silently re-routing to the unstable Chroma VM.
  "VECTOR_STORE=pgvector"
  # IDF-weighted dtag arm — weights rare tags (e.g. claims.timely_filing=17 chunks)
  # higher than broad parents (claims.general=31k) in RRF fusion. Eval-gated:
  # set to 1 for IDF head-to-head, unset for binary baseline.
  "DTAG_ARM_IDF=1"
  # Chroma kept as legacy fallback for ad-hoc admin /vector_search?store=chroma
  # comparisons; ignored at runtime when VECTOR_STORE=pgvector.
  "CHROMA_HOST=34.170.243.161"
  "CHROMA_PORT=8000"
  "CHROMA_SSL=0"
  # chat Postgres (same Cloud SQL instance, mobius_chat database)
  "CHAT_DATABASE_URL=${CHAT_DB_URL_FOR_RAG}"
  # Strategy (d) external web search — same service chat uses for
  # google_search. Required for the corpus_search_agent to fall back
  # to external sources when no corpus docs match the payer/query.
  "CHAT_SKILLS_GOOGLE_SEARCH_URL=https://mobius-google-search-ortabkknqa-uc.a.run.app/search"
  # Auto-publish on embed: when an embedding_job completes, the worker
  # immediately copies vectors into rag_published_embeddings + chat
  # Postgres so the doc is queryable end-to-end without a separate
  # "publish" admin call. Without this, embedded docs sit invisible to
  # chat retrieval forever — exactly the failure mode that stranded
  # 23 humana docs on 2026-04-27 after a deploy reset env vars.
  "AUTO_PUBLISH_ON_EMBED=1"
  # Org-docs DB: per-org namespace in mobius_org_docs (same Cloud SQL instance).
  # Gates POST /org-docs/ingest + GET /org-docs/search.
  "ORG_DOCS_DATABASE_URL=${ORG_DOCS_DB_URL}"

  # ---- Feature flags -------------------------------------------------------
  # DECLARE FLAGS HERE, NEVER OUT OF BAND.
  #
  # This deploy uses `--set-env-vars`, which REPLACES the whole environment.
  # Anything set afterwards with `gcloud run services update` survives only
  # until the next deploy silently wipes it. That is not hypothetical: the
  # AUTO_PUBLISH_ON_EMBED comment above records 23 humana docs stranded on
  # 2026-04-27 by exactly this, and on 2026-08-19 a deploy wiped TABLE_CAPTURE
  # mid-milestone — the same bug, two flags, four months apart. The lesson had
  # been applied to one variable instead of made general.
  #
  # So the script is the single source of truth for the environment. Each flag
  # takes its value from the shell when set, so a one-off run can flip it
  # (`TABLE_CAPTURE=off ./deploy/deploy_cloudrun_dev.sh`) without an edit, and
  # the default here is what dev returns to otherwise.

  # Table capture: rewrites page text, excising detected tables into
  # document_tables and leaving a breadcrumb. Changes what gets chunked AND what
  # the dedup gate compares, so it is a deliberate on, not a default on.
  "TABLE_CAPTURE=${TABLE_CAPTURE:-on}"
  # Passenger tables: attaches a retrieved chunk's table to the answer, by
  # breadcrumb or by (document_id, page_number) proximity.
  "PASSENGER_TABLE_RETRIEVAL=${PASSENGER_TABLE_RETRIEVAL:-true}"
)

COMMON_SECRETS=(
  "MOBIUS_SKILL_LLM_INTERNAL_KEY=mobius-skill-llm-internal-key:latest"
  "ADMIN_API_KEY=rag-admin-api-key:latest"
  "CHROMA_AUTH_TOKEN=chroma-auth-token:latest"
  "JWT_SECRET=jwt-secret:latest"
)

join_with() { local IFS="$1"; shift; echo "$*"; }

deploy_service() {
  local name="$1"
  local command="$2"         # comma-separated CMD override
  local min_instances="$3"
  local max_instances="$4"
  local cpu_throttling="$5"  # --no-cpu-throttling vs default
  local memory="$6"
  local extra_env="$7"       # optional per-service env, comma-separated

  # CONNECTION BUDGET (2026-08-21). Pool size is PER INSTANCE; max_connections
  # is GLOBAL. config.py sized the pool at 5+10 with the note "5+10 per service
  # instance leaves plenty of headroom" — true when each service was ONE
  # instance, false the moment the workers became a fleet.
  #
  #     12 chunking x (15 batch + 5 instant) + 6 embedding x 15 + API 15 = 360
  #     against max_connections = 200
  #
  # That is what took the DB to 203/200 today: the accounting panel rendered
  # -1 sentinels, ALTER TABLE convoyed behind them, and every worker claim
  # query queued behind the ALTER. Measured actual usage was ~3.3 connections
  # per instance — the pools were oversized about 5x.
  #
  # A worker is a SERIAL consumer: it claims one job at a time. It has no use
  # for a 15-connection fan-out pool; that shape is right for a request-serving
  # API and wrong for a queue worker. Workers get 2+3; the API keeps 5+10
  # because it is a single instance serving 20 concurrent requests.
  #
  #     8 chunking x (5+5) + 6 embedding x 5 + API 15 + other 15 = 140 of 190
  #
  # If you raise an instance count, redo this arithmetic first.

  echo ""
  echo "--- deploying ${name} ---"
  local flags=(
    --image="$IMAGE"
    --project="$PROJECT_ID"
    --region="$REGION"
    --platform=managed
    --allow-unauthenticated
    --memory="$memory"
    --cpu=4
    --timeout=3600
    --add-cloudsql-instances="$CLOUD_SQL_CONNECTION"
    --service-account="mobius-platform-dev@${PROJECT_ID}.iam.gserviceaccount.com"
    --set-env-vars="$(join_with ',' "${COMMON_ENV[@]}")${extra_env:+,${extra_env}}"
    --set-secrets="$(join_with ',' "${COMMON_SECRETS[@]}")"
    --min-instances="$min_instances"
    --max-instances="$max_instances"
    --quiet
  )
  if [[ "$cpu_throttling" == "no" ]]; then
    flags+=(--no-cpu-throttling)
  fi
  if [[ -n "$command" ]]; then
    flags+=(--command="$command")
  fi

  gcloud run deploy "$name" "${flags[@]}"
}

# 3. API service. min=max=1 (single instance) — REQUIRED: in-process background
#    state lives in memory on ONE instance (the eval/calibration runner AND the
#    nightly orchestrator's live status). At max>1, Cloud Run can route the
#    status poll to a different instance than the one running the job (empty
#    status) and can kill the instance mid-eval when it scales down (the 13/110
#    stall we hit). no-cpu-throttling keeps the long background task alive.
#    Dev-scale only; a multi-instance prod needs DB-backed job state instead.
#
#    CPU 2 -> 4 (2026-08-20). min=max=1 is a correctness constraint we cannot
#    lift today, but it means background extraction (restart_extraction spawns
#    asyncio.create_task IN THIS PROCESS) competes with query serving on the same
#    instance. Measured during a 30-document reingest: a retrieval query that
#    normally answers in 22s returned status="timeout" at 46s with zero chunks.
#    More cores does not make it multi-instance, it just stops one batch from
#    starving every reader.
#
#    THE REAL FIX, not done here: move extraction to its own self-polling worker
#    like chunking and embedding, so the API serves requests and nothing else.
#    That also removes the reason min=max=1 exists.
deploy_service "mobius-rag" "" 1 1 "no" "8Gi"   # 8Gi (2026-08-20, corpus-scale run): publishing a giant doc
                                                #      (~9k embeddings) OOM'd at 1Gi; 2Gi held for single
                                                #      documents but not for a sustained batch. Scale back
                                                #      down after the AHCA run.

# 4. Chunking worker. Self-polling supervisor (FOR UPDATE SKIP LOCKED
#    handles dedup across instances), so Cloud Run autoscaling never
#    fires from HTTP load — we have to pin min=max=N to get N parallel
#    pollers. With single instance, queue p50 wait was 2h (2026-04-27
#    perf scan); 5 pollers brings throughput from ~32 docs/24h to
#    theoretical ~9k/day. Drop min back to 1 if cost matters more
#    than instant-rag SLA.
deploy_service "mobius-rag-chunking-worker" \
  "uvicorn,app.worker_server_chunking:app,--host,0.0.0.0,--port,8080" \
  8 8 "no" "8Gi" "DB_POOL_SIZE=2,DB_MAX_OVERFLOW=3"

# 5. Embedding worker. Same self-polling shape as chunking, so instance count IS
#    the parallelism — and at 1 it was the serial bottleneck of the whole
#    pipeline: 12 chunking pollers fed a single embedder, which then also does
#    auto-publish-on-embed. For a corpus-scale rerun (AHCA, 2026-08-20) that one
#    instance is what everything queues behind. Raised to 6 rather than matching
#    chunking's 12 because each instance holds a giant document's ~9k embeddings
#    in memory at publish time, and Vertex quota is the next ceiling anyway.
deploy_service "mobius-rag-embedding-worker" \
  "uvicorn,app.worker_server_embedding:app,--host,0.0.0.0,--port,8080" \
  6 6 "no" "8Gi" "DB_POOL_SIZE=2,DB_MAX_OVERFLOW=3"   # 8Gi: auto-publish-on-embed loads a giant's ~9k embeddings into memory

# 6. Print URLs

# --- Verify the environment actually landed ------------------------------
# READ THE WRITE BACK. Declaring a var and assuming it deployed is precisely
# how TABLE_CAPTURE went missing: the deploy reported success, the flag was
# gone, and the only symptom was a feature quietly not running. This compares
# what COMMON_ENV declared against what the live revision actually serves, and
# fails loudly on a mismatch rather than leaving it to be discovered later.
echo ""
echo "--- verifying environment on mobius-rag ---"
live_env="$(gcloud run services describe mobius-rag \
  --project="$PROJECT_ID" --region="$REGION" \
  --format='value(spec.template.spec.containers[0].env)' 2>/dev/null)"
env_missing=0
for pair in "${COMMON_ENV[@]}"; do
  key="${pair%%=*}"
  if ! grep -q "'${key}'" <<<"$live_env"; then
    echo "  MISSING: ${key}"
    env_missing=$((env_missing + 1))
  fi
done
if [[ "$env_missing" -gt 0 ]]; then
  echo "  ERROR: ${env_missing} declared env var(s) are not on the live revision."
  echo "  The deploy reported success but the service is not running the declared"
  echo "  configuration. Do not treat this deploy as good."
  exit 1
fi
echo "  OK — all ${#COMMON_ENV[@]} declared env vars present on the live revision."

# READ THE INSTANCE COUNT BACK TOO — for the same reason, on a setting that
# already lied to us once today.
#
# `gcloud run deploy --min-instances/--max-instances` writes the knative
# annotation on the REVISION template. Cloud Run's v2 API also carries a
# SERVICE-level `scaling` block, and when that is set it wins. This service had
# service-level {min:4,max:4} while the script declared 12 12 and every revision
# annotation said 12. The revision sat at `MinInstancesProvisioned: Unknown /
# MinInstancesWarming` indefinitely and ran exactly 4 instances. Nothing failed;
# the fleet was simply a third of the declared size, for as long as nobody
# checked.
#
# gcloud cannot read or clear the service-level block, so this asks the v2 API
# directly and fails the deploy on a mismatch.
echo ""
echo "--- verifying instance scaling ---"
scaling_bad=0
_tok="$(gcloud auth print-access-token)"
check_scaling() {
  local svc="$1" want="$2"
  local got
  got="$(curl -s -H "Authorization: Bearer ${_tok}" \
    "https://run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/services/${svc}" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);s=d.get('scaling') or {};print(s.get('maxInstanceCount') or 0)" 2>/dev/null)"
  if [[ "$got" != "0" && "$got" != "$want" ]]; then
    echo "  MISMATCH ${svc}: script declared ${want}, service-level scaling caps at ${got}"
    echo "    fix: curl -X PATCH .../services/${svc}?updateMask=scaling \\"
    echo "         -d '{\"scaling\":{\"minInstanceCount\":${want},\"maxInstanceCount\":${want}}}'"
    scaling_bad=$((scaling_bad + 1))
  else
    printf "  OK  %-34s %s instances\n" "$svc" "$want"
  fi
}
check_scaling "mobius-rag"                   1
check_scaling "mobius-rag-chunking-worker"   8
check_scaling "mobius-rag-embedding-worker"  6
if [[ "$scaling_bad" -gt 0 ]]; then
  echo "  ERROR: the fleet is not the size this script declared. Connection-budget"
  echo "  arithmetic in deploy_service() assumes the declared counts."
  exit 1
fi

echo ""
echo "=============================================================="
echo "Deploy complete. URLs:"
for s in mobius-rag mobius-rag-chunking-worker mobius-rag-embedding-worker; do
  url=$(gcloud run services describe "$s" --project="$PROJECT_ID" --region="$REGION" --format='value(status.url)')
  printf "  %-34s %s\n" "$s" "$url"
done
echo ""
echo "Smoke check: curl $(gcloud run services describe mobius-rag --project=${PROJECT_ID} --region=${REGION} --format='value(status.url)')/health/deep"
