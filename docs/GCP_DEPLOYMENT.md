# GCP Deployment Guide: Shared Infrastructure for Mobius RAG, Mobius OS New & More

Deploy Mobius RAG, Mobius OS New, and future services on **shared GCP infrastructure**. Databases, storage, and other resources are centrally managed and accessible by all servers.

---

## Shared Infrastructure Model

All services run in the same project and VPC. Shared resources (Cloud SQL, GCS, Vertex AI) are accessible by every VM and service. Add new apps by deploying to VMs in the same network—they connect to the same databases and storage.

```
                    ┌──────────────────────────────────────────────────────────────────────┐
                    │              GCP Project: mobiusos-new (Shared Infrastructure)        │
                    │                                                                      │
                    │  ┌──────────────────────────────────────────────────────────────┐   │
                    │  │              SHARED RESOURCES (accessible by all servers)     │   │
                    │  │                                                               │   │
                    │  │  ┌─────────────────────────┐  ┌────────────────────────────┐ │   │
                    │  │  │ Cloud SQL (PostgreSQL)  │  │  GCS Buckets               │ │   │
                    │  │  │ • mobius_rag            │  │  • mobius-rag-uploads      │ │   │
                    │  │  │ • mobius_os_new (etc.)  │  │  • shared-assets (etc.)    │ │   │
                    │  │  │ + pgvector extension   │  │                            │ │   │
                    │  │  └───────────▲────────────┘  └────────────▲───────────────┘ │   │
                    │  │              │                            │                  │   │
                    │  │  ┌───────────┴────────────────────────────┴───────────────┐ │   │
                    │  │  │  Vertex AI (Gemini) – LLM + Embeddings                  │ │   │
                    │  │  └───────────────────────────────────────────────────────┘ │   │
                    │  └──────────────────────────────────────────────────────────────┘   │
                    │              ▲                    ▲                    ▲              │
                    │              │ Private IP / VPC   │                    │              │
                    │  ┌───────────┴────────────────────┴────────────────────┴──────────┐  │
                    │  │                    VPC Network (same subnet)                   │  │
                    │  │                                                                │  │
                    │  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │  │
                    │  │  │  VM 1            │  │  VM 2            │  │  VM N        │ │  │
                    │  │  │  Mobius RAG      │  │  Mobius OS New   │  │  (future)    │ │  │
                    │  │  │  • mragb         │  │  • app server    │  │              │ │  │
                    │  │  │  • mragw         │  │                  │  │              │ │  │
                    │  │  │  • mrage         │  │                  │  │              │ │  │
                    │  │  └──────────────────┘  └──────────────────┘  └──────────────┘ │  │
                    │  └───────────────────────────────────────────────────────────────┘  │
                    │                                                                      │
                    │  Load Balancer / Ingress → routes traffic to VMs by host/path       │
                    └──────────────────────────────────────────────────────────────────────┘
```

**Principles:**
- **One VPC** – All VMs in the same network; private connectivity to Cloud SQL and each other
- **Shared Cloud SQL** – One instance, multiple databases (`mobius_rag`, `mobius_os_new`, etc.); all servers connect via private IP
- **Shared GCS** – Buckets used by any service with the same service account
- **Shared service account** – One SA (or a few) with access to Cloud SQL, GCS, Vertex AI
- **Add services** – New VMs join the VPC, use same `DATABASE_URL` pattern and GCS bucket access

---

## Option A: Single VM First (Simplest Start)

Run Mobius RAG and Mobius OS New on the same VM initially. Same shared Cloud SQL and GCS. When you add more services, deploy new VMs in the same VPC—they connect to the same databases and storage.

### 1. GCP Prerequisites

- [ ] GCP project (e.g. `mobiusos-new`)
- [ ] `gcloud` CLI installed and authenticated
- [ ] Billing enabled

### 2. Create GCP Resources

**Firewall (allow HTTP):**
```bash
gcloud compute firewall-rules create allow-http --allow tcp:80 --source-ranges 0.0.0.0/0
```

```bash
# Set project
export PROJECT_ID=mobiusos-new
gcloud config set project $PROJECT_ID

# Enable APIs
gcloud services enable sqladmin.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable compute.googleapis.com

# Cloud SQL (PostgreSQL) – shared by all services
# Start with public IP; enable Private IP later (see step 4)
gcloud sql instances create mobius-platform-db \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=us-central1 \
  --storage-size=10GB

# Create databases for each service (add more as you grow)
gcloud sql databases create mobius_rag --instance=mobius-platform-db
gcloud sql databases create mobius_os_new --instance=mobius-platform-db

gcloud sql users set-password postgres --instance=mobius-platform-db --password=YOUR_SECURE_PASSWORD

# Enable pgvector (for Mobius RAG): Connect and run in mobius_rag database:
# CREATE EXTENSION IF NOT EXISTS vector;

# GCS buckets (shared by all services; add more as needed)
gsutil mb -l us-central1 gs://mobius-uploads-${PROJECT_ID}/
# gsutil mb -l us-central1 gs://mobius-assets-${PROJECT_ID}/

# Shared service account – all VMs/services use this to access Cloud SQL, GCS, Vertex AI
gcloud iam service-accounts create mobius-platform-sa \
  --display-name="Mobius Platform (shared)"

# Grant access to shared resources (all servers will use this SA)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:mobius-platform-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:mobius-platform-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:mobius-platform-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/cloudsql.client"

# Download key (store in Secret Manager in production; VMs mount or env-inject it)
gcloud iam service-accounts keys create mobius-platform-sa.json \
  --iam-account=mobius-platform-sa@${PROJECT_ID}.iam.gserviceaccount.com
```

### 3. Create VM

```bash
# VM in default VPC – will reach Cloud SQL via private IP when configured
gcloud compute instances create mobius-platform-vm \
  --zone=us-central1-a \
  --machine-type=e2-medium \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=30GB \
  --scopes=cloud-platform \
  --service-account=mobius-platform-sa@${PROJECT_ID}.iam.gserviceaccount.com
```

### 4. Cloud SQL Connection (Private IP – All Servers Reach DB)

For **shared access** from all VMs, use **Private IP**:

1. **Enable Private Services Access** (one-time per project):
   ```bash
   # Allocate IP range for private services
   gcloud compute addresses create google-managed-services-default \
     --global --purpose=VPC_PEERING --prefix-length=16 \
     --network=default
   gcloud services vpc-peerings connect \
     --service=servicenetworking.googleapis.com \
     --ranges=google-managed-services-default
   ```

2. **Configure Cloud SQL for Private IP** (after VPC peering):
   ```bash
   gcloud sql instances patch mobius-platform-db \
     --network=projects/$PROJECT_ID/global/networks/default
   # Or create new instance with --no-assign-public-ip
   ```

3. **Get private IP** and use in DATABASE_URL:
   ```bash
   gcloud sql instances describe mobius-platform-db --format='value(ipAddresses[0].ipAddress)'
   # Use this IP: DATABASE_URL=postgresql+asyncpg://postgres:PASS@10.x.x.x:5432/mobius_rag
   ```

**Alternative: Cloud SQL Auth Proxy** (if Private IP isn't set up yet):
```bash
./cloud-sql-proxy $PROJECT_ID:us-central1:mobius-platform-db &
# Listens on localhost:5432 – use 127.0.0.1 in DATABASE_URL
```

### 5. Deploy Mobius RAG on VM

```bash
# SSH into VM
gcloud compute ssh mobius-platform-vm --zone=us-central1-a

# On VM:
sudo apt update && sudo apt install -y python3.11 python3.11-venv git nginx

# Clone and setup
git clone https://github.com/YOUR_ORG/mobius-rag.git
cd mobius-rag

python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[vertex]"

# Create .env (use Secret Manager in production)
cat > .env << 'EOF'
ENV=prod
DATABASE_URL=postgresql+asyncpg://postgres:PASSWORD@127.0.0.1:5432/mobius_rag
GCS_BUCKET=mobius-uploads-mobiusos-new
VERTEX_PROJECT_ID=mobiusos-new
VERTEX_LOCATION=us-central1
VERTEX_MODEL=gemini-1.5-pro
LLM_PROVIDER=vertex
GOOGLE_APPLICATION_CREDENTIALS=/opt/mobius-platform/mobius-platform-sa.json
EOF

# Run migrations
python -m app.migrations.add_embedding_tables

# Run backend, workers, frontend (use systemd for production)
./mragb &   # Backend :8000
./mragw &   # Chunking worker
./mrage &   # Embedding worker

cd frontend && npm install && npm run build
# Serve frontend with nginx or: npx serve -s dist -l 5173
```

### 6. Run Mobius OS New on Same VM

If Mobius OS New is a separate app (Node, Python, etc.), run it on the same VM:

- **Same port space**: Mobius RAG backend :8000, frontend :5173 (or nginx :80). Mobius OS New on another port (e.g. :3000).
- **Reverse proxy**: Nginx routes by path or subdomain:
  - `rag.yourdomain.com` → Mobius RAG
  - `app.yourdomain.com` or `yourdomain.com` → Mobius OS New
- **Shared resources**: Same `mobius-platform-sa`, Cloud SQL (use `mobius_os_new` database), GCS bucket(s).

---

## Adding More Services to the Shared Infrastructure

When you add a new service (another app, worker, or API):

1. **Create a database** (if it needs its own schema):
   ```bash
   gcloud sql databases create my_new_service --instance=mobius-platform-db
   ```

2. **Create a new VM** (or use Cloud Run) in the same project:
   ```bash
   gcloud compute instances create my-service-vm \
     --zone=us-central1-a \
     --machine-type=e2-medium \
     --network=default \
     --service-account=mobius-platform-sa@${PROJECT_ID}.iam.gserviceaccount.com
   ```

3. **Configure the service** with:
   - `DATABASE_URL=postgresql+asyncpg://postgres:PASS@<Cloud-SQL-private-IP>:5432/my_new_service`
   - `GCS_BUCKET=mobius-uploads-${PROJECT_ID}` (or a dedicated bucket)
   - `GOOGLE_APPLICATION_CREDENTIALS` pointing to the shared SA key
   - Same VERTEX_PROJECT_ID if it uses Vertex AI

4. **Nginx / Load Balancer**: Add a route for the new service (host or path).

All VMs in the same VPC can reach Cloud SQL via its private IP. No proxy needed when Private IP is configured.

---

## Option B: Cloud Run (Managed, Scalable)

Deploy Mobius RAG to Cloud Run for a GCP-managed URL like `https://mobius-rag-xxx.run.app` (similar to Mobius OS New).

### Deploy to Cloud Run

```bash
# From project root
DATABASE_PASSWORD=YOUR_PASSWORD ./deploy/deploy_cloudrun.sh
```

This will:
1. Build the frontend (VITE_API_BASE= for same-origin)
2. Build the Docker image via Cloud Build
3. Deploy to Cloud Run with Cloud SQL connector
4. Output the service URL (e.g., `https://mobius-rag-xxxxx-uc.a.run.app`)

### Workers (Chunking + Embedding)

Cloud Run is request-based. The chunking and embedding workers are long-running processes. They can run on:

- **Existing VM** (`mobius-platform-vm`) – SSH in and start workers:
  ```bash
  gcloud compute ssh mobius-platform-vm --zone=us-central1-a --project=mobiusos-new
  sudo systemctl start mobius-rag-chunking-worker mobius-rag-embedding-worker
  ```
- **Cloud Run Jobs** (future) – Periodic or triggered jobs for batch processing

### Prerequisites

- Cloud SQL instance `mobius-platform-db` with `mobius_rag` database
- GCS bucket `mobius-uploads-mobiusos-new`
- Cloud Build API and Cloud Run API enabled

---

## Option C: GKE (Kubernetes)

For larger scale or multiple services, use GKE:

- Backend, workers, Mobius OS New as separate deployments
- Shared PostgreSQL (Cloud SQL), GCS, Vertex AI

---

## Environment Variables Checklist

| Variable | Required | Example |
|----------|----------|---------|
| `ENV` | Yes | `prod` |
| `DATABASE_URL` | Yes | `postgresql+asyncpg://user:pass@host:5432/mobius_rag` |
| `GCS_BUCKET` | Yes | `mobius-rag-uploads-mobiusos-new` |
| `VERTEX_PROJECT_ID` | Yes (for Vertex) | `mobiusos-new` |
| `VERTEX_LOCATION` | No | `us-central1` |
| `VERTEX_MODEL` | No | `gemini-1.5-pro` |
| `LLM_PROVIDER` | No | `vertex` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Yes (for Vertex/GCS) | Path to SA JSON |
| `EMBEDDING_PROVIDER` | No | `vertex` (default when VERTEX set) |
| `CHROMA_HOST` | No | Only if using ChromaDB |
| `CHROMA_PERSIST_DIR` | No | For persistent Chroma |

---

## Frontend API Base URL

The frontend uses `VITE_API_BASE` from `src/config.ts`. For production:

1. **Build-time**: Set `VITE_API_BASE` in `.env.production`:
   ```
   VITE_API_BASE=https://api.yourdomain.com
   ```
2. Build: `cd frontend && VITE_API_BASE=https://api.yourdomain.com npm run build`
3. Or copy `frontend/.env.production.example` to `frontend/.env.production` and set the URL.

---

## Recommended Next Steps

1. **Add `VITE_API_BASE`** – Make frontend API URL configurable for production.
2. **Dockerfile** – Containerize backend + workers for Cloud Run or GKE.
3. **systemd units** – For VM deployment: `mobius-rag-backend.service`, `mobius-rag-worker.service`, `mobius-rag-embedding.service`.
4. **Secret Manager** – Store DATABASE_URL, service account key via Secret Manager instead of .env file.
5. **Health checks** – Ensure `/health` exists and use it for load balancer / Cloud Run.

---

## Migrate Local Data to Cloud SQL

To move your existing Mobius RAG data (documents, chunks, facts, etc.) to Cloud SQL:

### Option A: Cloud SQL Auth Proxy + psql (recommended)

```bash
# 1. Set postgres password (if not set)
gcloud sql users set-password postgres --instance=mobius-platform-db --password=YOUR_PASSWORD

# 2. Dump local DB (if needed)
pg_dump -h localhost -U ananth -d mobius_rag --no-owner --no-acl > mobius_rag_backup.sql

# 3. Prepare dump: replace owner with postgres, remove DROP statements
sed -e 's/OWNER TO [a-zA-Z0-9_]*/OWNER TO postgres/g' -e '/^DROP /d' mobius_rag_backup.sql > prepared.sql

# 4. Run migration via proxy
CLOUD_SQL_PASSWORD=YOUR_PASSWORD ./scripts/migrate_to_gcp_psql.sh prepared.sql
```

### Option B: gcloud sql import (via GCS)

```bash
./scripts/migrate_to_gcp.sh ~/mobius_rag_backup.sql
```

If import fails with "permission denied", use Option A.

### After migration: Run embedding migration

```bash
# Start Cloud SQL Auth Proxy on port 5433
./cloud-sql-proxy mobiusos-new:us-central1:mobius-platform-db --port 5433 &

# Run embedding tables migration
DATABASE_URL="postgresql+asyncpg://postgres:PASSWORD@127.0.0.1:5433/mobius_rag" python -m app.migrations.add_embedding_tables
```

---

## Quick Start: Same VM as Mobius OS New

If Mobius OS New is already on a GCP VM:

1. Install Mobius RAG alongside it.
2. Use same GCP project, GCS bucket, and service account.
3. Use Cloud SQL or attach to existing PostgreSQL (if you have one).
4. Run `mragb`, `mragw`, `mrage` as additional processes.
5. Configure nginx/reverse proxy to route `/api` or `rag.yourdomain.com` to Mobius RAG backend.

---

## Cost Estimate (Option A, Single VM)

| Resource | Estimate |
|----------|----------|
| e2-medium VM | ~$25–30/month |
| Cloud SQL db-f1-micro | ~$10/month |
| GCS | ~$0.02/GB/month |
| Vertex AI | Pay per request (embedding + LLM) |
| **Total** | **~$40–50/month** (before Vertex usage) |
