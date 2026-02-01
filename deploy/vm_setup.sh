#!/usr/bin/env bash
# Run on VM: sudo bash deploy/vm_setup.sh (from app root)
# Env: DATABASE_PASSWORD, CLOUD_SQL_IP (optional), PROJECT_ID (optional)
set -e

DATABASE_PASSWORD="${DATABASE_PASSWORD:?Set DATABASE_PASSWORD}"
CLOUD_SQL_IP="${CLOUD_SQL_IP:-34.59.175.121}"
PROJECT_ID="${PROJECT_ID:-mobiusos-new}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Mobius RAG VM Setup (APP_DIR=$APP_DIR) ==="

apt-get update -qq
# Build deps for cffi/cryptography (google-cloud-storage, Vertex AI)
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-dev libffi-dev build-essential
# Python (frontend pre-built locally)
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3.11 python3.11-venv git nginx 2>/dev/null || \
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3 python3-venv git nginx

useradd -m -s /bin/bash mobius 2>/dev/null || true
chown -R mobius:mobius "$APP_DIR" 2>/dev/null || true

# Python venv (libffi-dev needed for cffi/cryptography)
sudo -u mobius bash -c "
  cd $APP_DIR
  python3.11 -m venv .venv 2>/dev/null || python3 -m venv .venv
  . .venv/bin/activate
  pip install -q -e \".[vertex]\"
  pip install -q --force-reinstall cffi cryptography || true
"

# .env (VM uses service account via metadata - no key file needed)
sudo -u mobius tee "$APP_DIR/.env" << EOF
ENV=prod
DATABASE_URL=postgresql+asyncpg://postgres:${DATABASE_PASSWORD}@${CLOUD_SQL_IP}:5432/mobius_rag
GCS_BUCKET=mobius-uploads-mobiusos-new
VERTEX_PROJECT_ID=${PROJECT_ID}
VERTEX_LOCATION=us-central1
VERTEX_MODEL=gemini-1.5-pro
LLM_PROVIDER=vertex
EOF

# Frontend: pre-built in tarball (dist/). If missing, try build (requires Node 18+)
if [[ ! -d "$APP_DIR/frontend/dist" ]]; then
  echo "Warning: frontend/dist not found - build locally and redeploy"
fi

# systemd units - fix paths in service files
for f in "$APP_DIR/deploy/systemd/"*.service; do
  [ -f "$f" ] || continue
  sed "s|/opt/mobius-platform/mobius-rag|$APP_DIR|g" "$f" > /etc/systemd/system/$(basename "$f")
done
systemctl daemon-reload
systemctl enable mobius-rag-backend mobius-rag-chunking-worker mobius-rag-embedding-worker
systemctl restart mobius-rag-backend mobius-rag-chunking-worker mobius-rag-embedding-worker 2>/dev/null || \
  systemctl start mobius-rag-backend mobius-rag-chunking-worker mobius-rag-embedding-worker

# Nginx
cp "$APP_DIR/deploy/nginx-mobius-rag.conf" /etc/nginx/sites-available/mobius-rag
sed -i "s|/opt/mobius-platform/mobius-rag|$APP_DIR|g" /etc/nginx/sites-available/mobius-rag
ln -sf /etc/nginx/sites-available/mobius-rag /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default 2>/dev/null || true
nginx -t && systemctl reload nginx

echo "=== Setup complete ==="
