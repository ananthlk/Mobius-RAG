#!/usr/bin/env bash
# Deploy Mobius RAG to mobius-platform-vm.
# Run from your local machine. Copies project to VM and runs setup.
#
# Usage:
#   DATABASE_PASSWORD=xxx ./deploy/deploy_to_vm.sh
#   DATABASE_PASSWORD=xxx REPO_URL=https://github.com/you/mobius-rag.git ./deploy/deploy_to_vm.sh
#
set -e

PROJECT_ID="${GCP_PROJECT_ID:-mobiusos-new}"
ZONE="${GCP_ZONE:-us-central1-a}"
VM_NAME="${GCP_VM_NAME:-mobius-platform-vm}"
CLOUD_SQL_IP="${CLOUD_SQL_IP:-34.59.175.121}"
REPO_URL="${REPO_URL:-}"

if [[ -z "$DATABASE_PASSWORD" ]]; then
  echo "Set DATABASE_PASSWORD"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Deploy Mobius RAG to $VM_NAME ==="

# Build frontend locally (VM may have old Node)
if [[ -d "$PROJECT_ROOT/frontend" ]]; then
  echo "Building frontend..."
  (cd "$PROJECT_ROOT/frontend" && npm ci 2>/dev/null || npm install && VITE_API_BASE= npm run build)
fi

# Create tarball (exclude .venv, node_modules except dist, __pycache__, .git)
TARBALL="/tmp/mobius-rag-deploy.tar.gz"
tar -czf "$TARBALL" -C "$PROJECT_ROOT" \
  --exclude='.venv' --exclude='node_modules' --exclude='__pycache__' \
  --exclude='.git' --exclude='*.pyc' --exclude='.env' \
  .

# Upload and extract on VM
gcloud compute scp --zone="$ZONE" --project="$PROJECT_ID" "$TARBALL" "$VM_NAME:/tmp/"
rm -f "$TARBALL"

# Run setup on VM
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT_ID" --command="
set -e
sudo mkdir -p /opt/mobius-platform/mobius-rag
sudo tar -xzf /tmp/mobius-rag-deploy.tar.gz -C /opt/mobius-platform/mobius-rag
sudo useradd -m mobius 2>/dev/null || true
sudo chown -R mobius:mobius /opt/mobius-platform
cd /opt/mobius-platform/mobius-rag
sudo DATABASE_PASSWORD='$DATABASE_PASSWORD' CLOUD_SQL_IP='$CLOUD_SQL_IP' PROJECT_ID='$PROJECT_ID' bash deploy/vm_setup.sh
"

echo ""
echo "=== Deploy complete ==="
VM_IP=$(gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --project="$PROJECT_ID" --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null)
echo "Frontend: http://${VM_IP:-<VM_IP>}/"
echo "Backend health: http://${VM_IP:-<VM_IP>}/health"
