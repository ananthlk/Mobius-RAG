#!/bin/bash
# Google Drive OAuth setup for RAG
# Run: ./scripts/setup_drive_oauth.sh

set -e
PROJECT="${GCP_PROJECT:-mobius-os-dev}"
REDIRECT_URI="http://localhost:8001/drive/callback"

echo "=== Google Drive OAuth Setup ==="
echo ""
echo "1. Enabling Drive API..."
gcloud services enable drive.googleapis.com --project="$PROJECT"

echo ""
echo "2. Opening Credentials page in browser..."
echo "   Create OAuth 2.0 Client ID (Web application)"
echo "   Add redirect URI: $REDIRECT_URI"
echo ""
open "https://console.cloud.google.com/apis/credentials?project=$PROJECT" 2>/dev/null || \
  echo "   Open: https://console.cloud.google.com/apis/credentials?project=$PROJECT"

echo ""
echo "3. In the Console:"
echo "   - Click 'Create Credentials' -> 'OAuth client ID'"
echo "   - Application type: Web application"
echo "   - Name: Mobius RAG Drive Import"
echo "   - Authorized redirect URIs: $REDIRECT_URI"
echo "   - Click Create"
echo ""
echo "4. Copy the Client ID and Client Secret, then add to mobius-rag/.env:"
echo ""
echo "   GOOGLE_DRIVE_CLIENT_ID=your-client-id.apps.googleusercontent.com"
echo "   GOOGLE_DRIVE_CLIENT_SECRET=your-client-secret"
echo ""
echo "5. Restart the RAG backend."
