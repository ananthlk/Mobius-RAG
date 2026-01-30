# Where to store credentials

**Do not store API keys, secrets, or service account keys in PostgreSQL.** The app resolves LLM and cloud credentials at runtime from environment variables or a secret manager. Storing secrets in the database is a security risk (backups, logs, access scope).

## Recommended: environment variables

Set credentials via environment variables so they are never committed to the repo.

- **Local / single host:** Use a `.env` file (must be in `.gitignore`; see `.env.example`). Load with `python-dotenv` (already used by the app).
- **Production / containers:** Set env vars in your deployment (e.g. Kubernetes secrets, Cloud Run env, GitHub Actions secrets). The app reads them at startup.

### LLM and cloud env vars

| Purpose | Variable | Notes |
|--------|----------|--------|
| **Vertex AI (Gemini)** | `VERTEX_PROJECT_ID` | Required for `provider: vertex`. |
| | `VERTEX_LOCATION` | Optional; default `us-central1`. |
| | `VERTEX_MODEL` | Optional; default `gemini-1.5-pro`. |
| | `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON, or use ADC. |
| **OpenAI** | `OPENAI_API_KEY` | Required for `provider: openai`. |
| **Ollama** | `OLLAMA_BASE_URL` | Optional; default `http://localhost:11434`. |
| **Database** | `DATABASE_URL` | Connection string; keep credentials here, not in code. |
| **GCS** | `GCS_BUCKET` | Bucket name; auth via ADC or `GOOGLE_APPLICATION_CREDENTIALS`. |

The app uses these when building LLM providers: config (YAML or DB) can specify provider and model, but **secrets are always resolved from env** (or a secret manager if you add one). The LLM Config UI does not persist real API keys to the database; it stores a placeholder and the app reads the actual key from the environment.

## Optional: secret manager

For production you can centralize secrets in a secret manager and inject them into the process as env vars, or resolve them in code.

- **Google Cloud:** [Secret Manager](https://cloud.google.com/secret-manager). At startup (or when building a provider), fetch the secret by name and set `OPENAI_API_KEY` or pass it into the provider. Alternatively run your app on GKE/Cloud Run with [Secret Manager integration](https://cloud.google.com/secret-manager/docs/access-secret-version) so secrets are mounted or set as env.
- **AWS:** [Secrets Manager](https://aws.amazon.com/secrets-manager/) or SSM Parameter Store; same idea: fetch at startup or via deployment and set env.
- **HashiCorp Vault:** Use the Vault agent or your deployment to populate env vars from Vault.

The app does not implement secret-manager calls by default; it reads only from the environment. You can add a small helper (e.g. `get_secret(name)` that checks env then falls back to Secret Manager) and use it in the LLM provider factories if you want.

## What is stored where

- **PostgreSQL:** Non-secret LLM config (provider name, model id, base URLs, project id as identifier). **No API keys or secrets.** If the UI or API sends an API key when saving config, the backend replaces it with a placeholder before writing to the DB.
- **YAML files (`app/llm_configs/`):** Same rule. Use `null` or placeholders for secrets and set the real values via env.
- **Environment / secret manager:** All secrets (API keys, DB password, service account path or JSON).
