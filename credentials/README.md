# GCP credentials (self-contained)

Copy your GCP service account JSON key **into this folder** so mobius-rag does not reference paths outside this repo.

- Copy your key file here, e.g. `gcp-service-account.json` or `mobiusos-new-090a058b63d9.json`.  
  If it is already in the mobius-rag repo root: `mv ../mobiusos-new-090a058b63d9.json .`
- In `mobius-rag/.env` set (path relative to repo root):
  ```bash
  GOOGLE_APPLICATION_CREDENTIALS=credentials/mobiusos-new-090a058b63d9.json
  ```
  Use the actual filename you have in this folder.
- Do **not** commit key files. This folder is gitignored except this README.
