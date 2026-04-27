/**
 * RAG API base URL (documents, chunking, embedding).
 *
 * Resolution order:
 *   1. VITE_API_BASE at build time   — explicit override
 *   2. dev server (vite dev)          — /api-rag (proxied to :8001, no CORS)
 *   3. production fallback            — ""  (relative to origin)
 *
 * The prod fallback was previously ``http://localhost:8001``, which
 * silently sent every hosted request to the dev laptop ("Failed to
 * fetch Is the backend running at http://localhost:8001?"). The
 * backend mounts the SPA at ``/``, so same-origin relative paths
 * are correct in prod.
 */
const envBase = import.meta.env?.VITE_API_BASE as string | undefined
const isDev = import.meta.env?.DEV === true
export const API_BASE: string =
  envBase !== undefined ? envBase : isDev ? '/api-rag' : ''

/**
 * Web scraper API base URL. For "Scrape from URL" in Document Input.
 * - VITE_SCRAPER_API_BASE: e.g. http://localhost:8002
 */
// Prefer build-arg `VITE_SCRAPER_API_BASE`. Fallback is empty string in
// hosted mode — same anti-localhost pattern as ``API_BASE`` (commit
// c49407d). Hosted deploys must pass the scraper URL at Docker build
// (Cloud Build --build-arg) since scraper is on a separate Cloud Run
// service, not same-origin. Local dev hits :8002 via vite proxy or
// explicit env.
const scraperBase = import.meta.env?.VITE_SCRAPER_API_BASE as string | undefined
// (``isDev`` already declared above for API_BASE — reuse it.)
export const SCRAPER_API_BASE: string =
  scraperBase !== undefined ? scraperBase : isDev ? 'http://localhost:8002' : ''
