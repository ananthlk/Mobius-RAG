/**
 * RAG API base URL (documents, chunking, embedding). In production, set VITE_API_BASE at build time.
 * - VITE_API_BASE="" (or unset): in dev use /api-rag (proxied to 8001, no CORS); in prod use http://localhost:8001
 * - VITE_API_BASE=https://api.example.com: custom API URL
 */
const envBase = import.meta.env?.VITE_API_BASE as string | undefined
const isDev = import.meta.env?.DEV === true
export const API_BASE: string =
  envBase !== undefined ? envBase : isDev ? '/api-rag' : 'http://localhost:8001'

/**
 * Web scraper API base URL. For "Scrape from URL" in Document Input.
 * - VITE_SCRAPER_API_BASE: e.g. http://localhost:8002
 */
const scraperBase = import.meta.env?.VITE_SCRAPER_API_BASE as string | undefined
export const SCRAPER_API_BASE: string =
  scraperBase !== undefined ? scraperBase : 'http://localhost:8002'
