/**
 * API base URL for backend. In production, set VITE_API_BASE at build time.
 * - VITE_API_BASE="" (or unset): use relative URLs (same origin - for nginx proxy)
 * - VITE_API_BASE=https://api.example.com: custom API URL
 * - undefined (dev): default http://localhost:8000
 */
const envBase = import.meta.env?.VITE_API_BASE as string | undefined
export const API_BASE: string =
  envBase !== undefined ? envBase : 'http://localhost:8000'
