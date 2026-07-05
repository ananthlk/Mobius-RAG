import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Force a single React copy across the bundle. @mobius/document-viewer
  // has React listed as a peer dep but historically ended up with its own
  // node_modules/react that gets picked up by node resolution, producing
  // two React copies in the prod bundle. The doc-viewer's hooks then run
  // against a React with a null dispatcher → "Cannot read properties of
  // null (reading 'useState')" the moment a slide-out reader mounts.
  resolve: {
    dedupe: ['react', 'react-dom'],
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    // Proxy RAG backend in dev so the browser makes same-origin requests (no CORS).
    // Override the target with RAG_PROXY_TARGET to point dev at a deployed
    // backend (e.g. the Cloud Run dev service) instead of a local :8001.
    proxy: {
      '/api-rag': {
        target: process.env.RAG_PROXY_TARGET || 'http://127.0.0.1:8001',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/api-rag/, ''),
      },
    },
  },
})
