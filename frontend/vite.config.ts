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
    // Proxy RAG backend in dev so the browser makes same-origin requests (no CORS)
    proxy: {
      '/api-rag': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api-rag/, ''),
      },
    },
  },
})
