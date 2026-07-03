import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { capturePlatformToken, installAuthFetch, ensureDevToken } from './platformToken'

// Grab the platform SSO token from the URL fragment (if launched from chat/hub)
// before the app renders, then install the fetch shim so API calls authenticate.
capturePlatformToken()
installAuthFetch()

// In dev, mint a token if the tab has none (long sessions outlive the launcher
// token). No-op in prod. Awaited so the first admin call already carries auth;
// resolves instantly when a token is already present.
ensureDevToken().finally(() => {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
})
