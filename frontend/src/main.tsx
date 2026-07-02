import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { capturePlatformToken, installAuthFetch } from './platformToken'

// Grab the platform SSO token from the URL fragment (if launched from chat/hub)
// before the app renders, so the first links already carry it, then install the
// fetch shim so the app's API calls authenticate with it.
capturePlatformToken()
installAuthFetch()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
