// Platform single-sign-on token plumbing.
//
// When a signed-in platform user opens this app from chat (or the hub), the
// launcher appends the mobius-os access token to the URL fragment as
// `#t=<jwt>`. The fragment is never sent to servers nor written to access
// logs, so it is a safe one-time carrier. We read it on first load, stash it
// in sessionStorage (per-tab, cleared when the tab closes), and immediately
// strip the hash so the token does not linger in the address bar / history.
//
// The token is then (a) sent as `Authorization: Bearer …` on API calls and
// (b) forwarded onward to tools this app launches (e.g. Lexicon Maintenance),
// so the chain chat → RAG → lexicon shares one login.

const STORAGE_KEY = 'mobius_platform_token'

/** Read `#t=` / `#token=` / `#access_token=` from the URL hash, persist it,
 *  and strip the fragment. Safe to call multiple times; no-op without a hash. */
export function capturePlatformToken(): void {
  try {
    const hash = window.location.hash || ''
    if (!hash || hash.length < 2) return
    const params = new URLSearchParams(hash.replace(/^#/, ''))
    const tok = params.get('t') || params.get('token') || params.get('access_token')
    if (tok) {
      sessionStorage.setItem(STORAGE_KEY, tok)
      // Strip the hash without reloading or pushing a history entry.
      const clean = window.location.pathname + window.location.search
      window.history.replaceState(null, '', clean)
    }
  } catch {
    /* sessionStorage / history unavailable — ignore */
  }
}

/** The captured platform token, or '' if none. */
export function getPlatformToken(): string {
  try {
    return sessionStorage.getItem(STORAGE_KEY) || ''
  } catch {
    return ''
  }
}

/** Append the platform token as a `#t=` fragment to a Mobius-internal URL so
 *  the next tool inherits the login. Returns the URL unchanged if no token. */
export function withPlatformToken(url: string): string {
  const tok = getPlatformToken()
  if (!url || !tok) return url
  return url + (url.includes('#') ? '&' : '#') + 't=' + encodeURIComponent(tok)
}

/** Install a one-time fetch shim that attaches `Authorization: Bearer <token>`
 *  to SAME-ORIGIN requests (the RAG API) when a platform token is present and
 *  the caller hasn't set its own Authorization. This authenticates the app's
 *  existing bare-`fetch` admin calls without editing each call site, and never
 *  leaks the token to cross-origin hosts. No-op when there is no token. */
export function installAuthFetch(): void {
  const tok = getPlatformToken()
  if (!tok || typeof window === 'undefined' || !window.fetch) return
  const orig = window.fetch.bind(window)
  window.fetch = (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    try {
      const url =
        typeof input === 'string' ? input
        : input instanceof URL ? input.href
        : (input as Request).url
      // Same-origin only: a leading '/' (relative) or our own origin. Anything
      // absolute to another host (scraper API, external pages) is left alone.
      const sameOrigin =
        url.startsWith('/') ||
        url.startsWith(window.location.origin)
      if (sameOrigin) {
        const headers = new Headers(
          init?.headers ?? (input instanceof Request ? input.headers : undefined),
        )
        if (!headers.has('Authorization')) {
          headers.set('Authorization', `Bearer ${tok}`)
          init = { ...init, headers }
        }
      }
    } catch {
      /* fall through to the original fetch unmodified */
    }
    return orig(input as RequestInfo, init)
  }
}
