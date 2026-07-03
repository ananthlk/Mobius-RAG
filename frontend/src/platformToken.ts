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

/** Persist a token to sessionStorage. */
function setPlatformToken(tok: string): void {
  try { if (tok) sessionStorage.setItem(STORAGE_KEY, tok) } catch { /* ignore */ }
}

// DEV convenience: when set (by the dev deploy's VITE_DEV_MINT_URL build arg),
// the app can mint a platform token itself so a long dev session doesn't 401
// once the launcher token expires. Unset in prod → auto-mint never happens.
const DEV_MINT_URL = (import.meta.env?.VITE_DEV_MINT_URL as string | undefined) || ''

/** Mint a fresh dev token from the configured mint endpoint (dev only). */
async function mintDevToken(): Promise<string> {
  if (!DEV_MINT_URL) return ''
  try {
    const r = await fetch(DEV_MINT_URL, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
    })
    if (!r.ok) return ''
    const tok = ((await r.json()) as { access_token?: string }).access_token || ''
    if (tok) setPlatformToken(tok)
    return tok
  } catch { return '' }
}

/** Ensure a token exists at startup — mint one in dev if the tab has none. */
export async function ensureDevToken(): Promise<void> {
  if (getPlatformToken()) return
  await mintDevToken()
}

/** Install a fetch shim that attaches `Authorization: Bearer <token>` to
 *  SAME-ORIGIN requests (the RAG API) when a platform token is present and the
 *  caller hasn't set its own Authorization — authenticating the app's bare
 *  `fetch` admin calls without editing each call site, never leaking the token
 *  cross-origin. Reads the token DYNAMICALLY each call (so a freshly-minted
 *  token is picked up), and in dev re-mints + retries once on a 401. */
export function installAuthFetch(): void {
  if (typeof window === 'undefined' || !window.fetch) return
  const orig = window.fetch.bind(window)
  const urlOf = (input: RequestInfo | URL): string =>
    typeof input === 'string' ? input : input instanceof URL ? input.href : (input as Request).url
  const withAuth = (tok: string, base?: RequestInit, input?: RequestInfo | URL): RequestInit => {
    const headers = new Headers(base?.headers ?? (input instanceof Request ? input.headers : undefined))
    if (tok && !headers.has('Authorization')) headers.set('Authorization', `Bearer ${tok}`)
    return { ...base, headers }
  }
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    let sameOrigin = false
    try {
      const url = urlOf(input)
      sameOrigin = url.startsWith('/') || url.startsWith(window.location.origin)
    } catch { /* leave sameOrigin false */ }
    if (!sameOrigin) return orig(input as RequestInfo, init)

    const tok = getPlatformToken()
    let res = await orig(input as RequestInfo, tok ? withAuth(tok, init, input) : init)
    // Dev: an /admin/* call 401'd → token missing/expired → mint fresh + retry
    // once. Scoped to /admin/ so the mint call itself can't recurse.
    if (res.status === 401 && DEV_MINT_URL && urlOf(input).includes('/admin/')) {
      const fresh = await mintDevToken()
      if (fresh) res = await orig(input as RequestInfo, withAuth(fresh, init, input))
    }
    return res
  }
}
