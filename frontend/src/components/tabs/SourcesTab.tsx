import { useState, useEffect, useCallback } from 'react'
import { API_BASE } from '../../config'
import { SourceTreeView } from './sources/SourceTreeView'
import type { SourceRow } from './sources/treeBuilder'
import './SourcesTab.css'


/**
 * Curator UI — browse the discovered_sources registry.
 *
 * Two views:
 *   - Tree: hierarchical per-entity (host) view. Coverage stats,
 *           biggest gaps, click-to-ingest on non-indexed URLs.
 *   - List: flat BM25 search across all sources.
 *
 * Backend already exists (Phase 13.2/13.3): /sources/search,
 * /sources/stats, /documents/import-from-html. This tab just renders.
 *
 * 2026-04-26 v0.1 — read + ingest. Curate / add new source come next.
 */

interface HostStat { host: string; count: number }
interface StatsResponse {
  by_host: Record<string, number>
  by_curation_status: Record<string, number>
  by_content_kind: Record<string, number>
  by_ingested: Record<string, number>
}


export function SourcesTab() {
  const [view, setView] = useState<'tree' | 'list'>('tree')

  // Stats: drives the entity dropdown
  const [hosts, setHosts] = useState<HostStat[]>([])
  const [statsLoading, setStatsLoading] = useState(true)
  const [statsError, setStatsError] = useState<string | null>(null)

  // Selected entity for tree view
  const [selectedHost, setSelectedHost] = useState<string>('')
  const [hostRows, setHostRows] = useState<SourceRow[]>([])
  const [hostLoading, setHostLoading] = useState(false)

  // Ingest in flight
  const [ingestingUrls, setIngestingUrls] = useState<Set<string>>(new Set())
  const [ingestLog, setIngestLog] = useState<{url: string; ok: boolean; msg: string}[]>([])

  // Free-text search (List view)
  const [query, setQuery] = useState('')
  const [searchResults, setSearchResults] = useState<SourceRow[]>([])
  const [searchLoading, setSearchLoading] = useState(false)


  // ── Load aggregate stats on mount ──────────────────────────────
  useEffect(() => {
    let cancelled = false
    setStatsLoading(true)
    fetch(`${API_BASE}/sources/stats`)
      .then(r => {
        if (!r.ok) throw new Error(`stats ${r.status}`)
        return r.json() as Promise<StatsResponse>
      })
      .then(s => {
        if (cancelled) return
        // Sort hosts by URL count desc; drop the test stub
        const hostList = Object.entries(s.by_host || {})
          .filter(([h]) => !h.includes('test.example.com'))
          .map(([host, count]) => ({ host, count }))
          .sort((a, b) => b.count - a.count)
        setHosts(hostList)
        // Auto-select biggest entity if none yet
        if (hostList.length > 0 && !selectedHost) {
          setSelectedHost(hostList[0].host)
        }
        setStatsLoading(false)
      })
      .catch(e => {
        if (cancelled) return
        setStatsError(String(e))
        setStatsLoading(false)
      })
    return () => { cancelled = true }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])


  // ── Load rows for selected host ────────────────────────────────
  useEffect(() => {
    if (!selectedHost || view !== 'tree') return
    let cancelled = false
    setHostLoading(true)
    // Server-side host filter (added 2026-04-26 — Phase 13.x). Earlier
    // version filtered client-side from a 500-row capped fetch which
    // couldn't surface all of SAMHSA's 1000 URLs.
    fetch(`${API_BASE}/sources/search?host=${encodeURIComponent(selectedHost)}&only_reachable=false&limit=500`)
      .then(r => r.json() as Promise<SourceRow[]>)
      .then(rows => {
        if (cancelled) return
        setHostRows(rows)
        setHostLoading(false)
      })
      .catch(_e => {
        if (cancelled) return
        setHostRows([])
        setHostLoading(false)
      })
    return () => { cancelled = true }
  }, [selectedHost, view])


  // ── Free-text search (List view) ───────────────────────────────
  const runSearch = useCallback(async (q: string) => {
    if (!q.trim()) {
      setSearchResults([])
      return
    }
    setSearchLoading(true)
    try {
      const r = await fetch(
        `${API_BASE}/sources/search?q=${encodeURIComponent(q)}&limit=50`
      )
      const rows = (await r.json()) as SourceRow[]
      setSearchResults(rows)
    } catch (_e) {
      setSearchResults([])
    } finally {
      setSearchLoading(false)
    }
  }, [])


  // ── Trigger ingest_url for one URL ─────────────────────────────
  const handleIngest = useCallback(async (url: string) => {
    if (ingestingUrls.has(url)) return
    setIngestingUrls(prev => new Set(prev).add(url))
    try {
      const resp = await fetch(`${API_BASE}/documents/import-from-html`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      })
      const ok = resp.ok || resp.status === 409   // 409 = already imported
      const body = await resp.json().catch(() => ({}))
      const msg = ok
        ? (resp.status === 409
            ? 'Already in corpus'
            : `Imported (doc=${(body as {document_id?: string}).document_id?.slice(0, 8)})`)
        : `Failed: HTTP ${resp.status}`
      setIngestLog(log => [{ url, ok, msg }, ...log].slice(0, 10))
      // Optimistic: mark this row as ingested in the tree
      setHostRows(rows => rows.map(r =>
        r.url === url ? { ...r, ingested: true } : r
      ))
    } catch (e) {
      setIngestLog(log => [{ url, ok: false, msg: String(e) }, ...log].slice(0, 10))
    } finally {
      setIngestingUrls(prev => {
        const next = new Set(prev)
        next.delete(url)
        return next
      })
    }
  }, [ingestingUrls])


  // ── Render ─────────────────────────────────────────────────────
  return (
    <div className="sources-tab">
      <div className="sources-header">
        <h2>Sources Registry</h2>
        <div className="view-toggle">
          <button
            className={view === 'tree' ? 'view-btn-active' : 'view-btn'}
            onClick={() => setView('tree')}
          >
            Tree view
          </button>
          <button
            className={view === 'list' ? 'view-btn-active' : 'view-btn'}
            onClick={() => setView('list')}
          >
            Search (BM25)
          </button>
        </div>
      </div>

      {statsError && (
        <div className="error-banner">Failed to load stats: {statsError}</div>
      )}

      {/* ── Tree view ─────────────────────────────────────────── */}
      {view === 'tree' && (
        <div className="view-tree">
          <div className="entity-selector">
            <label>Entity:&nbsp;</label>
            <select
              value={selectedHost}
              onChange={e => setSelectedHost(e.target.value)}
              disabled={statsLoading}
            >
              {hosts.map(({ host, count }) => (
                <option key={host} value={host}>
                  {host} ({count} URLs)
                </option>
              ))}
            </select>
          </div>

          {hostLoading && <div className="loading">Loading tree…</div>}
          {!hostLoading && selectedHost && hostRows.length === 0 && (
            <div className="empty">No rows for {selectedHost} (yet).</div>
          )}
          {!hostLoading && hostRows.length > 0 && (
            <SourceTreeView
              host={selectedHost}
              payerLabel={hostRows[0]?.payer || null}
              rows={hostRows}
              onIngest={handleIngest}
              ingestingUrls={ingestingUrls}
            />
          )}
        </div>
      )}

      {/* ── List / search view ───────────────────────────────── */}
      {view === 'list' && (
        <div className="view-list">
          <div className="search-bar">
            <input
              type="text"
              placeholder="Free-text search (BM25 over payer/host/path/authority)…"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') runSearch(query) }}
            />
            <button onClick={() => runSearch(query)} disabled={searchLoading}>
              {searchLoading ? 'Searching…' : 'Search'}
            </button>
          </div>
          <div className="search-results">
            {searchResults.length === 0 && !searchLoading && (
              <div className="empty">
                {query ? 'No matches.' : 'Enter a query to search the registry.'}
              </div>
            )}
            {searchResults.map(row => (
              <div
                key={row.id}
                className={`search-row ${row.ingested ? 'leaf-indexed' : 'leaf-not-indexed'}`}
              >
                <span className="leaf-marker">
                  {row.ingested ? '✓' : '○'}
                </span>
                <a href={row.url} target="_blank" rel="noopener noreferrer" className="leaf-url">
                  {row.url}
                </a>
                <span className="leaf-auth">{row.payer || '—'}</span>
                {!row.ingested && (
                  <button
                    className="leaf-ingest-btn"
                    onClick={() => handleIngest(row.url)}
                    disabled={ingestingUrls.has(row.url)}
                  >
                    {ingestingUrls.has(row.url) ? 'Ingesting…' : 'Ingest →'}
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Recent ingest log ────────────────────────────────── */}
      {ingestLog.length > 0 && (
        <div className="ingest-log">
          <h4>Recent ingests</h4>
          {ingestLog.map((e, i) => (
            <div key={i} className={e.ok ? 'log-ok' : 'log-err'}>
              <span className="log-icon">{e.ok ? '✓' : '✗'}</span>
              <span className="log-msg">{e.msg}</span>
              <span className="log-url">{e.url}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
