import { useState, useEffect, useCallback } from 'react'
import { API_BASE } from '../../config'
import { SourceTreeView } from './sources/SourceTreeView'
import { AddSourceDialog } from './sources/AddSourceDialog'
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

interface HostStat { host: string; count: number; corpusDocs: number; corpusPublished: number }
interface StatsResponse {
  by_host: Record<string, number>
  by_curation_status: Record<string, number>
  by_content_kind: Record<string, number>
  by_ingested: Record<string, number>
}
interface CorpusByHost {
  by_host: Record<string, { docs: number; published: number }>
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
  // Auto-publish toggle — when on, ingest waits for chunk+embed+publish
  // and the doc becomes queryable in chat in one button-press. Default
  // ON because that's the operator's mental model: "ingest" should
  // mean "live in the corpus", not "queued for processing".
  const [autoPublish, setAutoPublish] = useState(true)

  // "+ Add new source" wizard
  const [addDialogOpen, setAddDialogOpen] = useState(false)
  const [refreshTick, setRefreshTick] = useState(0)

  // Filters for the tree view
  const [filterStatus, setFilterStatus] = useState<'all' | 'not_indexed' | 'indexed' | 'blocked'>('all')
  const [filterSearch, setFilterSearch] = useState('')

  // Free-text search (List view)
  const [query, setQuery] = useState('')
  const [searchResults, setSearchResults] = useState<SourceRow[]>([])
  const [searchLoading, setSearchLoading] = useState(false)


  // ── Load aggregate stats + corpus counts on mount ──────────────
  useEffect(() => {
    let cancelled = false
    setStatsLoading(true)
    Promise.all([
      fetch(`${API_BASE}/sources/stats`).then(r => r.json() as Promise<StatsResponse>),
      fetch(`${API_BASE}/sources/corpus_by_host`).then(r => r.json() as Promise<CorpusByHost>)
        .catch(() => ({ by_host: {} })),  // back-compat: old rag deploys lack this endpoint
    ])
      .then(([s, c]) => {
        if (cancelled) return
        const corpusByHost = c.by_host || {}
        // Build host list: every host that appears in registry OR corpus.
        const allHosts = new Set<string>([
          ...Object.keys(s.by_host || {}),
          ...Object.keys(corpusByHost),
        ])
        const hostList: HostStat[] = Array.from(allHosts)
          .filter(h => !h.includes('test.example.com') && h !== '(no_host)')
          .map(host => ({
            host,
            count: s.by_host?.[host] ?? 0,
            corpusDocs: corpusByHost[host]?.docs ?? 0,
            corpusPublished: corpusByHost[host]?.published ?? 0,
          }))
          .sort((a, b) => (b.count + b.corpusDocs) - (a.count + a.corpusDocs))
        setHosts(hostList)
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
  }, [refreshTick])


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
        body: JSON.stringify({
          url,
          auto_publish: autoPublish,
          auto_publish_timeout_s: 180,
        }),
      })
      const ok = resp.ok || resp.status === 409   // 409 = already imported
      const body = await resp.json().catch(() => ({}))
      const docId = (body as {document_id?: string}).document_id
      const pub = (body as {auto_publish?: {ok: boolean; rows_written?: number; reason?: string}}).auto_publish
      let msg: string
      if (resp.status === 409) {
        msg = 'Already in corpus'
      } else if (ok) {
        if (autoPublish && pub) {
          msg = pub.ok
            ? `✓ Live in chat (${pub.rows_written} chunks published)`
            : `Imported but publish ${pub.reason || 'failed'}`
        } else {
          msg = `Imported (doc=${docId?.slice(0, 8)}, queued for chunking)`
        }
      } else {
        msg = `Failed: HTTP ${resp.status}`
      }
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
  }, [ingestingUrls, autoPublish])

  // ── Bulk ingest all non-indexed URLs in a sub-tree ─────────────
  // Sequential, not parallel — chunking workers are single-instance,
  // hammering them in parallel just queues anyway.
  const handleBulkIngest = useCallback(async (urls: string[]) => {
    for (const url of urls) {
      if (ingestingUrls.has(url)) continue
      // Reuse handleIngest's promise; it manages state correctly.
      // eslint-disable-next-line no-await-in-loop
      await handleIngest(url)
    }
  }, [handleIngest, ingestingUrls])


  // ── Render ─────────────────────────────────────────────────────
  return (
    <div className="sources-tab">
      <div className="sources-header">
        <h2>Sources Registry</h2>
        <div className="header-actions">
          <button
            className="add-source-btn"
            onClick={() => setAddDialogOpen(true)}
          >
            + Add new source
          </button>
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
      </div>

      {addDialogOpen && (
        <AddSourceDialog
          onClose={() => setAddDialogOpen(false)}
          onAdded={() => setRefreshTick(t => t + 1)}
        />
      )}

      {statsError && (
        <div className="error-banner">Failed to load stats: {statsError}</div>
      )}

      {/* ── Tree view ─────────────────────────────────────────── */}
      {view === 'tree' && (
        <div className="view-tree">
          <div className="toolbar">
            <div className="toolbar-left">
              <label>Entity:&nbsp;</label>
              <select
                value={selectedHost}
                onChange={e => setSelectedHost(e.target.value)}
                disabled={statsLoading}
              >
                {hosts.map(({ host, count, corpusDocs }) => (
                  <option key={host} value={host}>
                    {host} — {count} URLs · {corpusDocs} docs in corpus
                  </option>
                ))}
              </select>
            </div>
            <div className="toolbar-right">
              <label className="autopub-toggle" title="When on, ingest waits for chunk + embed + publish so the doc is queryable in chat in one step. When off, ingest just queues processing — doc lands in chat after manual /publish.">
                <input
                  type="checkbox"
                  checked={autoPublish}
                  onChange={e => setAutoPublish(e.target.checked)}
                />
                <span>Auto-publish (live in chat in one step)</span>
              </label>
            </div>
          </div>

          {/* Corpus banner — shows what's already indexed even when
              registry shows 0% (registry covers sitemap URLs; corpus
              has docs from other paths like /content/dam/*). */}
          {selectedHost && (() => {
            const hostStat = hosts.find(h => h.host === selectedHost)
            if (!hostStat) return null
            const { corpusDocs, corpusPublished } = hostStat
            if (corpusDocs === 0) return null
            return (
              <div className="corpus-banner">
                <strong>{corpusDocs}</strong> docs from this host already in corpus
                {' · '}
                <strong>{corpusPublished}</strong> published &amp; queryable in chat
                {corpusDocs > corpusPublished && (
                  <span className="corpus-banner-warn">
                    {' '}({corpusDocs - corpusPublished} embedded but not yet published)
                  </span>
                )}
              </div>
            )
          })()}

          {/* Filters row — narrow the tree by status + free-text */}
          {!hostLoading && hostRows.length > 0 && (
            <div className="filters-row">
              <label>Show:</label>
              {([
                ['all', 'All'],
                ['not_indexed', '○ Not indexed'],
                ['indexed', '✓ Indexed'],
                ['blocked', '⊘ Blocked / stale'],
              ] as const).map(([val, label]) => (
                <button
                  key={val}
                  className={`filter-pill ${filterStatus === val ? 'selected' : ''}`}
                  onClick={() => setFilterStatus(val)}
                >
                  {label}
                </button>
              ))}
              <input
                type="text"
                className="filter-search"
                placeholder="Filter URLs by substring…"
                value={filterSearch}
                onChange={e => setFilterSearch(e.target.value)}
              />
              {(() => {
                // Compute non-indexed urls in current filtered set
                // for the "Ingest all" button label.
                const filtered = filteredRows(hostRows, filterStatus, filterSearch)
                const ingestable = filtered
                  .filter(r => !r.ingested && (r.last_fetch_status ?? 0) < 400)
                  .map(r => r.url)
                if (ingestable.length < 2) return null
                return (
                  <button
                    className="ingest-all-btn"
                    onClick={() => handleBulkIngest(ingestable)}
                    title={`Ingest all ${ingestable.length} non-indexed URLs visible after filters`}
                  >
                    ▶ Ingest all {ingestable.length}
                  </button>
                )
              })()}
            </div>
          )}

          {hostLoading && <div className="loading">Loading tree…</div>}
          {!hostLoading && selectedHost && hostRows.length === 0 && (
            <div className="empty">
              No URLs in registry for <code>{selectedHost}</code>.
              {hosts.find(h => h.host === selectedHost)?.corpusDocs ? (
                <> But {hosts.find(h => h.host === selectedHost)?.corpusDocs} docs from this host are in the corpus (came in via paths the sitemap doesn't enumerate).</>
              ) : null}
            </div>
          )}
          {!hostLoading && hostRows.length > 0 && (
            <SourceTreeView
              host={selectedHost}
              payerLabel={hostRows[0]?.payer || null}
              rows={filteredRows(hostRows, filterStatus, filterSearch)}
              onIngest={handleIngest}
              onBulkIngest={handleBulkIngest}
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

      {/* (helper defined below) */}
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


/* ── Filtering helper ────────────────────────────────────── */

function filteredRows(
  rows: SourceRow[],
  status: 'all' | 'not_indexed' | 'indexed' | 'blocked',
  search: string,
): SourceRow[] {
  const q = search.trim().toLowerCase()
  return rows.filter(r => {
    if (status === 'indexed' && !r.ingested) return false
    if (status === 'not_indexed' && r.ingested) return false
    const fetchStatus = r.last_fetch_status ?? 0
    const blocked = fetchStatus >= 400
    if (status === 'blocked' && !blocked) return false
    if (status === 'not_indexed' && blocked) return false  // hide blocked from ingestable view
    if (q && !r.url.toLowerCase().includes(q)) return false
    return true
  })
}
