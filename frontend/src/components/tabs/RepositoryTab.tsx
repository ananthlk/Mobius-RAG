import { useEffect, useMemo, useState } from 'react'
import { API_BASE } from '../../config'
import { EntityCard } from './repository/EntityCard'
import { ReaderSlideOut } from './repository/ReaderSlideOut'
import { EntitySidebar, domainOf } from './repository/EntitySidebar'
import type { HostStatEnriched, CorpusStats, DomainFilter } from './repository/EntitySidebar'
import { SearchTracePanel } from './repository/SearchTracePanel'
import type { SearchTelemetry } from './repository/SearchTracePanel'
import { UploadedDocsPanel } from './repository/UploadedDocsPanel'
import './RepositoryTab.css'

// suppress unused-import warnings — these are kept per spec
void EntityCard
void EntitySidebar
void domainOf
void UploadedDocsPanel

const UPLOADED_HOST = '(uploaded)'

// ── Types ─────────────────────────────────────────────────────────────────────

interface DocLike {
  id: string
  filename: string
  display_name?: string | null
  payer?: string | null
  state?: string | null
  extraction_status?: string
  chunking_status?: string | null
  embedding_status?: string | null
  published_at?: string | null
  source_metadata?: { source_url?: string | null } | null
  source_url?: string | null
}

interface NavigateToRead {
  documentId: string
  pageNumber?: number
  factId?: string
  citeText?: string
}

interface RawHostStat {
  host: string
  count: number
  corpusDocs: number
  corpusPublished: number
}

interface StatsResponse {
  by_host: Record<string, number>
  by_curation_status: Record<string, number>
  by_content_kind: Record<string, number>
  by_ingested: Record<string, number>
}

interface CorpusByHost {
  by_host: Record<string, { docs: number; published: number }>
}

/** Chunk returned by /api/skills/v1/corpus_search */
interface CorpusChunk {
  id: string
  text: string
  document_id: string
  document_name: string
  page_number: number | null
  paragraph_index: number | null
  source_type: string
  similarity: number
  rerank_score: number
  confidence_label: 'high' | 'medium' | 'low' | 'abstain'
  retrieval_arms: string[]
  authority_level: string | null
  payer: string | null
  state: string | null
}

interface CorpusSearchResponse {
  chunks: CorpusChunk[]
  telemetry: SearchTelemetry
}

type SearchMode = 'corpus' | 'precision' | 'recall'

interface Props {
  documents: DocLike[]
  documentsLoading?: boolean
  selectedDocumentId: string | null
  navigateToRead: NavigateToRead | null
  onNavigateToReadConsumed: () => void
  onDocumentSelect: (documentId: string) => void
  onRefresh?: () => void
}

// ── Corpus search results ─────────────────────────────────────────────────────

const CONFIDENCE_COLORS: Record<string, string> = {
  high: '#10b981',
  medium: '#f59e0b',
  low: '#6b7280',
  abstain: '#d1d5db',
}

const ARM_LABELS: Record<string, string> = {
  bm25: 'BM25',
  vector: 'Vector',
}

// Kept (referenced for type compatibility) but not rendered after the
// 2026-04-29 cleanup that moved corpus search to the Test tab.
// @ts-ignore
function _CorpusSearchResults({
  query,
  mode,
  chunks,
  loading,
  telemetry,
  onOpenChunk,
}: {
  query: string
  mode: SearchMode
  chunks: CorpusChunk[] | null
  loading: boolean
  telemetry: SearchTelemetry | null
  onOpenChunk: (chunk: CorpusChunk) => void
}) {
  const [showTrace, setShowTrace] = useState(false)

  if (loading) return <div className="repo-search-status">Searching corpus…</div>
  if (!chunks) return null
  if (chunks.length === 0) {
    return (
      <div className="repo-search-status">
        No results for &ldquo;{query}&rdquo;
        {telemetry && (
          <span className="repo-search-telem">
            {' '}({telemetry.total_ms?.toFixed(0) ?? '?'}ms)
          </span>
        )}
      </div>
    )
  }

  const modeLabel = mode === 'corpus' ? 'Hybrid' : mode === 'precision' ? 'BM25' : 'Semantic'
  const hasTrace = !!(
    telemetry?.scoring_trace?.length ||
    telemetry?.arm_results?.bm25?.length ||
    telemetry?.arm_results?.vector?.length
  )

  return (
    <div className="repo-corpus-results">
      <div className="repo-corpus-results-hdr">
        <span>
          {chunks.length} result{chunks.length !== 1 ? 's' : ''} for{' '}
          <strong>&ldquo;{query}&rdquo;</strong>
        </span>
        <span className="repo-corpus-results-meta">
          {modeLabel}
          {telemetry && (
            <> · {telemetry.total_ms?.toFixed(0) ?? '?'}ms</>
          )}
          {telemetry?.arm_hits && (
            <> · BM25 {telemetry.arm_hits.bm25 ?? 0} / Vec {telemetry.arm_hits.vector ?? 0}</>
          )}
        </span>
        {/* Trace toggle */}
        {hasTrace && (
          <button
            type="button"
            className={`repo-corpus-trace-btn${showTrace ? ' active' : ''}`}
            onClick={() => setShowTrace((v) => !v)}
            title={showTrace ? 'Hide pipeline trace' : 'Show pipeline trace'}
          >
            {showTrace ? '▲ Trace' : '▼ Trace'}
          </button>
        )}
      </div>

      <ul className="repo-corpus-results-list">
        {chunks.map((chunk) => (
          <li key={chunk.id} className="repo-corpus-result-item">
            <button
              type="button"
              className="repo-corpus-result-btn"
              onClick={() => onOpenChunk(chunk)}
            >
              {/* Header row */}
              <div className="repo-corpus-result-head">
                <span className="repo-corpus-result-docname">{chunk.document_name}</span>
                {chunk.page_number != null && (
                  <span className="repo-corpus-result-page">p.{chunk.page_number}</span>
                )}
                {/* Arms */}
                <span className="repo-corpus-result-arms">
                  {chunk.retrieval_arms.map((arm) => (
                    <span key={arm} className={`repo-corpus-arm-badge arm-${arm}`}>
                      {ARM_LABELS[arm] ?? arm}
                    </span>
                  ))}
                </span>
                {/* Confidence */}
                <span
                  className="repo-corpus-confidence"
                  style={{ color: CONFIDENCE_COLORS[chunk.confidence_label] }}
                  title={`rerank score: ${chunk.rerank_score.toFixed(3)}`}
                >
                  {chunk.confidence_label}
                </span>
              </div>
              {/* Snippet */}
              <p className="repo-corpus-result-snippet">
                {chunk.text.slice(0, 220)}{chunk.text.length > 220 ? '…' : ''}
              </p>
              {/* Meta */}
              {(chunk.payer || chunk.authority_level) && (
                <div className="repo-corpus-result-footer">
                  {chunk.payer && <span className="repo-corpus-result-payer">{chunk.payer}</span>}
                  {chunk.authority_level && (
                    <span className="repo-corpus-result-auth">{chunk.authority_level}</span>
                  )}
                </div>
              )}
            </button>
          </li>
        ))}
      </ul>

      {/* Pipeline trace panel — expandable */}
      {showTrace && telemetry && (
        <div className="repo-corpus-trace-wrap">
          <SearchTracePanel telemetry={telemetry} />
        </div>
      )}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export function RepositoryTab({
  documents,
  documentsLoading = false,
  selectedDocumentId,
  navigateToRead,
  onNavigateToReadConsumed,
  onDocumentSelect,
  onRefresh,
}: Props) {
  // ── Entity / host data ───────────────────────────────────────────────────
  const [rawHosts, setRawHosts] = useState<RawHostStat[]>([])
  const [statsLoading, setStatsLoading] = useState(true)
  const [selectedHost, setSelectedHost] = useState(UPLOADED_HOST)

  // ── Sidebar UI state ─────────────────────────────────────────────────────
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [entitySearch, setEntitySearch] = useState('')
  const [domainFilter, setDomainFilter] = useState<DomainFilter>('all')

  // ── Search state ─────────────────────────────────────────────────────────
  const [searchQuery, setSearchQuery] = useState('')
  const [searchMode, setSearchMode] = useState<SearchMode>('corpus')
  // Search moved to Test tab (2026-04-29). State + useEffect kept stubbed
  // so the rest of the file's references compile, but the UI is gone.
  const [, setSearchResults] = useState<CorpusChunk[] | null>(null)
  const [, setSearchTelemetry] = useState<SearchTelemetry | null>(null)
  const [, setSearchLoading] = useState(false)

  // ── Reader state ─────────────────────────────────────────────────────────
  const [readerVisible, setReaderVisible] = useState(false)
  const [urlPreview, setUrlPreview] = useState<{ url: string; ingested: boolean } | null>(null)
  const [ingestingUrl, setIngestingUrl] = useState(false)
  // Local navigate-to (from clicking a search result chunk)
  const [localNavigateTo, setLocalNavigateTo] = useState<NavigateToRead | null>(null)

  // Auto-open reader when an external navigateToRead arrives
  useEffect(() => {
    if (navigateToRead) setReaderVisible(true)
  }, [navigateToRead])

  // ── Three-panel layout state ─────────────────────────────────────────────
  // expandedPanel: which panel (if any) is occupying full width
  const [expandedPanel, setExpandedPanel] = useState<'browse' | 'reader' | 'health' | null>(null)
  // healthOpen: independent toggle for health panel height within its strip
  const [healthOpen, setHealthOpen] = useState(false)

  // ── Browse panel state ───────────────────────────────────────────────────
  const [browseSearch, setBrowseSearch] = useState('')
  const [payerFilter, setPayerFilter] = useState('')
  const [stateFilter, setStateFilter] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  // Track which payer groups are collapsed
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>({})

  // ── Load aggregate stats ─────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false
    setStatsLoading(true)
    Promise.all([
      fetch(`${API_BASE}/sources/stats`)
        .then((r) => (r.ok ? (r.json() as Promise<StatsResponse>) : Promise.reject(r.status)))
        .catch((e): StatsResponse => {
          console.warn('[Repository] /sources/stats failed:', e)
          return { by_host: {}, by_curation_status: {}, by_content_kind: {}, by_ingested: {} }
        }),
      fetch(`${API_BASE}/sources/corpus_by_host`)
        .then((r) => (r.ok ? (r.json() as Promise<CorpusByHost>) : Promise.reject(r.status)))
        .catch((e): CorpusByHost => {
          console.warn('[Repository] /sources/corpus_by_host failed:', e)
          return { by_host: {} }
        }),
    ])
      .then(([s, c]) => {
        if (cancelled) return
        const corpusByHost = c.by_host || {}
        const allHosts = new Set<string>([
          ...Object.keys(s.by_host || {}),
          ...Object.keys(corpusByHost),
        ])
        const hostList: RawHostStat[] = Array.from(allHosts)
          .filter((h) => !h.includes('test.example.com') && h !== '(no_host)')
          .map((host) => ({
            host,
            count: s.by_host?.[host] ?? 0,
            corpusDocs: corpusByHost[host]?.docs ?? 0,
            corpusPublished: corpusByHost[host]?.published ?? 0,
          }))
          .sort((a, b) => b.count + b.corpusDocs - (a.count + a.corpusDocs))
        setRawHosts(hostList)
        setStatsLoading(false)
      })
      .catch((e) => {
        if (cancelled) return
        console.warn('[Repository] stats fetch failed:', e)
        setStatsLoading(false)
      })
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── Uploaded docs: no web source URL ────────────────────────────────────
  const uploadedDocs = useMemo(() => {
    return documents.filter((d) => {
      const url = d.source_metadata?.source_url || d.source_url || ''
      return !url
    })
  }, [documents])

  // ── Enrich hosts with domain classification + virtual "(uploaded)" host ──
  const enrichedHosts: HostStatEnriched[] = useMemo(() => {
    const webHosts = rawHosts.map((h) => {
      const payerMatch = documents.find((d) => {
        const url = d.source_metadata?.source_url || d.source_url || ''
        try { return new URL(url).hostname === h.host }
        catch { return false }
      })
      const payer = payerMatch?.payer ?? null
      return { ...h, payer, domain: domainOf(h.host, payer) }
    })
    // Always inject an "(uploaded)" virtual entry at the top
    const uploadedPublished = uploadedDocs.filter((d) => !!d.published_at).length
    const uploadedEntry: HostStatEnriched = {
      host: UPLOADED_HOST,
      count: 0,
      corpusDocs: uploadedDocs.length,
      corpusPublished: uploadedPublished,
      payer: null,
      domain: 'other' as const,
    }
    return [uploadedEntry, ...webHosts]
  }, [rawHosts, documents, uploadedDocs])

  // ── Corpus stats for sidebar mini-strip ──────────────────────────────────
  const corpusStats: CorpusStats = useMemo(() => {
    let published = 0, waiting = 0, failed = 0
    for (const d of documents) {
      if (d.published_at) published++
      else if (d.embedding_status === 'failed' || d.chunking_status === 'failed') failed++
      else waiting++
    }
    return { published, waiting, failed }
  }, [documents])

  // ── Payer label for selected host ────────────────────────────────────────
  const payerLabel = useMemo(
    () => enrichedHosts.find((h) => h.host === selectedHost)?.payer ?? null,
    [enrichedHosts, selectedHost],
  )

  // ── Corpus search — debounced 400ms, respects mode ───────────────────────
  useEffect(() => {
    if (!searchQuery.trim()) {
      setSearchResults(null)
      setSearchTelemetry(null)
      return
    }
    const timer = setTimeout(async () => {
      setSearchLoading(true)
      try {
        const resp = await fetch(`${API_BASE}/api/skills/v1/corpus_search`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: searchQuery.trim(),
            k: 20,
            mode: searchMode,
          }),
        })
        if (resp.ok) {
          const data: CorpusSearchResponse = await resp.json()
          setSearchResults(data.chunks)
          setSearchTelemetry(data.telemetry)
        } else {
          setSearchResults([])
          setSearchTelemetry(null)
        }
      } catch {
        setSearchResults([])
        setSearchTelemetry(null)
      } finally {
        setSearchLoading(false)
      }
    }, 400)
    return () => clearTimeout(timer)
  }, [searchQuery, searchMode])

  // ── Handlers ─────────────────────────────────────────────────────────────
  const handleSelectHost = (host: string) => {
    setSelectedHost(host)
    setSearchQuery('')
    setSearchResults(null)
  }

  // @ts-ignore — kept for symmetry; search UI moved to Test tab.
  const _handleSearchModeChange = (mode: SearchMode) => {
    setSearchMode(mode)
    // Re-run immediately if there's already a query
    if (searchQuery.trim()) {
      setSearchResults(null)
    }
  }

  const openDocument = (documentId: string, pageNumber?: number, citeText?: string) => {
    setUrlPreview(null)
    setLocalNavigateTo(pageNumber != null ? { documentId, pageNumber, citeText } : null)
    onDocumentSelect(documentId)
    setReaderVisible(true)
    setSidebarCollapsed(true)
    // Auto-switch to reader panel if reader is currently expanded, keep layout otherwise
    if (expandedPanel === 'reader') {
      // already expanded, stay
    } else if (expandedPanel !== null) {
      // some other panel expanded — switch to reader
      setExpandedPanel('reader')
    }
    // null → leave default split, user can expand if desired
  }

  // @ts-ignore — search UI moved to Test tab.
  const _openChunk = (chunk: CorpusChunk) => {
    openDocument(
      chunk.document_id,
      chunk.page_number ?? undefined,
      chunk.text.slice(0, 300),
    )
  }

  const openUrlPreview = (url: string, ingested: boolean) => {
    setUrlPreview({ url, ingested })
    setLocalNavigateTo(null)
    setReaderVisible(true)
    setSidebarCollapsed(true)
  }

  const closeReader = () => {
    setReaderVisible(false)
    setUrlPreview(null)
    setLocalNavigateTo(null)
    setSidebarCollapsed(false)
  }

  const handleIngestUrl = async (url: string) => {
    setIngestingUrl(true)
    try {
      const resp = await fetch(`${API_BASE}/documents/import-from-html`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, auto_publish: true, auto_publish_timeout_s: 180 }),
      })
      if (resp.ok || resp.status === 409) {
        setUrlPreview({ url, ingested: true })
      }
    } finally {
      setIngestingUrl(false)
    }
  }

  // Merge local navigation with external navigation (external wins if more recent)
  const activeNavigateTo = navigateToRead ?? localNavigateTo

  // ── Browse: filter + group logic ─────────────────────────────────────────
  const filteredDocs = useMemo(() => {
    const q = browseSearch.toLowerCase()
    return documents.filter((d) => {
      // text search
      if (q) {
        const haystack = [d.display_name, d.filename, d.payer, d.state]
          .filter(Boolean).join(' ').toLowerCase()
        if (!haystack.includes(q)) return false
      }
      // payer filter
      if (payerFilter && (d.payer ?? '') !== payerFilter) return false
      // state filter
      if (stateFilter && (d.state ?? '') !== stateFilter) return false
      // status filter
      if (statusFilter) {
        const docStatus = d.published_at
          ? 'published'
          : d.chunking_status === 'failed' || d.embedding_status === 'failed'
            ? 'failed'
            : 'processing'
        if (docStatus !== statusFilter) return false
      }
      return true
    })
  }, [documents, browseSearch, payerFilter, stateFilter, statusFilter])

  // Group by payer
  const groupedDocs = useMemo(() => {
    const groups: Record<string, DocLike[]> = {}
    for (const d of filteredDocs) {
      const key = d.payer || d.state || '(untagged)'
      if (!groups[key]) groups[key] = []
      groups[key].push(d)
    }
    return groups
  }, [filteredDocs])

  // Unique filter options
  const uniquePayers = useMemo(() =>
    Array.from(new Set(documents.map((d) => d.payer).filter(Boolean) as string[])).sort(),
    [documents])
  const uniqueStates = useMemo(() =>
    Array.from(new Set(documents.map((d) => d.state).filter(Boolean) as string[])).sort(),
    [documents])

  // Age formatter
  const fmtAge = (iso: string | null | undefined) => {
    if (!iso) return ''
    const diff = Date.now() - new Date(iso).getTime()
    const days = Math.floor(diff / 86400000)
    if (days === 0) return 'today'
    if (days === 1) return '1d'
    if (days < 30) return `${days}d`
    const months = Math.floor(days / 30)
    return `${months}mo`
  }

  // Status dot class
  const statusDotClass = (d: DocLike) => {
    if (d.published_at) return 'published'
    if (d.chunking_status === 'failed' || d.embedding_status === 'failed') return 'failed'
    if (d.chunking_status === 'completed' || d.embedding_status === 'completed') return 'processing'
    return 'pending'
  }

  // ── Panel class helpers ───────────────────────────────────────────────────
  const browsePanelClass = () => {
    if (expandedPanel === 'browse') return 'repo-panel repo-panel--browse panel-expanded'
    if (expandedPanel === 'reader') return 'repo-panel repo-panel--browse panel-slim'
    if (expandedPanel === 'health') return 'repo-panel repo-panel--browse'
    return 'repo-panel repo-panel--browse'
  }

  const readerPanelClass = () => {
    if (expandedPanel === 'reader') return 'repo-panel repo-panel--reader panel-expanded'
    if (expandedPanel === 'browse') return 'repo-panel repo-panel--reader panel-hidden'
    return 'repo-panel repo-panel--reader'
  }

  // Selected doc display name for Reader panel header
  const selectedDoc = useMemo(
    () => documents.find((d) => d.id === selectedDocumentId) ?? null,
    [documents, selectedDocumentId],
  )
  const readerTitle = selectedDoc
    ? (selectedDoc.display_name || selectedDoc.filename)
    : 'Reader'

  // Suppress unused vars from original code that are now no longer used in render
  void statsLoading
  void selectedHost
  void handleSelectHost
  void entitySearch
  void setEntitySearch
  void domainFilter
  void setDomainFilter
  void corpusStats
  void payerLabel
  void uploadedDocs
  void enrichedHosts
  void openUrlPreview

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="repo-tab">
      {/* ── Horizontal panels row ──────────────────────────────────────── */}
      <div className="repo-panels">

        {/* ── Panel 1: Browse ─────────────────────────────────────────── */}
        <div className={browsePanelClass()}>
          {/* Header */}
          <div className="repo-panel-header">
            <span>📂 Browse</span>
            <div className="repo-panel-header-actions">
              {expandedPanel === 'browse' ? (
                <button
                  type="button"
                  className="repo-panel-icon-btn"
                  title="Collapse browse"
                  onClick={() => setExpandedPanel(null)}
                >⊟</button>
              ) : (
                <button
                  type="button"
                  className="repo-panel-icon-btn"
                  title="Expand browse"
                  onClick={() => setExpandedPanel('browse')}
                >⊞</button>
              )}
            </div>
          </div>

          {/* Body — hidden in slim mode */}
          {expandedPanel !== 'reader' && (
            <div className="repo-panel-body">
              {/* Search + filters */}
              <div className="repo-browse-search">
                <input
                  type="search"
                  placeholder="Search documents, payer, org…"
                  value={browseSearch}
                  onChange={(e) => setBrowseSearch(e.target.value)}
                />
                <div className="repo-browse-filters">
                  <select value={payerFilter} onChange={(e) => setPayerFilter(e.target.value)}>
                    <option value="">Payer ▾</option>
                    {uniquePayers.map((p) => (
                      <option key={p} value={p}>{p}</option>
                    ))}
                  </select>
                  <select value={stateFilter} onChange={(e) => setStateFilter(e.target.value)}>
                    <option value="">State ▾</option>
                    {uniqueStates.map((s) => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                  <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
                    <option value="">Status ▾</option>
                    <option value="published">published</option>
                    <option value="processing">processing</option>
                    <option value="failed">failed</option>
                  </select>
                </div>
              </div>

              {/* Document list */}
              {filteredDocs.length === 0 ? (
                <div className="repo-browse-empty">
                  {documentsLoading
                    ? 'Loading documents…'
                    : documents.length === 0
                      ? 'No documents in corpus yet.'
                      : 'No documents match your search.'}
                </div>
              ) : (
                <ul className="repo-browse-list">
                  {Object.entries(groupedDocs).map(([group, docs]) => {
                    const isCollapsed = !!collapsedGroups[group]
                    return (
                      <li key={group}>
                        {/* Group header */}
                        <div
                          className="repo-browse-group-header"
                          onClick={() =>
                            setCollapsedGroups((prev) => ({ ...prev, [group]: !prev[group] }))
                          }
                        >
                          <span>{isCollapsed ? '▶' : '▼'}</span>
                          <span>{group}</span>
                          <span className="repo-browse-group-count">{docs.length}</span>
                        </div>
                        {/* Doc items */}
                        {!isCollapsed && docs.map((d) => (
                          <div
                            key={d.id}
                            className={`repo-browse-doc-item${d.id === selectedDocumentId ? ' selected' : ''}`}
                            onClick={() => {
                              openDocument(d.id)
                              // If reader was previously expanded keep it, else leave layout
                            }}
                          >
                            <span className={`repo-browse-doc-status ${statusDotClass(d)}`} />
                            <span className="repo-browse-doc-name" title={d.display_name || d.filename}>
                              {d.display_name || d.filename}
                            </span>
                            {d.published_at && (
                              <span style={{ fontSize: '0.65rem', color: 'var(--mobius-text-muted)', flexShrink: 0 }}>
                                {fmtAge(d.published_at)}
                              </span>
                            )}
                          </div>
                        ))}
                      </li>
                    )
                  })}
                </ul>
              )}
            </div>
          )}
        </div>

        {/* ── Panel 2: Reader ─────────────────────────────────────────── */}
        <div className={readerPanelClass()}>
          {/* Header */}
          <div className="repo-panel-header">
            <span>📖 {readerTitle}</span>
            <div className="repo-panel-header-actions">
              {expandedPanel === 'reader' ? (
                <button
                  type="button"
                  className="repo-panel-icon-btn"
                  title="Collapse reader"
                  onClick={() => setExpandedPanel(null)}
                >⊟</button>
              ) : (
                <button
                  type="button"
                  className="repo-panel-icon-btn"
                  title="Expand reader"
                  onClick={() => setExpandedPanel('reader')}
                >⊞</button>
              )}
              {(selectedDocumentId || urlPreview) && (
                <button
                  type="button"
                  className="repo-panel-icon-btn"
                  title="Close document"
                  onClick={closeReader}
                >✕</button>
              )}
            </div>
          </div>

          {/* Body */}
          <div className="repo-panel-body repo-reader-body">
            {!selectedDocumentId && !urlPreview ? (
              <div className="repo-reader-empty">
                Select a document from Browse to read it.
              </div>
            ) : (
              <ReaderSlideOut
                documents={documents}
                selectedDocumentId={selectedDocumentId}
                navigateToRead={activeNavigateTo}
                onNavigateToReadConsumed={() => {
                  onNavigateToReadConsumed()
                  setLocalNavigateTo(null)
                }}
                onDocumentSelect={onDocumentSelect}
                urlPreview={urlPreview}
                onIngestUrl={handleIngestUrl}
                ingestingUrl={ingestingUrl}
                onClose={closeReader}
              />
            )}
          </div>
        </div>
      </div>

      {/* ── Panel 3: Health strip ────────────────────────────────────────── */}
      <div className={`repo-health-strip${expandedPanel === 'health' ? ' health-fullscreen' : ''}`}>
        {/* Health bar — always visible */}
        {(() => {
          // Pipeline health pulled from /admin/pipeline_health every 10s.
          // Drives the green/yellow/red dots next to each stage. Local
          // hook is fine here — keeps the change contained and avoids
          // threading another global through this already-wide component.
          // eslint-disable-next-line react-hooks/rules-of-hooks
          const [health, setHealth] = useState<any>(null)
          // eslint-disable-next-line react-hooks/rules-of-hooks
          const [expanded, setExpanded] = useState(false)
          // eslint-disable-next-line react-hooks/rules-of-hooks
          const [stageExpanded, setStageExpanded] = useState<Record<string, boolean>>({})
          // eslint-disable-next-line react-hooks/rules-of-hooks
          useEffect(() => {
            let alive = true
            const tick = async () => {
              try {
                const r = await fetch(`${API_BASE}/pipeline_health`)
                if (r.ok && alive) setHealth(await r.json())
              } catch { /* ignore */ }
            }
            tick()
            const id = setInterval(tick, 10_000)
            return () => { alive = false; clearInterval(id) }
          }, [])
          const dotColor = (s?: string) =>
            s === 'green' ? '#10b981' : s === 'yellow' ? '#f59e0b' : s === 'red' ? '#ef4444' : '#cbd5e1'
          const Dot = ({ s, title }: { s?: string; title: string }) => (
            <span
              title={title}
              style={{
                display: 'inline-block', width: 8, height: 8, borderRadius: 999,
                background: dotColor(s), marginRight: 4,
              }}
              aria-label={title}
            />
          )
          // Prefer server-side totals (refreshed every 10s via the health
          // poll above) over the client-side ``documents`` array which
          // only loads once at page mount. Falls back to client counts
          // when health isn't loaded yet (first second after load).
          const t = health?.totals
          const total = t?.documents ?? documents.length
          const chunked = t?.chunked ?? documents.filter((d: any) => d.chunking_status === 'completed').length
          const embedded = t?.embedded ?? documents.filter((d: any) => d.embedding_status === 'completed').length
          const published = t?.published ?? documents.filter((d: any) => !!d.published_at).length
          const failed = documents.filter((d: any) =>
            d.chunking_status === 'failed' || d.embedding_status === 'failed').length
          const processing = total - published - failed
          const chk = health?.chunking
          const emb = health?.embedding
          const pub = health?.publishing
          const stageMeta: Array<{ label: string; n: number; terminal?: boolean; dot?: any; tip?: string }> = [
            { label: 'Documents', n: total },
            {
              label: 'Chunked', n: chunked,
              dot: chk?.status,
              tip: chk
                ? `Chunking: ${chk.status}\n${chk.active} workers active · ${chk.last_hour}/h · ${chk.pending} pending`
                : 'Chunking status loading…',
            },
            {
              label: 'Embedded', n: embedded,
              dot: emb?.status,
              tip: emb
                ? `Embedding: ${emb.status}\n${emb.active} workers active · ${emb.last_hour}/h · ${emb.pending} pending`
                : 'Embedding status loading…',
            },
            {
              label: 'Available in chat', n: published, terminal: true,
              dot: pub?.status,
              tip: pub
                ? `Publishing: ${pub.status}\n${pub.last_hour}/h · ${pub.embedded_unpublished} embedded but unpublished`
                : 'Publishing status loading…',
            },
          ]
          // Compute ETAs and rates for expanded view
          const fmtEta = (pending: number, perHour: number) => {
            if (!perHour || perHour <= 0) return pending > 0 ? '—' : '✓ caught up'
            const hours = pending / perHour
            if (hours < 1) return `${Math.round(hours * 60)} min`
            if (hours < 24) return `${hours.toFixed(1)} h`
            return `${(hours / 24).toFixed(1)} d`
          }
          const fmtSeconds = (s: number | null | undefined) => {
            if (s == null || !isFinite(s) || s <= 0) return '—'
            const h = s / 3600
            if (h < 1) return `${Math.round(s / 60)} min`
            if (h < 24) return `${h.toFixed(1)} h`
            return `${(h / 24).toFixed(1)} d`
          }
          // Tiny inline sparkline for 6 × 5-min buckets (oldest→newest).
          const Spark = ({ buckets }: { buckets: number[] }) => {
            const w = 78, h = 18, max = Math.max(1, ...buckets)
            const bw = (w - (buckets.length - 1) * 2) / buckets.length
            const bucketSpanMin = 5
            const totalBuckets = buckets.length
            return (
              <svg width={w} height={h} style={{ verticalAlign: 'middle' }}>
                {buckets.map((n, i) => {
                  const bh = (n / max) * (h - 2)
                  const minAgoEnd = (totalBuckets - 1 - i) * bucketSpanMin
                  const minAgoStart = minAgoEnd + bucketSpanMin
                  const span = minAgoEnd === 0
                    ? `last ${bucketSpanMin} min`
                    : `${minAgoEnd}-${minAgoStart} min ago`
                  return (
                    <rect
                      key={i}
                      x={i * (bw + 2)}
                      y={h - bh}
                      width={bw}
                      height={Math.max(1, bh)}
                      fill="#10b981"
                      opacity={n === 0 ? 0.25 : 1}
                      style={{ cursor: 'help' }}
                    >
                      <title>{`${span}: ${n} ${n === 1 ? 'event' : 'events'}`}</title>
                    </rect>
                  )
                })}
              </svg>
            )
          }

          return (
            <>
              {/* Collapsed bar */}
              <div
                className="repo-health-bar"
                onClick={() => setHealthOpen((v) => !v)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setHealthOpen((v) => !v) }}
                aria-expanded={healthOpen}
              >
                <span style={{ fontSize: 11, color: '#888', width: 12, display: 'inline-block' }} aria-hidden>
                  {healthOpen ? '▼' : '▶'}
                </span>
                <strong>Corpus Health</strong>
                <span style={{ color: '#666' }}>·</span>
                <span style={{ color: '#666' }}>{total.toLocaleString()} docs</span>
                <span style={{ color: '#666' }}>·</span>
                <span style={{ color: '#666' }}>{chunked.toLocaleString()} chunked</span>
                <span style={{ color: '#666' }}>·</span>
                <span style={{ color: '#666' }}>{embedded.toLocaleString()} embedded</span>
                {/* Retag button */}
                <button
                  type="button"
                  style={{
                    marginLeft: 'auto',
                    fontSize: 11, padding: '0.15rem 0.5rem',
                    border: '1px solid var(--mobius-border)',
                    borderRadius: 'var(--mobius-radius-sm)',
                    background: 'var(--mobius-bg-secondary)',
                    color: 'var(--mobius-text-secondary)',
                    cursor: 'pointer',
                  }}
                  onClick={(e) => {
                    e.stopPropagation()
                    onRefresh?.()
                  }}
                  title="Re-tag documents"
                >
                  Retag
                </button>
                {/* Expand health to fullscreen */}
                {expandedPanel === 'health' ? (
                  <button
                    type="button"
                    className="repo-panel-icon-btn"
                    title="Collapse health panel"
                    onClick={(e) => { e.stopPropagation(); setExpandedPanel(null) }}
                  >⊟</button>
                ) : (
                  <button
                    type="button"
                    className="repo-panel-icon-btn"
                    title="Expand health panel"
                    onClick={(e) => { e.stopPropagation(); setExpandedPanel('health') }}
                  >⊞</button>
                )}
              </div>

              {/* Expanded health body */}
              {(healthOpen || expandedPanel === 'health') && (
                <div className="repo-health-body">
                  {/* Collapsed bar content (the existing CorpusStatus inline JSX) */}
                  <div
                    className="repo-corpus-status"
                    role="group"
                    aria-label="Overall corpus pipeline status"
                    style={{ borderBottom: '1px solid #eee', fontSize: 13 }}
                  >
                    {/* Inner expand/collapse toggle */}
                    <div
                      onClick={() => setExpanded(e => !e)}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setExpanded(x => !x) }}
                      aria-expanded={expanded}
                      style={{
                        display: 'flex', alignItems: 'center', gap: 12,
                        padding: '8px 12px', cursor: 'pointer',
                        userSelect: 'none', flexWrap: 'wrap',
                      }}
                    >
                      <span
                        style={{ marginRight: 4, fontSize: 11, color: '#888', width: 12, display: 'inline-block' }}
                        aria-hidden
                      >{expanded ? '▼' : '▶'}</span>
                      <strong style={{ marginRight: 4 }}>Corpus</strong>
                      {stageMeta.map((s, i) => (
                        <span key={s.label} style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                          {s.dot !== undefined && <Dot s={s.dot} title={s.tip || s.label} />}
                          <span
                            title={s.tip}
                            style={{
                              fontWeight: 600,
                              color: s.terminal ? '#10b981' : (s.n > 0 ? '#111' : '#999'),
                            }}
                          >{s.n.toLocaleString()}</span>
                          <span style={{ color: '#666' }} title={s.tip}>{s.label}</span>
                          {i < stageMeta.length - 1 && <span style={{ color: '#ccc', marginLeft: 8 }}>→</span>}
                        </span>
                      ))}
                      {processing > 0 && (
                        <span style={{ marginLeft: 'auto', color: '#f59e0b' }}>
                          {processing.toLocaleString()} processing
                        </span>
                      )}
                      {failed > 0 && (
                        <span style={{ color: '#ef4444' }}>{failed.toLocaleString()} failed</span>
                      )}
                      {health?.integrity && (
                        <span
                          style={{
                            display: 'inline-flex', alignItems: 'center', gap: 4,
                            marginLeft: processing > 0 || failed > 0 ? 0 : 'auto',
                            paddingLeft: 8, borderLeft: '1px solid #eee',
                          }}
                          title={
                            health.integrity.chat_orphans === 0
                              ? 'Integrity OK\nrag and chat are fully aligned'
                              : `Integrity drift\n${health.integrity.chat_orphans} doc(s) shown by chat that rag does NOT have.\nFix: POST /admin/cleanup_chat_orphans?dry_run=false`
                          }
                        >
                          <Dot s={health.integrity.status} title="Integrity status" />
                          <span style={{ color: '#666' }}>Integrity</span>
                          {health.integrity.chat_orphans > 0 && (
                            <span style={{ color: '#ef4444', fontWeight: 600 }}>
                              {health.integrity.chat_orphans} drift
                            </span>
                          )}
                        </span>
                      )}
                    </div>

                    {/* Expanded panel — per-stage rate, pending, ETA */}
                    {expanded && health && (
                      <div
                        style={{
                          padding: '10px 14px 14px 28px',
                          background: '#fafbfc',
                          borderTop: '1px solid #f0f0f0',
                          display: 'grid',
                          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
                          gap: 12,
                        }}
                      >
                        {[
                          { label: 'Chunking', s: chk, terminalCount: chunked, inFlightLabel: 'Currently chunking', extraCol: 'paragraphs_done' },
                          { label: 'Embedding', s: emb, terminalCount: embedded, inFlightLabel: 'Currently embedding', extraCol: 'chunks_done' },
                          {
                            label: 'Publishing',
                            s: pub
                              ? { ...pub, active: undefined, pending: pub.embedded_unpublished }
                              : null,
                            terminalCount: published,
                            inFlightLabel: 'Recently published (last 30 min)',
                            extraCol: 'chunks_done',
                          },
                        ].map(({ label, s, terminalCount, inFlightLabel, extraCol }) => {
                          const r = s?.rolling
                          const inFlight: any[] = (s as any)?.in_flight || []
                          const isStageExpanded = !!stageExpanded[label]
                          return (
                          <div
                            key={label}
                            style={{
                              background: '#fff',
                              border: '1px solid #ececec',
                              borderRadius: 6,
                              padding: '10px 12px',
                            }}
                          >
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                              <Dot s={s?.status} title={`${label} status`} />
                              <strong style={{ fontSize: 13 }}>{label}</strong>
                              {inFlight.length > 0 && (
                                <span
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    setStageExpanded(prev => ({ ...prev, [label]: !prev[label] }))
                                  }}
                                  role="button"
                                  style={{
                                    marginLeft: 'auto', fontSize: 11, color: '#3b82f6',
                                    cursor: 'pointer', userSelect: 'none',
                                  }}
                                  title="Click to see what's in flight"
                                >
                                  {isStageExpanded ? '▼' : '▶'} {inFlight.length} in flight
                                </span>
                              )}
                            </div>
                            <div style={{ fontSize: 12, color: '#444', display: 'grid', gap: 3 }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                <span style={{ color: '#888' }}>Last 30min:</span>{' '}
                                <span style={{ fontWeight: 600 }}>
                                  {r ? `${r.rate_per_hour}/h` : `${s?.last_hour ?? 0}/h *`}
                                </span>
                                {r?.buckets_5min && <Spark buckets={r.buckets_5min} />}
                              </div>
                              {r && (
                                <div style={{ color: '#888', fontSize: 11 }}>
                                  95% CI: {r.rate_lo_per_hour}–{r.rate_hi_per_hour}/h
                                </div>
                              )}
                              {s?.active != null && (
                                <div>
                                  <span style={{ color: '#888' }}>Active workers:</span>{' '}
                                  <span style={{ fontWeight: 600 }}>{s.active}</span>
                                </div>
                              )}
                              <div>
                                <span style={{ color: '#888' }}>Pending:</span>{' '}
                                <span style={{ fontWeight: 600 }}>{(s?.pending ?? 0).toLocaleString()}</span>
                              </div>
                              {r && r.eta_seconds_p50 != null ? (
                                <>
                                  <div>
                                    <span style={{ color: '#888' }}>ETA (median):</span>{' '}
                                    <span style={{ fontWeight: 600 }}>{fmtSeconds(r.eta_seconds_p50)}</span>
                                  </div>
                                  <div style={{ color: '#888', fontSize: 11 }}>
                                    95% CI: {fmtSeconds(r.eta_seconds_p5)} – {fmtSeconds(r.eta_seconds_p95)}
                                  </div>
                                </>
                              ) : (
                                <div>
                                  <span style={{ color: '#888' }}>ETA:</span>{' '}
                                  <span style={{ fontWeight: 600, color: s?.last_hour ? '#111' : '#999' }}>
                                    {fmtEta(s?.pending ?? 0, s?.last_hour ?? 0)}
                                  </span>
                                </div>
                              )}
                              <div style={{ marginTop: 4, paddingTop: 4, borderTop: '1px dashed #f0f0f0' }}>
                                <span style={{ color: '#888' }}>Total {label.toLowerCase()}:</span>{' '}
                                <span style={{ fontWeight: 600 }}>{terminalCount.toLocaleString()}</span>
                              </div>
                            </div>
                            {/* In-flight panel */}
                            {isStageExpanded && inFlight.length > 0 && (
                              <div
                                style={{
                                  marginTop: 8, paddingTop: 8,
                                  borderTop: '1px solid #f0f0f0',
                                }}
                              >
                                <div style={{ fontSize: 11, color: '#666', marginBottom: 4 }}>
                                  {inFlightLabel}
                                </div>
                                <div
                                  style={{
                                    maxHeight: 220, overflowY: 'auto',
                                    fontSize: 11, fontFamily: 'ui-monospace, monospace',
                                  }}
                                >
                                  {inFlight.map((d: any, i: number) => {
                                    const elapsed = d.elapsed_s
                                    const elapsedTxt = elapsed < 60
                                      ? `${elapsed}s`
                                      : elapsed < 3600
                                        ? `${Math.floor(elapsed/60)}m${elapsed%60}s`
                                        : `${(elapsed/3600).toFixed(1)}h`
                                    const extra = d[extraCol] ?? 0
                                    return (
                                      <div
                                        key={d.doc_id + ':' + i}
                                        style={{
                                          display: 'grid',
                                          gridTemplateColumns: '1fr auto auto',
                                          gap: 6, padding: '2px 0',
                                          borderBottom: '1px dotted #f4f4f4',
                                        }}
                                        title={`${d.payer || '<unknown>'}\n${d.filename}\nelapsed: ${elapsedTxt}`}
                                      >
                                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                          {d.filename || d.doc_id.slice(0, 8)}
                                        </span>
                                        <span style={{ color: '#666' }}>
                                          {label === 'Chunking' && `${d.pages || 0}p · ${extra}¶`}
                                          {label === 'Embedding' && `${extra} chunks`}
                                          {label === 'Publishing' && `${extra} chunks`}
                                        </span>
                                        <span style={{ color: '#888', minWidth: 38, textAlign: 'right' }}>
                                          {elapsedTxt}
                                        </span>
                                      </div>
                                    )
                                  })}
                                </div>
                              </div>
                            )}
                          </div>
                        )})}
                        {/* Integrity card */}
                        {health.integrity && (() => {
                          const ig = health.integrity
                          const row = (label: string, n: number, hint: string) => (
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                              <span style={{ color: '#888' }} title={hint}>{label}:</span>
                              <span style={{ fontWeight: 600,
                                             color: n === 0 ? '#10b981' : (n >= 100 ? '#ef4444' : '#f59e0b') }}>
                                {n.toLocaleString()}
                              </span>
                            </div>
                          )
                          return (
                            <div
                              style={{
                                background: '#fff',
                                border: '1px solid #ececec',
                                borderRadius: 6,
                                padding: '10px 12px',
                                gridColumn: 'span 2',
                              }}
                            >
                              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                                <Dot s={ig.status} title="Integrity status" />
                                <strong style={{ fontSize: 13 }}>Integrity</strong>
                                <span style={{ marginLeft: 'auto', color: '#888', fontSize: 11 }}>
                                  4 dimensions
                                </span>
                              </div>
                              <div style={{ fontSize: 12, color: '#444', display: 'grid', gap: 3 }}>
                                {row('Chat orphans', ig.chat_orphans ?? 0,
                                  'docs visible to chat that rag does NOT have — phantom citation risk. Fix: POST /admin/cleanup_chat_orphans')}
                                {row('Sitemap orphans', ig.sitemap_orphans ?? 0,
                                  'rag docs without a discovered_sources registry row — per-host counts undercount. Fix: POST /admin/backfill_sitemap')}
                                {row('Blocked jobs', ig.blocked_jobs ?? 0,
                                  'chunking jobs at failure_count >= 3 — manual triage required. See: GET /admin/list_blocked_docs')}
                                {row('Metadata orphans', ig.metadata_orphans ?? 0,
                                  'docs missing payer / state / program tags — cannot be filtered or attributed in retrieval')}
                              </div>
                            </div>
                          )
                        })()}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </>
          )
        })()}
      </div>
    </div>
  )
}
