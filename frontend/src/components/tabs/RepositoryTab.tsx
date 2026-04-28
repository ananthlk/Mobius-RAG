import { useEffect, useMemo, useState } from 'react'
import { API_BASE } from '../../config'
import { EntityCard } from './repository/EntityCard'
import { ReaderSlideOut } from './repository/ReaderSlideOut'
import { EntitySidebar, domainOf } from './repository/EntitySidebar'
import type { HostStatEnriched, CorpusStats, DomainFilter } from './repository/EntitySidebar'
import { SearchTracePanel } from './repository/SearchTracePanel'
import type { SearchTelemetry } from './repository/SearchTracePanel'
import './RepositoryTab.css'

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
  selectedDocumentId: string | null
  navigateToRead: NavigateToRead | null
  onNavigateToReadConsumed: () => void
  onDocumentSelect: (documentId: string) => void
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

function CorpusSearchResults({
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
  selectedDocumentId,
  navigateToRead,
  onNavigateToReadConsumed,
  onDocumentSelect,
}: Props) {
  // ── Entity / host data ───────────────────────────────────────────────────
  const [rawHosts, setRawHosts] = useState<RawHostStat[]>([])
  const [statsLoading, setStatsLoading] = useState(true)
  const [selectedHost, setSelectedHost] = useState('')

  // ── Sidebar UI state ─────────────────────────────────────────────────────
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [entitySearch, setEntitySearch] = useState('')
  const [domainFilter, setDomainFilter] = useState<DomainFilter>('all')

  // ── Search state ─────────────────────────────────────────────────────────
  const [searchQuery, setSearchQuery] = useState('')
  const [searchMode, setSearchMode] = useState<SearchMode>('corpus')
  const [searchResults, setSearchResults] = useState<CorpusChunk[] | null>(null)
  const [searchTelemetry, setSearchTelemetry] = useState<SearchTelemetry | null>(null)
  const [searchLoading, setSearchLoading] = useState(false)

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

  const readerOpen = readerVisible && (!!selectedDocumentId || !!urlPreview)

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
        if (hostList.length > 0 && !selectedHost) {
          setSelectedHost(hostList[0].host)
        }
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

  // ── Enrich hosts with domain classification ──────────────────────────────
  const enrichedHosts: HostStatEnriched[] = useMemo(() => {
    return rawHosts.map((h) => {
      const payerMatch = documents.find((d) => {
        const url = d.source_metadata?.source_url || d.source_url || ''
        try { return new URL(url).hostname === h.host }
        catch { return false }
      })
      const payer = payerMatch?.payer ?? null
      return { ...h, payer, domain: domainOf(h.host, payer) }
    })
  }, [rawHosts, documents])

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

  const handleSearchModeChange = (mode: SearchMode) => {
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
  }

  const openChunk = (chunk: CorpusChunk) => {
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

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className={`repository-tab-v2${sidebarCollapsed ? ' sidebar-collapsed' : ''}`}>
      {/* ── Left: entity sidebar ──────────────────────────────────────── */}
      <EntitySidebar
        hosts={enrichedHosts}
        selectedHost={selectedHost}
        onSelectHost={handleSelectHost}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed((c) => !c)}
        entitySearch={entitySearch}
        onEntitySearch={setEntitySearch}
        domainFilter={domainFilter}
        onDomainFilter={setDomainFilter}
        stats={corpusStats}
      />

      {/* ── Right: main area ──────────────────────────────────────────── */}
      <div className="repo-main">
        {/* Corpus search bar — hidden when reader is open to reclaim vertical space */}
        {!readerOpen && (
          <div className="repo-search-bar">
            <div className="repo-search-modes" role="group" aria-label="Search mode">
              <button
                className={`repo-search-mode-btn${searchMode === 'corpus' ? ' active' : ''}`}
                title="Hybrid: BM25 + pgvector, RRF-fused and reranked (best all-round)"
                aria-pressed={searchMode === 'corpus'}
                onClick={() => handleSearchModeChange('corpus')}
              >
                Hybrid
              </button>
              <button
                className={`repo-search-mode-btn${searchMode === 'precision' ? ' active' : ''}`}
                title="BM25 exact-phrase search — best for codes, HCPCS, policy IDs"
                aria-pressed={searchMode === 'precision'}
                onClick={() => handleSearchModeChange('precision')}
              >
                BM25
              </button>
              <button
                className={`repo-search-mode-btn${searchMode === 'recall' ? ' active' : ''}`}
                title="Semantic (pgvector) search — best for paraphrased questions"
                aria-pressed={searchMode === 'recall'}
                onClick={() => handleSearchModeChange('recall')}
              >
                Semantic
              </button>
            </div>
            <div className="repo-search-field">
              <input
                type="search"
                className="repo-search-input"
                placeholder="Search the corpus — try a question or a code…"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                aria-label="Search corpus"
              />
              {searchLoading && (
                <span className="repo-search-spin" aria-hidden>⟳</span>
              )}
              {searchQuery && !searchLoading && (
                <button
                  className="repo-search-clear"
                  onClick={() => { setSearchQuery(''); setSearchResults(null) }}
                  aria-label="Clear search"
                  title="Clear"
                >
                  ×
                </button>
              )}
            </div>
          </div>
        )}

        {/* Content area: entity detail + optional reader */}
        <div className={`repo-content-area${readerOpen ? ' reader-open' : ''}`}>
          {/* Entity / search area */}
          <div className="repo-entity-area">
            {statsLoading ? (
              <div className="loading repo-loading">Loading sources…</div>
            ) : searchQuery.trim() ? (
              <CorpusSearchResults
                query={searchQuery}
                mode={searchMode}
                chunks={searchResults}
                loading={searchLoading}
                telemetry={searchTelemetry}
                onOpenChunk={openChunk}
              />
            ) : selectedHost ? (
              <EntityCard
                host={selectedHost}
                payerLabel={payerLabel}
                documents={documents}
                onSelectDoc={openDocument}
                onSelectUrl={openUrlPreview}
                selectedDocumentId={selectedDocumentId}
                selectedUrl={urlPreview?.url ?? null}
              />
            ) : enrichedHosts.length > 0 ? (
              <div className="repo-empty">
                Select a source from the left panel, or search the corpus above.
              </div>
            ) : null}
          </div>

          {/* Inline reader pane */}
          {readerOpen && (
            <div className="repo-reader-pane">
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
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
