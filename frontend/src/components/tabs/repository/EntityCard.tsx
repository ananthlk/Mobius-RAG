import { useEffect, useMemo, useState } from 'react'
import { API_BASE } from '../../../config'
import { SourceTreeView } from '../sources/SourceTreeView'
import type { SourceRow } from '../sources/treeBuilder'
import { IngestedDocsList } from './IngestedDocsList'

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

interface Props {
  host: string
  payerLabel: string | null
  documents: DocLike[]
  onSelectDoc: (documentId: string) => void
  onSelectUrl: (url: string, ingested: boolean) => void
  selectedDocumentId?: string | null
  selectedUrl?: string | null
}

type Tab = 'docs' | 'sitemap'
type ContentFilter = 'all' | 'html' | 'pdf' | 'doc'
type DocStatusFilter = 'all' | 'published' | 'processing' | 'failed'

const CONTENT_FILTERS: { key: ContentFilter; label: string }[] = [
  { key: 'all',  label: 'All types' },
  { key: 'html', label: 'HTML' },
  { key: 'pdf',  label: 'PDF' },
  { key: 'doc',  label: 'Docs' },
]

const DOC_STATUS_FILTERS: { key: DocStatusFilter; label: string }[] = [
  { key: 'all',        label: 'All' },
  { key: 'published',  label: 'Published' },
  { key: 'processing', label: 'Processing' },
  { key: 'failed',     label: 'Failed' },
]

// ── Helpers ────────────────────────────────────────────────────────────────

/** Build a sorted list of { tag, count } from an array of tag arrays. */
function tagCounts(tagArrays: (string[] | null | undefined)[]): { tag: string; count: number }[] {
  const map = new Map<string, number>()
  for (const tags of tagArrays) {
    for (const t of tags ?? []) {
      map.set(t, (map.get(t) ?? 0) + 1)
    }
  }
  return Array.from(map.entries())
    .map(([tag, count]) => ({ tag, count }))
    .sort((a, b) => b.count - a.count)
}

// ── Component ──────────────────────────────────────────────────────────────

export function EntityCard({
  host,
  payerLabel,
  documents,
  onSelectDoc,
  onSelectUrl,
  selectedDocumentId,
  selectedUrl,
}: Props) {
  const [hostRows, setHostRows] = useState<SourceRow[]>([])
  const [hostLoading, setHostLoading] = useState(false)
  const [ingestingUrls, setIngestingUrls] = useState<Set<string>>(new Set())
  const [autoPublish, setAutoPublish] = useState(true)

  // Tab + filter state
  const [activeTab, setActiveTab] = useState<Tab>('docs')
  const [contentFilter, setContentFilter] = useState<ContentFilter>('all')
  const [docStatusFilter, setDocStatusFilter] = useState<DocStatusFilter>('all')
  const [docTagFilter, setDocTagFilter] = useState<string | null>(null)
  const [sitemapTagFilter, setSitemapTagFilter] = useState<string | null>(null)

  // Reset tag filters when host changes
  useEffect(() => {
    setDocTagFilter(null)
    setSitemapTagFilter(null)
    setDocStatusFilter('all')
    setContentFilter('all')
  }, [host])

  useEffect(() => {
    if (!host) return
    let cancelled = false
    setHostLoading(true)
    fetch(`${API_BASE}/sources/search?host=${encodeURIComponent(host)}&only_reachable=false&limit=500`)
      .then((r) => r.json() as Promise<SourceRow[]>)
      .then((rows) => {
        if (cancelled) return
        setHostRows(rows)
        setHostLoading(false)
      })
      .catch(() => {
        if (cancelled) return
        setHostRows([])
        setHostLoading(false)
      })
    return () => { cancelled = true }
  }, [host])

  // ── Ingest handlers ──────────────────────────────────────────────────────
  const handleIngest = async (url: string) => {
    if (ingestingUrls.has(url)) return
    setIngestingUrls((prev) => new Set(prev).add(url))
    try {
      const resp = await fetch(`${API_BASE}/documents/import-from-html`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, auto_publish: autoPublish, auto_publish_timeout_s: 180 }),
      })
      if (resp.ok || resp.status === 409) {
        setHostRows((rows) => rows.map((r) => (r.url === url ? { ...r, ingested: true } : r)))
      }
    } catch { /* swallow */ }
    finally {
      setIngestingUrls((prev) => { const n = new Set(prev); n.delete(url); return n })
    }
  }

  const handleBulkIngest = async (urls: string[]) => {
    for (const url of urls) {
      if (ingestingUrls.has(url)) continue
      setIngestingUrls((prev) => new Set(prev).add(url))
      try {
        const resp = await fetch(`${API_BASE}/documents/import-from-html`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url, auto_publish: false }),
        })
        if (resp.status === 200 || resp.status === 409) {
          setHostRows((rows) => rows.map((r) => (r.url === url ? { ...r, ingested: true } : r)))
        }
      } catch { /* swallow */ }
      finally {
        setIngestingUrls((prev) => { const n = new Set(prev); n.delete(url); return n })
      }
    }
  }

  // ── Doc–host join via ingested_doc_id ────────────────────────────────────
  const docsForHost = useMemo(() => {
    const ingestedIds = new Set(
      hostRows
        .filter((r) => r.ingested && r.ingested_doc_id)
        .map((r) => r.ingested_doc_id as string),
    )
    if (ingestedIds.size > 0) {
      return documents.filter((d) => ingestedIds.has(d.id))
    }
    // Fallback: hostname or payer match for directly-uploaded docs.
    return documents.filter((d) => {
      const url = d.source_metadata?.source_url || d.source_url || ''
      if (url) {
        try { if (new URL(url).hostname === host) return true }
        catch { /* not a URL */ }
      }
      if (payerLabel && d.payer && d.payer === payerLabel) return true
      return false
    })
  }, [documents, host, payerLabel, hostRows])

  // ── docId → tags map (via source rows) ──────────────────────────────────
  // topic_tags live on discovered_sources, not on the document.  We bridge
  // through ingested_doc_id: each source row that points at a document also
  // carries that URL's topic_tags.
  const docTagsMap = useMemo(() => {
    const map = new Map<string, string[]>()
    for (const r of hostRows) {
      if (!r.ingested_doc_id || !r.topic_tags?.length) continue
      const existing = map.get(r.ingested_doc_id) ?? []
      // merge tags, deduplicate
      const merged = Array.from(new Set([...existing, ...r.topic_tags]))
      map.set(r.ingested_doc_id, merged)
    }
    return map
  }, [hostRows])

  // ── Doc status filter ────────────────────────────────────────────────────
  const statusFilteredDocs = useMemo(() => {
    if (docStatusFilter === 'all') return docsForHost
    return docsForHost.filter((d) => {
      const failed = d.embedding_status === 'failed' || d.chunking_status === 'failed'
      if (docStatusFilter === 'published')  return !!d.published_at
      if (docStatusFilter === 'failed')     return failed
      if (docStatusFilter === 'processing') return !d.published_at && !failed
      return true
    })
  }, [docsForHost, docStatusFilter])

  // ── Doc tag filter ───────────────────────────────────────────────────────
  const filteredDocs = useMemo(() => {
    if (!docTagFilter) return statusFilteredDocs
    return statusFilteredDocs.filter((d) =>
      (docTagsMap.get(d.id) ?? []).includes(docTagFilter),
    )
  }, [statusFilteredDocs, docTagFilter, docTagsMap])

  // Tag counts for the docs tab (from status-filtered set)
  const docTagList = useMemo(
    () => tagCounts(statusFilteredDocs.map((d) => docTagsMap.get(d.id))),
    [statusFilteredDocs, docTagsMap],
  )

  // ── Sitemap content-type filter ──────────────────────────────────────────
  const contentFilteredRows = useMemo(() => {
    if (contentFilter === 'all') return hostRows
    return hostRows.filter((r) => {
      const ext  = (r.extension || '').toLowerCase().replace(/^\./, '')
      const kind = (r.content_kind || '').toLowerCase()
      switch (contentFilter) {
        case 'html': return kind === 'html' || ext === 'html' || ext === 'htm' || ext === ''
        case 'pdf':  return kind === 'pdf'  || ext === 'pdf'
        case 'doc':  return ['docx', 'doc', 'xlsx', 'xls', 'pptx', 'ppt'].includes(ext)
        default:     return true
      }
    })
  }, [hostRows, contentFilter])

  // ── Sitemap tag filter ───────────────────────────────────────────────────
  const filteredRows = useMemo(() => {
    if (!sitemapTagFilter) return contentFilteredRows
    return contentFilteredRows.filter((r) =>
      r.topic_tags?.includes(sitemapTagFilter),
    )
  }, [contentFilteredRows, sitemapTagFilter])

  // Tag counts for the sitemap tab (from content-filtered set)
  const sitemapTagList = useMemo(
    () => tagCounts(contentFilteredRows.map((r) => r.topic_tags)),
    [contentFilteredRows],
  )

  // ── Pipeline counts ──────────────────────────────────────────────────────
  const counts = {
    scanned:   hostRows.length,
    ingested:  docsForHost.length,
    chunked:   docsForHost.filter((d) => d.chunking_status === 'completed').length,
    embedded:  docsForHost.filter((d) => d.embedding_status === 'completed').length,
    published: docsForHost.filter((d) => !!d.published_at).length,
    failed:    docsForHost.filter(
      (d) => d.embedding_status === 'failed' || d.chunking_status === 'failed',
    ).length,
  }

  const stages: { key: keyof typeof counts; label: string; terminal?: boolean; failed?: boolean }[] = [
    { key: 'scanned',   label: 'Scanned' },
    { key: 'ingested',  label: 'Ingested' },
    { key: 'chunked',   label: 'Chunked' },
    { key: 'embedded',  label: 'Embedded' },
    { key: 'published', label: 'Available in chat', terminal: true },
    ...(counts.failed > 0
      ? [{ key: 'failed' as const, label: 'Failed', failed: true }]
      : []),
  ]

  const docStatusCounts = {
    all:        docsForHost.length,
    published:  docsForHost.filter((d) => !!d.published_at).length,
    failed:     counts.failed,
    processing: docsForHost.length - docsForHost.filter((d) => !!d.published_at).length - counts.failed,
  }

  return (
    <div className="entity-card">
      {/* ── Header ──────────────────────────────────────────────────── */}
      <div className="entity-card-header">
        <h3 className="entity-card-title">{host}</h3>
        {payerLabel && <span className="entity-card-payer">{payerLabel}</span>}
        <label className="entity-card-autopub" title="Auto-publish to chat after ingest.">
          <input
            type="checkbox"
            checked={autoPublish}
            onChange={(e) => setAutoPublish(e.target.checked)}
          />
          <span>Auto-publish</span>
        </label>
      </div>

      {/* ── Pipeline strip ──────────────────────────────────────────── */}
      <div className="entity-pipeline" role="group" aria-label="Pipeline status">
        {stages.map((s, i) => (
          <div key={s.key} className="entity-pipeline-stage-wrap">
            {s.failed && <span className="entity-pipeline-arrow" aria-hidden>·</span>}
            <div
              className={[
                'entity-pipeline-stage',
                counts[s.key] > 0 ? 'has-count' : '',
                s.terminal ? 'is-terminal' : '',
                s.failed ? 'is-failed' : '',
              ].filter(Boolean).join(' ')}
            >
              <span className="entity-pipeline-count">{counts[s.key]}</span>
              <span className="entity-pipeline-label">{s.label}</span>
            </div>
            {!s.failed && i < stages.length - 1 && !stages[i + 1]?.failed && (
              <span className="entity-pipeline-arrow" aria-hidden>→</span>
            )}
          </div>
        ))}
      </div>

      {/* ── Tabs ────────────────────────────────────────────────────── */}
      <div className="entity-tabs" role="tablist">
        <button
          role="tab"
          aria-selected={activeTab === 'docs'}
          className={`entity-tab${activeTab === 'docs' ? ' active' : ''}`}
          onClick={() => setActiveTab('docs')}
        >
          Documents
          <span className="entity-tab-count">{docsForHost.length}</span>
        </button>
        <button
          role="tab"
          aria-selected={activeTab === 'sitemap'}
          className={`entity-tab${activeTab === 'sitemap' ? ' active' : ''}`}
          onClick={() => setActiveTab('sitemap')}
        >
          Sitemap
          <span className="entity-tab-count">{hostRows.length}</span>
        </button>
      </div>

      {/* ── Documents tab ───────────────────────────────────────────── */}
      {activeTab === 'docs' && (
        <div className="entity-tab-panel">
          {docsForHost.length > 0 && (
            <>
              {/* Status filter */}
              <div className="entity-filter-row" role="group" aria-label="Document status filter">
                {DOC_STATUS_FILTERS.map(({ key, label }) => {
                  const n = docStatusCounts[key]
                  if (key !== 'all' && n === 0) return null
                  return (
                    <button
                      key={key}
                      className={`entity-filter-btn${docStatusFilter === key ? ' active' : ''}${key === 'failed' ? ' danger' : ''}`}
                      onClick={() => setDocStatusFilter(key)}
                    >
                      {label}
                      {key !== 'all' && <span className="entity-filter-count">{n}</span>}
                    </button>
                  )
                })}
              </div>

              {/* Topic tag filter */}
              {docTagList.length > 0 && (
                <div className="entity-filter-row entity-tag-row" role="group" aria-label="Topic filter">
                  <span className="entity-filter-label">Topic</span>
                  {docTagFilter && (
                    <button
                      className="entity-filter-btn active"
                      onClick={() => setDocTagFilter(null)}
                      title="Clear tag filter"
                    >
                      {docTagFilter} ×
                    </button>
                  )}
                  {!docTagFilter && docTagList.slice(0, 10).map(({ tag, count }) => (
                    <button
                      key={tag}
                      className="entity-filter-btn entity-tag-btn"
                      onClick={() => setDocTagFilter(tag)}
                    >
                      {tag}
                      <span className="entity-filter-count">{count}</span>
                    </button>
                  ))}
                  {!docTagFilter && docTagList.length > 10 && (
                    <span className="entity-filter-more">+{docTagList.length - 10} more</span>
                  )}
                </div>
              )}
            </>
          )}

          <IngestedDocsList
            docs={filteredDocs}
            selectedDocumentId={selectedDocumentId}
            onSelect={onSelectDoc}
            tagLabels={docTagsMap}
          />
        </div>
      )}

      {/* ── Sitemap tab ─────────────────────────────────────────────── */}
      {activeTab === 'sitemap' && (
        <div className="entity-tab-panel">
          {hostRows.length > 0 && (
            <>
              {/* Content-type filter */}
              <div className="entity-filter-row" role="group" aria-label="Content type filter">
                {CONTENT_FILTERS.map(({ key, label }) => (
                  <button
                    key={key}
                    className={`entity-filter-btn${contentFilter === key ? ' active' : ''}`}
                    onClick={() => setContentFilter(key)}
                  >
                    {label}
                    {key !== 'all' && (
                      <span className="entity-filter-count">
                        {hostRows.filter((r) => {
                          const ext  = (r.extension || '').toLowerCase().replace(/^\./, '')
                          const kind = (r.content_kind || '').toLowerCase()
                          if (key === 'html') return kind === 'html' || ext === 'html' || ext === 'htm' || ext === ''
                          if (key === 'pdf')  return kind === 'pdf' || ext === 'pdf'
                          if (key === 'doc')  return ['docx','doc','xlsx','xls','pptx','ppt'].includes(ext)
                          return true
                        }).length}
                      </span>
                    )}
                  </button>
                ))}
              </div>

              {/* Topic tag filter */}
              {sitemapTagList.length > 0 && (
                <div className="entity-filter-row entity-tag-row" role="group" aria-label="Topic filter">
                  <span className="entity-filter-label">Topic</span>
                  {sitemapTagFilter && (
                    <button
                      className="entity-filter-btn active"
                      onClick={() => setSitemapTagFilter(null)}
                      title="Clear tag filter"
                    >
                      {sitemapTagFilter} ×
                    </button>
                  )}
                  {!sitemapTagFilter && sitemapTagList.slice(0, 10).map(({ tag, count }) => (
                    <button
                      key={tag}
                      className="entity-filter-btn entity-tag-btn"
                      onClick={() => setSitemapTagFilter(tag)}
                    >
                      {tag}
                      <span className="entity-filter-count">{count}</span>
                    </button>
                  ))}
                  {!sitemapTagFilter && sitemapTagList.length > 10 && (
                    <span className="entity-filter-more">+{sitemapTagList.length - 10} more</span>
                  )}
                </div>
              )}
            </>
          )}

          {hostLoading && <div className="loading">Loading sitemap…</div>}

          {!hostLoading && filteredRows.length > 0 && (
            <SourceTreeView
              host={host}
              payerLabel={hostRows[0]?.payer || payerLabel}
              rows={filteredRows}
              onIngest={handleIngest}
              onBulkIngest={handleBulkIngest}
              ingestingUrls={ingestingUrls}
              defaultCollapsed={true}
              hideHeader={true}
            />
          )}

          {!hostLoading && filteredRows.length === 0 && hostRows.length > 0 && (
            <div className="empty entity-tab-empty">No URLs match this filter.</div>
          )}

          {!hostLoading && hostRows.length === 0 && (
            <div className="empty entity-tab-empty">
              No URLs in registry for <code>{host}</code>.
            </div>
          )}
        </div>
      )}

      {/* URL preview hint */}
      {selectedUrl && (
        <div className="entity-card-url-hint">
          Previewing: <code>{selectedUrl}</code>
          <button type="button" className="btn btn-small" onClick={() => onSelectUrl(selectedUrl, false)}>
            Ingest
          </button>
        </div>
      )}
    </div>
  )
}
