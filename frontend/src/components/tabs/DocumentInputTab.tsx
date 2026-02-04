import { useState, useEffect, useCallback, useRef } from 'react'
import { API_BASE, SCRAPER_API_BASE } from '../../config'
import './DocumentInputTab.css'

interface DocumentInputTabProps {
  onUpload: (file: File) => Promise<void>
  uploading: boolean
  error: string | null
  onDocumentAdded?: () => void
}

interface ScrapedDoc {
  gcs_path: string
  filename: string
  content_type?: string
  source_url?: string
  size_bytes?: number
  source_page_url?: string
}

interface ScrapedPage {
  url: string
  text?: string
  html?: string
  final_url?: string
  depth?: number
  timestamp?: string
}

interface ScraperEvent {
  type: 'progress' | 'page' | 'document' | 'summary' | 'done'
  metadata: { timestamp?: string; job_id?: string; source_url?: string; final_url?: string; depth?: number; source_page_url?: string; content_type?: string; size_bytes?: number }
  payload?: Record<string, unknown>
}

export function DocumentInputTab({ onUpload, uploading, error, onDocumentAdded }: DocumentInputTabProps) {
  const [file, setFile] = useState<File | null>(null)
  const [dragActive, setDragActive] = useState(false)

  // Scrape from URL state
  const [scrapeUrl, setScrapeUrl] = useState('')
  const [scrapeMode, setScrapeMode] = useState<'regular' | 'tree'>('regular')
  const [scrapeMaxDepth, setScrapeMaxDepth] = useState(3)
  const [scrapeMaxPages, setScrapeMaxPages] = useState(50)
  const [scraping, setScraping] = useState(false)
  const [scrapeJobId, setScrapeJobId] = useState<string | null>(null)
  const [scrapeStatus, setScrapeStatus] = useState<'pending' | 'running' | 'completed' | 'failed' | null>(null)
  const [scrapeDocuments, setScrapeDocuments] = useState<ScrapedDoc[]>([])
  const [scrapePages, setScrapePages] = useState<ScrapedPage[]>([])
  const [scrapeSummary, setScrapeSummary] = useState<string | null>(null)
  const [scrapeContentMode, setScrapeContentMode] = useState<'text' | 'html' | 'both'>('text')
  const [scrapeDownloadDocuments, setScrapeDownloadDocuments] = useState(true)
  const [scrapeIncludeSummary, setScrapeIncludeSummary] = useState(false)
  const [scrapeError, setScrapeError] = useState<string | null>(null)
  const [importingDoc, setImportingDoc] = useState<string | null>(null)
  const [scrapeEvents, setScrapeEvents] = useState<ScraperEvent[]>([])
  const [scrapeProgressMessage, setScrapeProgressMessage] = useState<string | null>(null)
  const eventSourceRef = useRef<EventSource | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
    }
  }

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0])
    }
  }

  const handleUpload = async () => {
    if (file) {
      await onUpload(file)
      setFile(null)
      // Reset file input
      const fileInput = document.getElementById('file-input') as HTMLInputElement
      if (fileInput) {
        fileInput.value = ''
      }
    }
  }

  const startScrape = async () => {
    if (!scrapeUrl.trim()) return
    setScrapeError(null)
    setScraping(true)
    setScrapeStatus(null)
    setScrapeDocuments([])
    setScrapePages([])
    setScrapeSummary(null)
    setScrapeEvents([])
    setScrapeProgressMessage(null)
    try {
      const body: Record<string, unknown> = {
        url: scrapeUrl.trim(),
        mode: scrapeMode,
        content_mode: scrapeContentMode,
        download_documents: scrapeDownloadDocuments,
        summarize: scrapeIncludeSummary,
      }
      if (scrapeMode === 'tree') {
        body.max_depth = scrapeMaxDepth
        body.max_pages = scrapeMaxPages
      }
      const resp = await fetch(`${SCRAPER_API_BASE}/scrape`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}))
        throw new Error(err.detail || resp.statusText || 'Scrape request failed')
      }
      const { job_id } = await resp.json()
      setScrapeJobId(job_id)
      setScrapeStatus('pending')
    } catch (e) {
      setScrapeError(e instanceof Error ? e.message : 'Scrape failed')
      setScraping(false)
    }
  }

  const pollScrapeStatus = useCallback(async () => {
    if (!scrapeJobId) return
    try {
      const resp = await fetch(`${SCRAPER_API_BASE}/scrape/${scrapeJobId}`)
      if (!resp.ok) return
      const data = await resp.json()
      setScrapeStatus(data.status)
      setScrapeDocuments(data.documents || [])
      setScrapePages(data.pages || [])
      setScrapeSummary(data.summary || null)
      if (data.error) setScrapeError(data.error)
      if (data.status === 'completed' || data.status === 'failed') {
        setScraping(false)
      }
    } catch {
      // Ignore poll errors
    }
  }, [scrapeJobId])

  // Subscribe to SSE stream when job starts
  useEffect(() => {
    if (!scrapeJobId) return
    const url = `${SCRAPER_API_BASE}/scrape/${scrapeJobId}/stream`
    const es = new EventSource(url)
    eventSourceRef.current = es
    es.onmessage = (e) => {
      try {
        const evt: ScraperEvent = JSON.parse(e.data)
        setScrapeEvents((prev) => [...prev, evt])
        if (evt.type === 'progress' && evt.payload?.message) {
          setScrapeProgressMessage(String(evt.payload.message))
        }
        if (evt.type === 'done') {
          setScraping(false)
          const status = evt.payload?.status as string
          setScrapeStatus(status === 'failed' ? 'failed' : 'completed')
          if (evt.payload?.error) setScrapeError(String(evt.payload.error))
          es.close()
          eventSourceRef.current = null
        }
      } catch {
        // ignore parse errors
      }
    }
    es.onerror = () => {
      es.close()
      eventSourceRef.current = null
    }
    return () => {
      es.close()
      eventSourceRef.current = null
    }
  }, [scrapeJobId])

  useEffect(() => {
    if (!scrapeJobId || !scraping || scrapeStatus === 'completed' || scrapeStatus === 'failed') return
    const id = setInterval(pollScrapeStatus, 2000)
    return () => clearInterval(id)
  }, [scrapeJobId, scraping, scrapeStatus, pollScrapeStatus])

  const importToRag = async (gcsPath: string, filename: string) => {
    setImportingDoc(gcsPath)
    try {
      const resp = await fetch(`${API_BASE}/documents/import-from-gcs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gcs_path: gcsPath, filename }),
      })
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}))
        throw new Error(err.detail?.message || err.detail || 'Import failed')
      }
      onDocumentAdded?.()
    } catch (e) {
      setScrapeError(e instanceof Error ? e.message : 'Import failed')
    } finally {
      setImportingDoc(null)
    }
  }

  // Derive display data from events or poll response
  const displayDocs = scrapeEvents.length
    ? scrapeEvents.filter((e) => e.type === 'document').map((e) => ({
        gcs_path: (e.payload?.gcs_path as string) || '',
        filename: (e.payload?.filename as string) || '',
        content_type: e.metadata?.content_type,
        source_url: e.metadata?.source_url,
        size_bytes: e.metadata?.size_bytes,
        source_page_url: e.metadata?.source_page_url,
      }))
    : scrapeDocuments
  const displayPages = scrapeEvents.length
    ? scrapeEvents.filter((e) => e.type === 'page').map((e) => ({
        url: (e.payload?.url as string) || e.metadata?.source_url || '',
        text: e.payload?.text as string | undefined,
        html: e.payload?.html as string | undefined,
        final_url: e.metadata?.final_url,
        depth: e.metadata?.depth,
        timestamp: e.metadata?.timestamp,
      }))
    : scrapePages
  const displaySummary =
    scrapeEvents.length > 0
      ? (scrapeEvents.find((e) => e.type === 'summary')?.payload?.summary as string | undefined) ?? scrapeSummary
      : scrapeSummary

  return (
    <div className="document-input-tab">
      {/* Top half: Upload methods */}
      <div className="input-methods-top">
        {/* Upload Method */}
        <details className="input-method-card active collapsible" open>
          <summary className="collapsible-summary">
            <span className="collapsible-title">Upload Document</span>
            <span className="collapsible-desc">Upload a PDF document from your computer</span>
          </summary>
          <div className="collapsible-content">
          <div
            className={`upload-area ${dragActive ? 'drag-active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              type="file"
              onChange={handleFileChange}
              id="file-input"
              accept=".pdf"
              disabled={uploading}
            />
            <label htmlFor="file-input" className="file-label">
              {file ? file.name : 'Choose a file or drag it here'}
            </label>
            {file && (
              <div className="file-info">
                <span className="file-size">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </span>
              </div>
            )}
            <button
              onClick={handleUpload}
              disabled={!file || uploading}
              className="btn btn-primary upload-button"
            >
              {uploading ? 'Uploading…' : 'Upload'}
            </button>
          </div>
          
          {error && (
            <div className="error-message" role="alert">
              {error}
            </div>
          )}
          <p className="upload-hint services-hint">
            <strong>RAG</strong> = <code>{API_BASE}</code> (port 8001, documents &amp; chunking). <strong>Scraper</strong> = <code>{SCRAPER_API_BASE}</code> (port 8002, Scrape from URL). Start from Module Hub (mstart) or run RAG backend: <code>cd mobius-rag && .venv/bin/python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8001</code>. PDF only. Upload issues: check GCS in <code>.env</code>.
          </p>
          </div>
        </details>
      </div>

      {/* Bottom half: Scrape from URL (full-width) */}
      <details className="scrape-section collapsible" open>
        <summary className="collapsible-summary">
          <span className="collapsible-title">Scrape from URL</span>
          <span className="collapsible-desc">Extract content and download documents from a webpage</span>
          {scraping && <span className="collapsible-badge">Scraping…</span>}
          {(scrapeStatus === 'completed' || scrapeStatus === 'failed') && !scraping && (
            <span className={`collapsible-badge ${scrapeStatus === 'failed' ? 'badge-error' : 'badge-ok'}`}>
              {scrapeStatus === 'completed' ? 'Done' : 'Failed'}
            </span>
          )}
        </summary>
        <div className="collapsible-content">
          <div className="scrape-form">
            <input
              type="url"
              placeholder="https://example.com"
              value={scrapeUrl}
              onChange={(e) => setScrapeUrl(e.target.value)}
              className="scrape-url-input"
              disabled={scraping}
            />
            <div className="scrape-options">
              <label>
                <input
                  type="radio"
                  name="scrapeMode"
                  checked={scrapeMode === 'regular'}
                  onChange={() => setScrapeMode('regular')}
                  disabled={scraping}
                />
                Regular scan (single page)
              </label>
              <label>
                <input
                  type="radio"
                  name="scrapeMode"
                  checked={scrapeMode === 'tree'}
                  onChange={() => setScrapeMode('tree')}
                  disabled={scraping}
                />
                Tree scan (follow links)
              </label>
            </div>
            {scrapeMode === 'tree' && (
              <div className="scrape-tree-options">
                <label>
                  Max depth: <input type="number" min={1} max={10} value={scrapeMaxDepth} onChange={(e) => setScrapeMaxDepth(parseInt(e.target.value, 10) || 3)} disabled={scraping} className="scrape-number" />
                </label>
                <label>
                  Max pages: <input type="number" min={1} max={200} value={scrapeMaxPages} onChange={(e) => setScrapeMaxPages(parseInt(e.target.value, 10) || 50)} disabled={scraping} className="scrape-number" />
                </label>
              </div>
            )}
            <div className="scrape-content-options">
              <label>
                Content mode:
                <select
                  value={scrapeContentMode}
                  onChange={(e) => setScrapeContentMode(e.target.value as 'text' | 'html' | 'both')}
                  disabled={scraping}
                  className="scrape-select"
                >
                  <option value="text">Text only</option>
                  <option value="html">HTML only</option>
                  <option value="both">Both</option>
                </select>
              </label>
            </div>
            <label className="scrape-summary-option">
              <input
                type="checkbox"
                checked={scrapeDownloadDocuments}
                onChange={(e) => setScrapeDownloadDocuments(e.target.checked)}
                disabled={scraping}
              />
              Download documents (PDF, DOC, etc.)
            </label>
            <label className="scrape-summary-option">
              <input
                type="checkbox"
                checked={scrapeIncludeSummary}
                onChange={(e) => setScrapeIncludeSummary(e.target.checked)}
                disabled={scraping}
              />
              Include summary (requires OPENAI_API_KEY or Vertex in scraper)
            </label>
            <button
              onClick={startScrape}
              disabled={!scrapeUrl.trim() || scraping}
              className="btn btn-primary upload-button"
            >
              {scraping ? (scrapeStatus === 'running' ? 'Scraping…' : 'Waiting…') : 'Start scrape'}
            </button>
          </div>
          {scrapeError && (
            <div className="error-message" role="alert">
              {scrapeError}
            </div>
          )}
        {scraping && scrapeProgressMessage && (
          <div className="scrape-progress">
            <span className="scrape-progress-dot" />
            {scrapeProgressMessage}
          </div>
        )}
        {(scrapeStatus === 'completed' || scrapeStatus === 'failed') && (
          <div className="scrape-results">
            {displaySummary && (
              <div className="scrape-summary-block">
                <p className="scrape-results-title">Summary</p>
                <div className="scrape-summary-text">{displaySummary}</div>
              </div>
            )}
            {displayPages.length > 0 && (
              <div className="scrape-pages-block">
                <p className="scrape-results-title">Page content ({displayPages.length} page{displayPages.length !== 1 ? 's' : ''})</p>
                {displayPages.slice(0, 5).map((p, i) => {
                  const content = p.text ?? p.html ?? '(no content)'
                  const preview = content.slice(0, 2000) + (content.length > 2000 ? '...' : '')
                  return (
                    <details key={i} className="scrape-page-details">
                      <summary>
                        <span className="scrape-page-url">{p.url}</span>
                        {p.depth != null && <span className="scrape-page-meta">depth {p.depth}</span>}
                        {p.timestamp && <span className="scrape-page-meta">{p.timestamp}</span>}
                      </summary>
                      <pre className="scrape-page-text">{preview}</pre>
                    </details>
                  )
                })}
                {displayPages.length > 5 && <p className="scrape-more">... and {displayPages.length - 5} more pages</p>}
              </div>
            )}
            {displayDocs.length > 0 && (
              <>
                <p className="scrape-results-title">Downloaded documents:</p>
                <ul className="scrape-doc-list">
                  {displayDocs.map((d) => (
                    <li key={d.gcs_path}>
                      <div className="scrape-doc-info">
                        <span>{d.filename}</span>
                        {(d.size_bytes != null || d.content_type || d.source_page_url) && (
                          <span className="scrape-doc-meta">
                            {d.size_bytes != null && `${(d.size_bytes / 1024).toFixed(1)} KB`}
                            {d.content_type && ` · ${d.content_type}`}
                            {d.source_page_url && ` · from ${d.source_page_url}`}
                          </span>
                        )}
                      </div>
                      <button
                        type="button"
                        className="btn btn-small"
                        onClick={() => importToRag(d.gcs_path, d.filename)}
                        disabled={importingDoc === d.gcs_path}
                      >
                        {importingDoc === d.gcs_path ? 'Importing…' : 'Add to RAG'}
                      </button>
                    </li>
                  ))}
                </ul>
                <p className="upload-hint">PDF documents can be added to RAG for chunking and embedding.</p>
              </>
            )}
            {displayDocs.length === 0 && displayPages.length === 0 && !displaySummary && (
              <div className="scrape-empty-state">
                <p className="scrape-empty-title">No documents or page content extracted</p>
                <p className="scrape-empty-desc">
                  The scraper completed but found no downloadable files (PDF, DOC, etc.) and no extractable text from the page.
                  This often happens with JavaScript-heavy sites—the scraper fetches static HTML; content loaded by JS is not captured.
                  Try a simpler page, or use <strong>content mode: HTML</strong> to capture raw HTML.
                </p>
              </div>
            )}
          </div>
        )}
        <p className="upload-hint">
          Scraper API at <code>{SCRAPER_API_BASE}</code>. Ensure scraper API and worker are running.
        </p>
        </div>
      </details>

      <div className="input-methods-top" style={{ marginTop: '1.5rem' }}>
        {/* Google Drive Method - Coming Soon */}
        <details className="input-method-card disabled collapsible">
          <summary className="collapsible-summary">
            <span className="collapsible-title">Google Drive</span>
            <span className="collapsible-desc">Import documents from Google Drive</span>
            <span className="collapsible-badge">Coming Soon</span>
          </summary>
          <div className="collapsible-content">
            <div className="coming-soon-badge">Coming Soon</div>
          </div>
        </details>

        {/* SFTP to GCP Method - Coming Soon */}
        <details className="input-method-card disabled collapsible">
          <summary className="collapsible-summary">
            <span className="collapsible-title">SFTP to GCP</span>
            <span className="collapsible-desc">Sync documents via SFTP to Google Cloud Storage</span>
            <span className="collapsible-badge">Coming Soon</span>
          </summary>
          <div className="collapsible-content">
            <div className="coming-soon-badge">Coming Soon</div>
          </div>
        </details>
      </div>
    </div>
  )
}
