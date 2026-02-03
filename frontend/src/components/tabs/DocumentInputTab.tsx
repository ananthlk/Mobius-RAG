import { useState, useEffect, useCallback } from 'react'
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
}

interface ScrapedPage {
  url: string
  text: string
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
  const [scrapeIncludeSummary, setScrapeIncludeSummary] = useState(false)
  const [scrapeError, setScrapeError] = useState<string | null>(null)
  const [importingDoc, setImportingDoc] = useState<string | null>(null)

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
    try {
      const body: Record<string, unknown> = { url: scrapeUrl.trim(), mode: scrapeMode }
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

  return (
    <div className="document-input-tab">
      <div className="input-methods">
        {/* Upload Method */}
        <div className="input-method-card active">
          <h3 className="method-title">Upload Document</h3>
          <p className="method-description">Upload a PDF document from your computer</p>
          
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

        {/* Scrape from URL */}
        <div className="input-method-card active">
          <h3 className="method-title">Scrape from URL</h3>
          <p className="method-description">Extract content and download documents from a webpage</p>
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
          {scrapeStatus === 'completed' && (scrapeDocuments.length > 0 || scrapePages.length > 0 || scrapeSummary) && (
            <div className="scrape-results">
              {scrapeSummary && (
                <div className="scrape-summary-block">
                  <p className="scrape-results-title">Summary</p>
                  <div className="scrape-summary-text">{scrapeSummary}</div>
                </div>
              )}
              {scrapePages.length > 0 && (
                <div className="scrape-pages-block">
                  <p className="scrape-results-title">Page content ({scrapePages.length} page{scrapePages.length !== 1 ? 's' : ''})</p>
                  {scrapePages.slice(0, 5).map((p, i) => (
                    <details key={i} className="scrape-page-details">
                      <summary>{p.url}</summary>
                      <pre className="scrape-page-text">{p.text.slice(0, 2000)}{p.text.length > 2000 ? '...' : ''}</pre>
                    </details>
                  ))}
                  {scrapePages.length > 5 && <p className="scrape-more">... and {scrapePages.length - 5} more pages</p>}
                </div>
              )}
              {scrapeDocuments.length > 0 && (
                <>
              <p className="scrape-results-title">Downloaded documents:</p>
              <ul className="scrape-doc-list">
                {scrapeDocuments.map((d) => (
                  <li key={d.gcs_path}>
                    <span>{d.filename}</span>
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
            </div>
          )}
          <p className="upload-hint">
            Scraper API at <code>{SCRAPER_API_BASE}</code>. Ensure scraper API and worker are running.
          </p>
        </div>

        {/* Google Drive Method - Coming Soon */}
        <div className="input-method-card disabled">
          <h3 className="method-title">Google Drive</h3>
          <p className="method-description">Import documents from Google Drive</p>
          <div className="coming-soon-badge">Coming Soon</div>
        </div>

        {/* SFTP to GCP Method - Coming Soon */}
        <div className="input-method-card disabled">
          <h3 className="method-title">SFTP to GCP</h3>
          <p className="method-description">Sync documents via SFTP to Google Cloud Storage</p>
          <div className="coming-soon-badge">Coming Soon</div>
        </div>
      </div>
    </div>
  )
}
