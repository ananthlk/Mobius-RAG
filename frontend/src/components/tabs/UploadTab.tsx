import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { API_BASE, SCRAPER_API_BASE } from '../../config'
import { AUTHORITY_LEVEL_OPTIONS } from '../../lib/documentMetadata'
import './UploadTab.css'

interface DocLike {
  id: string
  filename: string
  display_name?: string | null
  created_at?: string
  extraction_status?: string
  chunking_status?: string | null
  embedding_status?: string | null
  chunking_total_paragraphs?: number
  chunking_completed_paragraphs?: number
  chunking_total_pages?: number
  chunking_current_page?: number
  published_at?: string | null
}

interface ProbeResult {
  url: string
  host: string
  fetch: { status: number; content_type: string | null; redirected_to: string | null }
  sitemap: { url: string; status: number; url_count: number; sample: string[] }
  robots?: { status: number; preview: string }
  classifier?: {
    host: string
    path: string
    payer: string | null
    state: string | null
    inferred_authority_level: string | null
    content_kind: string
    extension: string | null
  }
  recommended_strategy: 'scrape' | 'sitemap_only' | 'state_mirror' | 'manual_upload'
  recommended_reason: string
  mirror_suggestion?: {
    suggested_url: string
    search_url: string
    help: string
  } | null
}

interface DriveFile {
  id: string
  name: string
  mimeType?: string
  size?: string
  webViewLink?: string
}

interface DriveFolder {
  id: string
  name: string
}

interface Props {
  documents: DocLike[]
  onUpload: (file: File) => Promise<void>
  uploading: boolean
  error: string | null
  onDocumentAdded: () => void
}

interface QueueRow {
  id: string
  filename: string
  status: string
  stage: 'scraping' | 'extracting' | 'chunking' | 'embedding' | 'completed' | 'failed' | 'pending'
  progressPct: number
  recentlyCompleted: boolean
  meta?: string
}

interface ScrapeJob {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  url?: string | null
  pages_scraped?: number
  max_pages?: number | null
  documents_count?: number
  started_at?: number | null
  finished_at?: number | null
  error?: string | null
}

const ONE_HOUR_MS = 60 * 60 * 1000

type SourceType = 'computer' | 'url' | 'drive'
// All four strategies map to one of two API codepaths:
//   scrape           → POST /scrape (scraper service)
//   everything else  → POST /sources/upsert (rag service)
// state_mirror differs from manual_upload semantically: instead of
// uploading PDFs by hand, we register a *replacement URL* (typically
// the state agency's mirror, e.g. AHCA for FL Medicaid plans) when
// the original site is bot-walled.
type Strategy = 'scrape' | 'sitemap_only' | 'state_mirror' | 'manual_upload'

/** Derive queue rows from active scrape jobs (scraper-side). */
function scrapeJobsToRows(jobs: ScrapeJob[]): QueueRow[] {
  const now = Date.now()
  const rows: QueueRow[] = []
  for (const j of jobs) {
    const isActive = j.status === 'pending' || j.status === 'running'
    const finished = j.finished_at ? j.finished_at * 1000 : 0
    const recent = !isActive && finished > 0 && now - finished < ONE_HOUR_MS
    if (!isActive && !recent) continue

    let pct = 0
    if (j.max_pages && j.pages_scraped != null) {
      pct = Math.min(100, Math.round((j.pages_scraped / j.max_pages) * 100))
    } else if (!isActive) {
      pct = 100
    }

    let host = ''
    try {
      if (j.url) host = new URL(j.url).hostname.replace(/^www\./, '')
    } catch {
      // ignore — fall back to raw url
    }
    const filename = host || j.url || j.job_id.slice(0, 8)

    rows.push({
      id: `scrape:${j.job_id}`,
      filename: `Scrape · ${filename}`,
      status: j.status,
      stage:
        j.status === 'failed' ? 'failed' :
        isActive ? 'scraping' :
        'completed',
      progressPct: pct,
      recentlyCompleted: recent,
      meta:
        j.status === 'failed' ? (j.error || 'failed') :
        isActive
          ? `${j.pages_scraped || 0}${j.max_pages ? ` / ${j.max_pages}` : ''} pages`
          : `${j.pages_scraped || 0} pages · ${j.documents_count || 0} docs`,
    })
  }
  return rows
}

/** Derive the in-flight queue from the documents list. */
function deriveQueue(documents: DocLike[]): QueueRow[] {
  const now = Date.now()
  const rows: QueueRow[] = []
  for (const d of documents) {
    const ext = d.extraction_status || 'pending'
    const chk = d.chunking_status || 'idle'
    const emb = d.embedding_status || 'idle'

    let stage: QueueRow['stage'] = 'pending'
    let pct = 0
    let active = false

    if (ext === 'extracting' || ext === 'pending') {
      stage = 'extracting'
      active = true
    } else if (emb === 'processing' || emb === 'pending') {
      stage = 'embedding'
      active = true
      const total = d.chunking_total_paragraphs ?? 0
      const done = d.chunking_completed_paragraphs ?? 0
      pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0
    } else if (chk === 'in_progress' || chk === 'pending') {
      stage = 'chunking'
      active = true
      const total = d.chunking_total_paragraphs ?? 0
      const done = d.chunking_completed_paragraphs ?? 0
      pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0
    } else if (emb === 'completed' || chk === 'completed' || ext === 'completed') {
      stage = 'completed'
      pct = 100
    } else if (chk === 'failed' || emb === 'failed' || ext === 'failed') {
      stage = 'failed'
    }

    let recent = false
    if (stage === 'completed' && d.created_at) {
      const ts = Date.parse(d.created_at)
      if (!Number.isNaN(ts) && now - ts < ONE_HOUR_MS) recent = true
    }

    if (active || recent) {
      rows.push({
        id: d.id,
        filename: d.display_name?.trim() || d.filename,
        status: stage === 'completed' ? 'completed' : stage,
        stage,
        progressPct: pct,
        recentlyCompleted: recent,
      })
    }
  }
  return rows
}

/**
 * Upload tab — single source-selector row (Computer / URL / Drive),
 * one panel per source. Right side: in-flight queue.
 */
export function UploadTab({ documents, onUpload, uploading, error, onDocumentAdded }: Props) {
  const [sourceType, setSourceType] = useState<SourceType>('computer')

  // Poll scraper-side active jobs so the in-flight queue surfaces
  // running scrapes (otherwise they'd be invisible until the downstream
  // chunk/embed jobs land in /documents). 5s cadence — cheap.
  const [scrapeJobs, setScrapeJobs] = useState<ScrapeJob[]>([])
  useEffect(() => {
    let cancelled = false
    const fetchActive = async () => {
      if (!SCRAPER_API_BASE) return
      try {
        const r = await fetch(`${SCRAPER_API_BASE}/jobs/active`)
        if (!r.ok) return
        const data = await r.json() as { jobs: ScrapeJob[] }
        if (!cancelled) setScrapeJobs(data.jobs || [])
      } catch {
        // Silent — scraper unreachable shouldn't break the Upload tab.
      }
    }
    fetchActive()
    const interval = setInterval(fetchActive, 5000)
    return () => { cancelled = true; clearInterval(interval) }
  }, [])

  const queue = useMemo(() => {
    // Scrape jobs first (they're upstream of doc rows), then doc-stage rows.
    return [...scrapeJobsToRows(scrapeJobs), ...deriveQueue(documents)]
  }, [scrapeJobs, documents])

  return (
    <div className="upload-tab">
      <div className="upload-tab-grid">
        <div className="upload-card">
          <div className="upload-source-selector" role="tablist" aria-label="Upload source">
            <button
              type="button"
              role="tab"
              aria-selected={sourceType === 'computer'}
              className={`upload-source-btn ${sourceType === 'computer' ? 'active' : ''}`}
              onClick={() => setSourceType('computer')}
            >
              <span className="upload-source-icon" aria-hidden>📁</span>
              Computer
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={sourceType === 'url'}
              className={`upload-source-btn ${sourceType === 'url' ? 'active' : ''}`}
              onClick={() => setSourceType('url')}
            >
              <span className="upload-source-icon" aria-hidden>🌐</span>
              URL
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={sourceType === 'drive'}
              className={`upload-source-btn ${sourceType === 'drive' ? 'active' : ''}`}
              onClick={() => setSourceType('drive')}
            >
              <span className="upload-source-icon" aria-hidden>☁</span>
              Drive
            </button>
          </div>

          <div className="upload-source-panel">
            {sourceType === 'computer' && (
              <ComputerPanel
                onUpload={onUpload}
                uploading={uploading}
                error={error}
              />
            )}
            {sourceType === 'url' && (
              <UrlPanel onDocumentAdded={onDocumentAdded} />
            )}
            {sourceType === 'drive' && (
              <DrivePanel onDocumentAdded={onDocumentAdded} />
            )}
          </div>
        </div>

        <div className="upload-queue-card">
          <h3 className="upload-card-title">
            In-flight <span className="upload-queue-count">({queue.length})</span>
          </h3>
          {queue.length === 0 ? (
            <div className="upload-queue-empty">
              No active uploads. Drop a file or paste a URL to get started.
            </div>
          ) : (
            <ul className="upload-queue-list">
              {queue.map((row) => (
                <li key={row.id} className="upload-queue-row">
                  <div className="upload-queue-row-head">
                    <span className="upload-queue-name" title={row.filename}>
                      {row.filename}
                    </span>
                    <span className={`upload-queue-pill stage-${row.stage}`}>{row.stage}</span>
                  </div>
                  <div className="upload-queue-progress">
                    <div
                      className="upload-queue-progress-bar"
                      style={{ width: `${row.progressPct}%` }}
                    />
                  </div>
                  <div className="upload-queue-meta">
                    {row.progressPct > 0 ? `${row.progressPct}%` : row.stage}
                    {row.meta && ` · ${row.meta}`}
                    {row.recentlyCompleted && ' · just completed'}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  )
}

/* ───────────────────── Computer panel ─────────────────────── */
function ComputerPanel({
  onUpload,
  uploading,
  error,
}: {
  onUpload: (file: File) => Promise<void>
  uploading: boolean
  error: string | null
}) {
  const [file, setFile] = useState<File | null>(null)
  const [dragActive, setDragActive] = useState(false)

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true)
    else if (e.type === 'dragleave') setDragActive(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0])
  }

  const handleUpload = async () => {
    if (!file) return
    await onUpload(file)
    setFile(null)
    const fileInput = document.getElementById('upload-tab-file-input') as HTMLInputElement | null
    if (fileInput) fileInput.value = ''
  }

  return (
    <div className="upload-panel">
      <div
        className={`upload-dropzone ${dragActive ? 'drag-active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          id="upload-tab-file-input"
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) setFile(e.target.files[0])
          }}
          accept=".pdf"
          disabled={uploading}
          className="upload-dropzone-input"
        />
        <label htmlFor="upload-tab-file-input" className="upload-dropzone-label">
          {file ? file.name : 'Choose a file or drag it here'}
        </label>
        {file && (
          <div className="upload-dropzone-info">
            <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
          </div>
        )}
        <button
          type="button"
          onClick={handleUpload}
          disabled={!file || uploading}
          className="btn btn-primary upload-dropzone-btn"
        >
          {uploading ? 'Uploading…' : 'Upload'}
        </button>
      </div>
      {error && (
        <div className="error-message" role="alert">
          {error}
        </div>
      )}
      <p className="upload-hint">
        PDF only. Backend infers payer / state / authority from filename — edit later in Repository.
      </p>
    </div>
  )
}

/* ───────────────────────── URL panel ──────────────────────── */
function UrlPanel({ onDocumentAdded }: { onDocumentAdded: () => void }) {
  const [url, setUrl] = useState('')
  const [probing, setProbing] = useState(false)
  const [probe, setProbe] = useState<ProbeResult | null>(null)
  const [probeError, setProbeError] = useState<string | null>(null)

  const [strategy, setStrategy] = useState<Strategy>('scrape')
  const [mirrorUrl, setMirrorUrl] = useState('')
  const [maxDepth, setMaxDepth] = useState(3)
  const [maxPages, setMaxPages] = useState(200)
  const [includeHtml, setIncludeHtml] = useState(true)
  const [includePdfs, setIncludePdfs] = useState(true)

  const [payer, setPayer] = useState('')
  const [state, setState] = useState('')
  const [authority, setAuthority] = useState<string>('')
  const [program, setProgram] = useState('')
  const [autoPublish, setAutoPublish] = useState(true)

  const [submitting, setSubmitting] = useState(false)
  const [submitMsg, setSubmitMsg] = useState<{ ok: boolean; msg: string } | null>(null)

  const handleProbe = async () => {
    if (!url.trim()) return
    setProbing(true)
    setProbeError(null)
    setProbe(null)
    setSubmitMsg(null)
    try {
      const resp = await fetch(`${API_BASE}/sources/probe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url.trim() }),
      })
      if (!resp.ok) throw new Error(`probe ${resp.status}`)
      const data = (await resp.json()) as ProbeResult
      setProbe(data)
      // Auto-fill from classifier
      const c = data.classifier
      if (c) {
        setPayer(c.payer || '')
        setState(c.state || '')
        // Map classifier inferred_authority_level (free string) into the
        // canonical 3-value enum used everywhere else (lib/documentMetadata).
        const inferred = (c.inferred_authority_level || '').toLowerCase()
        if (inferred.includes('contract') || inferred.includes('primary') || inferred.includes('authoritat')) {
          setAuthority('contract_source_of_truth')
        } else if (inferred.includes('operational') || inferred.includes('secondary') || inferred.includes('suggest')) {
          setAuthority('operational_suggested')
        } else if (inferred.includes('fyi') || inferred.includes('tertiary') || inferred.includes('informational')) {
          setAuthority('fyi_not_citable')
        } else {
          setAuthority('')
        }
      }
      setStrategy(data.recommended_strategy as Strategy)
      // Pre-fill the mirror URL with whatever the backend suggested so the
      // operator gets a clickable AHCA landing page on first paint instead
      // of a blank input.
      if (data.mirror_suggestion?.suggested_url) {
        setMirrorUrl(data.mirror_suggestion.suggested_url)
      } else {
        setMirrorUrl('')
      }
    } catch (e) {
      setProbeError(String(e))
    } finally {
      setProbing(false)
    }
  }

  const contentMode: 'text' | 'html' | 'both' =
    includeHtml && includePdfs ? 'both' : includeHtml ? 'html' : 'text'

  const handleSubmit = async () => {
    if (!url.trim() || submitting || !probe) return
    setSubmitting(true)
    setSubmitMsg(null)
    try {
      let msg = ''
      if (strategy === 'scrape') {
        const resp = await fetch(`${SCRAPER_API_BASE}/scrape`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            url: url.trim(),
            mode: 'tree',
            max_depth: maxDepth,
            max_pages: maxPages,
            scope_mode: 'same_domain',
            include_content: true,
            include_summary: false,
            content_mode: contentMode,
            download_documents: includePdfs,
          }),
        })
        const data = await resp.json()
        if (!resp.ok) throw new Error(data?.detail || `scrape ${resp.status}`)
        msg = `Scrape queued (job ${String(data.job_id || '').slice(0, 8)}). URLs land in Repository as the scraper crawls.`
      } else if (strategy === 'state_mirror') {
        // State mirror: the original URL is bot-walled; register the
        // operator-supplied mirror URL (e.g. AHCA copy) as the canonical
        // source. The original gets recorded too so future operators see
        // why we redirected.
        const mirror = mirrorUrl.trim()
        if (!mirror) throw new Error('Mirror URL is required for state_mirror strategy.')
        const resp = await fetch(`${API_BASE}/sources/upsert`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            url: mirror,
            discovered_via: 'state_mirror',
            seed_url: url.trim(),
            payer_hint: payer || null,
            state_hint: state || null,
            authority_hint: authority || null,
            program_hint: program || null,
            auto_publish: autoPublish,
            curation_notes: `Mirror of bot-walled ${url.trim()}`,
          }),
        })
        if (!resp.ok) throw new Error(`upsert ${resp.status}`)
        msg = `Mirror URL registered. Run a scrape on ${new URL(mirror).hostname} to fetch the content.`
      } else {
        // sitemap_only / manual_upload share the same /sources/upsert codepath.
        const resp = await fetch(`${API_BASE}/sources/upsert`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            url: url.trim(),
            discovered_via: 'manual',
            payer_hint: payer || null,
            state_hint: state || null,
            authority_hint: authority || null,
            program_hint: program || null,
            auto_publish: autoPublish,
          }),
        })
        if (!resp.ok) throw new Error(`upsert ${resp.status}`)
        msg = strategy === 'sitemap_only'
          ? 'Source registered (sitemap_only). Run scripts/curator/backfill_from_sources_yaml.py to populate URLs from sitemap.xml.'
          : 'Source registered for manual upload. Operator should add PDFs through the Computer tab.'
      }
      setSubmitMsg({ ok: true, msg })
      onDocumentAdded()
    } catch (e) {
      setSubmitMsg({ ok: false, msg: String(e) })
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="upload-panel">
      <div className="upload-url-form">
        <input
          type="url"
          placeholder="https://example.com/providers/manual"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleProbe()
          }}
          className="upload-url-input"
          disabled={submitting}
        />
        <button
          type="button"
          className="btn btn-secondary"
          onClick={handleProbe}
          disabled={probing || !url.trim() || submitting}
        >
          {probing ? 'Probing…' : 'Probe'}
        </button>
      </div>
      {probeError && <div className="error-message">{probeError}</div>}

      {probe && (
        <>
          <div className="upload-probe-result">
            <div>
              <strong>HTTP {probe.fetch.status}</strong>
              {' · '}
              Sitemap: {probe.sitemap.status} ({probe.sitemap.url_count} URLs)
              {probe.robots && ` · robots: ${probe.robots.status}`}
            </div>
            {probe.classifier && (
              <div className="upload-probe-classifier">
                Detected:
                {probe.classifier.payer && ` payer=${probe.classifier.payer}`}
                {probe.classifier.state && ` · state=${probe.classifier.state}`}
                {probe.classifier.inferred_authority_level && ` · authority=${probe.classifier.inferred_authority_level}`}
              </div>
            )}
            <div className="upload-probe-rec">
              Recommended: <strong>{probe.recommended_strategy.replace(/_/g, ' ')}</strong>
              <small> — {probe.recommended_reason}</small>
            </div>
          </div>

          <fieldset className="upload-fieldset">
            <legend className="upload-fieldset-legend">Strategy</legend>
            <div className="upload-strategy-radios">
              {([
                ['scrape', 'Scrape (tree BFS) — best for open sites'],
                ['sitemap_only', 'Sitemap only — register URLs without crawling'],
                ['state_mirror', 'State mirror — register a different URL (e.g. AHCA) when this site is bot-walled'],
                ['manual_upload', 'Manual upload — operator will add PDFs by hand'],
              ] as [Strategy, string][]).map(([val, label]) => (
                <label key={val} className={`upload-strategy-row ${strategy === val ? 'selected' : ''}`}>
                  <input
                    type="radio"
                    name="upload-strategy"
                    value={val}
                    checked={strategy === val}
                    onChange={() => setStrategy(val)}
                  />
                  <span>{label}</span>
                </label>
              ))}
            </div>
            {strategy === 'state_mirror' && (
              <div className="upload-mirror-url">
                <label htmlFor="upload-mirror-url-input">
                  Mirror URL (where the same content actually lives)
                </label>
                <input
                  id="upload-mirror-url-input"
                  type="url"
                  value={mirrorUrl}
                  onChange={(e) => setMirrorUrl(e.target.value)}
                  placeholder="https://ahca.myflorida.com/medicaid/.../aetna-handbook.pdf"
                />
                {probe?.mirror_suggestion && (
                  <div className="upload-mirror-help">
                    <p>{probe.mirror_suggestion.help}</p>
                    <div className="upload-mirror-help-links">
                      <a
                        href={probe.mirror_suggestion.suggested_url}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        ↗ Open AHCA SMMC index
                      </a>
                      <a
                        href={probe.mirror_suggestion.search_url}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        🔎 Search AHCA for this payer
                      </a>
                    </div>
                  </div>
                )}
                <small className="upload-help-text">
                  We'll register the mirror URL in the corpus instead of the bot-walled
                  original. The original URL is kept as a reference (status 403).
                </small>
              </div>
            )}
          </fieldset>

          <details className="upload-advanced">
            <summary>Advanced</summary>
            <div className="upload-advanced-grid">
              <label>
                Max depth
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={maxDepth}
                  onChange={(e) => setMaxDepth(parseInt(e.target.value, 10) || 3)}
                  disabled={strategy !== 'scrape'}
                />
              </label>
              <label>
                Max pages
                <input
                  type="number"
                  min={1}
                  max={2000}
                  value={maxPages}
                  onChange={(e) => setMaxPages(parseInt(e.target.value, 10) || 200)}
                  disabled={strategy !== 'scrape'}
                />
              </label>
            </div>
            <div className="upload-advanced-content-mode">
              <span className="upload-advanced-content-mode-label">Content</span>
              <label>
                <input
                  type="checkbox"
                  checked={includeHtml}
                  onChange={(e) => setIncludeHtml(e.target.checked)}
                  disabled={strategy !== 'scrape'}
                />
                HTML pages
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={includePdfs}
                  onChange={(e) => setIncludePdfs(e.target.checked)}
                  disabled={strategy !== 'scrape'}
                />
                PDFs
              </label>
            </div>
          </details>

          <fieldset className="upload-fieldset">
            <legend className="upload-fieldset-legend">Metadata</legend>
            <div className="upload-metadata-grid">
              <label>
                Payer
                <input value={payer} onChange={(e) => setPayer(e.target.value)} placeholder="auto" />
              </label>
              <label>
                State
                <input
                  value={state}
                  onChange={(e) => setState(e.target.value.toUpperCase())}
                  placeholder="FL"
                  maxLength={2}
                />
              </label>
              <label>
                Authority
                <select value={authority} onChange={(e) => setAuthority(e.target.value)}>
                  <option value="">— auto / unknown —</option>
                  {AUTHORITY_LEVEL_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                </select>
              </label>
              <label>
                Program
                <input
                  value={program}
                  onChange={(e) => setProgram(e.target.value)}
                  placeholder="e.g. Medicaid"
                />
              </label>
            </div>
          </fieldset>

          <label className="upload-autopublish">
            <input
              type="checkbox"
              checked={autoPublish}
              onChange={(e) => setAutoPublish(e.target.checked)}
            />
            Auto-publish to chat
          </label>

          <div className="upload-actions">
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleSubmit}
              disabled={!probe || submitting}
            >
              {submitting ? 'Submitting…' : 'Submit'}
            </button>
          </div>
        </>
      )}

      {submitMsg && (
        <div className={`upload-submit-msg ${submitMsg.ok ? 'ok' : 'err'}`}>
          {submitMsg.msg}
        </div>
      )}
    </div>
  )
}

/* ──────────────────────── Drive panel ─────────────────────── */
function DrivePanel({ onDocumentAdded }: { onDocumentAdded: () => void }) {
  const [driveConnected, setDriveConnected] = useState(false)
  const [driveEmail, setDriveEmail] = useState<string | null>(null)
  const [driveEnabled, setDriveEnabled] = useState(true)
  const [driveCurrentFolderId, setDriveCurrentFolderId] = useState<string>('root')
  const [driveBreadcrumb, setDriveBreadcrumb] = useState<{ id: string; name: string }[]>([
    { id: 'root', name: 'My Drive' },
  ])
  const [driveFolders, setDriveFolders] = useState<DriveFolder[]>([])
  const [driveFiles, setDriveFiles] = useState<DriveFile[]>([])
  const [driveFilesLoading, setDriveFilesLoading] = useState(false)
  const [driveError, setDriveError] = useState<string | null>(null)
  const [selectedDriveFileIds, setSelectedDriveFileIds] = useState<Set<string>>(new Set())
  const [importingFromDrive, setImportingFromDrive] = useState(false)
  const [driveConnecting, setDriveConnecting] = useState(false)
  const [driveFolderLink, setDriveFolderLink] = useState('')

  // Optional metadata
  const [payer, setPayer] = useState('')
  const [stateMeta, setStateMeta] = useState('')
  const [authority, setAuthority] = useState<string>('')
  const [program, setProgram] = useState('')

  const driveSessionIdRef = useRef<string | null>(null)
  const driveFetchOpts = { credentials: 'include' as RequestCredentials }

  const fetchDriveStatus = useCallback(async () => {
    try {
      const headers: Record<string, string> = {}
      if (driveSessionIdRef.current) headers['X-RAG-Session'] = driveSessionIdRef.current
      const r = await fetch(`${API_BASE}/drive/status`, { ...driveFetchOpts, headers })
      const data = await r.json()
      setDriveConnected(!!data.connected)
      setDriveEmail(data.email || null)
      setDriveEnabled(data.enabled !== false)
    } catch {
      setDriveEnabled(false)
    }
  }, [])

  useEffect(() => {
    fetchDriveStatus()
  }, [fetchDriveStatus])

  useEffect(() => {
    const hash = window.location.hash
    if (hash.includes('drive') && (hash.includes('connected=1') || hash.includes('error='))) {
      fetchDriveStatus()
      const base = hash.split('?')[0] || '#/'
      window.history.replaceState(null, '', window.location.pathname + window.location.search + base)
    }
  }, [fetchDriveStatus])

  const loadDriveFolder = useCallback(async (folderId: string, breadcrumb: { id: string; name: string }[]) => {
    setDriveError(null)
    setDriveFilesLoading(true)
    try {
      const headers: Record<string, string> = {}
      if (driveSessionIdRef.current) headers['X-RAG-Session'] = driveSessionIdRef.current
      const r = await fetch(`${API_BASE}/drive/folders/${encodeURIComponent(folderId)}/files`, {
        ...driveFetchOpts,
        headers,
      })
      if (!r.ok) {
        if (r.status === 401) throw new Error('Connect Google Drive first')
        const err = await r.json().catch(() => ({}))
        throw new Error(err.detail || 'Failed to list folder')
      }
      const data = await r.json()
      const folderName = data.folder_name || (folderId === 'root' ? 'My Drive' : '')
      const crumb = [...breadcrumb]
      if (crumb.length > 0 && folderName) crumb[crumb.length - 1] = { ...crumb[crumb.length - 1], name: folderName }
      setDriveCurrentFolderId(folderId)
      setDriveBreadcrumb(crumb)
      setDriveFolders(data.folders || [])
      setDriveFiles(data.files || [])
      setSelectedDriveFileIds(new Set())
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'List failed'
      setDriveError(msg === 'Failed to fetch' ? 'RAG backend unreachable.' : msg)
      setDriveFolders([])
      setDriveFiles([])
    } finally {
      setDriveFilesLoading(false)
    }
  }, [])

  useEffect(() => {
    if (driveConnected) {
      loadDriveFolder('root', [{ id: 'root', name: 'My Drive' }])
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [driveConnected])

  const handleDriveConnect = async () => {
    setDriveError(null)
    setDriveConnecting(true)
    try {
      const r = await fetch(`${API_BASE}/drive/auth-url`, driveFetchOpts)
      if (!r.ok) {
        const err = await r.json().catch(() => ({}))
        const msg =
          typeof err.detail === 'string' ? err.detail : err.detail?.message || 'Failed to get auth URL'
        throw new Error(msg)
      }
      const { url, session_id } = await r.json()
      if (session_id) driveSessionIdRef.current = session_id
      if (url) window.location.href = url
      else throw new Error('No auth URL returned')
    } catch (e) {
      setDriveError(e instanceof Error ? e.message : 'Connect failed')
    } finally {
      setDriveConnecting(false)
    }
  }

  const handleDriveDisconnect = async () => {
    try {
      const headers: Record<string, string> = {}
      if (driveSessionIdRef.current) headers['X-RAG-Session'] = driveSessionIdRef.current
      await fetch(`${API_BASE}/drive/disconnect`, { method: 'DELETE', ...driveFetchOpts, headers })
      driveSessionIdRef.current = null
      setDriveConnected(false)
      setDriveEmail(null)
      setDriveFolders([])
      setDriveFiles([])
      setDriveCurrentFolderId('root')
      setDriveBreadcrumb([{ id: 'root', name: 'My Drive' }])
      setSelectedDriveFileIds(new Set())
      setDriveError(null)
    } catch (e) {
      setDriveError(e instanceof Error ? e.message : 'Disconnect failed')
    }
  }

  const handleDriveNavigateToFolder = (folder: DriveFolder) => {
    const nextCrumb = [...driveBreadcrumb, { id: folder.id, name: folder.name }]
    loadDriveFolder(folder.id, nextCrumb)
  }

  const handleDriveNavigateUp = () => {
    if (driveBreadcrumb.length <= 1) return
    const next = driveBreadcrumb.slice(0, -1)
    const parent = next[next.length - 1]
    loadDriveFolder(parent.id, next)
  }

  const handleDriveGoToFolder = () => {
    const trimmed = driveFolderLink.trim()
    if (!trimmed) return
    const match = trimmed.match(/\/folders\/([a-zA-Z0-9_-]+)/)
    const fid = match ? match[1] : trimmed
    loadDriveFolder(fid, [{ id: 'root', name: 'My Drive' }, { id: fid, name: fid }])
  }

  const toggleDriveFile = (id: string) => {
    setSelectedDriveFileIds((prev) => {
      const next = new Set(prev.size === 0 ? driveFiles.map((f) => f.id) : prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }
  const selectAllDriveFiles = () => setSelectedDriveFileIds(new Set(driveFiles.map((f) => f.id)))
  const clearDriveSelection = () => setSelectedDriveFileIds(new Set())

  const handleDriveImport = async () => {
    const ids = selectedDriveFileIds.size > 0 ? [...selectedDriveFileIds] : driveFiles.map((f) => f.id)
    if (ids.length === 0) return
    setDriveError(null)
    setImportingFromDrive(true)
    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' }
      if (driveSessionIdRef.current) headers['X-RAG-Session'] = driveSessionIdRef.current
      const r = await fetch(`${API_BASE}/documents/import-from-drive`, {
        method: 'POST',
        ...driveFetchOpts,
        headers,
        body: JSON.stringify({
          folder_id: driveCurrentFolderId,
          file_ids: ids,
          payer: payer.trim() || undefined,
          state: stateMeta.trim() || undefined,
          program: program.trim() || undefined,
          authority_level: authority || undefined,
        }),
      })
      if (!r.ok) {
        if (r.status === 401) throw new Error('Connect Google Drive first')
        const err = await r.json().catch(() => ({}))
        throw new Error(err.detail || 'Import failed')
      }
      const data = await r.json()
      const ok = (data.results || []).filter((x: { status: string }) => x.status === 'completed').length
      onDocumentAdded()
      setSelectedDriveFileIds(new Set())
      if (ok > 0) setDriveFiles((prev) => prev.filter((f) => !ids.includes(f.id)))
    } catch (e) {
      setDriveError(e instanceof Error ? e.message : 'Import failed')
    } finally {
      setImportingFromDrive(false)
    }
  }

  if (!driveEnabled) {
    return (
      <div className="upload-panel">
        <p className="upload-hint">Set DRIVE_API_ENABLED=true and OAuth credentials in .env to enable.</p>
      </div>
    )
  }

  if (!driveConnected) {
    return (
      <div className="upload-panel">
        <button
          type="button"
          className="btn btn-primary"
          onClick={handleDriveConnect}
          disabled={driveConnecting}
        >
          {driveConnecting ? 'Redirecting…' : 'Connect Google Drive'}
        </button>
        {driveError && (
          <div className="error-message" role="alert">
            {driveError}
          </div>
        )}
        <p className="upload-hint">Sign in with Google to access your Drive. PDFs and Google Docs supported.</p>
      </div>
    )
  }

  return (
    <div className="upload-panel">
      <div className="upload-drive-toolbar">
        <span className="upload-drive-email">Connected: {driveEmail || 'Google account'}</span>
        <button type="button" className="btn btn-secondary btn-sm" onClick={handleDriveDisconnect}>
          Disconnect
        </button>
        {driveBreadcrumb.length > 1 && (
          <button type="button" className="btn btn-secondary btn-sm" onClick={handleDriveNavigateUp}>
            ↑ Up
          </button>
        )}
      </div>

      <div className="upload-drive-breadcrumb">
        {driveBreadcrumb.map((b, i) => (
          <span key={b.id}>
            {i > 0 && ' › '}
            <button
              type="button"
              className="upload-drive-breadcrumb-link"
              onClick={() => loadDriveFolder(b.id, driveBreadcrumb.slice(0, i + 1))}
            >
              {b.name}
            </button>
          </span>
        ))}
      </div>

      <div className="upload-drive-goto">
        <input
          type="text"
          placeholder="Or paste folder link to jump"
          value={driveFolderLink}
          onChange={(e) => setDriveFolderLink(e.target.value)}
          className="upload-url-input"
          disabled={driveFilesLoading}
        />
        <button
          type="button"
          className="btn btn-secondary btn-sm"
          onClick={handleDriveGoToFolder}
          disabled={!driveFolderLink.trim() || driveFilesLoading}
        >
          Go
        </button>
      </div>

      {driveError && (
        <div className="error-message" role="alert">
          {driveError}
        </div>
      )}

      {(driveFolders.length > 0 || driveFiles.length > 0) && (
        <div className="upload-drive-list">
          {driveFolders.length > 0 && (
            <ul className="upload-drive-folders">
              {driveFolders.map((f) => (
                <li key={f.id}>
                  <button
                    type="button"
                    className="upload-drive-folder-item"
                    onClick={() => handleDriveNavigateToFolder(f)}
                  >
                    <span aria-hidden>📁</span>
                    <span>{f.name}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
          {driveFiles.length > 0 && (
            <>
              <div className="upload-drive-actions">
                <button type="button" className="btn btn-secondary btn-sm" onClick={selectAllDriveFiles}>
                  Select all
                </button>
                <button type="button" className="btn btn-secondary btn-sm" onClick={clearDriveSelection}>
                  Clear
                </button>
                <span className="upload-drive-count">
                  {selectedDriveFileIds.size || driveFiles.length} selected
                </span>
              </div>
              <ul className="upload-drive-files">
                {driveFiles.map((f) => (
                  <li key={f.id}>
                    <label>
                      <input
                        type="checkbox"
                        checked={selectedDriveFileIds.size === 0 ? true : selectedDriveFileIds.has(f.id)}
                        onChange={() => toggleDriveFile(f.id)}
                      />
                      <span>{f.name}</span>
                      {f.mimeType?.includes('document') && <span className="upload-drive-file-tag">(Doc)</span>}
                    </label>
                  </li>
                ))}
              </ul>
            </>
          )}
          {driveFolders.length === 0 && driveFiles.length === 0 && !driveFilesLoading && (
            <p className="upload-hint">This folder is empty or has no PDFs/Google Docs.</p>
          )}
        </div>
      )}

      <fieldset className="upload-fieldset">
        <legend className="upload-fieldset-legend">Metadata (applied to all imports)</legend>
        <div className="upload-metadata-grid">
          <label>
            Payer
            <input value={payer} onChange={(e) => setPayer(e.target.value)} placeholder="e.g. UnitedHealthcare" />
          </label>
          <label>
            State
            <input
              value={stateMeta}
              onChange={(e) => setStateMeta(e.target.value.toUpperCase())}
              placeholder="FL"
              maxLength={2}
            />
          </label>
          <label>
            Authority
            <select value={authority} onChange={(e) => setAuthority(e.target.value)}>
              <option value="">— auto / unknown —</option>
              {AUTHORITY_LEVEL_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </label>
          <label>
            Program
            <input value={program} onChange={(e) => setProgram(e.target.value)} placeholder="e.g. Medicaid" />
          </label>
        </div>
      </fieldset>

      {driveFiles.length > 0 && (
        <div className="upload-actions">
          <button
            type="button"
            className="btn btn-primary"
            onClick={handleDriveImport}
            disabled={importingFromDrive}
          >
            {importingFromDrive ? 'Importing…' : 'Import selected'}
          </button>
        </div>
      )}
    </div>
  )
}
