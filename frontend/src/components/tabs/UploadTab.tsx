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
  // Used to group docs under their parent scrape host in the queue.
  gcs_path?: string | null
  source_metadata?: {
    source_url?: string | null
    scrape_job_id?: string | null
    agent_scope?: string | null
  } | null
  expires_at?: string | null
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

interface UploadMeta {
  payer?: string
  state?: string
  program?: string
}

interface Props {
  documents: DocLike[]
  onUpload: (file: File, meta?: UploadMeta) => Promise<void>
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
  source?: 'chat' | 'drive' | 'url' | 'computer' | 'scrape'
  /** Optional child rows (per-doc detail when this row is a host group). */
  children?: QueueRow[]
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
      source: 'scrape',
      meta:
        j.status === 'failed' ? (j.error || 'failed') :
        isActive
          ? `${j.pages_scraped || 0}${j.max_pages ? ` / ${j.max_pages}` : ''} pages`
          : `${j.pages_scraped || 0} pages · ${j.documents_count || 0} docs`,
    })
  }
  return rows
}

/**
 * Infer the host a document came from. Order:
 *  1. agent_scope === 'chat' → chat upload (ephemeral, high-priority)
 *  2. source_metadata.source_url (set by import-from-gcs / -html)
 *  3. gcs_path (web URL for HTML imports, gs:// for scrape PDFs)
 *  4. '(uploaded)' bucket for hand-uploaded files
 */
function docHost(d: DocLike): string {
  if (d.source_metadata?.agent_scope === 'chat' || d.expires_at) return '(chat)'
  const sm = d.source_metadata?.source_url
  if (sm) {
    try {
      const h = new URL(sm).hostname.replace(/^www\./, '')
      if (h.includes('drive.google')) return '(drive)'
      return h
    } catch { /* fall through */ }
  }
  const gp = d.gcs_path || ''
  if (gp.startsWith('http')) {
    try { return new URL(gp).hostname.replace(/^www\./, '') } catch { /* fall through */ }
  }
  if (gp.startsWith('gs://')) return '(uploaded)'
  return '(uploaded)'
}

function hostToSource(host: string): QueueRow['source'] {
  if (host === '(chat)') return 'chat'
  if (host === '(drive)') return 'drive'
  if (host === '(uploaded)') return 'computer'
  return 'url'
}

/** Per-doc stage classification, separated so the group view can re-use it. */
function docStage(d: DocLike): { stage: QueueRow['stage']; active: boolean; pct: number; recent: boolean } {
  const now = Date.now()
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
  return { stage, active, pct, recent }
}

/** Stage priority for "what's the most-active stage in this group". */
const STAGE_RANK: Record<QueueRow['stage'], number> = {
  extracting: 5,
  chunking: 4,
  embedding: 3,
  scraping: 2, // unused at doc level but here for type completeness
  pending: 1,
  failed: 6,
  completed: 0,
}

/**
 * Derive the in-flight queue from the documents list.
 *
 * Rather than one row per doc (a humana scrape produces 100s of rows
 * which floods the panel), collapse active/recent docs by host. Each
 * group surfaces the dominant active stage + counts of docs in each
 * stage. Hand-uploaded files (no source URL) bucket into '(uploaded)'.
 */
function deriveQueue(documents: DocLike[]): QueueRow[] {
  // Bucket by host
  const groups = new Map<string, { host: string; docs: DocLike[]; stages: Map<QueueRow['stage'], number>; active: number; recent: number }>()

  for (const d of documents) {
    const info = docStage(d)
    if (!info.active && !info.recent) continue
    const host = docHost(d)
    let g = groups.get(host)
    if (!g) {
      g = { host, docs: [], stages: new Map(), active: 0, recent: 0 }
      groups.set(host, g)
    }
    g.docs.push(d)
    g.stages.set(info.stage, (g.stages.get(info.stage) || 0) + 1)
    if (info.active) g.active++
    if (info.recent) g.recent++
  }

  const rows: QueueRow[] = []
  for (const g of groups.values()) {
    // Pick the dominant stage = the highest-rank stage present
    let topStage: QueueRow['stage'] = 'completed'
    let topRank = -1
    for (const [stage, count] of g.stages.entries()) {
      if (!count) continue
      const rank = STAGE_RANK[stage] ?? 0
      if (rank > topRank) {
        topRank = rank
        topStage = stage
      }
    }
    // Aggregate progress = mean of contributing docs' pct
    let pctSum = 0
    let pctN = 0
    for (const d of g.docs) {
      const { active, pct } = docStage(d)
      if (active && pct > 0) { pctSum += pct; pctN++ }
    }
    const aggPct = pctN > 0 ? Math.round(pctSum / pctN) : (g.active === 0 ? 100 : 0)

    // Compose meta string showing per-stage counts
    const parts: string[] = []
    for (const stage of ['extracting', 'chunking', 'embedding', 'completed', 'failed'] as const) {
      const n = g.stages.get(stage) || 0
      if (n > 0) parts.push(`${n} ${stage}`)
    }

    // Per-doc child rows so the operator can drill in.
    const children: QueueRow[] = g.docs.map((d) => {
      const info = docStage(d)
      return {
        id: d.id,
        filename: d.display_name?.trim() || d.filename,
        status: info.stage,
        stage: info.stage,
        progressPct: info.pct || (info.stage === 'completed' ? 100 : 0),
        recentlyCompleted: info.recent && !info.active,
      }
    })
    // Sort children: active first, then completed
    children.sort((a, b) => {
      const aDone = a.stage === 'completed' || a.stage === 'failed'
      const bDone = b.stage === 'completed' || b.stage === 'failed'
      if (aDone !== bDone) return aDone ? 1 : -1
      return a.filename.localeCompare(b.filename)
    })

    rows.push({
      id: `host:${g.host}`,
      filename: g.host,
      status: topStage,
      stage: topStage,
      progressPct: aggPct,
      recentlyCompleted: g.active === 0 && g.recent > 0,
      source: hostToSource(g.host),
      meta: `${g.docs.length} ${g.docs.length === 1 ? 'doc' : 'docs'} · ${parts.join(' · ')}`,
      children,
    })
  }
  // Active groups first
  rows.sort((a, b) => {
    const aActive = a.stage !== 'completed' && a.stage !== 'failed'
    const bActive = b.stage !== 'completed' && b.stage !== 'failed'
    if (aActive !== bActive) return aActive ? -1 : 1
    return 0
  })
  return rows
}

/**
 * Upload tab — clean single-card layout with a collapsible in-flight footer.
 */
export function UploadTab({ documents, onUpload, uploading, error, onDocumentAdded }: Props) {
  const [sourceType, setSourceType] = useState<SourceType>('computer')
  const [footerOpen, setFooterOpen] = useState(false)

  // Poll scraper-side active jobs (5s cadence).
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
      } catch { /* scraper unreachable — silent */ }
    }
    fetchActive()
    const interval = setInterval(fetchActive, 5000)
    return () => { cancelled = true; clearInterval(interval) }
  }, [])

  const queue = useMemo(
    () => [...scrapeJobsToRows(scrapeJobs), ...deriveQueue(documents)],
    [scrapeJobs, documents],
  )

  // Auto-open footer when active items appear for the first time.
  const prevActive = useRef(0)
  useEffect(() => {
    const active = queue.filter(r => r.stage !== 'completed' && r.stage !== 'failed').length
    if (active > 0 && prevActive.current === 0) setFooterOpen(true)
    prevActive.current = active
  }, [queue])

  return (
    <div className="upload-tab">
      <div className="upload-main">
        <div className="upload-card">
          <div className="upload-source-selector" role="tablist" aria-label="Upload source">
            {([
              ['computer', '📁', 'Computer'],
              ['url', '🌐', 'URL'],
              ['drive', '☁', 'Drive'],
            ] as [SourceType, string, string][]).map(([type, icon, label]) => (
              <button
                key={type}
                type="button"
                role="tab"
                aria-selected={sourceType === type}
                className={`upload-source-btn ${sourceType === type ? 'active' : ''}`}
                onClick={() => setSourceType(type)}
              >
                <span className="upload-source-icon" aria-hidden>{icon}</span>
                {label}
              </button>
            ))}
          </div>

          <div className="upload-source-panel">
            {sourceType === 'computer' && (
              <ComputerPanel onUpload={onUpload} uploading={uploading} error={error} />
            )}
            {sourceType === 'url' && <UrlPanel onDocumentAdded={onDocumentAdded} />}
            {sourceType === 'drive' && <DrivePanel onDocumentAdded={onDocumentAdded} />}
          </div>
        </div>
      </div>

      <InflightFooter queue={queue} open={footerOpen} onToggle={() => setFooterOpen(v => !v)} />
    </div>
  )
}

/* ─── Collapsible in-flight footer ─────────────────────────────────────── */

const SOURCE_ICON: Record<string, string> = {
  chat: '💬',
  drive: '☁',
  computer: '📁',
  url: '🌐',
  scrape: '🔍',
}

function InflightFooter({
  queue,
  open,
  onToggle,
}: {
  queue: QueueRow[]
  open: boolean
  onToggle: () => void
}) {
  const active = queue.filter(r => r.stage !== 'completed' && r.stage !== 'failed')
  const chatActive = active.filter(r => r.source === 'chat').length
  const total = queue.length

  // Summary badge text
  let summary = ''
  if (total === 0) {
    summary = 'No active uploads'
  } else if (active.length === 0) {
    summary = `${total} recently completed`
  } else {
    summary = `${active.length} in‑flight`
    if (chatActive > 0) summary += ` · ${chatActive} from chat`
  }

  return (
    <div className={`upload-footer ${open ? 'upload-footer--open' : ''}`}>
      {open && (
        <div className="upload-footer-body">
          {total === 0 ? (
            <p className="upload-footer-empty">No active or recent uploads.</p>
          ) : (
            <ul className="upload-queue-list">
              {queue.map((row) => (
                <QueueRowView key={row.id} row={row} />
              ))}
            </ul>
          )}
        </div>
      )}

      <button
        type="button"
        className="upload-footer-bar"
        onClick={onToggle}
        aria-expanded={open}
      >
        <span className="upload-footer-indicator">
          {active.length > 0 && <span className="upload-footer-pulse" />}
          {active.length > 0 ? '⟳' : '✓'}
        </span>
        <span className="upload-footer-summary">{summary}</span>
        <span className="upload-footer-chevron">{open ? '▴' : '▾'}</span>
      </button>
    </div>
  )
}

/* ───────────────────── Queue row (collapsible) ─────────────────────── */
/* ───────────────────── Computer panel ─────────────────────── */
function QueueRowView({ row }: { row: QueueRow }) {
  const [expanded, setExpanded] = useState(false)
  const hasChildren = !!row.children && row.children.length > 0

  return (
    <li className="upload-queue-row">
      <div
        className={`upload-queue-row-head ${hasChildren ? 'upload-queue-row-head--toggle' : ''}`}
        onClick={() => hasChildren && setExpanded((v) => !v)}
        role={hasChildren ? 'button' : undefined}
        tabIndex={hasChildren ? 0 : undefined}
        onKeyDown={(e) => {
          if (hasChildren && (e.key === 'Enter' || e.key === ' ')) {
            e.preventDefault()
            setExpanded((v) => !v)
          }
        }}
      >
        {hasChildren && (
          <span className="upload-queue-toggle" aria-hidden>
            {expanded ? '▾' : '▸'}
          </span>
        )}
        {row.source && (
          <span className="upload-queue-source" title={row.source}>
            {SOURCE_ICON[row.source] || ''}
          </span>
        )}
        <span className="upload-queue-name" title={row.filename}>
          {row.filename}
        </span>
        {row.source === 'chat' && (
          <span className="upload-queue-pill upload-queue-pill--chat">chat</span>
        )}
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

      {expanded && hasChildren && (
        <ul className="upload-queue-children">
          {row.children!.map((c) => (
            <li key={c.id} className="upload-queue-child">
              <span className="upload-queue-child-name" title={c.filename}>
                {c.filename}
              </span>
              <span className={`upload-queue-pill stage-${c.stage}`}>{c.stage}</span>
              {c.progressPct > 0 && c.progressPct < 100 && (
                <span className="upload-queue-child-pct">{c.progressPct}%</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </li>
  )
}


function ComputerPanel({
  onUpload,
  uploading,
  error,
}: {
  onUpload: (file: File, meta?: UploadMeta) => Promise<void>
  uploading: boolean
  error: string | null
}) {
  const [files, setFiles] = useState<File[]>([])
  const [dragActive, setDragActive] = useState(false)
  const [payer, setPayer] = useState('')
  const [state, setState] = useState('')
  const [program, setProgram] = useState('')
  const [uploadingIdx, setUploadingIdx] = useState<number | null>(null)
  const [doneCount, setDoneCount] = useState(0)

  const addFiles = (incoming: FileList | null) => {
    if (!incoming) return
    const pdfs = Array.from(incoming).filter((f) => f.type === 'application/pdf' || f.name.endsWith('.pdf'))
    setFiles((prev) => {
      const existing = new Set(prev.map((f) => f.name + f.size))
      return [...prev, ...pdfs.filter((f) => !existing.has(f.name + f.size))]
    })
  }

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
    addFiles(e.dataTransfer.files)
  }

  const removeFile = (idx: number) => setFiles((prev) => prev.filter((_, i) => i !== idx))

  const handleUpload = async () => {
    if (files.length === 0) return
    const meta: UploadMeta = {
      payer: payer.trim() || undefined,
      state: state.trim() || undefined,
      program: program.trim() || undefined,
    }
    setDoneCount(0)
    for (let i = 0; i < files.length; i++) {
      setUploadingIdx(i)
      await onUpload(files[i], meta)
      setDoneCount(i + 1)
    }
    setUploadingIdx(null)
    setFiles([])
    const fileInput = document.getElementById('upload-tab-file-input') as HTMLInputElement | null
    if (fileInput) fileInput.value = ''
  }

  const isUploading = uploadingIdx !== null || uploading

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
          multiple
          onChange={(e) => addFiles(e.target.files)}
          accept=".pdf"
          disabled={isUploading}
          className="upload-dropzone-input"
        />
        <label htmlFor="upload-tab-file-input" className="upload-dropzone-label">
          {files.length === 0
            ? 'Choose files or drag them here'
            : `${files.length} file${files.length > 1 ? 's' : ''} selected — drop more or click to add`}
        </label>
        <button
          type="button"
          onClick={handleUpload}
          disabled={files.length === 0 || isUploading}
          className="btn btn-primary upload-dropzone-btn"
        >
          {isUploading
            ? `Uploading ${uploadingIdx !== null ? uploadingIdx + 1 : doneCount} / ${files.length}…`
            : files.length > 1
              ? `Upload ${files.length} files`
              : 'Upload'}
        </button>
      </div>

      {files.length > 0 && (
        <ul style={{ margin: '8px 0 0', padding: 0, listStyle: 'none', display: 'grid', gap: 4 }}>
          {files.map((f, i) => (
            <li
              key={f.name + f.size}
              style={{
                display: 'flex', alignItems: 'center', gap: 8, fontSize: 12,
                padding: '4px 8px', background: uploadingIdx === i ? '#eff6ff' : '#f9fafb',
                borderRadius: 4, border: `1px solid ${uploadingIdx === i ? '#93c5fd' : '#e5e7eb'}`,
              }}
            >
              <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={f.name}>
                {uploadingIdx === i ? '⏳ ' : doneCount > i ? '✓ ' : ''}{f.name}
              </span>
              <span style={{ color: '#6b7280', whiteSpace: 'nowrap' }}>
                {(f.size / 1024 / 1024).toFixed(1)} MB
              </span>
              {!isUploading && (
                <button
                  type="button"
                  onClick={() => removeFile(i)}
                  style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#9ca3af', padding: '0 2px', fontSize: 14 }}
                  title="Remove"
                >
                  ×
                </button>
              )}
            </li>
          ))}
        </ul>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginTop: 10 }}>
        <label style={{ fontSize: 12 }}>
          <div style={{ color: '#6b7280', marginBottom: 2 }}>Payer (optional)</div>
          <input
            value={payer}
            onChange={(e) => setPayer(e.target.value)}
            placeholder="e.g. Aetna Better Health"
            disabled={isUploading}
            style={{ width: '100%', fontSize: 12, padding: '4px 6px', border: '1px solid #d1d5db', borderRadius: 4, boxSizing: 'border-box' }}
          />
        </label>
        <label style={{ fontSize: 12 }}>
          <div style={{ color: '#6b7280', marginBottom: 2 }}>State (optional)</div>
          <input
            value={state}
            onChange={(e) => setState(e.target.value.toUpperCase())}
            placeholder="FL"
            maxLength={2}
            disabled={isUploading}
            style={{ width: '100%', fontSize: 12, padding: '4px 6px', border: '1px solid #d1d5db', borderRadius: 4, boxSizing: 'border-box' }}
          />
        </label>
        <label style={{ fontSize: 12 }}>
          <div style={{ color: '#6b7280', marginBottom: 2 }}>Program (optional)</div>
          <input
            value={program}
            onChange={(e) => setProgram(e.target.value)}
            placeholder="e.g. Medicaid"
            disabled={isUploading}
            style={{ width: '100%', fontSize: 12, padding: '4px 6px', border: '1px solid #d1d5db', borderRadius: 4, boxSizing: 'border-box' }}
          />
        </label>
      </div>

      {error && (
        <div className="error-message" role="alert" style={{ marginTop: 8 }}>
          {error}
        </div>
      )}
      <p className="upload-hint">
        PDF only. Metadata overrides filename inference — leave blank to let the backend infer from filename.
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

          <details className="upload-advanced" open>
            <summary>
              Advanced — depth {maxDepth}, max {maxPages} pages
              {strategy === 'scrape' ? '' : ' (not used for this strategy)'}
            </summary>
            <div className="upload-advanced-grid">
              <label>
                Max depth
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={maxDepth}
                  onChange={(e) => setMaxDepth(parseInt(e.target.value, 10) || 3)}
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
                />
                HTML pages
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={includePdfs}
                  onChange={(e) => setIncludePdfs(e.target.checked)}
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
