import { useState, useEffect, useCallback, useRef } from 'react'
import { API_BASE, SCRAPER_API_BASE } from '../../config'
import { STATE_OPTIONS, AUTHORITY_LEVEL_OPTIONS } from '../../lib/documentMetadata'
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

export function DocumentInputTab({ onUpload, uploading, error, onDocumentAdded }: DocumentInputTabProps) {
  const [file, setFile] = useState<File | null>(null)
  const [dragActive, setDragActive] = useState(false)

  // Scrape from URL state
  const [scrapeUrl, setScrapeUrl] = useState('')
  const [scrapeMode, setScrapeMode] = useState<'regular' | 'tree'>('regular')
  const [scrapeMaxDepth, setScrapeMaxDepth] = useState(3)
  const [scrapeMaxPages, setScrapeMaxPages] = useState(50)
  const [scrapePathPrefix, setScrapePathPrefix] = useState('')
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
  const [showAllPages, setShowAllPages] = useState(false)
  const [selectedPageUrls, setSelectedPageUrls] = useState<Set<string>>(new Set())
  const [selectedDocPaths, setSelectedDocPaths] = useState<Set<string>>(new Set())
  // Optional metadata when adding scraped pages/docs to RAG (carries forward to document)
  const [importMetadata, setImportMetadata] = useState<{ display_name: string; payer: string; state: string; program: string; authority_level: string }>({
    display_name: '',
    payer: '',
    state: '',
    program: '',
    authority_level: '',
  })
  const [importingPages, setImportingPages] = useState(false)
  const [readerPage, setReaderPage] = useState<ScrapedPage | null>(null)
  const [autoAddToRagWhenComplete, setAutoAddToRagWhenComplete] = useState(false)
  const eventSourceRef = useRef<EventSource | null>(null)
  const autoAddRunForJobRef = useRef<string | null>(null)

  // Google Drive state
  const [driveConnected, setDriveConnected] = useState(false)
  const [driveEmail, setDriveEmail] = useState<string | null>(null)
  const [driveEnabled, setDriveEnabled] = useState(true)
  const [driveCurrentFolderId, setDriveCurrentFolderId] = useState<string>('root')
  const [driveBreadcrumb, setDriveBreadcrumb] = useState<{ id: string; name: string }[]>([{ id: 'root', name: 'My Drive' }])
  const [driveFolders, setDriveFolders] = useState<DriveFolder[]>([])
  const [driveFiles, setDriveFiles] = useState<DriveFile[]>([])
  const [driveFolderName, setDriveFolderName] = useState<string>('My Drive')
  const [driveFilesLoading, setDriveFilesLoading] = useState(false)
  const [driveError, setDriveError] = useState<string | null>(null)
  const [selectedDriveFileIds, setSelectedDriveFileIds] = useState<Set<string>>(new Set())
  const [importingFromDrive, setImportingFromDrive] = useState(false)
  const [driveConnecting, setDriveConnecting] = useState(false)
  const [driveFolderLink, setDriveFolderLink] = useState('')
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

  useEffect(() => {
    if (driveConnected && !driveFilesLoading) {
      loadDriveFolder('root', [{ id: 'root', name: 'My Drive' }])
    }
  }, [driveConnected])

  const handleDriveConnect = async () => {
    setDriveError(null)
    setDriveConnecting(true)
    try {
      const r = await fetch(`${API_BASE}/drive/auth-url`, driveFetchOpts)
      if (!r.ok) {
        const err = await r.json().catch(() => ({}))
        const msg = typeof err.detail === 'string' ? err.detail : (err.detail?.message || 'Failed to get auth URL')
        throw new Error(msg)
      }
      const { url, session_id } = await r.json()
      if (session_id) driveSessionIdRef.current = session_id
      if (url) {
        window.location.href = url
      } else {
        throw new Error('No auth URL returned')
      }
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

  const loadDriveFolder = async (folderId: string, breadcrumb: { id: string; name: string }[]) => {
    setDriveError(null)
    setDriveFilesLoading(true)
    try {
      const headers: Record<string, string> = {}
      if (driveSessionIdRef.current) headers['X-RAG-Session'] = driveSessionIdRef.current
      const r = await fetch(`${API_BASE}/drive/folders/${encodeURIComponent(folderId)}/files`, { ...driveFetchOpts, headers })
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
      setDriveFolderName(folderName)
      setDriveFolders(data.folders || [])
      setDriveFiles(data.files || [])
      setSelectedDriveFileIds(new Set())
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'List failed'
      const hint = msg === 'Failed to fetch'
        ? 'RAG backend unreachable. Is it running on port 8001?'
        : msg
      setDriveError(hint)
      setDriveFolders([])
      setDriveFiles([])
    } finally {
      setDriveFilesLoading(false)
    }
  }

  const handleDriveBrowse = () => loadDriveFolder('root', [{ id: 'root', name: 'My Drive' }])

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

  const handleDriveGoToFolder = async () => {
    const trimmed = driveFolderLink.trim()
    if (!trimmed) return
    const match = trimmed.match(/\/folders\/([a-zA-Z0-9_-]+)/)
    const fid = match ? match[1] : trimmed
    loadDriveFolder(fid, [{ id: 'root', name: 'My Drive' }, { id: fid, name: fid }])
  }

  const handleDriveImport = async () => {
    const ids = selectedDriveFileIds.size > 0 ? [...selectedDriveFileIds] : driveFiles.map(f => f.id)
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
          payer: importMetadata.payer?.trim() || undefined,
          state: importMetadata.state?.trim() || undefined,
          program: importMetadata.program?.trim() || undefined,
          authority_level: importMetadata.authority_level?.trim() || undefined,
        }),
      })
      if (!r.ok) {
        if (r.status === 401) throw new Error('Connect Google Drive first')
        const err = await r.json().catch(() => ({}))
        throw new Error(err.detail || 'Import failed')
      }
      const data = await r.json()
      const ok = (data.results || []).filter((x: { status: string }) => x.status === 'completed').length
      onDocumentAdded?.()
      setDriveError(null)
      setSelectedDriveFileIds(new Set())
      if (ok > 0) setDriveFiles(prev => prev.filter(f => !ids.includes(f.id)))
    } catch (e) {
      setDriveError(e instanceof Error ? e.message : 'Import failed')
    } finally {
      setImportingFromDrive(false)
    }
  }

  const toggleDriveFile = (id: string) => {
    setSelectedDriveFileIds(prev => {
      const next = new Set(prev.size === 0 ? driveFiles.map(f => f.id) : prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }
  const selectAllDriveFiles = () => setSelectedDriveFileIds(new Set(driveFiles.map(f => f.id)))
  const clearDriveSelection = () => setSelectedDriveFileIds(new Set())

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
    setShowAllPages(false)
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
        let pathPrefix = scrapePathPrefix.trim()
        if (!pathPrefix) {
          try {
            const u = new URL(scrapeUrl.trim())
            const parts = u.pathname.split('/').filter(Boolean)
            if (parts.length > 0) {
              const first = parts[0]
              const dir = first.includes('.') ? first.split('.')[0] : first
              pathPrefix = '/' + dir
            }
          } catch {
            // leave pathPrefix empty
          }
        }
        if (pathPrefix) body.path_prefix = pathPrefix
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
          // Refresh pages/documents/summary from poll so displayPages has full content for "Add to RAG"
          pollScrapeStatus()
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
  }, [scrapeJobId, pollScrapeStatus])

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

  const addPageToRag = async (page: { url: string; text?: string; html?: string }) => {
    setImportingPages(true)
    try {
      const body: Record<string, unknown> = {
        pages: [{ url: page.url, text: page.text, html: page.html }],
      }
      if (importMetadata.display_name?.trim()) body.display_name = importMetadata.display_name.trim()
      if (importMetadata.payer?.trim()) body.payer = importMetadata.payer.trim()
      if (importMetadata.state?.trim()) body.state = importMetadata.state.trim()
      if (importMetadata.program?.trim()) body.program = importMetadata.program.trim()
      if (importMetadata.authority_level?.trim()) body.authority_level = importMetadata.authority_level.trim()
      const resp = await fetch(`${API_BASE}/documents/import-scraped-pages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}))
        throw new Error(err.detail?.message || err.detail || 'Import failed')
      }
      onDocumentAdded?.()
    } catch (e) {
      setScrapeError(e instanceof Error ? e.message : 'Import failed')
    } finally {
      setImportingPages(false)
    }
  }

  /** Add all displayed pages (as one doc) and all displayed docs to RAG. Used for auto-add when scrape completes. */
  const addAllDisplayedToRag = useCallback(async (pages: ScrapedPage[], docs: ScrapedDoc[]) => {
    if (pages.length === 0 && docs.length === 0) return
    setImportingPages(true)
    setScrapeError(null)
    try {
      if (pages.length > 0) {
        const body: Record<string, unknown> = {
          pages: pages.map((p) => ({ url: p.url, text: p.text, html: p.html })),
        }
        if (importMetadata.display_name?.trim()) body.display_name = importMetadata.display_name.trim()
        if (importMetadata.payer?.trim()) body.payer = importMetadata.payer.trim()
        if (importMetadata.state?.trim()) body.state = importMetadata.state.trim()
        if (importMetadata.program?.trim()) body.program = importMetadata.program.trim()
        if (importMetadata.authority_level?.trim()) body.authority_level = importMetadata.authority_level.trim()
        const resp = await fetch(`${API_BASE}/documents/import-scraped-pages`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        })
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}))
          throw new Error(err.detail?.message || err.detail || 'Import pages failed')
        }
        onDocumentAdded?.()
      }
      for (const d of docs) {
        setImportingDoc(d.gcs_path)
        try {
          const importBody: Record<string, string> = { gcs_path: d.gcs_path, filename: d.filename }
          if (importMetadata.payer?.trim()) importBody.payer = importMetadata.payer.trim()
          if (importMetadata.state?.trim()) importBody.state = importMetadata.state.trim()
          if (importMetadata.program?.trim()) importBody.program = importMetadata.program.trim()
          if (importMetadata.authority_level?.trim()) importBody.authority_level = importMetadata.authority_level.trim()
          const resp = await fetch(`${API_BASE}/documents/import-from-gcs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(importBody),
          })
          if (!resp.ok) {
            const err = await resp.json().catch(() => ({}))
            throw new Error(err.detail?.message || err.detail || 'Import document failed')
          }
          onDocumentAdded?.()
        } finally {
          setImportingDoc(null)
        }
      }
    } catch (e) {
      setScrapeError(e instanceof Error ? e.message : 'Auto-add to RAG failed')
    } finally {
      setImportingPages(false)
    }
  }, [importMetadata, onDocumentAdded])

  // Reset "already ran auto-add" when starting a new job
  useEffect(() => {
    autoAddRunForJobRef.current = null
  }, [scrapeJobId])

  // Auto-add to RAG when scrape completes and option is on (runs once per job)
  useEffect(() => {
    if (
      scrapeStatus !== 'completed' ||
      !autoAddToRagWhenComplete ||
      !scrapeJobId ||
      autoAddRunForJobRef.current === scrapeJobId
    ) return
    if (scrapePages.length === 0 && scrapeDocuments.length === 0) return
    autoAddRunForJobRef.current = scrapeJobId
    addAllDisplayedToRag(scrapePages, scrapeDocuments)
  }, [scrapeStatus, autoAddToRagWhenComplete, scrapeJobId, scrapePages, scrapeDocuments, addAllDisplayedToRag])

  const addSelectedToRag = async () => {
    const pageList = displayPages.filter((p) => selectedPageUrls.has(p.url))
    const docList = displayDocs.filter((d) => selectedDocPaths.has(d.gcs_path))
    if (pageList.length === 0 && docList.length === 0) return
    setImportingPages(true)
    setScrapeError(null)
    try {
      if (pageList.length > 0) {
        const body: Record<string, unknown> = {
          pages: pageList.map((p) => ({ url: p.url, text: p.text, html: p.html })),
        }
        if (importMetadata.display_name?.trim()) body.display_name = importMetadata.display_name.trim()
        if (importMetadata.payer?.trim()) body.payer = importMetadata.payer.trim()
        if (importMetadata.state?.trim()) body.state = importMetadata.state.trim()
        if (importMetadata.program?.trim()) body.program = importMetadata.program.trim()
        if (importMetadata.authority_level?.trim()) body.authority_level = importMetadata.authority_level.trim()
        const resp = await fetch(`${API_BASE}/documents/import-scraped-pages`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        })
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}))
          throw new Error(err.detail?.message || err.detail || 'Import pages failed')
        }
        onDocumentAdded?.()
      }
      for (const d of docList) {
        setImportingDoc(d.gcs_path)
        try {
          const importBody: Record<string, string> = { gcs_path: d.gcs_path, filename: d.filename }
          if (importMetadata.payer?.trim()) importBody.payer = importMetadata.payer.trim()
          if (importMetadata.state?.trim()) importBody.state = importMetadata.state.trim()
          if (importMetadata.program?.trim()) importBody.program = importMetadata.program.trim()
          if (importMetadata.authority_level?.trim()) importBody.authority_level = importMetadata.authority_level.trim()
          const resp = await fetch(`${API_BASE}/documents/import-from-gcs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(importBody),
          })
          if (!resp.ok) {
            const err = await resp.json().catch(() => ({}))
            throw new Error(err.detail?.message || err.detail || 'Import document failed')
          }
          onDocumentAdded?.()
        } finally {
          setImportingDoc(null)
        }
      }
    } catch (e) {
      setScrapeError(e instanceof Error ? e.message : 'Add to RAG failed')
    } finally {
      setImportingPages(false)
    }
  }

  const togglePageSelection = (url: string) => {
    setSelectedPageUrls((prev) => {
      const next = new Set(prev)
      if (next.has(url)) next.delete(url)
      else next.add(url)
      return next
    })
  }
  const toggleDocSelection = (gcsPath: string) => {
    setSelectedDocPaths((prev) => {
      const next = new Set(prev)
      if (next.has(gcsPath)) next.delete(gcsPath)
      else next.add(gcsPath)
      return next
    })
  }
  const selectAllPages = () => setSelectedPageUrls(new Set(displayPages.map((p) => p.url)))
  const selectAllDocs = () => setSelectedDocPaths(new Set(displayDocs.map((d) => d.gcs_path)))
  const clearSelection = () => {
    setSelectedPageUrls(new Set())
    setSelectedDocPaths(new Set())
  }
  const hasSelection = selectedPageUrls.size > 0 || selectedDocPaths.size > 0
  const anyImporting = importingDoc !== null || importingPages

  // Derive display data: use events when streaming; when job is done, prefer poll data if it has more (API is source of truth)
  const docsFromEvents = scrapeEvents
    .filter((e) => e.type === 'document')
    .map((e) => ({
      gcs_path: (e.payload?.gcs_path as string) || '',
      filename: (e.payload?.filename as string) || '',
      content_type: e.metadata?.content_type,
      source_url: e.metadata?.source_url,
      size_bytes: e.metadata?.size_bytes,
      source_page_url: e.metadata?.source_page_url,
    }))
  const pagesFromEvents = scrapeEvents.filter((e) => e.type === 'page').map((e) => ({
    url: (e.payload?.url as string) || e.metadata?.source_url || '',
    text: e.payload?.text as string | undefined,
    html: e.payload?.html as string | undefined,
    final_url: e.metadata?.final_url,
    depth: e.metadata?.depth,
    timestamp: e.metadata?.timestamp,
  }))
  const jobDone = scrapeStatus === 'completed' || scrapeStatus === 'failed'
  const displayDocs =
    jobDone && scrapeDocuments.length >= docsFromEvents.length
      ? scrapeDocuments
      : docsFromEvents.length > 0
        ? docsFromEvents
        : scrapeDocuments
  const displayPages =
    jobDone && scrapePages.length >= pagesFromEvents.length
      ? scrapePages
      : pagesFromEvents.length > 0
        ? pagesFromEvents
        : scrapePages
  const displaySummary =
    scrapeEvents.length > 0
      ? (scrapeEvents.find((e) => e.type === 'summary')?.payload?.summary as string | undefined) ?? scrapeSummary
      : scrapeSummary

  // Group documents by source page URL for hierarchy (page → documents)
  const pageGroups = displayPages.map((page) => ({
    page,
    docs: displayDocs.filter((d) => d.source_page_url === page.url),
  }))
  const remainingDocs = displayDocs.filter(
    (d) => !displayPages.some((p) => p.url === d.source_page_url)
  )

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
                <p className="scrape-tree-tip">
                  Tree scan follows links from this URL. For JS-heavy or SPA sites, use <strong>content mode: HTML</strong> below—the scraper fetches static HTML and does not run JavaScript.
                </p>
                <label>
                  Max depth: <input type="number" min={1} max={10} value={scrapeMaxDepth} onChange={(e) => setScrapeMaxDepth(parseInt(e.target.value, 10) || 3)} disabled={scraping} className="scrape-number" />
                </label>
                <label>
                  Max pages: <input type="number" min={1} max={200} value={scrapeMaxPages} onChange={(e) => setScrapeMaxPages(parseInt(e.target.value, 10) || 50)} disabled={scraping} className="scrape-number" />
                </label>
                <label className="scrape-path-scope">
                  Path scope (optional):{' '}
                  <input
                    type="text"
                    value={scrapePathPrefix}
                    onChange={(e) => setScrapePathPrefix(e.target.value)}
                    placeholder="/providers"
                    disabled={scraping}
                    className="scrape-path-input"
                    title="Only follow links under this path, e.g. /providers. Leave empty for whole site."
                  />
                  <button
                    type="button"
                    onClick={() => {
                      try {
                        const u = new URL(scrapeUrl.trim())
                        const parts = u.pathname.split('/').filter(Boolean)
                        if (parts.length === 0) {
                          setScrapePathPrefix('')
                          return
                        }
                        // Use first path segment as directory: providers.html -> /providers, providers/become... -> /providers
                        const first = parts[0]
                        const dir = first.includes('.') ? first.split('.')[0] : first
                        setScrapePathPrefix('/' + dir)
                      } catch {
                        setScrapePathPrefix('')
                      }
                    }}
                    disabled={scraping || !scrapeUrl.trim()}
                    className="scrape-auto-path-btn"
                    title="Derive path from URL (e.g. /providers from .../providers.html or .../providers/become-a-provider.html)"
                  >
                    Auto
                  </button>
                </label>
                <span className="scrape-path-hint">Leave empty for whole site.</span>
              </div>
            )}
            <div className={`scrape-content-options ${scrapeMode === 'tree' ? 'scrape-content-with-tree' : ''}`}>
              <label>
                Content mode:
                <select
                  value={scrapeContentMode}
                  onChange={(e) => setScrapeContentMode(e.target.value as 'text' | 'html' | 'both')}
                  disabled={scraping}
                  className="scrape-select"
                >
                  <option value="text">Text only</option>
                  <option value="html">HTML only (good for JS-heavy sites)</option>
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
            <label className="scrape-summary-option">
              <input
                type="checkbox"
                checked={autoAddToRagWhenComplete}
                onChange={(e) => setAutoAddToRagWhenComplete(e.target.checked)}
                disabled={scraping}
              />
              Auto-add to RAG when scrape completes (import + chunk + embed)
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
        {(scraping || scrapeStatus === 'completed' || scrapeStatus === 'failed') && (
          <div className="scrape-progress-block">
            {scraping && (
              <div className="scrape-progress">
                <span className="scrape-progress-dot" />
                <span className="scrape-progress-status">
                  {scrapeProgressMessage || 'Starting…'}
                </span>
                <span className="scrape-progress-counts">
                  {displayDocs.length} doc{displayDocs.length !== 1 ? 's' : ''}, {displayPages.length} page{displayPages.length !== 1 ? 's' : ''} so far
                </span>
              </div>
            )}
          </div>
        )}
        {(scraping || scrapeStatus === 'completed' || scrapeStatus === 'failed') && (
          <div className="scrape-results">
            {scraping && (displayDocs.length > 0 || displayPages.length > 0) && (
              <p className="scrape-results-live-title">
                Live — {displayDocs.length} document{displayDocs.length !== 1 ? 's' : ''}, {displayPages.length} page{displayPages.length !== 1 ? 's' : ''}
              </p>
            )}
            {displaySummary && (
              <div className="scrape-summary-block">
                <p className="scrape-results-title">Summary</p>
                <div className="scrape-summary-text">{displaySummary}</div>
              </div>
            )}
            {(displayPages.length > 0 || displayDocs.length > 0) && (
              <>
                <div className="scrape-results-toolbar">
                  <span className="scrape-results-toolbar-label">Selection:</span>
                  <button type="button" className="btn btn-small" onClick={selectAllPages} disabled={anyImporting}>
                    Select all {displayPages.length} pages
                  </button>
                  <button type="button" className="btn btn-small" onClick={selectAllDocs} disabled={anyImporting}>
                    Select all {displayDocs.length} documents
                  </button>
                  <button type="button" className="btn btn-small" onClick={clearSelection} disabled={!hasSelection || anyImporting}>
                    Clear
                  </button>
                  <button
                    type="button"
                    className="btn btn-primary btn-small"
                    onClick={addSelectedToRag}
                    disabled={!hasSelection || anyImporting}
                  >
                    {anyImporting ? 'Importing…' : `Add selected to RAG (${selectedPageUrls.size + selectedDocPaths.size})`}
                  </button>
                </div>
                <details className="scrape-import-metadata" open={false}>
                  <summary>Metadata (optional) — Payor, State, Program, Authority level</summary>
                  <p className="scrape-metadata-hint">Set these when adding pages or documents to RAG so the document carries this metadata for filtering and display.</p>
                  <div className="scrape-metadata-fields">
                    <label className="scrape-metadata-label">
                      <span>Payor name</span>
                      <input
                        type="text"
                        className="scrape-metadata-input"
                        placeholder="e.g. UnitedHealthcare"
                        value={importMetadata.payer}
                        onChange={e => setImportMetadata(prev => ({ ...prev, payer: e.target.value }))}
                      />
                    </label>
                    <label className="scrape-metadata-label">
                      <span>State</span>
                      <select
                        className="scrape-metadata-select"
                        value={importMetadata.state}
                        onChange={e => setImportMetadata(prev => ({ ...prev, state: e.target.value }))}
                      >
                        <option value="">—</option>
                        {STATE_OPTIONS.map(opt => (
                          <option key={opt.value} value={opt.value}>{opt.label}</option>
                        ))}
                      </select>
                    </label>
                    <label className="scrape-metadata-label">
                      <span>Program</span>
                      <input
                        type="text"
                        className="scrape-metadata-input"
                        placeholder="e.g. Medicaid, Medicare"
                        value={importMetadata.program}
                        onChange={e => setImportMetadata(prev => ({ ...prev, program: e.target.value }))}
                      />
                    </label>
                    <label className="scrape-metadata-label">
                      <span>Authority level</span>
                      <select
                        className="scrape-metadata-select"
                        value={importMetadata.authority_level}
                        onChange={e => setImportMetadata(prev => ({ ...prev, authority_level: e.target.value }))}
                      >
                        <option value="">—</option>
                        {AUTHORITY_LEVEL_OPTIONS.map(opt => (
                          <option key={opt.value} value={opt.value}>{opt.label}</option>
                        ))}
                      </select>
                    </label>
                    <label className="scrape-metadata-label">
                      <span>Display name</span>
                      <input
                        type="text"
                        className="scrape-metadata-input"
                        placeholder="Optional short name for the document"
                        value={importMetadata.display_name}
                        onChange={e => setImportMetadata(prev => ({ ...prev, display_name: e.target.value }))}
                      />
                    </label>
                  </div>
                </details>
                <p className="scrape-results-title">Results (by page)</p>
                {(showAllPages ? pageGroups : pageGroups.slice(0, 5)).map(({ page: p, docs }, i) => {
                  const content = p.text ?? p.html ?? '(no content)'
                  const preview = content.slice(0, 2000) + (content.length > 2000 ? '...' : '')
                  return (
                    <div key={`${p.url}-${i}`} className="scrape-page-group">
                      <details className="scrape-page-details">
                        <summary>
                          <span className="scrape-page-row-main">
                            <label className="scrape-row-select" onClick={e => e.stopPropagation()}>
                              <input
                                type="checkbox"
                                checked={selectedPageUrls.has(p.url)}
                                onChange={() => togglePageSelection(p.url)}
                                disabled={anyImporting}
                              />
                              <span className="scrape-checkbox-label">Select</span>
                            </label>
                            <button
                              type="button"
                              className="scrape-page-view-link"
                              onClick={(e) => {
                                e.preventDefault()
                                e.stopPropagation()
                                setReaderPage(p)
                              }}
                            >
                              {p.url}
                            </button>
                            <span className="scrape-page-view-hint">View</span>
                          </span>
                          {p.depth != null && <span className="scrape-page-meta">depth {p.depth}</span>}
                          {p.timestamp && <span className="scrape-page-meta">{p.timestamp}</span>}
                          <button
                            type="button"
                            className="btn btn-small scrape-page-add-rag"
                            onClick={(e) => {
                              e.preventDefault()
                              e.stopPropagation()
                              addPageToRag(p)
                            }}
                            disabled={importingPages}
                          >
                            {importingPages ? 'Importing…' : 'Add to RAG'}
                          </button>
                        </summary>
                        <pre className="scrape-page-text">{preview}</pre>
                        {docs.length > 0 && (
                          <ul className="scrape-doc-list scrape-doc-sublist">
                            {docs.map((d) => (
                              <li key={d.gcs_path}>
                                <label className="scrape-row-select">
                                  <input
                                    type="checkbox"
                                    checked={selectedDocPaths.has(d.gcs_path)}
                                    onChange={() => toggleDocSelection(d.gcs_path)}
                                    disabled={anyImporting}
                                  />
                                  <span>{d.filename}</span>
                                </label>
                                {(d.size_bytes != null || d.content_type) && (
                                  <span className="scrape-doc-meta">
                                    {d.size_bytes != null && `${(d.size_bytes / 1024).toFixed(1)} KB`}
                                    {d.content_type && ` · ${d.content_type}`}
                                  </span>
                                )}
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
                        )}
                      </details>
                    </div>
                  )
                })}
                {pageGroups.length > 5 && !showAllPages && (
                  <button
                    type="button"
                    className="btn btn-secondary scrape-show-more-pages"
                    onClick={() => setShowAllPages(true)}
                  >
                    Show all {displayPages.length} pages
                  </button>
                )}
                {remainingDocs.length > 0 && (
                  <div className="scrape-page-group">
                    <p className="scrape-results-title">Other downloaded documents</p>
                    <ul className="scrape-doc-list">
                      {remainingDocs.map((d) => (
                        <li key={d.gcs_path}>
                          <label className="scrape-row-select">
                            <input
                              type="checkbox"
                              checked={selectedDocPaths.has(d.gcs_path)}
                              onChange={() => toggleDocSelection(d.gcs_path)}
                              disabled={anyImporting}
                            />
                            <span>{d.filename}</span>
                          </label>
                          {(d.size_bytes != null || d.content_type || d.source_page_url) && (
                            <span className="scrape-doc-meta">
                              {d.size_bytes != null && `${(d.size_bytes / 1024).toFixed(1)} KB`}
                              {d.content_type && ` · ${d.content_type}`}
                              {d.source_page_url && ` · from ${d.source_page_url}`}
                            </span>
                          )}
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
                  </div>
                )}
                <p className="upload-hint">Page content and PDF documents can be added to RAG for chunking and embedding.</p>
              </>
            )}
            {displayDocs.length === 0 && displayPages.length === 0 && !displaySummary && (scrapeStatus === 'completed' || scrapeStatus === 'failed') && (
              <div className="scrape-empty-state">
                <p className="scrape-empty-title">No documents or page content extracted</p>
                <p className="scrape-empty-desc">
                  Try <strong>content mode: HTML</strong> (for JS-heavy sites) or a simpler page, then run the scan again.
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
        {/* Google Drive Method */}
        <details className={`input-method-card collapsible ${!driveEnabled ? 'disabled' : ''}`} open={driveConnected}>
          <summary className="collapsible-summary">
            <span className="collapsible-title">Google Drive</span>
            <span className="collapsible-desc">
              {driveConnected ? `Connected as ${driveEmail || 'Google account'}` : 'Import documents from Google Drive'}
            </span>
            {driveConnected && <span className="collapsible-badge badge-ok">Connected</span>}
            {!driveEnabled && <span className="collapsible-badge">Not configured</span>}
          </summary>
          <div className="collapsible-content">
            {!driveEnabled ? (
              <p className="upload-hint">Set DRIVE_API_ENABLED=true and OAuth credentials in .env to enable.</p>
            ) : !driveConnected ? (
              <div>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={handleDriveConnect}
                  disabled={driveConnecting}
                >
                  {driveConnecting ? 'Redirecting…' : 'Connect Google Drive'}
                </button>
                {driveError && (
                  <div className="error-message" role="alert" style={{ marginTop: '0.5rem' }}>
                    {driveError}
                  </div>
                )}
                <p className="upload-hint">Sign in with Google to access your Drive. PDFs and Google Docs supported.</p>
              </div>
            ) : (
              <div className="drive-import-section">
                <div className="drive-toolbar" style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'center', marginBottom: '0.75rem' }}>
                  <button type="button" className="btn btn-secondary btn-sm" onClick={handleDriveDisconnect}>
                    Disconnect
                  </button>
                  <button
                    type="button"
                    className="btn btn-primary btn-sm"
                    onClick={handleDriveBrowse}
                    disabled={driveFilesLoading}
                  >
                    {driveFilesLoading ? 'Loading…' : 'Browse My Drive'}
                  </button>
                  {driveBreadcrumb.length > 1 && (
                    <button type="button" className="btn btn-secondary btn-sm" onClick={handleDriveNavigateUp}>
                      ↑ Up
                    </button>
                  )}
                </div>
                <div className="drive-breadcrumb" style={{ fontSize: '0.875rem', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>
                  {driveBreadcrumb.map((b, i) => (
                    <span key={b.id}>
                      {i > 0 && ' › '}
                      <button
                        type="button"
                        className="drive-breadcrumb-link"
                        onClick={() => loadDriveFolder(b.id, driveBreadcrumb.slice(0, i + 1))}
                        style={{ background: 'none', border: 'none', padding: 0, cursor: 'pointer', color: 'inherit', textDecoration: 'underline' }}
                      >
                        {b.name}
                      </button>
                    </span>
                  ))}
                </div>
                <div className="drive-goto" style={{ marginBottom: '0.75rem', display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
                  <input
                    type="text"
                    placeholder="Or paste folder link to jump"
                    value={driveFolderLink}
                    onChange={e => setDriveFolderLink(e.target.value)}
                    className="scrape-url-input"
                    style={{ flex: 1, minWidth: 200 }}
                    disabled={driveFilesLoading}
                  />
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={handleDriveGoToFolder}
                    disabled={!driveFolderLink.trim() || driveFilesLoading}
                  >
                    Go to folder
                  </button>
                </div>
                {driveError && (
                  <div className="error-message" role="alert" style={{ marginBottom: '0.5rem' }}>
                    {driveError}
                  </div>
                )}
                {(driveFolders.length > 0 || driveFiles.length > 0) && (
                  <div className="drive-files-list" style={{ marginTop: '0.5rem' }}>
                    {driveFolders.length > 0 && (
                      <ul className="drive-folders-ul" style={{ listStyle: 'none', padding: 0, margin: '0 0 1rem 0' }}>
                        {driveFolders.map(f => (
                          <li key={f.id}>
                            <button
                              type="button"
                              className="drive-folder-item"
                              onClick={() => handleDriveNavigateToFolder(f)}
                              style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.35rem 0', background: 'none', border: 'none', cursor: 'pointer', color: 'inherit', fontSize: 'inherit', textAlign: 'left' }}
                            >
                              <span style={{ opacity: 0.7 }}>📁</span>
                              <span>{f.name}</span>
                            </button>
                          </li>
                        ))}
                      </ul>
                    )}
                    {driveFiles.length > 0 && (
                      <>
                        <div className="drive-files-actions" style={{ marginBottom: '0.5rem' }}>
                          <button type="button" className="btn btn-secondary btn-sm" onClick={selectAllDriveFiles}>
                            Select all
                          </button>
                          <button type="button" className="btn btn-secondary btn-sm" onClick={clearDriveSelection}>
                            Clear
                          </button>
                          <span className="drive-files-count" style={{ marginLeft: '0.5rem', fontSize: '0.875rem' }}>
                            {selectedDriveFileIds.size || driveFiles.length} selected
                          </span>
                        </div>
                        <ul className="drive-files-ul">
                          {driveFiles.map(f => (
                            <li key={f.id} className="drive-file-item">
                              <label>
                                <input
                                  type="checkbox"
                                  checked={selectedDriveFileIds.size === 0 ? true : selectedDriveFileIds.has(f.id)}
                                  onChange={() => toggleDriveFile(f.id)}
                                />
                                <span className="drive-file-name">{f.name}</span>
                                {f.mimeType?.includes('document') && <span className="drive-file-type">(Doc)</span>}
                              </label>
                            </li>
                          ))}
                        </ul>
                        <button
                          type="button"
                          className="btn btn-primary"
                          onClick={handleDriveImport}
                          disabled={importingFromDrive}
                          style={{ marginTop: '0.5rem' }}
                        >
                          {importingFromDrive ? 'Importing…' : 'Import selected'}
                        </button>
                      </>
                    )}
                    {driveFolders.length === 0 && driveFiles.length === 0 && !driveFilesLoading && (
                      <p className="upload-hint" style={{ margin: 0 }}>This folder is empty or has no PDFs/Google Docs.</p>
                    )}
                  </div>
                )}
                {driveFolders.length === 0 && driveFiles.length === 0 && !driveFilesLoading && driveCurrentFolderId !== 'root' && (
                  <p className="upload-hint">No folders or supported files here.</p>
                )}
              </div>
            )}
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

      {/* Reader modal: view page content */}
      {readerPage && (
        <div
          className="scrape-reader-overlay"
          role="dialog"
          aria-modal="true"
          aria-label="Page content reader"
          onClick={() => setReaderPage(null)}
        >
          <div
            className="scrape-reader-modal"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="scrape-reader-header">
              <a
                href={readerPage.url}
                target="_blank"
                rel="noopener noreferrer"
                className="scrape-reader-url"
                onClick={(e) => e.stopPropagation()}
              >
                {readerPage.url}
              </a>
              <button
                type="button"
                className="scrape-reader-close"
                onClick={() => setReaderPage(null)}
                aria-label="Close reader"
              >
                ×
              </button>
            </div>
            <div className="scrape-reader-body">
              {(() => {
                const raw = readerPage.text ?? readerPage.html ?? ''
                const text = readerPage.text
                  ? readerPage.text
                  : readerPage.html
                    ? raw
                        .replace(/<\s*br\s*\/?>/gi, '\n')
                        .replace(/<\s*\/\s*(p|div|li|tr)/gi, '\n')
                        .replace(/<[^>]+>/g, ' ')
                        .replace(/[ \t]+/g, ' ')
                        .replace(/\n\s*\n\s*\n/g, '\n\n')
                        .trim()
                    : ''
                return text ? (
                  <div className="scrape-reader-content">
                    {text}
                  </div>
                ) : (
                  <p className="scrape-reader-empty">No text content for this page.</p>
                )
              })()}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
