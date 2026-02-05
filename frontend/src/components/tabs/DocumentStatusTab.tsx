import React, { useState, useEffect } from 'react'
import { STATE_OPTIONS, AUTHORITY_LEVEL_OPTIONS } from '../../lib/documentMetadata'
import './DocumentStatusTab.css'

import { API_BASE } from '../../config'

interface Document {
  id: string
  filename: string
  display_name?: string | null
  extraction_status: string
  chunking_status: string | null
  embedding_status?: string | null
  created_at: string
  gcs_path: string
  has_errors?: string  // 'true' or 'false'
  error_count?: number
  critical_error_count?: number
  review_status?: string
  payer?: string | null
  state?: string | null
  program?: string | null
  authority_level?: string | null
  effective_date?: string | null
  termination_date?: string | null
  published_at?: string | null
  published_rows?: number | null
  publish_verification_passed?: boolean | null
  publish_verification_message?: string | null
  publish_count?: number
}

export interface ChunkingOptions {
  threshold: number
  critiqueEnabled: boolean
  maxRetries: number
  extractionEnabled: boolean
  promptVersions?: Record<string, string>
}

const DEFAULT_CHUNK_OPTIONS: ChunkingOptions = {
  threshold: 0.6,
  critiqueEnabled: true,
  maxRetries: 2,
  extractionEnabled: true,
}

/** Default termination date: 6 months from today (ISO date string). Used when document has no expiry. */
function defaultTerminationDate(): string {
  const d = new Date()
  d.setMonth(d.getMonth() + 6)
  return d.toISOString().slice(0, 10)
}

interface DocumentStatusTabProps {
  onStartChunking: (documentId: string, options: ChunkingOptions) => Promise<void>
  onStopChunking: (documentId: string) => Promise<void>
  onViewDocument: (documentId: string) => void
  onViewDocumentDetail?: (documentId: string) => void
  onDeleteDocument: (documentId: string) => Promise<void>
  onRestartChunking?: (documentId: string, options: ChunkingOptions) => Promise<void>
  onStartEmbedding?: (documentId: string) => Promise<void>
  onResetEmbedding?: (documentId: string) => Promise<void>
  onMarkReadyForChunking?: (documentId: string) => Promise<void>
}

export function DocumentStatusTab({ 
  onStartChunking, 
  onStopChunking,
  onViewDocument,
  onViewDocumentDetail,
  onDeleteDocument,
  onRestartChunking,
  onStartEmbedding,
  onResetEmbedding,
  onMarkReadyForChunking,
}: DocumentStatusTabProps) {
  const [documents, setDocuments] = useState<Document[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedDocs, setSelectedDocs] = useState<Set<string>>(new Set())
  const [searchQuery, setSearchQuery] = useState('')
  const [sortColumn, setSortColumn] = useState<string>('created_at')
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc')
  const [editingMetadataDocumentId, setEditingMetadataDocumentId] = useState<string | null>(null)
  const [metadataForm, setMetadataForm] = useState<{ display_name: string; payer: string; state: string; program: string; authority_level: string; effective_date: string; termination_date: string }>({ display_name: '', payer: '', state: '', program: '', authority_level: '', effective_date: '', termination_date: '' })
  const [metadataSaving, setMetadataSaving] = useState(false)
  const [metadataError, setMetadataError] = useState<string | null>(null)
  const [openChunkMenuDocId, setOpenChunkMenuDocId] = useState<string | null>(null)
  const [openEmbeddingMenuDocId, setOpenEmbeddingMenuDocId] = useState<string | null>(null)
  const [openPublishMenuDocId, setOpenPublishMenuDocId] = useState<string | null>(null)
  const [openActionsMenuDocId, setOpenActionsMenuDocId] = useState<string | null>(null)
  const [publishLoadingDocId, setPublishLoadingDocId] = useState<string | null>(null)
  const [publishMessage, setPublishMessage] = useState<{ docId: string; text: string } | null>(null)
  const [chunkStatusLoadingDocId, setChunkStatusLoadingDocId] = useState<string | null>(null)
  const [chunkStatusMessage, setChunkStatusMessage] = useState<{ docId: string; text: string } | null>(null)
  const [markReadyLoadingDocId, setMarkReadyLoadingDocId] = useState<string | null>(null)
  const [promptsConfig, setPromptsConfig] = useState<{ prompts: Record<string, string[]>; default: Record<string, string> } | null>(null)
  const [chunkOptionsByDoc, setChunkOptionsByDoc] = useState<Record<string, ChunkingOptions>>({})

  useEffect(() => {
    if (!openChunkMenuDocId) return
    const load = async () => {
      try {
        const res = await fetch(`${API_BASE}/config/prompts`)
        if (res.ok) {
          const data = await res.json()
          setPromptsConfig({ prompts: data.prompts || {}, default: data.default || {} })
        }
      } catch (e) {
        console.error('Failed to load prompts config:', e)
      }
    }
    load()
  }, [openChunkMenuDocId])

  const getChunkOptionsForDoc = (docId: string): ChunkingOptions => {
    const base = chunkOptionsByDoc[docId] ?? { ...DEFAULT_CHUNK_OPTIONS }
    if (promptsConfig && Object.keys(promptsConfig.prompts).length > 0 && !base.promptVersions) {
      return {
        ...base,
        promptVersions: { ...promptsConfig.default },
      }
    }
    return base
  }

  const setChunkOptionForDoc = (docId: string, updater: (prev: ChunkingOptions) => ChunkingOptions) => {
    setChunkOptionsByDoc(prev => ({
      ...prev,
      [docId]: updater(prev[docId] ?? { ...DEFAULT_CHUNK_OPTIONS, promptVersions: promptsConfig ? { ...promptsConfig.default } : {} }),
    }))
  }

  const openMetadataForm = (doc: Document) => {
    setEditingMetadataDocumentId(doc.id)
    const term = (doc.termination_date ?? '').trim()
    setMetadataForm({
      display_name: doc.display_name ?? '',
      payer: doc.payer ?? '',
      state: doc.state ?? '',
      program: doc.program ?? '',
      authority_level: doc.authority_level ?? '',
      effective_date: doc.effective_date ?? '',
      termination_date: term || defaultTerminationDate(),
    })
    setMetadataError(null)
  }

  const closeMetadataForm = () => {
    setEditingMetadataDocumentId(null)
    setMetadataError(null)
  }

  const isMetadataMissing = (doc: Document) => {
    const p = (doc.payer ?? '').trim()
    const s = (doc.state ?? '').trim()
    const prog = (doc.program ?? '').trim()
    return p === '' && s === '' && prog === ''
  }

  const loadDocuments = async () => {
    setLoading(true)
    try {
      const response = await fetch(`${API_BASE}/documents`)
      if (response.ok) {
        const data = await response.json()
        setDocuments(data.documents || [])
      }
    } catch (err) {
      console.error('Failed to load documents:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadDocuments()
    // Auto-refresh every 5 seconds when tab is active
    const interval = setInterval(loadDocuments, 5000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (!openChunkMenuDocId && !openEmbeddingMenuDocId && !openPublishMenuDocId && !openActionsMenuDocId) return
    const closeMenus = () => {
      setOpenChunkMenuDocId(null)
      setOpenEmbeddingMenuDocId(null)
      setOpenPublishMenuDocId(null)
      setOpenActionsMenuDocId(null)
    }
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeMenus()
    }
    const onDocClick = () => closeMenus()
    const t = setTimeout(() => document.addEventListener('click', onDocClick), 0)
      document.addEventListener('keydown', onKeyDown)
    return () => {
      clearTimeout(t)
      document.removeEventListener('click', onDocClick)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [openChunkMenuDocId, openEmbeddingMenuDocId, openPublishMenuDocId, openActionsMenuDocId])

  const handleSelectDoc = (docId: string) => {
    setSelectedDocs(prev => {
      const next = new Set(prev)
      if (next.has(docId)) {
        next.delete(docId)
      } else {
        next.add(docId)
      }
      return next
    })
  }

  const handleSelectAll = () => {
    if (selectedDocs.size === filteredDocuments.length) {
      setSelectedDocs(new Set())
    } else {
      setSelectedDocs(new Set(filteredDocuments.map(d => d.id)))
    }
  }

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortColumn(column)
      setSortDirection('asc')
    }
  }

  const handleBatchStartChunking = async () => {
    const options: ChunkingOptions = { ...DEFAULT_CHUNK_OPTIONS }
    for (const docId of selectedDocs) {
      try {
        await onStartChunking(docId, options)
      } catch (err) {
        console.error(`Failed to start chunking for ${docId}:`, err)
      }
    }
    setSelectedDocs(new Set())
    await loadDocuments()
  }

  const handleSaveMetadata = async (docId: string, payload: { display_name?: string; payer?: string; state?: string; program?: string; authority_level?: string; effective_date?: string; termination_date?: string }) => {
    setMetadataSaving(true)
    setMetadataError(null)
    try {
      const response = await fetch(`${API_BASE}/documents/${docId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        setMetadataError(err.detail || 'Failed to update metadata')
        return
      }
      await loadDocuments()
      setEditingMetadataDocumentId(null)
    } catch (err) {
      setMetadataError('Failed to update metadata')
    } finally {
      setMetadataSaving(false)
    }
  }

  const handleDelete = async (docId: string) => {
    try {
      await onDeleteDocument(docId)
      // Remove from selected if it was selected
      setSelectedDocs(prev => {
        const next = new Set(prev)
        next.delete(docId)
        return next
      })
      // Reload documents to refresh the list
      await loadDocuments()
    } catch (err) {
      console.error(`Failed to delete document ${docId}:`, err)
    }
  }

  const handlePublish = async (docId: string) => {
    setPublishMessage(null)
    setPublishLoadingDocId(docId)
    try {
      const res = await fetch(`${API_BASE}/documents/${docId}/publish`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
      const data = await res.json().catch(() => ({}))
      if (res.ok) {
        const n = data.rows_written ?? 0
        const verified = data.verification_passed === true
        const msg = verified
          ? `Published ${n} rows. ✓ Verified.`
          : data.verification_passed === false
            ? `Published ${n} rows. Verification failed: ${data.verification_message ?? 'unknown'}.`
            : `Published ${n} rows.`
        setPublishMessage({ docId, text: msg })
        await loadDocuments()
      } else {
        const msg = res.status === 404
          ? 'Publish not available yet (backend not connected).'
          : (data.detail || (typeof data.detail === 'string' ? data.detail : `Publish failed (${res.status})`))
        setPublishMessage({ docId, text: String(msg) })
      }
    } catch (err) {
      setPublishMessage({ docId, text: err instanceof Error ? err.message : 'Publish request failed.' })
    } finally {
      setPublishLoadingDocId(null)
    }
  }

  const handleMarkChunkingComplete = async (docId: string) => {
    setChunkStatusMessage(null)
    setChunkStatusLoadingDocId(docId)
    try {
      const res = await fetch(`${API_BASE}/documents/${docId}/chunking/status`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: 'completed' }),
      })
      const data = await res.json().catch(() => ({}))
      if (res.ok) {
        setChunkStatusMessage({ docId, text: 'Chunking status set to complete.' })
        setOpenChunkMenuDocId(null)
        await loadDocuments()
      } else {
        const msg = (data.detail && (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail))) || `Failed (${res.status})`
        setChunkStatusMessage({ docId, text: msg })
      }
    } catch (err) {
      setChunkStatusMessage({ docId, text: err instanceof Error ? err.message : 'Request failed.' })
    } finally {
      setChunkStatusLoadingDocId(null)
    }
  }

  const getStatusBadge = (status: string | null, _type: 'extraction' | 'chunking') => {
    if (!status) return <span className="status-badge status-idle">—</span>
    
    const statusClass = `status-badge status-${status}`
    const statusLabel = status.charAt(0).toUpperCase() + status.slice(1).replace('_', ' ')
    
    return <span className={statusClass}>{statusLabel}</span>
  }

  const filteredDocuments = documents.filter(doc => {
    const name = (doc.display_name || doc.filename || '').toLowerCase()
    const filename = (doc.filename || '').toLowerCase()
    const q = searchQuery.toLowerCase()
    return name.includes(q) || filename.includes(q)
  })

  const sortedDocuments = [...filteredDocuments].sort((a, b) => {
    let aVal: any = a[sortColumn as keyof Document]
    let bVal: any = b[sortColumn as keyof Document]
    
    if (sortColumn === 'created_at') {
      aVal = new Date(aVal).getTime()
      bVal = new Date(bVal).getTime()
    } else if (typeof aVal === 'string') {
      aVal = aVal.toLowerCase()
      bVal = bVal.toLowerCase()
    }
    
    if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1
    if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1
    return 0
  })

  return (
    <div className="document-status-tab">
      <div className="status-toolbar">
        <div className="toolbar-left">
          <input
            type="text"
            placeholder="Search documents..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="search-input"
          />
        </div>
        <div className="toolbar-right">
          {selectedDocs.size > 0 && (
            <button
              onClick={handleBatchStartChunking}
              className="btn btn-primary"
            >
              Invoke Extraction ({selectedDocs.size})
            </button>
          )}
          <button
            onClick={loadDocuments}
            className="btn btn-secondary"
            disabled={loading}
          >
            {loading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      </div>

      <div className="status-table-container">
        <table className="status-table">
          <thead>
            <tr>
              <th className="col-checkbox">
                <input
                  type="checkbox"
                  checked={selectedDocs.size === filteredDocuments.length && filteredDocuments.length > 0}
                  onChange={handleSelectAll}
                />
              </th>
              <th 
                className="col-name sortable"
                onClick={() => handleSort('filename')}
              >
                Document Name
                {sortColumn === 'filename' && (
                  <span className="sort-indicator">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                )}
              </th>
              <th className="col-status" title="Store (raw text) and Convert to MD (canonical markdown) per page">
                Store / MD
              </th>
              <th className="col-status">Chunk</th>
              <th className="col-status">Embedding</th>
              <th className="col-status">Publish</th>
              <th className="col-status" title="First publish vs subsequent republish">Last</th>
              <th className="col-errors">Errors</th>
              <th 
                className="col-date sortable"
                onClick={() => handleSort('created_at')}
              >
                Created At
                {sortColumn === 'created_at' && (
                  <span className="sort-indicator">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                )}
              </th>
              <th className="col-actions">Actions</th>
            </tr>
          </thead>
          <tbody>
            {sortedDocuments.length === 0 ? (
              <tr>
                <td colSpan={10} className="empty-state">
                  {loading ? 'Loading documents...' : 'No documents found'}
                </td>
              </tr>
            ) : (
              sortedDocuments.map((doc) => (
                <React.Fragment key={doc.id}>
                  <tr
                    className={isMetadataMissing(doc) ? 'doc-row-missing-metadata' : ''}
                  >
                    <td className="col-checkbox">
                      <input
                        type="checkbox"
                        checked={selectedDocs.has(doc.id)}
                        onChange={() => handleSelectDoc(doc.id)}
                      />
                    </td>
                    <td className="col-name">
                      <span className="document-display-name">{doc.display_name?.trim() || doc.filename}</span>
                      {doc.display_name?.trim() && (
                        <span className="document-filename-hint" title={`File: ${doc.filename}`}> ({doc.filename})</span>
                      )}
                      {isMetadataMissing(doc) && (
                        <span className="metadata-missing-badge" title="Payer, state, and program are empty">
                          Missing metadata
                        </span>
                      )}
                    </td>
                    <td className="col-status col-extraction-with-action">
                      {getStatusBadge(doc.extraction_status, 'extraction')}
                      {doc.extraction_status === 'uploaded' && onMarkReadyForChunking && (
                        <button
                          type="button"
                          className="btn btn-small btn-link scrape-mark-ready"
                          disabled={markReadyLoadingDocId === doc.id}
                          title="Scraped/web docs are already stored; mark ready so you can start chunking"
                          onClick={async (e) => {
                            e.stopPropagation()
                            setMarkReadyLoadingDocId(doc.id)
                            try {
                              await onMarkReadyForChunking(doc.id)
                            } finally {
                              setMarkReadyLoadingDocId(null)
                            }
                          }}
                        >
                          {markReadyLoadingDocId === doc.id ? 'Updating…' : 'Mark ready for chunking'}
                        </button>
                      )}
                    </td>
                    <td className="col-status col-chunk-with-menu">
                      <div className="chunk-cell">
                        {getStatusBadge(doc.chunking_status, 'chunking')}
                        {(() => {
                          const hasChunkActions =
                            doc.chunking_status === 'idle' || doc.chunking_status === null ||
                            doc.chunking_status === 'in_progress' || doc.chunking_status === 'queued' ||
                            ((doc.chunking_status === 'stopped' || doc.chunking_status === 'failed') && onRestartChunking)
                          if (!hasChunkActions) return null
                          return (
                            <div className="dropdown-wrap">
                              <button
                                type="button"
                                className="btn-icon btn-kebab"
                                onClick={(e) => { e.stopPropagation(); setOpenChunkMenuDocId(prev => prev === doc.id ? null : doc.id) }}
                                title="Chunk actions"
                                aria-haspopup="true"
                                aria-expanded={openChunkMenuDocId === doc.id}
                              >
                                ⋮
                              </button>
                              {openChunkMenuDocId === doc.id && (
                                <div className="dropdown-menu dropdown-menu-chunk-options" onClick={(e) => e.stopPropagation()}>
                                  {doc.chunking_status === 'in_progress' ? (
                                    <>
                                      <button
                                        type="button"
                                        className="dropdown-item dropdown-item-danger"
                                        onClick={() => { onStopChunking(doc.id); setOpenChunkMenuDocId(null) }}
                                      >
                                        Stop chunking
                                      </button>
                                      <button
                                        type="button"
                                        className="dropdown-item"
                                        disabled={chunkStatusLoadingDocId === doc.id}
                                        title="Set chunking status to complete (use when stuck in progress)"
                                        onClick={() => { handleMarkChunkingComplete(doc.id) }}
                                      >
                                        {chunkStatusLoadingDocId === doc.id ? 'Updating…' : 'Mark chunking complete'}
                                      </button>
                                    </>
                                  ) : doc.chunking_status === 'queued' ? (
                                    <button
                                      type="button"
                                      className="dropdown-item"
                                      disabled={chunkStatusLoadingDocId === doc.id}
                                      title="Set chunking status to complete"
                                      onClick={() => { handleMarkChunkingComplete(doc.id) }}
                                    >
                                      {chunkStatusLoadingDocId === doc.id ? 'Updating…' : 'Mark chunking complete'}
                                    </button>
                                  ) : (
                                    <>
                                      {(doc.chunking_status === 'idle' || doc.chunking_status === null || (doc.chunking_status === 'stopped' || doc.chunking_status === 'failed')) && (
                                        <div className="chunk-options-form">
                                          <div className="chunk-option-row">
                                            <label className="chunk-option-checkbox">
                                              <input
                                                type="checkbox"
                                                checked={getChunkOptionsForDoc(doc.id).extractionEnabled}
                                                onChange={e => setChunkOptionForDoc(doc.id, p => ({ ...p, extractionEnabled: e.target.checked }))}
                                              />
                                              Run atomic extraction (LLM facts + critique)
                                            </label>
                                          </div>
                                          <div className="chunk-option-row">
                                            <label className="chunk-option-checkbox">
                                              <input
                                                type="checkbox"
                                                checked={getChunkOptionsForDoc(doc.id).critiqueEnabled}
                                                onChange={e => setChunkOptionForDoc(doc.id, p => ({ ...p, critiqueEnabled: e.target.checked }))}
                                              />
                                              Run critique
                                            </label>
                                          </div>
                                          <div className="chunk-option-row">
                                            <label htmlFor={`chunk-retries-${doc.id}`}>Max retries</label>
                                            <input
                                              id={`chunk-retries-${doc.id}`}
                                              type="number"
                                              min={0}
                                              max={10}
                                              value={getChunkOptionsForDoc(doc.id).maxRetries}
                                              onChange={e => setChunkOptionForDoc(doc.id, p => ({ ...p, maxRetries: parseInt(e.target.value, 10) || 0 }))}
                                            />
                                          </div>
                                          <div className="chunk-option-row">
                                            <label htmlFor={`chunk-threshold-${doc.id}`}>Threshold (0–1)</label>
                                            <input
                                              id={`chunk-threshold-${doc.id}`}
                                              type="number"
                                              min={0}
                                              max={1}
                                              step={0.1}
                                              value={getChunkOptionsForDoc(doc.id).threshold}
                                              onChange={e => setChunkOptionForDoc(doc.id, p => ({ ...p, threshold: parseFloat(e.target.value) || 0.6 }))}
                                            />
                                          </div>
                                          {promptsConfig && Object.keys(promptsConfig.prompts).length > 0 && (
                                            <>
                                              {Object.entries(promptsConfig.prompts).map(([name, versions]) => (
                                                <div key={name} className="chunk-option-row">
                                                  <label htmlFor={`chunk-prompt-${doc.id}-${name}`}>{name}</label>
                                                  <select
                                                    id={`chunk-prompt-${doc.id}-${name}`}
                                                    value={(getChunkOptionsForDoc(doc.id).promptVersions?.[name]) ?? promptsConfig.default[name] ?? (versions[0] ?? '')}
                                                    onChange={e => setChunkOptionForDoc(doc.id, p => ({
                                                      ...p,
                                                      promptVersions: {
                                                        ...(p.promptVersions ?? promptsConfig.default),
                                                        [name]: e.target.value,
                                                      },
                                                    }))}
                                                  >
                                                    {versions.map(v => (
                                                      <option key={v} value={v}>{v}</option>
                                                    ))}
                                                  </select>
                                                </div>
                                              ))}
                                            </>
                                          )}
                                        </div>
                                      )}
                                      {(doc.chunking_status === 'idle' || doc.chunking_status === null) && (
                                        <button
                                          type="button"
                                          className="dropdown-item"
                                          disabled={doc.extraction_status !== 'completed'}
                                          onClick={() => { onStartChunking(doc.id, getChunkOptionsForDoc(doc.id)); setOpenChunkMenuDocId(null) }}
                                        >
                                          Start chunking
                                        </button>
                                      )}
                                      {((doc.chunking_status === 'stopped' || doc.chunking_status === 'failed') && onRestartChunking) && (
                                        <button
                                          type="button"
                                          className="dropdown-item"
                                          disabled={doc.extraction_status !== 'completed'}
                                          title="Restart from last completed paragraph"
                                          onClick={() => { onRestartChunking(doc.id, getChunkOptionsForDoc(doc.id)); setOpenChunkMenuDocId(null) }}
                                        >
                                          Restart chunking
                                        </button>
                                      )}
                                    </>
                                  )}
                                </div>
                              )}
                            </div>
                          )
                        })()}
                      </div>
                    </td>
                    <td className="col-status col-embedding-with-menu">
                      <div className="embedding-cell">
                        {getStatusBadge(doc.embedding_status ?? null, 'chunking')}
                        {(() => {
                          const hasEmbeddingActions =
                            (doc.chunking_status === 'completed' && (doc.embedding_status === 'idle' || doc.embedding_status === null || doc.embedding_status === 'failed') && onStartEmbedding) ||
                            ((doc.embedding_status === 'pending' || doc.embedding_status === 'processing') && onResetEmbedding)
                          if (!hasEmbeddingActions) return null
                          return (
                            <div className="dropdown-wrap">
                              <button
                                type="button"
                                className="btn-icon btn-kebab"
                                onClick={(e) => { e.stopPropagation(); setOpenEmbeddingMenuDocId(prev => prev === doc.id ? null : doc.id) }}
                                title="Embedding actions"
                                aria-haspopup="true"
                                aria-expanded={openEmbeddingMenuDocId === doc.id}
                              >
                                ⋮
                              </button>
                              {openEmbeddingMenuDocId === doc.id && (
                                <div className="dropdown-menu" onClick={(e) => e.stopPropagation()}>
                                  {doc.chunking_status === 'completed' && (doc.embedding_status === 'idle' || doc.embedding_status === null || doc.embedding_status === 'failed') && onStartEmbedding && (
                                    <button
                                      type="button"
                                      className="dropdown-item"
                                      title="Queue embedding job for embedding worker"
                                      onClick={() => { onStartEmbedding(doc.id); setOpenEmbeddingMenuDocId(null) }}
                                    >
                                      Start embedding
                                    </button>
                                  )}
                                  {(doc.embedding_status === 'pending' || doc.embedding_status === 'processing') && onResetEmbedding && (
                                    <button
                                      type="button"
                                      className="dropdown-item dropdown-item-danger"
                                      title="Reset stuck embedding job (use when worker was killed)"
                                      onClick={() => { onResetEmbedding(doc.id); setOpenEmbeddingMenuDocId(null) }}
                                    >
                                      Reset embedding
                                    </button>
                                  )}
                                </div>
                              )}
                            </div>
                          )
                        })()}
                      </div>
                    </td>
                    <td className="col-status col-publish-with-menu">
                      <div className="publish-cell">
                        {doc.published_at ? (
                          <>
                            <span
                              className={`status-badge publish-badge ${doc.publish_verification_passed === false ? 'status-failed' : 'status-completed'}`}
                              title={
                                doc.publish_verification_passed === false && doc.publish_verification_message
                                  ? `${doc.published_rows ?? 0} rows. Verification failed: ${doc.publish_verification_message}`
                                  : doc.publish_verification_passed === true
                                    ? `${doc.published_rows ?? 0} rows at ${doc.published_at}. Integrity check passed.`
                                    : `${doc.published_rows ?? 0} rows at ${doc.published_at}`
                              }
                            >
                              Complete
                              {doc.publish_verification_passed === true && ' ✓'}
                              {doc.publish_verification_passed === false && ' ⚠'}
                            </span>
                          </>
                        ) : (
                          <span className="no-publish">—</span>
                        )}
                        <div className="dropdown-wrap">
                          <button
                            type="button"
                            className="btn-icon btn-kebab"
                            onClick={(e) => { e.stopPropagation(); setOpenPublishMenuDocId(prev => prev === doc.id ? null : doc.id) }}
                            title="Publish options"
                            aria-haspopup="true"
                            aria-expanded={openPublishMenuDocId === doc.id}
                          >
                            ⋮
                          </button>
                          {openPublishMenuDocId === doc.id && (
                            <div className="dropdown-menu dropdown-menu-publish" onClick={(e) => e.stopPropagation()}>
                              {onViewDocumentDetail && (
                                <button
                                  type="button"
                                  className="dropdown-item"
                                  onClick={() => { onViewDocumentDetail(doc.id); setOpenPublishMenuDocId(null) }}
                                  title="Open document detail (metadata, errors, facts, publish)"
                                >
                                  Details
                                </button>
                              )}
                              <button
                                type="button"
                                className="dropdown-item"
                                disabled={publishLoadingDocId === doc.id || (doc.chunking_status !== 'idle' && doc.chunking_status !== 'completed') || doc.embedding_status !== 'completed'}
                                title={doc.publish_count && doc.publish_count > 0 ? 'Replace published data with current embeddings' : 'Publish entire document to dbt-consumed table'}
                                onClick={() => { handlePublish(doc.id); setOpenPublishMenuDocId(null) }}
                              >
                                {publishLoadingDocId === doc.id ? 'Publishing…' : (doc.publish_count && doc.publish_count > 0 ? 'Republish' : 'Publish')}
                              </button>
                            </div>
                          )}
                        </div>
                      </div>
                    </td>
                    <td className="col-status col-publish-last">
                      {doc.publish_count === 0 ? (
                        <span className="no-publish">—</span>
                      ) : doc.publish_count === 1 ? (
                        <span className="status-badge status-completed" title="First publish">Published</span>
                      ) : (
                        <span className="status-badge status-completed" title={`Republished ${doc.publish_count} times`}>Republished</span>
                      )}
                    </td>
                    <td className="col-errors">
                      {doc.has_errors === 'true' || (doc.error_count && doc.error_count > 0) ? (
                        <span className="error-indicator" title={`${doc.error_count || 0} errors (${doc.critical_error_count || 0} critical)`}>
                          {doc.critical_error_count && doc.critical_error_count > 0 ? (
                            <span className="error-badge error-critical">
                              {doc.critical_error_count} Critical
                            </span>
                          ) : (
                            <span className="error-badge error-warning">
                              {doc.error_count} Errors
                            </span>
                          )}
                        </span>
                      ) : (
                        <span className="no-errors">—</span>
                      )}
                    </td>
                    <td className="col-date">
                      {new Date(doc.created_at).toLocaleString()}
                    </td>
                    <td className="col-actions">
                      <div className="action-buttons">
                        <button
                          type="button"
                          onClick={() => editingMetadataDocumentId === doc.id ? closeMetadataForm() : openMetadataForm(doc)}
                          className={`btn btn-sm btn-secondary ${isMetadataMissing(doc) ? 'metadata-btn-missing' : ''} ${editingMetadataDocumentId === doc.id ? 'metadata-btn-open' : ''}`}
                          title="Edit document metadata (payer, state, program, authority level)"
                        >
                          {editingMetadataDocumentId === doc.id ? 'Metadata ▲' : 'Metadata'}
                        </button>
                        <button
                          type="button"
                          onClick={() => onViewDocument(doc.id)}
                          className="btn btn-sm btn-secondary"
                        >
                          View
                        </button>
                        <div className="dropdown-wrap">
                          <button
                            type="button"
                            className="btn-icon btn-kebab"
                            onClick={(e) => { e.stopPropagation(); setOpenActionsMenuDocId(prev => prev === doc.id ? null : doc.id) }}
                            title="More actions"
                            aria-haspopup="true"
                            aria-expanded={openActionsMenuDocId === doc.id}
                          >
                            ⋮
                          </button>
                          {openActionsMenuDocId === doc.id && (
                            <div className="dropdown-menu dropdown-menu-right" onClick={(e) => e.stopPropagation()}>
                              <button
                                type="button"
                                className="dropdown-item dropdown-item-danger"
                                onClick={() => { handleDelete(doc.id); setOpenActionsMenuDocId(null) }}
                                title="Delete document"
                              >
                                Delete document
                              </button>
                            </div>
                          )}
                        </div>
                        {publishMessage?.docId === doc.id && (
                          <span className="publish-result-message" title={publishMessage.text}>
                            {publishMessage.text}
                          </span>
                        )}
                        {chunkStatusMessage?.docId === doc.id && (
                          <span className="publish-result-message" title={chunkStatusMessage.text}>
                            {chunkStatusMessage.text}
                          </span>
                        )}
                      </div>
                    </td>
                  </tr>
                  {editingMetadataDocumentId === doc.id && (
                    <tr key={`${doc.id}-meta`}>
                      <td colSpan={10} className="metadata-form-cell">
                        <div className="metadata-form-panel">
                          <div className="metadata-form-grid">
                            <div className="metadata-form-field metadata-form-field-full">
                              <label htmlFor={`meta-display-name-${doc.id}`}>Display name</label>
                              <input
                                id={`meta-display-name-${doc.id}`}
                                type="text"
                                value={metadataForm.display_name}
                                onChange={e => setMetadataForm(prev => ({ ...prev, display_name: e.target.value }))}
                                placeholder="User-friendly name (shown instead of filename when set)"
                              />
                            </div>
                            <div className="metadata-form-field">
                              <label htmlFor={`meta-payer-${doc.id}`}>Payer</label>
                              <input
                                id={`meta-payer-${doc.id}`}
                                type="text"
                                value={metadataForm.payer}
                                onChange={e => setMetadataForm(prev => ({ ...prev, payer: e.target.value }))}
                                placeholder="Payor name"
                              />
                            </div>
                            <div className="metadata-form-field">
                              <label htmlFor={`meta-state-${doc.id}`}>State</label>
                              <select
                                id={`meta-state-${doc.id}`}
                                value={metadataForm.state}
                                onChange={e => setMetadataForm(prev => ({ ...prev, state: e.target.value }))}
                              >
                                <option value="">—</option>
                                {STATE_OPTIONS.map(({ value, label }) => (
                                  <option key={value} value={value}>{label} ({value})</option>
                                ))}
                                {metadataForm.state && !STATE_OPTIONS.some(s => s.value === metadataForm.state) && (
                                  <option value={metadataForm.state}>{metadataForm.state} (other)</option>
                                )}
                              </select>
                            </div>
                            <div className="metadata-form-field">
                              <label htmlFor={`meta-program-${doc.id}`}>Program</label>
                              <input
                                id={`meta-program-${doc.id}`}
                                type="text"
                                value={metadataForm.program}
                                onChange={e => setMetadataForm(prev => ({ ...prev, program: e.target.value }))}
                                placeholder="Program"
                              />
                            </div>
                            <div className="metadata-form-field">
                              <label htmlFor={`meta-authority-${doc.id}`}>Authority level</label>
                              <select
                                id={`meta-authority-${doc.id}`}
                                value={metadataForm.authority_level}
                                onChange={e => setMetadataForm(prev => ({ ...prev, authority_level: e.target.value }))}
                              >
                                <option value="">—</option>
                                {AUTHORITY_LEVEL_OPTIONS.map(({ value, label }) => (
                                  <option key={value} value={value}>{label}</option>
                                ))}
                                {metadataForm.authority_level && !AUTHORITY_LEVEL_OPTIONS.some(a => a.value === metadataForm.authority_level) && (
                                  <option value={metadataForm.authority_level}>{metadataForm.authority_level} (other)</option>
                                )}
                              </select>
                            </div>
                            <div className="metadata-form-field">
                              <label htmlFor={`meta-effective-${doc.id}`}>Effective date</label>
                              <input
                                id={`meta-effective-${doc.id}`}
                                type="text"
                                value={metadataForm.effective_date}
                                onChange={e => setMetadataForm(prev => ({ ...prev, effective_date: e.target.value }))}
                                placeholder="e.g. 2024-01-15 or Jan 2024"
                              />
                            </div>
                            <div className="metadata-form-field">
                              <label htmlFor={`meta-termination-${doc.id}`}>Termination date</label>
                              <input
                                id={`meta-termination-${doc.id}`}
                                type="text"
                                value={metadataForm.termination_date}
                                onChange={e => setMetadataForm(prev => ({ ...prev, termination_date: e.target.value }))}
                                placeholder={`Default: ${defaultTerminationDate()} (6 months from today)`}
                              />
                            </div>
                          </div>
                          {metadataError && <div className="metadata-form-error">{metadataError}</div>}
                          <div className="metadata-form-actions">
                            <button
                              type="button"
                              className="btn btn-secondary"
                              onClick={closeMetadataForm}
                              disabled={metadataSaving}
                            >
                              Cancel
                            </button>
                            <button
                              type="button"
                              className="btn btn-primary"
                              onClick={() => handleSaveMetadata(doc.id, metadataForm)}
                              disabled={metadataSaving}
                            >
                              {metadataSaving ? 'Saving...' : 'Save'}
                            </button>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
