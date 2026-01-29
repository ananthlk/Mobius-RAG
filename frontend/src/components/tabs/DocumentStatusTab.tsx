import { useState, useEffect } from 'react'
import './DocumentStatusTab.css'

interface Document {
  id: string
  filename: string
  extraction_status: string
  chunking_status: string | null
  created_at: string
  gcs_path: string
  has_errors?: string  // 'true' or 'false'
  error_count?: number
  critical_error_count?: number
  review_status?: string
}

interface DocumentStatusTabProps {
  onStartChunking: (documentId: string) => Promise<void>
  onStopChunking: (documentId: string) => Promise<void>
  onViewDocument: (documentId: string) => void
  onDeleteDocument: (documentId: string) => Promise<void>
  onRestartChunking?: (documentId: string) => Promise<void>
}

export function DocumentStatusTab({ 
  onStartChunking, 
  onStopChunking,
  onViewDocument,
  onDeleteDocument,
  onRestartChunking
}: DocumentStatusTabProps) {
  const [documents, setDocuments] = useState<Document[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedDocs, setSelectedDocs] = useState<Set<string>>(new Set())
  const [searchQuery, setSearchQuery] = useState('')
  const [sortColumn, setSortColumn] = useState<string>('created_at')
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc')

  const loadDocuments = async () => {
    setLoading(true)
    try {
      const response = await fetch('http://localhost:8000/documents')
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
    for (const docId of selectedDocs) {
      try {
        await onStartChunking(docId)
      } catch (err) {
        console.error(`Failed to start chunking for ${docId}:`, err)
      }
    }
    setSelectedDocs(new Set())
    await loadDocuments()
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

  const getStatusBadge = (status: string | null, type: 'extraction' | 'chunking') => {
    if (!status) return <span className="status-badge status-idle">—</span>
    
    const statusClass = `status-badge status-${status}`
    const statusLabel = status.charAt(0).toUpperCase() + status.slice(1).replace('_', ' ')
    
    return <span className={statusClass}>{statusLabel}</span>
  }

  const filteredDocuments = documents.filter(doc =>
    doc.filename.toLowerCase().includes(searchQuery.toLowerCase())
  )

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
      <p className="pipeline-copy" title="Upload stores file; Store = raw text per page; Convert to MD = canonical markdown per page; Chunk runs on markdown.">
        Pipeline: Upload → Store → Convert to MD → Chunk
      </p>
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
                <td colSpan={7} className="empty-state">
                  {loading ? 'Loading documents...' : 'No documents found'}
                </td>
              </tr>
            ) : (
              sortedDocuments.map((doc) => (
                <tr key={doc.id}>
                  <td className="col-checkbox">
                    <input
                      type="checkbox"
                      checked={selectedDocs.has(doc.id)}
                      onChange={() => handleSelectDoc(doc.id)}
                    />
                  </td>
                  <td className="col-name">{doc.filename}</td>
                  <td className="col-status">
                    {getStatusBadge(doc.extraction_status, 'extraction')}
                  </td>
                  <td className="col-status">
                    {getStatusBadge(doc.chunking_status, 'chunking')}
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
                      {doc.chunking_status === 'idle' || doc.chunking_status === null ? (
                        <button
                          onClick={() => onStartChunking(doc.id)}
                          className="btn btn-sm btn-primary"
                          disabled={doc.extraction_status !== 'completed'}
                        >
                          Start
                        </button>
                      ) : doc.chunking_status === 'in_progress' ? (
                        <button
                          onClick={() => onStopChunking(doc.id)}
                          className="btn btn-sm btn-danger"
                        >
                          Stop
                        </button>
                      ) : (doc.chunking_status === 'stopped' || doc.chunking_status === 'failed') && onRestartChunking ? (
                        <button
                          onClick={() => onRestartChunking(doc.id)}
                          className="btn btn-sm btn-primary"
                          disabled={doc.extraction_status !== 'completed'}
                          title="Restart from last completed paragraph"
                        >
                          Restart
                        </button>
                      ) : null}
                      <button
                        onClick={() => onViewDocument(doc.id)}
                        className="btn btn-sm btn-secondary"
                      >
                        View
                      </button>
                      <button
                        onClick={() => handleDelete(doc.id)}
                        className="btn btn-sm btn-danger"
                        title="Delete document"
                      >
                        Delete
                      </button>
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
