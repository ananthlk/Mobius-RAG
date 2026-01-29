import { useState, useEffect } from 'react'
import './ErrorReviewTab.css'

interface ProcessingError {
  id: string
  document_id: string
  paragraph_id: string | null
  error_type: string
  severity: 'critical' | 'warning' | 'info'
  error_message: string
  error_details: any
  stage: string
  resolved: boolean
  resolution: 'approved' | 'rejected' | 'reprocess' | null
  resolved_by: string | null
  resolved_at: string | null
  resolution_notes: string | null
  created_at: string
}

interface DocumentWithErrors {
  id: string
  filename: string
  error_count: number
  critical_error_count: number
  review_status: string
}

export function ErrorReviewTab() {
  const [errors, setErrors] = useState<ProcessingError[]>([])
  const [documents, setDocuments] = useState<DocumentWithErrors[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedError, setSelectedError] = useState<ProcessingError | null>(null)
  const [selectedDocument, setSelectedDocument] = useState<string | null>(null)
  
  // Filters
  const [filterSeverity, setFilterSeverity] = useState<string>('all')
  const [filterErrorType, setFilterErrorType] = useState<string>('all')
  const [filterResolved, setFilterResolved] = useState<string>('unresolved')
  const [filterDocument, setFilterDocument] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState('')

  const loadErrors = async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams()
      if (filterSeverity !== 'all') params.append('severity', filterSeverity)
      if (filterErrorType !== 'all') params.append('error_type', filterErrorType)
      if (filterResolved === 'resolved') params.append('resolved', 'true')
      else if (filterResolved === 'unresolved') params.append('resolved', 'false')
      if (filterDocument !== 'all') params.append('document_id', filterDocument)
      
      const response = await fetch(`http://localhost:8000/errors?${params.toString()}`)
      if (response.ok) {
        const data = await response.json()
        setErrors(data.errors || [])
      }
    } catch (err) {
      console.error('Failed to load errors:', err)
    } finally {
      setLoading(false)
    }
  }

  const loadDocumentsWithErrors = async () => {
    try {
      const response = await fetch('http://localhost:8000/documents')
      if (response.ok) {
        const data = await response.json()
        // Filter to only documents with errors
        const docsWithErrors = (data.documents || []).filter((doc: any) => 
          doc.has_errors === 'true' || doc.error_count > 0
        )
        setDocuments(docsWithErrors)
      }
    } catch (err) {
      console.error('Failed to load documents:', err)
    }
  }

  useEffect(() => {
    loadErrors()
    loadDocumentsWithErrors()
    // Auto-refresh every 10 seconds
    const interval = setInterval(() => {
      loadErrors()
      loadDocumentsWithErrors()
    }, 10000)
    return () => clearInterval(interval)
  }, [filterSeverity, filterErrorType, filterResolved, filterDocument])

  const handleResolve = async (errorId: string, resolution: 'approved' | 'rejected' | 'reprocess', notes: string = '') => {
    try {
      const response = await fetch(`http://localhost:8000/errors/${errorId}/resolve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          resolution,
          notes,
          resolved_by: 'dev_team' // In production, get from auth context
        })
      })
      if (response.ok) {
        await loadErrors()
        await loadDocumentsWithErrors()
        setSelectedError(null)
      } else {
        const errorData = await response.json().catch(() => null)
        alert(errorData?.detail || 'Failed to resolve error')
      }
    } catch (err) {
      alert('Failed to resolve error')
    }
  }

  const handleResolveAll = async (documentId: string, resolution: 'approved' | 'rejected' | 'reprocess', notes: string = '') => {
    if (!confirm(`Resolve all unresolved errors for this document as "${resolution}"?`)) return
    
    try {
      const response = await fetch(`http://localhost:8000/documents/${documentId}/errors/resolve-all`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          resolution,
          notes,
          resolved_by: 'dev_team'
        })
      })
      if (response.ok) {
        await loadErrors()
        await loadDocumentsWithErrors()
      } else {
        const errorData = await response.json().catch(() => null)
        alert(errorData?.detail || 'Failed to resolve errors')
      }
    } catch (err) {
      alert('Failed to resolve errors')
    }
  }

  const filteredErrors = errors.filter(err => {
    if (searchQuery && !err.error_message.toLowerCase().includes(searchQuery.toLowerCase())) {
      return false
    }
    return true
  })

  const errorTypes = Array.from(new Set(errors.map(e => e.error_type)))
  const severities: Array<'critical' | 'warning' | 'info'> = ['critical', 'warning', 'info']

  return (
    <div className="error-review-tab">
      <div className="error-review-header">
        <h2>Error Review</h2>
        <button onClick={loadErrors} className="btn btn-secondary" disabled={loading}>
          {loading ? 'Loading...' : 'Refresh'}
        </button>
      </div>

      <div className="error-review-layout">
        {/* Left Sidebar - Filters */}
        <div className="error-filters-sidebar">
          <h3>Filters</h3>
          
          <div className="filter-group">
            <label className="filter-label">Search</label>
            <input
              type="text"
              placeholder="Search error messages..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="filter-input"
            />
          </div>

          <div className="filter-group">
            <label className="filter-label">Severity</label>
            <select
              value={filterSeverity}
              onChange={(e) => setFilterSeverity(e.target.value)}
              className="filter-select"
            >
              <option value="all">All</option>
              {severities.map(sev => (
                <option key={sev} value={sev}>{sev.charAt(0).toUpperCase() + sev.slice(1)}</option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label className="filter-label">Error Type</label>
            <select
              value={filterErrorType}
              onChange={(e) => setFilterErrorType(e.target.value)}
              className="filter-select"
            >
              <option value="all">All</option>
              {errorTypes.map(type => (
                <option key={type} value={type}>{type.replace(/_/g, ' ')}</option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label className="filter-label">Resolution Status</label>
            <select
              value={filterResolved}
              onChange={(e) => setFilterResolved(e.target.value)}
              className="filter-select"
            >
              <option value="all">All</option>
              <option value="unresolved">Unresolved</option>
              <option value="resolved">Resolved</option>
            </select>
          </div>

          <div className="filter-group">
            <label className="filter-label">Document</label>
            <select
              value={filterDocument}
              onChange={(e) => setFilterDocument(e.target.value)}
              className="filter-select"
            >
              <option value="all">All Documents</option>
              {documents.map(doc => (
                <option key={doc.id} value={doc.id}>
                  {doc.filename} ({doc.error_count} errors)
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Main Area - Error List */}
        <div className="error-list-main">
          <div className="error-list-header">
            <h3>Errors ({filteredErrors.length})</h3>
            {selectedDocument && (
              <button
                onClick={() => {
                  const resolution = prompt('Enter resolution (approved/rejected/reprocess):')
                  if (resolution && ['approved', 'rejected', 'reprocess'].includes(resolution)) {
                    handleResolveAll(selectedDocument, resolution as 'approved' | 'rejected' | 'reprocess')
                  }
                }}
                className="btn btn-primary btn-sm"
              >
                Resolve All for Document
              </button>
            )}
          </div>

          {loading ? (
            <div className="loading-errors">Loading errors...</div>
          ) : filteredErrors.length === 0 ? (
            <div className="no-errors">No errors found matching filters</div>
          ) : (
            <div className="error-list">
              {filteredErrors.map((error) => (
                <div
                  key={error.id}
                  className={`error-item error-${error.severity} ${error.resolved ? 'resolved' : ''}`}
                  onClick={() => setSelectedError(error)}
                >
                  <div className="error-item-header">
                    <span className="error-type-badge">{error.error_type.replace(/_/g, ' ')}</span>
                    <span className={`severity-badge severity-${error.severity}`}>
                      {error.severity}
                    </span>
                    {error.resolved && (
                      <span className="resolved-badge">{error.resolution}</span>
                    )}
                  </div>
                  <div className="error-message">{error.error_message}</div>
                  <div className="error-meta">
                    <span>Document: {error.document_id.substring(0, 8)}...</span>
                    {error.paragraph_id && <span>Paragraph: {error.paragraph_id}</span>}
                    <span>Stage: {error.stage}</span>
                    <span>{new Date(error.created_at).toLocaleString()}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Error Detail Modal */}
      {selectedError && (
        <div className="error-modal-overlay" onClick={() => setSelectedError(null)}>
          <div className="error-modal" onClick={(e) => e.stopPropagation()}>
            <div className="error-modal-header">
              <h3>Error Details</h3>
              <button onClick={() => setSelectedError(null)} className="close-btn">×</button>
            </div>
            <div className="error-modal-content">
              <div className="error-detail-section">
                <label>Error Type:</label>
                <span className="error-type-badge">{selectedError.error_type.replace(/_/g, ' ')}</span>
              </div>
              
              <div className="error-detail-section">
                <label>Severity:</label>
                <span className={`severity-badge severity-${selectedError.severity}`}>
                  {selectedError.severity}
                </span>
              </div>
              
              <div className="error-detail-section">
                <label>Error Message:</label>
                <p>{selectedError.error_message}</p>
              </div>
              
              {selectedError.paragraph_id && (
                <div className="error-detail-section">
                  <label>Paragraph ID:</label>
                  <p>{selectedError.paragraph_id}</p>
                </div>
              )}
              
              <div className="error-detail-section">
                <label>Stage:</label>
                <p>{selectedError.stage}</p>
              </div>
              
              {selectedError.error_details && Object.keys(selectedError.error_details).length > 0 && (
                <div className="error-detail-section">
                  <label>Error Details:</label>
                  <pre className="error-details-json">
                    {JSON.stringify(selectedError.error_details, null, 2)}
                  </pre>
                </div>
              )}
              
              {selectedError.resolved && (
                <div className="error-detail-section">
                  <label>Resolution:</label>
                  <p>
                    <strong>{selectedError.resolution}</strong> by {selectedError.resolved_by} on{' '}
                    {selectedError.resolved_at ? new Date(selectedError.resolved_at).toLocaleString() : 'N/A'}
                  </p>
                  {selectedError.resolution_notes && (
                    <p className="resolution-notes">{selectedError.resolution_notes}</p>
                  )}
                </div>
              )}
              
              {!selectedError.resolved && (
                <div className="error-modal-actions">
                  <button
                    onClick={() => {
                      const notes = prompt('Resolution notes (optional):') || ''
                      handleResolve(selectedError.id, 'approved', notes)
                    }}
                    className="btn btn-success"
                  >
                    Approve
                  </button>
                  <button
                    onClick={() => {
                      const notes = prompt('Resolution notes (optional):') || ''
                      handleResolve(selectedError.id, 'rejected', notes)
                    }}
                    className="btn btn-danger"
                  >
                    Reject
                  </button>
                  <button
                    onClick={() => {
                      const notes = prompt('Resolution notes (optional):') || ''
                      handleResolve(selectedError.id, 'reprocess', notes)
                    }}
                    className="btn btn-primary"
                  >
                    Mark for Reprocess
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
