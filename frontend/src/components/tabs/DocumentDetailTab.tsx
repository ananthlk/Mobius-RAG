import React, { useState, useEffect, useCallback } from 'react'
import { API_BASE } from '../../config'
import './DocumentDetailTab.css'

interface DocumentDetailTabProps {
  documentId: string | null
  onViewDocument?: (documentId: string) => void
  onViewErrors?: () => void
  onViewFacts?: (documentId: string) => void
  onPublishSuccess?: () => void
}

interface DetailResponse {
  document: {
    id: string
    filename: string
    display_name: string
    payer: string
    state: string
    program: string
    authority_level: string
    effective_date: string
    termination_date: string
    status: string
    review_status: string
    created_at: string
    has_errors: string
    error_count: number
    critical_error_count: number
  }
  chunking_status: string
  embedding_status: string
  errors: { total: number; critical: number; unresolved: number }
  facts: { total: number; approved: number; pending: number; rejected: number }
  last_publish: {
    published_at: string
    published_by: string | null
    rows_written: number
    verification_passed?: boolean | null
    verification_message?: string | null
  } | null
  readiness: { chunking_done: boolean; embedding_done: boolean; no_critical_errors: boolean; ready: boolean }
}

export function DocumentDetailTab({
  documentId,
  onViewDocument,
  onViewErrors,
  onViewFacts,
  onPublishSuccess,
}: DocumentDetailTabProps) {
  const [detail, setDetail] = useState<DetailResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [publishLoading, setPublishLoading] = useState(false)
  const [publishMessage, setPublishMessage] = useState<string | null>(null)

  const fetchDetail = useCallback(async (id: string) => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/documents/${id}/detail`)
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        setError(data.detail || `Failed to load detail (${res.status})`)
        setDetail(null)
        return
      }
      const data: DetailResponse = await res.json()
      setDetail(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load detail')
      setDetail(null)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (documentId) {
      fetchDetail(documentId)
      setPublishMessage(null)
    } else {
      setDetail(null)
      setError(null)
    }
  }, [documentId, fetchDetail])

  const handlePublish = async () => {
    if (!documentId) return
    setPublishMessage(null)
    setPublishLoading(true)
    try {
      const res = await fetch(`${API_BASE}/documents/${documentId}/publish`, {
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
        setPublishMessage(msg)
        onPublishSuccess?.()
        fetchDetail(documentId)
      } else {
        setPublishMessage(data.detail || `Publish failed (${res.status})`)
      }
    } catch (e) {
      setPublishMessage(e instanceof Error ? e.message : 'Publish failed')
    } finally {
      setPublishLoading(false)
    }
  }

  if (!documentId) {
    return (
      <div className="document-detail-tab">
        <div className="empty-state">
          <p>No document selected.</p>
          <p>Go to <strong>Document Status</strong> and click <strong>Details</strong> on a document to view its metadata, errors, facts, chunks, and publish readiness here.</p>
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="document-detail-tab">
        <div className="empty-state">
          <p>Loading document details…</p>
        </div>
      </div>
    )
  }

  if (error || !detail) {
    return (
      <div className="document-detail-tab">
        <div className="empty-state">
          <p>{error || 'Failed to load detail'}</p>
        </div>
      </div>
    )
  }

  const doc = detail.document
  const displayName = doc.display_name?.trim() || doc.filename

  return (
    <div className="document-detail-tab">
      <div className="detail-header">
        <h2 className="detail-title">{displayName}</h2>
        <div className="detail-actions">
          {onViewDocument && (
            <button type="button" className="btn btn-sm btn-secondary" onClick={() => onViewDocument(doc.id)}>
              View document
            </button>
          )}
          <button
            type="button"
            className="btn btn-sm btn-primary"
            onClick={handlePublish}
            disabled={publishLoading || !detail.readiness.ready}
            title={detail.readiness.ready ? 'Publish entire document to dbt-consumed table' : 'Complete chunking and embedding with no critical errors first'}
          >
            {publishLoading ? 'Publishing…' : detail.last_publish ? 'Republish' : 'Publish'}
          </button>
        </div>
      </div>
      {publishMessage && (
        <p className="publish-message publish-result-message" style={{ marginTop: '0.5rem' }}>
          {publishMessage}
        </p>
      )}

      <section className="section">
        <h3 className="section-title">Metadata</h3>
        <div className="meta-grid">
          <div className="meta-item"><span className="meta-label">Filename</span><span className="meta-value">{doc.filename}</span></div>
          <div className="meta-item"><span className="meta-label">Payer</span><span className="meta-value">{doc.payer || '—'}</span></div>
          <div className="meta-item"><span className="meta-label">State</span><span className="meta-value">{doc.state || '—'}</span></div>
          <div className="meta-item"><span className="meta-label">Program</span><span className="meta-value">{doc.program || '—'}</span></div>
          <div className="meta-item"><span className="meta-label">Authority level</span><span className="meta-value">{doc.authority_level || '—'}</span></div>
          <div className="meta-item"><span className="meta-label">Effective date</span><span className="meta-value">{doc.effective_date || '—'}</span></div>
          <div className="meta-item"><span className="meta-label">Termination date</span><span className="meta-value">{doc.termination_date || '—'}</span></div>
          <div className="meta-item"><span className="meta-label">Store/MD status</span><span className="meta-value">{doc.status}</span></div>
          <div className="meta-item"><span className="meta-label">Review status</span><span className="meta-value">{doc.review_status}</span></div>
          <div className="meta-item"><span className="meta-label">Created</span><span className="meta-value">{new Date(doc.created_at).toLocaleString()}</span></div>
        </div>
      </section>

      <section className="section">
        <h3 className="section-title">Errors</h3>
        <div className="stats-row">
          <div className="stat-box"><span className="label">Total</span> <span className="value">{detail.errors.total}</span></div>
          <div className="stat-box errors-critical"><span className="label">Critical</span> <span className="value">{detail.errors.critical}</span></div>
          <div className="stat-box"><span className="label">Unresolved</span> <span className="value">{detail.errors.unresolved}</span></div>
        </div>
        {onViewErrors && (
          <button type="button" className="link-button" onClick={onViewErrors}>Review errors for this document</button>
        )}
      </section>

      <section className="section">
        <h3 className="section-title">Facts</h3>
        <div className="stats-row">
          <div className="stat-box"><span className="label">Total</span> <span className="value">{detail.facts.total}</span></div>
          <div className="stat-box facts-approved"><span className="label">Approved</span> <span className="value">{detail.facts.approved}</span></div>
          <div className="stat-box facts-pending"><span className="label">Pending</span> <span className="value">{detail.facts.pending}</span></div>
          <div className="stat-box"><span className="label">Rejected</span> <span className="value">{detail.facts.rejected}</span></div>
        </div>
        {onViewFacts && (
          <button type="button" className="link-button" onClick={() => onViewFacts(doc.id)}>Review facts for this document</button>
        )}
      </section>

      <section className="section">
        <h3 className="section-title">Chunking &amp; embedding</h3>
        <div className="meta-grid">
          <div className="meta-item"><span className="meta-label">Chunking</span><span className="meta-value">{detail.chunking_status}</span></div>
          <div className="meta-item"><span className="meta-label">Embedding</span><span className="meta-value">{detail.embedding_status}</span></div>
        </div>
      </section>

      <section className="section">
        <h3 className="section-title">Publish readiness</h3>
        <div className="stats-row">
          <span className={`readiness-badge ${detail.readiness.ready ? 'ready' : 'not-ready'}`}>
            {detail.readiness.ready ? 'Ready to publish' : 'Not ready'}
          </span>
          {!detail.readiness.ready && (
            <span className="meta-value" style={{ fontSize: '0.875rem' }}>
              {!detail.readiness.chunking_done && 'Chunking not done. '}
              {!detail.readiness.embedding_done && 'Embedding not done. '}
              {!detail.readiness.no_critical_errors && 'Critical errors present.'}
            </span>
          )}
        </div>
        {detail.last_publish && (
          <p className="publish-message" style={{ marginTop: '0.75rem' }}>
            Last published: {new Date(detail.last_publish.published_at).toLocaleString()}
            {detail.last_publish.published_by && ` by ${detail.last_publish.published_by}`}
            {' '}({detail.last_publish.rows_written} rows)
            {detail.last_publish.verification_passed === true && ' ✓ Integrity verified'}
            {detail.last_publish.verification_passed === false && (
              <span title={detail.last_publish.verification_message ?? undefined}>
                {' '}⚠ Verification failed{detail.last_publish.verification_message ? `: ${detail.last_publish.verification_message}` : ''}
              </span>
            )}
          </p>
        )}
      </section>
    </div>
  )
}
