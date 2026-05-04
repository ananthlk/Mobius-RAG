import { useState } from 'react'
import { API_BASE } from '../../../config'

interface DocLike {
  id: string
  filename: string
  display_name?: string | null
  payer?: string | null
  state?: string | null
  program?: string | null
  extraction_status?: string
  chunking_status?: string | null
  embedding_status?: string | null
  published_at?: string | null
}

interface Props {
  docs: DocLike[]
  selectedDocumentId?: string | null
  onSelectDoc: (documentId: string) => void
  onRefresh: () => void
}

function docStatus(d: DocLike) {
  if (d.published_at) return 'published'
  if (d.embedding_status === 'failed' || d.chunking_status === 'failed') return 'failed'
  if (d.embedding_status === 'completed') return 'embedded'
  if (d.chunking_status === 'completed') return 'chunked'
  if (d.chunking_status === 'in_progress') return 'chunking…'
  if (d.embedding_status === 'in_progress') return 'embedding…'
  if (d.extraction_status === 'completed') return 'extracted'
  return d.extraction_status || 'pending'
}

function canRetrigger(d: DocLike) {
  const s = docStatus(d)
  return s === 'failed' || s === 'extracted' || s === 'pending' || s === 'embedded'
}

function nextAction(d: DocLike): 'chunk' | 'embed' | 'publish' | null {
  if (d.published_at) return null
  if (d.embedding_status === 'completed') return 'publish'
  if (d.chunking_status === 'completed') return 'embed'
  return 'chunk'
}

interface EditState {
  payer: string
  state: string
  program: string
}

export function UploadedDocsPanel({ docs, selectedDocumentId, onSelectDoc, onRefresh }: Props) {
  const [editing, setEditing] = useState<string | null>(null)
  const [editValues, setEditValues] = useState<EditState>({ payer: '', state: '', program: '' })
  const [saving, setSaving] = useState(false)
  const [triggering, setTriggering] = useState<string | null>(null)
  const [bulkTriggering, setBulkTriggering] = useState(false)

  const startEdit = (d: DocLike) => {
    setEditing(d.id)
    setEditValues({
      payer: d.payer ?? '',
      state: d.state ?? '',
      program: d.program ?? '',
    })
  }

  const saveEdit = async (docId: string) => {
    setSaving(true)
    try {
      await fetch(`${API_BASE}/documents/${docId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          payer: editValues.payer || null,
          state: editValues.state || null,
          program: editValues.program || null,
        }),
      })
      setEditing(null)
      onRefresh()
    } finally {
      setSaving(false)
    }
  }

  const retrigger = async (d: DocLike) => {
    const action = nextAction(d)
    if (!action) return
    setTriggering(d.id)
    try {
      const url = action === 'chunk'
        ? `${API_BASE}/documents/${d.id}/chunking/start`
        : action === 'embed'
          ? `${API_BASE}/documents/${d.id}/embedding/start`
          : `${API_BASE}/documents/${d.id}/publish`
      await fetch(url, { method: 'POST' })
      onRefresh()
    } finally {
      setTriggering(null)
    }
  }

  const retriggerAll = async () => {
    setBulkTriggering(true)
    try {
      const stuck = docs.filter(canRetrigger)
      for (const d of stuck) {
        const action = nextAction(d)
        if (!action) continue
        const url = action === 'chunk'
          ? `${API_BASE}/documents/${d.id}/chunking/start`
          : action === 'embed'
            ? `${API_BASE}/documents/${d.id}/embedding/start`
            : `${API_BASE}/documents/${d.id}/publish`
        await fetch(url, { method: 'POST' })
      }
      onRefresh()
    } finally {
      setBulkTriggering(false)
    }
  }

  const stuckCount = docs.filter(canRetrigger).length

  return (
    <div style={{ padding: '12px 16px', minHeight: 200, minWidth: 0, overflow: 'hidden' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
        <h3 style={{ margin: 0, fontSize: 14, fontWeight: 600 }}>
          Uploaded Documents ({docs.length})
        </h3>
        {stuckCount > 0 && (
          <button
            onClick={retriggerAll}
            disabled={bulkTriggering}
            style={{
              marginLeft: 'auto',
              fontSize: 12,
              padding: '3px 10px',
              background: '#f59e0b',
              color: '#fff',
              border: 'none',
              borderRadius: 4,
              cursor: bulkTriggering ? 'not-allowed' : 'pointer',
              opacity: bulkTriggering ? 0.6 : 1,
            }}
          >
            {bulkTriggering ? 'Retriggering…' : `Retrigger ${stuckCount} stuck`}
          </button>
        )}
      </div>

      {docs.length === 0 && (
        <p style={{ color: '#888', fontSize: 13 }}>No uploaded documents found.</p>
      )}

      <div style={{ display: 'grid', gap: 6, minWidth: 0 }}>
        {docs.map((d) => {
          const status = docStatus(d)
          const isSelected = selectedDocumentId === d.id
          const isEditing = editing === d.id
          const name = (d.display_name && d.display_name.trim()) || d.filename

          const statusColor = status === 'published' ? '#10b981'
            : status === 'failed' ? '#ef4444'
            : status === 'embedded' ? '#3b82f6'
            : status === 'chunked' ? '#8b5cf6'
            : '#f59e0b'

          return (
            <div
              key={d.id}
              style={{
                border: `1px solid ${isSelected ? '#3b82f6' : '#e5e7eb'}`,
                borderRadius: 6,
                padding: '8px 10px',
                background: isSelected ? '#eff6ff' : '#fff',
                fontSize: 13,
                minWidth: 0,
                overflow: 'hidden',
              }}
            >
              {/* Header row */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: isEditing ? 8 : 0, minWidth: 0 }}>
                <button
                  onClick={() => onSelectDoc(d.id)}
                  style={{
                    flex: 1, textAlign: 'left', background: 'none', border: 'none',
                    cursor: 'pointer', fontWeight: 500, fontSize: 13,
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}
                  title={name}
                >
                  {name}
                </button>
                <span style={{
                  fontSize: 11, padding: '1px 6px', borderRadius: 10,
                  background: statusColor + '22', color: statusColor, whiteSpace: 'nowrap',
                }}>
                  {status}
                </span>
                {!isEditing && (
                  <button
                    onClick={() => startEdit(d)}
                    title="Edit metadata"
                    style={{
                      background: 'none', border: '1px solid #e5e7eb', borderRadius: 4,
                      padding: '2px 6px', fontSize: 11, cursor: 'pointer', color: '#666',
                    }}
                  >
                    ✏ metadata
                  </button>
                )}
                {canRetrigger(d) && !isEditing && (
                  <button
                    onClick={() => retrigger(d)}
                    disabled={triggering === d.id}
                    title={`Retrigger: ${nextAction(d)}`}
                    style={{
                      background: '#f59e0b', border: 'none', borderRadius: 4,
                      padding: '2px 8px', fontSize: 11, cursor: 'pointer', color: '#fff',
                      opacity: triggering === d.id ? 0.6 : 1,
                    }}
                  >
                    {triggering === d.id ? '…' : '↺ retrigger'}
                  </button>
                )}
              </div>

              {/* Metadata pills (when not editing) */}
              {!isEditing && (d.payer || d.state || d.program) && (
                <div style={{ display: 'flex', gap: 6, marginTop: 4, flexWrap: 'wrap' }}>
                  {d.payer && (
                    <span style={{ fontSize: 11, color: '#6366f1', background: '#eef2ff', padding: '1px 6px', borderRadius: 10 }}>
                      {d.payer}
                    </span>
                  )}
                  {d.state && (
                    <span style={{ fontSize: 11, color: '#0891b2', background: '#e0f2fe', padding: '1px 6px', borderRadius: 10 }}>
                      {d.state}
                    </span>
                  )}
                  {d.program && (
                    <span style={{ fontSize: 11, color: '#059669', background: '#d1fae5', padding: '1px 6px', borderRadius: 10 }}>
                      {d.program}
                    </span>
                  )}
                </div>
              )}

              {/* Missing metadata warning */}
              {!isEditing && !d.payer && (
                <div style={{ fontSize: 11, color: '#f59e0b', marginTop: 2 }}>
                  ⚠ No payer set — won't be filtered correctly in search
                </div>
              )}

              {/* Inline metadata editor */}
              {isEditing && (
                <div style={{ display: 'grid', gap: 6 }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6 }}>
                    <label style={{ fontSize: 11 }}>
                      <div style={{ color: '#888', marginBottom: 2 }}>Payer</div>
                      <input
                        value={editValues.payer}
                        onChange={(e) => setEditValues((v) => ({ ...v, payer: e.target.value }))}
                        placeholder="e.g. Aetna Better Health"
                        style={{
                          width: '100%', fontSize: 12, padding: '4px 6px',
                          border: '1px solid #d1d5db', borderRadius: 4, boxSizing: 'border-box',
                        }}
                      />
                    </label>
                    <label style={{ fontSize: 11 }}>
                      <div style={{ color: '#888', marginBottom: 2 }}>State</div>
                      <input
                        value={editValues.state}
                        onChange={(e) => setEditValues((v) => ({ ...v, state: e.target.value }))}
                        placeholder="e.g. FL"
                        style={{
                          width: '100%', fontSize: 12, padding: '4px 6px',
                          border: '1px solid #d1d5db', borderRadius: 4, boxSizing: 'border-box',
                        }}
                      />
                    </label>
                    <label style={{ fontSize: 11 }}>
                      <div style={{ color: '#888', marginBottom: 2 }}>Program</div>
                      <input
                        value={editValues.program}
                        onChange={(e) => setEditValues((v) => ({ ...v, program: e.target.value }))}
                        placeholder="e.g. Medicaid"
                        style={{
                          width: '100%', fontSize: 12, padding: '4px 6px',
                          border: '1px solid #d1d5db', borderRadius: 4, boxSizing: 'border-box',
                        }}
                      />
                    </label>
                  </div>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button
                      onClick={() => saveEdit(d.id)}
                      disabled={saving}
                      style={{
                        fontSize: 12, padding: '3px 10px', background: '#3b82f6',
                        color: '#fff', border: 'none', borderRadius: 4, cursor: 'pointer',
                        opacity: saving ? 0.6 : 1,
                      }}
                    >
                      {saving ? 'Saving…' : 'Save'}
                    </button>
                    <button
                      onClick={() => setEditing(null)}
                      style={{
                        fontSize: 12, padding: '3px 10px', background: 'none',
                        color: '#666', border: '1px solid #e5e7eb', borderRadius: 4, cursor: 'pointer',
                      }}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
