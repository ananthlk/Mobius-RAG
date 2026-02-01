import { useState, useEffect } from 'react'
import './PromptsModal.css'

import { API_BASE } from '../config'

function formatApiError(detail: unknown): string {
  if (detail == null) return 'Unknown error'
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail.map((e: { msg?: string; loc?: unknown }) => e?.msg ?? JSON.stringify(e)).join('; ')
  }
  return typeof (detail as { detail?: string }).detail === 'string'
    ? (detail as { detail: string }).detail
    : JSON.stringify(detail)
}

export interface PromptMeta {
  body: string
  variables: string[]
  description: string
  version: string
}

interface PromptsModalProps {
  open: boolean
  onClose: () => void
}

export function PromptsModal({ open, onClose }: PromptsModalProps) {
  const [promptNames, setPromptNames] = useState<string[]>([])
  const [promptsByName, setPromptsByName] = useState<Record<string, string[]>>({})
  const [selectedName, setSelectedName] = useState<string>('')
  const [selectedVersion, setSelectedVersion] = useState<string>('')
  const [meta, setMeta] = useState<PromptMeta | null>(null)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const [isNewVersion, setIsNewVersion] = useState(false)
  const [newVersionId, setNewVersionId] = useState('')
  const [newPromptName, setNewPromptName] = useState('')
  const [showNewPromptName, setShowNewPromptName] = useState(false)

  const loadList = async () => {
    try {
      const res = await fetch(`${API_BASE}/config/prompts`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      const names = data.names ?? Object.keys(data.prompts ?? {})
      setPromptNames(Array.isArray(names) ? names : [])
      setPromptsByName(data.prompts ?? {})
      if (!selectedName && names?.length) setSelectedName(names[0])
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load prompts')
    }
  }

  useEffect(() => {
    if (!open) return
    setError(null)
    setMessage(null)
    loadList()
  }, [open])

  useEffect(() => {
    if (!open || !selectedName) {
      setMeta(null)
      setSelectedVersion('')
      setIsNewVersion(false)
      return
    }
    const versions = promptsByName[selectedName] ?? []
    if (!selectedVersion && versions.length && !isNewVersion) setSelectedVersion(versions[0])
    if (!selectedVersion || (isNewVersion && selectedVersion === '__new__')) {
      if (!isNewVersion) setMeta(null)
      return
    }
    let cancelled = false
    setLoading(true)
    setError(null)
    fetch(`${API_BASE}/config/prompts/${encodeURIComponent(selectedName)}/${encodeURIComponent(selectedVersion)}`)
      .then((r) => {
        if (!r.ok) throw new Error(r.status === 404 ? 'Not found' : `HTTP ${r.status}`)
        return r.json()
      })
      .then((data) => {
        if (!cancelled) {
          setMeta({
            body: data.body ?? '',
            variables: Array.isArray(data.variables) ? data.variables : [],
            description: data.description ?? '',
            version: data.version ?? selectedVersion,
          })
        }
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Failed to load prompt')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => { cancelled = true }
  }, [open, selectedName, selectedVersion, isNewVersion])

  const versions = selectedName ? (promptsByName[selectedName] ?? []) : []

  const updateMeta = (updates: Partial<PromptMeta>) => {
    setMeta((prev) => (prev ? { ...prev, ...updates } : null))
  }

  const handleSave = async () => {
    const name = selectedName
    const version = isNewVersion ? newVersionId.trim() : selectedVersion
    if (!name || !version) {
      setError('Name and version are required')
      return
    }
    const bodyText = meta?.body ?? ''
    const description = meta?.description ?? ''
    const variables = meta?.variables ?? []
    setSaving(true)
    setError(null)
    setMessage(null)
    try {
      const url = `${API_BASE}/config/prompts/${encodeURIComponent(name)}/${encodeURIComponent(version)}`
      const method = isNewVersion ? 'POST' : 'PUT'
      const res = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ body: bodyText, description, variables }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        throw new Error(formatApiError(data.detail ?? data))
      }
      setMessage(
        isNewVersion
          ? `Created new version: ${name} / ${version}`
          : `Updated ${name} / ${version}`
      )
      setIsNewVersion(false)
      setNewVersionId('')
      await loadList()
      setSelectedVersion(version)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save')
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async () => {
    if (!selectedName || !selectedVersion || isNewVersion) return
    if (!confirm(`Delete "${selectedName}" / "${selectedVersion}"?`)) return
    setSaving(true)
    setError(null)
    setMessage(null)
    try {
      const res = await fetch(
        `${API_BASE}/config/prompts/${encodeURIComponent(selectedName)}/${encodeURIComponent(selectedVersion)}`,
        { method: 'DELETE' }
      )
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        throw new Error(formatApiError(data.detail ?? data))
      }
      setMessage('Deleted.')
      setMeta(null)
      setSelectedVersion('')
      await loadList()
      const remaining = (promptsByName[selectedName] ?? []).filter((v) => v !== selectedVersion)
      if (remaining.length) setSelectedVersion(remaining[0])
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to delete')
    } finally {
      setSaving(false)
    }
  }

  const handleCreateNewVersion = () => {
    setError(null)
    setMessage(null)
    setIsNewVersion(true)
    setSelectedVersion('__new__')
    setNewVersionId('')
    setMeta({ body: '', variables: [], description: '', version: '' })
  }

  const handleCreateNewPromptName = async () => {
    const name = newPromptName.trim()
    if (!name) return
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/config/prompts/names`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        throw new Error(formatApiError(data.detail ?? data))
      }
      setShowNewPromptName(false)
      setNewPromptName('')
      await loadList()
      setSelectedName(name)
      setSelectedVersion('')
      setMessage('Prompt name created. Add a version below.')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to create prompt name')
    }
  }

  const variablesStr = (meta?.variables ?? []).join(', ')
  const saveButtonLabel = isNewVersion ? 'Create new version' : 'Update version'
  const saveSuccessHint = isNewVersion
    ? `Creating new version for ${selectedName} (e.g. v2)`
    : `Editing ${selectedName} / ${selectedVersion}`
  const setVariablesStr = (s: string) => {
    const list = s.split(',').map((v) => v.trim()).filter(Boolean)
    updateMeta({ variables: list })
  }

  if (!open) return null

  return (
    <div className="prompts-modal-backdrop" onClick={onClose} aria-hidden>
      <div className="prompts-modal" onClick={(e) => e.stopPropagation()}>
        <div className="prompts-modal-header">
          <h2 className="prompts-modal-title">Manage prompts</h2>
          <button type="button" className="prompts-modal-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>
        <div className="prompts-modal-body">
          {error && (
            <div className="prompts-modal-error" role="alert">
              {error}
            </div>
          )}
          {message && (
            <div className="prompts-modal-message">{message}</div>
          )}

          <div className="prompts-modal-layout">
            <div className="prompts-modal-sidebar">
              <div className="prompts-modal-field">
                <label htmlFor="prompts-name">Prompt name</label>
                <div className="prompts-modal-name-row">
                  <select
                    id="prompts-name"
                    value={selectedName}
                    onChange={(e) => {
                      setSelectedName(e.target.value)
                      setSelectedVersion('')
                      setIsNewVersion(false)
                    }}
                  >
                    {promptNames.map((n) => (
                      <option key={n} value={n}>{n}</option>
                    ))}
                  </select>
                  <button
                    type="button"
                    className="prompts-modal-btn-small"
                    onClick={() => setShowNewPromptName(true)}
                    title="Create new prompt name"
                  >
                    + Name
                  </button>
                </div>
              </div>
              {showNewPromptName && (
                <div className="prompts-modal-new-name">
                  <input
                    type="text"
                    value={newPromptName}
                    onChange={(e) => setNewPromptName(e.target.value)}
                    placeholder="New prompt name"
                    onKeyDown={(e) => e.key === 'Enter' && handleCreateNewPromptName()}
                  />
                  <button type="button" className="prompts-modal-btn-small" onClick={handleCreateNewPromptName}>
                    Create
                  </button>
                  <button type="button" className="prompts-modal-btn-small" onClick={() => { setShowNewPromptName(false); setNewPromptName('') }}>
                    Cancel
                  </button>
                </div>
              )}

              <div className="prompts-modal-field">
                <label>Version</label>
                <div className="prompts-modal-version-list">
                  {versions.map((v) => (
                    <button
                      key={v}
                      type="button"
                      className={`prompts-modal-version-btn ${selectedVersion === v ? 'active' : ''}`}
                      onClick={() => { setSelectedVersion(v); setIsNewVersion(false) }}
                    >
                      {v}
                    </button>
                  ))}
                  <button
                    type="button"
                    className={`prompts-modal-version-btn new ${isNewVersion ? 'active' : ''}`}
                    onClick={handleCreateNewVersion}
                  >
                    + New version
                  </button>
                </div>
              </div>
            </div>

            <div className="prompts-modal-editor">
              {selectedName && versions.length === 0 && !isNewVersion && (
                <div className="prompts-modal-empty">
                  <p>No versions yet.</p>
                  <button type="button" className="prompts-modal-save" onClick={handleCreateNewVersion}>
                    Add first version
                  </button>
                </div>
              )}
              {selectedName && (selectedVersion || isNewVersion) && (versions.length > 0 || isNewVersion) && (
                <>
                  <p className="prompts-modal-context-hint" aria-live="polite">
                    {saveSuccessHint}
                  </p>
                  {isNewVersion && (
                    <div className="prompts-modal-field">
                      <label htmlFor="prompts-new-version-id">New version id</label>
                      <input
                        id="prompts-new-version-id"
                        type="text"
                        value={newVersionId}
                        onChange={(e) => setNewVersionId(e.target.value)}
                        placeholder="e.g. v2"
                      />
                    </div>
                  )}
                  {loading && !meta && !isNewVersion && <p className="prompts-modal-loading">Loading…</p>}
                  {(meta || isNewVersion) && (
                    <>
                      <div className="prompts-modal-field">
                        <label htmlFor="prompts-desc">Description</label>
                        <input
                          id="prompts-desc"
                          type="text"
                          value={meta?.description ?? ''}
                          onChange={(e) => updateMeta({ description: e.target.value })}
                          placeholder="Short description"
                        />
                      </div>
                      <div className="prompts-modal-field">
                        <label htmlFor="prompts-vars">Variables (comma-separated)</label>
                        <input
                          id="prompts-vars"
                          type="text"
                          value={variablesStr}
                          onChange={(e) => setVariablesStr(e.target.value)}
                          placeholder="e.g. paragraph_block, context"
                        />
                      </div>
                      <div className="prompts-modal-field">
                        <label htmlFor="prompts-body">Body</label>
                        <textarea
                          id="prompts-body"
                          className="prompts-modal-textarea"
                          value={meta?.body ?? ''}
                          onChange={(e) => updateMeta({ body: e.target.value })}
                          placeholder="Prompt template body…"
                          rows={14}
                        />
                      </div>
                      <div className="prompts-modal-actions">
                        <button
                          type="button"
                          className="prompts-modal-save"
                          onClick={handleSave}
                          disabled={saving || (isNewVersion && !newVersionId.trim())}
                        >
                          {saving ? 'Saving…' : saveButtonLabel}
                        </button>
                        {!isNewVersion && selectedVersion && (
                          <button
                            type="button"
                            className="prompts-modal-delete"
                            onClick={handleDelete}
                            disabled={saving}
                          >
                            Delete
                          </button>
                        )}
                      </div>
                    </>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
