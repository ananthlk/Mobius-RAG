import { useState, useEffect } from 'react'
import './LLMConfigModal.css'

const API_BASE = 'http://localhost:8000'

export interface LLMConfigShape {
  provider?: string
  model?: string
  version?: string
  options?: Record<string, unknown>
  ollama?: { base_url?: string }
  vertex?: { project_id?: string; location?: string }
  openai?: { api_key?: string; base_url?: string }
}

interface LLMConfigModalProps {
  open: boolean
  onClose: () => void
}

export function LLMConfigModal({ open, onClose }: LLMConfigModalProps) {
  const [configNames, setConfigNames] = useState<string[]>([])
  const [providers, setProviders] = useState<string[]>([])
  const [selectedConfig, setSelectedConfig] = useState<string>('')
  const [config, setConfig] = useState<LLMConfigShape | null>(null)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [testLoading, setTestLoading] = useState(false)
  const [testResult, setTestResult] = useState<{ ok: boolean; message?: string; error?: string } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  useEffect(() => {
    if (!open) return
    setError(null)
    setMessage(null)
    setTestResult(null)
    const load = async () => {
      setLoading(true)
      try {
        const [llmRes, provRes] = await Promise.all([
          fetch(`${API_BASE}/config/llm`),
          fetch(`${API_BASE}/config/llm/providers`),
        ])
        if (llmRes.ok) {
          const data = await llmRes.json()
          const names = (data.configs || []).filter((n: string) => n && !n.endsWith('.example'))
          setConfigNames(names.length ? names : ['default'])
          if (!selectedConfig && names.length) setSelectedConfig(names[0])
        }
        if (provRes.ok) {
          const data = await provRes.json()
          setProviders(data.providers || [])
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load config')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [open])

  useEffect(() => {
    if (!open || !selectedConfig) {
      setConfig(null)
      return
    }
    let cancelled = false
    setError(null)
    setLoading(true)
    fetch(`${API_BASE}/config/llm/${encodeURIComponent(selectedConfig)}`)
      .then((r) => {
        if (!r.ok) throw new Error(r.status === 404 ? 'Config not found' : `HTTP ${r.status}`)
        return r.json()
      })
      .then((data) => {
        if (!cancelled) {
          setConfig({
            provider: data.provider ?? '',
            model: data.model ?? '',
            version: data.version ?? selectedConfig,
            options: data.options ?? {},
            ollama: data.ollama ?? {},
            vertex: data.vertex ?? {},
            openai: data.openai ?? {},
          })
        }
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Failed to load config')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [open, selectedConfig])

  const updateConfig = (updates: Partial<LLMConfigShape>) => {
    setConfig((prev) => (prev ? { ...prev, ...updates } : null))
  }

  const updateNested = (key: 'ollama' | 'vertex' | 'openai', field: string, value: string) => {
    setConfig((prev) => {
      if (!prev) return prev
      const block = prev[key] ?? {}
      return { ...prev, [key]: { ...block, [field]: value } }
    })
  }

  const updateOption = (key: string, value: string | number) => {
    setConfig((prev) => {
      if (!prev) return prev
      const opts = prev.options ?? {}
      const num = typeof value === 'string' ? parseFloat(value) : value
      const final = (typeof value === 'string' && Number.isNaN(num)) ? value : num
      return { ...prev, options: { ...opts, [key]: final } }
    })
  }

  const handleSave = async () => {
    if (!selectedConfig || !config) return
    setSaving(true)
    setError(null)
    setMessage(null)
    try {
      const body = {
        provider: config.provider || undefined,
        model: config.model || undefined,
        version: config.version || undefined,
        options: config.options && Object.keys(config.options).length ? config.options : undefined,
        ollama: config.ollama && Object.keys(config.ollama).length ? config.ollama : undefined,
        vertex: config.vertex && Object.keys(config.vertex).length ? config.vertex : undefined,
        openai: config.openai && Object.keys(config.openai).length ? config.openai : undefined,
      }
      const res = await fetch(`${API_BASE}/config/llm/${encodeURIComponent(selectedConfig)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || `Save failed: ${res.status}`)
      }
      setMessage('Saved.')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save')
    } finally {
      setSaving(false)
    }
  }

  const handleTest = async () => {
    if (!selectedConfig) return
    setTestLoading(true)
    setTestResult(null)
    setError(null)
    try {
      const url = `${API_BASE}/config/llm/${encodeURIComponent(selectedConfig)}/test`
      const res = await fetch(url, { method: 'POST' })
      const data = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
      if (res.ok) {
        const msg = data.reply ?? data.message
        setTestResult({ ok: data.ok !== false, message: msg, error: data.error })
      } else {
        const errMsg = typeof data.detail === 'string' ? data.detail : data.error || JSON.stringify(data.detail) || `HTTP ${res.status}`
        setTestResult({ ok: false, error: errMsg })
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Request failed'
      setTestResult({
        ok: false,
        error: msg.includes('fetch') || msg.includes('Failed') ? `${msg}. Is the backend running at ${API_BASE}?` : msg,
      })
    } finally {
      setTestLoading(false)
    }
  }

  if (!open) return null

  return (
    <div className="llm-config-backdrop" onClick={onClose} aria-hidden>
      <div className="llm-config-modal" onClick={(e) => e.stopPropagation()}>
        <div className="llm-config-header">
          <h2 className="llm-config-title">Configure LLM provider</h2>
          <button type="button" className="llm-config-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>
        <div className="llm-config-body">
          {loading && !config && (
            <p className="llm-config-loading">Loading…</p>
          )}
          {error && (
            <div className="llm-config-error" role="alert">
              {error}
            </div>
          )}
          {message && (
            <div className="llm-config-message">{message}</div>
          )}

          <div className="llm-config-field llm-config-select-row">
            <label htmlFor="llm-config-select">Config</label>
            <div className="llm-config-select-and-test">
              <select
                id="llm-config-select"
                value={selectedConfig}
                onChange={(e) => {
                  setSelectedConfig(e.target.value)
                  setTestResult(null)
                }}
              >
                {configNames.map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
              <button
                type="button"
                className="llm-config-test"
                onClick={handleTest}
                disabled={!selectedConfig || testLoading}
                title="Test this LLM config (credentials and connectivity)"
              >
                {testLoading ? 'Testing…' : 'Test LLM'}
              </button>
            </div>
            {testResult && (
              <div
                className={testResult.ok ? 'llm-config-test-success' : 'llm-config-test-error'}
                role="alert"
                aria-live="polite"
              >
                {testResult.ok ? (
                  <>
                    <strong>Connected.</strong>{' '}
                    {testResult.message ? `LLM replied: “${testResult.message}”` : 'LLM responded.'}
                  </>
                ) : (
                  <>
                    <strong>Test failed.</strong> {testResult.error}
                  </>
                )}
              </div>
            )}
          </div>

          {config && (
            <>
              <div className="llm-config-field">
                <label htmlFor="llm-config-provider">Provider</label>
                <select
                  id="llm-config-provider"
                  value={config.provider ?? ''}
                  onChange={(e) => updateConfig({ provider: e.target.value })}
                >
                  <option value="">—</option>
                  {providers.map((p) => (
                    <option key={p} value={p}>{p}</option>
                  ))}
                </select>
              </div>
              <div className="llm-config-field">
                <label htmlFor="llm-config-model">Model</label>
                <input
                  id="llm-config-model"
                  type="text"
                  value={config.model ?? ''}
                  onChange={(e) => updateConfig({ model: e.target.value })}
                  placeholder="e.g. llama3.1:8b, gpt-4o-mini"
                />
              </div>

              <div className="llm-config-section">
                <h3 className="llm-config-section-title">Options</h3>
                <div className="llm-config-fields">
                  <div className="llm-config-field">
                    <label htmlFor="llm-opt-temperature">Temperature</label>
                    <input
                      id="llm-opt-temperature"
                      type="number"
                      min={0}
                      max={2}
                      step={0.1}
                      value={(config.options?.temperature ?? 0.1) as number}
                      onChange={(e) => updateOption('temperature', e.target.value)}
                    />
                  </div>
                  <div className="llm-config-field">
                    <label htmlFor="llm-opt-num_predict">Num predict (Ollama)</label>
                    <input
                      id="llm-opt-num_predict"
                      type="number"
                      min={1}
                      max={128000}
                      value={(config.options?.num_predict ?? 8192) as number}
                      onChange={(e) => updateOption('num_predict', e.target.value)}
                    />
                  </div>
                </div>
              </div>

              {config.provider === 'ollama' && (
                <div className="llm-config-section">
                  <h3 className="llm-config-section-title">Ollama</h3>
                  <div className="llm-config-field">
                    <label htmlFor="llm-ollama-base_url">Base URL</label>
                    <input
                      id="llm-ollama-base_url"
                      type="text"
                      value={config.ollama?.base_url ?? ''}
                      onChange={(e) => updateNested('ollama', 'base_url', e.target.value)}
                      placeholder="http://localhost:11434"
                    />
                  </div>
                </div>
              )}

              {config.provider === 'vertex' && (
                <div className="llm-config-section">
                  <h3 className="llm-config-section-title">Vertex AI</h3>
                  <p className="llm-config-provider-hint">
                    <strong>Setup:</strong> <code>.env</code> should have <code>VERTEX_PROJECT_ID</code> and <code>GOOGLE_APPLICATION_CREDENTIALS</code> (path to service account JSON). SDK: <code>pip install -e &quot;.[vertex]&quot;</code>. If you change <code>.env</code>, restart the backend.
                  </p>
                  <p className="llm-config-provider-hint">
                    <strong>Model:</strong> Use <code>gemini-1.5-pro</code>, <code>gemini-1.5-flash</code>, or <code>gemini-1.0-pro</code> (not gemini-3-flash).
                  </p>
                  <div className="llm-config-fields">
                    <div className="llm-config-field">
                      <label htmlFor="llm-vertex-project_id">Project ID</label>
                      <input
                        id="llm-vertex-project_id"
                        type="text"
                        value={config.vertex?.project_id ?? ''}
                        onChange={(e) => updateNested('vertex', 'project_id', e.target.value)}
                      />
                    </div>
                    <div className="llm-config-field">
                      <label htmlFor="llm-vertex-location">Location</label>
                      <input
                        id="llm-vertex-location"
                        type="text"
                        value={config.vertex?.location ?? ''}
                        onChange={(e) => updateNested('vertex', 'location', e.target.value)}
                        placeholder="us-central1"
                      />
                    </div>
                  </div>
                </div>
              )}

              {config.provider === 'openai' && (
                <div className="llm-config-section">
                  <h3 className="llm-config-section-title">OpenAI</h3>
                  <div className="llm-config-fields">
                    <div className="llm-config-field">
                      <label htmlFor="llm-openai-api_key">API key</label>
                      <input
                        id="llm-openai-api_key"
                        type="password"
                        autoComplete="off"
                        value={(config.openai?.api_key as string) ?? ''}
                        onChange={(e) => updateNested('openai', 'api_key', e.target.value)}
                        placeholder="*** or set OPENAI_API_KEY"
                      />
                    </div>
                    <div className="llm-config-field">
                      <label htmlFor="llm-openai-base_url">Base URL (optional)</label>
                      <input
                        id="llm-openai-base_url"
                        type="text"
                        value={config.openai?.base_url ?? ''}
                        onChange={(e) => updateNested('openai', 'base_url', e.target.value)}
                        placeholder="Azure / OpenAI-compatible endpoint"
                      />
                    </div>
                  </div>
                </div>
              )}

              <div className="llm-config-actions">
                <button
                  type="button"
                  className="llm-config-save"
                  onClick={handleSave}
                  disabled={saving}
                >
                  {saving ? 'Saving…' : 'Save'}
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
