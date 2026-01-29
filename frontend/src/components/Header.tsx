import { useState, useEffect } from 'react'
import './Header.css'

const API_BASE = 'http://localhost:8000'

export interface ChunkingOptions {
  threshold: number
  critiqueEnabled: boolean
  maxRetries: number
}

export interface ChunkingOptionsSetters {
  setThreshold: (v: number) => void
  setCritiqueEnabled: (v: boolean) => void
  setMaxRetries: (v: number) => void
}

interface HeaderProps {
  chunkingOptions?: ChunkingOptions
  onChunkingOptionsChange?: ChunkingOptionsSetters
  defaultLlmConfigVersion?: string | null
  onDefaultLlmConfigVersionChange?: (v: string | null) => void
  defaultPromptVersions?: Record<string, string> | null
  onDefaultPromptVersionsChange?: (v: Record<string, string> | null) => void
}

export function Header({
  chunkingOptions,
  onChunkingOptionsChange,
  defaultLlmConfigVersion,
  onDefaultLlmConfigVersionChange,
  defaultPromptVersions,
  onDefaultPromptVersionsChange,
}: HeaderProps) {
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [llmConfigs, setLlmConfigs] = useState<{ configs: string[]; default: string } | null>(null)
  const [promptsConfig, setPromptsConfig] = useState<{ prompts: Record<string, string[]>; default: Record<string, string> } | null>(null)

  useEffect(() => {
    if (!settingsOpen) return
    const load = async () => {
      try {
        const [llmRes, promptsRes] = await Promise.all([
          fetch(`${API_BASE}/config/llm`),
          fetch(`${API_BASE}/config/prompts`),
        ])
        if (llmRes.ok) {
          const data = await llmRes.json()
          setLlmConfigs({ configs: data.configs || [], default: data.default || 'default' })
        }
        if (promptsRes.ok) {
          const data = await promptsRes.json()
          setPromptsConfig({ prompts: data.prompts || {}, default: data.default || {} })
        }
      } catch (e) {
        console.error('Failed to load config for settings:', e)
      }
    }
    load()
  }, [settingsOpen])

  return (
    <header className="app-header">
      <div className="app-header-inner">
        <div className="app-header-left">
          <h1>Mobius RAG</h1>
          <p className="subtitle">Document Processing & Fact Extraction</p>
        </div>
        <div className="app-header-right">
          <button
            type="button"
            className="header-settings-btn"
            onClick={() => setSettingsOpen(!settingsOpen)}
            title="Settings"
            aria-label="Open settings"
          >
            <span className="header-settings-icon" aria-hidden>☰</span>
          </button>
        </div>
      </div>

      {settingsOpen && (
        <>
          <div
            className="settings-backdrop"
            onClick={() => setSettingsOpen(false)}
            aria-hidden
          />
          <div className="settings-panel">
            <div className="settings-panel-header">
              <h2 className="settings-panel-title">Settings</h2>
              <button
                type="button"
                className="settings-panel-close"
                onClick={() => setSettingsOpen(false)}
                aria-label="Close settings"
              >
                ×
              </button>
            </div>
            <div className="settings-panel-body">
              {/* Chunking defaults */}
              {chunkingOptions && onChunkingOptionsChange && (
                <section className="settings-section">
                  <h3 className="settings-section-title">Chunking defaults</h3>
                  <p className="settings-section-hint">Used when starting or restarting chunking.</p>
                  <div className="settings-fields">
                    <div className="settings-field">
                      <label htmlFor="settings-threshold">Critique pass threshold (0–1)</label>
                      <input
                        id="settings-threshold"
                        type="number"
                        min={0}
                        max={1}
                        step={0.1}
                        value={chunkingOptions.threshold}
                        onChange={e => onChunkingOptionsChange.setThreshold(parseFloat(e.target.value) || 0.6)}
                      />
                    </div>
                    <div className="settings-field">
                      <label className="settings-checkbox-label">
                        <input
                          type="checkbox"
                          checked={chunkingOptions.critiqueEnabled}
                          onChange={e => onChunkingOptionsChange.setCritiqueEnabled(e.target.checked)}
                        />
                        Run critique (QA)
                      </label>
                    </div>
                    <div className="settings-field">
                      <label htmlFor="settings-max-retries">Max retries on critique fail</label>
                      <input
                        id="settings-max-retries"
                        type="number"
                        min={0}
                        max={10}
                        value={chunkingOptions.maxRetries}
                        onChange={e => onChunkingOptionsChange.setMaxRetries(parseInt(e.target.value, 10) || 0)}
                      />
                    </div>
                  </div>
                </section>
              )}

              {/* LLM config */}
              {llmConfigs && onDefaultLlmConfigVersionChange && (
                <section className="settings-section">
                  <h3 className="settings-section-title">LLM config</h3>
                  <p className="settings-section-hint">Default config used for new chunking jobs.</p>
                  <div className="settings-field">
                    <label htmlFor="settings-llm-config">Config</label>
                    <select
                      id="settings-llm-config"
                      value={defaultLlmConfigVersion ?? ''}
                      onChange={e => onDefaultLlmConfigVersionChange(e.target.value || null)}
                    >
                      <option value="">Use server default</option>
                      {llmConfigs.configs.map(c => (
                        <option key={c} value={c}>{c}</option>
                      ))}
                    </select>
                  </div>
                </section>
              )}

              {/* Prompt versions */}
              {promptsConfig && onDefaultPromptVersionsChange && (
                <section className="settings-section">
                  <h3 className="settings-section-title">Prompt versions</h3>
                  <p className="settings-section-hint">Default prompt set for new chunking jobs.</p>
                  <div className="settings-fields">
                    {Object.entries(promptsConfig.prompts).map(([name, versions]) => (
                      <div key={name} className="settings-field">
                        <label htmlFor={`settings-prompt-${name}`}>{name}</label>
                        <select
                          id={`settings-prompt-${name}`}
                          value={(defaultPromptVersions && defaultPromptVersions[name]) ?? promptsConfig.default[name] ?? (versions[0] ?? '')}
                          onChange={e => {
                            const v = e.target.value
                            onDefaultPromptVersionsChange({
                              ...(defaultPromptVersions ?? promptsConfig.default),
                              [name]: v,
                            })
                          }}
                        >
                          {versions.map(ver => (
                            <option key={ver} value={ver}>{ver}</option>
                          ))}
                        </select>
                      </div>
                    ))}
                  </div>
                  <button
                    type="button"
                    className="settings-reset-prompts"
                    onClick={() => onDefaultPromptVersionsChange(null)}
                  >
                    Use server default set
                  </button>
                </section>
              )}
            </div>
          </div>
        </>
      )}
    </header>
  )
}
