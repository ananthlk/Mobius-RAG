import { useState, useEffect } from 'react'
import './Header.css'
import { LLMConfigModal } from './LLMConfigModal'
import { PromptsModal } from './PromptsModal'

import { API_BASE } from '../config'

interface HeaderProps {
  defaultLlmConfigVersion?: string | null
  onDefaultLlmConfigVersionChange?: (v: string | null) => void
}

export function Header({
  defaultLlmConfigVersion,
  onDefaultLlmConfigVersionChange,
}: HeaderProps) {
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [llmConfigOpen, setLlmConfigOpen] = useState(false)
  const [promptsModalOpen, setPromptsModalOpen] = useState(false)
  const [llmConfigs, setLlmConfigs] = useState<{ configs: string[]; default: string } | null>(null)

  useEffect(() => {
    if (!settingsOpen) return
    const load = async () => {
      try {
        const llmRes = await fetch(`${API_BASE}/config/llm`)
        if (llmRes.ok) {
          const data = await llmRes.json()
          setLlmConfigs({ configs: data.configs || [], default: data.default || 'default' })
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
          <div className="app-header-brand">
            <img src="/logo.svg" alt="Mobius" className="app-header-logo" width={32} height={32} />
            <h1>Mobius RAG</h1>
          </div>
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
              {/* LLM config: edit credentials and test */}
              <section className="settings-section">
                <h3 className="settings-section-title">LLM config</h3>
                <p className="settings-section-hint">Edit provider settings and test the connection.</p>
                {llmConfigs && onDefaultLlmConfigVersionChange && (
                  <div className="settings-field">
                    <label htmlFor="settings-llm-config">Default config for new jobs</label>
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
                )}
                <button
                  type="button"
                  className="settings-configure-llm"
                  onClick={() => {
                    setSettingsOpen(false)
                    setLlmConfigOpen(true)
                  }}
                >
                  Edit LLM config &amp; test
                </button>
              </section>

              {/* Prompt management: CRUD */}
              <section className="settings-section">
                <h3 className="settings-section-title">Prompts</h3>
                <p className="settings-section-hint">Create, edit, and delete prompt templates (extraction, critique, etc.).</p>
                <button
                  type="button"
                  className="settings-configure-llm"
                  onClick={() => {
                    setSettingsOpen(false)
                    setPromptsModalOpen(true)
                  }}
                >
                  Manage prompts
                </button>
              </section>
            </div>
          </div>
        </>
      )}

      <LLMConfigModal open={llmConfigOpen} onClose={() => setLlmConfigOpen(false)} />
      <PromptsModal open={promptsModalOpen} onClose={() => setPromptsModalOpen(false)} />
    </header>
  )
}
