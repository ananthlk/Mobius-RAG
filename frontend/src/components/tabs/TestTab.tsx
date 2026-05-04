/**
 * TestTab — interactive query playground for ``corpus_search_agent``.
 *
 * Send any query, optionally force a specific strategy or caller_mode,
 * and see the full thinking pipeline rendered the same way as the
 * EvalTab drilldown (parser → partition → pool → router → strategies
 * → assembler).
 *
 * History sidebar persists in localStorage so re-running yesterday's
 * one-off check is one click.
 */
import { useEffect, useState } from 'react'
import { API_BASE } from '../../config'
import { AgentPipelineTrace, type AgentResponse } from './AgentPipelineTrace'
import './EvalTab.css'   // reuse styles (run-header, kpi, section, etc.)
import './TestTab.css'

const CALLER_MODES = [
  'chat.default',
  'chat.thinking',
  'chat.copilot',
  'auth_agent',
  'research',
  'batch',
] as const

const STRATEGY_OVERRIDES = [
  { value: '', label: 'Auto (router decides)' },
  { value: 'a', label: 'Force a — BM25 cascade' },
  { value: 'b', label: 'Force b — Wide → themes' },
  { value: 'c', label: 'Force c — Reverse RAG' },
  { value: 'd', label: 'Force d — External (Google + scrape)' },
] as const

interface HistoryItem {
  query: string
  caller_mode: string
  mode: string
  ts: string
  strategy_used: string | null
  confidence: string | null
  total_ms: number | null
}

const HISTORY_KEY = 'rag-test-tab-history-v2'

export function TestTab() {
  const [query, setQuery] = useState('')
  const [callerMode, setCallerMode] = useState<typeof CALLER_MODES[number]>('chat.default')
  const [strategy, setStrategy] = useState<string>('')
  const [k, setK] = useState(5)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [response, setResponse] = useState<AgentResponse | null>(null)
  const [history, setHistory] = useState<HistoryItem[]>(() => {
    try {
      return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]')
    } catch {
      return []
    }
  })

  // Persist history.
  useEffect(() => {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 30)))
    } catch {
      // ignore quota errors
    }
  }, [history])

  async function run(q?: string) {
    const trimmed = (q ?? query).trim()
    if (!trimmed) return
    setLoading(true)
    setError(null)
    setResponse(null)
    try {
      const body: Record<string, unknown> = {
        query: trimmed,
        k,
        caller_mode: callerMode,
      }
      if (strategy) body.mode = strategy
      const resp = await fetch(`${API_BASE}/api/skills/v1/corpus_search_agent`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!resp.ok) {
        const txt = await resp.text().catch(() => resp.statusText)
        setError(`${resp.status} ${txt.slice(0, 300)}`)
        return
      }
      const data = (await resp.json()) as AgentResponse
      setResponse(data)
      setHistory((h) => [
        {
          query: trimmed,
          caller_mode: callerMode,
          mode: strategy,
          ts: new Date().toISOString(),
          strategy_used: data.strategy_used || null,
          confidence: data.confidence || null,
          total_ms: ((data.telemetry || {}) as Record<string, unknown>).total_ms as number || null,
        },
        ...h.filter((it) => it.query !== trimmed || it.mode !== strategy).slice(0, 29),
      ])
    } catch (e) {
      setError(`${e}`)
    } finally {
      setLoading(false)
    }
  }

  function rerun(item: HistoryItem) {
    setQuery(item.query)
    setCallerMode(item.caller_mode as typeof CALLER_MODES[number])
    setStrategy(item.mode)
    void run(item.query)
  }

  return (
    <div className="test-tab">
      {/* ── Top bar: input + controls ── */}
      <div className="test-controls">
        <textarea
          className="test-query-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
              e.preventDefault()
              void run()
            }
          }}
          placeholder="Type a query — Cmd/Ctrl+Enter to run.  E.g. 'What does Sunshine Health say about prior authorization for behavioral health?'"
          rows={2}
        />
        <div className="test-control-row">
          <label>
            <span className="ctrl-label">Caller mode</span>
            <select value={callerMode} onChange={(e) => setCallerMode(e.target.value as typeof CALLER_MODES[number])}>
              {CALLER_MODES.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </label>
          <label>
            <span className="ctrl-label">Strategy</span>
            <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
              {STRATEGY_OVERRIDES.map((s) => (
                <option key={s.value} value={s.value}>{s.label}</option>
              ))}
            </select>
          </label>
          <label>
            <span className="ctrl-label">k</span>
            <input
              type="number"
              min={1}
              max={20}
              value={k}
              onChange={(e) => setK(parseInt(e.target.value, 10) || 5)}
              className="ctrl-num"
            />
          </label>
          <button
            className="run-btn"
            onClick={() => void run()}
            disabled={loading || !query.trim()}
          >
            {loading ? 'Running…' : '▶ Run'}
          </button>
        </div>
      </div>

      {error && <div className="eval-error">{error}</div>}

      <div className="test-body">
        {/* ── Recent queries (left) ── */}
        <div className="test-history">
          <h3>Recent</h3>
          {history.length === 0 && (
            <div className="eval-empty">No queries yet. Type one above.</div>
          )}
          {history.map((item, i) => (
            <div
              key={i}
              className="history-row"
              onClick={() => rerun(item)}
              title={`Re-run as ${item.caller_mode}${item.mode ? ' / ' + item.mode : ''}`}
            >
              <div className="history-query">{item.query}</div>
              <div className="history-meta">
                <span className="badge mini">{item.caller_mode}</span>
                {item.mode && <span className="badge mini warn">force={item.mode}</span>}
                {item.strategy_used && (
                  <span className="badge mini ok">→ {item.strategy_used}</span>
                )}
                {item.confidence && <span className="badge mini">{item.confidence}</span>}
                {item.total_ms != null && (
                  <span className="badge mini dim">{item.total_ms}ms</span>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* ── Pipeline trace (right) ── */}
        <div className="test-trace">
          {!response && !loading && (
            <div className="eval-empty">
              Run a query to see the full pipeline trace.
            </div>
          )}
          {loading && (
            <div className="eval-empty">Running through the pipeline…</div>
          )}
          {response && (
            <>
              <RunSummaryHeader response={response} />
              <AgentPipelineTrace response={response} />
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function RunSummaryHeader({ response }: { response: AgentResponse }) {
  const tel = (response.telemetry || {}) as Record<string, unknown>
  const totalMs = (tel.total_ms as number) || null
  const routing = response.routing || {}
  return (
    <div className="run-header">
      <div className="run-title">
        <h2>
          <span className="qid">{response.strategy_used || '?'}</span>{' '}
          {response.confidence && (
            <span className={`verdict ${response.confidence === 'high' ? 'verdict-correct' : response.confidence === 'low' ? 'verdict-wrong' : 'verdict-partial'}`}>
              {response.confidence}
            </span>
          )}
          {totalMs && <span className="score">{totalMs}ms</span>}
        </h2>
        <div className="run-meta">
          <span className="dim-text">
            router picked <strong>{routing.strategy || '?'}</strong>
            {routing.fallback && <> · fallback <strong>{routing.fallback}</strong></>}
          </span>
          <span className="dim-text">· class {routing.query_class}</span>
          <span className="dim-text">· method {routing.method}</span>
        </div>
      </div>
    </div>
  )
}
