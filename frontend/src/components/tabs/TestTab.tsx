/**
 * TestTab — interactive query playground for ``corpus_search_agent``.
 *
 * Send any query, optionally force a specific strategy or caller_mode,
 * and see the full thinking pipeline rendered the same way as the
 * EvalTab drilldown (parser → partition → pool → router → strategies
 * → assembler).
 *
 * History sidebar: seeded from /api/routing/decisions on mount (shows
 * recent chat queries), then extended locally by queries run in this tab.
 */
import { useEffect, useRef, useState, Component, type ReactNode } from 'react'
import { API_BASE } from '../../config'
import { AgentPipelineTrace, type AgentResponse } from './AgentPipelineTrace'
import { TwoGradeBar, PerClaimLedger, type ClaimEntry } from './EvalTab'
import { QueryTraceDrilldown } from './QueryTraceDrilldown'
import './EvalTab.css'   // reuse styles (run-header, kv, section, etc.)
import './TestTab.css'

class TraceErrorBoundary extends Component<
  { children: ReactNode; boundaryKey?: string },
  { hasError: boolean; msg: string }
> {
  constructor(props: { children: ReactNode; boundaryKey?: string }) {
    super(props)
    this.state = { hasError: false, msg: '' }
  }
  static getDerivedStateFromError(e: unknown) {
    return { hasError: true, msg: String(e) }
  }
  componentDidUpdate(prev: { boundaryKey?: string }) {
    if (prev.boundaryKey !== this.props.boundaryKey && this.state.hasError) {
      this.setState({ hasError: false, msg: '' })
    }
  }
  render() {
    if (this.state.hasError) {
      return <div className="eval-error" style={{ fontSize: 12 }}>Trace render error: {this.state.msg}</div>
    }
    return this.props.children
  }
}

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
  id?: string          // decision UUID from rag_routing_decisions (server items only)
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
  const [historyRefreshing, setHistoryRefreshing] = useState(false)
  const [isStoredResult, setIsStoredResult] = useState(false)
  const [currentDecisionId, setCurrentDecisionId] = useState<string | null>(null)
  const [gradeData, setGradeData] = useState<{
    retrieval_grade: number | null
    synthesis_grade: number | null
    synthesis_gap: number | null
    per_claim_ledger: ClaimEntry[] | null
    fact_checker_version: string | null
  } | null>(null)
  const serverSeeded = useRef(false)

  function refreshHistory() {
    setHistoryRefreshing(true)
    fetch(`${API_BASE}/api/routing/decisions?limit=50`)
      .then((r) => r.ok ? r.json() : null)
      .then((data) => {
        if (!data?.decisions?.length) return
        const serverItems: HistoryItem[] = (data.decisions as any[]).map((d) => ({
          id: d.id as string,
          query: d.query as string,
          caller_mode: (d.caller_mode as string) || 'chat.default',
          mode: '',
          ts: d.ts as string,
          strategy_used: (d.strategy_executed as string) || null,
          confidence: (d.confidence as string) || null,
          total_ms: (d.total_ms as number) || null,
        }))
        setHistory((local) => {
          // Server items are the authoritative ordered list (ts DESC from DB).
          // Append local-only items (no server id, run from this tab) whose
          // query text isn't already covered by a server item — dedup by id,
          // NOT by query text, so the same query run twice both appear.
          const serverQueryTexts = new Set(serverItems.map(i => i.query))
          const localOnly = local.filter(i => !i.id && !serverQueryTexts.has(i.query))
          return [...serverItems, ...localOnly].slice(0, 50)
        })
      })
      .catch(() => { /* silently ignore — server history is best-effort */ })
      .finally(() => setHistoryRefreshing(false))
  }

  // On mount: seed history from server.
  useEffect(() => {
    if (serverSeeded.current) return
    serverSeeded.current = true
    refreshHistory()
  }, [])

  // Persist locally-run queries to localStorage.
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
    setIsStoredResult(false)
    setCurrentDecisionId(null)
    setGradeData(null)
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
      // Use the pre-generated decision ID returned in the response so the
      // drilldown always pins to exactly this run's rag_routing_decisions row.
      if (data.routing_decision_id) {
        setCurrentDecisionId(data.routing_decision_id)
      }
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

  async function loadStored(item: HistoryItem) {
    setQuery(item.query)
    setCallerMode(item.caller_mode as typeof CALLER_MODES[number])
    setStrategy(item.mode)
    // If we have a server-side decision id, fetch the stored result
    // without triggering a new run.
    if (item.id) {
      setLoading(true)
      setError(null)
      setResponse(null)
      setGradeData(null)
      try {
        const resp = await fetch(`${API_BASE}/api/routing/decisions/${item.id}`)
        if (!resp.ok) throw new Error(`${resp.status}`)
        const row = await resp.json() as Record<string, any>
        // Reconstruct a partial AgentResponse from the stored decision columns.
        const reconstructed: AgentResponse = {
          confidence: row.confidence,
          strategy_used: row.strategy_executed,
          query_profile: {
            query_type: row.query_type,
            coverage: row.coverage,
            tag_matches: row.tag_matches || [],
            literal_anchors: row.literal_anchors || [],
            untagged_meaningful_tokens: row.untagged_meaningful || [],
            raw_query: (row.prefs_received as any)?.query || row.query,
          },
          routing: {
            strategy: row.strategy_chosen,
            executed_strategy: row.strategy_executed,
            fallback: row.fallback_strategy,
            query_class: row.query_class,
            method: row.routing_method,
            scores: row.scores || {},
            self_assessments: row.self_assessments || {},
            withdrawn: row.withdrawn || [],
            prefs_resolved: row.prefs_resolved || {},
            priors_version: row.priors_version,
            fail_fast_reason: row.fail_fast_reason,
          },
          telemetry: {
            total_ms: row.total_ms,
            agent_id: row.agent_id,
            ...(row.per_strategy_telemetry || {}),
          } as any,
        }
        setResponse(reconstructed)
        setIsStoredResult(true)
        setCurrentDecisionId(item.id)
        // Extract two-grade QA fields (populated when EVAL agent has computed grades).
        if (row.retrieval_grade != null || row.per_claim_ledger != null) {
          setGradeData({
            retrieval_grade: row.retrieval_grade ?? null,
            synthesis_grade: row.synthesis_grade ?? null,
            synthesis_gap: row.synthesis_gap ?? null,
            per_claim_ledger: row.per_claim_ledger ?? null,
            fact_checker_version: row.fact_checker_version ?? null,
          })
        }
      } catch (e) {
        setError(`Could not load stored decision: ${e}`)
      } finally {
        setLoading(false)
      }
    } else {
      // Locally-run item with no stored id — just populate the input
      // so the user can re-run it manually.
    }
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
          <div className="test-history-header">
            <h3>Recent</h3>
            <button
              className="btn-icon"
              onClick={refreshHistory}
              disabled={historyRefreshing}
              title="Reload history from server"
            >
              {historyRefreshing ? '…' : '↺'}
            </button>
          </div>
          {history.length === 0 && (
            <div className="eval-empty">No recent queries found.</div>
          )}
          {history.map((item, i) => (
            <div
              key={i}
              className="history-row"
              onClick={() => void loadStored(item)}
              title={item.id ? 'Load stored result' : `Set query (no stored result)`}
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
            <TraceErrorBoundary boundaryKey={query + String(currentDecisionId)}>
              <>
                <div className="trace-run-header">
                  <RunSummaryHeader response={response} />
                  {isStoredResult && (
                    <div className="stored-result-bar">
                      <span className="badge mini dim">Stored result — routing &amp; scores only, no chunks</span>
                      <button
                        className="run-btn small"
                        onClick={() => void run()}
                        disabled={loading || !query.trim()}
                      >
                        ▶ Re-run live
                      </button>
                    </div>
                  )}
                </div>
                {currentDecisionId ? (
                  <QueryTraceDrilldown decisionId={currentDecisionId} />
                ) : (
                  <>
                    {gradeData && (gradeData.retrieval_grade != null || gradeData.synthesis_grade != null) && (
                      <TwoGradeBar
                        retrieval={gradeData.retrieval_grade}
                        synthesis={gradeData.synthesis_grade}
                        gap={gradeData.synthesis_gap}
                      />
                    )}
                    {gradeData?.per_claim_ledger && gradeData.per_claim_ledger.length > 0 && (
                      <PerClaimLedger claims={gradeData.per_claim_ledger} chunks={response.chunks ?? null} />
                    )}
                  </>
                )}
                {!isStoredResult && <AgentPipelineTrace response={response} />}
              </>
            </TraceErrorBoundary>
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
