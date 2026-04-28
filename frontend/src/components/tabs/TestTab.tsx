/**
 * TestTab — dedicated retrieval testing workbench.
 *
 * Layout:
 *   Top bar  — query input + mode + k + strategy + canonical_floor controls
 *   Main     — two columns:
 *               Left  (results): ranked chunks with confidence, scores, snippets
 *               Right (trace):   full SearchTracePanel, always open
 *
 * Intentionally standalone — no reader, no document nav. Pure retrieval QA.
 */
import { useEffect, useRef, useState } from 'react'
import { API_BASE } from '../../config'
import { SearchTracePanel } from './repository/SearchTracePanel'
import type { SearchTelemetry } from './repository/SearchTracePanel'
import './TestTab.css'

// ── Types ─────────────────────────────────────────────────────────────────────

interface CorpusChunk {
  id: string
  text: string
  document_id: string
  document_name: string
  page_number: number | null
  paragraph_index: number | null
  source_type: string
  similarity: number
  rerank_score: number
  confidence_label: 'high' | 'medium' | 'low' | 'abstain'
  retrieval_arms: string[]
  authority_level: string | null
  payer: string | null
  state: string | null
  jpd_tags: string[]   // dominant J/P/D families e.g. ["P","D"]
}

type SearchMode = 'corpus' | 'precision' | 'recall'
type AssemblyStrategy = 'score' | 'canonical_first' | 'balanced'

interface RerankSignals {
  sim_raw: number
  sim_weighted: number
  authority_raw: number
  authority_weighted: number
  length_raw: number
  length_weighted: number
  jpd_raw?: number
  jpd_weighted?: number
  jpd_tags?: string[]
  total_raw: number
  rerank_score: number
  max_weight: number
}

// ── Constants ─────────────────────────────────────────────────────────────────

const CONFIDENCE_COLOR: Record<string, string> = {
  high:    '#10b981',
  medium:  '#f59e0b',
  low:     '#6b7280',
  abstain: '#d1d5db',
}

const MODE_HELP: Record<SearchMode, string> = {
  corpus:    'Hybrid BM25 + pgvector → RRF → rerank',
  precision: 'BM25 only — exact phrase / code lookup',
  recall:    'pgvector only — semantic / paraphrase',
}

// ── Result card ───────────────────────────────────────────────────────────────

function ResultCard({
  chunk,
  rank,
  signals,
}: {
  chunk: CorpusChunk
  rank: number
  signals?: RerankSignals
}) {
  const [expanded, setExpanded] = useState(false)
  const isHybrid = chunk.retrieval_arms.length > 1

  return (
    <div className={`tt-result-card tt-result-card--${chunk.confidence_label}`}>
      {/* Header */}
      <div className="tt-result-hdr">
        <span className="tt-result-rank">#{rank}</span>
        <span className="tt-result-doc" title={chunk.document_name}>
          {chunk.document_name.length > 48
            ? `${chunk.document_name.slice(0, 48)}…`
            : chunk.document_name}
        </span>
        {chunk.page_number != null && (
          <span className="tt-result-page">p.{chunk.page_number}</span>
        )}
      </div>

      {/* Badges row */}
      <div className="tt-result-badges">
        {/* Arms */}
        {chunk.retrieval_arms.map((arm) => (
          <span key={arm} className={`tt-arm-badge tt-arm-badge--${arm}`}>
            {arm === 'bm25' ? 'BM25' : arm === 'vector' ? 'Vec' : arm}
          </span>
        ))}
        {isHybrid && <span className="tt-arm-badge tt-arm-badge--hybrid">BOTH</span>}
        {/* Confidence */}
        <span
          className="tt-conf-badge"
          style={{ color: CONFIDENCE_COLOR[chunk.confidence_label] }}
        >
          {chunk.confidence_label}
        </span>
        {/* Score */}
        <span className="tt-score-badge">
          {chunk.rerank_score.toFixed(3)}
        </span>
        {/* Authority */}
        {chunk.authority_level && (
          <span className="tt-auth-badge" title={chunk.authority_level}>
            {chunk.authority_level === 'contract_source_of_truth'
              ? 'CoT'
              : chunk.authority_level === 'payer_policy'
              ? 'PP'
              : chunk.authority_level === 'operational_suggested'
              ? 'Ops'
              : chunk.authority_level === 'fyi_not_citable'
              ? 'FYI'
              : chunk.authority_level.slice(0, 6)}
          </span>
        )}
        {chunk.payer && (
          <span className="tt-payer-badge">{chunk.payer}</span>
        )}
        {/* JPD family tags */}
        {chunk.jpd_tags && chunk.jpd_tags.map((tag) => (
          <span key={tag} className={`tt-jpd-badge tt-jpd-badge--${tag.toLowerCase()}`}>
            {tag}
          </span>
        ))}
      </div>

      {/* Snippet */}
      <button
        type="button"
        className="tt-result-snippet-btn"
        onClick={() => setExpanded((e) => !e)}
        title="Click to expand / collapse"
      >
        <p className={`tt-result-snippet${expanded ? ' expanded' : ''}`}>
          {chunk.text}
        </p>
        {!expanded && chunk.text.length > 280 && (
          <span className="tt-result-expand-hint">▼ show more</span>
        )}
        {expanded && (
          <span className="tt-result-expand-hint">▲ collapse</span>
        )}
      </button>

      {expanded && signals && (
        <div className="tt-card-signals">
          <div className="tt-card-signals-row">
            <span className="tt-sig-label">Sim</span>
            <span className="tt-sig-raw">{(signals.sim_raw ?? 0).toFixed(3)}</span>
            <span className="tt-sig-arrow">→</span>
            <span className="tt-sig-weighted">×0.30 = {(signals.sim_weighted ?? 0).toFixed(4)}</span>
          </div>
          <div className="tt-card-signals-row">
            <span className="tt-sig-label">Auth</span>
            <span className="tt-sig-raw">{(signals.authority_raw ?? 0).toFixed(3)}</span>
            <span className="tt-sig-arrow">→</span>
            <span className="tt-sig-weighted">×0.15 = {(signals.authority_weighted ?? 0).toFixed(4)}</span>
          </div>
          <div className="tt-card-signals-row">
            <span className="tt-sig-label">Len</span>
            <span className="tt-sig-raw">{(signals.length_raw ?? 0).toFixed(3)}</span>
            <span className="tt-sig-arrow">→</span>
            <span className="tt-sig-weighted">×0.10 = {(signals.length_weighted ?? 0).toFixed(4)}</span>
          </div>
          {signals.jpd_raw != null && signals.jpd_raw > 0 && (
            <div className="tt-card-signals-row">
              <span className="tt-sig-label">JPD</span>
              <span className="tt-sig-raw">{(signals.jpd_raw ?? 0).toFixed(3)}</span>
              <span className="tt-sig-arrow">→</span>
              <span className="tt-sig-weighted">×0.25 = {(signals.jpd_weighted ?? 0).toFixed(4)}</span>
              {signals.jpd_tags && signals.jpd_tags.length > 0 && (
                <span className="tt-sig-tags">
                  {signals.jpd_tags.map(t => (
                    <span key={t} className={`tt-jpd-badge tt-jpd-badge--${t.toLowerCase()}`}>{t}</span>
                  ))}
                </span>
              )}
            </div>
          )}
          <div className="tt-card-signals-total">
            {(signals.total_raw ?? 0).toFixed(4)} ÷ {(signals.max_weight ?? 1).toFixed(2)} = <strong>{(signals.rerank_score ?? 0).toFixed(4)}</strong>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export function TestTab() {
  // ── Query state ──────────────────────────────────────────────────────────
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<SearchMode>('corpus')
  const [k, setK] = useState(10)
  const [strategy, setStrategy] = useState<AssemblyStrategy>('score')
  const [canonicalFloor, setCanonicalFloor] = useState(0.5)

  // ── Results state ────────────────────────────────────────────────────────
  const [chunks, setChunks] = useState<CorpusChunk[] | null>(null)
  const [telemetry, setTelemetry] = useState<SearchTelemetry | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // ── History ──────────────────────────────────────────────────────────────
  const [history, setHistory] = useState<Array<{
    query: string
    mode: SearchMode
    k: number
    returned: number
    total_ms: number
  }>>([])

  const inputRef = useRef<HTMLInputElement>(null)

  // ── Auto-search (400ms debounce) ─────────────────────────────────────────
  useEffect(() => {
    if (!query.trim()) {
      setChunks(null)
      setTelemetry(null)
      setError(null)
      return
    }
    const timer = setTimeout(() => runSearch(), 400)
    return () => clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, mode, k, strategy, canonicalFloor])

  // ── Manual run ───────────────────────────────────────────────────────────
  const runSearch = async () => {
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    try {
      const body: Record<string, unknown> = {
        query: query.trim(),
        k,
        mode,
        assembly_strategy: strategy,
      }
      if (strategy === 'balanced') body.canonical_floor = canonicalFloor

      const resp = await fetch(`${API_BASE}/api/skills/v1/corpus_search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      if (resp.ok) {
        const data = await resp.json()
        setChunks(data.chunks ?? [])
        setTelemetry(data.telemetry ?? null)
        // Push to history
        if (data.telemetry) {
          setHistory((h) => [
            {
              query: query.trim(),
              mode,
              k,
              returned: data.telemetry.returned ?? data.chunks?.length ?? 0,
              total_ms: data.telemetry.total_ms ?? 0,
            },
            ...h.slice(0, 19),
          ])
        }
      } else {
        const detail = await resp.text().catch(() => resp.statusText)
        setError(`${resp.status} — ${detail}`)
        setChunks([])
        setTelemetry(null)
      }
    } catch (e) {
      setError(String(e))
      setChunks([])
      setTelemetry(null)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') runSearch()
  }

  const loadHistoryItem = (item: typeof history[0]) => {
    setQuery(item.query)
    setMode(item.mode)
    setK(item.k)
  }

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="test-tab">
      {/* ── Top control bar ──────────────────────────────────────────────── */}
      <div className="tt-controls">
        {/* Mode selector */}
        <div className="tt-mode-group" role="group" aria-label="Search mode">
          {(['corpus', 'precision', 'recall'] as SearchMode[]).map((m) => (
            <button
              key={m}
              type="button"
              className={`tt-mode-btn${mode === m ? ' active' : ''}`}
              title={MODE_HELP[m]}
              onClick={() => setMode(m)}
              aria-pressed={mode === m}
            >
              {m === 'corpus' ? 'Hybrid' : m === 'precision' ? 'BM25' : 'Semantic'}
            </button>
          ))}
        </div>

        {/* Query input */}
        <div className="tt-query-wrap">
          <input
            ref={inputRef}
            type="search"
            className="tt-query-input"
            placeholder="Ask a question or search a code / phrase…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            aria-label="Search query"
          />
          {loading && <span className="tt-spinner" aria-hidden>⟳</span>}
          {query && !loading && (
            <button
              className="tt-clear-btn"
              onClick={() => { setQuery(''); setChunks(null); setTelemetry(null) }}
              aria-label="Clear"
            >×</button>
          )}
        </div>

        {/* k selector */}
        <div className="tt-k-group" role="group" aria-label="Result count">
          {[5, 10, 20].map((n) => (
            <button
              key={n}
              type="button"
              className={`tt-k-btn${k === n ? ' active' : ''}`}
              onClick={() => setK(n)}
            >
              k={n}
            </button>
          ))}
        </div>

        {/* Assembly strategy */}
        <div className="tt-strategy-group" role="group" aria-label="Assembly strategy">
          {(['score', 'canonical_first', 'balanced'] as AssemblyStrategy[]).map((s) => (
            <button
              key={s}
              type="button"
              className={`tt-strategy-btn${strategy === s ? ' active' : ''}`}
              title={
                s === 'score'
                  ? 'Pure rerank score order'
                  : s === 'canonical_first'
                  ? 'Promote CoT above PP within confidence bands'
                  : 'Reserve floor% of slots for canonical docs'
              }
              onClick={() => setStrategy(s)}
            >
              {s === 'score' ? 'Score' : s === 'canonical_first' ? 'CoT-first' : 'Balanced'}
            </button>
          ))}
        </div>

        {/* Canonical floor (only for balanced) */}
        {strategy === 'balanced' && (
          <div className="tt-floor-wrap">
            <label className="tt-floor-label" htmlFor="tt-floor">
              floor {Math.round(canonicalFloor * 100)}%
            </label>
            <input
              id="tt-floor"
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={canonicalFloor}
              onChange={(e) => setCanonicalFloor(Number(e.target.value))}
              className="tt-floor-range"
            />
          </div>
        )}

        {/* Run button */}
        <button
          type="button"
          className="tt-run-btn"
          onClick={runSearch}
          disabled={loading || !query.trim()}
          title="Run search (Enter)"
        >
          {loading ? '…' : '▶ Run'}
        </button>
      </div>

      {/* ── Error banner ─────────────────────────────────────────────────── */}
      {error && (
        <div className="tt-error-banner">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* ── Main split area ──────────────────────────────────────────────── */}
      <div className="tt-body">
        {/* ── Left: history + results ──────────────────────────────────── */}
        <div className="tt-left">
          {/* Query history */}
          {history.length > 0 && (
            <div className="tt-history">
              <div className="tt-history-hdr">Recent queries</div>
              <ul className="tt-history-list">
                {history.map((item, i) => (
                  <li key={i}>
                    <button
                      type="button"
                      className={`tt-history-item${item.query === query && item.mode === mode ? ' active' : ''}`}
                      onClick={() => loadHistoryItem(item)}
                    >
                      <span className="tt-history-query" title={item.query}>
                        {item.query.length > 52 ? `${item.query.slice(0, 52)}…` : item.query}
                      </span>
                      <span className="tt-history-meta">
                        <span className={`tt-history-mode tt-history-mode--${item.mode}`}>
                          {item.mode === 'corpus' ? 'Hybrid' : item.mode === 'precision' ? 'BM25' : 'Sem'}
                        </span>
                        <span className="tt-history-stats">
                          {item.returned} result{item.returned !== 1 ? 's' : ''} · {item.total_ms.toFixed(0)}ms
                        </span>
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Results */}
          {!query.trim() && !loading && (
            <div className="tt-empty-state">
              <p className="tt-empty-title">Retrieval Test Workbench</p>
              <p className="tt-empty-hint">
                Type a query above — all 3 modes available. The full pipeline trace
                (BM25 arm, vector arm, RRF fusion, reranking signals, assembly) will
                appear on the right the moment results land.
              </p>
              <div className="tt-example-queries">
                <p className="tt-example-label">Try:</p>
                {[
                  'prior authorization behavioral health',
                  'H2019 per diem rate',
                  'inpatient mental health admission criteria',
                  'telehealth reimbursement policy',
                  'credentialing requirements for providers',
                ].map((q) => (
                  <button
                    key={q}
                    type="button"
                    className="tt-example-btn"
                    onClick={() => { setQuery(q); inputRef.current?.focus() }}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {loading && !chunks && (
            <div className="tt-loading">Searching…</div>
          )}

          {!loading && chunks !== null && chunks.length === 0 && (
            <div className="tt-no-results">
              No results for &ldquo;{query}&rdquo;
              {telemetry && (
                <span className="tt-no-results-time">
                  {' '}· {telemetry.total_ms?.toFixed(0)}ms
                </span>
              )}
            </div>
          )}

          {chunks !== null && chunks.length > 0 && (
            <div className="tt-results">
              <div className="tt-results-hdr">
                <span>
                  {chunks.length} result{chunks.length !== 1 ? 's' : ''}
                  {telemetry?.arm_hits && (
                    <span className="tt-results-arms">
                      {' '}· BM25 {telemetry.arm_hits.bm25} / Vec {telemetry.arm_hits.vector}
                    </span>
                  )}
                </span>
                {telemetry && (
                  <span className="tt-results-time">
                    {telemetry.total_ms?.toFixed(0)}ms
                  </span>
                )}
              </div>
              <div className="tt-results-list">
                {chunks.map((chunk, i) => {
                  const traceItem = telemetry?.scoring_trace?.find(
                    (t: any) => t.chunk_id === chunk.id
                  )
                  return (
                    <ResultCard
                      key={chunk.id}
                      chunk={chunk}
                      rank={i + 1}
                      signals={traceItem?.rerank_signals}
                    />
                  )
                })}
              </div>
            </div>
          )}
        </div>

        {/* ── Right: trace panel ────────────────────────────────────────── */}
        <div className="tt-right">
          {telemetry ? (
            <SearchTracePanel telemetry={telemetry} />
          ) : (
            <div className="tt-trace-placeholder">
              {loading ? (
                <span>Running pipeline…</span>
              ) : query.trim() ? (
                <span>Awaiting results…</span>
              ) : (
                <span>Trace will appear here</span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
