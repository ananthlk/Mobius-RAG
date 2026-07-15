/**
 * EvalTab — browse eval runs + drill into per-query results.
 *
 * Reads from:
 *   GET /api/eval/runs                — runs list
 *   GET /api/eval/runs/{id}           — run summary + per-query rows
 *   GET /api/eval/results/{id}        — full drilldown (chunks, judge reasoning)
 *
 * Layout: master/detail.
 *   Left:   list of runs (newest first)
 *   Middle: list of results in selected run
 *   Right:  full drilldown of selected result
 *           — query, expected, judge verdict + reasoning
 *           — chunks retrieved (text + doc + page + rerank)
 *           — routing decision linked from row
 */
import { useEffect, useState } from 'react'
import { API_BASE } from '../../config'
import { AgentPipelineTrace, type AgentResponse, type JudgeBlock } from './AgentPipelineTrace'
import './EvalTab.css'

interface EvalRunRow {
  id: string
  ts: string
  bank_path: string
  bank_version: string | null
  priors_version: string
  notes: string | null
  n_queries: number
  n_correct: number | null
  n_partial: number | null
  n_wrong: number | null
  n_unable: number | null
  routing_accuracy: number | null
  citation_hit_rate: number | null
  median_latency_ms: number | null
  p95_latency_ms: number | null
}

interface EvalResultRow {
  id: string
  query_id: string
  query: string
  expected: Record<string, unknown>
  strategy_chosen: string | null
  strategy_executed: string | null
  confidence: string | null
  total_ms: number | null
  n_chunks: number | null
  top_rerank: number | null
  routing_correct: boolean | null
  citation_hit: boolean | null
  judge_verdict: string | null
  judge_score: number | null
  judge_reasoning: string | null
  judge_model: string | null
  human_verdict?: string | null
  human_reasoning?: string | null
  human_verdict_at?: string | null
  human_verdict_by?: string | null
  effective_verdict?: string | null
  routing_decision_id: string | null
  // Two-grade QA — populated once EVAL agent wires up rag_query_decisions
  retrieval_grade?: number | null
  synthesis_grade?: number | null
  synthesis_gap?: number | null
}

export interface ClaimEntry {
  fact: string
  status: 'validated' | 'unvalidated' | 'contradicted'
  chunk_id: string | null
  quote: string | null
}

interface ChunkSummary {
  document_name: string | null
  page_number: number | null
  rerank_score: number | null
  text: string | null
}

interface ResultDetail {
  result: EvalResultRow & {
    llm_answer: string | null
    chunks_summary: ChunkSummary[]
    full_response: AgentResponse | null
    // Two-grade QA fields (from rag_query_decisions via JOIN)
    retrieval_grade?: number | null
    synthesis_grade?: number | null
    synthesis_gap?: number | null
    per_claim_ledger?: ClaimEntry[] | null
  }
  routing: {
    scores?: Record<string, number>
    self_assessments?: Record<string, { est_recall: number; reason: string }>
    withdrawn?: string[]
    prefs_resolved?: Record<string, unknown>
    query_class?: string
    routing_method?: string
    fallback_strategy?: string | null
    priors_version?: string
  } | null
}

const VERDICT_COLOR: Record<string, string> = {
  correct: 'verdict-correct',
  partial: 'verdict-partial',
  wrong: 'verdict-wrong',
  unable_to_verify: 'verdict-unable',
}

interface BankQuery {
  id: string
  query: string
  caller_mode?: string
  expected?: {
    strategy?: string
    query_class?: string
    answer_keywords?: string[]
    must_cite_doc?: string[]
    must_cite_url_contains?: string[]
    fail_fast_reason?: string
  }
  notes?: string
}

interface ActiveRun {
  active: boolean
  run_id?: string
  notes?: string
  n_queries?: number
  n_completed?: number
}


export function EvalTab() {
  const [runs, setRuns] = useState<EvalRunRow[]>([])
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [runDetail, setRunDetail] = useState<{ run: EvalRunRow; results: EvalResultRow[] } | null>(null)
  // 5-axis calibration summary (present only for forced-strategy calibration runs).
  const [calibSummary, setCalibSummary] = useState<any>(null)
  // Two-grade QA rollup per strategy (from rag_query_decisions, keyed by eval_run_id).
  const [gradeRollup, setGradeRollup] = useState<any>(null)
  const [expandedResultId, setExpandedResultId] = useState<string | null>(null)
  const [resultDetailCache, setResultDetailCache] = useState<Record<string, ResultDetail>>({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState<{ key: string; value: string } | null>(null)

  // Bank editor.
  const [bankOpen, setBankOpen] = useState(false)
  const [bank, setBank] = useState<BankQuery[]>([])
  const [bankDirty, setBankDirty] = useState(false)
  const [bankSaving, setBankSaving] = useState(false)

  // Run Eval state.
  const [active, setActive] = useState<ActiveRun | null>(null)
  const [runNotes, setRunNotes] = useState('from-ui')
  const [runsCollapsed, setRunsCollapsed] = useState(false)
  const [triggering, setTriggering] = useState(false)

  // Observability dashboard (timeline / A-B compare / drift) — default open.
  const [dashOpen, setDashOpen] = useState(true)

  // Initial: load runs.
  useEffect(() => {
    refreshRuns()
    refreshActive()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function refreshRuns() {
    fetch(`${API_BASE}/api/eval/runs?limit=50`)
      .then((r) => r.json())
      .then((d) => {
        setRuns(d.runs || [])
        setSelectedRunId((cur) => cur || ((d.runs || [])[0]?.id ?? null))
      })
      .catch((e) => setError(`runs: ${e}`))
  }

  function refreshActive() {
    fetch(`${API_BASE}/api/eval/active`)
      .then((r) => r.json())
      .then((d) => {
        if (d.active) {
          setActive(d)
          // Auto-select the running run so the user sees rows stream in.
          setSelectedRunId(d.run_id)
        } else {
          setActive(null)
        }
      })
      .catch(() => {/* no-op */})
  }

  // Poll active run when one is in flight (every 2s).
  // Side effect: if the active run is the one currently selected in
  // the runs list, refetch its detail too so per-query rows stream
  // in live as the calibration / eval writes them.
  useEffect(() => {
    if (!active?.run_id) return
    const t = setInterval(() => {
      fetch(`${API_BASE}/api/eval/runs/${active.run_id}/progress`)
        .then((r) => r.json())
        .then((d) => {
          if (!d.is_running) {
            setActive(null)
            refreshRuns()
            setSelectedRunId(d.run_id)
            return
          }
          setActive((cur) => cur ? { ...cur, n_completed: d.n_completed } : cur)
          // Live row stream: if the active run is what's selected,
          // re-fetch the run detail so new results appear.
          if (selectedRunId === active.run_id) {
            fetch(`${API_BASE}/api/eval/runs/${active.run_id}`)
              .then((r) => r.json())
              .then((d2) => setRunDetail(d2))
              .catch(() => {/* no-op */})
          }
        })
        .catch(() => {/* no-op */})
    }, 2000)
    return () => clearInterval(t)
  }, [active?.run_id, selectedRunId])

  // Bank load on open.
  useEffect(() => {
    if (!bankOpen || bank.length > 0) return
    fetch(`${API_BASE}/api/eval/bank`)
      .then((r) => r.json())
      .then((d) => setBank(d.queries || []))
      .catch((e) => setError(`bank: ${e}`))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bankOpen])

  function updateBankItem(i: number, patch: Partial<BankQuery>) {
    setBank((b) => b.map((q, idx) => (idx === i ? { ...q, ...patch } : q)))
    setBankDirty(true)
  }
  function updateBankExpected(i: number, patch: Partial<NonNullable<BankQuery['expected']>>) {
    setBank((b) => b.map((q, idx) => (idx === i ? { ...q, expected: { ...(q.expected || {}), ...patch } } : q)))
    setBankDirty(true)
  }
  function addBankItem() {
    setBank((b) => [
      ...b,
      {
        id: `q${String(b.length + 1).padStart(3, '0')}`,
        query: '',
        expected: { strategy: 'a' },
      },
    ])
    setBankDirty(true)
  }
  function deleteBankItem(i: number) {
    setBank((b) => b.filter((_, idx) => idx !== i))
    setBankDirty(true)
  }

  async function saveBank() {
    setBankSaving(true)
    setError(null)
    try {
      const resp = await fetch(`${API_BASE}/api/eval/bank`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ queries: bank }),
      })
      if (!resp.ok) {
        const t = await resp.text()
        setError(`save bank: ${resp.status} ${t.slice(0, 200)}`)
      } else {
        setBankDirty(false)
      }
    } finally {
      setBankSaving(false)
    }
  }

  async function triggerRun() {
    return await _trigger('/api/eval/trigger', 'eval')
  }

  async function triggerCalibration() {
    return await _trigger('/api/eval/calibrate/trigger', 'calibration')
  }

  async function _trigger(endpoint: string, kind: string) {
    setTriggering(true)
    setError(null)
    try {
      const resp = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes: runNotes || `from-ui (${kind})` }),
      })
      if (!resp.ok) {
        const t = await resp.text()
        setError(`${kind} trigger: ${resp.status} ${t.slice(0, 200)}`)
      } else {
        setTimeout(refreshActive, 1500)
        setTimeout(refreshRuns, 1500)
      }
    } finally {
      setTriggering(false)
    }
  }

  // When a run is selected, load its results.
  useEffect(() => {
    if (!selectedRunId) return
    setLoading(true)
    setExpandedResultId(null)
    setResultDetailCache({})
    setFilter(null)
    setCalibSummary(null)
    setGradeRollup(null)
    fetch(`${API_BASE}/api/eval/runs/${selectedRunId}`)
      .then((r) => r.json())
      .then((d) => setRunDetail(d))
      .catch((e) => setError(`run detail: ${e}`))
      .finally(() => setLoading(false))
    // 5-axis summary — populated only when the run has forced-strategy
    // calibration cells (fact_checker metrics); otherwise strategies is empty.
    fetch(`${API_BASE}/api/eval/runs/${selectedRunId}/calibration_summary`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => setCalibSummary(d && Object.keys(d.strategies || {}).length ? d : null))
      .catch(() => setCalibSummary(null))
    // Two-grade QA rollup — populated only when rag_query_decisions has rows for this run.
    fetch(`${API_BASE}/api/eval/runs/${selectedRunId}/grade_rollup`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => setGradeRollup(d && Object.keys(d.strategies || {}).length ? d : null))
      .catch(() => setGradeRollup(null))
  }, [selectedRunId])

  // Expand-on-click: lazy fetch full drilldown for selected result.
  function toggleExpand(resultId: string) {
    if (expandedResultId === resultId) {
      setExpandedResultId(null)
      return
    }
    setExpandedResultId(resultId)
    if (resultDetailCache[resultId]) return
    fetch(`${API_BASE}/api/eval/results/${resultId}`)
      .then((r) => r.json())
      .then((d) => setResultDetailCache((c) => ({ ...c, [resultId]: d })))
      .catch((e) => setError(`result detail: ${e}`))
  }

  // Save / clear a human override for a single result.
  async function setHumanVerdict(resultId: string, verdict: string | null, reasoning?: string) {
    const resp = await fetch(`${API_BASE}/api/eval/results/${resultId}/verdict`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ verdict, reasoning }),
    })
    if (!resp.ok) {
      const t = await resp.text()
      setError(`override: ${resp.status} ${t.slice(0, 200)}`)
      return
    }
    // Re-fetch the result detail (effective_verdict + new fields), the
    // run row (recomputed aggregates), and the runs list (counters).
    const [d2, runD] = await Promise.all([
      fetch(`${API_BASE}/api/eval/results/${resultId}`).then((r) => r.json()),
      selectedRunId
        ? fetch(`${API_BASE}/api/eval/runs/${selectedRunId}`).then((r) => r.json())
        : Promise.resolve(null),
    ])
    setResultDetailCache((c) => ({ ...c, [resultId]: d2 }))
    if (runD) setRunDetail(runD)
    refreshRuns()
  }

  // Filter the question list by clicking a stage tile.
  const filteredResults = !runDetail
    ? []
    : !filter
    ? runDetail.results
    : runDetail.results.filter((r) => {
        if (filter.key === 'verdict') return (r.effective_verdict || r.judge_verdict || 'unable_to_verify') === filter.value
        if (filter.key === 'strategy') return r.strategy_executed === filter.value
        if (filter.key === 'routing_correct') return String(r.routing_correct) === filter.value
        if (filter.key === 'confidence') return r.confidence === filter.value
        return true
      })

  return (
    <div className="eval-tab eval-vertical">
      {error && <div className="eval-error">{error}</div>}

      {/* ── Toolbar (above everything): Run Eval + Calibrate + Edit Bank ── */}
      <div className="eval-toolbar">
        <button
          className="eval-toolbar-btn primary"
          onClick={triggerRun}
          disabled={!!active?.active || triggering}
          title={active?.active ? 'A run is in progress' : 'Trigger a new eval run on the current bank'}
        >
          {triggering ? '⏳ Starting…'
            : active?.active
            ? `⏳ Running ${active.n_completed ?? 0} / ${active.n_queries ?? '?'}`
            : '▶ Run Eval'}
        </button>
        <button
          className="eval-toolbar-btn"
          onClick={triggerCalibration}
          disabled={!!active?.active || triggering}
          title="Run all queries × all 4 strategies (a/b/c/d) for prior calibration. ~25 min."
        >
          🎯 Run Calibration
        </button>
        <input
          type="text"
          className="eval-notes-input"
          placeholder="run notes (e.g. 'after PA→prior auth fix')"
          value={runNotes}
          onChange={(e) => setRunNotes(e.target.value)}
          disabled={!!active?.active || triggering}
        />
        <button
          className={`eval-toolbar-btn ${bankOpen ? 'active' : ''}`}
          onClick={() => setBankOpen(!bankOpen)}
        >
          {bankOpen ? '× Close Bank Editor' : '✎ Edit Bank'}
        </button>
        <button
          className={`eval-toolbar-btn ${dashOpen ? 'active' : ''}`}
          onClick={() => setDashOpen((v) => !v)}
          title="Toggle the eval observability dashboard (timeline / A-B compare / drift)"
        >
          {dashOpen ? '× Hide Dashboard' : '📈 Dashboard'}
        </button>
        {bankOpen && bankDirty && (
          <button
            className="eval-toolbar-btn save"
            onClick={saveBank}
            disabled={bankSaving}
          >
            {bankSaving ? 'Saving…' : '💾 Save Bank'}
          </button>
        )}
      </div>

      {/* ── Bank editor (slides down) ── */}
      {bankOpen && (
        <BankEditor
          queries={bank}
          onUpdate={updateBankItem}
          onUpdateExpected={updateBankExpected}
          onAdd={addBankItem}
          onDelete={deleteBankItem}
          dirty={bankDirty}
        />
      )}

      {/* ── Runs list (left nav, collapsible) ── */}
      <div className={`eval-pane eval-pane-runs ${runsCollapsed ? 'collapsed' : ''}`}>
        <div className="eval-pane-header">
          {!runsCollapsed && <h3>Eval Runs</h3>}
          <button
            className="eval-pane-toggle"
            onClick={() => setRunsCollapsed(v => !v)}
            title={runsCollapsed ? 'Expand runs list' : 'Collapse runs list'}
            aria-label={runsCollapsed ? 'Expand runs list' : 'Collapse runs list'}
          >
            {runsCollapsed ? '›' : '‹'}
          </button>
        </div>
        {!runsCollapsed && runs.map((r) => {
          const total = r.n_queries || 0
          const correct = r.n_correct || 0
          const wrong = r.n_wrong || 0
          const unable = r.n_unable || 0
          const pct = total ? Math.round((correct / total) * 100) : 0
          return (
            <div
              key={r.id}
              className={`eval-run-row ${selectedRunId === r.id ? 'selected' : ''}`}
              onClick={() => setSelectedRunId(r.id)}
            >
              <div className="eval-run-ts">{new Date(r.ts).toLocaleString()}</div>
              <div className="eval-run-notes">{r.notes || '(no notes)'}</div>
              <div className="eval-run-meta">
                <span className="badge">{total}q</span>
                <span className="badge ok">{correct}✓</span>
                <span className="badge warn">{wrong}✗</span>
                <span className="badge dim">{unable}?</span>
                <span className="badge">{pct}%</span>
                {r.routing_accuracy !== null && (
                  <span className="badge">rt {Math.round(r.routing_accuracy * 100)}%</span>
                )}
                {r.median_latency_ms !== null && (
                  <span className="badge dim">{r.median_latency_ms}ms</span>
                )}
              </div>
            </div>
          )
        })}
        {!runsCollapsed && runs.length === 0 && <div className="eval-empty">No runs yet.</div>}
      </div>

      {/* ── Main content (vertical: header → stats → funnel → questions) ── */}
      <div className="eval-main">
        {dashOpen && <ObservabilityDashboard onPick={setSelectedRunId} />}
        {!runDetail && !loading && !dashOpen && <div className="eval-empty">Pick a run.</div>}
        {loading && <div className="eval-empty">Loading…</div>}
        {runDetail && (
          <>
            <RunHeader run={runDetail.run} />
            {calibSummary && <CalibrationSummaryPanel data={calibSummary} />}
            {gradeRollup && <TwoGradeRollupPanel data={gradeRollup} />}
            <PRCurvePanel runId={runDetail.run.id} />
            <PipelineFunnel
              results={runDetail.results}
              filter={filter}
              onFilter={setFilter}
            />
            <QuestionList
              results={filteredResults}
              expandedResultId={expandedResultId}
              cache={resultDetailCache}
              onToggle={toggleExpand}
              onHumanVerdict={setHumanVerdict}
              filter={filter}
              clearFilter={() => setFilter(null)}
              totalResults={runDetail.results.length}
            />
          </>
        )}
      </div>
    </div>
  )
}

// ── Observability dashboard (timeline / A-B compare / drift) ─────────────
// Consumes the invocation-agnostic fingerprint endpoints:
//   GET /api/eval/timeline   — version-annotated trend
//   GET /api/eval/compare    — A/B diff + confound guard
//   GET /api/eval/drift      — mean ± 2σ band (metric moved, fingerprint fixed)

interface FingerprintObj {
  agent_revision?: string | null
  agent_git_sha?: string | null
  priors_version?: string | null
  retrieval_config_hash?: string | null
  lexicon_revision?: number | null
  corpus_snapshot_at?: string | null
  judge_model?: string | null
  bank_hash?: string | null
  stable?: boolean | null
}
interface TimelineRun {
  run_id: string
  ts: string
  notes: string | null
  fingerprint: FingerprintObj | null
  n_queries: number
  router_recall: number | null
  router_composite: number | null
  oracle_recall: number | null
  routing_headroom: number | null
  // Per-strategy recall — only populated for calibration runs (forced
  // a/b/c/d/natural passes); a plain eval run only has "natural".
  strategy_recall?: Record<string, number | null>
}

// Strategy line palette — natural (router) gets the darkest/most prominent
// color since it's the actual system behavior; a/b/c/d are distinguishable
// accents. Reuses mobius-design compat aliases, no new hex.
const TL_STRATEGY_COLOR: Record<string, string> = {
  natural: 'var(--text-primary)',
  a: 'var(--rag-accent)',
  b: 'var(--accent-green)',
  c: 'var(--accent-red)',
  d: 'var(--accent-amber)',
}
const TL_STRATEGY_LABEL: Record<string, string> = {
  natural: 'router', a: 'a', b: 'b', c: 'c', d: 'd',
}

// Dims that, if any change between adjacent runs, mark a "version change".
const FP_SIG_DIMS: (keyof FingerprintObj)[] = [
  'agent_git_sha', 'priors_version', 'retrieval_config_hash',
  'lexicon_revision', 'judge_model', 'corpus_snapshot_at',
]
function fpSig(fp: FingerprintObj | null): string {
  if (!fp) return ''
  return FP_SIG_DIMS.map((d) => String(fp[d] ?? '')).join('|')
}
const fmt3 = (v: number | null | undefined) => (v === null || v === undefined ? '—' : Number(v).toFixed(3))
const shortRun = (r: TimelineRun) => (r.notes || r.run_id.slice(0, 8)) as string

function ObservabilityDashboard({ onPick }: { onPick: (id: string) => void }) {
  const [runs, setRuns] = useState<TimelineRun[] | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    fetch(`${API_BASE}/api/eval/timeline?limit=30`)
      .then((r) => r.json())
      .then((d) => setRuns(d.runs || []))
      .catch((e) => setErr(String(e)))
  }, [])

  if (err) return <div className="obs-dash"><div className="eval-error">timeline: {err}</div></div>
  if (!runs) return <div className="obs-dash"><div className="obs-muted">Loading dashboard…</div></div>
  if (runs.length === 0)
    return <div className="obs-dash"><div className="obs-muted">No completed runs with fingerprints yet — run a calibration to populate the timeline.</div></div>

  return (
    <div className="obs-dash">
      <div className="obs-title">📈 Eval Observability</div>
      <div className="obs-grid">
        <TimelinePanel runs={runs} onPick={onPick} />
        <div className="obs-side">
          <DriftChip />
          <ComparePanel runs={runs} onPick={onPick} />
        </div>
      </div>
    </div>
  )
}

type TimelineView = 'router_oracle' | 'strategies'
type SeriesKey = 'router_recall' | 'oracle_recall' | 'a' | 'b' | 'c' | 'd' | 'natural'

function seriesVal(r: TimelineRun, key: SeriesKey): number | null | undefined {
  if (key === 'router_recall' || key === 'oracle_recall') return r[key]
  return r.strategy_recall?.[key]
}

function TimelinePanel({ runs, onPick }: { runs: TimelineRun[]; onPick: (id: string) => void }) {
  const [view, setView] = useState<TimelineView>('router_oracle')
  const series = [...runs].reverse() // oldest → newest (left → right)
  const W = 520, H = 150, PL = 30, PR = 10, PT = 12, PB = 18
  const n = series.length
  const xs = (i: number) => (n <= 1 ? PL : PL + (i * (W - PL - PR)) / (n - 1))

  const keys: SeriesKey[] = view === 'strategies' ? ['a', 'b', 'c', 'd', 'natural'] : ['router_recall', 'oracle_recall']
  // Only offer strategy lines that actually have at least one data point —
  // calibration runs populate all 5; plain eval runs only ever have 'natural'.
  const activeKeys = keys.filter((k) => series.some((r) => seriesVal(r, k) != null))
  const vmax = Math.max(0.6, ...series.flatMap((r) => activeKeys.map((k) => seriesVal(r, k) ?? 0)))
  const ys = (v: number) => PT + (1 - v / vmax) * (H - PT - PB)
  const linePath = (key: SeriesKey) =>
    series
      .map((r, i) => {
        const v = seriesVal(r, key)
        if (v === null || v === undefined) return null
        return `${i === 0 ? 'M' : 'L'}${xs(i).toFixed(1)},${ys(v).toFixed(1)}`
      })
      .filter(Boolean)
      .join(' ')
  const changes: number[] = []
  for (let i = 1; i < n; i++) if (fpSig(series[i].fingerprint) !== fpSig(series[i - 1].fingerprint)) changes.push(i)
  const gridlines = [0.2, 0.4, 0.6, 0.8].filter((g) => g <= vmax)
  const latest = series[n - 1]

  return (
    <div className="obs-panel obs-timeline">
      <div className="obs-panel-head">
        <span>{view === 'strategies' ? 'Strategy recall over time' : 'Router recall over time'}</span>
        <select
          className="obs-select obs-view-select"
          value={view}
          onChange={(e) => setView(e.target.value as TimelineView)}
          title="Choose what to trend"
        >
          <option value="router_oracle">Router vs oracle</option>
          <option value="strategies">All strategies (a/b/c/d/router)</option>
        </select>
        <span className="obs-legend">
          {view === 'strategies' ? (
            activeKeys.map((k) => (
              <span key={k} className="obs-legend-item">
                <i className="obs-swatch" style={{ background: TL_STRATEGY_COLOR[k] }} /> {TL_STRATEGY_LABEL[k]}
              </span>
            ))
          ) : (
            <>
              <i className="obs-swatch recall" /> router
              <i className="obs-swatch oracle" /> oracle ceiling
            </>
          )}
          <i className="obs-swatch tick" /> version change
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="obs-svg" preserveAspectRatio="xMidYMid meet" role="img"
           aria-label={view === 'strategies' ? 'Per-strategy recall over eval runs' : 'Router recall over eval runs'}>
        {gridlines.map((g) => (
          <g key={g}>
            <line x1={PL} x2={W - PR} y1={ys(g)} y2={ys(g)} className="obs-grid" />
            <text x={2} y={ys(g) + 3} className="obs-axis">{g.toFixed(1)}</text>
          </g>
        ))}
        {changes.map((i) => (
          <line key={`chg${i}`} x1={xs(i)} x2={xs(i)} y1={PT} y2={H - PB} className="obs-verline">
            <title>version change before {shortRun(series[i])}</title>
          </line>
        ))}
        {view === 'strategies' ? (
          activeKeys.map((k) => (
            <path key={k} d={linePath(k)} fill="none" style={{ stroke: TL_STRATEGY_COLOR[k], strokeWidth: k === 'natural' ? 2.4 : 1.6 }} />
          ))
        ) : (
          <>
            <path d={linePath('oracle_recall')} className="obs-path oracle" fill="none" />
            <path d={linePath('router_recall')} className="obs-path recall" fill="none" />
          </>
        )}
        {view === 'strategies'
          ? activeKeys.flatMap((k) =>
              series.map((r, i) => {
                const v = seriesVal(r, k)
                if (v == null) return null
                return (
                  <circle
                    key={`${k}-${r.run_id}`}
                    cx={xs(i)} cy={ys(v)} r={2.6}
                    style={{ fill: TL_STRATEGY_COLOR[k], stroke: '#fff', strokeWidth: 1, cursor: 'pointer' }}
                    onClick={() => onPick(r.run_id)}
                  >
                    <title>
                      {new Date(r.ts).toLocaleString()} · {r.notes || ''}
                      {'\n'}{TL_STRATEGY_LABEL[k]} recall {fmt3(v)}
                    </title>
                  </circle>
                )
              }),
            )
          : series.map((r, i) =>
              r.router_recall == null ? null : (
                <circle
                  key={r.run_id}
                  cx={xs(i)}
                  cy={ys(r.router_recall)}
                  r={3.2}
                  className={`obs-pt${changes.includes(i) ? ' chg' : ''}`}
                  onClick={() => onPick(r.run_id)}
                >
                  <title>
                    {new Date(r.ts).toLocaleString()} · {r.notes || ''}
                    {'\n'}router {fmt3(r.router_recall)} · oracle {fmt3(r.oracle_recall)} · composite {fmt3(r.router_composite)}
                  </title>
                </circle>
              ),
            )}
      </svg>
      <div className="obs-latest">
        latest <b>{fmt3(latest.router_recall)}</b> recall · {fmt3(latest.router_composite)} composite ·
        headroom {fmt3(latest.routing_headroom)}
        <span className="obs-muted"> · {shortRun(latest)}</span>
      </div>
    </div>
  )
}

interface CompareResp {
  fingerprint_diff: Record<string, { a: unknown; b: unknown }>
  confounded: boolean
  confound_reason: string | null
  deltas: Record<string, number | null>
  query_movers: { query: string; a: number; b: number; delta: number }[]
  detail?: string
}

function ComparePanel({ runs, onPick: _onPick }: { runs: TimelineRun[]; onPick: (id: string) => void }) {
  const [aId, setAId] = useState<string>(runs[1]?.run_id || runs[0]?.run_id || '')
  const [bId, setBId] = useState<string>(runs[0]?.run_id || '')
  const [data, setData] = useState<CompareResp | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (!aId || !bId || aId === bId) { setData(null); return }
    setBusy(true)
    fetch(`${API_BASE}/api/eval/compare?a=${aId}&b=${bId}`)
      .then((r) => r.json())
      .then((d) => setData(d.detail ? null : d))
      .catch(() => setData(null))
      .finally(() => setBusy(false))
  }, [aId, bId])

  const opt = (r: TimelineRun) => (
    <option key={r.run_id} value={r.run_id}>
      {new Date(r.ts).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })} · {shortRun(r).slice(0, 22)}
    </option>
  )
  const delta = (k: string, label: string) => {
    const v = data?.deltas?.[k]
    const cls = v == null ? '' : v > 0 ? 'up' : v < 0 ? 'down' : ''
    return (
      <div className="obs-delta">
        <span className="obs-delta-label">{label}</span>
        <span className={`obs-delta-val ${cls}`}>{v == null ? '—' : (v > 0 ? '+' : '') + v.toFixed(3)}</span>
      </div>
    )
  }
  const fpKeys = data ? Object.keys(data.fingerprint_diff || {}) : []

  return (
    <div className="obs-panel">
      <div className="obs-panel-head"><span>A / B compare</span></div>
      <div className="obs-cmp-selects">
        <select className="obs-select" value={aId} onChange={(e) => setAId(e.target.value)}>{runs.map(opt)}</select>
        <span className="obs-arrow">→</span>
        <select className="obs-select" value={bId} onChange={(e) => setBId(e.target.value)}>{runs.map(opt)}</select>
      </div>
      {aId === bId && <div className="obs-muted">Pick two different runs.</div>}
      {busy && <div className="obs-muted">Comparing…</div>}
      {data && !busy && (
        <>
          {data.confounded ? (
            <div className="obs-banner warn">⚠ {data.confound_reason}</div>
          ) : fpKeys.length === 0 ? (
            <div className="obs-banner ok">✓ same build — delta is noise / measurement only</div>
          ) : (
            <div className="obs-banner attrib">✓ attributable — 1 dim changed: {fpKeys.join(', ')}</div>
          )}
          <div className="obs-deltas">
            {delta('router_recall', 'recall')}
            {delta('router_composite', 'composite')}
            {delta('oracle_recall', 'oracle')}
            {delta('routing_headroom', 'headroom')}
          </div>
          {fpKeys.length > 0 && (
            <div className="obs-fpdiff">
              {Object.entries(data.fingerprint_diff).map(([k, v]) => (
                <div key={k} className="obs-fprow">
                  <span className="obs-fpk">{k}</span>
                  <span className="obs-fpv">{String(v.a)} → {String(v.b)}</span>
                </div>
              ))}
            </div>
          )}
          {data.query_movers?.length > 0 && (
            <div className="obs-movers">
              <div className="obs-movers-head">top query movers (router recall)</div>
              {data.query_movers.slice(0, 6).map((m) => (
                <div key={m.query} className="obs-mover">
                  <span className="obs-mq" title={m.query}>{m.query}</span>
                  <span className={`obs-md ${m.delta > 0 ? 'up' : 'down'}`}>{m.delta > 0 ? '+' : ''}{m.delta.toFixed(3)}</span>
                  <span className="obs-muted">{m.a.toFixed(2)}→{m.b.toFixed(2)}</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}

function DriftChip() {
  const [d, setD] = useState<any>(null)
  useEffect(() => {
    fetch(`${API_BASE}/api/eval/drift?metric=router_recall`)
      .then((r) => r.json())
      .then(setD)
      .catch(() => setD(null))
  }, [])
  if (!d) return null
  if (d.status === 'insufficient_data')
    return <div className="obs-chip neutral" title={d.detail}>drift: not enough same-config runs for a band</div>
  const cls = d.status === 'DRIFT' ? 'bad' : 'good'
  return (
    <div className={`obs-chip ${cls}`} title={`fingerprint group: ${JSON.stringify(d.fingerprint_group)}`}>
      drift ({d.metric}): {d.status === 'DRIFT' ? '⚠ DRIFT' : '✓ in band'} · μ {d.mean} ±2σ [{d.band_2sigma?.[0]}, {d.band_2sigma?.[1]}] · latest {d.latest?.value} · n={d.n_runs}
    </div>
  )
}

// ── Calibration summary (5-axis + oracle) ───────────────────────────────

// ── Two-grade QA Rollup (Panel B) ───────────────────────────────────────
//
// Per-strategy mean ± std for retrieval_grade / synthesis_grade / gap.
// Read from GET /api/eval/runs/{id}/grade_rollup (rag_query_decisions rows).
// Sigma band: delta < sigma_noise (~0.2) → gray out as within noise floor.

function TwoGradeRollupPanel({ data }: { data: any }) {
  const order = ['a', 'b', 'c', 'd', 'natural']
  const strats = order.filter((s) => data.strategies?.[s])
  const sigma = data.sigma_noise ?? 0.2
  const fmt = (v: any) => (v === null || v === undefined ? '—' : Number(v).toFixed(3))
  const label = (s: string) => (s === 'natural' ? 'router' : s)

  // Gap display: negative = dropped facts (synthesizer bottleneck, amber),
  // positive = hallucination bluff (red). Gray if within sigma noise.
  function GapCell({ mean, std }: { mean: number | null; std: number | null }) {
    if (mean === null || mean === undefined) return <td style={{ textAlign: 'right', opacity: 0.4 }}>—</td>
    const noisy = (std ?? 0) >= sigma || Math.abs(mean) < sigma
    const color = noisy ? 'var(--text-muted, #888)' : mean < 0 ? '#d97706' : '#dc2626'
    const title = noisy
      ? `Within noise floor (σ≈${sigma})`
      : mean < 0
      ? 'Synthesizer bottleneck — facts retrieved but dropped in answer'
      : 'Hallucination bluff — answer claims facts not in chunks'
    return (
      <td style={{ textAlign: 'right', fontWeight: 600, color }} title={title}>
        {mean > 0 ? '+' : ''}{fmt(mean)}
        {noisy && <span style={{ fontSize: 10, opacity: 0.6 }}>~</span>}
      </td>
    )
  }

  function MetricCell({ mean, std }: { mean: number | null; std: number | null }) {
    if (mean === null || mean === undefined) return <td style={{ textAlign: 'right', opacity: 0.4 }}>—</td>
    const noisy = (std ?? 0) >= sigma
    return (
      <td style={{ textAlign: 'right', opacity: noisy ? 0.55 : 1 }} title={noisy ? `High variance (σ=${fmt(std)}) — treat as noisy` : undefined}>
        {fmt(mean)}
        {std !== null && std !== undefined && (
          <span style={{ fontSize: 10, opacity: 0.55, marginLeft: 2 }}>±{fmt(std)}</span>
        )}
      </td>
    )
  }

  if (strats.length === 0) return null
  return (
    <div style={{ border: '1px solid var(--border, #333)', borderRadius: 6, padding: 12, margin: '8px 0' }}>
      <div style={{ fontWeight: 600, marginBottom: 6 }}>
        🧮 Two-Grade QA Rollup — retrieval vs synthesis (mean ± std)
        <span style={{ fontSize: 11, fontWeight: 400, marginLeft: 8, opacity: 0.6 }}>
          |gap| &lt; σ{sigma} shown dimmed — within noise floor
        </span>
      </div>
      <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ textAlign: 'right', opacity: 0.7 }}>
            <th style={{ textAlign: 'left' }}>strategy</th>
            <th>retrieval grade</th>
            <th>synthesis grade</th>
            <th>gap (syn − ret)</th>
            <th>n</th>
          </tr>
        </thead>
        <tbody>
          {strats.map((s) => {
            const c = data.strategies[s]
            return (
              <tr key={s} style={{ textAlign: 'right', borderTop: '1px solid var(--border, #2a2a2a)' }}>
                <td style={{ textAlign: 'left', fontWeight: 600 }}>{label(s)}</td>
                <MetricCell mean={c.retrieval_mean} std={c.retrieval_std} />
                <MetricCell mean={c.synthesis_mean} std={c.synthesis_std} />
                <GapCell mean={c.gap_mean} std={c.gap_std} />
                <td style={{ textAlign: 'right', opacity: 0.6 }}>{c.n}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ── GapBadge — inline row badge for QuestionList ────────────────────────

function GapBadge({ gap, synth }: { gap: number | null; synth: number }) {
  const sigma = 0.2
  const noisy = gap === null || Math.abs(gap) < sigma
  const color = noisy ? undefined : gap! < 0 ? '#d97706' : '#dc2626'
  const title = gap === null
    ? 'No synthesis (no answer generated)'
    : noisy
    ? 'Within noise floor'
    : gap < 0
    ? 'Synthesizer dropped facts'
    : 'Answer bluffs facts not in chunks'
  return (
    <span className="badge mini" style={{ color }} title={title}>
      syn {synth.toFixed(2)}
      {gap !== null && (
        <span style={{ opacity: 0.7, marginLeft: 2 }}>
          ({gap >= 0 ? '+' : ''}{gap.toFixed(2)})
        </span>
      )}
    </span>
  )
}

// ── TwoGradeBar — inline drilldown header ────────────────────────────────

export function TwoGradeBar({
  retrieval, synthesis, gap,
}: { retrieval: number | null; synthesis: number | null; gap: number | null }) {
  const sigma = 0.2
  const fmt2 = (v: number) => v.toFixed(3)
  const gapColor = gap === null || Math.abs(gap) < sigma
    ? 'var(--text-muted, #888)'
    : gap < 0 ? '#d97706' : '#dc2626'
  const gapLabel = gap === null
    ? 'n/a'
    : gap < 0
    ? 'synthesizer bottleneck'
    : 'hallucination bluff'
  return (
    <div style={{
      display: 'flex', gap: 12, padding: '8px 12px', margin: '6px 0',
      background: 'var(--surface-alt, #f8f9fb)', borderRadius: 6,
      border: '1px solid var(--border, #dde)', fontSize: 13,
    }}>
      <div>
        <span style={{ opacity: 0.6, marginRight: 4 }}>retrieval</span>
        <strong>{retrieval !== null ? fmt2(retrieval) : '—'}</strong>
      </div>
      <div style={{ opacity: 0.4 }}>→</div>
      <div>
        <span style={{ opacity: 0.6, marginRight: 4 }}>synthesis</span>
        <strong>{synthesis !== null ? fmt2(synthesis) : '—'}</strong>
      </div>
      {gap !== null && (
        <>
          <div style={{ opacity: 0.4 }}>=</div>
          <div>
            <span style={{ opacity: 0.6, marginRight: 4 }}>gap</span>
            <strong style={{ color: gapColor }}>
              {gap >= 0 ? '+' : ''}{fmt2(gap)}
            </strong>
            <span style={{ fontSize: 11, opacity: 0.6, marginLeft: 6 }}>({gapLabel})</span>
          </div>
        </>
      )}
    </div>
  )
}

// ── PerClaimLedger — per-claim fact-check evidence ───────────────────────
//
// Renders the per_claim_ledger JSONB field from rag_query_decisions.
// Layout: contradicted always prominent; validated collapsible; raw floats
// behind a toggle.

export function PerClaimLedger({ claims }: { claims: ClaimEntry[] }) {
  const [showValidated, setShowValidated] = useState(false)

  const contradicted = claims.filter((c) => c.status === 'contradicted')
  const unvalidated  = claims.filter((c) => c.status === 'unvalidated')
  const validated    = claims.filter((c) => c.status === 'validated')

  const STATUS_STYLE: Record<string, React.CSSProperties> = {
    contradicted: { color: '#dc2626', fontWeight: 600 },
    unvalidated:  { color: '#92400e', opacity: 0.8 },
    validated:    { color: '#16a34a' },
  }

  function ClaimRow({ claim }: { claim: ClaimEntry }) {
    const [open, setOpen] = useState(false)
    return (
      <div
        style={{
          padding: '6px 10px', borderRadius: 4, marginBottom: 3,
          background: claim.status === 'contradicted'
            ? 'rgba(220,38,38,0.06)'
            : claim.status === 'validated'
            ? 'rgba(22,163,74,0.05)'
            : 'rgba(0,0,0,0.03)',
          border: '1px solid',
          borderColor: claim.status === 'contradicted'
            ? 'rgba(220,38,38,0.25)'
            : claim.status === 'validated'
            ? 'rgba(22,163,74,0.2)'
            : 'var(--border, #e5e7eb)',
        }}
      >
        <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
          <span style={{ ...STATUS_STYLE[claim.status], fontSize: 11, flexShrink: 0, marginTop: 1 }}>
            {claim.status === 'contradicted' ? '✗' : claim.status === 'validated' ? '✓' : '?'}
          </span>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 13 }}>{claim.fact}</div>
            {claim.quote && (
              <div
                style={{
                  fontSize: 11, opacity: 0.7, marginTop: 3,
                  fontStyle: 'italic', cursor: 'pointer',
                }}
                onClick={() => setOpen((v) => !v)}
              >
                {open ? '▾' : '▸'} {open ? claim.quote : `"${claim.quote.slice(0, 80)}…"`}
              </div>
            )}
            {claim.chunk_id && (
              <div style={{ fontSize: 10, opacity: 0.5, marginTop: 2 }}>
                chunk {claim.chunk_id}
              </div>
            )}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div style={{ margin: '8px 0', fontSize: 13 }}>
      <div style={{ fontWeight: 600, marginBottom: 6, display: 'flex', alignItems: 'center', gap: 8 }}>
        📋 Per-claim ledger
        <span style={{ fontSize: 11, fontWeight: 400, opacity: 0.6 }}>
          {contradicted.length > 0 && <span style={{ color: '#dc2626', marginRight: 6 }}>{contradicted.length} contradicted</span>}
          {unvalidated.length > 0 && <span style={{ color: '#92400e', marginRight: 6 }}>{unvalidated.length} unvalidated</span>}
          {validated.length > 0 && <span style={{ color: '#16a34a' }}>{validated.length} validated</span>}
        </span>
      </div>

      {contradicted.map((c, i) => <ClaimRow key={`contra-${i}`} claim={c} />)}
      {unvalidated.map((c, i) => <ClaimRow key={`unv-${i}`} claim={c} />)}

      {validated.length > 0 && (
        <div>
          <button
            onClick={() => setShowValidated((v) => !v)}
            style={{
              background: 'none', border: 'none', cursor: 'pointer',
              fontSize: 12, color: '#16a34a', padding: '4px 0',
            }}
          >
            {showValidated ? '▾ hide validated' : `▸ show ${validated.length} validated claims`}
          </button>
          {showValidated && validated.map((c, i) => <ClaimRow key={`val-${i}`} claim={c} />)}
        </div>
      )}
    </div>
  )
}

function CalibrationSummaryPanel({ data }: { data: any }) {
  const order = ['a', 'b', 'c', 'd', 'natural']
  const strats = order.filter((s) => data.strategies?.[s])
  const label = (s: string) => (s === 'natural' ? 'router' : s)
  const fmt = (v: any, d = 2) => (v === null || v === undefined ? '—' : Number(v).toFixed(d))
  const ms = (v: any) => (v === null || v === undefined ? '—' : `${Math.round(v)}ms`)
  return (
    <div style={{ border: '1px solid var(--border, #333)', borderRadius: 6, padding: 12, margin: '8px 0' }}>
      <div style={{ fontWeight: 600, marginBottom: 6 }}>
        🎯 Retrieval Calibration — 5-axis (chunk-only recall)
      </div>
      <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ textAlign: 'right', opacity: 0.7 }}>
            <th style={{ textAlign: 'left' }}>strategy</th>
            <th>answer-rate</th><th>recall</th><th>precision</th><th>contra/cell</th>
            <th>median lat</th><th>p95 lat</th><th>n</th>
          </tr>
        </thead>
        <tbody>
          {strats.map((s) => {
            const c = data.strategies[s]
            return (
              <tr key={s} style={{ textAlign: 'right', borderTop: '1px solid var(--border, #2a2a2a)' }}>
                <td style={{ textAlign: 'left', fontWeight: 600 }}>{label(s)}</td>
                <td>{fmt(c.answer_rate)}</td>
                <td style={{ fontWeight: 600 }}>{fmt(c.recall)}</td>
                <td>{fmt(c.precision)}</td>
                <td style={{ color: c.contra_per_cell > 0 ? 'var(--warn, #d88)' : undefined }}>{fmt(c.contra_per_cell)}</td>
                <td>{ms(c.median_latency_ms)}</td>
                <td>{ms(c.p95_latency_ms)}</td>
                <td style={{ opacity: 0.6 }}>{c.n}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <div style={{ marginTop: 8, fontSize: 13 }}>
        <b>Oracle recall</b> {fmt(data.oracle_recall)} &nbsp;·&nbsp;
        router {fmt(data.router_recall)} &nbsp;·&nbsp;
        best single {fmt(data.best_single_recall)} &nbsp;·&nbsp;
        <b>routing headroom {fmt(data.routing_headroom)}</b> (oracle − router)
        <span style={{ opacity: 0.6 }}> &nbsp;({data.n_queries} queries)</span>
      </div>
    </div>
  )
}

// ── Run header ──────────────────────────────────────────────────────────

function RunHeader({ run }: { run: EvalRunRow }) {
  const total = run.n_queries || 0
  const correct = run.n_correct || 0
  const partial = run.n_partial || 0
  const wrong = run.n_wrong || 0
  const unable = run.n_unable || 0
  return (
    <div className="run-header">
      <div className="run-title">
        <h2>{run.notes || '(no notes)'}</h2>
        <div className="run-meta">
          <span className="dim-text">{new Date(run.ts).toLocaleString()}</span>
          <span className="dim-text">· bank {run.bank_path}@{run.bank_version || '?'}</span>
          <span className="dim-text">· priors {run.priors_version}</span>
        </div>
      </div>
      <div className="kpi-grid">
        <Kpi label="queries" value={total} />
        <Kpi label="correct" value={`${correct} (${pct(correct, total)}%)`} kind="ok" />
        <Kpi label="partial" value={partial} kind="warn" />
        <Kpi label="wrong" value={wrong} kind="bad" />
        <Kpi label="unable" value={unable} kind="dim" />
        <Kpi
          label="routing acc"
          value={run.routing_accuracy !== null ? `${Math.round(run.routing_accuracy * 100)}%` : '—'}
        />
        <Kpi
          label="cite hit"
          value={run.citation_hit_rate !== null ? `${Math.round(run.citation_hit_rate * 100)}%` : '—'}
        />
        <Kpi label="p50" value={run.median_latency_ms ? `${run.median_latency_ms}ms` : '—'} />
        <Kpi label="p95" value={run.p95_latency_ms ? `${run.p95_latency_ms}ms` : '—'} kind="dim" />
      </div>
    </div>
  )
}

function Kpi({ label, value, kind = 'neutral' }: { label: string; value: string | number; kind?: string }) {
  return (
    <div className={`kpi kpi-${kind}`}>
      <div className="kpi-value">{value}</div>
      <div className="kpi-label">{label}</div>
    </div>
  )
}

function pct(num: number, total: number) {
  return total ? Math.round((num / total) * 100) : 0
}

// ── PR-curve panel ──────────────────────────────────────────────────────
//
// Sweeps a confidence threshold τ and shows precision-recall per strategy,
// overlaid. Lets us see "strategy (a) hits 95% precision at τ=0.62 with
// 41% recall" — the operating-point story for each retrieval mode.
//
// Live-poll: if the run is still in progress (results streaming in), this
// panel refreshes every 4s so the curves fill in as data lands.

type PRPoint = {
  tau: number
  precision: number | null
  recall: number | null
  n_answered: number
  n_correct: number
  n_partial: number
  n_wrong: number
  n_total: number
}

type PRPayload = {
  run_id: string
  axis: string
  n_steps: number
  n_strategies: number
  n_total_per_strategy: Record<string, number>
  points: Record<string, PRPoint[]>
}

const STRATEGY_COLOR: Record<string, string> = {
  a: '#2563eb', // blue
  b: '#16a34a', // green
  c: '#9333ea', // purple
  d: '#ea580c', // orange
  e: '#6b7280', // gray
  // System curves: the end-to-end pipeline view.
  system_oracle: '#0f172a',  // near-black — bank's expected pick per query
  system_best:   '#dc2626',  // red — hindsight ceiling, the upper bound
}

const STRATEGY_LABEL: Record<string, string> = {
  system_oracle: 'system (oracle)',
  system_best:   'system (best)',
}

const SYSTEM_CURVE_KEYS = new Set(['system_oracle', 'system_best'])

function PRCurvePanel({ runId }: { runId: string }) {
  const [axis, setAxis] = useState<'top_rerank' | 'mean_top3' | 'confidence_tier'>('confidence_tier')
  const [data, setData] = useState<PRPayload | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hover, setHover] = useState<{ strategy: string; idx: number } | null>(null)

  // Fetch on runId / axis change. Also poll every 4s while the run looks
  // active (any strategy has < n_total decided rows).
  useEffect(() => {
    let alive = true
    const load = () => {
      setLoading(true)
      fetch(`${API_BASE}/api/eval/runs/${runId}/pr_curve?axis=${axis}`)
        .then(r => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))
        .then((d: PRPayload) => { if (alive) { setData(d); setError(null) } })
        .catch(e => { if (alive) setError(String(e)) })
        .finally(() => { if (alive) setLoading(false) })
    }
    load()
    const id = setInterval(load, 4000)
    return () => { alive = false; clearInterval(id) }
  }, [runId, axis])

  if (error) return <div className="pr-panel pr-error">PR curve: {error}</div>
  if (!data && loading) return <div className="pr-panel">Loading PR curve…</div>
  if (!data) return null

  // Detect degenerate top_rerank data: when rerank_score is null for most
  // rows, conf is treated as 0 — all rows only appear at τ=0, causing
  // precision and recall to both increase together (not the expected tradeoff).
  const isRerankDegenerate = axis === 'top_rerank' && (() => {
    let total = 0, nullConf = 0
    for (const pts of Object.values(data.points)) {
      for (const p of pts) {
        total++
        // At τ>0, a row only appears if its rerank score ≥ τ.
        // If nearly all non-τ=0 points have null precision, the data is degenerate.
        if (p.tau > 0 && p.precision === null) nullConf++
      }
    }
    return total > 0 && nullConf / total > 0.7
  })()

  // Sort strategies so system curves render LAST (on top of others)
  // and group nicely in the legend.
  const strategies = Object.keys(data.points).sort((a, b) => {
    const aSys = SYSTEM_CURVE_KEYS.has(a) ? 1 : 0
    const bSys = SYSTEM_CURVE_KEYS.has(b) ? 1 : 0
    if (aSys !== bSys) return aSys - bSys
    return a.localeCompare(b)
  })

  // Operating-point markers: find smallest τ where P ≥ target on each curve.
  const targets = [0.99, 0.95, 0.9]

  // SVG geometry
  const W = 720, H = 360, M = { l: 50, r: 16, t: 16, b: 36 }
  const plotW = W - M.l - M.r, plotH = H - M.t - M.b
  const xR = (r: number) => M.l + r * plotW
  const yP = (p: number) => M.t + (1 - p) * plotH

  return (
    <div className="pr-panel">
      <div className="pr-header">
        <div className="pr-title">
          <strong>Precision-Recall by strategy</strong>
          <span className="dim-text">
            {' · '}{data.n_strategies} strategies · axis = {data.axis}
          </span>
        </div>
        <div className="pr-controls">
          <label className="pr-axis-label">axis:</label>
          <select
            value={axis}
            onChange={e => setAxis(e.target.value as typeof axis)}
            className="pr-axis-select"
          >
            <option value="top_rerank">top rerank score</option>
            <option value="mean_top3">mean top-3 score</option>
            <option value="confidence_tier">confidence tier</option>
          </select>
        </div>
      </div>

      {isRerankDegenerate && (
        <div className="pr-warn">
          ⚠ Most rows have null rerank scores — the top_rerank axis will show precision
          and recall rising together (not the expected tradeoff). Switch to{' '}
          <button className="pr-warn-link" onClick={() => setAxis('confidence_tier')}>
            confidence tier
          </button>{' '}
          for a meaningful curve. The null rerank bug is a separate fix in the RAG service.
        </div>
      )}

      <div className="pr-chart-row">
        <svg width={W} height={H} className="pr-svg">
          {/* gridlines */}
          {[0, 0.25, 0.5, 0.75, 1.0].map(g => (
            <g key={`g${g}`}>
              <line x1={xR(g)} y1={M.t} x2={xR(g)} y2={M.t + plotH}
                stroke="#e5e7eb" strokeWidth={1} />
              <line x1={M.l} y1={yP(g)} x2={M.l + plotW} y2={yP(g)}
                stroke="#e5e7eb" strokeWidth={1} />
              <text x={xR(g)} y={M.t + plotH + 16} fontSize={11} fill="#6b7280" textAnchor="middle">
                {g.toFixed(2)}
              </text>
              <text x={M.l - 6} y={yP(g) + 4} fontSize={11} fill="#6b7280" textAnchor="end">
                {g.toFixed(2)}
              </text>
            </g>
          ))}
          {/* axis labels */}
          <text x={M.l + plotW / 2} y={H - 6} fontSize={12} fill="#374151" textAnchor="middle">
            recall
          </text>
          <text x={14} y={M.t + plotH / 2} fontSize={12} fill="#374151"
            textAnchor="middle" transform={`rotate(-90 14 ${M.t + plotH / 2})`}>
            precision
          </text>

          {/* per-strategy curves */}
          {strategies.map(s => {
            const pts = data.points[s] || []
            const filtered = pts.filter(p =>
              p.precision !== null && p.recall !== null &&
              // Drop degenerate anchor point: strategy answered but every
              // answer was wrong (precision=0, recall=0). Including it drags
              // the curve to the origin and creates a misleading upward slope.
              !(p.precision === 0 && p.recall === 0)
            )
            // Sort by τ descending: high τ (high precision, low recall) on the
            // left → low τ (lower precision, high recall) on the right. This
            // is the canonical PR curve direction and avoids zig-zags that
            // arise from sorting by recall when the data is non-monotonic.
            const ordered = [...filtered].sort((a, b) => b.tau - a.tau)
            const path = ordered.map((p, i) =>
              `${i === 0 ? 'M' : 'L'} ${xR(p.recall ?? 0)} ${yP(p.precision ?? 0)}`).join(' ')
            const color = STRATEGY_COLOR[s] || '#374151'
            const isSystem = SYSTEM_CURVE_KEYS.has(s)
            const strokeWidth = isSystem ? 3 : 2
            const dash = s === 'system_best' ? '6 4' : undefined
            return (
              <g key={s}>
                <path
                  d={path}
                  fill="none"
                  stroke={color}
                  strokeWidth={strokeWidth}
                  strokeDasharray={dash}
                />
                {pts.map((p, i) => p.precision !== null && p.recall !== null && (
                  <circle
                    key={i}
                    cx={xR(p.recall)}
                    cy={yP(p.precision)}
                    r={hover?.strategy === s && hover.idx === i ? 5 : 2.5}
                    fill={color}
                    onMouseEnter={() => setHover({ strategy: s, idx: i })}
                    onMouseLeave={() => setHover(null)}
                    style={{ cursor: 'crosshair' }}
                  />
                ))}
              </g>
            )
          })}

          {/* hover tooltip */}
          {hover && data.points[hover.strategy]?.[hover.idx] && (() => {
            const p = data.points[hover.strategy][hover.idx]
            if (p.precision === null || p.recall === null) return null
            const tx = xR(p.recall) + 8
            const ty = yP(p.precision) - 8
            return (
              <g>
                <rect x={tx} y={ty - 56} width={170} height={64}
                  fill="#111827" rx={4} opacity={0.95} />
                <text x={tx + 8} y={ty - 40} fontSize={11} fill="#fff">
                  ({hover.strategy}) τ={p.tau.toFixed(2)}
                </text>
                <text x={tx + 8} y={ty - 26} fontSize={11} fill="#fff">
                  P={p.precision.toFixed(2)} R={p.recall.toFixed(2)}
                </text>
                <text x={tx + 8} y={ty - 12} fontSize={11} fill="#9ca3af">
                  ans={p.n_answered} ✓{p.n_correct} ½{p.n_partial} ✗{p.n_wrong}/{p.n_total}
                </text>
              </g>
            )
          })()}
        </svg>

        <div className="pr-legend">
          <div className="pr-legend-title">strategies</div>
          {strategies.map(s => {
            // Tally at τ=0 — "how many of the N did this strategy
            // actually get right?" Plain integers beat decimals for
            // intuition.
            const points = data.points[s] || []
            const at0 = points.find(p => p.tau === 0) || points[0]
            const total = data.n_total_per_strategy[s] || 0
            return (
              <div key={s} className="pr-legend-row">
                <span className="pr-swatch" style={{ background: STRATEGY_COLOR[s] || '#374151' }} />
                <span className="pr-legend-label">{STRATEGY_LABEL[s] || `(${s})`}</span>
                {at0 ? (
                  <span className="dim-text">
                    {at0.n_correct}✓ {at0.n_partial}½ {at0.n_wrong}✗ / {total}
                  </span>
                ) : (
                  <span className="dim-text">n={total}</span>
                )}
              </div>
            )
          })}
          <div className="pr-legend-title" style={{ marginTop: 12 }}>operating points</div>
          {targets.map(t => (
            <div key={t} className="pr-op-block">
              <div className="pr-op-target">P ≥ {(t * 100).toFixed(0)}%</div>
              {strategies.map(s => {
                const ordered = [...(data.points[s] || [])]
                  .filter(p => p.precision !== null && p.recall !== null)
                  // walk from low τ → high τ; first point hitting target wins
                  // (highest recall at this precision floor)
                  .sort((a, b) => a.tau - b.tau)
                const hit = ordered.find(p => (p.precision ?? 0) >= t)
                return (
                  <div key={s} className="pr-op-row">
                    <span className="pr-swatch" style={{ background: STRATEGY_COLOR[s] || '#374151' }} />
                    <span className="pr-op-strategy">{STRATEGY_LABEL[s] || `(${s})`}</span>
                    {hit ? (
                      <span className="pr-op-detail">
                        τ={hit.tau.toFixed(2)} R={(hit.recall ?? 0).toFixed(2)}
                      </span>
                    ) : (
                      <span className="dim-text">never</span>
                    )}
                  </div>
                )
              })}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Pipeline funnel ────────────────────────────────────────────────────

function PipelineFunnel({
  results,
  filter,
  onFilter,
}: {
  results: EvalResultRow[]
  filter: { key: string; value: string } | null
  onFilter: (f: { key: string; value: string } | null) => void
}) {
  const total = results.length

  // Roll up per-stage counts from the result rows.
  const verdicts: Record<string, number> = {}
  const strategies: Record<string, number> = {}
  const confidences: Record<string, number> = {}
  let routingCorrect = 0
  let routingWrong = 0
  let citationHit = 0
  let citationMiss = 0

  for (const r of results) {
    const effV = r.effective_verdict || r.judge_verdict || 'unable_to_verify'
    verdicts[effV] = (verdicts[effV] || 0) + 1
    if (r.strategy_executed) strategies[r.strategy_executed] = (strategies[r.strategy_executed] || 0) + 1
    if (r.confidence) confidences[r.confidence] = (confidences[r.confidence] || 0) + 1
    if (r.routing_correct === true) routingCorrect++
    else if (r.routing_correct === false) routingWrong++
    if (r.citation_hit === true) citationHit++
    else if (r.citation_hit === false) citationMiss++
  }

  function isFiltered(key: string, value: string) {
    return filter?.key === key && filter?.value === value
  }
  function toggle(key: string, value: string) {
    if (isFiltered(key, value)) onFilter(null)
    else onFilter({ key, value })
  }

  const stageLabel = (k: string) =>
    k === 'a' ? 'a · BM25 cascade'
    : k === 'b' ? 'b · Wide→themes'
    : k === 'c' ? 'c · Reverse RAG'
    : k === 'd' ? 'd · External'
    : k === 'e' ? 'e · Fail-fast'
    : k

  return (
    <div className="funnel">
      <h3>Pipeline Funnel</h3>
      <div className="dim-text" style={{ marginBottom: 8 }}>
        Click any tile to filter the question list below.
      </div>

      {/* Strategy execution (where queries went) */}
      <div className="funnel-row">
        <div className="funnel-row-label">Strategy executed</div>
        <div className="funnel-tiles">
          {(['a', 'b', 'c', 'd', 'e'] as const).map((s) => (
            <FunnelTile
              key={s}
              label={stageLabel(s)}
              count={strategies[s] || 0}
              total={total}
              active={isFiltered('strategy', s)}
              onClick={() => toggle('strategy', s)}
              kind={s === 'e' ? 'fail' : 'default'}
            />
          ))}
        </div>
      </div>

      {/* Confidence distribution */}
      <div className="funnel-row">
        <div className="funnel-row-label">Confidence</div>
        <div className="funnel-tiles">
          {(['high', 'medium', 'low'] as const).map((c) => (
            <FunnelTile
              key={c}
              label={c}
              count={confidences[c] || 0}
              total={total}
              active={isFiltered('confidence', c)}
              onClick={() => toggle('confidence', c)}
              kind={c === 'high' ? 'ok' : c === 'medium' ? 'warn' : 'dim'}
            />
          ))}
        </div>
      </div>

      {/* Routing-vs-expected */}
      <div className="funnel-row">
        <div className="funnel-row-label">Routing vs label</div>
        <div className="funnel-tiles">
          <FunnelTile
            label="match"
            count={routingCorrect}
            total={routingCorrect + routingWrong}
            active={isFiltered('routing_correct', 'true')}
            onClick={() => toggle('routing_correct', 'true')}
            kind="ok"
          />
          <FunnelTile
            label="miss"
            count={routingWrong}
            total={routingCorrect + routingWrong}
            active={isFiltered('routing_correct', 'false')}
            onClick={() => toggle('routing_correct', 'false')}
            kind="warn"
          />
        </div>
      </div>

      {/* Citation hit */}
      {(citationHit + citationMiss > 0) && (
        <div className="funnel-row">
          <div className="funnel-row-label">Citation hit</div>
          <div className="funnel-tiles">
            <FunnelTile label="hit" count={citationHit} total={citationHit + citationMiss} kind="ok" />
            <FunnelTile label="miss" count={citationMiss} total={citationHit + citationMiss} kind="warn" />
          </div>
        </div>
      )}

      {/* Verdict distribution */}
      <div className="funnel-row">
        <div className="funnel-row-label">Judge verdict</div>
        <div className="funnel-tiles">
          {(['correct', 'partial', 'wrong', 'unable_to_verify'] as const).map((v) => (
            <FunnelTile
              key={v}
              label={v.replace('_to_verify', '')}
              count={verdicts[v] || 0}
              total={total}
              active={isFiltered('verdict', v)}
              onClick={() => toggle('verdict', v)}
              kind={v === 'correct' ? 'ok' : v === 'wrong' ? 'bad' : v === 'partial' ? 'warn' : 'dim'}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

function FunnelTile({
  label,
  count,
  total,
  active = false,
  kind = 'default',
  onClick,
}: {
  label: string
  count: number
  total: number
  active?: boolean
  kind?: string
  onClick?: () => void
}) {
  const p = total ? (count / total) * 100 : 0
  return (
    <button
      type="button"
      className={`funnel-tile tile-${kind} ${active ? 'active' : ''} ${onClick ? 'clickable' : ''}`}
      onClick={onClick}
      title={`${count} of ${total} (${Math.round(p)}%)`}
    >
      <div className="tile-count">{count}</div>
      <div className="tile-label">{label}</div>
      <div className="tile-bar" style={{ width: `${p}%` }} />
    </button>
  )
}

// ── Question list (with inline expand) ─────────────────────────────────

function QuestionList({
  results,
  expandedResultId,
  cache,
  onToggle,
  onHumanVerdict,
  filter,
  clearFilter,
  totalResults,
}: {
  results: EvalResultRow[]
  expandedResultId: string | null
  cache: Record<string, ResultDetail>
  onToggle: (resultId: string) => void
  onHumanVerdict: (resultId: string, v: string | null, reasoning?: string) => Promise<void> | void
  filter: { key: string; value: string } | null
  clearFilter: () => void
  totalResults: number
}) {
  return (
    <div className="question-list">
      <h3>
        Questions ({results.length}{filter ? ` of ${totalResults}` : ''})
        {filter && (
          <button className="clear-filter" onClick={clearFilter}>
            clear filter ({filter.key}={filter.value}) ✕
          </button>
        )}
      </h3>
      {results.map((res) => {
        const expanded = expandedResultId === res.id
        const detail = cache[res.id]
        return (
          <div key={res.id} className="question-row-wrapper">
            <div
              className={`question-row ${expanded ? 'expanded' : ''}`}
              onClick={() => onToggle(res.id)}
            >
              <span className="row-toggle">{expanded ? '▾' : '▸'}</span>
              <span className="qid">{res.query_id}</span>
              {(() => {
                const eff = res.effective_verdict || res.judge_verdict || 'unable_to_verify'
                const overridden = !!res.human_verdict && res.human_verdict !== res.judge_verdict
                return (
                  <span
                    className={`verdict ${VERDICT_COLOR[eff]}`}
                    title={overridden ? `Human override (judge said ${res.judge_verdict})` : undefined}
                  >
                    {eff.replace('_to_verify', '')}
                    {overridden && <span className="human-marker"> 👤</span>}
                  </span>
                )
              })()}
              <span className="strat-mini">
                {res.strategy_chosen}/{res.strategy_executed}
              </span>
              <span className="row-query">{res.query}</span>
              <span className="row-meta">
                {res.confidence && <span className="badge mini">{res.confidence}</span>}
                {res.total_ms !== null && <span className="badge mini dim">{res.total_ms}ms</span>}
                {res.routing_correct === false && <span className="badge mini warn">rt miss</span>}
                {res.retrieval_grade != null && (
                  <span className="badge mini grade-ret" title="retrieval grade">
                    ret {res.retrieval_grade.toFixed(2)}
                  </span>
                )}
                {res.synthesis_grade != null && (
                  <GapBadge gap={res.synthesis_gap ?? null} synth={res.synthesis_grade} />
                )}
              </span>
            </div>
            {expanded && (
              <div className="question-drilldown">
                {!detail && <div className="eval-empty">Loading pipeline trace…</div>}
                {detail && (
                  <ResultDetailView
                    detail={detail}
                    onHumanVerdict={(v, reasoning) => onHumanVerdict(res.id, v, reasoning)}
                  />
                )}
              </div>
            )}
          </div>
        )
      })}
      {results.length === 0 && (
        <div className="eval-empty">No questions match this filter.</div>
      )}
    </div>
  )
}

function ResultDetailView({
  detail,
  onHumanVerdict,
}: {
  detail: ResultDetail
  onHumanVerdict?: (verdict: string | null, reasoning?: string) => void | Promise<void>
}) {
  const r = detail.result

  // Prefer the captured full_response (post-migration runs); for legacy
  // rows we don't have it — synthesize a minimal AgentResponse from
  // the persisted columns so the trace still renders something useful.
  const responseForTrace: AgentResponse =
    r.full_response ||
    ({
      chunks: (r.chunks_summary || []).map((c) => ({
        text: c.text || '',
        document_name: c.document_name,
        page_number: c.page_number,
        rerank_score: c.rerank_score ?? undefined,
      })),
      confidence: r.confidence || undefined,
      strategy_used: r.strategy_executed || undefined,
      llm_answer: r.llm_answer ?? undefined,
      routing: detail.routing
        ? {
            strategy: r.strategy_chosen || undefined,
            executed_strategy: r.strategy_executed || undefined,
            fallback: detail.routing.fallback_strategy || null,
            query_class: detail.routing.query_class,
            method: detail.routing.routing_method,
            scores: detail.routing.scores,
            self_assessments: detail.routing.self_assessments,
            withdrawn: detail.routing.withdrawn,
            priors_version: detail.routing.priors_version,
          }
        : undefined,
    } as AgentResponse)

  const judge: JudgeBlock = {
    verdict: r.judge_verdict,
    score: r.judge_score,
    reasoning: r.judge_reasoning,
    model: r.judge_model,
  }

  return (
    <div>
      <h3 className="result-detail-header">
        <span className="qid">{r.query_id}</span>{' '}
        <span className={`verdict ${VERDICT_COLOR[r.judge_verdict || 'unable_to_verify']}`}>
          {r.judge_verdict}
        </span>
        {r.judge_score !== null && (
          <span className="score">score {r.judge_score.toFixed(2)}</span>
        )}
      </h3>

      {onHumanVerdict && (
        <HumanVerdictCard
          judgeVerdict={r.judge_verdict || null}
          humanVerdict={r.human_verdict || null}
          humanReasoning={r.human_reasoning || null}
          humanVerdictAt={r.human_verdict_at || null}
          onSave={onHumanVerdict}
        />
      )}

      {(r.retrieval_grade != null || r.synthesis_grade != null) && (
        <TwoGradeBar
          retrieval={r.retrieval_grade ?? null}
          synthesis={r.synthesis_grade ?? null}
          gap={r.synthesis_gap ?? null}
        />
      )}

      {r.per_claim_ledger && r.per_claim_ledger.length > 0 && (
        <PerClaimLedger claims={r.per_claim_ledger} />
      )}

      <AgentPipelineTrace
        response={responseForTrace}
        query={r.query}
        expected={r.expected}
        judge={judge}
      />
    </div>
  )
}


function HumanVerdictCard({
  judgeVerdict,
  humanVerdict,
  humanReasoning,
  humanVerdictAt,
  onSave,
}: {
  judgeVerdict: string | null
  humanVerdict: string | null
  humanReasoning: string | null
  humanVerdictAt: string | null
  onSave: (verdict: string | null, reasoning?: string) => void | Promise<void>
}) {
  const [reasoning, setReasoning] = useState(humanReasoning || '')
  const [saving, setSaving] = useState(false)

  // Re-sync reasoning when the underlying row changes.
  useEffect(() => {
    setReasoning(humanReasoning || '')
  }, [humanReasoning])

  async function pick(v: string | null) {
    setSaving(true)
    try {
      await onSave(v, reasoning || undefined)
    } finally {
      setSaving(false)
    }
  }

  const overridden = humanVerdict && humanVerdict !== judgeVerdict
  return (
    <div className="human-verdict-card">
      <div className="human-verdict-header">
        <strong>👤 Human Override</strong>
        {humanVerdict ? (
          <span className="dim-text">
            you said <span className={`verdict ${VERDICT_COLOR[humanVerdict]}`}>{humanVerdict}</span>
            {humanVerdictAt && <> · {new Date(humanVerdictAt).toLocaleString()}</>}
            {overridden && <span className="dim-text"> · differs from judge</span>}
          </span>
        ) : (
          <span className="dim-text">
            judge said <span className={`verdict ${VERDICT_COLOR[judgeVerdict || 'unable_to_verify']}`}>{judgeVerdict || '—'}</span>
            {' '}— override if you disagree
          </span>
        )}
      </div>
      <div className="human-verdict-pills">
        {(['correct', 'partial', 'wrong', 'unable_to_verify'] as const).map((v) => (
          <button
            key={v}
            type="button"
            className={`hv-pill ${humanVerdict === v ? 'hv-pill-active' : ''} ${VERDICT_COLOR[v]}`}
            onClick={() => pick(v)}
            disabled={saving}
          >
            {v === 'correct' && '✓ '}
            {v === 'partial' && '◐ '}
            {v === 'wrong' && '✗ '}
            {v === 'unable_to_verify' && '? '}
            {v.replace('_to_verify', '')}
          </button>
        ))}
        {humanVerdict && (
          <button
            type="button"
            className="hv-pill hv-pill-clear"
            onClick={() => pick(null)}
            disabled={saving}
            title="Clear override — revert to judge verdict"
          >
            × clear override
          </button>
        )}
      </div>
      <textarea
        className="human-verdict-textarea"
        placeholder="Optional reasoning — why you chose this verdict (helps recalibrate the judge later)."
        value={reasoning}
        onChange={(e) => setReasoning(e.target.value)}
        rows={2}
      />
    </div>
  )
}


// ── Bank editor ────────────────────────────────────────────────────────

const STRATEGIES_AVAIL = ['a', 'b', 'c', 'd', 'e'] as const
const QCLASS_AVAIL = ['literal_anchor', 'tight_pool', 'wide_pool', 'exploratory', 'vague'] as const

function BankEditor({
  queries,
  onUpdate,
  onUpdateExpected,
  onAdd,
  onDelete,
  dirty,
}: {
  queries: BankQuery[]
  onUpdate: (i: number, patch: Partial<BankQuery>) => void
  onUpdateExpected: (i: number, patch: Partial<NonNullable<BankQuery['expected']>>) => void
  onAdd: () => void
  onDelete: (i: number) => void
  dirty: boolean
}) {
  return (
    <div className="bank-editor">
      <div className="bank-editor-header">
        <h3>Question Bank Editor</h3>
        <span className="dim-text">{queries.length} queries{dirty ? ' · unsaved changes' : ''}</span>
        <button className="eval-toolbar-btn small" onClick={onAdd}>+ Add Query</button>
      </div>
      {queries.map((q, i) => (
        <div key={i} className="bank-query-card">
          <div className="bank-row">
            <input
              type="text"
              className="bank-id"
              value={q.id}
              onChange={(e) => onUpdate(i, { id: e.target.value })}
              placeholder="q###"
            />
            <input
              type="text"
              className="bank-query-text"
              value={q.query}
              onChange={(e) => onUpdate(i, { query: e.target.value })}
              placeholder="Query text…"
            />
            <button
              className="eval-toolbar-btn small danger"
              onClick={() => onDelete(i)}
              title="Delete this query"
            >
              ×
            </button>
          </div>
          <div className="bank-row bank-expected-row">
            <label>
              <span className="ctrl-label">strategy</span>
              <select
                value={q.expected?.strategy || ''}
                onChange={(e) => onUpdateExpected(i, { strategy: e.target.value || undefined })}
              >
                <option value="">—</option>
                {STRATEGIES_AVAIL.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </label>
            <label>
              <span className="ctrl-label">qclass</span>
              <select
                value={q.expected?.query_class || ''}
                onChange={(e) => onUpdateExpected(i, { query_class: e.target.value || undefined })}
              >
                <option value="">—</option>
                {QCLASS_AVAIL.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </label>
            <label className="grow">
              <span className="ctrl-label">answer keywords (comma-sep)</span>
              <input
                type="text"
                value={(q.expected?.answer_keywords || []).join(', ')}
                onChange={(e) =>
                  onUpdateExpected(i, {
                    answer_keywords: e.target.value.split(',').map((s) => s.trim()).filter(Boolean),
                  })
                }
                placeholder="e.g. 365, year, filing"
              />
            </label>
          </div>
          <div className="bank-row bank-expected-row">
            <label className="grow">
              <span className="ctrl-label">must_cite_doc (comma-sep)</span>
              <input
                type="text"
                value={(q.expected?.must_cite_doc || []).join(', ')}
                onChange={(e) =>
                  onUpdateExpected(i, {
                    must_cite_doc: e.target.value.split(',').map((s) => s.trim()).filter(Boolean),
                  })
                }
                placeholder="e.g. Sunshine Provider Manual"
              />
            </label>
            <label className="grow">
              <span className="ctrl-label">must_cite_url contains (comma-sep)</span>
              <input
                type="text"
                value={(q.expected?.must_cite_url_contains || []).join(', ')}
                onChange={(e) =>
                  onUpdateExpected(i, {
                    must_cite_url_contains: e.target.value.split(',').map((s) => s.trim()).filter(Boolean),
                  })
                }
                placeholder="e.g. cms.gov, ahca"
              />
            </label>
            <label>
              <span className="ctrl-label">FF reason</span>
              <input
                type="text"
                value={q.expected?.fail_fast_reason || ''}
                onChange={(e) =>
                  onUpdateExpected(i, { fail_fast_reason: e.target.value || undefined })
                }
                placeholder="(only for e)"
              />
            </label>
          </div>
          <div className="bank-row">
            <input
              type="text"
              className="bank-notes"
              value={q.notes || ''}
              onChange={(e) => onUpdate(i, { notes: e.target.value })}
              placeholder="notes (analyst-only)"
            />
            <input
              type="text"
              className="bank-caller-mode"
              value={q.caller_mode || ''}
              onChange={(e) => onUpdate(i, { caller_mode: e.target.value || undefined })}
              placeholder="caller_mode (default: chat.default)"
            />
          </div>
        </div>
      ))}
      <button className="eval-toolbar-btn" onClick={onAdd}>+ Add Query</button>
    </div>
  )
}
