/**
 * SearchTracePanel — full pipeline visualization for corpus_search telemetry.
 *
 * Stages displayed:
 *   1. Timing bar   — embed / BM25 / vec / rerank wall times
 *   2. Retrieval    — per-arm hit tables (BM25 ts_rank, vector cosine)
 *   3. Fusion + Reranking — scoring_trace with signal breakdown per chunk
 *   4. Assembly     — canonical ratio, tier breakdown, strategy
 *   5. Neighbourhood — adjacent-page preview (Phase 3 placeholder)
 */
import { Fragment, useState } from 'react'
import './SearchTracePanel.css'

// ── Types ─────────────────────────────────────────────────────────────────────

interface ArmResultItem {
  chunk_id: string
  document_name: string
  document_id: string
  page_number: number | null
  authority_level: string | null
  authority_tier: number
  payer: string | null
  ts_rank?: number
  cosine?: number
  text_preview: string
}

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

interface ScoringTraceItem {
  rank: number
  chunk_id: string
  document_name: string
  page_number: number | null
  retrieval_arms: string[]
  authority_level: string | null
  authority_tier: number
  confidence_label: string
  text_preview: string
  arm_scores: Record<string, number>
  arm_ranks: Record<string, number>
  rrf_score: number
  rerank_signals: RerankSignals
}

interface AssemblyInfo {
  strategy: string
  canonical_floor: number | null
  canonical_ratio: number
  strict_canonical_ratio: number
  tier_breakdown: Record<string, number>
  total_selected: number
}

export interface SearchTelemetry {
  search_id: string
  mode: string
  k: number
  query?: string
  bm25_normalized_query?: string | null
  embed_ms: number
  bm25_ms: number
  vec_ms: number
  rerank_ms: number
  total_ms: number
  arm_hits: { bm25: number; vector: number }
  arm_results: { bm25: ArmResultItem[]; vector: ArmResultItem[] }
  candidates: number
  returned: number
  min_label_applied: string
  reranker: string
  assembly: AssemblyInfo
  scoring_trace: ScoringTraceItem[]
}

// ── Constants ─────────────────────────────────────────────────────────────────

const AUTH_SHORT: Record<string, string> = {
  contract_source_of_truth: 'CoT',
  payer_policy:             'PP',
  operational_suggested:    'Ops',
  fyi_not_citable:          'FYI',
}

const AUTH_TIER_COLOR: Record<number, string> = {
  0: '#16a34a',
  1: '#2563eb',
  2: '#d97706',
  3: '#9ca3af',
}

const CONFIDENCE_COLOR: Record<string, string> = {
  high:    '#10b981',
  medium:  '#f59e0b',
  low:     '#6b7280',
  abstain: '#d1d5db',
}

const TIER_LABEL: Record<string, string> = {
  contract_source_of_truth: 'CoT',
  payer_policy:             'PP',
  operational_suggested:    'Ops',
  fyi_not_citable:          'FYI',
}

const TIER_COLOR: Record<string, string> = {
  contract_source_of_truth: '#16a34a',
  payer_policy:             '#2563eb',
  operational_suggested:    '#0891b2',
  fyi_not_citable:          '#d97706',
}

const TIER_ORDER = [
  'contract_source_of_truth',
  'payer_policy',
  'operational_suggested',
  'fyi_not_citable',
]

// ── Helpers ───────────────────────────────────────────────────────────────────

function authShort(level: string | null): string {
  if (!level) return '—'
  return AUTH_SHORT[level.toLowerCase()] ?? level.slice(0, 4)
}

function authColor(tier: number): string {
  return AUTH_TIER_COLOR[tier] ?? '#9ca3af'
}

function clamp(min: number, max: number, val: number) {
  return Math.min(max, Math.max(min, val))
}

// ── Micro-components ─────────────────────────────────────────────────────────

function ScoreBar({
  value,
  max = 1,
  color = '#2563eb',
  trackPx = 80,
}: {
  value: number
  max?: number
  color?: string
  trackPx?: number
}) {
  const pct = clamp(0, 100, (value / (max || 1)) * 100)
  return (
    <span className="stp-score-bar">
      <span
        className="stp-score-track"
        style={{ width: trackPx }}
      >
        <span
          className="stp-score-fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </span>
      <code className="stp-score-val">{value.toFixed(3)}</code>
    </span>
  )
}

function Section({
  title,
  badge,
  children,
  defaultOpen = true,
}: {
  title: string
  badge?: string
  children: React.ReactNode
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="stp-section">
      <button
        type="button"
        className="stp-section-hdr"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <span className="stp-chevron" aria-hidden>{open ? '▼' : '▶'}</span>
        <span className="stp-section-title">{title}</span>
        {badge && <span className="stp-section-badge">{badge}</span>}
      </button>
      {open && <div className="stp-section-body">{children}</div>}
    </div>
  )
}

// ── Stage 1: Timing ──────────────────────────────────────────────────────────

function TimingBar({ t }: { t: SearchTelemetry }) {
  const total = t.total_ms || 1
  const phases: Array<{ label: string; ms: number; color: string }> = [
    { label: 'embed',  ms: t.embed_ms,  color: '#6366f1' },
    { label: 'BM25',   ms: t.bm25_ms,   color: '#ea580c' },
    { label: 'vec',    ms: t.vec_ms,    color: '#7c3aed' },
    { label: 'rerank', ms: t.rerank_ms, color: '#0891b2' },
  ]

  return (
    <div className="stp-timing">
      {/* Proportional bar */}
      <div className="stp-timing-bar">
        {phases.map((p) => {
          const pct = clamp(0.5, 100, (p.ms / total) * 100)
          if (p.ms < 0.5) return null
          return (
            <span
              key={p.label}
              className="stp-timing-seg"
              style={{ width: `${pct}%`, background: p.color }}
              title={`${p.label}: ${p.ms.toFixed(1)}ms`}
            />
          )
        })}
      </div>
      {/* Legend pills */}
      <div className="stp-timing-legend">
        {phases.map((p) => (
          <span
            key={p.label}
            className="stp-timing-pill"
            style={{ borderColor: p.color }}
          >
            <span className="stp-timing-dot" style={{ background: p.color }} />
            {p.label} {p.ms.toFixed(0)}ms
          </span>
        ))}
        <span className="stp-timing-pill stp-timing-total">
          total {t.total_ms.toFixed(0)}ms
        </span>
      </div>
    </div>
  )
}

// ── Stage 2: Arm tables ───────────────────────────────────────────────────────

function ArmTable({
  items,
  scoreKey,
}: {
  items: ArmResultItem[]
  scoreKey: 'ts_rank' | 'cosine'
}) {
  if (!items.length)
    return <p className="stp-empty">No hits from this arm</p>

  const color = scoreKey === 'ts_rank' ? '#ea580c' : '#6366f1'

  return (
    <table className="stp-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Document</th>
          <th>p.</th>
          <th>{scoreKey === 'ts_rank' ? 'ts_rank' : 'cosine'}</th>
          <th>Auth</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item, i) => {
          const score =
            scoreKey === 'ts_rank'
              ? (item.ts_rank ?? 0)
              : (item.cosine ?? 0)
          return (
            <tr key={item.chunk_id}>
              <td className="stp-td-num">{i + 1}</td>
              <td
                className="stp-td-doc"
                title={`${item.document_name}\n${item.text_preview}`}
              >
                {item.document_name.length > 42
                  ? `${item.document_name.slice(0, 42)}…`
                  : item.document_name}
              </td>
              <td className="stp-td-num">{item.page_number ?? '—'}</td>
              <td>
                <ScoreBar value={score} color={color} trackPx={72} />
              </td>
              <td>
                <span
                  className="stp-auth-badge"
                  style={{ color: authColor(item.authority_tier) }}
                  title={item.authority_level ?? 'untagged'}
                >
                  {authShort(item.authority_level)}
                </span>
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

// ── Stage 3: Fusion + Reranking ───────────────────────────────────────────────

function RerankTable({ items }: { items: ScoringTraceItem[] }) {
  const [expanded, setExpanded] = useState<string | null>(null)

  if (!items.length) return <p className="stp-empty">No scored candidates</p>

  return (
    <div className="stp-rerank-wrap">
      <table className="stp-table stp-table--rerank">
        <thead>
          <tr>
            <th>#</th>
            <th>Document</th>
            <th>p.</th>
            <th>Arms</th>
            <th title="sim × 0.30  (best raw arm score rescaled)">Sim ×.30</th>
            <th title="authority_weight × 0.15">Auth ×.15</th>
            <th title="length_norm × 0.10">Len ×.10</th>
            <th title="JPD tag-match × 0.25 — J=Eligibility P=Prior-auth D=Documentation">JPD ×.25</th>
            <th title="RRF 1/(60+rank) fusion score">RRF</th>
            <th title="rerank_score = total / max_weight">Final</th>
            <th>Conf</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => {
            const s = item.rerank_signals ?? {} as RerankSignals
            const isHybrid = (item.retrieval_arms ?? []).length > 1
            const isExpanded = expanded === item.chunk_id

            return (
              <Fragment key={item.chunk_id}>
                <tr
                  className={`stp-tr${isHybrid ? ' stp-tr--hybrid' : ''}`}
                  onClick={() =>
                    setExpanded((e) => (e === item.chunk_id ? null : item.chunk_id))
                  }
                  title="Click to toggle text preview"
                  style={{ cursor: 'pointer' }}
                >
                  <td className="stp-td-num">{item.rank}</td>
                  <td className="stp-td-doc" title={item.document_name}>
                    {item.document_name.length > 38
                      ? `${item.document_name.slice(0, 38)}…`
                      : item.document_name}
                  </td>
                  <td className="stp-td-num">{item.page_number ?? '—'}</td>
                  <td className="stp-td-arms">
                    {item.retrieval_arms.map((arm) => (
                      <span
                        key={arm}
                        className={`stp-arm-badge stp-arm-badge--${arm}`}
                      >
                        {arm === 'bm25' ? 'BM25' : arm === 'vector' ? 'Vec' : arm}
                      </span>
                    ))}
                    {isHybrid && (
                      <span className="stp-arm-badge stp-arm-badge--hybrid">BOTH</span>
                    )}
                  </td>
                  <td>
                    <ScoreBar
                      value={s.sim_weighted ?? 0}
                      max={0.3}
                      color="#6366f1"
                      trackPx={56}
                    />
                  </td>
                  <td>
                    <ScoreBar
                      value={s.authority_weighted ?? 0}
                      max={0.15}
                      color="#16a34a"
                      trackPx={44}
                    />
                  </td>
                  <td>
                    <ScoreBar
                      value={s.length_weighted ?? 0}
                      max={0.1}
                      color="#0891b2"
                      trackPx={36}
                    />
                  </td>
                  <td>
                    {(s.jpd_weighted != null && s.jpd_weighted > 0) ? (
                      <span className="stp-jpd-cell">
                        <ScoreBar
                          value={s.jpd_weighted}
                          max={0.25}
                          color="#d97706"
                          trackPx={44}
                        />
                        {s.jpd_tags && s.jpd_tags.map((t) => (
                          <span key={t} className={`stp-jpd-tag stp-jpd-tag--${t.toLowerCase()}`}>{t}</span>
                        ))}
                      </span>
                    ) : (
                      <span className="stp-td-dim">—</span>
                    )}
                  </td>
                  <td className="stp-td-num stp-td-dim">
                    {item.rrf_score != null ? item.rrf_score.toFixed(4) : '—'}
                  </td>
                  <td>
                    <ScoreBar
                      value={s.rerank_score ?? 0}
                      max={1}
                      color="#f59e0b"
                      trackPx={64}
                    />
                  </td>
                  <td>
                    <span
                      className="stp-conf-badge"
                      style={{
                        color:
                          CONFIDENCE_COLOR[item.confidence_label] ?? '#9ca3af',
                      }}
                    >
                      {item.confidence_label}
                    </span>
                  </td>
                </tr>
                {isExpanded && (
                  <tr className="stp-tr-expand">
                    <td />
                    <td colSpan={10} className="stp-expand-body">
                      <div className="stp-expand-signals">
                        <span>
                          sim_raw={(s.sim_raw ?? 0).toFixed(4)} →{' '}
                          ×0.30={(s.sim_weighted ?? 0).toFixed(4)}
                        </span>
                        <span>
                          auth_raw={(s.authority_raw ?? 0).toFixed(3)} →{' '}
                          ×0.15={(s.authority_weighted ?? 0).toFixed(4)}
                        </span>
                        <span>
                          len_raw={(s.length_raw ?? 0).toFixed(3)} →{' '}
                          ×0.10={(s.length_weighted ?? 0).toFixed(4)}
                        </span>
                        {s.jpd_raw != null && (
                          <span>
                            jpd_raw={(s.jpd_raw ?? 0).toFixed(3)} →{' '}
                            ×0.25={(s.jpd_weighted ?? 0).toFixed(4)}
                            {s.jpd_tags && s.jpd_tags.length > 0 && (
                              <> [{s.jpd_tags.join('+')}]</>
                            )}
                          </span>
                        )}
                        <span>
                          total_raw={(s.total_raw ?? 0).toFixed(4)} ÷{' '}
                          {(s.max_weight ?? 1).toFixed(2)} ={' '}
                          <strong>{(s.rerank_score ?? 0).toFixed(4)}</strong>
                        </span>
                      </div>
                      {Object.keys(item.arm_scores ?? {}).length > 0 && (
                        <div className="stp-expand-arm-scores">
                          {Object.entries(item.arm_scores ?? {}).map(([arm, score]) => (
                            <span key={arm}>
                              {arm}: {score.toFixed(4)}{' '}
                              (rank {item.arm_ranks?.[arm] ?? '?'})
                            </span>
                          ))}
                        </div>
                      )}
                      <p className="stp-expand-preview">{item.text_preview}</p>
                    </td>
                  </tr>
                )}
              </Fragment>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ── Stage 4: Assembly ─────────────────────────────────────────────────────────

function AssemblyPanel({ a }: { a: AssemblyInfo }) {
  const canonPct = Math.round(clamp(0, 100, (a.canonical_ratio ?? 0) * 100))
  const strictPct = Math.round(clamp(0, 100, (a.strict_canonical_ratio ?? 0) * 100))

  const tierBreakdown = a.tier_breakdown ?? {}
  // Untagged might be stored as key "untagged" or "null"
  const untagged =
    (tierBreakdown['untagged'] ?? 0) +
    (tierBreakdown['null'] ?? 0) +
    (tierBreakdown['None'] ?? 0)

  return (
    <div className="stp-assembly">
      {/* Meta row */}
      <div className="stp-assembly-meta">
        <span className="stp-kv">
          <span className="stp-k">strategy</span>
          <code className="stp-v">{a.strategy}</code>
        </span>
        {a.canonical_floor != null && (
          <span className="stp-kv">
            <span className="stp-k">floor</span>
            <code className="stp-v">{Math.round(a.canonical_floor * 100)}%</code>
          </span>
        )}
        <span className="stp-kv">
          <span className="stp-k">selected</span>
          <code className="stp-v">{a.total_selected}</code>
        </span>
      </div>

      {/* Canonical ratio gauges */}
      <div className="stp-ratio-block">
        <div className="stp-ratio-row">
          <span className="stp-ratio-label">Canonical (CoT + PP)</span>
          <div className="stp-ratio-track">
            <div
              className="stp-ratio-fill"
              style={{ width: `${canonPct}%`, background: '#2563eb' }}
            />
          </div>
          <span className="stp-ratio-pct">{canonPct}%</span>
        </div>
        <div className="stp-ratio-row">
          <span className="stp-ratio-label">Strict (CoT only)</span>
          <div className="stp-ratio-track">
            <div
              className="stp-ratio-fill"
              style={{ width: `${strictPct}%`, background: '#16a34a' }}
            />
          </div>
          <span className="stp-ratio-pct">{strictPct}%</span>
        </div>
      </div>

      {/* Tier breakdown pills */}
      <div className="stp-tier-row">
        {TIER_ORDER.map((tier) => {
          const n = tierBreakdown[tier] ?? 0
          if (n === 0) return null
          return (
            <span
              key={tier}
              className="stp-tier-pill"
              style={{ borderColor: TIER_COLOR[tier], color: TIER_COLOR[tier] }}
              title={tier}
            >
              {TIER_LABEL[tier]} × {n}
            </span>
          )
        })}
        {untagged > 0 && (
          <span
            className="stp-tier-pill"
            style={{ borderColor: '#9ca3af', color: '#9ca3af' }}
          >
            untagged × {untagged}
          </span>
        )}
      </div>
    </div>
  )
}

// ── Stage 5: Neighbourhood (placeholder) ─────────────────────────────────────

function NeighbourhoodPanel({ items }: { items: ScoringTraceItem[] }) {
  const top = items.filter((i) => i.page_number != null).slice(0, 5)
  return (
    <div className="stp-neighbourhood">
      <p className="stp-neighbourhood-hint">
        Phase 3 — neighbourhood fetch (±1 page context) is not yet wired.
        Below shows which page windows would be loaded for the top results.
      </p>
      {top.map((item) => (
        <div key={item.chunk_id} className="stp-neighbour-row">
          <span
            className="stp-neighbour-doc"
            title={item.document_name}
          >
            {item.document_name.length > 44
              ? `${item.document_name.slice(0, 44)}…`
              : item.document_name}
          </span>
          <span className="stp-neighbour-pages">
            p.{(item.page_number! - 1)} ·{' '}
            <strong>p.{item.page_number}</strong> ·{' '}
            p.{(item.page_number! + 1)}
          </span>
        </div>
      ))}
    </div>
  )
}

// ── Root export ───────────────────────────────────────────────────────────────

interface Props {
  telemetry: SearchTelemetry
}

export function SearchTracePanel({ telemetry: t }: Props) {
  const armHits = t.arm_hits ?? { bm25: 0, vector: 0 }
  const armResults = t.arm_results ?? { bm25: [], vector: [] }
  const scoringTrace = (t.scoring_trace ?? []) as ScoringTraceItem[]
  const hybridCount = scoringTrace.filter((x) => x.retrieval_arms.length > 1).length

  return (
    <div className="search-trace-panel">
      {/* ── Header strip ────────────────────────────────────────── */}
      <div className="stp-hdr">
        <span className="stp-hdr-title">Pipeline Trace</span>
        <span className="stp-hdr-meta">
          <span className="stp-meta-pill">{t.mode}</span>
          <span className="stp-meta-pill">{t.total_ms?.toFixed(0)}ms</span>
          <span className="stp-meta-pill">
            id:{(t.search_id ?? '').slice(0, 8)}
          </span>
        </span>
      </div>

      {/* ── 1. Timing ────────────────────────────────────────────── */}
      <Section title="Timing" defaultOpen>
        <TimingBar t={t} />
      </Section>

      {/* ── 2. Retrieval arms ────────────────────────────────────── */}
      <Section
        title="Retrieval"
        badge={`BM25 ${armHits.bm25} · Vec ${armHits.vector}`}
        defaultOpen
      >
        {t.bm25_normalized_query && t.bm25_normalized_query !== t.query && (
          <div className="stp-normalized-query">
            <span className="stp-nq-label">BM25 query rewritten</span>
            <span className="stp-nq-original" title="Original query">{t.query}</span>
            <span className="stp-nq-arrow">→</span>
            <span className="stp-nq-normalized" title="After noise stripping">{t.bm25_normalized_query}</span>
          </div>
        )}
        <div className="stp-arms-grid">
          <div className="stp-arm-col">
            <div className="stp-arm-col-hdr stp-arm-col-hdr--bm25">
              BM25 arm &nbsp;({armHits.bm25} hits)
            </div>
            <ArmTable items={armResults.bm25} scoreKey="ts_rank" />
          </div>
          <div className="stp-arm-col">
            <div className="stp-arm-col-hdr stp-arm-col-hdr--vec">
              Vector arm &nbsp;({armHits.vector} hits)
            </div>
            <ArmTable items={armResults.vector} scoreKey="cosine" />
          </div>
        </div>
      </Section>

      {/* ── 3. Fusion + Reranking ──────────────────────────────────── */}
      <Section
        title="Fusion + Reranking"
        badge={`${t.candidates ?? '?'} candidates → ${t.returned ?? '?'} selected · ${hybridCount} hybrid`}
        defaultOpen
      >
        <p className="stp-reranker-label">{t.reranker}</p>
        <p className="stp-reranker-hint">Click any row to expand signals + preview.</p>
        <RerankTable items={scoringTrace} />
      </Section>

      {/* ── 4. Assembly ──────────────────────────────────────────── */}
      {t.assembly && (
        <Section
          title="Assembly"
          badge={`${t.assembly.strategy} · ${Math.round((t.assembly.canonical_ratio ?? 0) * 100)}% canonical`}
          defaultOpen
        >
          <AssemblyPanel a={t.assembly} />
        </Section>
      )}

      {/* ── 5. Neighbourhood ─────────────────────────────────────── */}
      <Section title="Neighbourhood" defaultOpen={false}>
        <NeighbourhoodPanel items={scoringTrace} />
      </Section>
    </div>
  )
}
