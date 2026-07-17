/**
 * QueryTraceDrilldown — full per-query trace for a stored routing decision.
 *
 * Fetches GET /api/routing/decisions/{id} and renders:
 *   - Quick summary bar: grades + routing choice + latency
 *   - 9-stage path indicator: query→cleanup→rewrite→classify→route→retrieve→rerank→assemble→synthesize→grade
 *   - Two-grade hero: TwoGradeBar (retrieval, synthesis, gap)
 *   - Routing decision: feature vector + linear score bars + contribution table
 *   - Action tree: taken/not-taken leaf nodes
 *   - Per-claim ledger: PerClaimLedger (open by default)
 *   - Raw JSON toggle
 */
import React, { useEffect, useState } from 'react'
import { API_BASE } from '../../config'
import { TwoGradeBar, PerClaimLedger, type ClaimEntry } from './EvalTab'
import './EvalTab.css'

// Linear router weights — mirrors LIN_BASE / LIN_WEIGHTS in EvalTab.tsx
const LIN_BASE: Record<string, number> = { a: 0.40, b: 0.20, c: 0.05, d: 0.20 }
const LIN_WEIGHTS: Record<string, Record<string, number>> = {
  a: { exclusivity: 0.30, literal: 0.25, corpus_depth: 0.20, thematic_policy: -0.10, wide_pool: -0.15, inheritance: 0.05 },
  b: { thematic_policy: 0.40, corpus_depth: 0.20, exclusivity: 0.05, literal: -0.20 },
  c: {},
  d: { crawlability: 0.40, wide_pool: 0.25, literal: -0.05, corpus_depth: -0.15, thematic_policy: -0.20, inheritance: -0.25 },
}

const TREE_NODES: Array<{ id: string; label: string; status: 'live' | 'partial' | 'not_built' }> = [
  { id: 's_skim',      label: 's · skim',         status: 'not_built' },
  { id: 'a',           label: 'a · BM25 cascade', status: 'live' },
  { id: 'b',           label: 'b · wide+themes',  status: 'live' },
  { id: 'c',           label: 'c · reverse RAG',  status: 'live' },
  { id: 'd',           label: 'd · external',     status: 'live' },
  { id: 'union',       label: 'union [a+b]',      status: 'live' },
  { id: 'reformulate', label: 'reformulate',       status: 'partial' },
  { id: 'f_floor',     label: 'f · honest-floor', status: 'not_built' },
  { id: 'm_cached',    label: 'm · cached-replay',status: 'not_built' },
]

const PATH_STAGES = [
  'query', 'cleanup', 'rewrite', 'classify',
  'route', 'retrieve', 'rerank', 'assemble', 'synthesize', 'grade',
]

function takenFromLeafKey(leafKey: string | null | undefined): Set<string> {
  if (!leafKey) return new Set()
  const [action, armsStr] = leafKey.split(':')
  const arms = (armsStr ?? '').split('+').filter(Boolean)
  const taken = new Set<string>(arms)
  if (arms.length > 1) taken.add('union')
  if (action === 'union') taken.add('union')
  if (action === 'reformulate') taken.add('reformulate')
  if (action === 'skim') taken.add('s_skim')
  if (action === 'floor') taken.add('f_floor')
  return taken
}

function computeLinearScores(fv: Record<string, number> | null | undefined): Record<string, number> {
  const out: Record<string, number> = {}
  for (const s of ['a', 'b', 'c', 'd']) {
    let total = LIN_BASE[s] ?? 0
    for (const [feat, w] of Object.entries(LIN_WEIGHTS[s] ?? {})) {
      total += w * (fv?.[feat] ?? 0)
    }
    out[s] = Math.max(0, Math.min(2, total))
  }
  return out
}

interface DecisionRow {
  id: string
  query: string
  ts: string
  strategy_chosen: string | null
  strategy_executed: string | null
  confidence: string | null
  total_ms: number | null
  // from rag_routing_decisions — always present
  scores: Record<string, number> | null
  // from rag_query_decisions JOIN — present only if EVAL agent has graded
  leaf_key: string | null
  invoke_all: string[] | null
  feature_vector: Record<string, number> | null
  strategy_scores: Record<string, number> | null
  retrieval_grade: number | null
  synthesis_grade: number | null
  synthesis_gap: number | null
  per_claim_ledger: ClaimEntry[] | string | null
  fact_checker_version: string | null
  query_type: string | null
  query_class: string | null
  coverage: string | null
  tag_matches: string[] | null
  priors_version: string | null
  routing_method: string | null
  fallback_strategy: string | null
}

export function QueryTraceDrilldown({ decisionId }: { decisionId: string }) {
  const [row, setRow] = useState<DecisionRow | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showRaw, setShowRaw] = useState(false)

  useEffect(() => {
    setLoading(true)
    setError(null)
    setRow(null)
    fetch(`${API_BASE}/api/routing/decisions/${decisionId}`)
      .then((r) => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))
      .then(setRow)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false))
  }, [decisionId])

  if (loading) return <div className="eval-empty">Loading trace…</div>
  if (error) return <div className="eval-error">trace: {error}</div>
  if (!row) return null

  const SIGMA = 0.2
  const fv = row.feature_vector
  // Score priority: eval-augmented strategy_scores > routing scores > computed linear fallback
  const stratScores: Record<string, number> =
    row.strategy_scores ?? row.scores ?? computeLinearScores(fv)
  const scoreEntries = Object.entries(stratScores).sort(([, a], [, b]) => b - a)
  const argmax = scoreEntries[0]?.[0] ?? null
  const maxScore = scoreEntries.length > 0 ? Math.max(...Object.values(stratScores), 0.01) : 0.01
  const routeGap = argmax ? (stratScores[argmax] - (scoreEntries[1]?.[1] ?? 0)) : 0
  const invokeAll = row.invoke_all
  const isUnion = (invokeAll && invokeAll.length > 1) || routeGap < 0.08
  const taken = takenFromLeafKey(row.leaf_key)
  const gapGrade = row.synthesis_gap ?? null
  // Defensive parse: per_claim_ledger may come back as a JSON string from older DB rows
  let ledger: ClaimEntry[] | null = null
  try {
    const raw = row.per_claim_ledger
    if (Array.isArray(raw)) ledger = raw
    else if (typeof raw === 'string' && raw.length > 0) ledger = JSON.parse(raw) as ClaimEntry[]
  } catch { ledger = null }
  const validated = ledger?.filter((c) => c.status === 'validated').length ?? 0
  const totalClaims = ledger?.length ?? 0
  const s2 = (n: number | null | undefined) => (n == null ? '—' : n.toFixed(3))

  // Which stages are "lit" based on available data
  const activeStages = new Set<string>([
    'query',
    ...(row.query_class || row.query_type ? ['cleanup', 'rewrite', 'classify'] : []),
    ...(row.strategy_chosen ? ['route'] : []),
    ...(row.strategy_executed ? ['retrieve', 'rerank', 'assemble'] : []),
    ...(row.synthesis_grade != null ? ['synthesize'] : []),
    ...(row.retrieval_grade != null ? ['grade'] : []),
  ])

  const confColor = row.confidence === 'high' ? '#15803d'
    : row.confidence === 'low' ? '#dc2626' : '#92400e'
  const confBg = row.confidence === 'high' ? 'rgba(22,163,74,0.12)'
    : row.confidence === 'low' ? 'rgba(220,38,38,0.10)' : 'rgba(217,119,6,0.10)'

  return (
    <div style={{ fontSize: 13, padding: '8px 0' }}>

      {/* ── Quick summary bar ─────────────────────────────── */}
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center', marginBottom: 12 }}>
        {row.strategy_executed && (
          <span style={{ fontSize: 12, padding: '3px 9px', borderRadius: 4, background: 'var(--rag-accent, #6d28d9)', color: '#fff', fontWeight: 700 }}>
            {row.leaf_key ?? row.strategy_executed}
          </span>
        )}
        {row.confidence && (
          <span style={{ fontSize: 11, padding: '2px 7px', borderRadius: 4, background: confBg, color: confColor, fontWeight: 600 }}>
            {row.confidence}
          </span>
        )}
        {row.retrieval_grade != null && (
          <span style={{ fontSize: 11, padding: '2px 6px', background: 'var(--surface-alt)', borderRadius: 4, fontVariantNumeric: 'tabular-nums' }}>
            ret <strong>{s2(row.retrieval_grade)}</strong>
          </span>
        )}
        {row.synthesis_grade != null && (
          <span style={{ fontSize: 11, padding: '2px 6px', background: 'var(--surface-alt)', borderRadius: 4, fontVariantNumeric: 'tabular-nums' }}>
            syn <strong>{s2(row.synthesis_grade)}</strong>
          </span>
        )}
        {gapGrade !== null && (
          <span style={{
            fontSize: 11, padding: '2px 6px', borderRadius: 4,
            background: 'var(--surface-alt)', fontVariantNumeric: 'tabular-nums',
            color: Math.abs(gapGrade) < SIGMA ? undefined : gapGrade < 0 ? '#d97706' : '#dc2626',
            fontWeight: Math.abs(gapGrade) >= SIGMA ? 600 : undefined,
          }}>
            gap {gapGrade >= 0 ? '+' : ''}{s2(gapGrade)}
          </span>
        )}
        {row.total_ms != null && (
          <span style={{ fontSize: 11, opacity: 0.4 }}>{row.total_ms}ms</span>
        )}
        <button
          onClick={() => setShowRaw((v) => !v)}
          style={{ marginLeft: 'auto', fontSize: 11, opacity: 0.5, background: 'none', border: '1px solid var(--border)', borderRadius: 4, padding: '1px 7px', cursor: 'pointer' }}
        >
          {showRaw ? '× raw' : '{ } raw'}
        </button>
      </div>

      {/* ── Raw JSON ─────────────────────────────────────── */}
      {showRaw && (
        <pre style={{ fontSize: 10, overflow: 'auto', maxHeight: 300, background: 'var(--surface-alt)', borderRadius: 4, padding: 10, marginBottom: 10 }}>
          {JSON.stringify(row, null, 2)}
        </pre>
      )}

      {/* ── 9-stage path indicator ────────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 0, marginBottom: 14, overflowX: 'auto', paddingBottom: 2 }}>
        {PATH_STAGES.map((stage, i) => {
          const active = activeStages.has(stage)
          return (
            <React.Fragment key={stage}>
              <div style={{
                fontSize: 10, padding: '3px 8px', borderRadius: 3, flexShrink: 0,
                background: active ? 'rgba(109,40,217,0.10)' : 'var(--surface-alt)',
                color: active ? 'var(--rag-accent, #6d28d9)' : 'var(--text-muted)',
                border: active ? '1px solid rgba(109,40,217,0.3)' : '1px solid var(--border)',
                fontWeight: active ? 600 : 400,
                opacity: active ? 1 : 0.4,
              }}>
                {stage}
              </div>
              {i < PATH_STAGES.length - 1 && (
                <div style={{ width: 12, textAlign: 'center', fontSize: 10, opacity: 0.2, color: 'var(--text-muted)', flexShrink: 0 }}>→</div>
              )}
            </React.Fragment>
          )
        })}
      </div>

      {/* ── Two-grade hero ────────────────────────────────── */}
      {(row.retrieval_grade != null || row.synthesis_grade != null) && (
        <TwoGradeBar
          retrieval={row.retrieval_grade}
          synthesis={row.synthesis_grade}
          gap={gapGrade}
        />
      )}

      {/* ── Routing decision ─────────────────────────────── */}
      {scoreEntries.length > 0 && (
        <details style={{ marginTop: 10 }} open>
          <summary style={{ cursor: 'pointer', fontWeight: 600, padding: '5px 0', userSelect: 'none' }}>
            Routing Decision
            {argmax && (
              <span style={{ fontSize: 11, fontWeight: 400, opacity: 0.45, marginLeft: 8 }}>
                linear → {argmax} ({s2(stratScores[argmax] ?? 0)})
              </span>
            )}
          </summary>
          <div style={{ paddingTop: 8 }}>
            {fv && Object.keys(fv).length > 0 && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginBottom: 10 }}>
                {Object.entries(fv).map(([k, v]) => (
                  <span key={k} style={{
                    fontSize: 11, padding: '2px 6px', borderRadius: 4,
                    border: '1px solid var(--border)',
                    background: v > 0.5 ? 'rgba(109,40,217,0.07)' : 'transparent',
                    opacity: v === 0 ? 0.35 : 1, fontVariantNumeric: 'tabular-nums',
                  }}>
                    {k} <strong>{v != null ? v.toFixed(2) : '—'}</strong>
                  </span>
                ))}
              </div>
            )}
            <div style={{ marginBottom: 10 }}>
              {scoreEntries.map(([s, score]) => {
                const isWinner = s === argmax
                const pct = (score / maxScore) * 100
                return (
                  <div key={s} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 3 }}>
                    <span style={{ width: 18, fontSize: 11, fontWeight: isWinner ? 700 : 400, opacity: isWinner ? 1 : 0.55 }}>{s}</span>
                    <div style={{ flex: 1, height: 10, borderRadius: 2, overflow: 'hidden', background: 'var(--surface-alt, #f3f4f6)' }}>
                      <div style={{ height: '100%', width: `${pct}%`, background: isWinner ? 'var(--rag-accent, #6d28d9)' : '#9ca3af', transition: 'width 0.25s' }} />
                    </div>
                    <span style={{ fontSize: 11, width: 34, textAlign: 'right', fontVariantNumeric: 'tabular-nums', fontWeight: isWinner ? 600 : 400, opacity: isWinner ? 1 : 0.55 }}>
                      {s2(score)}
                    </span>
                  </div>
                )
              })}
            </div>
            {argmax && fv && Object.keys(LIN_WEIGHTS[argmax] ?? {}).length > 0 && (
              <div style={{ fontSize: 12, borderTop: '1px solid var(--border)', paddingTop: 8, marginBottom: 8 }}>
                <div style={{ fontWeight: 600, marginBottom: 5, opacity: 0.65 }}>Why {argmax} won — contribution table</div>
                <table style={{ borderCollapse: 'collapse', width: '100%', fontVariantNumeric: 'tabular-nums' }}>
                  <tbody>
                    <tr>
                      <td style={{ padding: '2px 4px', opacity: 0.55 }}>base</td>
                      <td />
                      <td style={{ padding: '2px 4px', textAlign: 'right' }}>{(LIN_BASE[argmax] ?? 0).toFixed(2)}</td>
                    </tr>
                    {Object.entries(LIN_WEIGHTS[argmax] ?? {})
                      .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
                      .map(([feat, w]) => {
                        const fval = fv[feat] ?? 0
                        const contrib = w * fval
                        return (
                          <tr key={feat} style={{ opacity: fval !== 0 ? 1 : 0.3 }}>
                            <td style={{ padding: '2px 4px', opacity: 0.6 }}>{feat}</td>
                            <td style={{ padding: '2px 4px', opacity: 0.45, fontSize: 11 }}>
                              {fval.toFixed(2)} × {w >= 0 ? '+' : ''}{w.toFixed(2)}
                            </td>
                            <td style={{ padding: '2px 4px', textAlign: 'right', color: contrib > 0 ? '#16a34a' : contrib < 0 ? '#dc2626' : undefined }}>
                              {contrib >= 0 ? '+' : ''}{contrib.toFixed(2)}
                            </td>
                          </tr>
                        )
                      })}
                    <tr style={{ borderTop: '1px solid var(--border)' }}>
                      <td style={{ padding: '3px 4px', fontWeight: 600 }}>total</td>
                      <td />
                      <td style={{ padding: '3px 4px', textAlign: 'right', fontWeight: 600 }}>{s2(stratScores[argmax] ?? 0)}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            )}
            {argmax && scoreEntries.length > 1 && (
              <div style={{ fontSize: 11, opacity: 0.5 }}>
                Gap to runner-up ({scoreEntries[1][0]}): {routeGap.toFixed(2)}
                {' → '}
                {invokeAll && invokeAll.length > 1
                  ? `multi-invoke union [${invokeAll.join('+')}]`
                  : isUnion ? 'narrow gap — may multi-invoke' : 'single arm'}
              </div>
            )}
          </div>
        </details>
      )}

      {/* ── Action tree ──────────────────────────────────── */}
      {row.leaf_key && (
        <details style={{ marginTop: 8 }} open>
          <summary style={{ cursor: 'pointer', fontWeight: 600, padding: '5px 0', userSelect: 'none' }}>
            Action Tree
            <span style={{ fontSize: 11, fontWeight: 400, opacity: 0.45, marginLeft: 8 }}>
              {row.leaf_key} · green=taken · dashed=planned
            </span>
          </summary>
          <div style={{ paddingTop: 8, display: 'flex', flexWrap: 'wrap', gap: 5 }}>
            {TREE_NODES.map((node) => {
              const isTaken = taken.has(node.id)
              let nodeStyle: React.CSSProperties
              if (node.status === 'live' && isTaken) {
                nodeStyle = { padding: '3px 8px', fontSize: 11, borderRadius: 4, border: '1px solid #16a34a', color: '#15803d', background: 'rgba(22,163,74,0.08)', fontWeight: 600 }
              } else if (node.status === 'live') {
                nodeStyle = { padding: '3px 8px', fontSize: 11, borderRadius: 4, border: '1px solid var(--border)', color: 'var(--text-muted)', opacity: 0.6 }
              } else if (node.status === 'partial') {
                nodeStyle = { padding: '3px 8px', fontSize: 11, borderRadius: 4, border: '1px dashed #d97706', color: '#92400e', background: 'rgba(217,119,6,0.05)' }
              } else {
                nodeStyle = { padding: '3px 8px', fontSize: 11, borderRadius: 4, border: '1px dashed var(--border)', color: 'var(--text-muted)', opacity: 0.38 }
              }
              return (
                <span key={node.id} style={nodeStyle}
                  title={node.status === 'not_built' ? 'Planned — not yet built' : node.status === 'partial' ? 'Partially built' : isTaken ? 'Taken this query' : 'Built, not taken'}>
                  {isTaken ? '→ ' : ''}{node.label}
                </span>
              )
            })}
          </div>
        </details>
      )}

      {/* ── Per-claim ledger ─────────────────────────────── */}
      {ledger && ledger.length > 0 && (
        <details style={{ marginTop: 8 }} open>
          <summary style={{ cursor: 'pointer', fontWeight: 600, padding: '5px 0', userSelect: 'none' }}>
            Per-Claim Ledger
            <span style={{ fontSize: 11, fontWeight: 400, opacity: 0.45, marginLeft: 8 }}>
              {validated}/{totalClaims} validated
            </span>
          </summary>
          <div style={{ paddingTop: 6 }}>
            <PerClaimLedger claims={ledger} />
          </div>
        </details>
      )}

      {/* ── No grade data notice ─────────────────────────── */}
      {row.retrieval_grade == null && row.synthesis_grade == null && !ledger?.length && (
        <div style={{ padding: '10px 0', fontSize: 12, opacity: 0.5 }}>
          No grade data for this decision — grades populate after the EVAL agent runs a synthesis pass.
        </div>
      )}
    </div>
  )
}
