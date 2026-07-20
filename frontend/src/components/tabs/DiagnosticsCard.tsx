/**
 * DiagnosticsCard — unified glance + drill component for RAG query traces.
 *
 * Three mount modes:
 *   chat        — glance strip only, drill collapsed behind a toggle
 *   eval        — glance + drill expanded to level 1
 *   diagnostics — glance + drill fully open, deeper default depth
 *
 * Data contract: accepts a DiagnosticsTree built by the caller from
 * rag_query_decisions + full_response telemetry. EVAL Agent owns the
 * authoritative field→level mapping; this component is tree-agnostic.
 *
 * Usage:
 *   import { DiagnosticsCard, STUB_TREE } from './DiagnosticsCard'
 *   <DiagnosticsCard tree={myTree} mode="eval" />
 */
import { useState, useCallback } from 'react'
import './DiagnosticsCard.css'

// ─────────────────────────────────────────────
// Data types (interface EVAL Agent satisfies)
// ─────────────────────────────────────────────

export interface DiagnosticsTree {
  query: string
  answer: string
  route: { strategy: string; confidence: 'high' | 'medium' | 'low' | string }
  focusTags: string[]
  grades: { retrieval: number | null; synthesis: number | null; gap: number | null }
  claims: { passed: number; total: number } | null
  latencyMs: number
  decisionId: string
  root: TreeNode
}

export interface TreeNode {
  id: string
  title: string
  summary: string
  latencyMs?: number
  status: 'ok' | 'warn' | 'gray'
  children?: TreeNode[]
  telemetry?: Record<string, unknown>
  strategyScores?: Record<string, number>  // present at routing step → triggers "why X won" view
}

export type MountMode = 'chat' | 'eval' | 'diagnostics'

interface Props {
  tree: DiagnosticsTree
  mode?: MountMode
  className?: string
}

// ─────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────

function StatusDot({ status }: { status: TreeNode['status'] }) {
  return <span className={`dc-dot dc-dot-${status}`} aria-hidden />
}

function GradeBar({ value, label }: { value: number | null; label: string }) {
  const pct = value != null ? Math.round(value * 100) : 0
  const warn = value != null && value < 0.5
  return (
    <span className="dc-grade-bar" title={`${label}: ${value != null ? pct + '%' : '—'}`}>
      <span className="dc-grade-label">{label}</span>
      <span className="dc-grade-track">
        <span className={`dc-grade-fill${warn ? ' dc-grade-warn' : ''}`} style={{ width: `${pct}%` }} />
      </span>
      <span className="dc-grade-val">{value != null ? pct + '%' : '—'}</span>
    </span>
  )
}

function StrategyCompare({ scores, winner }: { scores: Record<string, number>; winner: string }) {
  const sorted = Object.entries(scores).sort(([, a], [, b]) => b - a)
  const max = sorted[0]?.[1] ?? 1
  return (
    <div className="dc-strategy-compare">
      <div className="dc-strategy-title">Why {winner} won</div>
      {sorted.map(([strat, score]) => (
        <div key={strat} className={`dc-strat-row${strat === winner ? ' dc-strat-winner' : ''}`}>
          <span className="dc-strat-label">{strat}</span>
          <div className="dc-strat-track">
            <div className="dc-strat-fill" style={{ width: `${(score / max) * 100}%` }} />
          </div>
          <span className="dc-strat-score">{score.toFixed(3)}</span>
        </div>
      ))}
    </div>
  )
}

function TelemetryLeaf({ data }: { data: Record<string, unknown> }) {
  return (
    <div className="dc-telemetry">
      <table className="dc-telem-table">
        <tbody>
          {Object.entries(data).map(([k, v]) => (
            <tr key={k}>
              <td className="dc-telem-key">{k}</td>
              <td className="dc-telem-val">
                {typeof v === 'object' ? JSON.stringify(v) : String(v)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function fmtMs(ms?: number): string {
  if (ms == null) return '—'
  return ms >= 1000 ? (ms / 1000).toFixed(1) + 's' : ms + 'ms'
}

// ─────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────

export function DiagnosticsCard({ tree, mode = 'eval', className = '' }: Props) {
  const [drillOpen, setDrillOpen] = useState(mode !== 'chat')
  // path = breadcrumb stack; always starts at root
  const [path, setPath] = useState<TreeNode[]>([tree.root])

  const current = path[path.length - 1]
  const isLeaf = !current.children?.length

  const drillInto = useCallback((node: TreeNode) => {
    setPath(p => [...p, node])
  }, [])

  const crumbTo = useCallback((idx: number) => {
    setPath(p => p.slice(0, idx + 1))
  }, [])

  const goBack = useCallback(() => {
    setPath(p => p.length > 1 ? p.slice(0, -1) : p)
  }, [])

  const { route, focusTags, grades, claims, latencyMs } = tree

  return (
    <div className={`dc dc-${mode}${className ? ' ' + className : ''}`}>

      {/* ══════════════════ GLANCE STRIP ══════════════════ */}
      <div className="dc-glance">

        {/* Question + answer */}
        <div className="dc-qa">
          <span className="dc-question">{tree.query}</span>
          <span className="dc-answer">{tree.answer}</span>
        </div>

        {/* Thinking strip */}
        <div className="dc-strip">

          {/* Route badge: a · high */}
          <span className={`dc-route-badge dc-conf-${route.confidence}`}>
            {route.strategy} · {route.confidence}
          </span>

          {/* Focused on */}
          {focusTags.length > 0 && (
            <span className="dc-focused">
              <span className="dc-focused-label">focused on</span>
              {focusTags.map(t => <span key={t} className="dc-tag-chip">{t}</span>)}
            </span>
          )}

          {/* Retrieval + synthesis mini-bars */}
          <span className="dc-grades">
            <GradeBar value={grades.retrieval} label="ret" />
            <GradeBar value={grades.synthesis} label="syn" />
          </span>

          {/* Claims count */}
          {claims && (
            <span className={`dc-claims${claims.passed === claims.total ? ' dc-claims-ok' : ' dc-claims-warn'}`}>
              {claims.passed}/{claims.total}&nbsp;{claims.passed === claims.total ? '✓' : '⚠'}
            </span>
          )}

          {/* Latency */}
          <span className="dc-latency">{fmtMs(latencyMs)}</span>

          {/* Expand toggle (chat mode only) */}
          {mode === 'chat' && (
            <button
              className="dc-toggle"
              onClick={() => setDrillOpen(o => !o)}
              title={drillOpen ? 'Collapse trace' : 'Expand trace'}
            >
              {drillOpen ? '▲' : '▼'}
            </button>
          )}

        </div>
      </div>

      {/* ══════════════════ DRILL PANEL ══════════════════ */}
      {drillOpen && (
        <div className="dc-drill">

          {/* Breadcrumb */}
          <nav className="dc-breadcrumb" aria-label="Trace navigation">
            {path.map((node, idx) => (
              <span key={node.id} className="dc-crumb-item">
                {idx > 0 && <span className="dc-crumb-sep" aria-hidden>›</span>}
                <button
                  className={`dc-crumb${idx === path.length - 1 ? ' dc-crumb-active' : ''}`}
                  onClick={() => idx < path.length - 1 && crumbTo(idx)}
                  disabled={idx === path.length - 1}
                >
                  {node.title}
                </button>
              </span>
            ))}
          </nav>

          {/* Level summary card */}
          <div className="dc-level-card">
            <div className="dc-level-header">
              <span className="dc-level-title">{current.title}</span>
              {current.latencyMs != null && (
                <span className="dc-level-timing">{fmtMs(current.latencyMs)}</span>
              )}
            </div>
            <div className="dc-level-summary">{current.summary}</div>
          </div>

          {/* Strategy comparison — routing nodes only */}
          {current.strategyScores && (
            <StrategyCompare scores={current.strategyScores} winner={route.strategy} />
          )}

          {/* Child rows — drill deeper */}
          {!isLeaf && current.children && (
            <div className="dc-children" role="list">
              {current.children.map(child => (
                <button
                  key={child.id}
                  className="dc-child-row"
                  role="listitem"
                  onClick={() => drillInto(child)}
                >
                  <StatusDot status={child.status} />
                  <span className="dc-child-title">{child.title}</span>
                  <span className="dc-child-summary">{child.summary}</span>
                  {child.latencyMs != null && (
                    <span className="dc-child-time">{fmtMs(child.latencyMs)}</span>
                  )}
                  <span className="dc-child-chevron" aria-hidden>›</span>
                </button>
              ))}
            </div>
          )}

          {/* Telemetry leaf */}
          {isLeaf && current.telemetry && (
            <TelemetryLeaf data={current.telemetry} />
          )}

          {/* Back */}
          {path.length > 1 && (
            <button className="dc-back" onClick={goBack}>← back</button>
          )}

        </div>
      )}

    </div>
  )
}

// ─────────────────────────────────────────────
// Stub tree — development + storybook
// EVAL Agent replaces this with the real mapping.
// Shape: Loop → Classify / Route / Retrieve·a / Synthesize
// ─────────────────────────────────────────────

export const STUB_TREE: DiagnosticsTree = {
  query: 'Why was this claim denied and how do I fix it?',
  answer: 'The claim was denied with CARC 197: prior authorization not on file. The denial is typically recoverable through a retroactive authorization request, where the payer permits it, or a formal appeal supported by medical-necessity documentation.',
  route: { strategy: 'a', confidence: 'high' },
  focusTags: ['sunshine_health', 'prior_auth', 'denial'],
  grades: { retrieval: 0.82, synthesis: 0.74, gap: 0.08 },
  claims: { passed: 6, total: 6 },
  latencyMs: 1140,
  decisionId: 'stub-001',
  root: {
    id: 'loop',
    title: 'Loop',
    summary: 'Single-pass · strategy a · 1140ms total · 6/6 claims',
    latencyMs: 1140,
    status: 'ok',
    children: [
      {
        id: 'classify',
        title: 'Classify',
        summary: 'denial_appeal · exclusivity 0.82 · literal match: CARC 197',
        latencyMs: 48,
        status: 'ok',
        children: [{
          id: 'classify-telem',
          title: 'Classifier telemetry',
          summary: 'Full feature vector',
          status: 'ok',
          telemetry: {
            query_type: 'denial_appeal', coverage: 'focused',
            exclusivity: 0.82, literal: 0.91, corpus_depth: 0.67,
            thematic_policy: 0.44,
            tag_matches: ['sunshine_health', 'prior_auth', 'denial'],
          },
        }],
      },
      {
        id: 'route',
        title: 'Route',
        summary: 'Strategy a chosen · BM25 cascade · conf high · linear router',
        latencyMs: 12,
        status: 'ok',
        strategyScores: { a: 0.782, b: 0.441, c: 0.119, d: 0.094 },
        children: [{
          id: 'route-telem',
          title: 'Router telemetry',
          summary: 'Linear score breakdown + priors',
          status: 'ok',
          telemetry: {
            routing_method: 'linear', strategy_chosen: 'a',
            strategy_executed: 'a', confidence: 'high',
            priors_version: 'v3', fail_fast_reason: null,
          },
        }],
      },
      {
        id: 'retrieve-a',
        title: 'Retrieve · a',
        summary: '12 chunks · tag-boosted · top score 0.89',
        latencyMs: 640,
        status: 'ok',
        children: [{
          id: 'merge',
          title: 'Merge + tag-boost',
          summary: '18 → 12 after dedup · sunshine_health ×1.5',
          latencyMs: 38,
          status: 'ok',
          children: [{
            id: 'chunk-scores',
            title: 'Raw chunk scores',
            summary: 'Top 5 of 12',
            status: 'ok',
            telemetry: {
              chunk_1: 0.89, chunk_3: 0.84, chunk_7: 0.81,
              chunk_2: 0.77, chunk_9: 0.72,
            },
          }],
        }],
      },
      {
        id: 'synthesize',
        title: 'Synthesize',
        summary: 'synthesis_grade 0.74 · 6/6 claims · gap 0.08',
        latencyMs: 380,
        status: 'warn',
        children: [{
          id: 'synth-telem',
          title: 'Synthesizer telemetry',
          summary: 'Per-claim ledger + model metadata',
          status: 'warn',
          telemetry: {
            synthesis_grade: 0.74, synthesis_gap: 0.08,
            claims_passed: 6, claims_total: 6,
            finish_reason: 'stop', model: 'claude-sonnet-4-6',
          },
        }],
      },
    ],
  },
}
