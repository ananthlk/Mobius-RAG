import { Fragment, useEffect, useMemo, useState } from 'react'
import { API_BASE } from '../../config'
import { EntityCard } from './repository/EntityCard'
import { ReaderSlideOut } from './repository/ReaderSlideOut'
import { EntitySidebar, domainOf } from './repository/EntitySidebar'
import type { HostStatEnriched, CorpusStats, DomainFilter } from './repository/EntitySidebar'
import { SearchTracePanel } from './repository/SearchTracePanel'
import type { SearchTelemetry } from './repository/SearchTracePanel'
import { UploadedDocsPanel } from './repository/UploadedDocsPanel'
import './RepositoryTab.css'

// suppress unused-import warnings — these are kept per spec
void EntityCard
void EntitySidebar
void domainOf
void UploadedDocsPanel

const UPLOADED_HOST = '(uploaded)'

// ── Types ─────────────────────────────────────────────────────────────────────

interface DocLike {
  id: string
  filename: string
  display_name?: string | null
  payer?: string | null
  state?: string | null
  program?: string | null
  authority_level?: string | null
  effective_date?: string | null
  termination_date?: string | null
  status?: string | null
  extraction_status?: string
  chunking_status?: string | null
  embedding_status?: string | null
  published_at?: string | null
  source_metadata?: { source_url?: string | null } | null
  source_url?: string | null
}

interface NavigateToRead {
  documentId: string
  pageNumber?: number
  factId?: string
  citeText?: string
}

interface RawHostStat {
  host: string
  count: number
  corpusDocs: number
  corpusPublished: number
}

interface StatsResponse {
  by_host: Record<string, number>
  by_curation_status: Record<string, number>
  by_content_kind: Record<string, number>
  by_ingested: Record<string, number>
}

interface CorpusByHost {
  by_host: Record<string, { docs: number; published: number }>
}

/** Chunk returned by /api/skills/v1/corpus_search */
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
}

interface CorpusSearchResponse {
  chunks: CorpusChunk[]
  telemetry: SearchTelemetry
}

type SearchMode = 'corpus' | 'precision' | 'recall'

interface Props {
  documents: DocLike[]
  documentsLoading?: boolean
  selectedDocumentId: string | null
  navigateToRead: NavigateToRead | null
  onNavigateToReadConsumed: () => void
  onDocumentSelect: (documentId: string) => void
  onRefresh?: () => void
}

// ── Corpus search results ─────────────────────────────────────────────────────

const CONFIDENCE_COLORS: Record<string, string> = {
  high: '#10b981',
  medium: '#f59e0b',
  low: '#6b7280',
  abstain: '#d1d5db',
}

const ARM_LABELS: Record<string, string> = {
  bm25: 'BM25',
  vector: 'Vector',
}

// Kept (referenced for type compatibility) but not rendered after the
// 2026-04-29 cleanup that moved corpus search to the Test tab.
// @ts-ignore
function _CorpusSearchResults({
  query,
  mode,
  chunks,
  loading,
  telemetry,
  onOpenChunk,
}: {
  query: string
  mode: SearchMode
  chunks: CorpusChunk[] | null
  loading: boolean
  telemetry: SearchTelemetry | null
  onOpenChunk: (chunk: CorpusChunk) => void
}) {
  const [showTrace, setShowTrace] = useState(false)

  if (loading) return <div className="repo-search-status">Searching corpus…</div>
  if (!chunks) return null
  if (chunks.length === 0) {
    return (
      <div className="repo-search-status">
        No results for &ldquo;{query}&rdquo;
        {telemetry && (
          <span className="repo-search-telem">
            {' '}({telemetry.total_ms?.toFixed(0) ?? '?'}ms)
          </span>
        )}
      </div>
    )
  }

  const modeLabel = mode === 'corpus' ? 'Hybrid' : mode === 'precision' ? 'BM25' : 'Semantic'
  const hasTrace = !!(
    telemetry?.scoring_trace?.length ||
    telemetry?.arm_results?.bm25?.length ||
    telemetry?.arm_results?.vector?.length
  )

  return (
    <div className="repo-corpus-results">
      <div className="repo-corpus-results-hdr">
        <span>
          {chunks.length} result{chunks.length !== 1 ? 's' : ''} for{' '}
          <strong>&ldquo;{query}&rdquo;</strong>
        </span>
        <span className="repo-corpus-results-meta">
          {modeLabel}
          {telemetry && (
            <> · {telemetry.total_ms?.toFixed(0) ?? '?'}ms</>
          )}
          {telemetry?.arm_hits && (
            <> · BM25 {telemetry.arm_hits.bm25 ?? 0} / Vec {telemetry.arm_hits.vector ?? 0}</>
          )}
        </span>
        {/* Trace toggle */}
        {hasTrace && (
          <button
            type="button"
            className={`repo-corpus-trace-btn${showTrace ? ' active' : ''}`}
            onClick={() => setShowTrace((v) => !v)}
            title={showTrace ? 'Hide pipeline trace' : 'Show pipeline trace'}
          >
            {showTrace ? '▲ Trace' : '▼ Trace'}
          </button>
        )}
      </div>

      <ul className="repo-corpus-results-list">
        {chunks.map((chunk) => (
          <li key={chunk.id} className="repo-corpus-result-item">
            <button
              type="button"
              className="repo-corpus-result-btn"
              onClick={() => onOpenChunk(chunk)}
            >
              {/* Header row */}
              <div className="repo-corpus-result-head">
                <span className="repo-corpus-result-docname">{chunk.document_name}</span>
                {chunk.page_number != null && (
                  <span className="repo-corpus-result-page">p.{chunk.page_number}</span>
                )}
                {/* Arms */}
                <span className="repo-corpus-result-arms">
                  {chunk.retrieval_arms.map((arm) => (
                    <span key={arm} className={`repo-corpus-arm-badge arm-${arm}`}>
                      {ARM_LABELS[arm] ?? arm}
                    </span>
                  ))}
                </span>
                {/* Confidence */}
                <span
                  className="repo-corpus-confidence"
                  style={{ color: CONFIDENCE_COLORS[chunk.confidence_label] }}
                  title={`rerank score: ${chunk.rerank_score.toFixed(3)}`}
                >
                  {chunk.confidence_label}
                </span>
              </div>
              {/* Snippet */}
              <p className="repo-corpus-result-snippet">
                {chunk.text.slice(0, 220)}{chunk.text.length > 220 ? '…' : ''}
              </p>
              {/* Meta */}
              {(chunk.payer || chunk.authority_level) && (
                <div className="repo-corpus-result-footer">
                  {chunk.payer && <span className="repo-corpus-result-payer">{chunk.payer}</span>}
                  {chunk.authority_level && (
                    <span className="repo-corpus-result-auth">{chunk.authority_level}</span>
                  )}
                </div>
              )}
            </button>
          </li>
        ))}
      </ul>

      {/* Pipeline trace panel — expandable */}
      {showTrace && telemetry && (
        <div className="repo-corpus-trace-wrap">
          <SearchTracePanel telemetry={telemetry} />
        </div>
      )}
    </div>
  )
}

// ── StatusPanel component ─────────────────────────────────────────────────────
// Extracted from the inline IIFE so hooks are called at the component top level.

interface StatusPanelProps {
  documents: DocLike[]
  onRefresh?: () => void
}

function StatusPanel({ documents, onRefresh }: StatusPanelProps) {
  const [health, setHealth] = useState<any>(null)
  const [expanded, setExpanded] = useState(false)
  const [stageExpanded, setStageExpanded] = useState<Record<string, boolean>>({})

  // ── One-touch data integrity (check + fix all) ──────────────────────────
  const [integ, setInteg] = useState<any>(null)
  const [integBusy, setIntegBusy] = useState(false)
  const [remediate, setRemediate] = useState<any>(null)
  const [integErr, setIntegErr] = useState<string | null>(null)

  const checkIntegrity = async () => {
    setIntegBusy(true); setIntegErr(null)
    try {
      const r = await fetch(`${API_BASE}/admin/integrity/report`)
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      setInteg(await r.json())
    } catch (e) { setIntegErr(String(e)) } finally { setIntegBusy(false) }
  }

  const runRemediate = async () => {
    // No confirm() — blocked in the platform iframe (see PipelinePanel.run).
    // Fix-all is idempotent remediation, safe to trigger directly.
    setIntegBusy(true); setIntegErr(null)
    try {
      const r = await fetch(`${API_BASE}/admin/integrity/remediate`, { method: 'POST' })
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      // Poll status until the queueing pass finishes (fast — it only enqueues).
      for (let i = 0; i < 30; i++) {
        await new Promise(res => setTimeout(res, 1500))
        const s = await fetch(`${API_BASE}/admin/integrity/remediate/status`)
        if (s.ok) {
          const st = await s.json(); setRemediate(st)
          if (!st.running) break
        }
      }
      await checkIntegrity()
    } catch (e) { setIntegErr(String(e)) } finally { setIntegBusy(false) }
  }

  useEffect(() => {
    let alive = true
    const tick = async () => {
      try {
        const r = await fetch(`${API_BASE}/pipeline_health`)
        if (r.ok && alive) setHealth(await r.json())
      } catch { /* ignore */ }
    }
    tick()
    const id = setInterval(tick, 10_000)
    return () => { alive = false; clearInterval(id) }
  }, [])

  const dotColor = (s?: string) =>
    s === 'green' ? '#10b981' : s === 'yellow' ? '#f59e0b' : s === 'red' ? '#ef4444' : '#cbd5e1'

  const Dot = ({ s, title }: { s?: string; title: string }) => (
    <span
      title={title}
      style={{
        display: 'inline-block', width: 8, height: 8, borderRadius: 999,
        background: dotColor(s), marginRight: 4,
      }}
      aria-label={title}
    />
  )

  const fmtEta = (pending: number, perHour: number) => {
    if (!perHour || perHour <= 0) return pending > 0 ? '—' : '✓ caught up'
    const hours = pending / perHour
    if (hours < 1) return `${Math.round(hours * 60)} min`
    if (hours < 24) return `${hours.toFixed(1)} h`
    return `${(hours / 24).toFixed(1)} d`
  }

  const fmtSeconds = (s: number | null | undefined) => {
    if (s == null || !isFinite(s) || s <= 0) return '—'
    const h = s / 3600
    if (h < 1) return `${Math.round(s / 60)} min`
    if (h < 24) return `${h.toFixed(1)} h`
    return `${(h / 24).toFixed(1)} d`
  }

  const Spark = ({ buckets }: { buckets: number[] }) => {
    const w = 78, h = 18, max = Math.max(1, ...buckets)
    const bw = (w - (buckets.length - 1) * 2) / buckets.length
    const bucketSpanMin = 5
    const totalBuckets = buckets.length
    return (
      <svg width={w} height={h} style={{ verticalAlign: 'middle' }}>
        {buckets.map((n, i) => {
          const bh = (n / max) * (h - 2)
          const minAgoEnd = (totalBuckets - 1 - i) * bucketSpanMin
          const minAgoStart = minAgoEnd + bucketSpanMin
          const span = minAgoEnd === 0
            ? `last ${bucketSpanMin} min`
            : `${minAgoEnd}-${minAgoStart} min ago`
          return (
            <rect
              key={i}
              x={i * (bw + 2)}
              y={h - bh}
              width={bw}
              height={Math.max(1, bh)}
              fill="#10b981"
              opacity={n === 0 ? 0.25 : 1}
              style={{ cursor: 'help' }}
            >
              <title>{`${span}: ${n} ${n === 1 ? 'event' : 'events'}`}</title>
            </rect>
          )
        })}
      </svg>
    )
  }

  const t = health?.totals
  const total = t?.documents ?? documents.length
  const chunked = t?.chunked ?? documents.filter((d: any) => d.chunking_status === 'completed').length
  const embedded = t?.embedded ?? documents.filter((d: any) => d.embedding_status === 'completed').length
  const published = t?.published ?? documents.filter((d: any) => !!d.published_at).length
  const failed = documents.filter((d: any) =>
    d.chunking_status === 'failed' || d.embedding_status === 'failed').length
  const processing = total - published - failed
  const chk = health?.chunking
  const emb = health?.embedding
  const pub = health?.publishing

  const stageMeta: Array<{ label: string; n: number; terminal?: boolean; dot?: any; tip?: string }> = [
    { label: 'Documents', n: total },
    {
      label: 'Chunked', n: chunked,
      dot: chk?.status,
      tip: chk
        ? `Chunking: ${chk.status}\n${chk.active} workers active · ${chk.last_hour}/h · ${chk.pending} pending`
        : 'Chunking status loading…',
    },
    {
      label: 'Embedded', n: embedded,
      dot: emb?.status,
      tip: emb
        ? `Embedding: ${emb.status}\n${emb.active} workers active · ${emb.last_hour}/h · ${emb.pending} pending`
        : 'Embedding status loading…',
    },
    {
      label: 'Available in chat', n: published, terminal: true,
      dot: pub?.status,
      tip: pub
        ? `Publishing: ${pub.status}\n${pub.last_hour}/h · ${pub.embedded_unpublished} embedded but unpublished`
        : 'Publishing status loading…',
    },
  ]

  return (
    <>
      {/* Summary bar */}
      <div
        className="repo-corpus-status"
        role="group"
        aria-label="Overall corpus pipeline status"
        style={{ borderBottom: '1px solid #eee', fontSize: 13, marginBottom: 12 }}
      >
        <div
          onClick={() => setExpanded(e => !e)}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setExpanded(x => !x) }}
          aria-expanded={expanded}
          style={{
            display: 'flex', alignItems: 'center', gap: 12,
            padding: '8px 12px', cursor: 'pointer',
            userSelect: 'none', flexWrap: 'wrap',
          }}
        >
          <span
            style={{ marginRight: 4, fontSize: 11, color: '#888', width: 12, display: 'inline-block' }}
            aria-hidden
          >{expanded ? '▼' : '▶'}</span>
          <strong style={{ marginRight: 4 }}>Corpus</strong>
          {stageMeta.map((s, i) => (
            <span key={s.label} style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
              {s.dot !== undefined && <Dot s={s.dot} title={s.tip || s.label} />}
              <span
                title={s.tip}
                style={{
                  fontWeight: 600,
                  color: s.terminal ? '#10b981' : (s.n > 0 ? '#111' : '#999'),
                }}
              >{s.n.toLocaleString()}</span>
              <span style={{ color: '#666' }} title={s.tip}>{s.label}</span>
              {i < stageMeta.length - 1 && <span style={{ color: '#ccc', marginLeft: 8 }}>→</span>}
            </span>
          ))}
          {processing > 0 && (
            <span style={{ marginLeft: 'auto', color: '#f59e0b' }}>
              {processing.toLocaleString()} processing
            </span>
          )}
          {failed > 0 && (
            <span style={{ color: '#ef4444' }}>{failed.toLocaleString()} failed</span>
          )}
          {health?.integrity && (
            <span
              style={{
                display: 'inline-flex', alignItems: 'center', gap: 4,
                marginLeft: processing > 0 || failed > 0 ? 0 : 'auto',
                paddingLeft: 8, borderLeft: '1px solid #eee',
              }}
              title={
                health.integrity.chat_orphans === 0
                  ? 'Integrity OK\nrag and chat are fully aligned'
                  : `Integrity drift\n${health.integrity.chat_orphans} doc(s) shown by chat that rag does NOT have.\nFix: POST /admin/cleanup_chat_orphans?dry_run=false`
              }
            >
              <Dot s={health.integrity.status} title="Integrity status" />
              <span style={{ color: '#666' }}>Integrity</span>
              {health.integrity.chat_orphans > 0 && (
                <span style={{ color: '#ef4444', fontWeight: 600 }}>
                  {health.integrity.chat_orphans} drift
                </span>
              )}
            </span>
          )}
        </div>

        {/* Expanded panel — per-stage rate, pending, ETA */}
        {expanded && health && (
          <div
            style={{
              padding: '10px 14px 14px 28px',
              background: '#fafbfc',
              borderTop: '1px solid #f0f0f0',
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
              gap: 12,
            }}
          >
            {[
              { label: 'Chunking', s: chk, terminalCount: chunked, inFlightLabel: 'Currently chunking', extraCol: 'paragraphs_done' },
              { label: 'Embedding', s: emb, terminalCount: embedded, inFlightLabel: 'Currently embedding', extraCol: 'chunks_done' },
              {
                label: 'Publishing',
                s: pub
                  ? { ...pub, active: undefined, pending: pub.embedded_unpublished }
                  : null,
                terminalCount: published,
                inFlightLabel: 'Recently published (last 30 min)',
                extraCol: 'chunks_done',
              },
            ].map(({ label, s, terminalCount, inFlightLabel, extraCol }) => {
              const r = s?.rolling
              const inFlight: any[] = (s as any)?.in_flight || []
              const isStageExpanded = !!stageExpanded[label]
              return (
              <div
                key={label}
                style={{
                  background: '#fff',
                  border: '1px solid #ececec',
                  borderRadius: 6,
                  padding: '10px 12px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                  <Dot s={s?.status} title={`${label} status`} />
                  <strong style={{ fontSize: 13 }}>{label}</strong>
                  {inFlight.length > 0 && (
                    <span
                      onClick={(e) => {
                        e.stopPropagation()
                        setStageExpanded(prev => ({ ...prev, [label]: !prev[label] }))
                      }}
                      role="button"
                      style={{
                        marginLeft: 'auto', fontSize: 11, color: '#3b82f6',
                        cursor: 'pointer', userSelect: 'none',
                      }}
                      title="Click to see what's in flight"
                    >
                      {isStageExpanded ? '▼' : '▶'} {inFlight.length} in flight
                    </span>
                  )}
                </div>
                <div style={{ fontSize: 12, color: '#444', display: 'grid', gap: 3 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ color: '#888' }}>Last 30min:</span>{' '}
                    <span style={{ fontWeight: 600 }}>
                      {r ? `${r.rate_per_hour}/h` : `${s?.last_hour ?? 0}/h *`}
                    </span>
                    {r?.buckets_5min && <Spark buckets={r.buckets_5min} />}
                  </div>
                  {r && (
                    <div style={{ color: '#888', fontSize: 11 }}>
                      95% CI: {r.rate_lo_per_hour}–{r.rate_hi_per_hour}/h
                    </div>
                  )}
                  {s?.active != null && (
                    <div>
                      <span style={{ color: '#888' }}>Active workers:</span>{' '}
                      <span style={{ fontWeight: 600 }}>{s.active}</span>
                    </div>
                  )}
                  <div>
                    <span style={{ color: '#888' }}>Pending:</span>{' '}
                    <span style={{ fontWeight: 600 }}>{(s?.pending ?? 0).toLocaleString()}</span>
                  </div>
                  {r && r.eta_seconds_p50 != null ? (
                    <>
                      <div>
                        <span style={{ color: '#888' }}>ETA (median):</span>{' '}
                        <span style={{ fontWeight: 600 }}>{fmtSeconds(r.eta_seconds_p50)}</span>
                      </div>
                      <div style={{ color: '#888', fontSize: 11 }}>
                        95% CI: {fmtSeconds(r.eta_seconds_p5)} – {fmtSeconds(r.eta_seconds_p95)}
                      </div>
                    </>
                  ) : (
                    <div>
                      <span style={{ color: '#888' }}>ETA:</span>{' '}
                      <span style={{ fontWeight: 600, color: s?.last_hour ? '#111' : '#999' }}>
                        {fmtEta(s?.pending ?? 0, s?.last_hour ?? 0)}
                      </span>
                    </div>
                  )}
                  <div style={{ marginTop: 4, paddingTop: 4, borderTop: '1px dashed #f0f0f0' }}>
                    <span style={{ color: '#888' }}>Total {label.toLowerCase()}:</span>{' '}
                    <span style={{ fontWeight: 600 }}>{terminalCount.toLocaleString()}</span>
                  </div>
                </div>
                {/* In-flight panel */}
                {isStageExpanded && inFlight.length > 0 && (
                  <div
                    style={{
                      marginTop: 8, paddingTop: 8,
                      borderTop: '1px solid #f0f0f0',
                    }}
                  >
                    <div style={{ fontSize: 11, color: '#666', marginBottom: 4 }}>
                      {inFlightLabel}
                    </div>
                    <div
                      style={{
                        maxHeight: 220, overflowY: 'auto',
                        fontSize: 11, fontFamily: 'ui-monospace, monospace',
                      }}
                    >
                      {inFlight.map((d: any, i: number) => {
                        const elapsed = d.elapsed_s
                        const elapsedTxt = elapsed < 60
                          ? `${elapsed}s`
                          : elapsed < 3600
                            ? `${Math.floor(elapsed/60)}m${elapsed%60}s`
                            : `${(elapsed/3600).toFixed(1)}h`
                        const extra = d[extraCol] ?? 0
                        return (
                          <div
                            key={d.doc_id + ':' + i}
                            style={{
                              display: 'grid',
                              gridTemplateColumns: '1fr auto auto',
                              gap: 6, padding: '2px 0',
                              borderBottom: '1px dotted #f4f4f4',
                            }}
                            title={`${d.payer || '<unknown>'}\n${d.filename}\nelapsed: ${elapsedTxt}`}
                          >
                            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                              {d.filename || d.doc_id.slice(0, 8)}
                            </span>
                            <span style={{ color: '#666' }}>
                              {label === 'Chunking' && `${d.pages || 0}p · ${extra}¶`}
                              {label === 'Embedding' && `${extra} chunks`}
                              {label === 'Publishing' && `${extra} chunks`}
                            </span>
                            <span style={{ color: '#888', minWidth: 38, textAlign: 'right' }}>
                              {elapsedTxt}
                            </span>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
              </div>
            )})}
            {/* Integrity card */}
            {health.integrity && (() => {
              const ig = health.integrity
              const row = (label: string, n: number, hint: string) => (
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                  <span style={{ color: '#888' }} title={hint}>{label}:</span>
                  <span style={{ fontWeight: 600,
                                 color: n === 0 ? '#10b981' : (n >= 100 ? '#ef4444' : '#f59e0b') }}>
                    {n.toLocaleString()}
                  </span>
                </div>
              )
              return (
                <div
                  style={{
                    background: '#fff',
                    border: '1px solid #ececec',
                    borderRadius: 6,
                    padding: '10px 12px',
                    gridColumn: 'span 2',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                    <Dot s={ig.status} title="Integrity status" />
                    <strong style={{ fontSize: 13 }}>Integrity</strong>
                    <span style={{ marginLeft: 'auto', color: '#888', fontSize: 11 }}>
                      4 dimensions
                    </span>
                  </div>
                  <div style={{ fontSize: 12, color: '#444', display: 'grid', gap: 3 }}>
                    {row('Chat orphans', ig.chat_orphans ?? 0,
                      'docs visible to chat that rag does NOT have — phantom citation risk. Fix: POST /admin/cleanup_chat_orphans')}
                    {row('Sitemap orphans', ig.sitemap_orphans ?? 0,
                      'rag docs without a discovered_sources registry row — per-host counts undercount. Fix: POST /admin/backfill_sitemap')}
                    {row('Blocked jobs', ig.blocked_jobs ?? 0,
                      'chunking jobs at failure_count >= 3 — manual triage required. See: GET /admin/list_blocked_docs')}
                    {row('Metadata orphans', ig.metadata_orphans ?? 0,
                      'docs missing payer / state / program tags — cannot be filtered or attributed in retrieval')}
                  </div>
                </div>
              )
            })()}
          </div>
        )}
      </div>

      {/* Retag button */}
      {onRefresh && (
        <div style={{ padding: '0 12px 12px' }}>
          <button
            type="button"
            style={{
              fontSize: 11, padding: '0.2rem 0.6rem',
              border: '1px solid var(--mobius-border)',
              borderRadius: 'var(--mobius-radius-sm)',
              background: 'var(--mobius-bg-secondary)',
              color: 'var(--mobius-text-secondary)',
              cursor: 'pointer',
            }}
            onClick={onRefresh}
          >
            Retag documents
          </button>
        </div>
      )}

      {/* ── One-touch data integrity ─────────────────────────────────────── */}
      <div style={{ padding: '0 12px 14px' }}>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
          <strong style={{ fontSize: 12, color: 'var(--mobius-text-secondary)' }}>Data integrity</strong>
          {integ && <Dot s={integ.status} title={`Integrity: ${integ.status}`} />}
          <button type="button" disabled={integBusy}
            style={{ fontSize: 11, padding: '0.2rem 0.6rem', border: '1px solid var(--mobius-border)',
                     borderRadius: 'var(--mobius-radius-sm)', background: 'var(--mobius-bg-secondary)',
                     color: 'var(--mobius-text-secondary)', cursor: integBusy ? 'default' : 'pointer' }}
            onClick={checkIntegrity}>{integBusy ? 'Checking…' : 'Check integrity'}</button>
          {integ && (integ.gaps?.unpublished_total > 0 || integ.gaps?.failed_docs > 0 ||
                     integ.gaps?.stuck_docs > 0 || integ.gaps?.stale_tags > 0 ||
                     integ.gaps?.cancelled_jobs_prunable > 0 || integ.gaps?.sitemap_orphans > 0 ||
                     integ.gaps?.metadata_orphans > 0) && (
            <button type="button" disabled={integBusy}
              style={{ fontSize: 11, padding: '0.2rem 0.6rem', border: '1px solid #ef4444',
                       borderRadius: 'var(--mobius-radius-sm)', background: 'rgba(239,68,68,0.08)',
                       color: '#b91c1c', cursor: integBusy ? 'default' : 'pointer' }}
              onClick={runRemediate}>{integBusy ? 'Working…' : 'Fix all'}</button>
          )}
        </div>
        {integErr && <div style={{ fontSize: 11, color: '#b91c1c' }}>{integErr}</div>}
        {integ && (
          <div style={{ fontSize: 11, color: 'var(--mobius-text-secondary)', display: 'grid',
                        gridTemplateColumns: '1fr auto', gap: '1px 10px' }}>
            {([
              ['Published', `${integ.published} / ${integ.documents_total}`],
              ['Need publish (have vectors)', integ.gaps?.need_publish],
              ['Need re-embed (have chunks)', integ.gaps?.need_embed],
              ['Need re-chunk (has pages · Path B)', integ.gaps?.need_rechunk],
              ['Need re-ingest (no pages)', integ.gaps?.need_reingest],
              ['Stuck (extracting/extracted)', integ.gaps?.stuck_docs],
              ['Blocked jobs', integ.gaps?.blocked_jobs],
              ['Cancelled jobs (prunable)', integ.gaps?.cancelled_jobs_prunable],
              ['Stale tags (need retag)', integ.gaps?.stale_tags],
              ['Sitemap orphans', integ.gaps?.sitemap_orphans],
              ['Metadata orphans', integ.gaps?.metadata_orphans],
            ] as [string, any][]).map(([label, val]) => (
              <Fragment key={label}>
                <span>{label}</span>
                <span style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums',
                               color: (typeof val === 'number' && val > 0) ? '#b45309' : 'inherit' }}>
                  {val ?? '—'}
                </span>
              </Fragment>
            ))}
          </div>
        )}
        {integ?.latency && (
          <div style={{ fontSize: 11, color: 'var(--mobius-text-secondary)', marginTop: 6 }}>
            Latency (24h): chunk {integ.latency.chunking?.avg_s ?? 0}s avg / {integ.latency.chunking?.max_s ?? 0}s max
            · embed {integ.latency.embedding?.avg_s ?? 0}s avg / {integ.latency.embedding?.max_s ?? 0}s max
          </div>
        )}
        {remediate && (
          <div style={{ fontSize: 11, color: 'var(--mobius-text-secondary)', marginTop: 6 }}>
            Fix: pruned {remediate.pruned_jobs}, reset {remediate.stuck_reset}, re-chunk {remediate.rechunk_enqueued},
            re-embed {remediate.reembed_enqueued}{remediate.retag_triggered ? ', retag queued' : ''}
            {remediate.sitemap_backfilled ? ', sitemap fixed' : ''}{remediate.metadata_backfilled ? ', metadata fixed' : ''}
            {remediate.running ? ' …' : ' ✓'}
            {remediate.error && <span style={{ color: '#b91c1c' }}> — {remediate.error}</span>}
          </div>
        )}
      </div>
    </>
  )
}

// ── PipelinePanel — one-button nightly loop with live stage tracker ──────────
const _STEP_ORDER = ['infra_up', 'baseline_eval', 'publish', 'retag', 'chunk',
  'embed', 'gate', 'freeze', 'final_eval', 'push', 'infra_down', 'lift']

const _stepColor = (s?: string) =>
  s === 'done' ? '#10b981' : s === 'running' ? '#3b82f6'
    : s === 'failed' ? '#ef4444' : s === 'skipped' ? '#94a3b8' : '#cbd5e1'

function _fmtDur(a?: string | null, b?: string | null): string {
  if (!a) return ''
  const end = b ? new Date(b).getTime() : Date.now()
  const s = Math.max(0, (end - new Date(a).getTime()) / 1000)
  if (s < 60) return `${Math.round(s)}s`
  if (s < 3600) return `${Math.round(s / 60)}m`
  return `${(s / 3600).toFixed(1)}h`
}

function PipelinePanel({ documents, onRefresh }: StatusPanelProps) {
  const [rep, setRep] = useState<any>(null)
  const [runs, setRuns] = useState<any[]>([])
  const [nly, setNly] = useState<any>(null)          // GLOBAL orchestrator status (the live run, if any)
  const [selId, setSelId] = useState<string | null>(null)
  const [detail, setDetail] = useState<any>(null)    // fetched detail for a SELECTED past run
  const [includeEval, setIncludeEval] = useState(true)
  const [dryRun, setDryRun] = useState(false)
  const [busy, setBusy] = useState(false)
  const [runErr, setRunErr] = useState<string | null>(null)

  // "A run is actually executing" comes from the live orchestrator — NOT from
  // whichever history row is selected (a killed run persists a stale running flag).
  const globalRunning = !!nly?.running
  const liveRunId = nly?.run_id
  const viewingLive = globalRunning && !!liveRunId && (selId === liveRunId || !selId)
  // What the right pane shows: the live orchestrator state when viewing the active
  // run, else the fetched (past) run detail.
  const shownDetail = viewingLive ? nly : detail

  const fetchReport = async () => {
    try { const r = await fetch(`${API_BASE}/admin/integrity/report`); if (r.ok) setRep(await r.json()) } catch { /* ignore */ }
  }
  const fetchNly = async () => {
    try { const r = await fetch(`${API_BASE}/admin/nightly/status`); if (r.ok) setNly(await r.json()) } catch { /* ignore */ }
  }
  const fetchRuns = async () => {
    try { const r = await fetch(`${API_BASE}/admin/nightly/runs`); if (r.ok) { const j = await r.json(); setRuns(j.runs || []) } } catch { /* ignore */ }
  }
  const fetchDetail = async (id: string) => {
    try { const r = await fetch(`${API_BASE}/admin/nightly/runs/${id}`); if (r.ok) setDetail(await r.json()) } catch { /* ignore */ }
  }
  useEffect(() => { fetchReport(); fetchNly(); fetchRuns() }, [])
  // Selection: follow the live run when one is executing; otherwise keep the
  // current selection or default to the newest.
  useEffect(() => {
    if (globalRunning && liveRunId) setSelId(liveRunId)
    else setSelId(prev => prev || runs[0]?.id || null)
  }, [globalRunning, liveRunId, runs])
  // Fetch detail only for a SELECTED PAST run (the live run comes from nly).
  useEffect(() => { if (selId && !viewingLive) fetchDetail(selId) }, [selId, viewingLive])
  useEffect(() => {
    const iv = setInterval(() => {
      fetchNly(); fetchRuns()
      if (selId && !viewingLive) fetchDetail(selId)
      if (!globalRunning) fetchReport()
    }, globalRunning ? 4000 : 12000)
    return () => clearInterval(iv)
  }, [selId, globalRunning, liveRunId, viewingLive])

  const run = async () => {
    // No confirm() dialog: this app runs inside the platform iframe, which
    // blocks window.confirm/alert (no allow-modals) — a confirm() gate would
    // silently return false and the run would never fire. The dry-run / eval
    // toggles make intent explicit instead.
    setBusy(true); setRunErr(null)
    try {
      const r = await fetch(`${API_BASE}/admin/nightly/run`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ include_eval: includeEval, dry_run: dryRun }),
      })
      if (!r.ok) {
        const body = await r.text().catch(() => '')
        setRunErr(r.status === 401 || r.status === 403
          ? 'Not authenticated — open the RAG app from the platform (chat tool tile), not a bare URL.'
          : `Run failed: HTTP ${r.status}${body ? ` — ${body.slice(0, 160)}` : ''}`)
        return
      }
      const j = await r.json()
      if (j.status === 'already_running') { setRunErr('A run is already in progress.') }
      if (j.run_id) setSelId(j.run_id)
      await Promise.all([fetchNly(), fetchRuns()])   // flip the button to Stop + track the live run
    } catch (e) {
      setRunErr(`Run failed: ${String(e)}`)
    } finally { setBusy(false) }
  }
  const stop = async () => {
    setRunErr(null)
    try {
      const r = await fetch(`${API_BASE}/admin/nightly/stop`, { method: 'POST' })
      if (!r.ok) setRunErr(`Stop failed: HTTP ${r.status}`)
      await fetchNly()   // reflect the stop in the button immediately
    } catch (e) { setRunErr(`Stop failed: ${String(e)}`) }
  }
  const resume = async (fromId: string) => {
    setBusy(true); setRunErr(null)
    try {
      const r = await fetch(`${API_BASE}/admin/nightly/run`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ include_eval: includeEval, dry_run: dryRun, resume_from: fromId }),
      })
      if (!r.ok) {
        setRunErr(r.status === 401 || r.status === 403 ? 'Not authenticated — reload the page.' : `Resume failed: HTTP ${r.status}`)
        return
      }
      const j = await r.json()
      if (j.run_id) setSelId(j.run_id)
      await Promise.all([fetchNly(), fetchRuns()])
    } catch (e) { setRunErr(`Resume failed: ${String(e)}`) } finally { setBusy(false) }
  }

  const gaps = rep?.gaps || {}
  const openGaps = ['unpublished_total', 'failed_docs', 'stuck_docs', 'blocked_jobs', 'stale_tags'].reduce((a, k) => a + (gaps[k] || 0), 0)
  const lastLift = runs.find(r => typeof r.router_delta === 'number')?.router_delta
  const card = (label: string, value: any, sub?: string, color?: string) => (
    <div style={{ background: 'var(--mobius-surface-elevated, #f8fafc)', borderRadius: 8, padding: '10px 14px' }}>
      <div style={{ fontSize: 12, color: 'var(--mobius-text-secondary)' }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 600, color: color || 'inherit' }}>{value}
        {sub && <span style={{ fontSize: 12, color: 'var(--mobius-text-secondary)', fontWeight: 400 }}> {sub}</span>}</div>
    </div>
  )
  const fmtWhen = (iso?: string | null) => iso ? new Date(iso).toLocaleString([], { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '—'
  const steps: any[] = shownDetail?.steps || []
  const lift = shownDetail?.lift

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* header */}
      <div>
        <div style={{ fontSize: 15, fontWeight: 600 }}>Nightly pipeline</div>
        <div style={{ fontSize: 12, color: 'var(--mobius-text-secondary)' }}>Corpus + lexicon → RAG → chat, bracketed by eval</div>
      </div>

      {/* metric cards (current corpus) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0,1fr))', gap: 12 }}>
        {card('Published', (rep?.published ?? '—').toLocaleString?.() ?? rep?.published ?? '—', rep ? `/ ${rep.documents_total?.toLocaleString?.()}` : '')}
        {card('Lexicon rev', rep?.current_lexicon_revision ?? '—')}
        {card('Open gaps', rep ? openGaps : '—')}
        {card('Last lift Δrouter', typeof lastLift === 'number' ? `${lastLift >= 0 ? '+' : ''}${lastLift.toFixed(3)}` : '—', '',
          typeof lastLift === 'number' ? (lastLift >= 0 ? '#10b981' : '#ef4444') : undefined)}
      </div>

      {/* two columns: left = new run + history · right = selected run detail */}
      <div style={{ display: 'flex', gap: 14, alignItems: 'flex-start' }}>
        {/* LEFT RAIL */}
        <div style={{ width: 234, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div style={{ border: '1px solid var(--mobius-border, #e2e8f0)', borderRadius: 12, padding: '12px 14px' }}>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>New run</div>
            <label style={{ fontSize: 12, color: 'var(--mobius-text-secondary)', display: 'flex', gap: 6, alignItems: 'center', marginBottom: 4 }}>
              <input type="checkbox" checked={includeEval} disabled={globalRunning} onChange={e => setIncludeEval(e.target.checked)} /> eval bracket (~50 min)</label>
            <label style={{ fontSize: 12, color: 'var(--mobius-text-secondary)', display: 'flex', gap: 6, alignItems: 'center', marginBottom: 10 }}>
              <input type="checkbox" checked={dryRun} disabled={globalRunning} onChange={e => setDryRun(e.target.checked)} /> dry run</label>
            {globalRunning ? (
              <button type="button" onClick={stop}
                style={{ width: '100%', padding: '8px 0', borderRadius: 8, border: '1px solid #ef4444', color: '#ef4444', background: 'transparent', fontSize: 13, fontWeight: 600, cursor: 'pointer' }}>Stop run</button>
            ) : (
              <button type="button" onClick={run} disabled={busy}
                style={{ width: '100%', padding: '8px 0', borderRadius: 8, border: 'none', color: '#fff', background: '#2563eb', fontSize: 13, fontWeight: 600, cursor: busy ? 'wait' : 'pointer' }}>
                {busy ? 'Starting…' : '▶ Run pipeline'}</button>
            )}
            {runErr && (
              <div style={{ marginTop: 8, fontSize: 12, color: '#b91c1c', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 6, padding: '6px 8px' }}>{runErr}</div>
            )}
          </div>

          <div style={{ border: '1px solid var(--mobius-border, #e2e8f0)', borderRadius: 12, padding: '8px 8px', maxHeight: 380, overflow: 'auto' }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--mobius-text-secondary)', padding: '4px 6px' }}>History</div>
            {runs.length === 0 && <div style={{ fontSize: 12, color: 'var(--mobius-text-secondary)', padding: '6px' }}>no runs yet</div>}
            {runs.map(rn => {
              const sel = rn.id === selId
              const d = rn.router_delta
              return (
                <div key={rn.id} onClick={() => setSelId(rn.id)}
                  style={{ display: 'flex', gap: 8, alignItems: 'center', padding: '7px 6px', borderRadius: 8, cursor: 'pointer',
                    background: sel ? 'var(--mobius-surface-elevated, #eef2ff)' : 'transparent' }}>
                  <span style={{ width: 8, height: 8, borderRadius: 999, flexShrink: 0, background: _stepColor(rn.status === 'done' ? 'done' : rn.status === 'running' ? 'running' : rn.status === 'failed' ? 'failed' : 'skipped') }} />
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <div style={{ fontSize: 12, fontWeight: sel ? 600 : 400 }}>{fmtWhen(rn.started_at)}</div>
                    <div style={{ fontSize: 11, color: 'var(--mobius-text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {rn.status}{rn.dry_run ? ' · dry' : ''}{rn.include_eval ? ' · eval' : ''}
                      {typeof d === 'number' ? ` · Δ${d >= 0 ? '+' : ''}${d.toFixed(3)}` : ''}</div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* RIGHT DETAIL */}
        <div style={{ flex: 1, minWidth: 0 }}>
          {!shownDetail ? (
            <div style={{ fontSize: 13, color: 'var(--mobius-text-secondary)', padding: '24px 0', textAlign: 'center' }}>Select a run to see its stages and eval bracket.</div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div style={{ background: 'var(--mobius-surface, #fff)', border: '1px solid var(--mobius-border, #e2e8f0)', borderRadius: 12, padding: '12px 16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 13, fontWeight: 600, marginBottom: 10 }}>
                  <span>Run progress</span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 10, color: 'var(--mobius-text-secondary)', fontWeight: 400 }}>
                    <span>
                      {shownDetail.running ? `running · ${_fmtDur(shownDetail.started_at)}` : shownDetail.finished_at ? `${(shownDetail.error ? 'failed' : 'done')} · ${_fmtDur(shownDetail.started_at, shownDetail.finished_at)}` : 'idle'}
                      {shownDetail.error ? ` · ${shownDetail.error}` : ''}
                    </span>
                    {!globalRunning && shownDetail.error && selId && (
                      <button type="button" onClick={() => resume(selId)} disabled={busy}
                        style={{ border: '1px solid #2563eb', color: '#2563eb', background: 'transparent', borderRadius: 6, fontSize: 12, fontWeight: 600, padding: '3px 10px', cursor: busy ? 'wait' : 'pointer' }}>
                        {busy ? 'Resuming…' : '↻ Resume from failure'}</button>
                    )}
                  </span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  {[...steps].sort((a, b) => _STEP_ORDER.indexOf(a.key) - _STEP_ORDER.indexOf(b.key)).map(s => {
                    const log: string[] = s.log || []
                    const hasDetail = log.length > 0 || s.started_at
                    const summaryRow = (
                      <span style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 13, flex: 1 }}>
                        <span style={{ width: 9, height: 9, borderRadius: 999, background: _stepColor(s.status), flexShrink: 0 }} />
                        <span style={{ width: 150, fontWeight: s.status === 'running' ? 600 : 400 }}>{s.label}</span>
                        <span style={{ flex: 1, color: 'var(--mobius-text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s.detail || (s.status === 'pending' ? '—' : s.status)}</span>
                        <span style={{ color: 'var(--mobius-text-secondary)', fontSize: 12 }}>{s.status === 'running' || s.status === 'done' || s.status === 'failed' ? _fmtDur(s.started_at, s.ended_at) : ''}</span>
                      </span>
                    )
                    const body = (
                      <div style={{ margin: '2px 0 8px 19px', padding: '8px 10px', background: 'var(--mobius-surface-elevated, #f8fafc)', borderRadius: 6, fontSize: 12 }}>
                        <div style={{ color: 'var(--mobius-text-secondary)', marginBottom: 6 }}>
                          status <b style={{ color: _stepColor(s.status) }}>{s.status}</b>
                          {s.started_at && <> · started {new Date(s.started_at).toLocaleTimeString()}</>}
                          {s.ended_at && <> · ended {new Date(s.ended_at).toLocaleTimeString()}</>}
                          {(s.key === 'baseline_eval' || s.key === 'final_eval') && shownDetail?.eval_run_id && s.status === 'running' && <> · run {String(shownDetail.eval_run_id).slice(0, 8)}</>}
                        </div>
                        {s.key === 'gate' && shownDetail?.gate && (
                          <div style={{ fontFamily: 'monospace', marginBottom: 6 }}>
                            published {shownDetail.gate.published}/{shownDetail.gate.documents_total} · frac {shownDetail.gate.frac} · stale_tags {shownDetail.gate.stale_tags} → {shownDetail.gate.passed ? 'PASS' : 'FAIL'}
                          </div>
                        )}
                        {log.length > 0 ? (
                          <div style={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap', lineHeight: 1.5, maxHeight: 220, overflow: 'auto', color: 'var(--mobius-text-secondary)' }}>{log.join('\n')}</div>
                        ) : <div style={{ color: 'var(--mobius-text-secondary)' }}>no log yet</div>}
                      </div>
                    )
                    return hasDetail ? (
                      <details key={s.key} style={{ padding: '3px 0' }}>
                        <summary style={{ cursor: 'pointer', listStyle: 'none', display: 'flex', alignItems: 'center' }}>{summaryRow}</summary>
                        {body}
                      </details>
                    ) : (
                      <div key={s.key} style={{ padding: '5px 0', display: 'flex' }}>{summaryRow}</div>
                    )
                  })}
                </div>
              </div>

              {lift && (
                <div style={{ background: 'var(--mobius-surface, #fff)', border: '1px solid var(--mobius-border, #e2e8f0)', borderRadius: 12, padding: '12px 16px' }}>
                  <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Eval bracket — baseline → final</div>
                  <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
                    <thead><tr style={{ color: 'var(--mobius-text-secondary)', fontSize: 12 }}>
                      <td>metric</td><td style={{ textAlign: 'right' }}>baseline</td><td style={{ textAlign: 'right' }}>final</td><td style={{ textAlign: 'right' }}>Δ</td></tr></thead>
                    <tbody>
                      {['router_recall', 'oracle_recall', 'best_single_recall', 'routing_headroom'].map(k => {
                        const row = lift[k] || {}; const d = row.delta
                        return (
                          <tr key={k}>
                            <td style={{ padding: '4px 0' }}>{k}</td>
                            <td style={{ textAlign: 'right' }}>{row.baseline ?? '—'}</td>
                            <td style={{ textAlign: 'right' }}>{row.final ?? '—'}</td>
                            <td style={{ textAlign: 'right', color: typeof d === 'number' ? (d >= 0 ? '#10b981' : '#ef4444') : 'inherit' }}>
                              {typeof d === 'number' ? `${d >= 0 ? '+' : ''}${d.toFixed(3)}` : '—'}</td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* legacy integrity/health — demoted to a collapsed section */}
      <details style={{ fontSize: 13 }}>
        <summary style={{ cursor: 'pointer', color: 'var(--mobius-text-secondary)', padding: '4px 0' }}>Integrity + pipeline health (details)</summary>
        <div style={{ marginTop: 8 }}>
          <StatusPanel documents={documents} onRefresh={onRefresh} />
        </div>
      </details>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export function RepositoryTab({
  documents,
  documentsLoading: _documentsLoading = false,
  selectedDocumentId,
  navigateToRead,
  onNavigateToReadConsumed,
  onDocumentSelect,
  onRefresh,
}: Props) {
  // ── Entity / host data ───────────────────────────────────────────────────
  const [rawHosts, setRawHosts] = useState<RawHostStat[]>([])
  const [statsLoading, setStatsLoading] = useState(true)
  const [selectedHost, setSelectedHost] = useState(UPLOADED_HOST)

  // ── Sidebar UI state ─────────────────────────────────────────────────────
  const [entitySearch, setEntitySearch] = useState('')
  const [domainFilter, setDomainFilter] = useState<DomainFilter>('all')

  // ── Search state ─────────────────────────────────────────────────────────
  const [searchQuery, setSearchQuery] = useState('')
  const [searchMode, setSearchMode] = useState<SearchMode>('corpus')
  // Corpus search — live in the Library "Search corpus" mode.
  const [searchResults, setSearchResults] = useState<CorpusChunk[] | null>(null)
  const [searchTelemetry, setSearchTelemetry] = useState<SearchTelemetry | null>(null)
  const [searchLoading, setSearchLoading] = useState(false)

  // ── Reader state ─────────────────────────────────────────────────────────
  const [urlPreview, setUrlPreview] = useState<{ url: string; ingested: boolean } | null>(null)
  const [ingestingUrl, setIngestingUrl] = useState(false)
  // Local navigate-to (from clicking a search result chunk)
  const [localNavigateTo, setLocalNavigateTo] = useState<NavigateToRead | null>(null)

  // ── New layout state ─────────────────────────────────────────────────────
  const [pageTab, setPageTab] = useState<'library' | 'pipeline'>('library')
  const [libMode, setLibMode] = useState<'browse' | 'search'>('browse')
  const [viewingDoc, setViewingDoc] = useState(false)  // reader open (over search results in search mode)

  // Auto-open the reader when an external navigateToRead arrives
  useEffect(() => {
    if (navigateToRead) { setPageTab('library'); setViewingDoc(true) }
  }, [navigateToRead])

  // ── Browse panel state ───────────────────────────────────────────────────
  const [browseSearch, setBrowseSearch] = useState('')
  const [_payerFilter, setPayerFilter] = useState('')
  const [stateFilter, setStateFilter] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  // Track which payer groups are collapsed
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>({ __other__: true })

  // ── Payor-filtered doc loading ────────────────────────────────────────────
  // Canonical managed care payors — always shown in nav even if 0 docs
  const MANAGED_CARE_PAYORS = [
    'Sunshine Health',
    'Simply Healthcare',
    'United Healthcare',
    'Aetna',
    'Molina Healthcare of Florida',
  ]
  const [payerDocs, setPayerDocs] = useState<DocLike[]>([])
  const [payerDocsLoading, setPayerDocsLoading] = useState(false)
  const [activePayer, setActivePayer] = useState<string | null>(null)

  const selectPayer = (payer: string | null) => {
    setActivePayer(payer)
    setPayerFilter(payer ?? '')
    if (payer === null) { setPayerDocs([]); return }
    setPayerDocsLoading(true)
    const param = payer === '' ? '&payer=' : `&payer=${encodeURIComponent(payer)}`
    fetch(`${API_BASE}/documents?limit=500${param}`)
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(d => setPayerDocs((d.documents || []) as DocLike[]))
      .catch(() => setPayerDocs([]))
      .finally(() => setPayerDocsLoading(false))
  }

  // ── Load aggregate stats ─────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false
    setStatsLoading(true)
    Promise.all([
      fetch(`${API_BASE}/sources/stats`)
        .then((r) => (r.ok ? (r.json() as Promise<StatsResponse>) : Promise.reject(r.status)))
        .catch((e): StatsResponse => {
          console.warn('[Repository] /sources/stats failed:', e)
          return { by_host: {}, by_curation_status: {}, by_content_kind: {}, by_ingested: {} }
        }),
      fetch(`${API_BASE}/sources/corpus_by_host`)
        .then((r) => (r.ok ? (r.json() as Promise<CorpusByHost>) : Promise.reject(r.status)))
        .catch((e): CorpusByHost => {
          console.warn('[Repository] /sources/corpus_by_host failed:', e)
          return { by_host: {} }
        }),
    ])
      .then(([s, c]) => {
        if (cancelled) return
        const corpusByHost = c.by_host || {}
        const allHosts = new Set<string>([
          ...Object.keys(s.by_host || {}),
          ...Object.keys(corpusByHost),
        ])
        const hostList: RawHostStat[] = Array.from(allHosts)
          .filter((h) => !h.includes('test.example.com') && h !== '(no_host)')
          .map((host) => ({
            host,
            count: s.by_host?.[host] ?? 0,
            corpusDocs: corpusByHost[host]?.docs ?? 0,
            corpusPublished: corpusByHost[host]?.published ?? 0,
          }))
          .sort((a, b) => b.count + b.corpusDocs - (a.count + a.corpusDocs))
        setRawHosts(hostList)
        setStatsLoading(false)
      })
      .catch((e) => {
        if (cancelled) return
        console.warn('[Repository] stats fetch failed:', e)
        setStatsLoading(false)
      })
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── Uploaded docs: no web source URL ────────────────────────────────────
  const uploadedDocs = useMemo(() => {
    return documents.filter((d) => {
      const url = d.source_metadata?.source_url || d.source_url || ''
      return !url
    })
  }, [documents])

  // ── Enrich hosts with domain classification + virtual "(uploaded)" host ──
  const enrichedHosts: HostStatEnriched[] = useMemo(() => {
    const webHosts = rawHosts.map((h) => {
      const payerMatch = documents.find((d) => {
        const url = d.source_metadata?.source_url || d.source_url || ''
        try { return new URL(url).hostname === h.host }
        catch { return false }
      })
      const payer = payerMatch?.payer ?? null
      return { ...h, payer, domain: domainOf(h.host, payer) }
    })
    // Always inject an "(uploaded)" virtual entry at the top
    const uploadedPublished = uploadedDocs.filter((d) => !!d.published_at).length
    const uploadedEntry: HostStatEnriched = {
      host: UPLOADED_HOST,
      count: 0,
      corpusDocs: uploadedDocs.length,
      corpusPublished: uploadedPublished,
      payer: null,
      domain: 'other' as const,
    }
    return [uploadedEntry, ...webHosts]
  }, [rawHosts, documents, uploadedDocs])

  // ── Corpus stats for sidebar mini-strip ──────────────────────────────────
  const corpusStats: CorpusStats = useMemo(() => {
    let published = 0, waiting = 0, failed = 0
    for (const d of documents) {
      if (d.published_at) published++
      else if (d.embedding_status === 'failed' || d.chunking_status === 'failed') failed++
      else waiting++
    }
    return { published, waiting, failed }
  }, [documents])

  // ── Payer label for selected host ────────────────────────────────────────
  const payerLabel = useMemo(
    () => enrichedHosts.find((h) => h.host === selectedHost)?.payer ?? null,
    [enrichedHosts, selectedHost],
  )

  // ── Corpus search — debounced 400ms, respects mode ───────────────────────
  useEffect(() => {
    if (!searchQuery.trim()) {
      setSearchResults(null)
      setSearchTelemetry(null)
      return
    }
    const timer = setTimeout(async () => {
      setSearchLoading(true)
      try {
        const resp = await fetch(`${API_BASE}/api/skills/v1/corpus_search`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: searchQuery.trim(),
            k: 20,
            mode: searchMode,
          }),
        })
        if (resp.ok) {
          const data: CorpusSearchResponse = await resp.json()
          setSearchResults(data.chunks)
          setSearchTelemetry(data.telemetry)
        } else {
          setSearchResults([])
          setSearchTelemetry(null)
        }
      } catch {
        setSearchResults([])
        setSearchTelemetry(null)
      } finally {
        setSearchLoading(false)
      }
    }, 400)
    return () => clearTimeout(timer)
  }, [searchQuery, searchMode])

  // ── Handlers ─────────────────────────────────────────────────────────────
  const handleSelectHost = (host: string) => {
    setSelectedHost(host)
    setSearchQuery('')
    setSearchResults(null)
  }

  // @ts-ignore — kept for symmetry; search UI moved to Test tab.
  const _handleSearchModeChange = (mode: SearchMode) => {
    setSearchMode(mode)
    // Re-run immediately if there's already a query
    if (searchQuery.trim()) {
      setSearchResults(null)
    }
  }

  const openDocument = (documentId: string, pageNumber?: number, citeText?: string) => {
    setUrlPreview(null)
    setLocalNavigateTo(pageNumber != null ? { documentId, pageNumber, citeText } : null)
    onDocumentSelect(documentId)
    setPageTab('library')
    setViewingDoc(true)   // show the reader (over search results if in search mode)
  }

  // @ts-ignore — search UI moved to Test tab.
  const _openChunk = (chunk: CorpusChunk) => {
    openDocument(
      chunk.document_id,
      chunk.page_number ?? undefined,
      chunk.text.slice(0, 300),
    )
  }

  const openUrlPreview = (url: string, ingested: boolean) => {
    setUrlPreview({ url, ingested })
    setLocalNavigateTo(null)
    setPageTab('library')
    setViewingDoc(true)
  }

  const closeReader = () => {
    setUrlPreview(null)
    setLocalNavigateTo(null)
    setViewingDoc(false)   // back to the doc list / search results
  }

  const handleIngestUrl = async (url: string) => {
    setIngestingUrl(true)
    try {
      const resp = await fetch(`${API_BASE}/documents/import-from-html`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, auto_publish: true, auto_publish_timeout_s: 180 }),
      })
      if (resp.ok || resp.status === 409) {
        setUrlPreview({ url, ingested: true })
      }
    } finally {
      setIngestingUrl(false)
    }
  }

  // Merge local navigation with external navigation (external wins if more recent)
  const activeNavigateTo = navigateToRead ?? localNavigateTo

  // ── Browse: filter + group logic ─────────────────────────────────────────
  // When a managed care payor is active, use the freshly loaded payerDocs; otherwise use the sample
  const browseSource = activePayer !== null ? payerDocs : documents

  const filteredDocs = useMemo(() => {
    const q = browseSearch.toLowerCase()
    return browseSource.filter((d) => {
      if (q) {
        const haystack = [d.display_name, d.filename, d.payer, d.state]
          .filter(Boolean).join(' ').toLowerCase()
        if (!haystack.includes(q)) return false
      }
      if (stateFilter && (d.state ?? '') !== stateFilter) return false
      if (statusFilter) {
        const docStatus = d.published_at
          ? 'published'
          : d.chunking_status === 'failed' || d.embedding_status === 'failed'
            ? 'failed'
            : 'processing'
        if (docStatus !== statusFilter) return false
      }
      return true
    })
  }, [browseSource, browseSearch, stateFilter, statusFilter])

  // Group non-managed-care docs by payer for the "Other" section
  const groupedDocs = useMemo(() => {
    if (activePayer !== null) return {}   // payor mode: flat list, no groups needed
    const groups: Record<string, DocLike[]> = {}
    for (const d of filteredDocs) {
      const key = d.payer || '(untagged)'
      if (MANAGED_CARE_PAYORS.includes(key)) continue   // shown in pinned section
      if (!groups[key]) groups[key] = []
      groups[key].push(d)
    }
    return groups
  }, [filteredDocs, activePayer])

  // Counts for managed care payors from the full document sample
  const managedCareCounts = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const p of MANAGED_CARE_PAYORS) counts[p] = 0
    for (const d of documents) {
      if (d.payer && MANAGED_CARE_PAYORS.includes(d.payer)) counts[d.payer]++
    }
    return counts
  }, [documents])

  // Unique filter options
  const uniqueStates = useMemo(() =>
    Array.from(new Set(documents.map((d) => d.state).filter(Boolean) as string[])).sort(),
    [documents])

  // ── Payor sitemap (sources for selected payor) ───────────────────────────
  const [payorSources, setPayorSources] = useState<any[]>([])
  const [sourcesLoading, setSourcesLoading] = useState(false)

  useEffect(() => {
    if (!activePayer) { setPayorSources([]); return }
    setSourcesLoading(true)
    fetch(`${API_BASE}/sources/search?q=${encodeURIComponent(activePayer)}&limit=200`)
      .then(r => r.ok ? r.json() : [])
      .then(d => setPayorSources(Array.isArray(d) ? d : []))
      .catch(() => setPayorSources([]))
      .finally(() => setSourcesLoading(false))
  }, [activePayer])

  // ── Doc view (read / tags / info) ───────────────────────────────────────
  const [docView, setDocView] = useState<'reader' | 'analytics' | 'metadata'>('reader')
  const [analyticsData, setAnalyticsData] = useState<any>(null)
  const [analyticsLoading, setAnalyticsLoading] = useState(false)

  // Reset to reader whenever a new doc is opened
  useEffect(() => {
    if (selectedDocumentId) {
      setDocView('reader')
      setAnalyticsData(null)
    }
  }, [selectedDocumentId])

  const loadAnalytics = (docId: string) => {
    if (analyticsData?.document_id === docId) return   // already loaded
    setAnalyticsLoading(true)
    fetch(`${API_BASE}/documents/${docId}/policy-line-tags`)
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(d => setAnalyticsData(d))
      .catch(() => setAnalyticsData(null))
      .finally(() => setAnalyticsLoading(false))
  }

  // Aggregate policy-line-tags into { tag, count, pages[] } per tag type
  const aggregateTags = (lines: any[], key: 'j_tags' | 'p_tags' | 'd_tags') => {
    const map: Record<string, { count: number; pages: Set<number>; topScore: number }> = {}
    for (const line of lines || []) {
      const tags = line[key]
      if (!tags) continue
      for (const [tag, score] of Object.entries(tags) as [string, number][]) {
        if (!map[tag]) map[tag] = { count: 0, pages: new Set(), topScore: 0 }
        map[tag].count++
        map[tag].pages.add(line.page_number)
        if (score > map[tag].topScore) map[tag].topScore = score
      }
    }
    return Object.entries(map)
      .map(([tag, { count, pages, topScore }]) => ({ tag, count, pages: [...pages].sort((a, b) => a - b), topScore }))
      .sort((a, b) => b.count - a.count)
  }

  const jumpToPage = (pageNumber: number) => {
    if (!selectedDocumentId) return
    setDocView('reader')
    setViewingDoc(true)
    setLocalNavigateTo({ documentId: selectedDocumentId, pageNumber, citeText: undefined })
  }

  // Status dot class
  const statusDotClass = (d: DocLike) => {
    if (d.published_at) return 'published'
    if (d.chunking_status === 'failed' || d.embedding_status === 'failed') return 'failed'
    if (d.chunking_status === 'completed' || d.embedding_status === 'completed') return 'processing'
    return 'pending'
  }

  // Pipeline progress pips: extract → chunk → embed → publish
  const PipelinePips = ({ d }: { d: DocLike }) => {
    const steps = [
      { key: 'E', done: d.extraction_status === 'completed', failed: d.extraction_status === 'failed', label: 'Extract: ' + (d.extraction_status ?? '—') },
      { key: 'C', done: d.chunking_status === 'completed', failed: d.chunking_status === 'failed', label: 'Chunk: ' + (d.chunking_status ?? '—') },
      { key: 'M', done: d.embedding_status === 'completed', failed: d.embedding_status === 'failed', label: 'Embed: ' + (d.embedding_status ?? '—') },
      { key: 'P', done: !!d.published_at, failed: false, label: d.published_at ? `Published (${(d as any).published_rows ?? '?'} rows)` : 'Not published' },
    ]
    const allDone = steps.every(s => s.done)
    return (
      <span className="pipeline-pips" title={steps.map(s => s.label).join('\n')}>
        {steps.map(s => (
          <span key={s.key} className={`pip ${s.failed ? 'pip-fail' : s.done ? 'pip-done' : 'pip-todo'}`}>{s.key}</span>
        ))}
        {!allDone && <span className="pip-flag">!</span>}
      </span>
    )
  }

  // Suppress unused vars from original code that are now no longer used in render
  void statsLoading
  void selectedHost
  void handleSelectHost
  void entitySearch
  void setEntitySearch
  void domainFilter
  void setDomainFilter
  void corpusStats
  void payerLabel
  void uploadedDocs
  void enrichedHosts
  void openUrlPreview

  // ── Render ────────────────────────────────────────────────────────────────
  const CorpusResults = _CorpusSearchResults
  const searchInput = libMode === 'search' ? searchQuery : browseSearch
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 56px)', overflow: 'hidden', background: 'var(--mobius-bg-primary)' }}>
      {/* ── Top-level tabs: Library | Pipeline ── */}
      <div className="repo-page-tabs" style={{ display: 'flex', gap: 4, alignItems: 'center', flexShrink: 0, borderBottom: '1px solid var(--mobius-border, #e2e8f0)', padding: '0 8px' }}>
        {(['library', 'pipeline'] as const).map((t) => (
          <button
            key={t}
            onClick={() => setPageTab(t)}
            style={{
              background: 'none', border: 'none', cursor: 'pointer', padding: '10px 16px',
              fontSize: 14, fontWeight: pageTab === t ? 600 : 400,
              color: pageTab === t ? 'var(--mobius-text-primary, #0f172a)' : 'var(--mobius-text-secondary)',
              borderBottom: pageTab === t ? '2px solid #2563eb' : '2px solid transparent',
            }}
          >{t === 'library' ? 'Library' : 'Pipeline'}</button>
        ))}
        <div style={{ flex: 1 }} />
        {onRefresh && pageTab === 'library' && (
          <button onClick={onRefresh} title="Refresh documents"
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--mobius-text-secondary)', fontSize: 16, padding: '6px 10px' }}>↺</button>
        )}
      </div>

      {pageTab === 'pipeline' ? (
        <div className="repo-pipeline-body" style={{ flex: 1, minHeight: 0, overflow: 'auto', padding: '14px 16px' }}>
          <PipelinePanel documents={documents} onRefresh={onRefresh} />
        </div>
      ) : (
        <div className="repo-body" style={{ display: 'flex', flex: 1, minHeight: 0, overflow: 'hidden' }}>
          {/* ── LEFT: browse / search controls ── */}
          <div className="repo-nav" style={{ width: 258, flexShrink: 0, borderRight: '1px solid var(--mobius-border, #e2e8f0)' }}>
            <div className="repo-nav-content">
              {/* mode toggle */}
              <div style={{ display: 'flex', border: '1px solid var(--mobius-border, #e2e8f0)', borderRadius: 8, overflow: 'hidden', margin: '10px 10px 8px' }}>
                {(['browse', 'search'] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => { setLibMode(m); if (m === 'search') setViewingDoc(false) }}
                    style={{
                      flex: 1, padding: '7px 0', fontSize: 12.5, cursor: 'pointer', border: 'none',
                      background: libMode === m ? '#2563eb' : 'transparent',
                      color: libMode === m ? '#fff' : 'var(--mobius-text-secondary)',
                      fontWeight: libMode === m ? 600 : 400,
                    }}
                  >{m === 'browse' ? 'Browse' : 'Search corpus'}</button>
                ))}
              </div>

              {/* search + filters */}
              <div className="repo-nav-search">
                <input
                  type="search"
                  placeholder={libMode === 'search' ? 'Search corpus content…' : 'Search documents, payer…'}
                  value={searchInput}
                  onChange={(e) => libMode === 'search' ? setSearchQuery(e.target.value) : setBrowseSearch(e.target.value)}
                />
                {libMode === 'search' && (
                  <div className="repo-search-mode-btns" style={{ display: 'flex', gap: 5, margin: '8px 0' }}>
                    {(['corpus', 'precision', 'recall'] as const).map((m) => (
                      <button key={m} className={`repo-search-mode-btn${searchMode === m ? ' active' : ''}`} onClick={() => setSearchMode(m)}>
                        {m === 'corpus' ? 'Hybrid' : m === 'precision' ? 'BM25' : 'Semantic'}
                      </button>
                    ))}
                  </div>
                )}
                <div className="repo-nav-filters">
                  <select value={stateFilter} onChange={(e) => setStateFilter(e.target.value)}>
                    <option value="">State ▾</option>
                    {uniqueStates.map((s) => <option key={s} value={s}>{s}</option>)}
                  </select>
                  <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
                    <option value="">Status ▾</option>
                    <option value="published">published</option>
                    <option value="processing">processing</option>
                    <option value="failed">failed</option>
                  </select>
                </div>
              </div>

              {libMode === 'browse' ? (
                <>
                  {/* ── Managed care payor pins ── */}
                  <div style={{ padding: '6px 10px 2px', fontSize: 11, fontWeight: 600, color: 'var(--mobius-text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Managed Care
                  </div>
                  <ul className="repo-nav-list" style={{ marginBottom: 0 }}>
                    {/* "All docs" row */}
                    <li>
                      <div
                        className={`repo-nav-doc repo-nav-payor-row${activePayer === null ? ' active' : ''}`}
                        onClick={() => selectPayer(null)}
                      >
                        <span className="repo-nav-doc-name" style={{ fontWeight: activePayer === null ? 600 : 400 }}>All documents</span>
                        <span className="repo-nav-doc-age">{documents.length.toLocaleString()}</span>
                      </div>
                    </li>
                    {MANAGED_CARE_PAYORS.map((p) => (
                      <li key={p}>
                        <div
                          className={`repo-nav-doc repo-nav-payor-row${activePayer === p ? ' active' : ''}`}
                          onClick={() => selectPayer(activePayer === p ? null : p)}
                        >
                          <span className="repo-nav-doc-name" style={{ fontWeight: activePayer === p ? 600 : 400 }}>{p}</span>
                          <span className="repo-nav-doc-age">
                            {activePayer === p && payerDocsLoading ? '…' : (managedCareCounts[p] || 0)}
                          </span>
                        </div>
                      </li>
                    ))}
                  </ul>

                  {/* ── Doc list for active payor ── */}
                  {activePayer !== null && (
                    <>
                      <div style={{ padding: '8px 10px 2px', fontSize: 11, fontWeight: 600, color: 'var(--mobius-text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        {activePayer} — {filteredDocs.length} doc{filteredDocs.length !== 1 ? 's' : ''}
                      </div>
                      <ul className="repo-nav-list">
                        {payerDocsLoading ? (
                          <li><div className="repo-nav-empty">Loading…</div></li>
                        ) : filteredDocs.length === 0 ? (
                          <li><div className="repo-nav-empty">No documents yet for this payor.</div></li>
                        ) : filteredDocs.map((d) => (
                          <li key={d.id}>
                            <div
                              className={`repo-nav-doc${d.id === selectedDocumentId ? ' active' : ''}`}
                              onClick={() => openDocument(d.id)}
                              title={d.display_name || d.filename}
                            >
                              <span className={`repo-nav-dot ${statusDotClass(d)}`} />
                              <span className="repo-nav-doc-name">{d.display_name || d.filename}</span>
                              <PipelinePips d={d} />
                            </div>
                          </li>
                        ))}
                      </ul>
                    </>
                  )}

                  {/* ── Other sources (collapsed by default) ── */}
                  {activePayer === null && (
                    <ul className="repo-nav-list">
                      <li>
                        <div
                          className="repo-nav-group-header"
                          onClick={() => setCollapsedGroups(prev => ({ ...prev, __other__: !prev.__other__ }))}
                          style={{ marginTop: 8 }}
                        >
                          <span className="repo-nav-group-arrow">{collapsedGroups.__other__ ? '▶' : '▼'}</span>
                          <span className="repo-nav-group-name">Other sources</span>
                          <span className="repo-nav-group-count">{Object.values(groupedDocs).reduce((n, g) => n + g.length, 0)}</span>
                        </div>
                        {!collapsedGroups.__other__ && Object.entries(groupedDocs).sort(([a], [b]) => a.localeCompare(b)).map(([group, docs]) => {
                          const isCollapsed = !!collapsedGroups[group]
                          return (
                            <Fragment key={group}>
                              <div
                                className="repo-nav-group-header"
                                style={{ paddingLeft: 20, fontSize: 12 }}
                                onClick={() => setCollapsedGroups(prev => ({ ...prev, [group]: !prev[group] }))}
                              >
                                <span className="repo-nav-group-arrow">{isCollapsed ? '▶' : '▼'}</span>
                                <span className="repo-nav-group-name">{group}</span>
                                <span className="repo-nav-group-count">{docs.length}</span>
                              </div>
                              {!isCollapsed && docs.map((d) => (
                                <div
                                  key={d.id}
                                  className={`repo-nav-doc${d.id === selectedDocumentId ? ' active' : ''}`}
                                  style={{ paddingLeft: 28 }}
                                  onClick={() => openDocument(d.id)}
                                  title={d.display_name || d.filename}
                                >
                                  <span className={`repo-nav-dot ${statusDotClass(d)}`} />
                                  <span className="repo-nav-doc-name">{d.display_name || d.filename}</span>
                                  <PipelinePips d={d} />
                                </div>
                              ))}
                            </Fragment>
                          )
                        })}
                      </li>
                    </ul>
                  )}
                </>
              ) : (
                <div className="repo-nav-count">
                  {searchQuery.trim()
                    ? (searchLoading ? 'Searching…' : `${searchResults?.length ?? 0} chunk result${(searchResults?.length ?? 0) !== 1 ? 's' : ''}`)
                    : 'Type to search corpus content.'}
                </div>
              )}
            </div>
          </div>

          {/* ── RIGHT: search results OR reader ── */}
          <div className="repo-main-area" style={{ flex: 1, minWidth: 0 }}>
            <div className="repo-right-content">
              {libMode === 'search' && !viewingDoc ? (
                <div className="repo-tab-pane repo-search-pane-wrap" style={{ overflowY: 'auto', padding: '4px 12px 12px' }}>
                  {!searchQuery.trim() ? (
                    <div className="repo-reader-empty">
                      <p>Search the corpus by content — Hybrid, BM25, or Semantic. Click a hit to open the document at that page.</p>
                    </div>
                  ) : (
                    <CorpusResults
                      query={searchQuery}
                      mode={searchMode}
                      chunks={searchResults}
                      loading={searchLoading}
                      telemetry={searchTelemetry}
                      onOpenChunk={(c) => openDocument(c.document_id, c.page_number ?? undefined, c.text.slice(0, 120))}
                    />
                  )}
                </div>
              ) : (activePayer !== null && !viewingDoc && !urlPreview && !selectedDocumentId) ? (
                // payor active, no doc open → sitemap
                <div className="repo-tab-pane repo-sitemap-pane">
                  <div className="repo-sitemap-header">
                    <span className="repo-sitemap-title">Source sitemap — {activePayer}</span>
                    <span className="repo-sitemap-count">
                      {sourcesLoading ? 'Loading…' : `${payorSources.length} source${payorSources.length !== 1 ? 's' : ''}`}
                    </span>
                  </div>
                  {sourcesLoading ? (
                    <div className="repo-sitemap-loading">Loading sources…</div>
                  ) : payorSources.length === 0 ? (
                    <div className="repo-sitemap-empty">No discovered sources for {activePayer}.</div>
                  ) : (
                    <div className="repo-sitemap-table-wrap">
                      <table className="repo-sitemap-table">
                        <thead>
                          <tr>
                            <th>Source</th>
                            <th>Type</th>
                            <th>Authority</th>
                            <th>Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {payorSources.map((s, i) => {
                            const name = s.filename || s.url || s.source_url || '—'
                            const short = name.length > 60 ? '…' + name.slice(-57) : name
                            const ingested = s.ingested_doc_id != null
                            return (
                              <tr key={i} className={ingested ? 'sitemap-row-ingested' : 'sitemap-row-missing'}
                                onClick={() => {
                                  if (ingested && s.ingested_doc_id) openDocument(s.ingested_doc_id)
                                }}
                                style={{ cursor: ingested ? 'pointer' : 'default' }}
                              >
                                <td title={name}><span className="repo-sitemap-source-name">{short}</span></td>
                                <td>{s.source_type || '—'}</td>
                                <td>{s.authority_level || '—'}</td>
                                <td>
                                  <span className={`repo-sitemap-badge ${ingested ? 'badge-ingested' : 'badge-missing'}`}>
                                    {ingested ? 'ingested' : 'not ingested'}
                                  </span>
                                </td>
                              </tr>
                            )
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              ) : (selectedDocumentId || urlPreview) ? (
                // doc open → 3-view panel (Read / Tags / Info)
                <div className="repo-tab-pane repo-doc-panel">
                  {/* top bar: back + view tabs */}
                  <div className="repo-doc-topbar">
                    {libMode === 'search' ? (
                      <button onClick={() => setViewingDoc(false)} className="repo-pane-back-btn">‹ Results</button>
                    ) : activePayer !== null ? (
                      <button onClick={closeReader} className="repo-pane-back-btn">‹ {activePayer}</button>
                    ) : <span />}
                    {selectedDocumentId && (
                      <div className="repo-doc-tabs">
                        <button
                          className={`repo-doc-tab${docView === 'reader' ? ' active' : ''}`}
                          onClick={() => { setDocView('reader'); setViewingDoc(true) }}
                        >Read</button>
                        <button
                          className={`repo-doc-tab${docView === 'analytics' ? ' active' : ''}`}
                          onClick={() => { setDocView('analytics'); loadAnalytics(selectedDocumentId) }}
                        >Tags</button>
                        <button
                          className={`repo-doc-tab${docView === 'metadata' ? ' active' : ''}`}
                          onClick={() => setDocView('metadata')}
                        >Info</button>
                      </div>
                    )}
                  </div>

                  {/* Read view */}
                  {docView === 'reader' && (
                    <div className="repo-doc-view repo-doc-view--reader">
                      <ReaderSlideOut
                        documents={documents}
                        selectedDocumentId={selectedDocumentId}
                        navigateToRead={activeNavigateTo}
                        onNavigateToReadConsumed={() => { onNavigateToReadConsumed(); setLocalNavigateTo(null) }}
                        onDocumentSelect={onDocumentSelect}
                        urlPreview={urlPreview}
                        onIngestUrl={handleIngestUrl}
                        ingestingUrl={ingestingUrl}
                        onClose={closeReader}
                      />
                    </div>
                  )}

                  {/* Tags analytics view */}
                  {docView === 'analytics' && (
                    <div className="repo-doc-view repo-doc-view--analytics">
                      {analyticsLoading ? (
                        <div className="repo-analytics-empty">Loading tag data…</div>
                      ) : !analyticsData ? (
                        <div className="repo-analytics-empty">No tag data available for this document.</div>
                      ) : (() => {
                        const lines = analyticsData.lines || []
                        const jTags = aggregateTags(lines, 'j_tags')
                        const pTags = aggregateTags(lines, 'p_tags')
                        const dTags = aggregateTags(lines, 'd_tags')
                        const total = analyticsData.total || 0
                        return (
                          <>
                            <div className="repo-analytics-summary">
                              <span className="repo-analytics-stat">{total} tagged paragraphs</span>
                              <span className="repo-analytics-sep">·</span>
                              <span className="repo-analytics-stat">{jTags.length} jurisdictions</span>
                              <span className="repo-analytics-sep">·</span>
                              <span className="repo-analytics-stat">{pTags.length} policy tags</span>
                              <span className="repo-analytics-sep">·</span>
                              <span className="repo-analytics-stat">{dTags.length} domain tags</span>
                            </div>

                            {[
                              { label: 'Jurisdiction (j:)', tags: jTags, cls: 'tag-j' },
                              { label: 'Policy (p:)', tags: pTags, cls: 'tag-p' },
                              { label: 'Domain (d:)', tags: dTags, cls: 'tag-d' },
                            ].map(({ label, tags, cls }) => tags.length > 0 && (
                              <div key={label} className="repo-analytics-section">
                                <div className="repo-analytics-section-title">{label}</div>
                                <div className="repo-analytics-tag-list">
                                  {tags.map(({ tag, count, pages }) => (
                                    <div key={tag} className="repo-analytics-tag-row">
                                      <span className={`repo-analytics-tag-pill ${cls}`}>{tag}</span>
                                      <span className="repo-analytics-tag-count">{count}×</span>
                                      <div className="repo-analytics-page-chips">
                                        {pages.slice(0, 12).map(pg => (
                                          <button
                                            key={pg}
                                            className="repo-analytics-page-chip"
                                            onClick={() => jumpToPage(pg)}
                                            title={`Jump to page ${pg}`}
                                          >p{pg}</button>
                                        ))}
                                        {pages.length > 12 && (
                                          <span className="repo-analytics-page-more">+{pages.length - 12}</span>
                                        )}
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            ))}
                          </>
                        )
                      })()}
                    </div>
                  )}

                  {/* Info / metadata view */}
                  {docView === 'metadata' && (() => {
                    const doc = documents.find(d => d.id === selectedDocumentId)
                    if (!doc) return <div className="repo-analytics-empty">Document not found.</div>
                    const fields: [string, string | null | undefined][] = [
                      ['Filename', doc.filename],
                      ['Display name', doc.display_name],
                      ['Payer', doc.payer],
                      ['State', doc.state],
                      ['Program', (doc as any).program],
                      ['Authority', (doc as any).authority_level],
                      ['Effective date', (doc as any).effective_date],
                      ['Termination date', (doc as any).termination_date],
                      ['Status', doc.status],
                    ]
                    return (
                      <div className="repo-doc-view repo-doc-view--metadata">
                        <div className="repo-meta-section">
                          <div className="repo-meta-section-title">Classification</div>
                          {fields.map(([label, val]) => val ? (
                            <div key={label} className="repo-meta-row">
                              <span className="repo-meta-label">{label}</span>
                              <span className="repo-meta-value">{val}</span>
                            </div>
                          ) : null)}
                        </div>
                        <div className="repo-meta-section">
                          <div className="repo-meta-section-title">Pipeline</div>
                          <div className="repo-meta-row">
                            <span className="repo-meta-label">Extraction</span>
                            <span className={`repo-meta-status repo-meta-status--${doc.extraction_status || 'none'}`}>{doc.extraction_status || '—'}</span>
                          </div>
                          <div className="repo-meta-row">
                            <span className="repo-meta-label">Chunking</span>
                            <span className={`repo-meta-status repo-meta-status--${doc.chunking_status || 'none'}`}>{doc.chunking_status || '—'}</span>
                          </div>
                          <div className="repo-meta-row">
                            <span className="repo-meta-label">Embedding</span>
                            <span className={`repo-meta-status repo-meta-status--${doc.embedding_status || 'none'}`}>{doc.embedding_status || '—'}</span>
                          </div>
                          <div className="repo-meta-row">
                            <span className="repo-meta-label">Published</span>
                            <span className="repo-meta-value">{doc.published_at ? new Date(doc.published_at).toLocaleDateString() : '—'}</span>
                          </div>
                        </div>
                        <div className="repo-meta-pipeline-pips">
                          <PipelinePips d={doc} />
                        </div>
                      </div>
                    )
                  })()}
                </div>
              ) : (
                <div className="repo-tab-pane">
                  <div className="repo-reader-empty">
                    <p>Select a document from the left to open it.</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
