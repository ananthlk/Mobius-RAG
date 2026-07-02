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
    if (!confirm('Fix all integrity gaps? This prunes cancelled jobs, resets stuck docs, and queues re-chunk / re-embed / retag work for the workers to drain.')) return
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
  const [nly, setNly] = useState<any>(null)
  const [rep, setRep] = useState<any>(null)
  const [includeEval, setIncludeEval] = useState(true)
  const [dryRun, setDryRun] = useState(false)
  const [busy, setBusy] = useState(false)
  const running = !!nly?.running

  const fetchStatus = async () => {
    try { const r = await fetch(`${API_BASE}/admin/nightly/status`); if (r.ok) setNly(await r.json()) } catch { /* ignore */ }
  }
  const fetchReport = async () => {
    try { const r = await fetch(`${API_BASE}/admin/integrity/report`); if (r.ok) setRep(await r.json()) } catch { /* ignore */ }
  }
  useEffect(() => { fetchStatus(); fetchReport() }, [])
  useEffect(() => {
    const id = setInterval(() => { fetchStatus(); if (!running) fetchReport() }, running ? 4000 : 15000)
    return () => clearInterval(id)
  }, [running])

  const run = async () => {
    const msg = `Run nightly pipeline?${includeEval ? '\nIncludes the eval bracket (~50 min).' : ''}${dryRun ? '\n(dry run — no mutations)' : ''}`
    if (!confirm(msg)) return
    setBusy(true)
    try {
      await fetch(`${API_BASE}/admin/nightly/run`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ include_eval: includeEval, dry_run: dryRun }),
      })
      await fetchStatus()
    } finally { setBusy(false) }
  }
  const stop = async () => {
    if (!confirm('Stop the running pipeline after the current step?')) return
    await fetch(`${API_BASE}/admin/nightly/stop`, { method: 'POST' })
    await fetchStatus()
  }

  const steps: any[] = nly?.steps || []
  const lift = nly?.lift
  const gaps = rep?.gaps || {}
  const openGaps = ['unpublished_total', 'failed_docs', 'stuck_docs', 'blocked_jobs', 'stale_tags']
    .reduce((a, k) => a + (gaps[k] || 0), 0)
  const lastLift = lift?.router_recall?.delta
  const card = (label: string, value: any, sub?: string, color?: string) => (
    <div style={{ background: 'var(--mobius-surface-elevated, #f8fafc)', borderRadius: 8, padding: '10px 14px' }}>
      <div style={{ fontSize: 12, color: 'var(--mobius-text-secondary)' }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 600, color: color || 'inherit' }}>{value}
        {sub && <span style={{ fontSize: 12, color: 'var(--mobius-text-secondary)', fontWeight: 400 }}> {sub}</span>}</div>
    </div>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* header + actions */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
        <div>
          <div style={{ fontSize: 15, fontWeight: 600 }}>Nightly pipeline</div>
          <div style={{ fontSize: 12, color: 'var(--mobius-text-secondary)' }}>Corpus + lexicon → RAG → chat, bracketed by eval</div>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <label style={{ fontSize: 12, color: 'var(--mobius-text-secondary)', display: 'flex', gap: 4, alignItems: 'center' }}>
            <input type="checkbox" checked={includeEval} disabled={running} onChange={e => setIncludeEval(e.target.checked)} /> eval bracket</label>
          <label style={{ fontSize: 12, color: 'var(--mobius-text-secondary)', display: 'flex', gap: 4, alignItems: 'center' }}>
            <input type="checkbox" checked={dryRun} disabled={running} onChange={e => setDryRun(e.target.checked)} /> dry run</label>
          {running ? (
            <button type="button" onClick={stop}
              style={{ padding: '7px 14px', borderRadius: 8, border: '1px solid #ef4444', color: '#ef4444', background: 'transparent', fontSize: 13, fontWeight: 600, cursor: 'pointer' }}>Stop</button>
          ) : (
            <button type="button" onClick={run} disabled={busy}
              style={{ padding: '7px 14px', borderRadius: 8, border: 'none', color: '#fff', background: '#2563eb', fontSize: 13, fontWeight: 600, cursor: busy ? 'wait' : 'pointer' }}>
              {busy ? 'Starting…' : '▶ Run pipeline'}</button>
          )}
        </div>
      </div>

      {/* metric cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0,1fr))', gap: 12 }}>
        {card('Published', (rep?.published ?? '—').toLocaleString?.() ?? rep?.published ?? '—', rep ? `/ ${rep.documents_total?.toLocaleString?.()}` : '')}
        {card('Lexicon rev', rep?.current_lexicon_revision ?? '—')}
        {card('Open gaps', rep ? openGaps : '—')}
        {card('Last lift Δrouter', typeof lastLift === 'number' ? `${lastLift >= 0 ? '+' : ''}${lastLift.toFixed(3)}` : '—', '',
          typeof lastLift === 'number' ? (lastLift >= 0 ? '#10b981' : '#ef4444') : undefined)}
      </div>

      {/* live stage tracker */}
      {steps.length > 0 && (
        <div style={{ background: 'var(--mobius-surface, #fff)', border: '1px solid var(--mobius-border, #e2e8f0)', borderRadius: 12, padding: '12px 16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, fontWeight: 600, marginBottom: 10 }}>
            <span>Run progress</span>
            <span style={{ color: 'var(--mobius-text-secondary)', fontWeight: 400 }}>
              {nly?.running ? `running · ${_fmtDur(nly?.started_at)}` : nly?.finished_at ? `done · ${_fmtDur(nly?.started_at, nly?.finished_at)}` : 'idle'}
              {nly?.error ? ` · error: ${nly.error}` : ''}
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
                    {(s.key === 'baseline_eval' || s.key === 'final_eval') && nly?.eval_run_id && s.status === 'running' && <> · run {String(nly.eval_run_id).slice(0, 8)}</>}
                  </div>
                  {s.key === 'gate' && nly?.gate && (
                    <div style={{ fontFamily: 'monospace', marginBottom: 6 }}>
                      published {nly.gate.published}/{nly.gate.documents_total} · frac {nly.gate.frac} · stale_tags {nly.gate.stale_tags} → {nly.gate.passed ? 'PASS' : 'FAIL'}
                    </div>
                  )}
                  {log.length > 0 ? (
                    <div style={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap', lineHeight: 1.5, maxHeight: 220, overflow: 'auto', color: 'var(--mobius-text-secondary)' }}>
                      {log.join('\n')}
                    </div>
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
      )}

      {/* eval bracket */}
      {lift && (
        <div style={{ background: 'var(--mobius-surface, #fff)', border: '1px solid var(--mobius-border, #e2e8f0)', borderRadius: 12, padding: '12px 16px' }}>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Eval bracket — baseline → final</div>
          <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
            <thead><tr style={{ color: 'var(--mobius-text-secondary)', fontSize: 12 }}>
              <td>metric</td><td style={{ textAlign: 'right' }}>baseline</td><td style={{ textAlign: 'right' }}>final</td><td style={{ textAlign: 'right' }}>Δ</td></tr></thead>
            <tbody>
              {['router_recall', 'oracle_recall', 'best_single_recall', 'routing_headroom'].map(k => {
                const row = lift[k] || {}
                const d = row.delta
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
  documentsLoading = false,
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
  // Search moved to Test tab (2026-04-29). State + useEffect kept stubbed
  // so the rest of the file's references compile, but the UI is gone.
  const [, setSearchResults] = useState<CorpusChunk[] | null>(null)
  const [, setSearchTelemetry] = useState<SearchTelemetry | null>(null)
  const [, setSearchLoading] = useState(false)

  // ── Reader state ─────────────────────────────────────────────────────────
  const [urlPreview, setUrlPreview] = useState<{ url: string; ingested: boolean } | null>(null)
  const [ingestingUrl, setIngestingUrl] = useState(false)
  // Local navigate-to (from clicking a search result chunk)
  const [localNavigateTo, setLocalNavigateTo] = useState<NavigateToRead | null>(null)

  // ── New layout state ─────────────────────────────────────────────────────
  const [leftCollapsed, setLeftCollapsed] = useState(false)
  const [rightTab, setRightTab] = useState<'reader' | 'search' | 'status'>('reader')

  // Auto-open reader tab when an external navigateToRead arrives
  useEffect(() => {
    if (navigateToRead) setRightTab('reader')
  }, [navigateToRead])

  // ── Browse panel state ───────────────────────────────────────────────────
  const [browseSearch, setBrowseSearch] = useState('')
  const [payerFilter, setPayerFilter] = useState('')
  const [stateFilter, setStateFilter] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  // Track which payer groups are collapsed
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>({})

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
    setRightTab('reader')  // auto-switch to reader tab
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
    setRightTab('reader')
  }

  const closeReader = () => {
    setUrlPreview(null)
    setLocalNavigateTo(null)
    // don't change rightTab — stay on reader tab showing empty state
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
  const filteredDocs = useMemo(() => {
    const q = browseSearch.toLowerCase()
    return documents.filter((d) => {
      // text search
      if (q) {
        const haystack = [d.display_name, d.filename, d.payer, d.state]
          .filter(Boolean).join(' ').toLowerCase()
        if (!haystack.includes(q)) return false
      }
      // payer filter
      if (payerFilter && (d.payer ?? '') !== payerFilter) return false
      // state filter
      if (stateFilter && (d.state ?? '') !== stateFilter) return false
      // status filter
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
  }, [documents, browseSearch, payerFilter, stateFilter, statusFilter])

  // Group by payer
  const groupedDocs = useMemo(() => {
    const groups: Record<string, DocLike[]> = {}
    for (const d of filteredDocs) {
      const key = d.payer || d.state || '(untagged)'
      if (!groups[key]) groups[key] = []
      groups[key].push(d)
    }
    return groups
  }, [filteredDocs])

  // Unique filter options
  const uniquePayers = useMemo(() =>
    Array.from(new Set(documents.map((d) => d.payer).filter(Boolean) as string[])).sort(),
    [documents])
  const uniqueStates = useMemo(() =>
    Array.from(new Set(documents.map((d) => d.state).filter(Boolean) as string[])).sort(),
    [documents])

  // Age formatter
  const fmtAge = (iso: string | null | undefined) => {
    if (!iso) return ''
    const diff = Date.now() - new Date(iso).getTime()
    const days = Math.floor(diff / 86400000)
    if (days === 0) return 'today'
    if (days === 1) return '1d'
    if (days < 30) return `${days}d`
    const months = Math.floor(days / 30)
    return `${months}mo`
  }

  // Status dot class
  const statusDotClass = (d: DocLike) => {
    if (d.published_at) return 'published'
    if (d.chunking_status === 'failed' || d.embedding_status === 'failed') return 'failed'
    if (d.chunking_status === 'completed' || d.embedding_status === 'completed') return 'processing'
    return 'pending'
  }

  // Selected doc
  const selectedDoc = useMemo(
    () => documents.find((d) => d.id === selectedDocumentId) ?? null,
    [documents, selectedDocumentId],
  )

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
  return (
    <div className={`repo-layout${leftCollapsed ? ' nav-collapsed' : ''}`}>
      {/* Left nav */}
      <div className="repo-nav">
        {/* Toggle rail — always visible */}
        <div className="repo-nav-rail">
          <button
            className="repo-nav-toggle"
            onClick={() => setLeftCollapsed(v => !v)}
            title={leftCollapsed ? 'Expand panel' : 'Collapse panel'}
          >
            {leftCollapsed ? '›' : '‹'}
          </button>
        </div>

        {/* Nav content — hidden when collapsed */}
        {!leftCollapsed && (
          <div className="repo-nav-content">
            {/* Search + filters */}
            <div className="repo-nav-search">
              <input
                type="search"
                placeholder="Search documents, payer…"
                value={browseSearch}
                onChange={(e) => setBrowseSearch(e.target.value)}
              />
              <div className="repo-nav-filters">
                <select value={payerFilter} onChange={(e) => setPayerFilter(e.target.value)}>
                  <option value="">Payer ▾</option>
                  {uniquePayers.map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
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

            {/* Doc count badge */}
            <div className="repo-nav-count">
              {filteredDocs.length.toLocaleString()} document{filteredDocs.length !== 1 ? 's' : ''}
              {documents.length > filteredDocs.length && ` of ${documents.length.toLocaleString()}`}
            </div>

            {/* Document list grouped by payer */}
            {filteredDocs.length === 0 ? (
              <div className="repo-nav-empty">
                {documentsLoading ? 'Loading…' : documents.length === 0 ? 'No documents yet.' : 'No matches.'}
              </div>
            ) : (
              <ul className="repo-nav-list">
                {Object.entries(groupedDocs).map(([group, docs]) => {
                  const isCollapsed = !!collapsedGroups[group]
                  return (
                    <li key={group}>
                      <div
                        className="repo-nav-group-header"
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
                          onClick={() => openDocument(d.id)}
                          title={d.display_name || d.filename}
                        >
                          <span className={`repo-nav-dot ${statusDotClass(d)}`} />
                          <span className="repo-nav-doc-name">{d.display_name || d.filename}</span>
                          {d.published_at && (
                            <span className="repo-nav-doc-age">{fmtAge(d.published_at)}</span>
                          )}
                        </div>
                      ))}
                    </li>
                  )
                })}
              </ul>
            )}
          </div>
        )}
      </div>

      {/* Right main area */}
      <div className="repo-main-area">
        {/* Tab bar */}
        <div className="repo-right-tabs">
          {(['reader', 'search', 'status'] as const).map((tab) => (
            <button
              key={tab}
              className={`repo-right-tab-btn${rightTab === tab ? ' active' : ''}`}
              onClick={() => setRightTab(tab)}
            >
              {tab === 'reader'
                ? selectedDoc
                  ? `Reader · ${(selectedDoc.display_name || selectedDoc.filename).slice(0, 28)}${(selectedDoc.display_name || selectedDoc.filename).length > 28 ? '…' : ''}`
                  : 'Reader'
                : tab === 'search' ? 'Search'
                : 'Status'}
            </button>
          ))}
          {/* Refresh button on the right */}
          {onRefresh && (
            <button
              className="repo-right-refresh"
              onClick={onRefresh}
              title="Refresh documents"
            >↺</button>
          )}
        </div>

        {/* Tab content */}
        <div className="repo-right-content">
          {/* Reader tab */}
          {rightTab === 'reader' && (
            <div className="repo-tab-pane">
              {!selectedDocumentId && !urlPreview ? (
                <div className="repo-reader-empty">
                  <p>Select a document from the left panel to open it.</p>
                </div>
              ) : (
                <ReaderSlideOut
                  documents={documents}
                  selectedDocumentId={selectedDocumentId}
                  navigateToRead={activeNavigateTo}
                  onNavigateToReadConsumed={() => {
                    onNavigateToReadConsumed()
                    setLocalNavigateTo(null)
                  }}
                  onDocumentSelect={onDocumentSelect}
                  urlPreview={urlPreview}
                  onIngestUrl={handleIngestUrl}
                  ingestingUrl={ingestingUrl}
                  onClose={closeReader}
                />
              )}
            </div>
          )}

          {/* Search tab */}
          {rightTab === 'search' && (
            <div className="repo-tab-pane repo-search-pane-wrap">
              <div className="repo-search-bar-row">
                <input
                  type="search"
                  className="repo-search-input-main"
                  placeholder="Search corpus semantically…"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
                <div className="repo-search-mode-btns">
                  {(['corpus', 'precision', 'recall'] as const).map((m) => (
                    <button
                      key={m}
                      className={`repo-search-mode-btn${searchMode === m ? ' active' : ''}`}
                      onClick={() => setSearchMode(m)}
                    >
                      {m === 'corpus' ? 'Hybrid' : m === 'precision' ? 'BM25' : 'Semantic'}
                    </button>
                  ))}
                </div>
              </div>
              <div className="repo-search-results-area">
                {!searchQuery.trim() && (
                  <div className="repo-reader-empty">
                    <p>Type to search corpus content semantically.</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Status tab */}
          {rightTab === 'status' && (
            <div className="repo-tab-pane repo-status-pane">
              <PipelinePanel documents={documents} onRefresh={onRefresh} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
