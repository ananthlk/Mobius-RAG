import { useCallback, useEffect, useState } from 'react'
import { API_BASE } from '../../config'
import './CorpusHealthTab.css'

/**
 * Corpus Health — the whole pipeline in one view.
 *
 *   sources → GCS → extract → classify → chunk → embed → version → dedup → publish
 *
 * Every stage reports what reached it, what is stuck, WHY, and the action that
 * clears it. Every number opens the list behind it — a count you cannot open is
 * a count nobody acts on.
 *
 * Spec: docs/versioning-dedup-gate-spec.md §12.2 / §12.3.
 */

interface Stage {
  stage: string
  label: string
  reached: number
  missing: number
  of: string
  missing_reason: string
  action: string
  action_key: string
  drill: string
  tone: 'good' | 'bad' | 'warn'
}

interface Source {
  source: string
  label: string
  blurb: string
  documents: number | null
  last_7d: number | null
  not_chunked: number | null
  drill: string | null
  /** Org upload and personal vault live in other stores; this corpus cannot count them. */
  external?: boolean
  owner?: string
  why?: string
}

interface Classifier {
  key: string
  label: string
  owner: string
  what: string
  external: boolean
  why?: string
  scored: number | null
  coverage_pct: number | null
  /** A gating classifier can halt the pipeline, not just enrich it. */
  gating?: boolean
  blocked?: number | null
}

interface Health {
  payer: string | null
  documents_total: number
  sources: Source[]
  classifiers: Classifier[]
  inflight: Record<string, number>
  stages: Stage[]
  gate: {
    measured: boolean
    run_id?: string
    measured_at?: string
    documents_scored?: number
    by_decision?: Record<string, number>
    awaiting_adjudication?: number
    chunks_carried?: number
    chunks_reembedded?: number
  }
  ordering_clock: {
    with_publication_date: number
    with_filename_date: number
    total: number
    coverage_pct: number
  }
}

interface DrillDoc {
  id: string
  filename: string
  display_name: string | null
  payer: string | null
  status: string
  created_at: string | null
  effective_date: string | null
}

const n = (v: number | undefined) => (v ?? 0).toLocaleString()

export function CorpusHealthTab() {
  const [payer, setPayer] = useState<string>('AHCA')
  const [health, setHealth] = useState<Health | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [drill, setDrill] = useState<{ key: string; label: string } | null>(null)
  const [drillDocs, setDrillDocs] = useState<DrillDoc[] | null>(null)
  const [drillLoading, setDrillLoading] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const q = payer ? `?payer=${encodeURIComponent(payer)}` : ''
      const r = await fetch(`${API_BASE}/corpus/health${q}`)
      if (!r.ok) throw new Error(`health ${r.status}`)
      setHealth(await r.json())
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load corpus health')
    } finally {
      setLoading(false)
    }
  }, [payer])

  useEffect(() => { load() }, [load])

  const openDrill = async (key: string, label: string) => {
    setDrill({ key, label })
    setDrillDocs(null)
    setDrillLoading(true)
    try {
      const q = payer ? `?payer=${encodeURIComponent(payer)}` : ''
      const r = await fetch(`${API_BASE}/corpus/health/drill/${encodeURIComponent(key)}${q}`)
      if (!r.ok) throw new Error(`drill ${r.status}`)
      const d = await r.json()
      setDrillDocs(d.documents || [])
    } catch {
      setDrillDocs([])
    } finally {
      setDrillLoading(false)
    }
  }

  const g = health?.gate
  const oc = health?.ordering_clock

  return (
    <div className="ch-root">
      <header className="ch-head">
        <div>
          <h2>Corpus health</h2>
          <p className="ch-sub">
            Where documents come from, how far they get, and what to do about the ones that stop.
          </p>
        </div>
        <div className="ch-controls">
          <select value={payer} onChange={e => setPayer(e.target.value)} className="ch-select">
            <option value="">All payers</option>
            <option value="AHCA">AHCA</option>
            <option value="Sunshine Health">Sunshine Health</option>
            <option value="Humana">Humana</option>
            <option value="Samhsa">Samhsa</option>
          </select>
          <button className="ch-btn" onClick={load} disabled={loading}>
            {loading ? 'Loading…' : 'Refresh'}
          </button>
        </div>
      </header>

      {error && <div className="ch-error">{error}</div>}

      {health && (
        <>
          {/* ── SOURCES ─────────────────────────────────────────────── */}
          <section className="ch-section">
            <h3>Sources — how documents arrive</h3>
            <p className="ch-note ch-note-top">
              Every source lands the raw file in GCS, and from there a single ingestion pipeline
              runs. New sources (email is next) join here without changing anything downstream.
            </p>
            <div className="ch-sources">
              {health.sources
                .filter(s => s.external || (s.documents ?? 0) > 0)
                .map(s => s.external ? (
                  <div key={s.source} className="ch-source ch-source-ext">
                    <div className="ch-source-top">
                      <span className="ch-source-label">{s.label}</span>
                      <span className="ch-ext-tag">elsewhere</span>
                    </div>
                    <div className="ch-source-count ch-muted">—</div>
                    <div className="ch-source-blurb">{s.blurb}</div>
                    <div className="ch-source-why">{s.why}</div>
                    <div className="ch-source-owner">owned by {s.owner}</div>
                  </div>
                ) : (
                  <button key={s.source} className="ch-source"
                          onClick={() => s.drill && openDrill(s.drill, s.label)}>
                    <div className="ch-source-top">
                      <span className="ch-source-label">{s.label}</span>
                      {(s.last_7d ?? 0) > 0 && <span className="ch-live">● {n(s.last_7d!)} this week</span>}
                    </div>
                    <div className="ch-source-count">{n(s.documents!)}</div>
                    <div className="ch-source-blurb">{s.blurb}</div>
                    {(s.not_chunked ?? 0) > 0 && (
                      <div className="ch-source-warn">{n(s.not_chunked!)} never chunked</div>
                    )}
                  </button>
                ))}
            </div>
            <div className="ch-inflight">
              <span>In flight now:</span>
              <b>{n(health.inflight.chunking_pending)}</b> chunking
              <span className="ch-dot">·</span>
              <b className={health.inflight.chunking_blocked ? 'ch-red' : ''}>
                {n(health.inflight.chunking_blocked)}
              </b> blocked
              <span className="ch-dot">·</span>
              <b className={health.inflight.chunking_failed_24h ? 'ch-red' : ''}>
                {n(health.inflight.chunking_failed_24h)}
              </b> failed 24h
            </div>
          </section>

          <div className="ch-converge">
            <span>all sources land in</span><b>GCS</b><span>→ one ingestion pipeline</span>
          </div>

          {/* ── PIPELINE WATERFALL ──────────────────────────────────── */}
          <section className="ch-section">
            <h3>Pipeline — what reached each stage</h3>
            <table className="ch-table">
              <thead>
                <tr>
                  <th>Stage</th>
                  <th className="num">Reached</th>
                  <th className="num">Stuck</th>
                  <th>Why it stopped</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {health.stages.map(st => (
                  <tr key={st.stage} className={st.missing > 0 ? `tone-${st.tone}` : ''}>
                    <td className="ch-stage-label">
                      <span className={`ch-pip tone-${st.missing === 0 ? 'good' : st.tone}`} />
                      {st.label}
                    </td>
                    <td className="num">{n(st.reached)}</td>
                    <td className="num">
                      {st.missing > 0 ? (
                        <button className="ch-num-link" onClick={() => openDrill(st.drill, st.label)}>
                          {n(st.missing)}
                        </button>
                      ) : <span className="ch-zero">—</span>}
                    </td>
                    <td className="ch-reason">{st.missing > 0 ? st.missing_reason : ''}</td>
                    <td className="ch-action">{st.missing > 0 ? st.action : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="ch-note">
              “Stuck” counts documents that reached the previous stage but not this one — the
              population an action would actually move.
            </p>
          </section>

          {/* ── CLASSIFIERS ─────────────────────────────────────────── */}
          <section className="ch-section">
            <h3>Classifiers — “classified” is several things</h3>
            <p className="ch-note ch-note-top">
              A document scored by one classifier and not another is not unclassified. Each runs
              independently and more will be added.
            </p>
            <table className="ch-table">
              <thead>
                <tr><th>Classifier</th><th>What it decides</th><th>Owner</th>
                    <th className="num">Scored</th><th className="num">Coverage</th></tr>
              </thead>
              <tbody>
                {health.classifiers.map(cf => (
                  <tr key={cf.key} className={cf.external ? 'ch-ext-row' : ''}>
                    <td className="ch-stage-label">
                      <span className={`ch-pip ${cf.external ? '' :
                        (cf.coverage_pct! > 90 ? 'tone-good' : cf.coverage_pct! > 10 ? 'tone-warn' : 'tone-bad')}`} />
                      {cf.label}
                      {cf.gating && <span className="ch-gate-tag">gate</span>}
                    </td>
                    <td className="ch-reason">
                      {cf.what}
                      {cf.gating && (cf.blocked ?? 0) > 0 && (
                        <span className="ch-blocked"> · {n(cf.blocked!)} blocked</span>
                      )}
                    </td>
                    <td className="ch-owner">{cf.owner}</td>
                    <td className="num">{cf.external ? '—' : n(cf.scored!)}</td>
                    <td className="num">
                      {cf.external
                        ? <span className="ch-ext-tag">elsewhere</span>
                        : `${cf.coverage_pct}%`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          {/* ── VERSIONING & DEDUP ──────────────────────────────────── */}
          <section className="ch-section">
            <h3>Versioning &amp; deduplication</h3>
            {!g?.measured ? (
              <div className="ch-empty">
                The gate has not run yet. Version and duplicate state is unmeasured — not zero.
              </div>
            ) : (
              <>
                <div className="ch-cards">
                  <div className="ch-card">
                    <div className="ch-card-n">{n(g.documents_scored)}</div>
                    <div className="ch-card-l">documents scored</div>
                  </div>
                  <div className="ch-card">
                    <div className="ch-card-n ch-amber">{n(g.awaiting_adjudication)}</div>
                    <div className="ch-card-l">awaiting a human</div>
                    <div className="ch-card-s">each one is an extra active version competing in retrieval</div>
                  </div>
                  <div className="ch-card">
                    <div className="ch-card-n ch-green">{n(g.chunks_carried)}</div>
                    <div className="ch-card-l">chunks carried forward</div>
                    <div className="ch-card-s">embeddings reused, not recomputed</div>
                  </div>
                  <div className="ch-card">
                    <div className="ch-card-n">{n(g.chunks_reembedded)}</div>
                    <div className="ch-card-l">chunks re-embedded</div>
                  </div>
                </div>
                <div className="ch-decisions">
                  {Object.entries(g.by_decision || {})
                    .sort((a, b) => b[1] - a[1])
                    .map(([k, v]) => (
                      <span key={k} className={`ch-chip ch-chip-${k}`}>
                        {k.replace(/_/g, ' ')} <b>{n(v)}</b>
                      </span>
                    ))}
                </div>
                <p className="ch-note">
                  Last run {g.measured_at ? new Date(g.measured_at).toLocaleString() : '—'}
                </p>
              </>
            )}
          </section>

          {/* ── ORDERING CLOCK ──────────────────────────────────────── */}
          <section className="ch-section">
            <h3>Ordering clock</h3>
            <p className="ch-note ch-note-top">
              A version chain can only be ordered if each edition carries a date. Publication date
              (from the file) covers most of the corpus; where it is missing, ordering falls back to
              when we first saw the document, which is unreliable for anything backfilled.
            </p>
            <div className="ch-bar">
              <div className="ch-bar-fill" style={{ width: `${oc?.coverage_pct ?? 0}%` }} />
            </div>
            <div className="ch-bar-legend">
              <span><b>{n(oc?.with_publication_date)}</b> with a publication date</span>
              <span><b>{oc?.coverage_pct}%</b> of {n(oc?.total)}</span>
            </div>
          </section>
        </>
      )}

      {/* ── DRILL-DOWN ────────────────────────────────────────────── */}
      {drill && (
        <div className="ch-drill-backdrop" onClick={() => setDrill(null)}>
          <div className="ch-drill" onClick={e => e.stopPropagation()}>
            <header>
              <h4>{drill.label}</h4>
              <button className="ch-close" onClick={() => setDrill(null)}>✕</button>
            </header>
            {drillLoading && <div className="ch-empty">Loading…</div>}
            {!drillLoading && drillDocs && drillDocs.length === 0 && (
              <div className="ch-empty">Nothing here.</div>
            )}
            {!drillLoading && drillDocs && drillDocs.length > 0 && (
              <table className="ch-table ch-drill-table">
                <thead>
                  <tr><th>Document</th><th>Payer</th><th>Status</th><th>Added</th></tr>
                </thead>
                <tbody>
                  {drillDocs.map(d => (
                    <tr key={d.id}>
                      <td className="ch-fn">{d.display_name || d.filename}</td>
                      <td>{d.payer || '—'}</td>
                      <td>{d.status}</td>
                      <td>{d.created_at ? d.created_at.slice(0, 10) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
