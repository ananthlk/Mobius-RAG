import { useCallback, useEffect, useState } from 'react'
import { API_BASE, PAYOR_BASE, PAYOR_QUEUE_PATH } from '../../config'
import './CorpusHealthTab.css'

/**
 * Corpus Health — v3 layout.
 *
 *   sources of entry → GCS → time to serve → pipeline → stopped → classifiers → versioning
 *
 * Reads the gate's telemetry (gate_decisions) rather than recomputing over 2M
 * chunks. That is both correct — the page reports what the gate DECIDED, not a
 * second opinion — and the difference between 0.5s and 110s. The first cut
 * recomputed everything per request and the tab never finished loading.
 *
 * Spec: docs/versioning-dedup-gate-spec.md §12.2 / §12.3.
 */

interface Source {
  source: string; label: string; blurb: string
  documents: number | null; last_7d: number | null; drill: string | null
  external?: boolean; owner?: string; why?: string
}
interface Stage {
  stage: string; label: string; reached: number; missing: number; stopped: number
  missing_reason: string; action: string; action_key: string | null
  drill: string; tone: 'good' | 'bad' | 'warn'
}
interface Stopped {
  status: string; label: string; why: string; unblocked_by: string
  owner: string; count: number; drill: string
}
interface Classifier {
  key: string; label: string; owner: string; what: string
  external: boolean; why?: string; gating?: boolean
  blocked?: number | null; scored: number | null; coverage_pct: number | null
}
interface Health {
  payer: string | null; documents_total: number; as_of: string | null
  sources: Source[]; stages: Stage[]; stopped: Stopped[]; classifiers: Classifier[]
  inflight: Record<string, number>
  gate: {
    measured: boolean; measured_at?: string; documents_scored?: number
    by_decision?: Record<string, number>; awaiting_adjudication?: number
    successors?: number; chunks_carried?: number; chunks_reembedded?: number
    tracked?: number; unpublishable?: number
  }
  duplicates: {
    measured: boolean; measured_at?: string; run_id?: string
    by_kind?: Record<string, number>
    retirable?: number; held_no_date?: number
    documents?: number; high_value_documents?: number; high_value_basis?: string
    managed_documents?: number; unmanaged_documents?: number
    managed_by_kind?: Record<string, number>; managed_basis?: string
  }
  queue: {
    measured: boolean
    scored?: { managed: number; unmanaged: number }
    buckets?: Record<string, { managed: number; unmanaged: number }>
  }
}
interface StageLat {
  step: string; label: string; kind: 'wait' | 'work'
  p50_min: number; p90_min: number; documents: number; share_pct: number
}
interface TTS {
  source: string; label: string; documents: number
  p50_min: number | null; p90_min: number | null
}
interface DrillDoc {
  id: string; filename: string; display_name: string | null; payer: string | null
  status: string; created_at: string | null; effective_date: string | null
}

const n = (v: number | null | undefined) => (v ?? 0).toLocaleString()

/** Collapsible section. Sections carrying an attention count show it in the
 *  header, so a collapsed section still tells you whether it needs you. */
function Section({ title, badge, tone, defaultOpen = true, children }: {
  title: string
  badge?: string | number | null
  tone?: 'good' | 'warn' | 'bad'
  defaultOpen?: boolean
  children: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <section className={`ch-acc ${open ? 'is-open' : ''}`}>
      <button className="ch-acc-head" onClick={() => setOpen(o => !o)} aria-expanded={open}>
        <svg className="ch-chev" viewBox="0 0 12 12" aria-hidden="true">
          <path d="M4 2.5 L8 6 L4 9.5" fill="none" stroke="currentColor"
                strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span className="ch-acc-title">{title}</span>
        {badge != null && badge !== '' && (
          <span className={`ch-acc-badge${tone ? ' ' + tone : ''}`}>{badge}</span>
        )}
      </button>
      {open && <div className="ch-acc-body">{children}</div>}
    </section>
  )
}
const mins = (v: number | null) =>
  v == null ? '—' : v < 1 ? `${Math.round(v * 60)}s` : v < 90 ? `${v}m` : `${(v / 60).toFixed(1)}h`

export function CorpusHealthTab() {
  const [payer, setPayer] = useState('AHCA')
  // Scope is global — payer AND window apply to every section, so a range
  // isolates an issue across the whole page, not only in ingestion.
  const [since, setSince] = useState('')
  const [until, setUntil] = useState('')
  const [health, setHealth] = useState<Health | null>(null)
  const [tts, setTts] = useState<TTS[] | null>(null)
  const [lat, setLat] = useState<StageLat[] | null>(null)
  // source -> its own stage breakdown. Crossing the two is the only view that
  // says whether a slow source is slow for the same reason as everything else.
  const [srcLat, setSrcLat] = useState<Record<string, StageLat[] | 'loading'>>({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [q, setQ] = useState('')
  const [drill, setDrill] = useState<{ key: string; label: string } | null>(null)
  const [drillDocs, setDrillDocs] = useState<DrillDoc[] | null>(null)

  const scopeQS = useCallback(() => {
    const p = new URLSearchParams()
    if (payer) p.set('payer', payer)
    if (since) p.set('since', since)
    if (until) p.set('until', until)
    return p.toString() ? `?${p}` : ''
  }, [payer, since, until])

  const load = useCallback(async () => {
    setLoading(true); setError(null)
    try {
      const r = await fetch(`${API_BASE}/corpus/health${scopeQS()}`)
      if (!r.ok) throw new Error(`health ${r.status}`)
      setHealth(await r.json())
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load corpus health')
    } finally { setLoading(false) }
  }, [scopeQS])

  useEffect(() => { load() }, [load])

  // Time to serve is the one genuinely slow query left; fetched separately so
  // the rest of the page renders without waiting for it.
  useEffect(() => {
    let dead = false
    const qs = scopeQS()
    fetch(`${API_BASE}/corpus/health/time-to-serve${qs || '?'}${qs ? '&' : ''}days=30`)
      .then(r => (r.ok ? r.json() : null))
      .then(d => { if (!dead && d) setTts(d.sources) })
      .catch(() => { })
    const q2 = scopeQS()
    fetch(`${API_BASE}/corpus/health/stage-latency${q2 || '?'}${q2 ? '&' : ''}days=30`)
      .then(r => (r.ok ? r.json() : null))
      .then(d => { if (!dead && d) setLat(d.steps) })
      .catch(() => { })
    return () => { dead = true }
  }, [scopeQS])

  const toggleSourceLat = async (src: string) => {
    if (srcLat[src]) { setSrcLat(p => { const c = { ...p }; delete c[src]; return c }) ; return }
    setSrcLat(p => ({ ...p, [src]: 'loading' }))
    const qs = scopeQS()
    try {
      const r = await fetch(
        `${API_BASE}/corpus/health/stage-latency${qs || '?'}${qs ? '&' : ''}days=30&source=${encodeURIComponent(src)}`)
      const d = r.ok ? await r.json() : null
      setSrcLat(p => ({ ...p, [src]: d?.steps ?? [] }))
    } catch { setSrcLat(p => ({ ...p, [src]: [] })) }
  }

  const openDrill = async (key: string, label: string) => {
    setDrill({ key, label }); setDrillDocs(null)
    try {
      const r = await fetch(`${API_BASE}/corpus/health/drill/${encodeURIComponent(key)}${scopeQS()}`)
      setDrillDocs(r.ok ? (await r.json()).documents || [] : [])
    } catch { setDrillDocs([]) }
  }

  const g = health?.gate
  const d = health?.duplicates
  const qq = health?.queue
  const asOf = health?.as_of ? new Date(health.as_of).toLocaleString() : null

  // Which preset is active is DERIVED from the dates rather than stored, so it
  // stays truthful when the range is edited by hand — a stored flag would keep
  // "30d" lit while the inputs said something else.
  const PRESETS: [string, number][] = [['7d', 7], ['30d', 30], ['90d', 90]]
  const iso = (d: Date) => d.toISOString().slice(0, 10)
  const activePreset = (() => {
    if (!since && !until) return 'all'
    if (until !== iso(new Date())) return null
    const hit = PRESETS.find(([, d]) => since === iso(new Date(Date.now() - d * 864e5)))
    return hit ? hit[0] : null
  })()

  return (
    <div className="ch-root">
      <div className="ch-scope">
        <select value={payer} onChange={e => setPayer(e.target.value)}>
          <option value="">All payers</option>
          <option value="AHCA">AHCA</option>
          <option value="Sunshine Health">Sunshine Health</option>
          <option value="Humana">Humana</option>
          <option value="Samhsa">Samhsa</option>
        </select>
        <input
          placeholder="Search a document by name — see its own journey"
          value={q}
          onChange={e => setQ(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && q.trim()) openDrill(`search:${q.trim()}`, `“${q.trim()}”`) }}
        />
        <button className="ch-go" onClick={() => q.trim() && openDrill(`search:${q.trim()}`, `“${q.trim()}”`)}>
          Search
        </button>
        <span className="ch-range">
          <input type="date" value={since} max={until || undefined}
                 onChange={e => setSince(e.target.value)} aria-label="From" />
          <span className="ch-dash">→</span>
          <input type="date" value={until} min={since || undefined}
                 onChange={e => setUntil(e.target.value)} aria-label="To" />
        </span>
        <span className="ch-presets" role="group" aria-label="Date range preset">
          {PRESETS.map(([lab, d]) => (
            <button
              key={lab}
              className={`ch-preset${activePreset === lab ? ' is-on' : ''}`}
              aria-pressed={activePreset === lab}
              onClick={() => {
                setSince(iso(new Date(Date.now() - d * 864e5)))
                setUntil(iso(new Date()))
              }}
            >{lab}</button>
          ))}
          <button
            className={`ch-preset${activePreset === 'all' ? ' is-on' : ''}`}
            aria-pressed={activePreset === 'all'}
            onClick={() => { setSince(''); setUntil('') }}
          >all</button>
        </span>
        <button className="ch-btn" onClick={load} disabled={loading}>
          {loading ? 'Loading…' : 'Refresh'}
        </button>
        {asOf && <span className="ch-asof">gate run {asOf}</span>}
      </div>

      {error && <div className="ch-error">{error}</div>}
      {loading && !health && <div className="ch-empty">Loading…</div>}

      {health && (
        <>
          <Section title="Sources of entry"
                   badge={`${health.sources.filter(x => !x.external).length} routes`}>
          <p className="ch-note ch-top">
            Where documents come in. Every one lands the raw file in GCS and a single pipeline
            runs from there, so this only answers “how did it get here”.
          </p>
          <div className="ch-scroll">
            <table className="ch-table ch-narrow">
              <thead><tr>
                <th>Source</th><th className="num">Documents</th>
                <th className="num">This week</th><th>What it is</th>
              </tr></thead>
              <tbody>
                {health.sources.map(s => (
                  <tr key={s.source} className={s.external ? 'ch-ext' : ''}>
                    <td className="ch-name">
                      {s.drill
                        ? <button className="ch-lnk-plain" onClick={() => openDrill(s.drill!, s.label)}>{s.label}</button>
                        : s.label}
                      {s.external && <span className="ch-tag">elsewhere</span>}
                    </td>
                    <td className="num">{s.external ? <span className="ch-zero">—</span> : n(s.documents)}</td>
                    <td className="num">
                      {(s.last_7d ?? 0) > 0
                        ? <span className="ch-live">{n(s.last_7d)}</span>
                        : <span className="ch-zero">—</span>}
                    </td>
                    <td className="ch-why">{s.external ? `${s.why} · ${s.owner}` : s.blurb}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="ch-converge">
            <span>all of the above land in</span><b>GCS</b><span>→ one pipeline</span>
          </div>
          </Section>

          <Section title="Time to serve" badge={tts?.length ? `${tts.length} sources` : null}>
          <p className="ch-note ch-top">
            The number a customer feels: document landing → first published vector, last 30 days.
          </p>
          {!tts ? <div className="ch-empty">Measuring…</div> : (
            <div className="ch-scroll">
              <table className="ch-table ch-narrow">
                <thead><tr><th>Source</th><th className="num">Docs</th>
                  <th className="num">p50</th><th className="num">p90</th></tr></thead>
                <tbody>
                  {tts.map(t => {
                    const sub = srcLat[t.source]
                    return (
                      <>
                        <tr key={t.source} className="ch-row-click" onClick={() => toggleSourceLat(t.source)}>
                          <td className="ch-name">
                            <svg className={`ch-chev ch-chev-sm${sub ? ' is-open' : ''}`} viewBox="0 0 12 12" aria-hidden="true">
                              <path d="M4 2.5 L8 6 L4 9.5" fill="none" stroke="currentColor"
                                    strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                            </svg>
                            <span className={`ch-pip ${(t.p50_min ?? 0) < 5 ? 'good' : 'warn'}`} />{t.label}
                          </td>
                          <td className="num">{n(t.documents)}</td>
                          <td className="num"><b>{mins(t.p50_min)}</b></td>
                          <td className="num">{mins(t.p90_min)}</td>
                        </tr>
                        {sub === 'loading' && (
                          <tr key={t.source + '-l'}><td colSpan={4} className="ch-sub-cell">measuring…</td></tr>
                        )}
                        {Array.isArray(sub) && sub.length > 0 && (
                          <tr key={t.source + '-s'}>
                            <td colSpan={4} className="ch-sub-cell">
                              <div className="ch-sub-head">where {t.label.toLowerCase()}’s time goes</div>
                              {sub.map(st => (
                                <div key={st.step} className="ch-sub-row">
                                  <span className={`ch-pip ${st.kind === 'wait' && st.share_pct > 50 ? 'bad' : st.kind === 'wait' ? 'warn' : 'good'}`} />
                                  <span className="ch-sub-lab">{st.label}</span>
                                  <span className="ch-sub-bar-wrap">
                                    <span className="ch-sub-bar" style={{ width: `${Math.min(st.share_pct, 100)}%` }} />
                                  </span>
                                  <span className="ch-sub-pct">{st.share_pct}%</span>
                                  <span className="ch-sub-val">{mins(st.p50_min)}</span>
                                </div>
                              ))}
                            </td>
                          </tr>
                        )}
                      </>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}

          </Section>

          <Section title="Where the time goes"
                   badge={lat ? `${lat.reduce((a, x) => a + x.p50_min, 0).toFixed(1)}m total` : null}>
            <p className="ch-note ch-top">
              The same wait, decomposed by transition. <b>Waiting</b> is capacity or scheduling;
              <b> working</b> is code. They have different fixes, so they are counted separately.
            </p>
            {!lat ? <div className="ch-empty">Measuring…</div> : (
              <div className="ch-scroll">
                <table className="ch-table ch-narrow">
                  <thead><tr><th>Transition</th><th className="num">p50</th>
                    <th className="num">p90</th><th className="num">share</th><th>Kind</th></tr></thead>
                  <tbody>
                    {lat.map(st => (
                      <tr key={st.step}>
                        <td className="ch-name">
                          <span className={`ch-pip ${st.kind === 'wait' && st.share_pct > 50 ? 'bad'
                            : st.kind === 'wait' ? 'warn' : 'good'}`} />{st.label}
                        </td>
                        <td className="num"><b>{mins(st.p50_min)}</b></td>
                        <td className="num">{mins(st.p90_min)}</td>
                        <td className="num ch-share-cell">
                          <span className="ch-share">
                            <span className="ch-share-track">
                              <span className="ch-share-bar" style={{ width: `${Math.min(st.share_pct, 100)}%` }} />
                            </span>
                            <span className="ch-share-n">{st.share_pct}%</span>
                          </span>
                        </td>
                        <td className="ch-why">{st.kind === 'wait' ? 'queue' : 'processing'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Section>

          <Section title="Pipeline"
                   badge={health.stages.reduce((a, x) => a + x.missing, 0) || null}
                   tone="bad">
          <div className="ch-scroll">
            <table className="ch-table">
              <thead><tr>
                <th>Stage</th><th className="num">Reached</th><th className="num">Stuck</th>
                <th className="num">Stopped</th><th>Why it stopped</th><th>Action</th>
              </tr></thead>
              <tbody>
                {health.stages.map(st => (
                  <tr key={st.stage}>
                    <td className="ch-name">
                      <span className={`ch-pip ${st.missing === 0 ? 'good' : st.tone}`} />{st.label}
                    </td>
                    <td className="num">{n(st.reached)}</td>
                    <td className="num">
                      {st.missing > 0
                        ? <button className="ch-lnk" onClick={() => openDrill(st.drill, st.label)}>{n(st.missing)}</button>
                        : <span className="ch-zero">—</span>}
                    </td>
                    <td className="num">
                      {st.stopped > 0
                        ? <span className="ch-stopn">{n(st.stopped)}</span>
                        : <span className="ch-zero">—</span>}
                    </td>
                    <td className="ch-why">{st.missing > 0 ? st.missing_reason : ''}</td>
                    <td className="ch-act">{st.missing > 0 ? st.action : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="ch-note">
            <b>Stuck</b> reached the previous stage and should have progressed — re-triggering is
            the fix. <b>Stopped</b> cannot progress with the capability we have; folding it into
            “stuck” turns a permanent gap into a queue nobody can clear.
            <span className="ch-inflight"> In flight now: <b>{n(health.inflight.chunking_pending)}</b> chunking
              · <b className={health.inflight.chunking_blocked ? 'ch-red' : ''}>
                {n(health.inflight.chunking_blocked)}</b> blocked</span>
          </p>

          </Section>

          {health.stopped.some(t => t.count > 0) && (
            <Section title="Stopped — not a backlog"
                     badge={health.stopped.reduce((a, t) => a + t.count, 0)}
                     tone="warn" defaultOpen={false}>
              <div className="ch-scroll">
                <table className="ch-table">
                  <thead><tr><th>State</th><th className="num">Docs</th><th>Why</th>
                    <th>What would unblock it</th><th>Owner</th></tr></thead>
                  <tbody>
                    {health.stopped.filter(t => t.count > 0).map(t => (
                      <tr key={t.status}>
                        <td className="ch-name"><span className="ch-pip stop" />{t.label}</td>
                        <td className="num">
                          <button className="ch-lnk" onClick={() => openDrill(t.drill, t.label)}>{n(t.count)}</button>
                        </td>
                        <td className="ch-why">{t.why}</td>
                        <td className="ch-act">{t.unblocked_by}</td>
                        <td className="ch-owner">{t.owner}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          )}

          <Section title="Classifiers" badge={`${health.classifiers.length}`} defaultOpen={false}>
          <div className="ch-scroll">
            <table className="ch-table">
              <thead><tr><th>Classifier</th><th>What it decides</th><th>Owner</th>
                <th className="num">Scored</th><th className="num">Coverage</th></tr></thead>
              <tbody>
                {health.classifiers.map(cf => (
                  <tr key={cf.key} className={cf.external ? 'ch-ext' : ''}>
                    <td className="ch-name">
                      <span className={`ch-pip ${cf.external ? '' :
                        (cf.coverage_pct! > 90 ? 'good' : cf.coverage_pct! > 10 ? 'warn' : 'bad')}`} />
                      {cf.label}{cf.gating && <span className="ch-tag gate">gate</span>}
                    </td>
                    <td className="ch-why">{cf.what}</td>
                    <td className="ch-owner">{cf.owner}</td>
                    <td className="num">{cf.external ? <span className="ch-zero">—</span> : n(cf.scored)}</td>
                    <td className="num">
                      {cf.external ? <span className="ch-tag">elsewhere</span> : `${cf.coverage_pct}%`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          </Section>

          <Section title="Cleanup queue"
                   badge={qq?.measured
                     ? ((qq.buckets?.awaiting_duplicate?.managed || 0)
                        + (qq.buckets?.awaiting_versioning?.managed || 0)) || null
                     : null} tone="warn">
          {!qq?.measured ? <div className="ch-empty">Nothing scored yet.</div> : (
            <>
              <p className="ch-note">
                Every scored document sits in exactly one bucket, so these sum to the total —
                that is what makes <b>clean</b> a number worth trusting. A document waiting on
                both determinations is counted once, in the one that must clear first.
                <b> Managed</b> documents are the ones the Payor platform classified and can act
                on; the rest are the long tail nobody owns.
              </p>
              <div className="ch-tablewrap">
                <table className="ch-table ch-queue">
                  <thead><tr>
                    <th>Queue</th><th className="num">Managed</th>
                    <th className="num">Unmanaged</th><th className="num">Total</th>
                  </tr></thead>
                  <tbody>
                    {([
                      ['awaiting_duplicate', 'Awaiting duplicate determination',
                       'in a candidate pair — held until a person decides, none auto-retired'],
                      ['awaiting_versioning', 'Awaiting versioning determination',
                       'overlap or ordering the gate would not call on its own'],
                      ['unpublishable', 'Unpublishable',
                       'no pages or no chunks — neither determination is possible until fixed'],
                      ['clean', 'Clean', 'scored, unambiguous, nothing pending'],
                    ] as const).map(([k, label, why]) => {
                      const b = qq.buckets?.[k] || { managed: 0, unmanaged: 0 }
                      return (
                        <tr key={k} className={k === 'clean' ? 'is-clean' : ''}>
                          <td><b>{label}</b><div className="ch-card-s">{why}</div></td>
                          <td className="num">{n(b.managed)}</td>
                          <td className="num">{n(b.unmanaged)}</td>
                          <td className="num">{n(b.managed + b.unmanaged)}</td>
                        </tr>
                      )
                    })}
                    <tr className="is-total">
                      <td><b>Documents scored</b></td>
                      <td className="num">{n(qq.scored?.managed)}</td>
                      <td className="num">{n(qq.scored?.unmanaged)}</td>
                      <td className="num">{n((qq.scored?.managed || 0) + (qq.scored?.unmanaged || 0))}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </>
          )}
          </Section>

          <Section title="Versioning &amp; deduplication"
                   badge={g?.awaiting_adjudication || null} tone="warn">
          {!g?.measured ? <div className="ch-empty">The gate has not run yet.</div> : (
            <>
              <div className="ch-cards">
                <div className="ch-card"><div className="ch-card-n">{n(g.documents_scored)}</div>
                  <div className="ch-card-l">documents scored</div></div>
                <a className="ch-card ch-card-link"
                   href={`${PAYOR_BASE}${PAYOR_QUEUE_PATH}`}
                   target="_blank" rel="noopener noreferrer">
                  <div className="ch-card-n amber">{n(g.awaiting_adjudication)}</div>
                  <div className="ch-card-l">awaiting a human <span className="ch-out">↗</span></div>
                  <div className="ch-card-s">
                    decided in Fact Store's working queue, not here — each pending one is an
                    extra active version competing in retrieval
                  </div>
                </a>
                <div className="ch-card"><div className="ch-card-n green">{n(g.chunks_carried)}</div>
                  <div className="ch-card-l">chunks carried forward</div>
                  <div className="ch-card-s">embeddings reused, not recomputed</div></div>
                <div className="ch-card"><div className="ch-card-n">{n(g.chunks_reembedded)}</div>
                  <div className="ch-card-l">chunks re-embedded</div></div>
              </div>
              <div className="ch-chips">
                {Object.entries(g.by_decision || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                  <span key={k} className={`ch-chip ch-chip-${k}`}>
                    {k.replace(/_/g, ' ')} <b>{n(v)}</b>
                  </span>
                ))}
              </div>
            </>
          )}
          </Section>

          <Section title="Duplicates"
                   badge={d?.measured ? (d.by_kind?.duplicate || null) : null} tone="warn">
          {!d?.measured ? <div className="ch-empty">Duplicate determination has not run yet.</div> : (
            <>
              <p className="ch-note">
                Duplication is not lineage — the versioning gate only compares documents that
                share a doc_key, and most of the corpus has none, so copies never met inside it.
                This is a separate pass over normalized page text.
                <b> A pair counts as a duplicate only when every signal agrees</b> — identical
                text, length, page count, reporting period and product. Everything else is held,
                because a blank annual form is identical every year and one product's copy of a
                policy is identical to another's; both are legitimately separate documents.
              </p>
              <div className="ch-cards">
                <div className="ch-card"><div className="ch-card-n">{n(d.by_kind?.duplicate || 0)}</div>
                  <div className="ch-card-l">true duplicates</div>
                  <div className="ch-card-s">every signal agreed</div></div>
                <div className="ch-card"><div className="ch-card-n green">{n(d.retirable || 0)}</div>
                  <div className="ch-card-l">safe to retire now</div>
                  <div className="ch-card-s">
                    {d.retirable ? 'canonical rests on a real edition date'
                                 : 'none — every duplicate pair lacks the dates to pick a canonical'}
                  </div></div>
                <div className="ch-card"><div className="ch-card-n amber">{n(d.held_no_date || 0)}</div>
                  <div className="ch-card-l">held, needs a human</div>
                  <div className="ch-card-s">identical but undated — picking a winner would be a coin flip</div></div>
                <a className="ch-card ch-card-link"
                   href={`${PAYOR_BASE}${PAYOR_QUEUE_PATH}`}
                   target="_blank" rel="noopener noreferrer">
                  <div className="ch-card-n amber">{n(d.managed_documents || 0)}</div>
                  <div className="ch-card-l">managed — actionable <span className="ch-out">↗</span></div>
                  <div className="ch-card-s">
                    resolved in the Payor platform's Deduplicate queue, which is scoped to
                    documents it manages. Only these are actionable there.
                  </div></a>
                <div className="ch-card"><div className="ch-card-n">{n(d.unmanaged_documents || 0)}</div>
                  <div className="ch-card-l">unmanaged</div>
                  <div className="ch-card-s">
                    scraped nav pages, bulk sources and uploads nobody manages — the long tail,
                    and the reason this page's totals exceed the Payor queue's
                  </div></div>
              </div>
              <div className="ch-chips">
                {Object.entries(d.by_kind || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                  <span key={k} className={`ch-chip ch-chip-${k}`}>
                    {k.replace(/_/g, ' ')} <b>{n(v)}</b>
                    {d.managed_by_kind?.[k] != null && (
                      <i className="ch-chip-sub">{n(d.managed_by_kind[k])} managed</i>
                    )}
                  </span>
                ))}
              </div>
              <p className="ch-note ch-note-dim">
                period series = same form, different reporting period · product variant = same
                template, different product under Medicaid · ordering unknown = no usable edition
                date on either side · product unknown = neither document declares its product.
                None of these are duplicates and nothing in them is ever retired automatically.
              </p>
            </>
          )}
          </Section>
        </>
      )}

      {drill && (
        <div className="ch-drill-backdrop" onClick={() => setDrill(null)}>
          <div className="ch-drill" onClick={e => e.stopPropagation()}>
            <header><h4>{drill.label}</h4>
              <button className="ch-close" onClick={() => setDrill(null)}>✕</button></header>
            {!drillDocs && <div className="ch-empty">Loading…</div>}
            {drillDocs && drillDocs.length === 0 && <div className="ch-empty">Nothing here.</div>}
            {drillDocs && drillDocs.length > 0 && (
              <table className="ch-table">
                <thead><tr><th>Document</th><th>Payer</th><th>Status</th><th>Added</th></tr></thead>
                <tbody>
                  {drillDocs.map(d => (
                    <tr key={d.id}>
                      <td className="ch-fn">{d.display_name || d.filename}</td>
                      <td>{d.payer || '—'}</td><td>{d.status}</td>
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
