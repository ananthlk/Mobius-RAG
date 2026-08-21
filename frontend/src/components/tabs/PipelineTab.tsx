import { useCallback, useEffect, useRef, useState } from 'react';
import './PipelineTab.css';

/**
 * Pipeline — the whole chain, end to end, refreshing on its own.
 *
 * The Repository banner already showed chunking → embedding → publishing →
 * integrity. That is the BACK HALF. A document that never scraped, never landed
 * in GCS, failed extraction, or is held by the classifier never reaches chunking
 * at all — so it was simply absent from the panel, and "9,910 documents, 9,753
 * chunked" gave no way to see where the other 157 stopped.
 *
 * This tab shows every stage in order and polls while you watch it, so movement
 * is visible at any time rather than only when someone thinks to reload.
 */

type Stage = Record<string, unknown>;
type Health = Record<string, Stage> & { totals?: Record<string, number> };
type DrillItem = { document_id: string; filename: string; at: string | null;
                   age_seconds: number | null; detail: string | null };

const REFRESH_MS = 10_000;

// "all" matters more than it looks: a stall is usually a document that entered a
// stage days ago and never left, which an hour-scoped view hides completely.
const WINDOWS: [string, string][] = [
  ['5m', '5m'], ['15m', '15m'], ['30m', '30m'], ['1h', '1h'], ['24h', '24h'], ['7d', '7d'], ['all', 'all time'],
];

// The backend already buckets the last 30 minutes into 6 × 5-minute slices
// (oldest → newest), so short windows need no extra query — they are a slice of
// data the card is already carrying.
const BUCKET_SLICE: Record<string, number> = { '5m': 1, '15m': 3, '30m': 6 };

function windowedThroughput(st: Stage, win: string, lens: 'tx' | 'docs' = 'tx'): number | null {
  if (lens === 'docs') {
    const m: Record<string, string> = {
      '1h': 'docs_last_hour', '24h': 'docs_last_24h', '7d': 'docs_last_7d', 'all': 'docs_all_time',
    };
    const dv = st[m[win]];
    if (typeof dv === 'number') return dv;
    return null;                      // short windows have no distinct-doc series
  }
  const rolling = st.rolling as { buckets_5min?: number[] } | undefined;
  const b = rolling?.buckets_5min;
  if (b && BUCKET_SLICE[win]) return b.slice(-BUCKET_SLICE[win]).reduce((a, c) => a + c, 0);
  const direct: Record<string, string> = {
    '1h': 'last_hour', '24h': 'last_24h', '7d': 'last_7d', 'all': 'all_time',
  };
  const v = st[direct[win]];
  return typeof v === 'number' ? v : null;
}

function age(sec: number | null): string {
  if (sec == null) return '—';
  if (sec < 90) return `${sec}s`;
  if (sec < 5400) return `${Math.round(sec / 60)}m`;
  if (sec < 172800) return `${Math.round(sec / 3600)}h`;
  return `${Math.round(sec / 86400)}d`;
}

// Stage order IS the pipeline order. Rendering it in any other order would
// misrepresent what feeds what.
const STAGES: { key: string; label: string; fields: [string, string][] }[] = [
  { key: 'scrape', label: 'Scrape', fields: [
      ['total', 'crawled documents'], ['last_24h', 'last 24h'],
      ['with_source_url', 'with source URL'] ] },
  { key: 'gcs', label: 'GCS', fields: [
      ['@throughput', 'landed'], ['stored', 'objects stored'],
      ['missing_object', 'row without object|good'] ] },
  { key: 'extract', label: 'Extract', fields: [
      ['extracting', 'in flight'], ['no_text', 'produced no text'],
      ['failed_typed', 'typed failures'], ['tables_captured', 'tables captured'] ] },
  { key: 'classify', label: 'Classify', fields: [
      ['classified', 'classified'], ['held_for_human', 'held for human'],
      ['unclassified', 'not yet classified'] ] },
  { key: 'chunking', label: 'Chunking', fields: [
      ['@throughput', 'completed'], ['active', 'active workers'], ['pending', 'pending|good'] ] },
  { key: 'embedding', label: 'Embedding', fields: [
      ['@throughput', 'completed'], ['active', 'active workers'], ['pending', 'pending|good'] ] },
  { key: 'versioning', label: 'Versioning / dedup', fields: [
      ['pairs_scored', 'pairs scored'], ['duplicates', 'duplicates'],
      ['retired', 'retired'], ['shelved', 'shelved'] ] },
  { key: 'publishing', label: 'Publishing', fields: [
      ['embedded_unpublished', 'genuinely unpublished'],
      ['excluded_retired', 'excluded: retired'], ['excluded_shelved', 'excluded: shelved'],
      ['excluded_no_chunks', 'excluded: no chunks'] ] },
];

function fmt(v: unknown): string {
  if (v === -1 || v === null || v === undefined) return '—';   // -1 means the query failed
  if (typeof v === 'number') return v.toLocaleString();
  return String(v);
}

/** A value that changed since the last poll gets a brief highlight — that is
 *  the whole point of a tab you leave open. */
function Metric({ label, value, good }: { label: string; value: unknown; good?: boolean }) {
  const prev = useRef<unknown>(value);
  const [bump, setBump] = useState(false);
  useEffect(() => {
    if (prev.current !== undefined && prev.current !== value) {
      setBump(true);
      const t = setTimeout(() => setBump(false), 1200);
      prev.current = value;
      return () => clearTimeout(t);
    }
    prev.current = value;
  }, [value]);
  // A zero that means "healthy" and a zero that means "stalled" look identical
  // in a count. `good` marks the ones where empty IS the correct state, so the
  // card does not read as a problem when the queue has simply drained.
  const zeroIsGood = good && (value === 0 || value === '0');
  return (
    <div className={`pl-metric${bump ? ' pl-bump' : ''}`}>
      <span className={`pl-mv${zeroIsGood ? ' pl-ok0' : ''}`}>{fmt(value)}</span>
      <span className="pl-ml">{label}{zeroIsGood ? ' ✓' : ''}</span>
    </div>
  );
}

export function PipelineTab() {
  const [h, setH] = useState<Health | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [at, setAt] = useState<Date | null>(null);
  const [live, setLive] = useState(true);
  const [integ, setInteg] = useState<any | null>(null);
  const [win, setWin] = useState('24h');
  // Corpus health counts DOCUMENTS; this tab counts TRANSACTIONS. A document is
  // chunked 2.5x on average, so the two can never match and it is not a bug —
  // it is two different questions. Naming the lens is the fix.
  const [lens, setLens] = useState<'tx' | 'docs'>('tx');
  const winRef = useRef('24h');
  useEffect(() => { winRef.current = win; }, [win]);
  const [drill, setDrill] = useState<{ stage: string; label: string } | null>(null);
  // The modal keeps its OWN window, defaulting to all-time.
  //
  // It used to inherit the page window, which made clicking a card showing
  // "155 produced no text" open a modal saying "0 documents · 24h" — the card
  // number is cumulative, the modal was windowed, and the two disagreed at a
  // glance. You open a bucket to find what is STUCK, and stuck things are old,
  // so all-time is the honest default. Narrowing is still one click.
  const [drillWin, setDrillWin] = useState('all');
  const [items, setItems] = useState<DrillItem[] | null>(null);
  const [dErr, setDErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const r = await fetch('/pipeline_health');
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setH(await r.json());
      setErr(null);
      setAt(new Date());
      // Same window as the cards, so the accounting and the stages can never
      // describe different periods.
      try {
        const ri = await fetch(`/pipeline_integrity?window=${winRef.current}`);
        if (ri.ok) setInteg(await ri.json());
      } catch { /* accounting is additive; never break the page for it */ }
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => { load(); }, [win, load]);

  useEffect(() => {
    load();
    if (!live) return;
    const t = setInterval(load, REFRESH_MS);
    return () => clearInterval(t);
  }, [load, live]);

  // Opening a bucket answers "which ones", which is always the question a
  // stuck count provokes.
  useEffect(() => {
    if (!drill) { setItems(null); setDErr(null); return; }
    let dead = false;
    (async () => {
      try {
        const r = await fetch(`/pipeline_health/stage/${drill.stage}?window=${drillWin}&limit=200`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const j = await r.json();
        if (!dead) { setItems(j.items || []); setDErr(null); }
      } catch (e) {
        if (!dead) { setItems([]); setDErr(e instanceof Error ? e.message : String(e)); }
      }
    })();
    return () => { dead = true; };
  }, [drill, drillWin]);

  const t = h?.totals || {};
  const inFlight = STAGES.flatMap(s => {
    const st = (h?.[s.key] || {}) as Stage;
    const jobs = (st.in_flight as unknown[]) || [];
    return jobs.map(j => ({ stage: s.label, job: j }));
  });

  return (
    <div className="pl-wrap">
      <div className="pl-head">
        <div>
          <h2>Pipeline</h2>
          <p className="pl-sub">
            Every stage from crawl to index. Refreshes every {REFRESH_MS / 1000}s;
            changed numbers flash.
          </p>
        </div>
        <div className="pl-controls">
          <button className={`pl-btn${live ? ' on' : ''}`} onClick={() => setLive(v => !v)}>
            {live ? '● live' : '❙❙ paused'}
          </button>
          <button className="pl-btn" onClick={load}>refresh</button>
          <span className="pl-winbar" title="Transactions counts processing runs; documents counts distinct documents. A document is chunked ~2.5x on average.">
            <button className={`pl-win${lens === 'tx' ? ' on' : ''}`} onClick={() => setLens('tx')}>transactions</button>
            <button className={`pl-win${lens === 'docs' ? ' on' : ''}`} onClick={() => setLens('docs')}>unique docs</button>
          </span>
          <span className="pl-winbar">
            {WINDOWS.map(([k, label]) => (
              <button key={k} className={`pl-win${win === k ? ' on' : ''}`}
                      onClick={() => setWin(k)}>{label}</button>
            ))}
          </span>
          <span className="pl-when">{at ? `updated ${at.toLocaleTimeString()}` : 'loading…'}</span>
        </div>
      </div>

      {err && <div className="pl-err">could not read /pipeline_health — {err}</div>}

      <div className="pl-funnel">
        {['documents', 'chunked', 'embedded', 'published'].map((k, i) => (
          <div key={k} className="pl-fstep">
            <div className="pl-fnum">{fmt(t[k])}</div>
            <div className="pl-flabel">{k === 'published' ? 'available in chat' : k}</div>
            {i < 3 && <span className="pl-arrow">→</span>}
          </div>
        ))}
      </div>

      {/* LIVE CRAWL — what is being scraped right NOW.
          The Scrape card counts documents already in RAG, so a crawl in flight
          was invisible until it landed: during the AHCA run the bucket held 525
          objects while the card sat frozen at 6,813. This reads the scraper's
          own downloads block — the seat that owns the fact — so the in-flight
          lag is visible rather than inferred. */}
      {(h?.active_crawl as any)?.status ? (() => {
        const a = h!.active_crawl as any;
        const live = a.status === 'running';
        return (
          <div className={`pl-crawl${live ? ' pl-crawl-live' : ''}`}>
            <div className="pl-crawlhead">
              <span className={`pl-dot ${live ? 'pl-d-green' : 'pl-d-grey'}`} />
              <strong>{live ? 'Crawl running' : `Crawl ${a.status}`}</strong>
              <code className="pl-runid">{String(a.run_id || '').slice(0, 8)}</code>
              {a.conserved === false && <span className="pl-gap">not conserved</span>}
              {a.push_failed > 0 && <span className="pl-gap">{a.push_failed} push failures</span>}
            </div>
            <div className="pl-crawlflow">
              {[['gcs_objects', 'in GCS'], ['in_rag', 'in RAG'], ['awaiting_push', 'awaiting push'],
                ['pages_scraped', 'pages'], ['files_discovered', 'files found'],
                ['suppressed_cpt', 'CPT-suppressed'], ['downloaded', 'downloaded'],
                ['download_failed', 'download failed'], ['push_sent', 'pushed'],
                ['push_duplicate', 'already held']].map(([k, label]) => (
                <span key={k} className="pl-cstep">
                  <b className={((k === 'download_failed' || k === 'push_failed') && a[k] > 0)
                                 || (k === 'awaiting_push' && a[k] > 200) ? 'pl-gap' : ''}
                                 title={k === 'awaiting_push' ? (a.awaiting_push_basis || '') : undefined}>
                    {(a[k] ?? 0).toLocaleString()}
                  </b> {label}
                </span>
              ))}
            </div>
            {a.error ? <div className="pl-acctnote">scraper unreachable ({a.error}) — counts are last known</div> : null}
            {a.gcs_objects != null && a.pages_scraped === 0 ? (
              <div className="pl-acctnote">
                The scraper reports progress only at job completion, so its counters read 0 mid-run.
                <b> in GCS</b> is the live signal; <b>awaiting push</b> is what has been fetched but not yet ingested.
              </div>
            ) : null}
          </div>
        );
      })() : null}

      {/* THE ACCOUNTING: one cohort, followed down the chain.
          Every stage subtracts from the one above it, and `gap` is what left a
          stage and arrived nowhere. A non-zero gap is always a bug. */}
      {integ ? (
        <div className={`pl-acct${integ.worst_gap ? ' pl-acct-bad' : ''}`}>
          <div className="pl-accthead">
            <strong>Accounting</strong>
            <span className="pl-lenstag">unique documents</span>
            <span className="pl-sub">
              {integ.cohort.toLocaleString()} documents discovered · {WINDOWS.find(w => w[0] === win)?.[1] ?? win}
            </span>
            <span className={integ.worst_gap ? 'pl-gap' : 'pl-bal'}>
              {integ.worst_gap ? `${integ.total_gap.toLocaleString()} unaccounted` : 'every document accounted for ✓'}
            </span>
          </div>
          <table className="pl-acctable">
            <thead><tr><th>stage</th><th>in</th><th>reached</th><th>stopped</th><th>why it stopped</th><th>gap</th></tr></thead>
            <tbody>
              {integ.stages.map((st: any) => (
                <tr key={st.stage} className={st.gap ? 'pl-rowbad' : undefined}>
                  <td>{st.stage}</td>
                  <td className="pl-num">{(st.in ?? 0).toLocaleString()}</td>
                  <td className="pl-num"><b>{(st.reached ?? 0).toLocaleString()}</b></td>
                  <td className="pl-num">{st.stopped_total ? st.stopped_total.toLocaleString() : '—'}</td>
                  <td className="pl-why">
                    {st.stopped.filter((x: any) => x.count > 0).map((x: any) => x.reason).join('; ') || '—'}
                  </td>
                  <td className="pl-num">{st.gap ? <b className="pl-gap">{st.gap.toLocaleString()}</b> : '0'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {integ.classify_gate ? (
            <p className="pl-acctnote">
              <b>classify gate</b> (beside the chain, not in it):
              {' '}{integ.classify_gate.classified.toLocaleString()} of {integ.classify_gate.extracted.toLocaleString()} extracted are classified
              {integ.classify_gate.unclassified ? ` · ${integ.classify_gate.unclassified.toLocaleString()} never classified` : ''}
              {integ.classify_gate.held ? ` · ${integ.classify_gate.held.toLocaleString()} held for a human` : ''}
            </p>
          ) : null}
          <p className="pl-acctnote">
            <b>stopped</b> cannot progress and the reason is stated — deliberate, not a fault.
            <b> gap</b> left the stage above and arrived nowhere; it is always a bug, never a state.
          </p>
        </div>
      ) : null}

      <div className="pl-grid">
        {STAGES.map(s => {
          const st = (h?.[s.key] || {}) as Stage;
          const status = (st.status as string) || 'grey';
          return (
            <section key={s.key} className={`pl-card pl-${status} pl-click`}
                     role="button" tabIndex={0}
                     onClick={() => { setDrillWin('all'); setDrill({ stage: s.key, label: s.label }); }}
                     onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { setDrillWin('all'); setDrill({ stage: s.key, label: s.label }); } }}>
              <header>
                <span className={`pl-dot pl-d-${status}`} />
                <h3>{s.label}</h3>
                <span className="pl-open">open ›</span>
              </header>
              {s.fields.map(([f, rawLabel]) => {
                const [label, flag] = rawLabel.split('|');
                const value = f === '@throughput'
                  ? windowedThroughput(st, win, lens)
                  : st[f];
                const shown = f === '@throughput'
                  ? `${label} (${WINDOWS.find(w => w[0] === win)?.[1]}, ${lens === 'tx' ? 'runs' : 'docs'})`
                  : label;
                return <Metric key={f} label={shown} value={value} good={flag === 'good'} />;
              })}
              {s.key === 'versioning' && st.last_run_at ? (
                <div className="pl-note">last gate run {String(st.last_run_at).slice(0, 19)}</div>
              ) : null}
            </section>
          );
        })}
      </div>

      {drill && (
        <div className="pl-modal" onClick={() => setDrill(null)}>
          <div className="pl-sheet" onClick={e => e.stopPropagation()}>
            <div className="pl-sheethead">
              <div>
                <h3>{drill.label}</h3>
                <p className="pl-sub">
                  {items == null ? 'loading…'
                    : `${items.length} document${items.length === 1 ? '' : 's'} · ${WINDOWS.find(w => w[0] === drillWin)?.[1]}`}
                </p>
              </div>
              <div className="pl-controls">
                <span className="pl-winbar">
                  {WINDOWS.map(([k, label]) => (
                    <button key={k} className={`pl-win${drillWin === k ? ' on' : ''}`}
                            onClick={() => setDrillWin(k)}>{label}</button>
                  ))}
                </span>
                <button className="pl-btn" onClick={() => setDrill(null)}>close</button>
              </div>
            </div>
            {dErr && <div className="pl-err">could not load — {dErr}</div>}
            <div className="pl-scroll pl-sheetbody">
              <table className="pl-table">
                <thead><tr><th>age</th><th>document</th><th>state</th><th>entered</th></tr></thead>
                <tbody>
                  {(items || []).map(it => (
                    <tr key={it.document_id + String(it.at)}>
                      <td className={`pl-mono${(it.age_seconds ?? 0) > 3600 ? ' pl-stale' : ''}`}>
                        {age(it.age_seconds)}
                      </td>
                      <td className="pl-mono">{(it.filename || it.document_id).slice(0, 62)}</td>
                      <td className="pl-mono pl-dim">{(it.detail || '—').slice(0, 70)}</td>
                      <td className="pl-mono pl-dim">{(it.at || '—').slice(0, 19)}</td>
                    </tr>
                  ))}
                  {items && items.length === 0 && (
                    <tr><td colSpan={4} className="pl-dim" style={{ padding: '16px' }}>
                      Nothing in this stage for the selected window.
                    </td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      <h3 className="pl-h3">In-process jobs {inFlight.length ? `(${inFlight.length})` : ''}</h3>
      {inFlight.length === 0 ? (
        <div className="pl-idle">
          Nothing in flight. An idle pipeline and a stalled one look the same in a
          count — the stage cards above are where a stall shows.
        </div>
      ) : (
        <div className="pl-scroll">
          <table className="pl-table">
            <thead>
              <tr><th>stage</th><th>document</th><th>detail</th></tr>
            </thead>
            <tbody>
              {inFlight.map((r, i) => {
                const j = r.job as Record<string, unknown>;
                return (
                  <tr key={i}>
                    <td>{r.stage}</td>
                    <td className="pl-mono">
                      {String(j.filename || j.document_name || j.document_id || '—').slice(0, 60)}
                    </td>
                    <td className="pl-mono pl-dim">
                      {Object.entries(j)
                        .filter(([k]) => !['filename', 'document_name', 'document_id'].includes(k))
                        .map(([k, v]) => `${k}=${v}`)
                        .join('  ')
                        .slice(0, 90)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
