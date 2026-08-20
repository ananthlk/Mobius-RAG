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
const WINDOWS: [string, string][] = [['1h', 'last hour'], ['24h', '24h'], ['7d', '7d'], ['all', 'all time']];

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
      ['stored', 'objects stored'], ['missing_object', 'row without object'] ] },
  { key: 'extract', label: 'Extract', fields: [
      ['extracting', 'in flight'], ['no_text', 'produced no text'],
      ['failed_typed', 'typed failures'], ['tables_captured', 'tables captured'] ] },
  { key: 'classify', label: 'Classify', fields: [
      ['classified', 'classified'], ['held_for_human', 'held for human'],
      ['unclassified', 'not yet classified'] ] },
  { key: 'chunking', label: 'Chunking', fields: [
      ['active', 'active workers'], ['pending', 'pending'], ['last_hour', 'last hour'] ] },
  { key: 'embedding', label: 'Embedding', fields: [
      ['active', 'active workers'], ['pending', 'pending'], ['last_hour', 'last hour'] ] },
  { key: 'versioning', label: 'Versioning / dedup', fields: [
      ['pairs_scored', 'pairs scored'], ['duplicates', 'duplicates'],
      ['retired', 'retired'], ['shelved', 'shelved'] ] },
  { key: 'publishing', label: 'Publishing', fields: [
      ['last_hour', 'last hour'], ['embedded_unpublished', 'genuinely unpublished'],
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
function Metric({ label, value }: { label: string; value: unknown }) {
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
  return (
    <div className={`pl-metric${bump ? ' pl-bump' : ''}`}>
      <span className="pl-mv">{fmt(value)}</span>
      <span className="pl-ml">{label}</span>
    </div>
  );
}

export function PipelineTab() {
  const [h, setH] = useState<Health | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [at, setAt] = useState<Date | null>(null);
  const [live, setLive] = useState(true);
  const [win, setWin] = useState('1h');
  const [drill, setDrill] = useState<{ stage: string; label: string } | null>(null);
  const [items, setItems] = useState<DrillItem[] | null>(null);
  const [dErr, setDErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const r = await fetch('/pipeline_health');
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setH(await r.json());
      setErr(null);
      setAt(new Date());
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }, []);

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
        const r = await fetch(`/pipeline_health/stage/${drill.stage}?window=${win}&limit=200`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const j = await r.json();
        if (!dead) { setItems(j.items || []); setDErr(null); }
      } catch (e) {
        if (!dead) { setItems([]); setDErr(e instanceof Error ? e.message : String(e)); }
      }
    })();
    return () => { dead = true; };
  }, [drill, win]);

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

      <div className="pl-grid">
        {STAGES.map(s => {
          const st = (h?.[s.key] || {}) as Stage;
          const status = (st.status as string) || 'grey';
          return (
            <section key={s.key} className={`pl-card pl-${status} pl-click`}
                     role="button" tabIndex={0}
                     onClick={() => setDrill({ stage: s.key, label: s.label })}
                     onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') setDrill({ stage: s.key, label: s.label }); }}>
              <header>
                <span className={`pl-dot pl-d-${status}`} />
                <h3>{s.label}</h3>
                <span className="pl-open">open ›</span>
              </header>
              {s.fields.map(([f, label]) => (
                <Metric key={f} label={label} value={st[f]} />
              ))}
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
                    : `${items.length} document${items.length === 1 ? '' : 's'} · ${WINDOWS.find(w => w[0] === win)?.[1]}`}
                </p>
              </div>
              <div className="pl-controls">
                <span className="pl-winbar">
                  {WINDOWS.map(([k, label]) => (
                    <button key={k} className={`pl-win${win === k ? ' on' : ''}`}
                            onClick={() => setWin(k)}>{label}</button>
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
                      Nothing in this stage for the selected window. Widen to “all time” — a
                      stalled document usually entered long before the last hour.
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
