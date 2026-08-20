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

const REFRESH_MS = 10_000;

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
            <section key={s.key} className={`pl-card pl-${status}`}>
              <header>
                <span className={`pl-dot pl-d-${status}`} />
                <h3>{s.label}</h3>
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
