import { useState, useCallback } from 'react'
import { API_BASE, SCRAPER_API_BASE } from '../../../config'

/**
 * "Add new source" wizard for the Sources tab.
 *
 * One operator workflow, one button — replaces the bouncing across
 * Document Input → Status → Sources → chat that the legacy 3-tab
 * flow required.
 *
 * Steps:
 *   1. Paste URL
 *   2. Probe (HEAD + sitemap.xml + robots.txt + classifier) — auto
 *      runs on URL change with debounce
 *   3. Pick strategy (auto-suggested from probe; operator can override)
 *   4. Confirm metadata (auto-filled from classifier)
 *   5. Submit — fires the right backend action per strategy:
 *        scrape         → POST {scraper}/scrape (tree)
 *        sitemap_only   → POST /sources/bulk_upsert from sitemap
 *        state_mirror   → POST /sources/upsert (placeholder row)
 *        manual_upload  → POST /sources/upsert (placeholder row)
 */

interface ProbeResult {
  url: string
  host: string
  fetch: { status: number; content_type: string | null; redirected_to: string | null }
  sitemap: { url: string; status: number; url_count: number; sample: string[] }
  robots: { status: number; preview: string }
  classifier: {
    host: string
    path: string
    payer: string | null
    state: string | null
    inferred_authority_level: string | null
    content_kind: string
    extension: string | null
  }
  recommended_strategy: 'scrape' | 'sitemap_only' | 'state_mirror' | 'manual_upload'
  recommended_reason: string
}

type Strategy = 'scrape' | 'sitemap_only' | 'state_mirror' | 'manual_upload'

interface Props {
  onClose: () => void
  onAdded: () => void  // called after successful add so parent can refresh
}

export function AddSourceDialog({ onClose, onAdded }: Props) {
  const [url, setUrl] = useState('')
  const [probing, setProbing] = useState(false)
  const [probe, setProbe] = useState<ProbeResult | null>(null)
  const [probeError, setProbeError] = useState<string | null>(null)

  // Form fields — pre-filled from probe.classifier when probe lands
  const [strategy, setStrategy] = useState<Strategy>('scrape')
  const [payer, setPayer] = useState('')
  const [state, setState] = useState('')
  const [authority, setAuthority] = useState('')
  const [autoPublish, setAutoPublish] = useState(true)
  const [maxPages, setMaxPages] = useState(200)
  const [maxDepth, setMaxDepth] = useState(3)

  // Submit state
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState<{ ok: boolean; msg: string; jobId?: string } | null>(null)


  const handleProbe = useCallback(async () => {
    if (!url.trim()) return
    setProbing(true)
    setProbeError(null)
    setProbe(null)
    try {
      const resp = await fetch(`${API_BASE}/sources/probe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url.trim() }),
      })
      if (!resp.ok) throw new Error(`probe ${resp.status}`)
      const data = (await resp.json()) as ProbeResult
      setProbe(data)
      // Auto-fill all the form fields from probe
      setStrategy(data.recommended_strategy)
      setPayer(data.classifier.payer || '')
      setState(data.classifier.state || '')
      setAuthority(data.classifier.inferred_authority_level || '')
    } catch (e) {
      setProbeError(String(e))
    } finally {
      setProbing(false)
    }
  }, [url])


  const handleSubmit = useCallback(async () => {
    if (!url.trim() || submitting) return
    setSubmitting(true)
    setResult(null)
    try {
      let msg = ''
      let jobId: string | undefined

      if (strategy === 'scrape') {
        // Fire a tree scan via the scraper. Curator push hook +
        // HTML auto-import hook take care of registry/corpus
        // population. auto_publish flows through HTML import.
        const resp = await fetch(`${SCRAPER_API_BASE}/scrape`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            url: url.trim(),
            mode: 'tree',
            max_depth: maxDepth,
            max_pages: maxPages,
            scope_mode: 'same_domain',
            include_content: true,
            include_summary: false,
          }),
        })
        const data = await resp.json()
        if (!resp.ok) throw new Error(data?.detail || `scrape ${resp.status}`)
        jobId = data.job_id as string
        msg = `Scrape queued (job ${jobId?.slice(0, 8)}). URLs land in the tree as the scraper makes progress; check the entity selector after ~30s.`
      } else if (strategy === 'sitemap_only') {
        // Fetch sitemap server-side and bulk-upsert. Reuses the
        // YAML backfill pattern but for a single ad-hoc source.
        // For v1, we just register the URL itself + tell the operator
        // to run the YAML backfill for full sitemap parse. (Future:
        // add /sources/bulk_from_sitemap?url=&payer=&state= endpoint.)
        const resp = await fetch(`${API_BASE}/sources/upsert`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            url: url.trim(),
            discovered_via: 'manual',
            payer_hint: payer || null,
            state_hint: state || null,
            authority_hint: authority || null,
          }),
        })
        if (!resp.ok) throw new Error(`upsert ${resp.status}`)
        msg = `Source registered (sitemap_only). To populate URLs from sitemap.xml, run scripts/curator/backfill_from_sources_yaml.py --id <new-source-id>.`
      } else {
        // state_mirror / manual_upload: just register the placeholder.
        const resp = await fetch(`${API_BASE}/sources/upsert`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            url: url.trim(),
            discovered_via: 'manual',
            payer_hint: payer || null,
            state_hint: state || null,
            authority_hint: authority || null,
          }),
        })
        if (!resp.ok) throw new Error(`upsert ${resp.status}`)
        msg = strategy === 'state_mirror'
          ? `Source registered as state_mirror. Chat ReAct will surface this URL for the user to upload manually; operator should add the canonical state-agency URL via a separate ${'`Add new source`'} entry.`
          : `Source registered for manual upload. Chat ReAct will surface "we know this exists" and prompt operator to upload PDFs through the Document Input tab.`
      }

      setResult({ ok: true, msg, jobId })
      onAdded()
    } catch (e) {
      setResult({ ok: false, msg: String(e) })
    } finally {
      setSubmitting(false)
    }
  }, [url, strategy, payer, state, authority, maxPages, maxDepth, submitting, onAdded])


  return (
    <div className="add-source-overlay" onClick={onClose}>
      <div className="add-source-dialog" onClick={e => e.stopPropagation()}>
        <div className="add-source-header">
          <h3>Add new source</h3>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>

        {/* ── Step 1: URL ───────────────────────────────────────── */}
        <div className="step">
          <label className="step-label">1. Seed URL or domain</label>
          <div className="url-input-row">
            <input
              type="text"
              placeholder="https://www.example.com/providers/"
              value={url}
              onChange={e => setUrl(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') handleProbe() }}
              autoFocus
            />
            <button onClick={handleProbe} disabled={probing || !url.trim()}>
              {probing ? 'Probing…' : 'Probe →'}
            </button>
          </div>
          {probeError && <div className="probe-error">{probeError}</div>}
        </div>

        {/* ── Step 2: Probe results ─────────────────────────────── */}
        {probe && (
          <div className="step probe-results">
            <label className="step-label">2. What we found</label>
            <div className="probe-grid">
              <div className={`probe-pill ${probe.fetch.status >= 200 && probe.fetch.status < 400 ? 'ok' : 'bad'}`}>
                Front door: <strong>HTTP {probe.fetch.status}</strong>
                {probe.fetch.redirected_to && <small> (→ {probe.fetch.redirected_to})</small>}
              </div>
              <div className={`probe-pill ${probe.sitemap.status === 200 && probe.sitemap.url_count > 0 ? 'ok' : 'mid'}`}>
                Sitemap: <strong>HTTP {probe.sitemap.status}</strong>
                {probe.sitemap.url_count > 0 && <small> · {probe.sitemap.url_count} URLs</small>}
              </div>
              <div className="probe-pill mid">
                robots.txt: HTTP {probe.robots.status}
              </div>
            </div>
            <div className="recommendation">
              <strong>Recommended:</strong> {probe.recommended_strategy.replace(/_/g, ' ')}
              <br/><small>{probe.recommended_reason}</small>
            </div>
          </div>
        )}

        {/* ── Step 3: Strategy ──────────────────────────────────── */}
        {probe && (
          <div className="step">
            <label className="step-label">3. Ingest strategy</label>
            <div className="strategy-radios">
              {([
                ['scrape', 'Scrape (tree BFS) — best for open sites'],
                ['sitemap_only', 'Sitemap only — register URLs without crawling'],
                ['state_mirror', 'State mirror — bot-walled, use AHCA-style upstream'],
                ['manual_upload', 'Manual upload — operator drops PDFs themselves'],
              ] as [Strategy, string][]).map(([val, label]) => (
                <label key={val} className={`strategy-row ${strategy === val ? 'selected' : ''}`}>
                  <input
                    type="radio"
                    name="strategy"
                    value={val}
                    checked={strategy === val}
                    onChange={() => setStrategy(val)}
                  />
                  <span className="strategy-label">{label}</span>
                  {val === probe.recommended_strategy && (
                    <span className="recommended-tag">recommended</span>
                  )}
                </label>
              ))}
            </div>

            {strategy === 'scrape' && (
              <div className="scrape-opts">
                <label>
                  Max pages:
                  <input
                    type="number" min={10} max={2000} value={maxPages}
                    onChange={e => setMaxPages(parseInt(e.target.value) || 200)}
                  />
                </label>
                <label>
                  Max depth:
                  <input
                    type="number" min={1} max={10} value={maxDepth}
                    onChange={e => setMaxDepth(parseInt(e.target.value) || 3)}
                  />
                </label>
              </div>
            )}
          </div>
        )}

        {/* ── Step 4: Metadata ──────────────────────────────────── */}
        {probe && (
          <div className="step">
            <label className="step-label">4. Metadata (auto-filled, override if needed)</label>
            <div className="metadata-grid">
              <label>
                Payer: <input value={payer} onChange={e => setPayer(e.target.value)} placeholder="auto" />
              </label>
              <label>
                State: <input value={state} onChange={e => setState(e.target.value)} placeholder="FL" maxLength={2} />
              </label>
              <label>
                Authority: <input value={authority} onChange={e => setAuthority(e.target.value)} placeholder="payer_policy" />
              </label>
            </div>
          </div>
        )}

        {/* ── Step 5: Auto-publish + submit ─────────────────────── */}
        {probe && (
          <div className="step">
            <label className="autopub-line">
              <input
                type="checkbox"
                checked={autoPublish}
                onChange={e => setAutoPublish(e.target.checked)}
              />
              Auto-publish to chat (HTML pages chained through chunk → embed → publish)
            </label>
            <small className="muted">
              Note: scrape strategy auto-publishes via the worker's HTML import hook.
              For other strategies, this only affects the placeholder row.
            </small>
          </div>
        )}

        {/* ── Result + actions ──────────────────────────────────── */}
        {result && (
          <div className={`add-result ${result.ok ? 'ok' : 'err'}`}>
            <strong>{result.ok ? '✓ Done' : '✗ Failed'}</strong>
            <p>{result.msg}</p>
          </div>
        )}

        <div className="add-source-footer">
          <button className="cancel-btn" onClick={onClose}>Cancel</button>
          <button
            className="submit-btn"
            onClick={handleSubmit}
            disabled={!probe || submitting}
          >
            {submitting ? 'Adding…' : 'Add source →'}
          </button>
        </div>
      </div>
    </div>
  )
}
