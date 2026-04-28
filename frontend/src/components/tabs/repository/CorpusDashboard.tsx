import { useMemo } from 'react'

interface DocLike {
  id: string
  payer?: string | null
  embedding_status?: string | null
  chunking_status?: string | null
  published_at?: string | null
  source_metadata?: { source_url?: string | null } | null
  source_url?: string | null
}

interface HostStat {
  host: string
  count: number        // URLs in discovered_sources
  corpusDocs: number   // docs in documents table
  corpusPublished: number  // published to pgvector
}

interface Props {
  documents: DocLike[]
  hosts: HostStat[]
}

function cleanHostLabel(host: string): string {
  return host.replace(/^www\./, '')
}

/**
 * Global corpus health dashboard — sits above the entity selector.
 *
 * Shows three KPIs (in chat / ready to publish / stalled) plus a
 * per-payer bar breakdown so the operator can see coverage at a glance
 * without drilling into each entity card.
 */
export function CorpusDashboard({ documents, hosts }: Props) {
  // ── Global pipeline stage counts from the documents list ──────────
  const stats = useMemo(() => {
    let published = 0
    let embeddedWaiting = 0
    let chunkedWaiting = 0
    let failed = 0

    for (const d of documents) {
      if (d.published_at) {
        published++
        continue
      }
      if (d.embedding_status === 'failed') {
        failed++
        continue
      }
      if (d.embedding_status === 'completed') {
        embeddedWaiting++
        continue
      }
      if (d.chunking_status === 'completed') {
        chunkedWaiting++
      }
    }

    return {
      published,
      waiting: embeddedWaiting + chunkedWaiting,
      embeddedWaiting,
      chunkedWaiting,
      failed,
    }
  }, [documents])

  // ── Payer display name: first doc matching this host's payer field ──
  const payerByHost = useMemo(() => {
    const map: Record<string, string> = {}
    for (const d of documents) {
      const url = d.source_metadata?.source_url || d.source_url || ''
      try {
        const h = new URL(url).hostname
        if (!map[h] && d.payer) map[h] = d.payer
      } catch {
        // not a URL
      }
    }
    return map
  }, [documents])

  // ── Top hosts by total activity ────────────────────────────────────
  const topHosts = useMemo(
    () =>
      hosts
        .filter((h) => h.corpusDocs > 0 || h.count > 5)
        .slice(0, 10),
    [hosts],
  )

  const maxDocs = useMemo(
    () => Math.max(...topHosts.map((h) => h.corpusDocs), 1),
    [topHosts],
  )

  if (documents.length === 0 && hosts.length === 0) return null

  return (
    <div className="corpus-dashboard">
      {/* ── KPI strip ────────────────────────────────────────────── */}
      <div className="corpus-kpis">
        <div className="corpus-kpi corpus-kpi--published">
          <span className="corpus-kpi-value">{stats.published.toLocaleString()}</span>
          <span className="corpus-kpi-label">in chat</span>
        </div>

        <div className="corpus-kpi-sep" aria-hidden>·</div>

        <div className={`corpus-kpi ${stats.waiting > 0 ? 'corpus-kpi--waiting' : ''}`}>
          <span className="corpus-kpi-value">{stats.waiting.toLocaleString()}</span>
          <span
            className="corpus-kpi-label"
            title={`${stats.embeddedWaiting} embedded (needs publish) · ${stats.chunkedWaiting} chunked (needs embedding)`}
          >
            ready to publish
          </span>
        </div>

        {stats.failed > 0 && (
          <>
            <div className="corpus-kpi-sep" aria-hidden>·</div>
            <div className="corpus-kpi corpus-kpi--failed">
              <span className="corpus-kpi-value">{stats.failed.toLocaleString()}</span>
              <span className="corpus-kpi-label">stalled</span>
            </div>
          </>
        )}
      </div>

      {/* ── Per-payer breakdown ───────────────────────────────────── */}
      {topHosts.length > 0 && (
        <div className="corpus-payer-breakdown">
          {topHosts.map((h) => {
            const label = payerByHost[h.host] || cleanHostLabel(h.host)
            const totalPct = (h.corpusDocs / maxDocs) * 100
            const publishedPct =
              h.corpusDocs > 0 ? (h.corpusPublished / h.corpusDocs) * 100 : 0
            const staged = h.corpusDocs - h.corpusPublished

            return (
              <div key={h.host} className="corpus-payer-row">
                <span className="corpus-payer-name" title={h.host}>
                  {label}
                </span>

                <div className="corpus-payer-track" role="presentation">
                  <div
                    className="corpus-payer-fill corpus-payer-fill--total"
                    style={{ width: `${totalPct}%` }}
                  >
                    {publishedPct > 0 && (
                      <div
                        className="corpus-payer-fill corpus-payer-fill--published"
                        style={{ width: `${publishedPct}%` }}
                      />
                    )}
                  </div>
                </div>

                <span className="corpus-payer-counts">
                  {h.corpusPublished > 0 && (
                    <span className="corpus-payer-count--published">
                      {h.corpusPublished}
                    </span>
                  )}
                  {staged > 0 && (
                    <span className="corpus-payer-count--staged">
                      +{staged} staged
                    </span>
                  )}
                  {h.corpusDocs === 0 && h.count > 0 && (
                    <span className="corpus-payer-count--urls">
                      {h.count} URLs
                    </span>
                  )}
                </span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
