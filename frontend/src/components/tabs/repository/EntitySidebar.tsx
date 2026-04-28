import { useMemo } from 'react'

// ── Types ─────────────────────────────────────────────────────────────────────

export type Domain = 'payer' | 'government' | 'association' | 'clinical' | 'other'

export interface HostStatEnriched {
  host: string
  count: number          // URLs in discovered_sources
  corpusDocs: number     // docs in documents table
  corpusPublished: number
  payer: string | null
  domain: Domain
}

export interface CorpusStats {
  published: number
  waiting: number
  failed: number
}

export type DomainFilter = 'all' | Domain

interface Props {
  hosts: HostStatEnriched[]
  selectedHost: string
  onSelectHost: (host: string) => void
  collapsed: boolean
  onToggle: () => void
  entitySearch: string
  onEntitySearch: (q: string) => void
  domainFilter: DomainFilter
  onDomainFilter: (f: DomainFilter) => void
  stats: CorpusStats
}

// ── Domain helpers ─────────────────────────────────────────────────────────────

export function domainOf(host: string, payer: string | null | undefined): Domain {
  const h = host.toLowerCase()
  if (
    h.endsWith('.gov') ||
    h.includes('myflorida') ||
    h.includes('govinfo') ||
    h.includes('congress.') ||
    h.includes('cms.') ||
    h.includes('samhsa') ||
    h.includes('hhs.') ||
    h.includes('nih.') ||
    h.includes('cdc.')
  )
    return 'government'

  const payerDomains = [
    'sunshinehealth', 'humana', 'wellcare', 'molina',
    'aetna', 'uhc', 'simplyhealthcare', 'staywell', 'careplus',
  ]
  if (payerDomains.some((p) => h.includes(p))) return 'payer'
  if (
    payer &&
    ['sunshine', 'humana', 'wellcare', 'molina', 'aetna', 'uhc'].some((p) =>
      payer.toLowerCase().includes(p),
    )
  )
    return 'payer'

  const assocDomains = ['fbha', 'thenationalcouncil', 'fadaa', 'nasw', 'bacb']
  if (assocDomains.some((p) => h.includes(p))) return 'association'

  const clinicalDomains = ['apa.org', 'nctsn', 'nhlbi', 'goldcopd', 'accme', 'cpeip', 'netdna-ssl']
  if (clinicalDomains.some((p) => h.includes(p))) return 'clinical'

  if (h.includes('aspire')) return 'payer' // Aspire Health Partners = FL MCO

  return 'other'
}

const DOMAIN_META: Record<
  DomainFilter,
  { label: string; color: string; bg: string }
> = {
  all:         { label: 'All',      color: '#64748b', bg: '#f1f5f9' },
  payer:       { label: 'Payer',    color: '#2563eb', bg: '#eff6ff' },
  government:  { label: 'Gov',      color: '#7c3aed', bg: '#f5f3ff' },
  association: { label: 'Assoc',    color: '#b45309', bg: '#fffbeb' },
  clinical:    { label: 'Clinical', color: '#0d9488', bg: '#f0fdfa' },
  other:       { label: 'Other',    color: '#64748b', bg: '#f8fafc' },
}

function DomainDot({ domain, size = 8 }: { domain: Domain; size?: number }) {
  const { color } = DOMAIN_META[domain] ?? DOMAIN_META.other
  return (
    <span
      style={{
        display: 'inline-block',
        width: size,
        height: size,
        borderRadius: '50%',
        background: color,
        flexShrink: 0,
      }}
    />
  )
}

// ── Component ─────────────────────────────────────────────────────────────────

const DOMAIN_FILTERS: DomainFilter[] = ['all', 'payer', 'government', 'association', 'clinical']

export function EntitySidebar({
  hosts,
  selectedHost,
  onSelectHost,
  collapsed,
  onToggle,
  entitySearch,
  onEntitySearch,
  domainFilter,
  onDomainFilter,
  stats,
}: Props) {
  const filtered = useMemo(() => {
    let list = hosts
    if (domainFilter !== 'all') list = list.filter((h) => h.domain === domainFilter)
    if (entitySearch.trim()) {
      const q = entitySearch.trim().toLowerCase()
      list = list.filter(
        (h) =>
          h.host.toLowerCase().includes(q) ||
          (h.payer ?? '').toLowerCase().includes(q),
      )
    }
    return list
  }, [hosts, domainFilter, entitySearch])

  if (collapsed) {
    return (
      <aside className="repo-sidebar repo-sidebar--collapsed">
        <button
          className="repo-sidebar-toggle"
          onClick={onToggle}
          title="Expand sidebar"
          aria-label="Expand sidebar"
        >
          ›
        </button>
        <div className="repo-sidebar-dots">
          {filtered.slice(0, 12).map((h) => (
            <button
              key={h.host}
              className={`repo-sidebar-dot-btn ${selectedHost === h.host ? 'active' : ''}`}
              onClick={() => onSelectHost(h.host)}
              title={h.payer ?? h.host}
            >
              <DomainDot domain={h.domain} size={10} />
            </button>
          ))}
        </div>
      </aside>
    )
  }

  return (
    <aside className="repo-sidebar">
      {/* ── Header + toggle ────────────────────────────────────────── */}
      <div className="repo-sidebar-header">
        <span className="repo-sidebar-title">Sources</span>
        <button
          className="repo-sidebar-toggle"
          onClick={onToggle}
          title="Collapse sidebar"
          aria-label="Collapse sidebar"
        >
          ‹
        </button>
      </div>

      {/* ── Corpus mini-stats ────────────────────────────────────────── */}
      <div className="repo-sidebar-stats">
        <span className="repo-sidebar-stat repo-sidebar-stat--published">
          {stats.published.toLocaleString()} in chat
        </span>
        <span className="repo-sidebar-stat-sep">·</span>
        {stats.waiting > 0 && (
          <>
            <span className="repo-sidebar-stat repo-sidebar-stat--waiting">
              {stats.waiting.toLocaleString()} ready
            </span>
            <span className="repo-sidebar-stat-sep">·</span>
          </>
        )}
        {stats.failed > 0 && (
          <span className="repo-sidebar-stat repo-sidebar-stat--failed">
            {stats.failed.toLocaleString()} stalled
          </span>
        )}
      </div>

      {/* ── Search ──────────────────────────────────────────────────── */}
      <div className="repo-sidebar-search-wrap">
        <span className="repo-sidebar-search-icon" aria-hidden>⌕</span>
        <input
          className="repo-sidebar-search"
          type="search"
          placeholder="Filter sources…"
          value={entitySearch}
          onChange={(e) => onEntitySearch(e.target.value)}
          aria-label="Filter sources"
        />
      </div>

      {/* ── Domain filter pills ──────────────────────────────────────── */}
      <div className="repo-sidebar-domains" role="group" aria-label="Domain filter">
        {DOMAIN_FILTERS.map((d) => {
          const { label, color, bg } = DOMAIN_META[d]
          const active = domainFilter === d
          return (
            <button
              key={d}
              className={`repo-domain-pill ${active ? 'active' : ''}`}
              style={
                active
                  ? { background: bg, color, borderColor: color }
                  : undefined
              }
              onClick={() => onDomainFilter(d)}
            >
              {d !== 'all' && <DomainDot domain={d as Domain} size={6} />}
              {label}
            </button>
          )
        })}
      </div>

      {/* ── Entity list ─────────────────────────────────────────────── */}
      <div className="repo-sidebar-list" role="listbox" aria-label="Source entities">
        {filtered.length === 0 && (
          <div className="repo-sidebar-empty">No sources match</div>
        )}
        {filtered.map((h) => {
          const label = h.payer ?? h.host.replace(/^www\./, '')
          const sub = h.payer ? h.host.replace(/^www\./, '') : null
          const publishedPct =
            h.corpusDocs > 0 ? (h.corpusPublished / h.corpusDocs) * 100 : 0
          const staged = h.corpusDocs - h.corpusPublished
          const isSelected = selectedHost === h.host
          return (
            <button
              key={h.host}
              role="option"
              aria-selected={isSelected}
              className={`repo-sidebar-entity ${isSelected ? 'selected' : ''}`}
              onClick={() => onSelectHost(h.host)}
            >
              <div className="repo-entity-row-top">
                <DomainDot domain={h.domain} />
                <span className="repo-entity-name">{label}</span>
                <span className="repo-entity-count">
                  {h.corpusPublished > 0 ? (
                    <span className="repo-entity-count--published">{h.corpusPublished}</span>
                  ) : null}
                  {staged > 0 && (
                    <span className="repo-entity-count--staged">+{staged}</span>
                  )}
                </span>
              </div>
              {sub && <div className="repo-entity-host">{sub}</div>}
              {h.corpusDocs > 0 && (
                <div className="repo-entity-bar-track">
                  <div
                    className="repo-entity-bar-fill"
                    style={{ width: `${publishedPct}%` }}
                  />
                </div>
              )}
            </button>
          )
        })}
      </div>
    </aside>
  )
}
