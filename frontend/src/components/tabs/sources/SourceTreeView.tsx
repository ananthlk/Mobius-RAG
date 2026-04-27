import { useState, useCallback } from 'react'
import { API_BASE } from '../../../config'
import {
  type SourceRow,
  type TreeNode,
  buildTree,
  coverage,
  sortedChildren,
  topGaps,
} from './treeBuilder'

interface Props {
  host: string
  payerLabel: string | null
  rows: SourceRow[]
  onIngest: (url: string) => Promise<void>
  onBulkIngest?: (urls: string[]) => Promise<void>
  ingestingUrls: Set<string>
}

/**
 * Hierarchical tree view of all known URLs for one entity (host).
 *
 * Top-level shows aggregate stats + biggest gaps. Each TreeNode is a
 * collapsible row showing path-segment, URL count, indexed count,
 * status pill. Leaf rows show ingest action when not yet indexed.
 *
 * Default expansion: root + first level (depth 0-1). Deeper levels
 * are collapsed to keep the page scannable. Operator clicks ▶ to
 * drill in.
 */
export function SourceTreeView({ host, payerLabel, rows, onIngest, onBulkIngest, ingestingUrls }: Props) {
  const root = buildTree(rows)
  const cov = coverage(root)
  const gaps = topGaps(root, 5)

  return (
    <div className="source-tree-view">
      {/* ── Entity header + coverage ───────────────────────────── */}
      <div className="entity-header">
        <h3 className="entity-name">
          {payerLabel || host}
          <span className="entity-host">({host})</span>
        </h3>
        <div className="entity-stats">
          <span className="stat-pill">{root.urlCount} URLs</span>
          <span className={`stat-pill stat-indexed`}>{root.indexedCount} ✓ indexed</span>
          <span className={`stat-pill stat-coverage cov-${bucket(cov)}`}>
            {cov}% coverage
          </span>
        </div>
        {/* Coverage bar */}
        <div className="coverage-bar" title={`${root.indexedCount} of ${root.urlCount} URLs indexed`}>
          <div
            className="coverage-fill"
            style={{ width: `${cov}%` }}
          />
        </div>
      </div>

      {/* ── Biggest gaps callout ───────────────────────────────── */}
      {gaps.length > 0 && root.indexedCount < root.urlCount && (
        <div className="gaps-callout">
          <strong>Biggest gaps:</strong>{' '}
          {gaps.map(({ node, gap }) => (
            <span key={node.fullPath} className="gap-chip">
              {node.fullPath} ({gap} URLs)
            </span>
          ))}
        </div>
      )}

      {/* ── Tree ─────────────────────────────────────────────────── */}
      <div className="tree-root">
        {sortedChildren(root).map(child => (
          <TreeRow
            key={child.fullPath}
            node={child}
            depth={0}
            onIngest={onIngest}
            onBulkIngest={onBulkIngest}
            ingestingUrls={ingestingUrls}
          />
        ))}
      </div>
    </div>
  )
}


/** Walk a sub-tree and collect every leaf URL where ingested=false
 *  AND status looks reachable (not 4xx). Used by the per-node
 *  "Ingest all N" bulk button. */
function collectIngestableUrls(node: TreeNode): string[] {
  const out: string[] = []
  function walk(n: TreeNode) {
    for (const r of n.rows) {
      const blocked = (r.last_fetch_status ?? 0) >= 400
      if (!r.ingested && !blocked) out.push(r.url)
    }
    for (const c of n.children.values()) walk(c)
  }
  walk(node)
  return out
}


/* ── Recursive tree row ────────────────────────────────────────────── */

interface TreeRowProps {
  node: TreeNode
  depth: number
  onIngest: (url: string) => Promise<void>
  onBulkIngest?: (urls: string[]) => Promise<void>
  ingestingUrls: Set<string>
}

function TreeRow({ node, depth, onIngest, onBulkIngest, ingestingUrls }: TreeRowProps) {
  // Default-expand depth 0; collapse below.
  const [expanded, setExpanded] = useState(depth === 0)
  const [bulkInFlight, setBulkInFlight] = useState(false)
  const children = sortedChildren(node)
  const hasChildren = children.length > 0
  const isLeaf = node.rows.length > 0
  const cov = coverage(node)
  const fullyIndexed = node.indexedCount === node.urlCount && node.urlCount > 0

  const toggle = useCallback(() => {
    if (hasChildren) setExpanded(e => !e)
  }, [hasChildren])

  // How many ingestable (non-indexed, non-blocked) URLs in this sub-tree?
  // Used to decide whether to show "Ingest all N" button.
  const ingestable = isLeaf || hasChildren ? collectIngestableUrls(node) : []
  const showBulk = ingestable.length >= 2 && onBulkIngest && !fullyIndexed

  const handleBulk = async () => {
    if (!onBulkIngest || bulkInFlight) return
    setBulkInFlight(true)
    try {
      await onBulkIngest(ingestable)
    } finally {
      setBulkInFlight(false)
    }
  }

  return (
    <div className="tree-row" style={{ paddingLeft: `${depth * 16}px` }}>
      <div className="tree-row-line">
        <button
          className="tree-toggle"
          onClick={toggle}
          disabled={!hasChildren}
          aria-label={expanded ? 'collapse' : 'expand'}
        >
          {hasChildren ? (expanded ? '▼' : '▶') : '·'}
        </button>

        <span className="tree-segment">/{node.segment}</span>

        {hasChildren && (
          <span className="tree-counts">
            {node.urlCount} URLs · {node.indexedCount} indexed
          </span>
        )}

        {fullyIndexed && hasChildren && (
          <span className="tree-status tree-status-done">✓ done</span>
        )}
        {!fullyIndexed && hasChildren && (
          <span className={`tree-status tree-status-cov cov-${bucket(cov)}`}>
            {cov}%
          </span>
        )}

        {showBulk && (
          <button
            className="bulk-ingest-btn"
            onClick={handleBulk}
            disabled={bulkInFlight}
            title={`Ingest the ${ingestable.length} non-indexed URLs in this sub-tree (sequential — chunking workers are single-instance)`}
          >
            {bulkInFlight ? 'Ingesting…' : `▶ Ingest all ${ingestable.length}`}
          </button>
        )}
      </div>

      {/* Leaf rows attached to this node (not just nested children) */}
      {isLeaf && node.rows.map(row => (
        <LeafRow
          key={row.id}
          row={row}
          depth={depth + 1}
          onIngest={onIngest}
          ingesting={ingestingUrls.has(row.url)}
        />
      ))}

      {/* Recurse into children */}
      {expanded && children.map(child => (
        <TreeRow
          key={child.fullPath}
          node={child}
          depth={depth + 1}
          onIngest={onIngest}
          onBulkIngest={onBulkIngest}
          ingestingUrls={ingestingUrls}
        />
      ))}
    </div>
  )
}


/* ── Leaf row (one actual URL) ─────────────────────────────────────── */

interface LeafRowProps {
  row: SourceRow
  depth: number
  onIngest: (url: string) => Promise<void>
  ingesting: boolean
}

function LeafRow({ row, depth, onIngest, ingesting }: LeafRowProps) {
  const filename = row.path.split('/').pop() || row.path
  const blocked = (row.last_fetch_status ?? 0) >= 400
  const stale = row.last_fetch_status === 404 || row.curation_status === 'stale'

  return (
    <div
      className={`leaf-row ${row.ingested ? 'leaf-indexed' : 'leaf-not-indexed'}`}
      style={{ paddingLeft: `${depth * 16}px` }}
    >
      <span className="leaf-marker">
        {row.ingested ? '✓' : blocked ? '⊘' : stale ? '✗' : '○'}
      </span>
      <a
        href={row.url}
        target="_blank"
        rel="noopener noreferrer"
        className="leaf-url"
        title={row.url}
      >
        {filename || '(root)'}
      </a>
      {row.effective_authority_level && (
        <span className="leaf-auth">{row.effective_authority_level}</span>
      )}
      {row.ingested && (
        <span className="leaf-status leaf-status-indexed">indexed</span>
      )}
      {!row.ingested && !blocked && !stale && (
        <button
          className="leaf-ingest-btn"
          onClick={() => onIngest(row.url)}
          disabled={ingesting}
        >
          {ingesting ? 'Ingesting…' : 'Ingest →'}
        </button>
      )}
      {blocked && (
        <span className="leaf-status leaf-status-blocked">
          blocked ({row.last_fetch_status})
        </span>
      )}
      {stale && !blocked && (
        <span className="leaf-status leaf-status-stale">stale (404)</span>
      )}
    </div>
  )
}


/** Discrete coverage bucket → CSS class for color */
function bucket(pct: number): 'low' | 'mid' | 'high' | 'full' {
  if (pct === 100) return 'full'
  if (pct >= 67) return 'high'
  if (pct >= 33) return 'mid'
  return 'low'
}
