/**
 * Pure functions: flat list of discovered_sources rows → nested tree
 * grouped by URL path prefix. Used by SourceTreeView to render the
 * "what do we know about this entity" hierarchy.
 *
 * Why client-side: 226-1000 URLs per source comfortably fits in memory.
 * Tree assembly is O(n log n) and runs sub-millisecond. Adding a
 * server endpoint would be premature optimization.
 *
 * Tree node shape:
 *   - segment        path segment for this node ('providers', 'Billing-manual')
 *   - fullPath       cumulative path so far ('/providers/Billing-manual')
 *   - children       map of segment → child node
 *   - rows           the discovered_sources rows whose path ends at THIS node
 *                    (i.e., this is a leaf URL, not just a directory)
 *   - urlCount       total URLs under this subtree (including this leaf)
 *   - indexedCount   URLs with ingested=true under this subtree
 */

export interface SourceRow {
  id: string
  url: string
  host: string
  path: string
  payer: string | null
  state: string | null
  inferred_authority_level: string | null
  curated_authority_level: string | null
  effective_authority_level: string | null
  topic_tags: string[] | null
  content_kind: string
  extension: string | null
  last_seen_at: string | null
  last_fetch_status: number | null
  ingested: boolean
  ingested_doc_id: string | null
  curation_status: string
}

export interface TreeNode {
  segment: string
  fullPath: string
  children: Map<string, TreeNode>
  rows: SourceRow[]
  urlCount: number
  indexedCount: number
}


function newNode(segment: string, fullPath: string): TreeNode {
  return {
    segment,
    fullPath,
    children: new Map(),
    rows: [],
    urlCount: 0,
    indexedCount: 0,
  }
}


/**
 * Build a tree from a flat list of source rows. Rows are grouped by
 * the segments of their ``path`` field, splitting on '/'. The root
 * represents the host; children represent first-level path segments;
 * leaves are the actual URLs (rows live in ``node.rows``).
 *
 * Empty path segments (from leading/trailing slashes) are dropped.
 */
export function buildTree(rows: SourceRow[]): TreeNode {
  const root = newNode('', '')
  for (const row of rows) {
    const segments = (row.path || '/').split('/').filter(s => s.length > 0)
    let node = root
    let cumulative = ''
    for (let i = 0; i < segments.length; i++) {
      const seg = segments[i]
      cumulative += '/' + seg
      let child = node.children.get(seg)
      if (!child) {
        child = newNode(seg, cumulative)
        node.children.set(seg, child)
      }
      node = child
    }
    // Leaf: the row lives at this terminal node. Multiple rows can
    // share a leaf (e.g. /providers/ as a directory + /providers as a
    // page) — keep them all.
    node.rows.push(row)
  }
  // Walk the tree once to compute aggregate counts bottom-up.
  recomputeCounts(root)
  return root
}


/** Recursive bottom-up count fill. */
function recomputeCounts(node: TreeNode): void {
  let urlCount = node.rows.length
  let indexedCount = node.rows.filter(r => r.ingested).length
  for (const child of node.children.values()) {
    recomputeCounts(child)
    urlCount += child.urlCount
    indexedCount += child.indexedCount
  }
  node.urlCount = urlCount
  node.indexedCount = indexedCount
}


/** Sort children alphabetically for stable display. Returns array. */
export function sortedChildren(node: TreeNode): TreeNode[] {
  return Array.from(node.children.values()).sort((a, b) =>
    a.segment.localeCompare(b.segment)
  )
}


/** Coverage percentage 0-100; safe for 0-row entities. */
export function coverage(node: TreeNode): number {
  if (node.urlCount === 0) return 0
  return Math.round((node.indexedCount / node.urlCount) * 100)
}


/** Find sub-trees with the biggest gaps (most non-ingested URLs).
 *  Returns [node, gap] tuples sorted descending by gap. Used for
 *  the "biggest gaps" callout in the entity overview. */
export function topGaps(root: TreeNode, max: number = 5): { node: TreeNode; gap: number }[] {
  const gaps: { node: TreeNode; gap: number }[] = []
  function walk(node: TreeNode) {
    const gap = node.urlCount - node.indexedCount
    if (gap > 0 && node.fullPath !== '') {
      gaps.push({ node, gap })
    }
    for (const c of node.children.values()) walk(c)
  }
  walk(root)
  return gaps.sort((a, b) => b.gap - a.gap).slice(0, max)
}
