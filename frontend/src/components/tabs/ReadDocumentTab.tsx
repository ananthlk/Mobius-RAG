import { useState, useEffect, useCallback, useRef, type ReactNode } from 'react'
import { approveFactApi, deleteFactApi, patchFactApi, rejectFactApi } from '../../lib/factActions'
import './ReadDocumentTab.css'

const API_BASE = 'http://localhost:8000'

// Category keys and labels (match backend CATEGORY_NAMES)
const CATEGORIES: { key: string; label: string }[] = [
  { key: 'contacting_marketing_members', label: 'Contacting / marketing members' },
  { key: 'member_eligibility_molina', label: 'Member eligibility (Molina)' },
  { key: 'benefit_access_limitations', label: 'Benefit access / limitations' },
  { key: 'prior_authorization_required', label: 'Prior authorization required' },
  { key: 'claims_authorization_submissions', label: 'Claims / authorization / submissions' },
  { key: 'compliant_claim_requirements', label: 'Compliant claim requirements' },
  { key: 'claim_disputes', label: 'Claim disputes' },
  { key: 'credentialing', label: 'Credentialing' },
  { key: 'claim_submission_important', label: 'Claim submission (important)' },
  { key: 'coordination_of_benefits', label: 'Coordination of benefits' },
  { key: 'other_important', label: 'Other important' },
]

interface Page {
  page_number: number
  text: string | null
  text_markdown?: string | null
  extraction_status: string
}

/** Detect if a block looks like a list (most lines start with bullet or number). */
const LIST_BULLET = /^[\s]*[•\-*]\s+/
const LIST_NUMBER = /^[\s]*\d+[.)]\s+/
function isListBlock(block: string): boolean {
  const lines = block.split('\n').filter(l => l.trim())
  if (lines.length < 2) return false
  const listLike = lines.filter(l => LIST_BULLET.test(l) || LIST_NUMBER.test(l)).length
  return listLike >= Math.min(2, lines.length) || listLike >= lines.length * 0.6
}

/** Render one block: list as <ul><li> or paragraph(s) as <p>. */
function renderBlock(block: string, keyPrefix: string): ReactNode {
  const trimmed = block.trim()
  if (!trimmed) return null
  if (isListBlock(trimmed)) {
    const lines = trimmed.split('\n').filter(l => l.trim())
    return (
      <ul key={keyPrefix} className="reader-list">
        {lines.map((line, i) => {
          const bulletMatch = line.match(LIST_BULLET) || line.match(LIST_NUMBER)
          const content = bulletMatch ? line.slice(bulletMatch[0].length).trim() : line.trim()
          return <li key={`${keyPrefix}-${i}`}>{content}</li>
        })}
      </ul>
    )
  }
  return (
    <p key={keyPrefix} className="reader-raw-p">
      {trimmed.replace(/\n/g, ' ')}
    </p>
  )
}

/** Simple markdown-style render: ## as h2, lists as ul/li, paragraphs preserved. No extra deps. */
function SimpleMarkdown({ content }: { content: string }) {
  if (!content.trim()) return null
  const parts: ReactNode[] = []
  const sections = content.split(/\n(?=## )/).filter(Boolean)
  for (const section of sections) {
    if (section.startsWith('## ')) {
      const firstNewline = section.indexOf('\n')
      const title = firstNewline >= 0 ? section.slice(3, firstNewline).trim() : section.slice(3).trim()
      const body = firstNewline >= 0 ? section.slice(firstNewline + 1).trim() : ''
      parts.push(<h2 key={parts.length} className="reader-h2">{title}</h2>)
      if (body) {
        parts.push(
          <div key={parts.length} className="reader-section-body">
            {body.split(/\n\n+/).map((p, i) => renderBlock(p, `body-${parts.length}-${i}`))}
          </div>
        )
      }
    } else {
      section.split(/\n\n+/).forEach((p, i) => {
        const node = renderBlock(p, `sec-${parts.length}-${i}`)
        if (node) parts.push(node)
      })
    }
  }
  return <div className="reader-markdown">{parts}</div>
}

interface Document {
  id: string
  filename: string
  display_name?: string | null
}

interface ReadDocumentTabProps {
  documents: Document[]
  selectedDocumentId?: string | null
  navigateToRead?: { documentId: string; pageNumber?: number; factId?: string } | null
  onNavigateToReadConsumed?: () => void
  onDocumentSelect?: (documentId: string) => void
}

type HighlightSource = 'user' | 'llm'

export interface HighlightRange {
  start: number
  end: number
  factId?: string
  factText?: string
  categoryScores?: Record<string, { score: number | null; direction: number | null }>
  isPertinent?: boolean
  verificationStatus?: string | null
}

/** Short labels for compact tooltip display */
const CATEGORY_SHORT: Record<string, string> = {
  contacting_marketing_members: 'Contacting members',
  member_eligibility_molina: 'Eligibility',
  benefit_access_limitations: 'Benefits / access',
  prior_authorization_required: 'Prior auth',
  claims_authorization_submissions: 'Claims / auth',
  compliant_claim_requirements: 'Compliant claims',
  claim_disputes: 'Disputes',
  credentialing: 'Credentialing',
  claim_submission_important: 'Claim submission',
  coordination_of_benefits: 'COB',
  other_important: 'Other',
}

export type TooltipData = {
  pertinent: boolean
  topCategories: Array<{ key: string; label: string; score: number; direction: number | null }>
}

function getTooltipData(r: HighlightRange): TooltipData | null {
  const pertinent = r.isPertinent === true
  let topCategories: TooltipData['topCategories'] = []
  if (r.categoryScores && typeof r.categoryScores === 'object') {
    topCategories = Object.entries(r.categoryScores)
      .filter(([, v]) => v && typeof v.score === 'number' && v.score > 0)
      .sort(([, a], [, b]) => (b?.score ?? 0) - (a?.score ?? 0))
      .slice(0, 2)
      .map(([k, v]) => ({
        key: k,
        label: CATEGORY_SHORT[k] ?? k.replace(/_/g, ' '),
        score: v?.score ?? 0,
        direction: v?.direction ?? null,
      }))
  }
  return { pertinent, topCategories }
}

function dirSymbol(d: number | null): string {
  if (d === 1) return '↑'
  if (d === 0) return '↓'
  return '→'
}

/** Modern fact tooltip: pertinent status + top 2 categories. Renders above the anchor. */
function FactTooltip({
  data,
  anchorRect,
}: {
  data: TooltipData
  anchorRect: DOMRect
}) {
  return (
    <div
      className="reader-fact-tooltip"
      style={{
        left: anchorRect.left + anchorRect.width / 2,
        top: anchorRect.top - 8,
      }}
    >
      <div className="reader-fact-tooltip-arrow" />
      <div className="reader-fact-tooltip-inner">
        <div className={`reader-fact-tooltip-pertinent ${data.pertinent ? 'yes' : 'no'}`}>
          {data.pertinent ? '✓' : '○'} {data.pertinent ? 'Pertinent' : 'Not pertinent'} to claims & members
        </div>
        {data.topCategories.length > 0 && (
          <div className="reader-fact-tooltip-categories">
            {data.topCategories.map(({ key, label, score, direction }) => (
              <div key={key} className="reader-fact-tooltip-cat-row">
                <span className="reader-fact-tooltip-cat-label">{label}</span>
                <span className="reader-fact-tooltip-cat-meta">
                  {(score * 100).toFixed(0)}% {dirSymbol(direction)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

/** Render one segment (paragraph) of text with highlighted ranges. segStart/segEnd are offsets in full text; ranges are clipped to this segment and rendered with segment-relative offsets. */
function SegmentWithHighlights({
  segmentText,
  segStart,
  segEnd,
  ranges,
  normalizedLen: _normalizedLen,
  onHighlightHover,
  onHighlightLeave,
}: {
  segmentText: string
  segStart: number
  segEnd: number
  ranges: Array<{ start: number; end: number; source: HighlightSource; factId?: string; factText?: string; categoryScores?: Record<string, { score: number | null; direction: number | null }>; isPertinent?: boolean; verificationStatus?: string | null }>
  normalizedLen: number
  onHighlightHover?: (r: HighlightRange, rect: DOMRect) => void
  onHighlightLeave?: () => void
}) {
  const nodes: ReactNode[] = []
  let pos = 0
  const segLen = segmentText.length
  for (const r of ranges) {
    const rStart = Math.max(segStart, r.start)
    const rEnd = Math.min(segEnd, r.end)
    if (rStart >= rEnd) continue
    const relStart = rStart - segStart
    const relEnd = rEnd - segStart
    if (relStart > pos) {
      nodes.push(<span key={`${pos}-pre`}>{segmentText.slice(pos, relStart)}</span>)
    }
    const approved = r.verificationStatus === 'approved'
    let baseClass = r.source === 'user' ? 'reader-fact-highlight' : 'reader-llm-fact-highlight'
    if (r.source === 'llm' && r.isPertinent === false) baseClass += ' reader-llm-fact-highlight-non-pertinent'
    const className = approved ? `${baseClass} reader-fact-highlight-approved` : baseClass
    const fallbackTitle = [r.isPertinent ? '✓ Pertinent' : '○ Not pertinent'].concat(
      r.categoryScores && Object.keys(r.categoryScores).length
        ? ['Hover for details']
        : []
    ).join(' · ')
    nodes.push(
      <span
        key={r.factId ? `${r.factId}-hl` : `${relStart}-${r.source}-hl`}
        className={className}
        title={fallbackTitle}
        data-fact-id={r.factId || undefined}
        data-source={r.source}
        onMouseEnter={(e) => {
          onHighlightHover?.(r, e.currentTarget.getBoundingClientRect())
        }}
        onMouseLeave={() => onHighlightLeave?.()}
      >
        {segmentText.slice(relStart, relEnd)}
      </span>
    )
    pos = relEnd
  }
  if (pos < segLen) {
    nodes.push(<span key={`${pos}-post`}>{segmentText.slice(pos)}</span>)
  }
  return <>{nodes}</>
}

/** Get the first section header label from page text (for sidebar/main title). Returns null if no header found. */
function getFirstSectionHeader(text: string | null | undefined): string | null {
  if (!text?.trim()) return null
  const normalized = text
    .split('\n')
    .map(line => line.replace(/\s+/g, ' ').trim())
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
  const blocks = normalized.split(/\n\n+/)
  for (const block of blocks) {
    const firstLine = block.split('\n')[0]?.trim() ?? ''
    if (isSectionHeader(firstLine)) {
      if (firstLine.startsWith('## ')) return firstLine.slice(3).trim()
      if (firstLine.endsWith(':')) return firstLine.slice(0, -1).trim()
      return firstLine
    }
  }
  return null
}

/** Detect section header: colon/numbered (backend-style) or short title-like line (e.g. "Contact information", "Provider services"). */
function isSectionHeader(firstLine: string): boolean {
  if (!firstLine || firstLine.length > 120) return false
  const line = firstLine.trim()
  if (line.startsWith('## ')) return true
  if (line.endsWith(':')) return true
  if (/^[A-Z][A-Z\s]+:/.test(line)) return true
  if (/^\d+\.\s+[A-Z]/.test(line)) return true
  if (/^\d+\.\d+\s+/.test(line)) return true
  // Short title-like line (no period): e.g. "Contact information", "Provider services"
  if (line.length <= 60 && !line.endsWith('.') && /^[A-Z]/.test(line)) {
    const wordCount = line.split(/\s+/).length
    if (wordCount >= 1 && wordCount <= 8) return true
  }
  return false
}

type Segment = { start: number; end: number; text: string; isHeader: boolean; headerLabel?: string }

/** Build segments from text: section header (h2) or ## Title + body or plain paragraph. For markdown, first line "## X" becomes headerLabel "X". Segment text is the exact slice so highlights align. */
function buildSegments(normalized: string): Segment[] {
  const segments: Segment[] = []
  const blocks = normalized.split(/\n\n+/)
  let offset = 0
  for (let i = 0; i < blocks.length; i++) {
    const block = blocks[i]
    const blockStart = offset
    offset += block.length + (i < blocks.length - 1 ? 2 : 0)
    const lines = block.split('\n')
    const firstLine = lines[0]?.trim() ?? ''
    if (lines.length > 0 && isSectionHeader(firstLine)) {
      const headerRaw = lines[0] ?? ''
      const headerLen = headerRaw.length
      const headerLabel = firstLine.startsWith('## ')
        ? firstLine.slice(3).trim()
        : firstLine.endsWith(':')
          ? firstLine.slice(0, -1)
          : firstLine
      segments.push({
        start: blockStart,
        end: blockStart + headerLen,
        text: normalized.slice(blockStart, blockStart + headerLen),
        isHeader: true,
        headerLabel,
      })
      if (lines.length > 1) {
        const bodyStart = blockStart + headerLen + 1
        segments.push({ start: bodyStart, end: offset, text: normalized.slice(bodyStart, offset), isHeader: false })
      }
    } else {
      segments.push({ start: blockStart, end: offset, text: normalized.slice(blockStart, offset), isHeader: false })
    }
  }
  return segments
}

/** Normalize whitespace for comparison. */
function normWs(s: string): string {
  return s.replace(/\s+/g, ' ').trim()
}

/** Try to fix fragment highlights like "pro lorida." -> use "Florida." when the slice looks like junk + word fragment. */
function tryFixFragmentHighlight(text: string, slice: string): { start: number; end: number } | null {
  const trimmed = slice.trim()
  if (!trimmed) return null
  const parts = trimmed.split(/\s+/)
  if (parts.length < 2) return null
  const first = parts[0]
  const rest = parts.slice(1).join(' ')
  if (first.length > 4 || !rest) return null
  const capitalized = rest.charAt(0).toUpperCase() + rest.slice(1)
  let idx = text.indexOf(capitalized)
  if (idx >= 0) return { start: idx, end: idx + capitalized.length }
  idx = text.indexOf(rest)
  if (idx > 0) {
    const charBefore = text[idx - 1]
    if (charBefore === 'F' || charBefore === 'f') {
      const start = idx - 1
      const end = idx + rest.length
      return { start, end }
    }
    return { start: idx, end: idx + rest.length }
  }
  return null
}

/** If the range has factText and the text slice at (start,end) doesn't match it, find factText in text and return corrected start/end so highlight matches fact text. */
function correctRangeToMatchFactText(
  text: string,
  r: HighlightRange & { source: HighlightSource }
): { start: number; end: number } {
  const start = Math.max(0, Number(r.start))
  const end = Math.min(text.length, Math.max(start, Number(r.end)))
  const slice = text.slice(start, end)
  if (r.factText && r.factText.trim()) {
    if (normWs(slice) === normWs(r.factText)) {
      const fixed = tryFixFragmentHighlight(text, slice)
      if (fixed) return fixed
      return { start, end }
    }
    let idx = text.indexOf(r.factText)
    if (idx >= 0) return { start: idx, end: idx + r.factText.length }
    try {
      const escaped = r.factText.trim().replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
      const flexiblePattern = escaped.replace(/\s+/g, '\\s+')
      const match = text.match(new RegExp(flexiblePattern))
      if (match && match.index !== undefined) return { start: match.index, end: match.index + match[0].length }
    } catch {
      // regex invalid
    }
    const fixed = tryFixFragmentHighlight(text, slice)
    if (fixed) return fixed
  }
  return { start, end }
}

/** Render page text (markdown or raw) with highlighted ranges. When isMarkdown=true, text is used as-is (offsets are in markdown). userRanges use .reader-fact-highlight; llmRanges use .reader-llm-fact-highlight. */
function PageTextWithHighlights({
  text,
  userRanges,
  llmRanges,
  isMarkdown = false,
  onHighlightHover,
  onHighlightLeave,
}: {
  text: string
  userRanges: HighlightRange[]
  llmRanges: HighlightRange[]
  isMarkdown?: boolean
  onHighlightHover?: (r: HighlightRange, rect: DOMRect) => void
  onHighlightLeave?: () => void
}) {
  const withSource: Array<HighlightRange & { source: HighlightSource }> = [
    ...userRanges.map(r => ({ ...r, source: 'user' as HighlightSource })),
    ...llmRanges.map(r => ({ ...r, source: 'llm' as HighlightSource })),
  ]
  const normalized = isMarkdown
    ? text
    : text
        .split('\n')
        .map(line => line.replace(/\s+/g, ' ').trim())
        .join('\n')
        .replace(/\n{3,}/g, '\n\n')
        .trim()
  const len = normalized.length
  const segments = buildSegments(normalized)
  // Correct ranges so highlight span matches factText when stored offsets are wrong (formatting/LLM drift)
  const correctedRanges = withSource.map(r => {
    const { start, end } = correctRangeToMatchFactText(normalized, r)
    return { ...r, start, end }
  })

  if (!correctedRanges.length) {
    return (
      <div className="reader-markdown">
        {segments.map((seg, i) =>
          seg.isHeader ? (
            <h2 key={`h-${i}`} className="reader-h2">{seg.headerLabel ?? seg.text}</h2>
          ) : (
            <div key={`p-${i}`} className="reader-section-body">
              {seg.text.split(/\n\n+/).map((p, j) => renderBlock(p, `seg-${i}-${j}`))}
            </div>
          )
        )}
      </div>
    )
  }
  const sorted = [...correctedRanges].sort((a, b) => a.start - b.start)
  const clamped: Array<HighlightRange & { source: HighlightSource }> = []
  for (const r of sorted) {
    const start = Math.max(0, Number(r.start))
    const end = Math.min(len, Math.max(start, Number(r.end)))
    if (start >= end) continue
    clamped.push({ ...r, start, end })
  }
  const merged: Array<HighlightRange & { source: HighlightSource }> = []
  for (const r of clamped) {
    const last = merged[merged.length - 1]
    const sameFact = r.factId && last?.factId === r.factId
    if (merged.length && r.source === last?.source && (sameFact || (!r.factId && !last?.factId)) && r.start <= last.end) {
      merged[merged.length - 1].end = Math.max(last.end, r.end)
    } else {
      merged.push({ ...r })
    }
  }
  const segmentEls: ReactNode[] = []
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i]
    const rangesInSeg = merged.filter(r => r.end > seg.start && r.start < seg.end)
    const rangesClipped = rangesInSeg.map(r => ({
      ...r,
      start: Math.max(r.start, seg.start),
      end: Math.min(r.end, seg.end),
    }))
    if (seg.isHeader) {
      segmentEls.push(<h2 key={`h-${i}`} className="reader-h2">{seg.headerLabel ?? seg.text}</h2>)
    } else {
      segmentEls.push(
        <div key={`p-${i}`} className="reader-section-body">
          <p className={`reader-raw-p ${rangesClipped.length > 0 ? 'reader-raw-p-with-highlights' : ''}`}>
            <SegmentWithHighlights
              segmentText={seg.text}
              segStart={seg.start}
              segEnd={seg.end}
              ranges={rangesClipped}
              normalizedLen={len}
              onHighlightHover={onHighlightHover}
              onHighlightLeave={onHighlightLeave}
            />
          </p>
        </div>
      )
    }
  }
  return <div className="reader-markdown">{segmentEls}</div>
}

export function ReadDocumentTab({ documents, selectedDocumentId: selectedDocumentIdProp, navigateToRead, onNavigateToReadConsumed, onDocumentSelect }: ReadDocumentTabProps) {
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(selectedDocumentIdProp ?? null)
  const [pages, setPages] = useState<Page[]>([])
  const [selectedPage, setSelectedPage] = useState<number | null>(null)
  const [pageZoom, setPageZoom] = useState(1.0)
  const [loading, setLoading] = useState(false)
  const [userHighlightedRangesByPage, setUserHighlightedRangesByPage] = useState<Record<number, HighlightRange[]>>({})
  const [llmHighlightedRangesByPage, setLlmHighlightedRangesByPage] = useState<Record<number, HighlightRange[]>>({})

  const [contextMenu, setContextMenu] = useState<{ x: number; y: number } | null>(null)
  const [contextMenuFactId, setContextMenuFactId] = useState<string | null>(null)
  const [contextMenuFactContext, setContextMenuFactContext] = useState<{
    factText: string
    categoryScores: Record<string, { score: number; direction: number }>
    isPertinent: boolean
  } | null>(null)
  const [selectedText, setSelectedText] = useState('')
  const [selectedPageForMenu, setSelectedPageForMenu] = useState<number | null>(null)
  const [modalMode, setModalMode] = useState<'add' | 'edit'>('add')
  const [modalFactId, setModalFactId] = useState<string | null>(null)

  const [modalOpen, setModalOpen] = useState(false)
  const [modalFactText, setModalFactText] = useState('')
  const [modalPageNumber, setModalPageNumber] = useState<number | null>(null)
  const [modalStartOffset, setModalStartOffset] = useState<number | undefined>(undefined)
  const [modalEndOffset, setModalEndOffset] = useState<number | undefined>(undefined)
  const [modalPertinent, setModalPertinent] = useState(true)
  const [modalCategoryScores, setModalCategoryScores] = useState<Record<string, { score: number; direction: number }>>({})
  const [modalSubmitting, setModalSubmitting] = useState(false)
  const [modalError, setModalError] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)
  const [sectionsSidebarOpen, setSectionsSidebarOpen] = useState(true)
  const [toolbarExpanded, setToolbarExpanded] = useState(true)
  const [factTooltip, setFactTooltip] = useState<{ data: TooltipData; rect: DOMRect } | null>(null)
  const tooltipShowRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const tooltipHideRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pageScrollRef = useRef<HTMLDivElement | null>(null)

  const handleHighlightHover = useCallback((r: HighlightRange, rect: DOMRect) => {
    if (tooltipHideRef.current) {
      clearTimeout(tooltipHideRef.current)
      tooltipHideRef.current = null
    }
    const data = getTooltipData(r)
    if (!data) {
      setFactTooltip(null)
      return
    }
    tooltipShowRef.current = setTimeout(() => {
      setFactTooltip({ data, rect })
      tooltipShowRef.current = null
    }, 120)
  }, [])

  const handleHighlightLeave = useCallback(() => {
    if (tooltipShowRef.current) {
      clearTimeout(tooltipShowRef.current)
      tooltipShowRef.current = null
    }
    tooltipHideRef.current = setTimeout(() => {
      setFactTooltip(null)
      tooltipHideRef.current = null
    }, 80)
  }, [])

  const fetchPages = useCallback(async (documentId: string) => {
    setLoading(true)
    try {
      const response = await fetch(`${API_BASE}/documents/${documentId}/pages`)
      if (response.ok) {
        const data = await response.json()
        const newPages = data.pages || []
        setPages(newPages)
        if (newPages.length > 0) {
          setSelectedPage(prev => prev ?? newPages[0].page_number)
        }
      }
    } catch (err) {
      console.error('Failed to load pages:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  const fetchFactsForHighlights = useCallback(async (documentId: string) => {
    try {
      const response = await fetch(`${API_BASE}/documents/${documentId}/facts`)
      if (!response.ok) return
      const data = await response.json()
      const chunks = data.chunks || []
      const userByPage: Record<number, HighlightRange[]> = {}
      const llmByPage: Record<number, HighlightRange[]> = {}
      const chunkPageNum = (chunk: { page_number?: number }) => (chunk.page_number != null ? Number(chunk.page_number) : null)
      const toScore = (v: unknown): number | null =>
        v != null && typeof v === 'number' && !isNaN(v) ? v : typeof v === 'string' ? parseFloat(v) : null
      for (const chunk of chunks) {
        const isUserChunk = chunkPageNum(chunk) === 0
        const chunkPage = chunkPageNum(chunk)
        const facts = chunk.facts || []
        for (const fact of facts) {
          // Use fact.page_number with fallback to chunk.page_number (AI facts have offsets in page markdown)
          const pn = fact.page_number != null ? Number(fact.page_number) : chunkPage
          const so = fact.start_offset != null ? Number(fact.start_offset) : null
          const eo = fact.end_offset != null ? Number(fact.end_offset) : null
          if (pn == null || so == null || eo == null || so >= eo) continue
          const isPertinent = fact.is_pertinent_to_claims_or_members === true || fact.is_pertinent_to_claims_or_members === 'true'
          const categoryScores: Record<string, { score: number | null; direction: number | null }> = {}
          if (fact.category_scores && typeof fact.category_scores === 'object') {
            for (const [k, v] of Object.entries(fact.category_scores as Record<string, { score?: unknown; direction?: unknown }>)) {
              const score = toScore(v?.score)
              const direction = toScore(v?.direction)
              if (score != null || direction != null) categoryScores[k] = { score, direction }
            }
          }
          const range: HighlightRange = {
            start: so,
            end: eo,
            factId: fact.id,
            factText: fact.fact_text,
            categoryScores: Object.keys(categoryScores).length ? categoryScores : undefined,
            isPertinent,
            verificationStatus: fact.verification_status ?? null,
          }
          if (isUserChunk) {
            if (!userByPage[pn]) userByPage[pn] = []
            userByPage[pn].push(range)
          } else {
            /* LLM facts: show both pertinent (blue) and non-pertinent (grey) */
            if (!llmByPage[pn]) llmByPage[pn] = []
            llmByPage[pn].push(range)
          }
        }
      }
      setUserHighlightedRangesByPage(userByPage)
      setLlmHighlightedRangesByPage(llmByPage)
    } catch (err) {
      console.error('Failed to load facts for highlights:', err)
    }
  }, [])

  useEffect(() => {
    if (selectedDocumentId) {
      fetchPages(selectedDocumentId)
      fetchFactsForHighlights(selectedDocumentId)
    }
  }, [selectedDocumentId, fetchPages, fetchFactsForHighlights])

  // Sync document when navigating from Review Facts or when App selection changes
  useEffect(() => {
    if (navigateToRead?.documentId) {
      setSelectedDocumentId(navigateToRead.documentId)
    }
  }, [navigateToRead?.documentId])

  useEffect(() => {
    if (selectedDocumentIdProp != null) {
      setSelectedDocumentId(selectedDocumentIdProp)
    }
  }, [selectedDocumentIdProp])

  // Set page when navigating from Review Facts (after pages load)
  useEffect(() => {
    if (
      navigateToRead?.pageNumber != null &&
      selectedDocumentId === navigateToRead?.documentId &&
      pages.length > 0
    ) {
      setSelectedPage(navigateToRead.pageNumber)
    }
  }, [navigateToRead?.documentId, navigateToRead?.pageNumber, selectedDocumentId, pages.length])

  // Scroll to fact highlight when navigating from Review Facts
  useEffect(() => {
    if (
      !navigateToRead?.factId ||
      selectedDocumentId !== navigateToRead?.documentId ||
      !selectedPage
    ) {
      return
    }
    const factId = navigateToRead.factId
    const timer = setTimeout(() => {
      const el = document.querySelector(`[data-fact-id="${factId}"]`)
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' })
      }
      onNavigateToReadConsumed?.()
    }, 300)
    return () => clearTimeout(timer)
  }, [
    navigateToRead?.documentId,
    navigateToRead?.factId,
    selectedDocumentId,
    selectedPage,
    onNavigateToReadConsumed,
  ])

  const handleDocumentChange = (documentId: string) => {
    setSelectedDocumentId(documentId)
    setPages([])
    setSelectedPage(null)
    setUserHighlightedRangesByPage({})
    setLlmHighlightedRangesByPage({})
    setContextMenu(null)
    setContextMenuFactId(null)
    setContextMenuFactContext(null)
    setModalOpen(false)
    if (onDocumentSelect) {
      onDocumentSelect(documentId)
    }
  }

  const handlePageSelect = (pageNumber: number) => {
    setSelectedPage(pageNumber)
  }

  const goToPreviousPage = () => {
    if (selectedPage && selectedPage > 1) {
      setSelectedPage(selectedPage - 1)
    }
  }

  const goToNextPage = () => {
    if (selectedPage && pages.length > 0) {
      const maxPage = Math.max(...pages.map(p => p.page_number))
      if (selectedPage < maxPage) {
        setSelectedPage(selectedPage + 1)
      }
    }
  }

  const zoomIn = () => setPageZoom(prev => Math.min(prev + 0.25, 3.0))
  const zoomOut = () => setPageZoom(prev => Math.max(prev - 0.25, 0.5))
  const resetZoom = () => setPageZoom(1.0)

  // Keyboard zoom: Ctrl/Cmd + Plus/Minus/0 (reading focus)
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (!e.ctrlKey && !e.metaKey) return
      if (e.key === '=' || e.key === '+') {
        e.preventDefault()
        setPageZoom(prev => Math.min(prev + 0.25, 3.0))
      } else if (e.key === '-') {
        e.preventDefault()
        setPageZoom(prev => Math.max(prev - 0.25, 0.5))
      } else if (e.key === '0') {
        e.preventDefault()
        setPageZoom(1.0)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  const normalizeText = (text: string | null | undefined): string => {
    if (!text) return 'No text available'
    return text
      .split('\n')
      .map(line => line.replace(/\s+/g, ' ').trim())
      .join('\n')
      .replace(/\n{3,}/g, '\n\n')
      .trim()
  }

  const handleContextMenu = (e: React.MouseEvent) => {
    const target = e.target as HTMLElement
    const span = target.closest?.('.reader-fact-highlight, .reader-llm-fact-highlight')
    const factId = span?.getAttribute?.('data-fact-id')
    if (factId && selectedPage != null) {
      const userR = (userHighlightedRangesByPage[selectedPage] || []).find(r => r.factId === factId)
      const llmR = (llmHighlightedRangesByPage[selectedPage] || []).find(r => r.factId === factId)
      const r = userR ?? llmR
      if (r) {
        e.preventDefault()
        setContextMenuFactId(factId)
        const scores: Record<string, { score: number; direction: number }> = {}
        for (const { key } of CATEGORIES) {
          const v = r.categoryScores?.[key]
          scores[key] = {
            score: v && typeof v.score === 'number' ? v.score : 0,
            direction: v && typeof v.direction === 'number' ? v.direction : 0.5,
          }
        }
        setContextMenuFactContext({
          factText: r.factText ?? '',
          categoryScores: Object.keys(scores).length ? scores : Object.fromEntries(CATEGORIES.map(({ key }) => [key, { score: 0, direction: 0.5 }])),
          isPertinent: r.isPertinent ?? true,
        })
        setContextMenu({ x: e.clientX, y: e.clientY })
        return
      }
    }
    const sel = window.getSelection()
    const text = (sel?.toString() ?? '').trim()
    if (text) {
      e.preventDefault()
      setSelectedText(text)
      setSelectedPageForMenu(selectedPage)
      setContextMenuFactId(null)
      setContextMenuFactContext(null)
      setContextMenu({ x: e.clientX, y: e.clientY })
    }
  }

  const openMarkAsFactModal = () => {
    setContextMenu(null)
    setContextMenuFactId(null)
    setContextMenuFactContext(null)
    setModalMode('add')
    setModalFactId(null)
    if (!selectedDocumentId || selectedPageForMenu == null) return
    const pageForModal = pages.find(p => p.page_number === selectedPageForMenu)
    if (!pageForModal) return
    const sourceText = pageForModal.text_markdown ?? pageForModal.text ?? ''
    const idx = selectedText ? sourceText.indexOf(selectedText) : -1
    let startOffset: number | undefined
    let endOffset: number | undefined
    if (idx >= 0) {
      startOffset = idx
      endOffset = idx + selectedText.length
    }
    setModalFactText(selectedText)
    setModalPageNumber(selectedPageForMenu)
    setModalStartOffset(startOffset)
    setModalEndOffset(endOffset)
    setModalPertinent(true)
    setModalCategoryScores(
      Object.fromEntries(CATEGORIES.map(({ key }) => [key, { score: 0, direction: 0.5 }]))
    )
    setModalError(null)
    setModalOpen(true)
  }

  const openEditFactModal = () => {
    if (!contextMenuFactId || !contextMenuFactContext) return
    setModalMode('edit')
    setModalFactId(contextMenuFactId)
    setModalFactText(contextMenuFactContext.factText)
    setModalPageNumber(selectedPage)
    setModalStartOffset(undefined)
    setModalEndOffset(undefined)
    setModalPertinent(contextMenuFactContext.isPertinent)
    setModalCategoryScores(contextMenuFactContext.categoryScores)
    setModalError(null)
    setModalOpen(true)
    setContextMenu(null)
    setContextMenuFactId(null)
    setContextMenuFactContext(null)
  }

  const deleteFact = async () => {
    if (!selectedDocumentId || !contextMenuFactId) return
    try {
      const response = await deleteFactApi(selectedDocumentId, contextMenuFactId, API_BASE)
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        setSuccessMessage(err.detail || 'Failed to delete fact')
        setTimeout(() => setSuccessMessage(null), 3000)
        return
      }
      setContextMenu(null)
      setContextMenuFactId(null)
      setContextMenuFactContext(null)
      if (selectedDocumentId) fetchFactsForHighlights(selectedDocumentId)
      setSuccessMessage('Fact deleted')
      setTimeout(() => setSuccessMessage(null), 3000)
    } catch {
      setSuccessMessage('Failed to delete fact')
      setTimeout(() => setSuccessMessage(null), 3000)
    }
  }

  const approveFact = async () => {
    if (!selectedDocumentId || !contextMenuFactId) return
    try {
      const response = await approveFactApi(selectedDocumentId, contextMenuFactId, API_BASE)
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        setSuccessMessage(err.detail || 'Failed to approve fact')
        setTimeout(() => setSuccessMessage(null), 3000)
        return
      }
      setContextMenu(null)
      setContextMenuFactId(null)
      setContextMenuFactContext(null)
      if (selectedDocumentId) fetchFactsForHighlights(selectedDocumentId)
      setSuccessMessage('Fact approved')
      setTimeout(() => setSuccessMessage(null), 3000)
    } catch {
      setSuccessMessage('Failed to approve fact')
      setTimeout(() => setSuccessMessage(null), 3000)
    }
  }

  const rejectFact = async () => {
    if (!selectedDocumentId || !contextMenuFactId) return
    try {
      const response = await rejectFactApi(selectedDocumentId, contextMenuFactId, API_BASE)
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        setSuccessMessage(err.detail || 'Failed to reject fact')
        setTimeout(() => setSuccessMessage(null), 3000)
        return
      }
      setContextMenu(null)
      setContextMenuFactId(null)
      setContextMenuFactContext(null)
      if (selectedDocumentId) fetchFactsForHighlights(selectedDocumentId)
      setSuccessMessage('Fact rejected')
      setTimeout(() => setSuccessMessage(null), 3000)
    } catch {
      setSuccessMessage('Failed to reject fact')
      setTimeout(() => setSuccessMessage(null), 3000)
    }
  }

  const togglePertinentFact = async () => {
    if (!selectedDocumentId || !contextMenuFactId || !contextMenuFactContext) return
    const next = !contextMenuFactContext.isPertinent
    try {
      const response = await patchFactApi(
        selectedDocumentId,
        contextMenuFactId,
        { is_pertinent_to_claims_or_members: next },
        API_BASE
      )
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        setSuccessMessage(err.detail || 'Failed to update pertinence')
        setTimeout(() => setSuccessMessage(null), 3000)
        return
      }
      setContextMenu(null)
      setContextMenuFactId(null)
      setContextMenuFactContext(null)
      if (selectedDocumentId) fetchFactsForHighlights(selectedDocumentId)
      setSuccessMessage(next ? 'Marked pertinent' : 'Marked not pertinent')
      setTimeout(() => setSuccessMessage(null), 3000)
    } catch {
      setSuccessMessage('Failed to update pertinence')
      setTimeout(() => setSuccessMessage(null), 3000)
    }
  }

  const closeModal = () => {
    setModalOpen(false)
    setModalError(null)
    setModalFactId(null)
    setModalMode('add')
  }

  const handleCategoryScoreChange = (key: string, score: number) => {
    setModalCategoryScores(prev => ({
      ...prev,
      [key]: { score, direction: prev[key]?.direction ?? 0.5 },
    }))
  }

  const submitReaderFact = async () => {
    const factText = modalFactText.trim()
    if (!factText || !selectedDocumentId) return
    if (modalMode === 'add' && modalPageNumber == null) return
    setModalSubmitting(true)
    setModalError(null)
    try {
      const category_scores: Record<string, { score: number; direction: number }> = {}
      for (const { key } of CATEGORIES) {
        const entry = modalCategoryScores[key]
        if (entry && entry.score > 0) {
          category_scores[key] = { score: entry.score, direction: entry.direction }
        }
      }
      if (modalMode === 'edit' && modalFactId) {
        const response = await patchFactApi(
          selectedDocumentId,
          modalFactId,
          {
            fact_text: factText,
            is_pertinent_to_claims_or_members: modalPertinent,
            category_scores,
          },
          API_BASE
        )
        if (!response.ok) {
          const errData = await response.json().catch(() => ({}))
          setModalError(errData.detail || response.statusText || 'Failed to update fact')
          return
        }
        if (selectedDocumentId) fetchFactsForHighlights(selectedDocumentId)
        setSuccessMessage('Fact updated')
        setTimeout(() => setSuccessMessage(null), 3000)
        closeModal()
        return
      }
      let startOffset = modalStartOffset
      let endOffset = modalEndOffset
      if ((startOffset == null || endOffset == null) && modalPageNumber != null) {
        const pageForFact = pages.find(p => p.page_number === modalPageNumber)
        const sourceText = pageForFact?.text_markdown ?? pageForFact?.text ?? ''
        if (sourceText && factText) {
          const isMd = !!pageForFact?.text_markdown
          const searchIn = isMd ? sourceText : normalizeText(sourceText)
          const searchFact = isMd ? factText : normalizeText(factText)
          if (searchFact) {
            let idx = searchIn.indexOf(searchFact)
            if (idx >= 0) {
              startOffset = idx
              endOffset = idx + (isMd ? factText.length : searchFact.length)
            } else {
              const lengths = [500, 300, 200, 150, 100, 80, 50, 30]
              for (const len of lengths) {
                if (searchFact.length <= len) continue
                const prefix = searchFact.slice(0, len)
                idx = searchIn.indexOf(prefix)
                if (idx >= 0) {
                  startOffset = idx
                  endOffset = idx + (isMd ? factText.slice(0, len).length : prefix.length)
                  break
                }
              }
            }
            if ((startOffset == null || endOffset == null) && isMd && factText.trim()) {
              const escaped = factText.trim().replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
              const flexiblePattern = escaped.replace(/\s+/g, '\\s+')
              try {
                const re = new RegExp(flexiblePattern)
                const match = sourceText.match(re)
                if (match && match.index !== undefined) {
                  startOffset = match.index
                  endOffset = match.index + match[0].length
                }
              } catch {
                // regex invalid, skip
              }
            }
          }
        }
      }
      const body: Record<string, unknown> = {
        fact_text: factText,
        page_number: modalPageNumber,
        is_pertinent_to_claims_or_members: modalPertinent,
        category_scores,
      }
      if (startOffset != null && endOffset != null && startOffset < endOffset) {
        body.start_offset = startOffset
        body.end_offset = endOffset
      }
      const response = await fetch(`${API_BASE}/documents/${selectedDocumentId}/reader-facts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        setModalError(errData.detail || response.statusText || 'Failed to add fact')
        return
      }
      if (startOffset != null && endOffset != null && modalPageNumber != null) {
        const newRange: HighlightRange = { start: startOffset, end: endOffset }
        setUserHighlightedRangesByPage(prev => {
          const list = prev[modalPageNumber] || []
          return { ...prev, [modalPageNumber]: [...list, newRange] }
        })
      }
      setSuccessMessage('Fact added')
      setTimeout(() => setSuccessMessage(null), 3000)
      closeModal()
      // Refetch after a short delay so the optimistic highlight is visible first and the DB has the new fact
      if (selectedDocumentId) {
        setTimeout(() => fetchFactsForHighlights(selectedDocumentId), 400)
      }
    } catch (err) {
      setModalError(err instanceof Error ? err.message : 'Request failed')
    } finally {
      setModalSubmitting(false)
    }
  }

  useEffect(() => {
    const hideContextMenu = () => {
      setContextMenu(null)
      setContextMenuFactId(null)
      setContextMenuFactContext(null)
    }
    document.addEventListener('click', hideContextMenu)
    return () => document.removeEventListener('click', hideContextMenu)
  }, [])

  useEffect(() => {
    const el = pageScrollRef.current
    if (!el) return
    const onScroll = () => {
      if (tooltipShowRef.current) {
        clearTimeout(tooltipShowRef.current)
        tooltipShowRef.current = null
      }
      setFactTooltip(null)
    }
    el.addEventListener('scroll', onScroll, { passive: true })
    return () => el.removeEventListener('scroll', onScroll)
  }, [selectedPage])

  const currentPage = pages.find(p => p.page_number === selectedPage)
  const totalPages = pages.length
  const userHighlightsForPage = selectedPage != null ? userHighlightedRangesByPage[selectedPage] || [] : []
  const llmHighlightsForPage = selectedPage != null ? llmHighlightedRangesByPage[selectedPage] || [] : []
  const hasHighlights = userHighlightsForPage.length > 0 || llmHighlightsForPage.length > 0
  const showRawWithHighlights = hasHighlights && currentPage
  const selectedDoc = documents.find(d => d.id === selectedDocumentId)
  const documentDisplayName = selectedDoc?.display_name?.trim() || selectedDoc?.filename || 'Document'
  const showReadingView = !!selectedDocumentId

  return (
    <div className="read-document-tab">
      {/* Top bar: collapsible when a document is selected (reading view) */}
      {showReadingView && !toolbarExpanded ? (
        <div className="read-tab-topbar read-tab-topbar-minimal">
          <span className="read-tab-doc-name" title={documentDisplayName}>{documentDisplayName}</span>
          <button
            type="button"
            className="btn-read-toolbar-expand"
            onClick={() => setToolbarExpanded(true)}
            title="Show document & zoom controls"
            aria-label="Show controls"
          >
            ▾ Show controls
          </button>
        </div>
      ) : (
        <div className="read-tab-topbar">
          <div className="document-selector">
            <label htmlFor="document-select" className="selector-label">
              Select Document:
            </label>
            <select
              id="document-select"
              value={selectedDocumentId || ''}
              onChange={(e) => handleDocumentChange(e.target.value)}
              className="document-select"
            >
              <option value="">-- Choose a document --</option>
              {documents.map((doc) => (
                <option key={doc.id} value={doc.id}>
                  {doc.display_name?.trim() || doc.filename}
                </option>
              ))}
            </select>
          </div>
          {showReadingView && (
            <button
              type="button"
              className="btn-read-toolbar-hide"
              onClick={() => setToolbarExpanded(false)}
              title="Hide toolbar for reading"
              aria-label="Hide toolbar"
            >
              Hide ›
            </button>
          )}
        </div>
      )}

      {/* Floating zoom: always available when reading (keyboard: Ctrl/Cmd + − / + / 0) */}
      {showReadingView && (
        <div className="read-zoom-float" title="Zoom (or use Ctrl/Cmd + − / + / 0)">
          <button type="button" className="read-zoom-float-btn" onClick={zoomOut} disabled={pageZoom <= 0.5} aria-label="Zoom out">−</button>
          <span className="read-zoom-float-level">{Math.round(pageZoom * 100)}%</span>
          <button type="button" className="read-zoom-float-btn" onClick={zoomIn} disabled={pageZoom >= 3.0} aria-label="Zoom in">+</button>
          <button type="button" className="read-zoom-float-reset" onClick={resetZoom} aria-label="Reset zoom">Reset</button>
        </div>
      )}

      {selectedDocumentId && (
        <div className={`page-viewer-layout ${sectionsSidebarOpen ? '' : 'sidebar-hidden'}`}>
          {/* Left Sidebar - Section (page) list; collapsible */}
          <div className="pages-sidebar" aria-hidden={!sectionsSidebarOpen}>
            <h3 className="sidebar-title">Sections</h3>
            {loading ? (
              <div className="loading-pages">Loading sections...</div>
            ) : (
              <div className="pages-list">
                {pages.map((page) => {
                  const firstHeader = getFirstSectionHeader(page.text_markdown ?? page.text)
                  const label = firstHeader ? `${page.page_number} – ${firstHeader}` : String(page.page_number)
                  return (
                    <button
                      key={page.page_number}
                      className={`page-item ${selectedPage === page.page_number ? 'active' : ''}`}
                      onClick={() => handlePageSelect(page.page_number)}
                      title={firstHeader ? `Page ${page.page_number}: ${firstHeader}` : `Page ${page.page_number}`}
                    >
                      <span className="page-item-label">{label}</span>
                      {page.extraction_status !== 'success' && (
                        <span className="page-status-badge">{page.extraction_status}</span>
                      )}
                    </button>
                  )
                })}
              </div>
            )}
          </div>

          {/* Chevron to collapse/expand sections panel */}
          <button
            type="button"
            className="sidebar-chevron"
            onClick={() => setSectionsSidebarOpen(prev => !prev)}
            title={sectionsSidebarOpen ? 'Collapse sections' : 'Expand sections'}
            aria-label={sectionsSidebarOpen ? 'Collapse sections' : 'Expand sections'}
          >
            {sectionsSidebarOpen ? '‹' : '›'}
          </button>

          {/* Main Area - Page Content */}
          <div className="page-content-area">
            {currentPage ? (
              <>
                {successMessage && <div className="reader-success-message">{successMessage}</div>}
                {/* Page content with left/right chevrons for simple navigation */}
                <div className="page-reader-with-nav-chevrons">
                  <button
                    type="button"
                    className="page-nav-chevron page-nav-chevron-left"
                    onClick={goToPreviousPage}
                    disabled={!selectedPage || selectedPage === 1}
                    title="Previous section"
                    aria-label="Previous section"
                  >
                    ‹
                  </button>
                  <div ref={pageScrollRef} className="page-content-wrapper" style={{ zoom: pageZoom }}>
                    <div className="book-page">
                    <h4>
                      {(() => {
                        const firstHeader = getFirstSectionHeader(currentPage.text_markdown ?? currentPage.text)
                        const headerLine = firstHeader ? `${currentPage.page_number} – ${firstHeader}` : `Page ${currentPage.page_number}`
                        return (
                          <>
                            {headerLine}
                            {currentPage.text_markdown && (
                              <span className="reader-canonical-note" title="Content from markdown (canonical source)">
                                {' '}MD
                              </span>
                            )}
                          </>
                        )
                      })()}
                    </h4>
                    <div
                      className="page-text-content"
                      onContextMenu={handleContextMenu}
                    >
                      {showRawWithHighlights ? (
                        <PageTextWithHighlights
                          text={currentPage.text_markdown ?? currentPage.text ?? ''}
                          userRanges={userHighlightsForPage}
                          llmRanges={llmHighlightsForPage}
                          isMarkdown={!!currentPage.text_markdown}
                          onHighlightHover={handleHighlightHover}
                          onHighlightLeave={handleHighlightLeave}
                        />
                      ) : currentPage.text_markdown ? (
                        <SimpleMarkdown content={currentPage.text_markdown} />
                      ) : (
                        <PageTextWithHighlights
                          text={currentPage.text ?? ''}
                          userRanges={[]}
                          llmRanges={[]}
                          isMarkdown={false}
                          onHighlightHover={handleHighlightHover}
                          onHighlightLeave={handleHighlightLeave}
                        />
                      )}
                    </div>
                  </div>
                </div>
                  <button
                    type="button"
                    className="page-nav-chevron page-nav-chevron-right"
                    onClick={goToNextPage}
                    disabled={!selectedPage || selectedPage === totalPages}
                    title="Next section"
                    aria-label="Next section"
                  >
                    ›
                  </button>
                </div>

                {/* Fact hover tooltip */}
                {factTooltip && (
                  <FactTooltip data={factTooltip.data} anchorRect={factTooltip.rect} />
                )}

                {/* Context menu */}
                {contextMenu && (
                  <div
                    className="reader-context-menu"
                    style={{ left: contextMenu.x, top: contextMenu.y }}
                    onClick={(e) => e.stopPropagation()}
                  >
                    {contextMenuFactId ? (
                      <>
                        {(() => {
                          const ur = (userHighlightedRangesByPage[selectedPage ?? 0] || []).find(r => r.factId === contextMenuFactId)
                          const lr = (llmHighlightedRangesByPage[selectedPage ?? 0] || []).find(r => r.factId === contextMenuFactId)
                          const r = ur ?? lr
                          const isApproved = r?.verificationStatus === 'approved'
                          const isRejected = r?.verificationStatus === 'rejected'
                          return (
                            <>
                              {isApproved ? (
                                <button type="button" className="reader-context-menu-item" disabled>
                                  Approved
                                </button>
                              ) : (
                                <button type="button" className="reader-context-menu-item" onClick={approveFact}>
                                  Approve
                                </button>
                              )}
                              {isRejected ? (
                                <button type="button" className="reader-context-menu-item" disabled>
                                  Rejected
                                </button>
                              ) : (
                                <button type="button" className="reader-context-menu-item" onClick={rejectFact}>
                                  Reject
                                </button>
                              )}
                            </>
                          )
                        })()}
                        <button type="button" className="reader-context-menu-item" onClick={togglePertinentFact}>
                          {contextMenuFactContext?.isPertinent ? 'Mark not pertinent' : 'Mark pertinent'}
                        </button>
                        <button type="button" className="reader-context-menu-item" onClick={openEditFactModal}>
                          Edit fact
                        </button>
                        <button type="button" className="reader-context-menu-item" onClick={deleteFact}>
                          Delete fact
                        </button>
                      </>
                    ) : (
                      <button type="button" className="reader-context-menu-item" onClick={openMarkAsFactModal}>
                        Mark as fact
                      </button>
                    )}
                  </div>
                )}

                {/* Navigation */}
                <div className="page-navigation">
                  <button
                    onClick={goToPreviousPage}
                    disabled={!selectedPage || selectedPage === 1}
                    className="btn btn-secondary"
                  >
                    ← Previous
                  </button>
                  <span className="page-indicator">
                    Page {selectedPage} of {totalPages}
                  </span>
                  <button
                    onClick={goToNextPage}
                    disabled={!selectedPage || selectedPage === totalPages}
                    className="btn btn-secondary"
                  >
                    Next →
                  </button>
                </div>
              </>
            ) : (
              <div className="no-page-selected">
                {loading ? 'Loading...' : 'Select a section from the sidebar'}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Modal: Add or Edit fact */}
      {modalOpen && (
        <div className="reader-modal-overlay" onClick={closeModal}>
          <div className="reader-modal" onClick={(e) => e.stopPropagation()}>
            <h3 className="reader-modal-title">{modalMode === 'edit' ? 'Edit fact' : 'Add fact from selection'}</h3>
            {modalPageNumber != null && modalMode === 'add' && (
              <p className="reader-modal-page">From page {modalPageNumber}</p>
            )}
            <div className="reader-modal-field">
              <label htmlFor="reader-modal-fact-text">Fact text</label>
              <textarea
                id="reader-modal-fact-text"
                value={modalFactText}
                onChange={(e) => setModalFactText(e.target.value)}
                rows={4}
                className="reader-modal-textarea"
              />
            </div>
            <div className="reader-modal-field">
              <label>Category relevance (sliding scale 0–1)</label>
              <p className="reader-modal-category-hint">Set score per category; 0 = not relevant.</p>
              <div className="reader-modal-categories reader-modal-category-sliders">
                {CATEGORIES.map(({ key, label }) => (
                  <div key={key} className="reader-modal-category-row">
                    <label className="reader-modal-category-label" htmlFor={`reader-cat-${key}`}>
                      {label}
                    </label>
                    <div className="reader-modal-category-slider-wrap">
                      <input
                        id={`reader-cat-${key}`}
                        type="range"
                        min={0}
                        max={1}
                        step={0.1}
                        value={modalCategoryScores[key]?.score ?? 0}
                        onChange={(e) => handleCategoryScoreChange(key, parseFloat(e.target.value))}
                      />
                      <span className="reader-modal-category-value">
                        {(modalCategoryScores[key]?.score ?? 0).toFixed(1)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="reader-modal-field">
              <label className="reader-modal-pertinent-check">
                <input
                  type="checkbox"
                  checked={modalPertinent}
                  onChange={(e) => setModalPertinent(e.target.checked)}
                />
                Pertinent to claims or members
              </label>
            </div>
            {modalError && <div className="reader-modal-error">{modalError}</div>}
            <div className="reader-modal-actions">
              <button type="button" className="btn btn-secondary" onClick={closeModal}>
                Cancel
              </button>
              <button
                type="button"
                className="btn btn-primary"
                onClick={submitReaderFact}
                disabled={modalSubmitting || !modalFactText.trim()}
              >
                {modalSubmitting ? (modalMode === 'edit' ? 'Saving...' : 'Adding...') : modalMode === 'edit' ? 'Save changes' : 'Add fact'}
              </button>
            </div>
          </div>
        </div>
      )}

      {!selectedDocumentId && (
        <div className="no-document-selected">
          <p>Select a document to view its pages</p>
        </div>
      )}
    </div>
  )
}
