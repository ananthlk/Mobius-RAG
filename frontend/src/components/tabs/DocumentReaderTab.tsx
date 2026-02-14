/**
 * DocumentReaderTab – RAG wrapper for the shared @mobius/document-viewer.
 *
 * Fetches pages, facts, and tags for a document, converts them to generic
 * Highlight[], and provides an InteractionConfig that defines:
 *   - Tooltips for facts and tags
 *   - Right-click on highlight: approve/reject/delete fact, or remove tag
 *   - Right-click on selected text: "Mark as fact" + "Add tag > {category}"
 */
import { useState, useEffect, useCallback, useMemo } from 'react'
import {
  DocumentViewer,
  type Highlight,
  type PageData,
  type InteractionConfig,
  type TextSelectionContext,
} from '@mobius/document-viewer'
import '@mobius/document-viewer/dist/index.css'
import { API_BASE } from '../../config'
import { approveFactApi, deleteFactApi, rejectFactApi } from '../../lib/factActions'
import { FactTooltipContent, TagTooltipContent, PolicyLineTagTooltipContent, type FactTooltipData, type TagTooltipData } from '../FactTooltip'

/* ─── Tag taxonomy (domain > leaf tags) for right-click "Add tag" menu ─── */

interface TagDomain {
  domain: string
  label: string
  tags: readonly { key: string; label: string }[]
}

const TAG_TAXONOMY: readonly TagDomain[] = [
  {
    domain: 'claims', label: 'Claims', tags: [
      { key: 'claims.submission', label: 'Submission' },
      { key: 'claims.denial', label: 'Denial' },
      { key: 'claims.appeals_grievances', label: 'Appeals & Grievances' },
      { key: 'claims.clean_claim', label: 'Clean Claim' },
      { key: 'claims.timely_filing', label: 'Timely Filing' },
      { key: 'claims.coordination_of_benefits', label: 'Coordination of Benefits' },
      { key: 'claims.electronic_claims', label: 'Electronic Claims' },
      { key: 'claims.corrected_claims', label: 'Corrected Claims' },
    ],
  },
  {
    domain: 'eligibility', label: 'Eligibility', tags: [
      { key: 'eligibility.verification', label: 'Verification' },
      { key: 'eligibility.enrollment', label: 'Enrollment' },
      { key: 'eligibility.member_status', label: 'Member Status' },
      { key: 'eligibility.plan_assignment', label: 'Plan Assignment' },
    ],
  },
  {
    domain: 'utilization_management', label: 'Utilization Management', tags: [
      { key: 'utilization_management.prior_authorization', label: 'Prior Authorization' },
      { key: 'utilization_management.referrals', label: 'Referrals' },
      { key: 'utilization_management.medical_necessity', label: 'Medical Necessity' },
    ],
  },
  {
    domain: 'credentialing', label: 'Credentialing', tags: [
      { key: 'credentialing.general', label: 'Credentialing (general)' },
    ],
  },
  {
    domain: 'compliance', label: 'Compliance', tags: [
      { key: 'compliance.fraud_waste_abuse', label: 'Fraud, Waste & Abuse' },
      { key: 'compliance.hipaa', label: 'HIPAA' },
      { key: 'compliance.audits', label: 'Audits' },
      { key: 'compliance.nondiscrimination', label: 'Nondiscrimination' },
    ],
  },
  {
    domain: 'provider', label: 'Provider', tags: [
      { key: 'provider.network', label: 'Network' },
      { key: 'provider.relations', label: 'Relations' },
      { key: 'provider.services', label: 'Services' },
    ],
  },
  {
    domain: 'health_care_services', label: 'Health Care Services', tags: [
      { key: 'health_care_services.behavioral_health', label: 'Behavioral Health' },
      { key: 'health_care_services.primary_care', label: 'Primary Care' },
      { key: 'health_care_services.urgent_care', label: 'Urgent Care' },
    ],
  },
  {
    domain: 'pharmacy', label: 'Pharmacy', tags: [
      { key: 'pharmacy.pharmacy_benefit', label: 'Pharmacy Benefit' },
      { key: 'pharmacy.preferred_drug_list', label: 'Preferred Drug List' },
      { key: 'pharmacy.specialty_pharmacy', label: 'Specialty Pharmacy' },
    ],
  },
  {
    domain: 'responsibilities', label: 'Responsibilities', tags: [
      { key: 'responsibilities.continuity_of_care', label: 'Continuity of Care' },
      { key: 'responsibilities.training', label: 'Training' },
      { key: 'responsibilities.abuse_neglect_reporting', label: 'Abuse/Neglect Reporting' },
    ],
  },
  {
    domain: 'contact_information', label: 'Contact Information', tags: [
      { key: 'contact_information.phone', label: 'Phone' },
      { key: 'contact_information.portal', label: 'Portal' },
      { key: 'contact_information.provider_contact', label: 'Provider Contact' },
    ],
  },
] as const

// Flat list for backward compat (used by context menu rendering)
const _CATEGORIES = TAG_TAXONOMY.flatMap((d) => d.tags)
void _CATEGORIES // suppress unused warning

const CATEGORY_LABELS: Record<string, string> = Object.fromEntries(
  TAG_TAXONOMY.flatMap((d) => d.tags.map((t) => [t.key, `${d.label} > ${t.label}`])),
)

/* ─── Types ─── */
interface Document {
  id: string
  filename: string
  display_name?: string | null
  file_path?: string | null
}

interface DocumentReaderTabProps {
  documents: Document[]
  selectedDocumentId?: string | null
  navigateToRead?: { documentId: string; pageNumber?: number; factId?: string } | null
  onNavigateToReadConsumed?: () => void
  onDocumentSelect?: (documentId: string) => void
}

/* ─── Component ─── */
export function DocumentReaderTab({
  documents,
  selectedDocumentId: selectedDocumentIdProp,
  navigateToRead,
  onNavigateToReadConsumed,
  onDocumentSelect,
}: DocumentReaderTabProps) {
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(
    selectedDocumentIdProp ?? null,
  )
  const [pages, setPages] = useState<PageData[]>([])
  const [loading, setLoading] = useState(false)

  // Raw data from API
  const [factsRaw, setFactsRaw] = useState<any[]>([])
  const [tagsRaw, setTagsRaw] = useState<any[]>([])
  const [policyLineTagsRaw, setPolicyLineTagsRaw] = useState<any[]>([])

  // Refresh key to trigger refetch
  const [refreshKey, setRefreshKey] = useState(0)
  const triggerRefresh = () => setRefreshKey((k) => k + 1)

  /* ─── Sync external selection ─── */
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

  /* ─── Fetch pages ─── */
  const fetchPages = useCallback(async (documentId: string) => {
    setLoading(true)
    try {
      const response = await fetch(`${API_BASE}/documents/${documentId}/pages`)
      if (response.ok) {
        const data = await response.json()
        setPages(data.pages || [])
      }
    } catch (err) {
      console.error('Failed to load pages:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  /* ─── Fetch facts ─── */
  const fetchFacts = useCallback(async (documentId: string) => {
    try {
      const response = await fetch(`${API_BASE}/documents/${documentId}/facts`)
      if (!response.ok) return
      const data = await response.json()
      const chunks = data.chunks || []
      const allFacts: any[] = []
      for (const chunk of chunks) {
        for (const fact of chunk.facts || []) {
          // Propagate chunk's page_number when the fact doesn't have its own
          const factPageNumber = fact.page_number != null ? fact.page_number : chunk.page_number
          allFacts.push({
            ...fact,
            page_number: factPageNumber,
            _isUserChunk: chunk.page_number === 0,
            _chunkText: chunk.text, // Keep chunk text for offset computation
          })
        }
      }
      setFactsRaw(allFacts)
    } catch (err) {
      console.error('Failed to load facts:', err)
    }
  }, [])

  /* ─── Fetch user text-tags ─── */
  const fetchTags = useCallback(async (documentId: string) => {
    try {
      const response = await fetch(`${API_BASE}/documents/${documentId}/text-tags`)
      if (!response.ok) return
      const data = await response.json()
      setTagsRaw(data.tags || [])
    } catch (err) {
      console.error('Failed to load tags:', err)
    }
  }, [])

  /* ─── Fetch policy line tags (from Path B pipeline) ─── */
  const fetchPolicyLineTags = useCallback(async (documentId: string) => {
    try {
      const response = await fetch(`${API_BASE}/documents/${documentId}/policy-line-tags`)
      if (!response.ok) return
      const data = await response.json()
      setPolicyLineTagsRaw(data.lines || [])
    } catch (err) {
      console.error('Failed to load policy line tags:', err)
    }
  }, [])

  useEffect(() => {
    if (selectedDocumentId) {
      fetchPages(selectedDocumentId)
      fetchFacts(selectedDocumentId)
      fetchTags(selectedDocumentId)
      fetchPolicyLineTags(selectedDocumentId)
    } else {
      setPages([])
      setFactsRaw([])
      setTagsRaw([])
      setPolicyLineTagsRaw([])
    }
  }, [selectedDocumentId, fetchPages, fetchFacts, fetchTags, fetchPolicyLineTags, refreshKey])

  /* ─── Helper: build a lookup of page text for offset computation ─── */
  const pageTextByNumber = useMemo(() => {
    const map: Record<number, string> = {}
    for (const p of pages) {
      map[p.page_number] = p.text_markdown ?? p.text ?? ''
    }
    return map
  }, [pages])

  /* ─── Build generic Highlight[] from facts + tags ─── */
  const highlights = useMemo(() => {
    const byPage: Record<number, Highlight[]> = {}

    const toScore = (v: unknown): number | null =>
      v != null && typeof v === 'number' && !isNaN(v) ? v : typeof v === 'string' ? parseFloat(v) : null

    // Facts
    for (const fact of factsRaw) {
      const pn = fact.page_number != null ? Number(fact.page_number) : null
      if (pn == null) continue // Must have at least a page number

      let so = fact.start_offset != null ? Number(fact.start_offset) : null
      let eo = fact.end_offset != null ? Number(fact.end_offset) : null

      // If offsets are missing, compute them by searching for fact_text in the page text
      if ((so == null || eo == null || so >= eo) && fact.fact_text) {
        const pageText = pageTextByNumber[pn] ?? ''
        const idx = pageText.indexOf(fact.fact_text)
        if (idx >= 0) {
          so = idx
          eo = idx + fact.fact_text.length
        } else {
          // Try case-insensitive search
          const lowerIdx = pageText.toLowerCase().indexOf(fact.fact_text.toLowerCase())
          if (lowerIdx >= 0) {
            so = lowerIdx
            eo = lowerIdx + fact.fact_text.length
          } else {
            // No match found – still include the highlight with dummy offsets
            // (won't appear in markdown mode but will appear in PDF via matchText)
            so = 0
            eo = 0
          }
        }
      }
      if (so == null || eo == null) continue

      const isPertinent =
        fact.is_pertinent_to_claims_or_members === true ||
        fact.is_pertinent_to_claims_or_members === 'true'
      const categoryScores: Record<string, { score: number | null; direction: number | null }> = {}
      if (fact.category_scores && typeof fact.category_scores === 'object') {
        for (const [k, v] of Object.entries(
          fact.category_scores as Record<string, { score?: unknown; direction?: unknown }>,
        )) {
          const score = toScore(v?.score)
          const direction = toScore(v?.direction)
          if (score != null || direction != null) categoryScores[k] = { score, direction }
        }
      }

      const isUserFact = fact._isUserChunk === true
      const approved = fact.verification_status === 'approved'
      let cls = isUserFact ? 'dv-fact-highlight' : 'dv-llm-fact-highlight'
      if (!isUserFact && !isPertinent) cls += ' dv-llm-fact-highlight-non-pertinent'
      if (approved) cls += ' dv-fact-highlight-approved'

      const hl: Highlight = {
        id: fact.id,
        start: so,
        end: eo,
        label: isPertinent ? 'Pertinent fact' : 'Non-pertinent fact',
        className: cls,
        data: {
          type: 'fact',
          matchText: fact.fact_text,
          factText: fact.fact_text,
          isPertinent,
          verificationStatus: fact.verification_status ?? null,
          categoryScores: Object.keys(categoryScores).length ? categoryScores : undefined,
          isUserFact,
        } satisfies FactTooltipData & Record<string, unknown>,
      }

      if (!byPage[pn]) byPage[pn] = []
      byPage[pn].push(hl)
    }

    // User text-tags
    for (const tag of tagsRaw) {
      const pn = tag.page_number != null ? Number(tag.page_number) : null
      const so = tag.start_offset != null ? Number(tag.start_offset) : null
      const eo = tag.end_offset != null ? Number(tag.end_offset) : null
      if (pn == null || so == null || eo == null || so >= eo) continue

      const tagLabel = CATEGORY_LABELS[tag.tag] ?? tag.tag.replace(/_/g, ' ')
      const hl: Highlight = {
        id: tag.id,
        start: so,
        end: eo,
        label: `Tag: ${tagLabel}`,
        className: `dv-tag-highlight dv-tag-${tag.tag.replace(/\./g, '-')}`,
        data: {
          type: 'tag',
          matchText: tag.tagged_text,
          tag: tag.tag,
          tagLabel,
          taggedText: tag.tagged_text,
        } satisfies TagTooltipData & Record<string, unknown>,
      }

      if (!byPage[pn]) byPage[pn] = []
      byPage[pn].push(hl)
    }

    // Policy line tags (from Path B pipeline)
    for (const line of policyLineTagsRaw) {
      const pn = line.page_number != null ? Number(line.page_number) : null
      if (pn == null || !line.text) continue

      const lineText = (line.text as string).trim()
      if (!lineText) continue

      // Determine primary tag type and build a label
      const pTags = line.p_tags as Record<string, number> | null
      const dTags = line.d_tags as Record<string, number> | null
      const jTags = line.j_tags as Record<string, number> | null

      const allTagParts: string[] = []
      let primaryCls = 'dv-pltag-highlight'

      // Helper: tag code -> CSS-safe class suffix (dots -> hyphens)
      const cssSafe = (code: string) => code.replace(/\./g, '-').replace(/_/g, '-')
      // Helper: extract domain prefix (e.g. "claims.denial" -> "claims")
      const domainOf = (code: string) => code.includes('.') ? code.split('.')[0] : code

      if (pTags && Object.keys(pTags).length) {
        const keys = Object.keys(pTags)
        allTagParts.push(...keys.map((k) => `p:${k.replace(/\./g, ' > ').replace(/_/g, ' ')}`))
        primaryCls += ` dv-pltag-p dv-pltag-p-${cssSafe(domainOf(keys[0]))}`
      }
      if (dTags && Object.keys(dTags).length) {
        const keys = Object.keys(dTags)
        allTagParts.push(...keys.map((k) => `d:${k.replace(/\./g, ' > ').replace(/_/g, ' ')}`))
        if (!pTags || !Object.keys(pTags).length) {
          primaryCls += ` dv-pltag-d dv-pltag-d-${cssSafe(domainOf(keys[0]))}`
        }
      }
      if (jTags && Object.keys(jTags).length) {
        const keys = Object.keys(jTags)
        allTagParts.push(...keys.map((k) => `j:${k.replace(/\./g, ' > ').replace(/_/g, ' ')}`))
        if (!pTags && !dTags) {
          primaryCls += ` dv-pltag-j`
        }
      }

      const label = allTagParts.join(', ')

      // Compute offsets by text search if not available
      let so = line.start_offset != null ? Number(line.start_offset) : null
      let eo = line.end_offset != null ? Number(line.end_offset) : null
      if (so == null || eo == null || so >= eo) {
        const pageText = pageTextByNumber[pn] ?? ''
        const idx = pageText.indexOf(lineText)
        if (idx >= 0) {
          so = idx
          eo = idx + lineText.length
        } else {
          // Fallback: set dummy offsets; matchText will be used for PDF highlighting
          so = 0
          eo = 0
        }
      }

      const hl: Highlight = {
        id: line.id,
        start: so,
        end: eo,
        label,
        className: primaryCls,
        data: {
          type: 'policy-line-tag',
          matchText: lineText,
          p_tags: pTags,
          d_tags: dTags,
          j_tags: jTags,
          tagLabel: label,
          taggedText: lineText,
        },
      }

      if (!byPage[pn]) byPage[pn] = []
      byPage[pn].push(hl)
    }

    return byPage
  }, [factsRaw, tagsRaw, policyLineTagsRaw, pageTextByNumber])

  /* ─── Actions ─── */
  const markAsFact = async (selection: TextSelectionContext) => {
    if (!selectedDocumentId) return
    try {
      const res = await fetch(`${API_BASE}/documents/${selectedDocumentId}/reader-facts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          fact_text: selection.text,
          page_number: selection.pageNumber,
          start_offset: selection.startOffset,
          end_offset: selection.endOffset,
        }),
      })
      if (res.ok) triggerRefresh()
    } catch (err) {
      console.error('Failed to create fact:', err)
    }
  }

  const addTag = async (selection: TextSelectionContext, tagKey: string) => {
    if (!selectedDocumentId) return
    try {
      const res = await fetch(`${API_BASE}/documents/${selectedDocumentId}/text-tags`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          page_number: selection.pageNumber,
          start_offset: selection.startOffset,
          end_offset: selection.endOffset,
          tagged_text: selection.text,
          tag: tagKey,
        }),
      })
      if (res.ok) triggerRefresh()
    } catch (err) {
      console.error('Failed to create tag:', err)
    }
  }

  const approveFact = async (factId: string) => {
    if (!selectedDocumentId) return
    try {
      const res = await approveFactApi(selectedDocumentId, factId, API_BASE)
      if (res.ok) triggerRefresh()
    } catch (err) {
      console.error('Failed to approve fact:', err)
    }
  }

  const rejectFact = async (factId: string) => {
    if (!selectedDocumentId) return
    try {
      const res = await rejectFactApi(selectedDocumentId, factId, API_BASE)
      if (res.ok) triggerRefresh()
    } catch (err) {
      console.error('Failed to reject fact:', err)
    }
  }

  const deleteFact = async (factId: string) => {
    if (!selectedDocumentId) return
    try {
      const res = await deleteFactApi(selectedDocumentId, factId, API_BASE)
      if (res.ok) triggerRefresh()
    } catch (err) {
      console.error('Failed to delete fact:', err)
    }
  }

  const deleteTag = async (tagId: string) => {
    if (!selectedDocumentId) return
    try {
      const res = await fetch(`${API_BASE}/documents/${selectedDocumentId}/text-tags/${tagId}`, {
        method: 'DELETE',
      })
      if (res.ok) triggerRefresh()
    } catch (err) {
      console.error('Failed to delete tag:', err)
    }
  }

  /* ─── Interaction config (the key integration point) ─── */
  const interaction: InteractionConfig = useMemo(
    () => ({
      textSelectionEnabled: true,

      renderTooltip: (highlight) => {
        const d = highlight.data
        if (d?.type === 'fact') {
          return <FactTooltipContent data={d as unknown as FactTooltipData} />
        }
        if (d?.type === 'tag') {
          return <TagTooltipContent data={d as unknown as TagTooltipData} />
        }
        if (d?.type === 'policy-line-tag') {
          return <PolicyLineTagTooltipContent data={d as Record<string, unknown>} />
        }
        return null
      },

      renderHighlightMenu: (highlight, dismiss) => {
        const d = highlight.data
        if (d?.type === 'fact') {
          const verified = d.verificationStatus as string | null | undefined
          return (
            <>
              {verified !== 'approved' && (
                <button
                  type="button"
                  className="dv-context-menu-item"
                  onClick={() => { approveFact(highlight.id); dismiss() }}
                >
                  Approve fact
                </button>
              )}
              {verified !== 'rejected' && (
                <button
                  type="button"
                  className="dv-context-menu-item"
                  onClick={() => { rejectFact(highlight.id); dismiss() }}
                >
                  Reject fact
                </button>
              )}
              <button
                type="button"
                className="dv-context-menu-item"
                onClick={() => { deleteFact(highlight.id); dismiss() }}
              >
                Delete fact
              </button>
            </>
          )
        }
        if (d?.type === 'tag') {
          return (
            <button
              type="button"
              className="dv-context-menu-item"
              onClick={() => { deleteTag(highlight.id); dismiss() }}
            >
              Remove tag
            </button>
          )
        }
        return null
      },

      renderSelectionMenu: (selection, dismiss) => (
        <>
          <button
            type="button"
            className="dv-context-menu-item"
            onClick={() => { markAsFact(selection); dismiss() }}
          >
            Mark as fact
          </button>
          <div className="dv-context-menu-divider" />
          <span className="dv-context-menu-label">Add tag</span>
          {TAG_TAXONOMY.map((domain) => (
            <div key={domain.domain}>
              <span className="dv-context-menu-sublabel">{domain.label}</span>
              {domain.tags.map((tag) => (
                <button
                  key={tag.key}
                  type="button"
                  className="dv-context-menu-item dv-context-menu-item-indented"
                  onClick={() => { addTag(selection, tag.key); dismiss() }}
                >
                  {tag.label}
                </button>
              ))}
            </div>
          ))}
        </>
      ),
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [selectedDocumentId],
  )

  /* ─── Determine original file availability ─── */
  const selectedDoc = documents.find((d) => d.id === selectedDocumentId)
  const filePath = (selectedDoc as any)?.gcs_path ?? (selectedDoc as any)?.file_path ?? ''
  const hasOriginalFile = typeof filePath === 'string' && filePath.startsWith('gs://')
  const originalFileUrl = hasOriginalFile && selectedDocumentId
    ? `${API_BASE}/documents/${selectedDocumentId}/file`
    : undefined
  const markdownDownloadUrl = selectedDocumentId
    ? `${API_BASE}/documents/${selectedDocumentId}/download/markdown`
    : undefined

  /* ─── Document selector (shown when nothing is selected) ─── */
  if (!selectedDocumentId) {
    return (
      <div className="dv-root" style={{ maxWidth: 1400, margin: '0 auto' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
          <label htmlFor="dv-doc-select" style={{ fontWeight: 500, fontSize: '0.9375rem' }}>
            Select Document:
          </label>
          <select
            id="dv-doc-select"
            value=""
            onChange={(e) => {
              const docId = e.target.value
              setSelectedDocumentId(docId)
              onDocumentSelect?.(docId)
            }}
            style={{
              flex: 1,
              maxWidth: 400,
              padding: '0.625rem 1rem',
              borderRadius: 8,
              border: '1px solid var(--border-light, #e5e7eb)',
              fontSize: '0.9375rem',
            }}
          >
            <option value="">-- Choose a document --</option>
            {documents.map((doc) => (
              <option key={doc.id} value={doc.id}>
                {doc.display_name?.trim() || doc.filename}
              </option>
            ))}
          </select>
        </div>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: 400,
          background: 'var(--bg-card, #fff)',
          border: '1px solid var(--border-light, #e5e7eb)',
          borderRadius: 12,
          color: 'var(--text-tertiary, #9ca3af)',
        }}>
          <p>Select a document to view its pages</p>
        </div>
      </div>
    )
  }

  return (
    <div>
      {/* Document selector header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
        <label htmlFor="dv-doc-select-active" style={{ fontWeight: 500, fontSize: '0.9375rem' }}>
          Document:
        </label>
        <select
          id="dv-doc-select-active"
          value={selectedDocumentId}
          onChange={(e) => {
            const docId = e.target.value
            setSelectedDocumentId(docId)
            setPages([])
            setFactsRaw([])
            setTagsRaw([])
            onDocumentSelect?.(docId)
          }}
          style={{
            flex: 1,
            maxWidth: 400,
            padding: '0.5rem 0.75rem',
            borderRadius: 8,
            border: '1px solid var(--border-light, #e5e7eb)',
            fontSize: '0.875rem',
          }}
        >
          {documents.map((doc) => (
            <option key={doc.id} value={doc.id}>
              {doc.display_name?.trim() || doc.filename}
            </option>
          ))}
        </select>
      </div>

      <DocumentViewer
        documentId={selectedDocumentId}
        pages={pages}
        loading={loading}
        highlights={highlights}
        initialPage={navigateToRead?.pageNumber}
        navigateTo={
          navigateToRead && navigateToRead.documentId === selectedDocumentId
            ? { pageNumber: navigateToRead.pageNumber, highlightId: navigateToRead.factId }
            : null
        }
        onNavigateConsumed={onNavigateToReadConsumed}
        hasOriginalFile={hasOriginalFile}
        originalFileUrl={originalFileUrl}
        markdownDownloadUrl={markdownDownloadUrl}
        interaction={interaction}
        onPageChange={() => {}}
      />
    </div>
  )
}
