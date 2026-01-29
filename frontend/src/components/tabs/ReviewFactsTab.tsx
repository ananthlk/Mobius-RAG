import { useState, useEffect } from 'react'
import { approveFactApi, deleteFactApi, patchFactApi, rejectFactApi } from '../../lib/factActions'
import { getStateLabel } from '../../lib/documentMetadata'
import './ReviewFactsTab.css'

// Category keys and labels (match backend CATEGORY_NAMES) for edit modal importance
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

interface Fact {
  id: string
  fact_text: string
  fact_type: string | null
  who_eligible: string | null
  how_verified: string | null
  conflict_resolution: string | null
  when_applies: string | null
  limitations: string | null
  is_pertinent_to_claims_or_members: string | null
  is_eligibility_related: string | null
  category_scores: Record<string, { score: number; direction: number | null }> | null
  document_id: string
  document_filename?: string
  document_display_name?: string | null
  payer?: string
  state?: string
  program?: string
  effective_date?: string | null
  termination_date?: string | null
  page_number?: number | null
  is_verified?: string | null
  confidence?: string | number | null
  verification_status?: string | null
  verified_by?: string | null
  verified_at?: string | null
}

interface ReviewFactsTabProps {
  onViewDocument?: (documentId: string, options?: { pageNumber?: number; factId?: string }) => void
}

/** Document metadata from GET /documents for filter options and fact enrichment. */
interface DocumentMeta {
  id: string
  filename: string
  display_name?: string | null
  payer?: string | null
  state?: string | null
  program?: string | null
  effective_date?: string | null
  termination_date?: string | null
}

export function ReviewFactsTab({ onViewDocument }: ReviewFactsTabProps) {
  const [facts, setFacts] = useState<Fact[]>([])
  const [documents, setDocuments] = useState<DocumentMeta[]>([])
  const [loading, setLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedFact, setSelectedFact] = useState<Fact | null>(null)
  
  // Filters
  const [selectedPayers, setSelectedPayers] = useState<string[]>([])
  const [selectedStates, setSelectedStates] = useState<string[]>([])
  const [selectedPrograms, setSelectedPrograms] = useState<string[]>([])
  const [categoryThresholds, setCategoryThresholds] = useState<Record<string, number>>({})
  const [selectedFactTypes, setSelectedFactTypes] = useState<string[]>([])
  const [isPertinentFilter, setIsPertinentFilter] = useState<'all' | 'yes' | 'no'>('all')
  const [isEligibilityFilter, setIsEligibilityFilter] = useState<'all' | 'yes' | 'no'>('all')
  const [approvalFilter, setApprovalFilter] = useState<'all' | 'pending' | 'approved' | 'rejected'>('all')
  const [filtersOpen, setFiltersOpen] = useState(false)

  // Actions dropdown (one card at a time)
  const [actionsOpenFactId, setActionsOpenFactId] = useState<string | null>(null)

  // Edit modal
  const [editingFact, setEditingFact] = useState<Fact | null>(null)
  const [editFactText, setEditFactText] = useState('')
  const [editPertinent, setEditPertinent] = useState(true)
  const [editCategoryScores, setEditCategoryScores] = useState<Record<string, { score: number; direction: number }>>({})
  const [editSaving, setEditSaving] = useState(false)
  const [editError, setEditError] = useState<string | null>(null)

  const API_BASE = 'http://localhost:8000'

  const closeActionsMenu = () => setActionsOpenFactId(null)

  /** Shared helper: approve fact via API and update local state. */
  const handleApprove = async (fact: Fact) => {
    try {
      const res = await approveFactApi(fact.document_id, fact.id, API_BASE)
      if (res.ok) {
        const updated = await res.json()
        setFacts(prev =>
          prev.map(f =>
            f.id === fact.id
              ? { ...f, verification_status: 'approved', verified_by: updated.verified_by, verified_at: updated.verified_at }
              : f
          )
        )
        if (selectedFact?.id === fact.id) {
          setSelectedFact(prev => (prev?.id === fact.id ? { ...prev, verification_status: 'approved', verified_by: updated.verified_by, verified_at: updated.verified_at } : prev))
        }
      }
    } catch (e) {
      console.error('Failed to approve fact:', e)
    }
  }

  /** Shared helper: reject fact via API and update local state. */
  const handleReject = async (fact: Fact) => {
    try {
      const res = await rejectFactApi(fact.document_id, fact.id, API_BASE)
      if (res.ok) {
        const updated = await res.json()
        setFacts(prev =>
          prev.map(f =>
            f.id === fact.id
              ? { ...f, verification_status: 'rejected', verified_by: updated.verified_by, verified_at: updated.verified_at }
              : f
          )
        )
        if (selectedFact?.id === fact.id) {
          setSelectedFact(prev => (prev?.id === fact.id ? { ...prev, verification_status: 'rejected', verified_by: updated.verified_by, verified_at: updated.verified_at } : prev))
        }
      }
    } catch (e) {
      console.error('Failed to reject fact:', e)
    }
  }

  /** Shared helper: delete fact via API and update local state. */
  const handleDelete = async (fact: Fact, e: React.MouseEvent) => {
    e.stopPropagation()
    if (!confirm('Delete this fact? This cannot be undone.')) return
    try {
      const res = await deleteFactApi(fact.document_id, fact.id, API_BASE)
      if (res.ok) {
        setFacts(prev => prev.filter(f => f.id !== fact.id))
        if (selectedFact?.id === fact.id) setSelectedFact(null)
        if (editingFact?.id === fact.id) setEditingFact(null)
      } else {
        const err = await res.json().catch(() => ({}))
        alert(err.detail || 'Failed to delete fact')
      }
    } catch (e) {
      console.error('Failed to delete fact:', e)
      alert('Failed to delete fact')
    }
  }

  /** One-click toggle pertinence (is_pertinent_to_claims_or_members). */
  const handleTogglePertinent = async (fact: Fact, e: React.MouseEvent) => {
    e.stopPropagation()
    const next = fact.is_pertinent_to_claims_or_members !== 'true'
    try {
      const res = await patchFactApi(fact.document_id, fact.id, { is_pertinent_to_claims_or_members: next }, API_BASE)
      if (res.ok) {
        const val = next ? 'true' : 'false'
        setFacts(prev => prev.map(f => (f.id === fact.id ? { ...f, is_pertinent_to_claims_or_members: val } : f)))
        if (selectedFact?.id === fact.id) {
          setSelectedFact(prev => (prev?.id === fact.id ? { ...prev, is_pertinent_to_claims_or_members: val } : prev))
        }
        if (editingFact?.id === fact.id) {
          setEditPertinent(next)
        }
      }
    } catch (err) {
      console.error('Failed to toggle pertinence:', err)
    }
  }

  /** Fetch GET /documents and return doc list + map by id (string) for merging into facts. */
  const fetchDocumentsMap = async (): Promise<{ list: DocumentMeta[]; map: Record<string, DocumentMeta> }> => {
    const res = await fetch(`${API_BASE}/documents`)
    if (!res.ok) return { list: [], map: {} }
    const json = await res.json()
    const raw = Array.isArray(json) ? json : (json.documents || [])
    const list: DocumentMeta[] = []
    const map: Record<string, DocumentMeta> = {}
    for (const d of raw) {
      const id = d.id ?? d.document_id
      if (id) {
        const sid = String(id)
        const meta: DocumentMeta = {
          id: sid,
          filename: d.filename || d.name || 'Unknown Document',
          display_name: d.display_name ?? null,
          payer: d.payer ?? null,
          state: d.state ?? null,
          program: d.program ?? null,
          effective_date: d.effective_date ?? null,
          termination_date: d.termination_date ?? null,
        }
        list.push(meta)
        map[sid] = meta
      }
    }
    return { list, map }
  }

  /** Merge document metadata (GET /documents) into facts so display name, filename, payer/state/program, and dates are up to date. */
  const mergeDocumentMetadataIntoFacts = (
    factsList: Fact[],
    docMap: Record<string, DocumentMeta>
  ): Fact[] => {
    return factsList.map((f) => {
      const meta = docMap[String(f.document_id)]
      if (!meta) return f
      const displayName = (meta.display_name && meta.display_name.trim()) ? meta.display_name.trim() : null
      return {
        ...f,
        document_filename: meta.filename || f.document_filename,
        document_display_name: displayName ?? f.document_display_name ?? null,
        payer: meta.payer ?? f.payer,
        state: meta.state ?? f.state,
        program: meta.program ?? f.program,
        effective_date: meta.effective_date ?? f.effective_date,
        termination_date: meta.termination_date ?? f.termination_date,
      }
    })
  }

  const loadFacts = async () => {
    setLoading(true)
    try {
      const { list: docList, map: docMap } = await fetchDocumentsMap()
      setDocuments(docList)

      // 1) Try admin table endpoint first
      const response = await fetch(`${API_BASE}/admin/db/tables/extracted_facts/records?limit=1000`)
      if (response.ok) {
        const data = await response.json()
        const records = data.records || []
        if (records.length > 0) {
          const factsData = records.map((record: any) => ({
            id: record.id,
            fact_text: record.fact_text,
            fact_type: record.fact_type,
            who_eligible: record.who_eligible,
            how_verified: record.how_verified,
            conflict_resolution: record.conflict_resolution,
            when_applies: record.when_applies,
            limitations: record.limitations,
            is_pertinent_to_claims_or_members: record.is_pertinent_to_claims_or_members,
            is_eligibility_related: record.is_eligibility_related,
            is_verified: record.is_verified,
            confidence: record.confidence,
            category_scores: record.category_scores ?? null,
            document_id: record.document_id,
            page_number: record.page_number != null ? Number(record.page_number) : null,
            verification_status: record.verification_status ?? null,
            verified_by: record.verified_by ?? null,
            verified_at: record.verified_at ?? null,
          }))
          const factsWithIds = factsData.map((f: Fact) => ({ ...f, document_id: String(f.document_id) }))
          const enriched = mergeDocumentMetadataIntoFacts(factsWithIds, docMap)
          setFacts(enriched)
          return
        }
      }

      // 2) Fallback: load by document (normalized facts from GET /documents/:id/facts)
      const allFacts: Fact[] = []
      for (const doc of docList) {
        const id = doc.id
        try {
          const factsResponse = await fetch(`${API_BASE}/documents/${id}/facts`)
          if (!factsResponse.ok) continue
          const factsData = await factsResponse.json()
          const chunks = factsData.chunks || []
          const displayName = (doc.display_name && doc.display_name.trim()) ? doc.display_name.trim() : null
          const filename = doc.filename || 'Unknown Document'
          for (const chunk of chunks) {
            for (const f of chunk.facts || []) {
              allFacts.push({
                id: f.id,
                fact_text: f.fact_text,
                fact_type: f.fact_type ?? null,
                who_eligible: f.who_eligible ?? null,
                how_verified: f.how_verified ?? null,
                conflict_resolution: f.conflict_resolution ?? null,
                when_applies: f.when_applies ?? null,
                limitations: f.limitations ?? null,
                is_pertinent_to_claims_or_members: f.is_pertinent_to_claims_or_members ?? null,
                is_eligibility_related: f.is_eligibility_related ?? null,
                category_scores: f.category_scores ?? null,
                document_id: id,
                document_filename: filename,
                document_display_name: displayName ?? null,
                payer: doc.payer ?? undefined,
                state: doc.state ?? undefined,
                program: doc.program ?? undefined,
                effective_date: doc.effective_date ?? undefined,
                termination_date: doc.termination_date ?? undefined,
                page_number: f.page_number != null ? Number(f.page_number) : null,
                verification_status: f.verification_status ?? null,
                verified_by: f.verified_by ?? null,
                verified_at: f.verified_at ?? null,
              })
            }
          }
        } catch (err) {
          console.error('Failed to load facts for document', id, err)
        }
      }
      setFacts(allFacts)
    } catch (err) {
      console.error('Failed to load facts:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadFacts()
  }, [])

  useEffect(() => {
    if (!filtersOpen) return
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setFiltersOpen(false)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [filtersOpen])

  const toNum = (x: number | string | null | undefined): number | null => {
    if (x == null) return null
    if (typeof x === 'number') return isNaN(x) ? null : x
    const n = parseFloat(String(x))
    return isNaN(n) ? null : n
  }

  const getTopCategories = (categoryScores: Record<string, { score: number | null; direction: number | null }> | null) => {
    if (!categoryScores) return []
    return Object.entries(categoryScores)
      .filter(([_, data]) => (toNum(data?.score) ?? 0) > 0)
      .sort(([_, a], [__, b]) => (toNum(b?.score) ?? 0) - (toNum(a?.score) ?? 0))
      .slice(0, 5)
      .map(([category, data]) => ({ category, score: toNum(data?.score) ?? 0, direction: data?.direction ?? null }))
  }

  const filteredFacts = facts.filter(fact => {
    // Text search
    if (searchQuery && !fact.fact_text.toLowerCase().includes(searchQuery.toLowerCase())) {
      return false
    }
    
    // Payer filter
    if (selectedPayers.length > 0 && (!fact.payer || !selectedPayers.includes(fact.payer))) {
      return false
    }
    
    // State filter
    if (selectedStates.length > 0 && (!fact.state || !selectedStates.includes(fact.state))) {
      return false
    }
    
    // Program filter
    if (selectedPrograms.length > 0 && (!fact.program || !selectedPrograms.includes(fact.program))) {
      return false
    }
    
    // Fact type filter
    if (selectedFactTypes.length > 0 && (!fact.fact_type || !selectedFactTypes.includes(fact.fact_type))) {
      return false
    }
    
    // Category filter - check if all category thresholds are met (AND logic)
    // If a category has a threshold set, the fact must have that category with score >= threshold
    if (Object.keys(categoryThresholds).length > 0) {
      const allThresholdsMet = Object.entries(categoryThresholds).every(([category, threshold]) => {
        if (threshold === 0) return true // Threshold of 0 means "show all" for this category
        if (!fact.category_scores || typeof fact.category_scores !== 'object') return false
        const categoryData = fact.category_scores[category]
        if (!categoryData || typeof categoryData !== 'object') return false
        const score = toNum(categoryData.score)
        if (score == null) return false
        return score >= threshold
      })
      if (!allThresholdsMet) {
        return false
      }
    }
    
    // is_pertinent filter
    if (isPertinentFilter !== 'all') {
      const isPertinent = fact.is_pertinent_to_claims_or_members === 'true'
      if ((isPertinentFilter === 'yes' && !isPertinent) || (isPertinentFilter === 'no' && isPertinent)) {
        return false
      }
    }
    
    // is_eligibility_related filter
    if (isEligibilityFilter !== 'all') {
      const isEligible = fact.is_eligibility_related === 'true'
      if ((isEligibilityFilter === 'yes' && !isEligible) || (isEligibilityFilter === 'no' && isEligible)) {
        return false
      }
    }

    // Approval / verification status filter
    if (approvalFilter !== 'all') {
      const status = fact.verification_status || 'pending'
      if (status !== approvalFilter) return false
    }

    return true
  })

  // Filter options from document metadata (so dropdowns stay populated after metadata updates)
  const uniquePayers = Array.from(new Set(documents.map(d => d.payer).filter(Boolean))).sort() as string[]
  const uniqueStates = Array.from(new Set(documents.map(d => d.state).filter(Boolean))).sort() as string[]
  const uniquePrograms = Array.from(new Set(documents.map(d => d.program).filter(Boolean))).sort() as string[]
  const uniqueFactTypes = Array.from(new Set(facts.map(f => f.fact_type).filter(Boolean))) as string[]

  const categories = [
    'contacting_marketing_members',
    'member_eligibility_molina',
    'benefit_access_limitations',
    'prior_authorization_required',
    'claims_authorization_submissions',
    'compliant_claim_requirements',
    'claim_disputes',
    'credentialing',
    'claim_submission_important',
    'coordination_of_benefits',
    'other_important',
  ]

  const activeFiltersCount =
    (searchQuery.trim() ? 1 : 0) +
    (selectedPayers.length || 0) +
    (selectedStates.length || 0) +
    (selectedPrograms.length || 0) +
    (selectedFactTypes.length || 0) +
    (isPertinentFilter !== 'all' ? 1 : 0) +
    (isEligibilityFilter !== 'all' ? 1 : 0) +
    (approvalFilter !== 'all' ? 1 : 0) +
    (Object.keys(categoryThresholds).filter(k => (categoryThresholds[k] || 0) > 0).length)

  const clearFilters = () => {
    setSelectedPayers([])
    setSelectedStates([])
    setSelectedPrograms([])
    setCategoryThresholds({})
    setSelectedFactTypes([])
    setIsPertinentFilter('all')
    setIsEligibilityFilter('all')
    setApprovalFilter('all')
    setSearchQuery('')
  }

  const openEditModal = (fact: Fact, e: React.MouseEvent) => {
    e.stopPropagation()
    setEditingFact(fact)
    setEditFactText(fact.fact_text)
    setEditPertinent(fact.is_pertinent_to_claims_or_members === 'true')
    const scores: Record<string, { score: number; direction: number }> = {}
    for (const { key } of CATEGORIES) {
      const data = fact.category_scores?.[key]
      const score = typeof data?.score === 'number' ? data.score : 0
      const direction = typeof data?.direction === 'number' ? data.direction : 0.5
      scores[key] = { score, direction }
    }
    setEditCategoryScores(scores)
    setEditError(null)
  }

  const closeEditModal = () => {
    setEditingFact(null)
    setEditError(null)
  }

  const handleEditCategoryScoreChange = (key: string, score: number) => {
    setEditCategoryScores(prev => ({
      ...prev,
      [key]: { score, direction: prev[key]?.direction ?? 0.5 },
    }))
  }

  const saveEditFact = async () => {
    if (!editingFact || !editFactText.trim()) return
    setEditSaving(true)
    setEditError(null)
    try {
      const category_scores: Record<string, { score: number; direction: number }> = {}
      for (const { key } of CATEGORIES) {
        const entry = editCategoryScores[key]
        if (entry && entry.score > 0) {
          category_scores[key] = { score: entry.score, direction: entry.direction ?? 0.5 }
        }
      }
      const res = await patchFactApi(
        editingFact.document_id,
        editingFact.id,
        {
          fact_text: editFactText.trim(),
          is_pertinent_to_claims_or_members: editPertinent,
          category_scores,
        },
        API_BASE
      )
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        setEditError(err.detail || 'Failed to update fact')
        return
      }
      const updatedPertinent = editPertinent ? 'true' : 'false'
      const updatedScores: Record<string, { score: number; direction: number | null }> = {}
      for (const { key } of CATEGORIES) {
        const v = editCategoryScores[key] ?? { score: 0, direction: 0.5 }
        updatedScores[key] = { score: v.score, direction: v.direction }
      }
      setFacts(prev =>
        prev.map(f =>
          f.id === editingFact.id
            ? { ...f, fact_text: editFactText.trim(), is_pertinent_to_claims_or_members: updatedPertinent, category_scores: updatedScores }
            : f
        )
      )
      if (selectedFact?.id === editingFact.id) {
        setSelectedFact(prev =>
          prev?.id === editingFact.id
            ? { ...prev, fact_text: editFactText.trim(), is_pertinent_to_claims_or_members: updatedPertinent, category_scores: updatedScores }
            : prev
        )
      }
      closeEditModal()
    } catch (e) {
      setEditError('Failed to update fact')
    } finally {
      setEditSaving(false)
    }
  }

  // Consolidated action handlers for card and expanded (consistent with Read Document context menu)
  const onApprove = (fact: Fact) => (e: React.MouseEvent) => {
    e.stopPropagation()
    handleApprove(fact)
  }
  const onReject = (fact: Fact) => (e: React.MouseEvent) => {
    e.stopPropagation()
    handleReject(fact)
  }
  const onEdit = (fact: Fact) => (e: React.MouseEvent) => openEditModal(fact, e)
  const onDelete = (fact: Fact) => (e: React.MouseEvent) => handleDelete(fact, e)

  return (
    <div className="review-facts-tab">
      {/* Backdrop: click outside to close filters panel */}
      {filtersOpen && (
        <div
          className="filters-float-backdrop"
          onClick={() => setFiltersOpen(false)}
          aria-hidden
        />
      )}
      {/* Floating filters panel (toggle to open) */}
      <div className={`filters-float ${filtersOpen ? 'filters-float-open' : ''}`}>
        <div className="filters-float-inner">
          <div className="filters-header">
            <h3 className="filters-title">Filters</h3>
            <div className="filters-header-actions">
              <button onClick={clearFilters} className="clear-filters-btn">
                Clear All
              </button>
              <button type="button" className="filters-close-btn" onClick={() => setFiltersOpen(false)} aria-label="Close filters">
                ×
              </button>
            </div>
          </div>

          <div className="filters-content">
            {/* Search is in the header toolbar */}

            {/* Document Metadata Filters (autopopulated from documents) */}
            <div className="filter-group">
              <label className="filter-label">Payer</label>
              <select
                multiple
                size={Math.min(6, uniquePayers.length + 1)}
                value={selectedPayers}
                onChange={(e) => {
                  const selected = Array.from(e.target.selectedOptions, (o) => o.value)
                  setSelectedPayers(selected)
                }}
                className="filter-select"
              >
                {uniquePayers.map((payer) => (
                  <option key={payer} value={payer}>{payer}</option>
                ))}
              </select>
              {uniquePayers.length === 0 && (
                <span className="filter-hint">No payers in document metadata yet.</span>
              )}
            </div>

            <div className="filter-group">
              <label className="filter-label">State</label>
              <select
                multiple
                size={Math.min(6, uniqueStates.length + 1)}
                value={selectedStates}
                onChange={(e) => {
                  const selected = Array.from(e.target.selectedOptions, (o) => o.value)
                  setSelectedStates(selected)
                }}
                className="filter-select"
              >
                {uniqueStates.map((state) => (
                  <option key={state} value={state}>{getStateLabel(state) || state}</option>
                ))}
              </select>
              {uniqueStates.length === 0 && (
                <span className="filter-hint">No states in document metadata yet.</span>
              )}
            </div>

            <div className="filter-group">
              <label className="filter-label">Program</label>
              <select
                multiple
                size={Math.min(6, uniquePrograms.length + 1)}
                value={selectedPrograms}
                onChange={(e) => {
                  const selected = Array.from(e.target.selectedOptions, (o) => o.value)
                  setSelectedPrograms(selected)
                }}
                className="filter-select"
              >
                {uniquePrograms.map((program) => (
                  <option key={program} value={program}>{program}</option>
                ))}
              </select>
              {uniquePrograms.length === 0 && (
                <span className="filter-hint">No programs in document metadata yet.</span>
              )}
            </div>

            {/* Category Filters - Sliders */}
            <div className="filter-group">
              <label className="filter-label">Categories (Score Threshold)</label>
              <div className="category-sliders">
                {categories.map((category) => {
                  const threshold = categoryThresholds[category] || 0
                  const displayName = category.replace(/_/g, ' ')
                  return (
                    <div key={category} className="category-slider-item">
                      <div className="category-slider-header">
                        <label className="category-slider-label">{displayName}</label>
                        <span className="category-slider-value">{threshold.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={threshold}
                        onChange={(e) => {
                          const newValue = parseFloat(e.target.value)
                          if (newValue === 0) {
                            // Remove from thresholds if set to 0
                            const newThresholds = { ...categoryThresholds }
                            delete newThresholds[category]
                            setCategoryThresholds(newThresholds)
                          } else {
                            setCategoryThresholds({
                              ...categoryThresholds,
                              [category]: newValue
                            })
                          }
                        }}
                        className="category-slider"
                      />
                      <div className="category-slider-labels">
                        <span>0</span>
                        <span>1</span>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Fact Type Filters */}
            <div className="filter-group">
              <label className="filter-label">Fact Type</label>
              <div className="filter-checkboxes">
                {uniqueFactTypes.map((type) => (
                  <label key={type} className="filter-checkbox">
                    <input
                      type="checkbox"
                      checked={selectedFactTypes.includes(type)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedFactTypes([...selectedFactTypes, type])
                        } else {
                          setSelectedFactTypes(selectedFactTypes.filter(t => t !== type))
                        }
                      }}
                    />
                    <span>{type}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Boolean Filters */}
            <div className="filter-group">
              <label className="filter-label">Pertinent to Claims/Members</label>
              <select
                value={isPertinentFilter}
                onChange={(e) => setIsPertinentFilter(e.target.value as 'all' | 'yes' | 'no')}
                className="filter-select"
              >
                <option value="all">All</option>
                <option value="yes">Yes</option>
                <option value="no">No</option>
              </select>
            </div>

            <div className="filter-group">
              <label className="filter-label">Eligibility Related</label>
              <select
                value={isEligibilityFilter}
                onChange={(e) => setIsEligibilityFilter(e.target.value as 'all' | 'yes' | 'no')}
                className="filter-select"
              >
                <option value="all">All</option>
                <option value="yes">Yes</option>
                <option value="no">No</option>
              </select>
            </div>

            <div className="filter-group">
              <label className="filter-label">Approval status</label>
              <select
                value={approvalFilter}
                onChange={(e) => setApprovalFilter(e.target.value as 'all' | 'pending' | 'approved' | 'rejected')}
                className="filter-select"
              >
                <option value="all">All</option>
                <option value="pending">Pending</option>
                <option value="approved">Approved</option>
                <option value="rejected">Rejected</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Full-width main area - Fact Cards */}
      <div className="facts-main">
        <header className="facts-toolbar">
          <div className="facts-toolbar-left">
            <h2 className="facts-toolbar-title">Facts</h2>
            <span className="facts-toolbar-count" title={`${filteredFacts.length} of ${facts.length} facts`}>
              {filteredFacts.length}{facts.length !== filteredFacts.length ? ` / ${facts.length}` : ''}
            </span>
            {activeFiltersCount > 0 && (
              <span className="facts-toolbar-filters-badge" title={`${activeFiltersCount} filter(s) active`}>
                {activeFiltersCount}
              </span>
            )}
          </div>
          <div className="facts-toolbar-search">
            <input
              type="search"
              placeholder="Search facts..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="facts-toolbar-search-input"
              aria-label="Search facts"
            />
          </div>
          <div className="facts-toolbar-actions">
            <button
              type="button"
              className={`facts-toolbar-btn ${filtersOpen ? 'facts-toolbar-btn-active' : ''}`}
              onClick={() => setFiltersOpen(prev => !prev)}
              aria-label={filtersOpen ? 'Close filters' : 'Open filters'}
              title={filtersOpen ? 'Close filters' : 'Open filters panel'}
            >
              Filters{activeFiltersCount > 0 ? ` (${activeFiltersCount})` : ''}{filtersOpen ? ' ▲' : ' ▼'}
            </button>
            <button
              type="button"
              className="facts-toolbar-btn"
              onClick={loadFacts}
              disabled={loading}
              title="Reload facts from server"
              aria-label="Refresh"
            >
              {loading ? '…' : '↻ Refresh'}
            </button>
          </div>
        </header>

          {loading ? (
            <div className="loading-facts">Loading facts...</div>
          ) : filteredFacts.length === 0 ? (
            <div className="no-facts">
              {facts.length === 0
                ? 'No facts yet. Run chunking on a document (Document Status or Live Updates), then click Refresh.'
                : 'No facts found matching your filters'}
            </div>
          ) : (
            <div className="facts-grid">
              {filteredFacts.map((fact) => {
                const topCategories = getTopCategories(fact.category_scores)
                const isPertinent = fact.is_pertinent_to_claims_or_members === 'true'
                const isEligible = fact.is_eligibility_related === 'true'
                const isExpanded = selectedFact?.id === fact.id
                
                const actionsOpen = actionsOpenFactId === fact.id

                return (
                  <div
                    key={fact.id}
                    className={`fact-card ${isExpanded ? 'expanded' : ''}`}
                    onClick={() => { setSelectedFact(isExpanded ? null : fact); closeActionsMenu() }}
                  >
                    <div className="fact-card-content">
                      <div className="fact-card-header">
                        <div className="fact-card-header-main">
                          <span className="fact-document-name" title={fact.document_display_name && fact.document_filename ? `File: ${fact.document_filename}` : undefined}>
                            {fact.document_display_name?.trim() || fact.document_filename || 'Unknown Document'}
                          </span>
                        </div>
                        <div className="fact-card-actions-wrap" onClick={e => e.stopPropagation()}>
                          <button
                            type="button"
                            className="fact-card-menu-btn"
                            onClick={(e) => { e.stopPropagation(); setActionsOpenFactId(actionsOpen ? null : fact.id) }}
                            aria-expanded={actionsOpen}
                            aria-haspopup="true"
                            title="Actions"
                          >
                            <span className="fact-card-menu-dots" aria-hidden>⋮</span>
                          </button>
                          {actionsOpen && (
                            <>
                              <div className="fact-card-menu-backdrop" onClick={closeActionsMenu} aria-hidden />
                              <div className="fact-card-menu-dropdown" role="menu">
                                {fact.verification_status === 'approved' ? (
                                  <button type="button" className="fact-card-menu-item" disabled>Approved</button>
                                ) : (
                                  <button type="button" className="fact-card-menu-item" onClick={() => { onApprove(fact)({} as React.MouseEvent); closeActionsMenu() }} role="menuitem">Approve</button>
                                )}
                                {fact.verification_status === 'rejected' ? (
                                  <button type="button" className="fact-card-menu-item" disabled>Rejected</button>
                                ) : (
                                  <button type="button" className="fact-card-menu-item" onClick={() => { onReject(fact)({} as React.MouseEvent); closeActionsMenu() }} role="menuitem">Reject</button>
                                )}
                                <button type="button" className="fact-card-menu-item" onClick={(e) => { onEdit(fact)(e); closeActionsMenu() }} role="menuitem">Edit</button>
                                <button type="button" className="fact-card-menu-item fact-card-menu-item-danger" onClick={(e) => { onDelete(fact)(e); closeActionsMenu() }} role="menuitem">Delete</button>
                                {onViewDocument && (
                                  <button
                                    type="button"
                                    className="fact-card-menu-item fact-card-menu-item-primary"
                                    onClick={() => {
                                      onViewDocument(fact.document_id, { pageNumber: fact.page_number ?? undefined, factId: fact.id })
                                      closeActionsMenu()
                                    }}
                                    role="menuitem"
                                  >
                                    View in Document
                                  </button>
                                )}
                              </div>
                            </>
                          )}
                        </div>
                      </div>

                      <div className="fact-text">
                        {isExpanded
                          ? fact.fact_text
                          : fact.fact_text.length > 180
                            ? `${fact.fact_text.substring(0, 180)}…`
                            : fact.fact_text}
                      </div>

                      <div className="fact-card-footer">
                        <div className="fact-status-group">
                          <button
                            type="button"
                            className={`pertinent-badge pertinent-badge-clickable ${isPertinent ? 'yes' : 'no'}`}
                            onClick={e => handleTogglePertinent(fact, e)}
                            title={isPertinent ? 'Mark not pertinent' : 'Mark pertinent'}
                          >
                            {isPertinent ? 'Pertinent' : 'Not pertinent'}
                          </button>
                          <button
                            type="button"
                            className={`fact-verification-badge fact-verification-badge-btn fact-verification-${fact.verification_status || 'pending'}`}
                            onClick={fact.verification_status === 'pending' || !fact.verification_status ? (e) => { e.stopPropagation(); handleApprove(fact) } : undefined}
                            disabled={fact.verification_status === 'approved' || fact.verification_status === 'rejected'}
                            title={fact.verification_status === 'pending' || !fact.verification_status ? 'Click to approve' : undefined}
                          >
                            {fact.verification_status === 'approved' ? 'Approved' : fact.verification_status === 'rejected' ? 'Rejected' : 'Pending'}
                          </button>
                          {isEligible && <span className="eligibility-badge">Eligibility related</span>}
                        </div>
                        {topCategories.length > 0 && (
                          <div className="fact-categories">
                            {topCategories.slice(0, isExpanded ? 10 : 3).map((cat) => {
                              const tier = (cat.score >= 0.7) ? 'high' : (cat.score >= 0.4) ? 'medium' : 'low'
                              return (
                                <span key={cat.category} className={`category-tag category-tag-${tier}`} title={`Relevance: ${(cat.score * 100).toFixed(0)}%`}>
                                  {cat.category.replace(/_/g, ' ')}
                                </span>
                              )
                            })}
                          </div>
                        )}
                        <span className="fact-card-footer-spacer" />
                        <div className="fact-card-footer-line">
                          {(fact.payer || fact.state || fact.program) && (
                            <span className="fact-metadata-inline">
                              {[fact.payer, fact.state, fact.program].filter(Boolean).join(' · ')}
                            </span>
                          )}
                          {(fact.effective_date || fact.termination_date) && (
                            <span className="fact-dates-inline">
                              Exp: {fact.effective_date || '—'}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                    
                    {/* Expanded Details */}
                    {isExpanded && (
                      <div className="fact-expanded-details">
                        {fact.fact_type && (
                          <div className="fact-detail-section">
                            <label>Classification:</label>
                            <p>{fact.fact_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</p>
                          </div>
                        )}
                        <div className="fact-detail-section">
                          <label>Full Fact Text:</label>
                          <p className="fact-full-text">{fact.fact_text}</p>
                        </div>
                        
                        {fact.who_eligible && (
                          <div className="fact-detail-section">
                            <label>Who Eligible:</label>
                            <p>{fact.who_eligible}</p>
                          </div>
                        )}
                        
                        {fact.how_verified && (
                          <div className="fact-detail-section">
                            <label>How Verified:</label>
                            <p>{fact.how_verified}</p>
                          </div>
                        )}
                        
                        {fact.conflict_resolution && (
                          <div className="fact-detail-section">
                            <label>Conflict Resolution:</label>
                            <p>{fact.conflict_resolution}</p>
                          </div>
                        )}
                        
                        {fact.when_applies && (
                          <div className="fact-detail-section">
                            <label>When Applies:</label>
                            <p>{fact.when_applies}</p>
                          </div>
                        )}
                        
                        {fact.limitations && (
                          <div className="fact-detail-section">
                            <label>Limitations:</label>
                            <p>{fact.limitations}</p>
                          </div>
                        )}
                        
                        {(fact.is_verified != null && fact.is_verified !== '') && (
                          <div className="fact-detail-section">
                            <label>Is Verified:</label>
                            <p>{fact.is_verified === 'true' ? 'Yes' : fact.is_verified === 'false' ? 'No' : fact.is_verified}</p>
                          </div>
                        )}
                        {(fact.confidence != null && fact.confidence !== '') && (
                          <div className="fact-detail-section">
                            <label>Confidence:</label>
                            <p>{fact.confidence}</p>
                          </div>
                        )}
                        
                        <div className="fact-detail-section">
                          <label>Category Scores (all):</label>
                          <div className="category-scores-grid">
                            {fact.category_scores && Object.keys(fact.category_scores).length > 0
                              ? Object.entries(fact.category_scores)
                                  .sort(([_, a], [__, b]) => (typeof b.score === 'number' ? b.score : 0) - (typeof a.score === 'number' ? a.score : 0))
                                  .map(([category, data]: [string, { score: number | string | null; direction?: number | null }]) => {
                                    const scoreVal = toNum(data?.score)
                                    return (
                                    <div key={category} className="category-score-card">
                                      <div className="category-score-header">
                                        <span className="category-name">{category.replace(/_/g, ' ')}</span>
                                        <span className="category-score-value">
                                          {scoreVal != null ? scoreVal.toFixed(2) : '—'}
                                        </span>
                                      </div>
                                      {data?.direction !== null && data?.direction !== undefined && (
                                        <div className="category-direction">
                                          {data.direction === 1.0 ? 'Encourages' : data.direction === 0.5 ? 'Neutral' : data.direction === 0 ? 'Restricts' : '—'}
                                        </div>
                                      )}
                                    </div>
                                  )})
                              : <p className="fact-muted">No category scores for this fact.</p>
                            }
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>

      {/* Edit fact modal */}
      {editingFact && (
        <div className="fact-card-modal-overlay" onClick={closeEditModal}>
          <div className="fact-card-modal" onClick={e => e.stopPropagation()}>
            <h3 className="fact-card-modal-title">Edit fact</h3>
            <div className="fact-card-modal-body">
              <p className="fact-card-modal-doc-name" title={editingFact?.document_display_name && editingFact?.document_filename ? `File: ${editingFact.document_filename}` : undefined}>
                {editingFact?.document_display_name?.trim() || editingFact?.document_filename || 'Unknown Document'}
                {(editingFact?.effective_date || editingFact?.termination_date) && (
                  <span className="fact-document-dates">
                    {' '}(Effective: {editingFact?.effective_date || '—'} – Termination: {editingFact?.termination_date || '—'})
                  </span>
                )}
              </p>
              <div className="fact-card-modal-field">
                <label className="filter-label">Fact text</label>
                <textarea
                  className="fact-card-modal-textarea"
                  value={editFactText}
                  onChange={e => setEditFactText(e.target.value)}
                  rows={5}
                />
              </div>
              <div className="fact-card-modal-field">
                <label className="fact-card-modal-checkbox-label">
                  <input
                    type="checkbox"
                    checked={editPertinent}
                    onChange={e => setEditPertinent(e.target.checked)}
                  />
                  Pertinent to claims or members
                </label>
              </div>
              <div className="fact-card-modal-field">
                <label className="filter-label">Category relevance (0–1)</label>
                <p className="fact-card-modal-hint">Set score per category; 0 = not relevant.</p>
                <div className="fact-card-modal-categories fact-card-modal-category-sliders">
                  {CATEGORIES.map(({ key, label }) => (
                    <div key={key} className="fact-card-modal-category-row">
                      <label className="fact-card-modal-category-label" htmlFor={`edit-cat-${key}`}>
                        {label}
                      </label>
                      <div className="fact-card-modal-category-slider-wrap">
                        <input
                          id={`edit-cat-${key}`}
                          type="range"
                          min={0}
                          max={1}
                          step={0.1}
                          value={editCategoryScores[key]?.score ?? 0}
                          onChange={e => handleEditCategoryScoreChange(key, parseFloat(e.target.value))}
                        />
                        <span className="fact-card-modal-category-value">
                          {(editCategoryScores[key]?.score ?? 0).toFixed(1)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              {editError && <div className="fact-card-modal-error">{editError}</div>}
              <div className="fact-card-modal-actions">
                <button type="button" className="btn btn-secondary" onClick={closeEditModal}>
                  Cancel
                </button>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={saveEditFact}
                  disabled={editSaving || !editFactText.trim()}
                >
                  {editSaving ? 'Saving...' : 'Save'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
