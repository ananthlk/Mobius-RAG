import { useState, useEffect, useCallback, useRef } from 'react'
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
  
  // Filters (server-side)
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([])
  const [sectionFilter, setSectionFilter] = useState<string>('')
  const [pageNumberFilter, setPageNumberFilter] = useState<string>('')
  const [selectedPayers, setSelectedPayers] = useState<string[]>([])
  const [selectedStates, setSelectedStates] = useState<string[]>([])
  const [selectedPrograms, setSelectedPrograms] = useState<string[]>([])
  const [categoryThresholds, setCategoryThresholds] = useState<Record<string, number>>({})
  const [selectedFactTypes, setSelectedFactTypes] = useState<string[]>([])
  const [isPertinentFilter, setIsPertinentFilter] = useState<'all' | 'yes' | 'no'>('all')
  const [isEligibilityFilter, setIsEligibilityFilter] = useState<'all' | 'yes' | 'no'>('all')
  const [approvalFilter, setApprovalFilter] = useState<'all' | 'pending' | 'approved' | 'rejected'>('all')
  const [filtersOpen, setFiltersOpen] = useState(true)

  // Pagination
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(100)
  const [total, setTotal] = useState(0)

  // Filter options from API
  const [filterOptions, setFilterOptions] = useState<{ payers: string[]; states: string[]; programs: string[]; fact_types: string[] }>({ payers: [], states: [], programs: [], fact_types: [] })
  const [sections, setSections] = useState<string[]>([])
  const [documentSearchQuery, setDocumentSearchQuery] = useState('')
  const [metadataDropdownOpen, setMetadataDropdownOpen] = useState<'payer' | 'state' | 'program' | 'factType' | null>(null)
  const [statusDropdownOpen, setStatusDropdownOpen] = useState<'pertinent' | 'eligibility' | 'approval' | null>(null)

  const [searchQueryDebounced, setSearchQueryDebounced] = useState('')
  const searchDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const metadataDropdownRef = useRef<HTMLDivElement | null>(null)
  const statusDropdownRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      const target = e.target as Node
      if (metadataDropdownOpen && metadataDropdownRef.current && !metadataDropdownRef.current.contains(target)) {
        setMetadataDropdownOpen(null)
      }
      if ((statusDropdownOpen || metadataDropdownOpen === 'factType') && statusDropdownRef.current && !statusDropdownRef.current.contains(target)) {
        setStatusDropdownOpen(null)
        if (metadataDropdownOpen === 'factType') setMetadataDropdownOpen(null)
      }
    }
    document.addEventListener('click', onDocClick)
    return () => document.removeEventListener('click', onDocClick)
  }, [metadataDropdownOpen, statusDropdownOpen])

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

  /** Shared helper: approve fact via API and refresh from server. */
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

  /** Shared helper: delete fact via API and refresh from server. */
  const handleDelete = async (fact: Fact, e: React.MouseEvent) => {
    e.stopPropagation()
    if (!confirm('Delete this fact? This cannot be undone.')) return
    try {
      const res = await deleteFactApi(fact.document_id, fact.id, API_BASE)
      if (res.ok) {
        if (selectedFact?.id === fact.id) setSelectedFact(null)
        if (editingFact?.id === fact.id) setEditingFact(null)
        await loadFacts()
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

  const buildFactsParams = useCallback(() => {
    const params = new URLSearchParams()
    params.set('skip', String((page - 1) * pageSize))
    params.set('limit', String(pageSize))
    if (searchQueryDebounced.trim()) params.set('search', searchQueryDebounced.trim())
    selectedDocuments.forEach(id => params.append('document_id', id))
    if (sectionFilter.trim()) params.set('section_path', sectionFilter.trim())
    const pn = parseInt(pageNumberFilter, 10)
    if (!isNaN(pn) && pn > 0) params.set('page_number', String(pn))
    selectedPayers.forEach(p => params.append('payer', p))
    selectedStates.forEach(s => params.append('state', s))
    selectedPrograms.forEach(p => params.append('program', p))
    selectedFactTypes.forEach(t => params.append('fact_type', t))
    if (isPertinentFilter !== 'all') params.set('is_pertinent', isPertinentFilter)
    if (isEligibilityFilter !== 'all') params.set('is_eligibility', isEligibilityFilter)
    if (approvalFilter !== 'all') params.set('verification_status', approvalFilter)
    const catMins = Object.fromEntries(
      Object.entries(categoryThresholds).filter(([, v]) => v > 0)
    )
    if (Object.keys(catMins).length > 0) {
      params.set('category_min_scores', JSON.stringify(catMins))
    }
    return params.toString()
  }, [page, pageSize, searchQueryDebounced, selectedDocuments, sectionFilter, pageNumberFilter,
      selectedPayers, selectedStates, selectedPrograms, selectedFactTypes,
      isPertinentFilter, isEligibilityFilter, approvalFilter, categoryThresholds])

  const loadFacts = useCallback(async () => {
    setLoading(true)
    try {
      const qs = buildFactsParams()
      const response = await fetch(`${API_BASE}/facts?${qs}`)
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        console.error('Failed to load facts:', err.detail || response.statusText)
        setFacts([])
        setTotal(0)
        return
      }
      const data = await response.json()
      const records = data.records || []
      const docs = data.documents || []
      const opts = data.filter_options || { payers: [], states: [], programs: [], fact_types: [] }
      setDocuments(docs.map((d: any) => ({
        id: String(d.id),
        filename: d.filename || '',
        display_name: d.display_name ?? null,
        payer: d.payer ?? null,
        state: d.state ?? null,
        program: d.program ?? null,
        effective_date: d.effective_date ?? null,
        termination_date: d.termination_date ?? null,
      })))
      setFilterOptions({
        payers: opts.payers || [],
        states: opts.states || [],
        programs: opts.programs || [],
        fact_types: opts.fact_types || [],
      })
      setTotal(data.total ?? 0)
      setFacts(records.map((r: any) => ({
        ...r,
        document_id: String(r.document_id),
        category_scores: r.category_scores ?? null,
      })))
    } catch (err) {
      console.error('Failed to load facts:', err)
      setFacts([])
      setTotal(0)
    } finally {
      setLoading(false)
    }
  }, [buildFactsParams])

  const loadSections = useCallback(async () => {
    try {
      const params = new URLSearchParams()
      selectedDocuments.forEach(id => params.append('document_id', id))
      const qs = params.toString()
      const url = qs ? `${API_BASE}/facts/sections?${qs}` : `${API_BASE}/facts/sections`
      const res = await fetch(url)
      if (res.ok) {
        const data = await res.json()
        setSections(data.sections || [])
      } else {
        setSections([])
      }
    } catch {
      setSections([])
    }
  }, [selectedDocuments])

  useEffect(() => {
    loadFacts()
  }, [loadFacts])

  useEffect(() => {
    loadSections()
  }, [loadSections])

  // Debounce search input
  useEffect(() => {
    if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current)
    searchDebounceRef.current = setTimeout(() => {
      setSearchQueryDebounced(searchQuery)
      setPage(1)
      searchDebounceRef.current = null
    }, 300)
    return () => {
      if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current)
    }
  }, [searchQuery])

  // Reset to page 1 when filters change (except page, pageSize)
  useEffect(() => {
    setPage(1)
  }, [selectedDocuments, sectionFilter, pageNumberFilter, selectedPayers, selectedStates,
      selectedPrograms, selectedFactTypes, isPertinentFilter, isEligibilityFilter, approvalFilter,
      categoryThresholds])

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

  // Filter options from API; fallback to documents for payer/state/program when empty
  const uniquePayers = filterOptions.payers.length > 0
    ? filterOptions.payers
    : Array.from(new Set(documents.map(d => d.payer).filter(Boolean))).sort() as string[]
  const uniqueStates = filterOptions.states.length > 0
    ? filterOptions.states
    : Array.from(new Set(documents.map(d => d.state).filter(Boolean))).sort() as string[]
  const uniquePrograms = filterOptions.programs.length > 0
    ? filterOptions.programs
    : Array.from(new Set(documents.map(d => d.program).filter(Boolean))).sort() as string[]
  const uniqueFactTypes = filterOptions.fact_types

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
    (searchQueryDebounced.trim() ? 1 : 0) +
    (selectedDocuments.length || 0) +
    (sectionFilter.trim() ? 1 : 0) +
    (pageNumberFilter.trim() ? 1 : 0) +
    (selectedPayers.length || 0) +
    (selectedStates.length || 0) +
    (selectedPrograms.length || 0) +
    (selectedFactTypes.length || 0) +
    (isPertinentFilter !== 'all' ? 1 : 0) +
    (isEligibilityFilter !== 'all' ? 1 : 0) +
    (approvalFilter !== 'all' ? 1 : 0) +
    (Object.keys(categoryThresholds).filter(k => (categoryThresholds[k] || 0) > 0).length)

  const clearFilters = () => {
    setSelectedDocuments([])
    setDocumentSearchQuery('')
    setMetadataDropdownOpen(null)
    setStatusDropdownOpen(null)
    setSectionFilter('')
    setPageNumberFilter('')
    setSelectedPayers([])
    setSelectedStates([])
    setSelectedPrograms([])
    setCategoryThresholds({})
    setSelectedFactTypes([])
    setIsPertinentFilter('all')
    setIsEligibilityFilter('all')
    setApprovalFilter('all')
    setSearchQuery('')
    setSearchQueryDebounced('')
    setPage(1)
  }

  const totalPages = Math.max(1, Math.ceil(total / pageSize))
  const startItem = total === 0 ? 0 : (page - 1) * pageSize + 1
  const endItem = Math.min(page * pageSize, total)

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
      {/* Full-width main area - Filters + Fact Cards */}
      <div className="facts-main">
        {/* Inline expandable filters - full width */}
        <div className={`filters-inline ${filtersOpen ? 'filters-inline-open' : ''}`}>
          <div className="filters-inline-header">
            <h3 className="filters-inline-title">Filters</h3>
            <div className="filters-inline-actions">
              <button onClick={clearFilters} className="clear-filters-btn">
                Clear All
              </button>
              <button
                type="button"
                className="filters-inline-toggle"
                onClick={() => setFiltersOpen(prev => !prev)}
                aria-label={filtersOpen ? 'Collapse filters' : 'Expand filters'}
              >
                {filtersOpen ? '▲ Collapse' : '▼ Filters'}
              </button>
            </div>
          </div>

          {filtersOpen && (
            <div className="filters-inline-content">
              <div className="filters-grid">
                {/* Categories - most important, first */}
                <div className="filter-section filter-section-categories">
                  <h4 className="filter-section-title">Categories (score threshold)</h4>
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

                {/* Document + Status (Status to the right of Document) */}
                <div className="filter-section filter-section-document-status">
                  <div className="document-status-row">
                    <div className="document-filters">
                      <h4 className="filter-section-title">Document</h4>
                      <div className="filter-group">
                    <label className="filter-label">Document</label>
                    <div className="document-multiselect">
                <input
                  type="text"
                  placeholder="Search documents..."
                  value={documentSearchQuery}
                  onChange={(e) => setDocumentSearchQuery(e.target.value)}
                  className="document-multiselect-search"
                  aria-label="Search documents"
                />
                <div className="document-multiselect-list">
                  {(() => {
                    const q = documentSearchQuery.trim().toLowerCase()
                    const filtered = q
                      ? documents.filter(
                          (d) =>
                            (d.display_name?.trim() || d.filename || '').toLowerCase().includes(q) ||
                            (d.filename || '').toLowerCase().includes(q) ||
                            (d.payer || '').toLowerCase().includes(q) ||
                            (d.state || '').toLowerCase().includes(q) ||
                            (d.program || '').toLowerCase().includes(q)
                        )
                      : documents
                    return documents.length === 0 ? (
                      <span className="filter-hint">No documents yet.</span>
                    ) : filtered.length === 0 ? (
                      <span className="filter-hint">No documents match &quot;{documentSearchQuery}&quot;</span>
                    ) : (
                      filtered.map((doc) => {
                        const label = doc.display_name?.trim() || doc.filename || doc.id
                        const isSelected = selectedDocuments.includes(doc.id)
                        return (
                          <label key={doc.id} className={`document-multiselect-item ${isSelected ? 'document-multiselect-item-selected' : ''}`}>
                            <input
                              type="checkbox"
                              checked={isSelected}
                              onChange={() => {
                                setSelectedDocuments((prev) =>
                                  isSelected ? prev.filter((id) => id !== doc.id) : [...prev, doc.id]
                                )
                              }}
                            />
                            <span
                              className="document-multiselect-item-label"
                              title={doc.display_name?.trim() && doc.filename && doc.display_name !== doc.filename ? `${doc.display_name} (${doc.filename})` : doc.filename || label}
                            >
                              {label}
                            </span>
                          </label>
                        )
                      })
                    )
                  })()}
                </div>
                {selectedDocuments.length > 0 && (
                  <button
                    type="button"
                    className="document-multiselect-clear"
                    onClick={() => setSelectedDocuments([])}
                  >
                    Clear ({selectedDocuments.length} selected)
                  </button>
                )}
                    </div>
                  </div>
                  <div className="filter-group">
                    <label className="filter-label">Section</label>
                    <select
                      value={sectionFilter}
                      onChange={(e) => setSectionFilter(e.target.value)}
                      className="filter-select"
                    >
                      <option value="">All sections</option>
                      {sections.map((s) => (
                        <option key={s} value={s}>{s}</option>
                      ))}
                    </select>
                  </div>
                  <div className="filter-group">
                    <label className="filter-label">Page number</label>
                    <input
                      type="number"
                      min={1}
                      placeholder="All pages"
                      value={pageNumberFilter}
                      onChange={(e) => setPageNumberFilter(e.target.value)}
                      className="filter-input"
                    />
                  </div>
                    </div>

                    {/* Status - to the right of Document */}
                    <div className="status-filters" ref={statusDropdownRef}>
                      <h4 className="filter-section-title">Status</h4>
                      <div className="status-toggles">
                        <div className="status-toggle">
                          <label className="filter-label">Pertinent</label>
                          <div className="status-dropdown-wrap">
                            <button
                              type="button"
                              className={`status-dropdown-trigger ${statusDropdownOpen === 'pertinent' ? 'open' : ''} ${isPertinentFilter !== 'all' ? 'has-selection' : ''}`}
                              onClick={() => setStatusDropdownOpen(prev => prev === 'pertinent' ? null : 'pertinent')}
                            >
                              {isPertinentFilter === 'all' ? 'All' : isPertinentFilter === 'yes' ? 'Yes' : 'No'}
                            </button>
                            {statusDropdownOpen === 'pertinent' && (
                              <div className="status-dropdown-panel" onClick={e => e.stopPropagation()}>
                                {(['all', 'yes', 'no'] as const).map((v) => (
                                  <button key={v} type="button" className={`status-dropdown-option ${isPertinentFilter === v ? 'selected' : ''}`} onClick={() => { setIsPertinentFilter(v); setStatusDropdownOpen(null) }}>
                                    {v === 'all' ? 'All' : v === 'yes' ? 'Yes' : 'No'}
                                  </button>
                                ))}
                              </div>
                            )}
                          </div>
                        </div>
                        <div className="status-toggle">
                          <label className="filter-label">Eligibility</label>
                          <div className="status-dropdown-wrap">
                            <button
                              type="button"
                              className={`status-dropdown-trigger ${statusDropdownOpen === 'eligibility' ? 'open' : ''} ${isEligibilityFilter !== 'all' ? 'has-selection' : ''}`}
                              onClick={() => setStatusDropdownOpen(prev => prev === 'eligibility' ? null : 'eligibility')}
                            >
                              {isEligibilityFilter === 'all' ? 'All' : isEligibilityFilter === 'yes' ? 'Yes' : 'No'}
                            </button>
                            {statusDropdownOpen === 'eligibility' && (
                              <div className="status-dropdown-panel" onClick={e => e.stopPropagation()}>
                                {(['all', 'yes', 'no'] as const).map((v) => (
                                  <button key={v} type="button" className={`status-dropdown-option ${isEligibilityFilter === v ? 'selected' : ''}`} onClick={() => { setIsEligibilityFilter(v); setStatusDropdownOpen(null) }}>
                                    {v === 'all' ? 'All' : v === 'yes' ? 'Yes' : 'No'}
                                  </button>
                                ))}
                              </div>
                            )}
                          </div>
                        </div>
                        <div className="status-toggle">
                          <label className="filter-label">Approval</label>
                          <div className="status-dropdown-wrap">
                            <button
                              type="button"
                              className={`status-dropdown-trigger ${statusDropdownOpen === 'approval' ? 'open' : ''} ${approvalFilter !== 'all' ? 'has-selection' : ''}`}
                              onClick={() => setStatusDropdownOpen(prev => prev === 'approval' ? null : 'approval')}
                            >
                              {approvalFilter === 'all' ? 'All' : approvalFilter}
                            </button>
                            {statusDropdownOpen === 'approval' && (
                              <div className="status-dropdown-panel" onClick={e => e.stopPropagation()}>
                                {(['all', 'pending', 'approved', 'rejected'] as const).map((v) => (
                                  <button key={v} type="button" className={`status-dropdown-option ${approvalFilter === v ? 'selected' : ''}`} onClick={() => { setApprovalFilter(v); setStatusDropdownOpen(null) }}>
                                    {v === 'all' ? 'All' : v.charAt(0).toUpperCase() + v.slice(1)}
                                  </button>
                                ))}
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="filter-group">
                        <label className="filter-label">Fact type</label>
                        <div className="status-dropdown-wrap">
                          <button
                            type="button"
                            className={`metadata-dropdown-trigger ${metadataDropdownOpen === 'factType' ? 'open' : ''} ${selectedFactTypes.length > 0 ? 'has-selection' : ''}`}
                            onClick={() => setMetadataDropdownOpen(prev => prev === 'factType' ? null : 'factType')}
                          >
                            {selectedFactTypes.length === 0 ? 'All types' : `${selectedFactTypes.length} selected`}
                          </button>
                          {metadataDropdownOpen === 'factType' && (
                            <div className="metadata-dropdown-panel" onClick={e => e.stopPropagation()}>
                              <div className="metadata-dropdown-options">
                                {uniqueFactTypes.map((type) => (
                                  <label key={type} className="metadata-dropdown-option">
                                    <input type="checkbox" checked={selectedFactTypes.includes(type)} onChange={(e) => {
                                      setSelectedFactTypes(prev => e.target.checked ? [...prev, type] : prev.filter(x => x !== type))
                                    }} />
                                    <span>{type}</span>
                                  </label>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Metadata: Payer, State, Program - compact dropdown multi-selects */}
                <div className="filter-section filter-section-metadata" ref={metadataDropdownRef}>
                  <h4 className="filter-section-title">Metadata</h4>
                  <div className="metadata-dropdowns">
                    <div className="metadata-dropdown">
                      <label className="filter-label">Payer</label>
                      <div className="metadata-dropdown-trigger-wrap">
                        <button
                          type="button"
                          className={`metadata-dropdown-trigger ${metadataDropdownOpen === 'payer' ? 'open' : ''} ${selectedPayers.length > 0 ? 'has-selection' : ''}`}
                          onClick={() => setMetadataDropdownOpen(prev => prev === 'payer' ? null : 'payer')}
                          aria-expanded={metadataDropdownOpen === 'payer'}
                        >
                          {selectedPayers.length === 0 ? 'All payers' : `${selectedPayers.length} selected`}
                        </button>
                        {metadataDropdownOpen === 'payer' && (
                          <div className="metadata-dropdown-panel" onClick={e => e.stopPropagation()}>
                            <div className="metadata-dropdown-options">
                              {uniquePayers.length === 0 ? (
                                <span className="filter-hint">No payers yet</span>
                              ) : (
                                uniquePayers.map((v) => (
                                  <label key={v} className="metadata-dropdown-option">
                                    <input type="checkbox" checked={selectedPayers.includes(v)} onChange={(e) => {
                                      setSelectedPayers(prev => e.target.checked ? [...prev, v] : prev.filter(x => x !== v))
                                    }} />
                                    <span>{v}</span>
                                  </label>
                                ))
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="metadata-dropdown">
                      <label className="filter-label">State</label>
                      <div className="metadata-dropdown-trigger-wrap">
                        <button
                          type="button"
                          className={`metadata-dropdown-trigger ${metadataDropdownOpen === 'state' ? 'open' : ''} ${selectedStates.length > 0 ? 'has-selection' : ''}`}
                          onClick={() => setMetadataDropdownOpen(prev => prev === 'state' ? null : 'state')}
                          aria-expanded={metadataDropdownOpen === 'state'}
                        >
                          {selectedStates.length === 0 ? 'All states' : `${selectedStates.length} selected`}
                        </button>
                        {metadataDropdownOpen === 'state' && (
                          <div className="metadata-dropdown-panel" onClick={e => e.stopPropagation()}>
                            <div className="metadata-dropdown-options">
                              {uniqueStates.length === 0 ? (
                                <span className="filter-hint">No states yet</span>
                              ) : (
                                uniqueStates.map((v) => (
                                  <label key={v} className="metadata-dropdown-option">
                                    <input type="checkbox" checked={selectedStates.includes(v)} onChange={(e) => {
                                      setSelectedStates(prev => e.target.checked ? [...prev, v] : prev.filter(x => x !== v))
                                    }} />
                                    <span>{getStateLabel(v) || v}</span>
                                  </label>
                                ))
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="metadata-dropdown">
                      <label className="filter-label">Program</label>
                      <div className="metadata-dropdown-trigger-wrap">
                        <button
                          type="button"
                          className={`metadata-dropdown-trigger ${metadataDropdownOpen === 'program' ? 'open' : ''} ${selectedPrograms.length > 0 ? 'has-selection' : ''}`}
                          onClick={() => setMetadataDropdownOpen(prev => prev === 'program' ? null : 'program')}
                          aria-expanded={metadataDropdownOpen === 'program'}
                        >
                          {selectedPrograms.length === 0 ? 'All programs' : `${selectedPrograms.length} selected`}
                        </button>
                        {metadataDropdownOpen === 'program' && (
                          <div className="metadata-dropdown-panel" onClick={e => e.stopPropagation()}>
                            <div className="metadata-dropdown-options">
                              {uniquePrograms.length === 0 ? (
                                <span className="filter-hint">No programs yet</span>
                              ) : (
                                uniquePrograms.map((v) => (
                                  <label key={v} className="metadata-dropdown-option">
                                    <input type="checkbox" checked={selectedPrograms.includes(v)} onChange={(e) => {
                                      setSelectedPrograms(prev => e.target.checked ? [...prev, v] : prev.filter(x => x !== v))
                                    }} />
                                    <span>{v}</span>
                                  </label>
                                ))
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
        <header className="facts-toolbar">
          <div className="facts-toolbar-left">
            <h2 className="facts-toolbar-title">Facts</h2>
            <span className="facts-toolbar-count" title={`Showing ${startItem}-${endItem} of ${total} facts`}>
              {total === 0 ? '0' : `${startItem}-${endItem} of ${total}`}
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
          ) : facts.length === 0 ? (
            <div className="no-facts">
              {total === 0
                ? 'No facts yet. Run chunking on a document (Document Status or Live Updates), then click Refresh.'
                : 'No facts found matching your filters'}
            </div>
          ) : (
            <div className="facts-grid">
              {facts.map((fact) => {
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

          {/* Pagination */}
          {total > 0 && (
            <div className="facts-pagination">
              <div className="facts-pagination-left">
                <label className="facts-pagination-label">
                  Per page:
                  <select
                    value={pageSize}
                    onChange={(e) => { setPageSize(Number(e.target.value)); setPage(1) }}
                    className="facts-pagination-select"
                  >
                    {[25, 50, 100, 250, 500].map((n) => (
                      <option key={n} value={n}>{n}</option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="facts-pagination-center">
                <button
                  type="button"
                  className="facts-pagination-btn"
                  onClick={() => setPage(1)}
                  disabled={page <= 1}
                  aria-label="First page"
                >
                  «
                </button>
                <button
                  type="button"
                  className="facts-pagination-btn"
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page <= 1}
                  aria-label="Previous page"
                >
                  ‹
                </button>
                <span className="facts-pagination-info">
                  Page {page} of {totalPages}
                </span>
                <button
                  type="button"
                  className="facts-pagination-btn"
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={page >= totalPages}
                  aria-label="Next page"
                >
                  ›
                </button>
                <button
                  type="button"
                  className="facts-pagination-btn"
                  onClick={() => setPage(totalPages)}
                  disabled={page >= totalPages}
                  aria-label="Last page"
                >
                  »
                </button>
              </div>
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
