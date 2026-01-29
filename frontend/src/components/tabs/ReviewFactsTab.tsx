import { useState, useEffect } from 'react'
import './ReviewFactsTab.css'

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
  payer?: string
  state?: string
  program?: string
  page_number?: number | null
}

interface ReviewFactsTabProps {
  onViewDocument?: (documentId: string, options?: { pageNumber?: number; factId?: string }) => void
}

export function ReviewFactsTab({ onViewDocument }: ReviewFactsTabProps) {
  const [facts, setFacts] = useState<Fact[]>([])
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

  const API_BASE = 'http://localhost:8000'

  const loadFacts = async () => {
    setLoading(true)
    try {
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
          }))
          const factsWithMetadata = await Promise.all(
            factsData.map(async (fact: Fact) => {
              try {
                const docResponse = await fetch(`${API_BASE}/admin/db/tables/documents/records/${fact.document_id}`)
                if (docResponse.ok) {
                  const docData = await docResponse.json()
                  return {
                    ...fact,
                    document_filename: docData.record?.filename,
                    payer: docData.record?.payer,
                    state: docData.record?.state,
                    program: docData.record?.program,
                  }
                }
              } catch (err) {
                console.error('Failed to fetch document metadata:', err)
              }
              return fact
            })
          )
          setFacts(factsWithMetadata)
          return
        }
      }

      // 2) Fallback: load by document (normalized facts from GET /documents/:id/facts)
      const docsResponse = await fetch(`${API_BASE}/documents`)
      if (!docsResponse.ok) return
      const docsData = await docsResponse.json()
      const docs = Array.isArray(docsData) ? docsData : docsData.documents || []
      const allFacts: Fact[] = []
      for (const doc of docs) {
        const id = doc.id ?? doc.document_id
        if (!id) continue
        try {
          const factsResponse = await fetch(`${API_BASE}/documents/${id}/facts`)
          if (!factsResponse.ok) continue
          const factsData = await factsResponse.json()
          const chunks = factsData.chunks || []
          const filename = doc.filename || doc.name || 'Unknown Document'
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
                payer: doc.payer,
                state: doc.state,
                program: doc.program,
                page_number: f.page_number != null ? Number(f.page_number) : null,
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
    
    return true
  })

  const uniquePayers = Array.from(new Set(facts.map(f => f.payer).filter(Boolean))) as string[]
  const uniqueStates = Array.from(new Set(facts.map(f => f.state).filter(Boolean))) as string[]
  const uniquePrograms = Array.from(new Set(facts.map(f => f.program).filter(Boolean))) as string[]
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

  const clearFilters = () => {
    setSelectedPayers([])
    setSelectedStates([])
    setSelectedPrograms([])
    setCategoryThresholds({})
    setSelectedFactTypes([])
    setIsPertinentFilter('all')
    setIsEligibilityFilter('all')
    setSearchQuery('')
  }

  return (
    <div className="review-facts-tab">
      <div className="facts-layout">
        {/* Left Sidebar - Filters */}
        <div className="filters-sidebar">
          <div className="filters-header">
            <h3 className="filters-title">Filters</h3>
            <button onClick={clearFilters} className="clear-filters-btn">
              Clear All
            </button>
          </div>

          <div className="filters-content">
            {/* Search */}
            <div className="filter-group">
              <label className="filter-label">Search</label>
              <input
                type="text"
                placeholder="Search facts..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="filter-input"
              />
            </div>

            {/* Document Metadata Filters */}
            <div className="filter-group">
              <label className="filter-label">Payer</label>
              <div className="filter-checkboxes">
                {uniquePayers.map((payer) => (
                  <label key={payer} className="filter-checkbox">
                    <input
                      type="checkbox"
                      checked={selectedPayers.includes(payer)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedPayers([...selectedPayers, payer])
                        } else {
                          setSelectedPayers(selectedPayers.filter(p => p !== payer))
                        }
                      }}
                    />
                    <span>{payer}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="filter-group">
              <label className="filter-label">State</label>
              <div className="filter-checkboxes">
                {uniqueStates.map((state) => (
                  <label key={state} className="filter-checkbox">
                    <input
                      type="checkbox"
                      checked={selectedStates.includes(state)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedStates([...selectedStates, state])
                        } else {
                          setSelectedStates(selectedStates.filter(s => s !== state))
                        }
                      }}
                    />
                    <span>{state}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="filter-group">
              <label className="filter-label">Program</label>
              <div className="filter-checkboxes">
                {uniquePrograms.map((program) => (
                  <label key={program} className="filter-checkbox">
                    <input
                      type="checkbox"
                      checked={selectedPrograms.includes(program)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedPrograms([...selectedPrograms, program])
                        } else {
                          setSelectedPrograms(selectedPrograms.filter(p => p !== program))
                        }
                      }}
                    />
                    <span>{program}</span>
                  </label>
                ))}
              </div>
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
          </div>
        </div>

        {/* Main Area - Fact Cards */}
        <div className="facts-main">
          <div className="facts-header">
            <h2 className="facts-title">
              Facts ({filteredFacts.length})
            </h2>
            <button onClick={loadFacts} className="btn btn-secondary" disabled={loading}>
              {loading ? 'Loading...' : 'Refresh'}
            </button>
          </div>

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
                
                return (
                  <div
                    key={fact.id}
                    className={`fact-card ${isExpanded ? 'expanded' : ''}`}
                    onClick={() => setSelectedFact(isExpanded ? null : fact)}
                  >
                    <div className="fact-card-content">
                      <div className="fact-card-header">
                        <div className="fact-document-name">
                          {fact.document_filename || 'Unknown Document'}
                        </div>
                        {fact.fact_type && (
                          <span className="fact-type-badge">{fact.fact_type}</span>
                        )}
                      </div>
                      
                      <div className="fact-text">
                        {isExpanded 
                          ? fact.fact_text
                          : fact.fact_text.length > 200
                            ? `${fact.fact_text.substring(0, 200)}...`
                            : fact.fact_text}
                      </div>
                      
                      {topCategories.length > 0 && (
                        <div className="fact-categories">
                          {topCategories.slice(0, isExpanded ? 10 : 3).map((cat) => (
                            <span key={cat.category} className="category-tag">
                              {cat.category.replace(/_/g, ' ')} ({cat.score.toFixed(2)})
                            </span>
                          ))}
                        </div>
                      )}
                      
                      <div className="fact-indicators">
                        <span className={`pertinent-badge ${isPertinent ? 'yes' : 'no'}`}>
                          {isPertinent ? '✓ Pertinent' : '✗ Not Pertinent'}
                        </span>
                        {isEligible && (
                          <span className="eligibility-badge">Eligibility Related</span>
                        )}
                      </div>
                      
                      {fact.payer || fact.state || fact.program ? (
                        <div className="fact-metadata">
                          {fact.payer && <span>{fact.payer}</span>}
                          {fact.state && <span>{fact.state}</span>}
                          {fact.program && <span>{fact.program}</span>}
                        </div>
                      ) : null}
                    </div>
                    
                    {/* Expanded Details */}
                    {isExpanded && (
                      <div className="fact-expanded-details">
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
                        
                        {onViewDocument && (
                          <div className="fact-expanded-actions">
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                onViewDocument(fact.document_id, {
                                  pageNumber: fact.page_number ?? undefined,
                                  factId: fact.id,
                                })
                              }}
                              className="btn btn-primary"
                            >
                              View in Document
                            </button>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>

    </div>
  )
}
