/**
 * RAG-specific fact modal – for editing facts with category sliders.
 * Kept here for future use; the primary flow is now a simplified "Mark as fact".
 */
import { useState } from 'react'

interface CategoryDef {
  key: string
  label: string
}

interface FactModalProps {
  mode: 'add' | 'edit'
  factText: string
  pageNumber: number | null
  isPertinent: boolean
  categoryScores: Record<string, { score: number; direction: number }>
  categories: CategoryDef[]
  onSubmit: (params: {
    factText: string
    isPertinent: boolean
    categoryScores: Record<string, { score: number; direction: number }>
  }) => Promise<boolean>
  onClose: () => void
}

export function FactModal({
  mode,
  factText: initialFactText,
  pageNumber,
  isPertinent: initialPertinent,
  categoryScores: initialScores,
  categories,
  onSubmit,
  onClose,
}: FactModalProps) {
  const [factText, setFactText] = useState(initialFactText)
  const [isPertinent, setIsPertinent] = useState(initialPertinent)
  const [categoryScores, setCategoryScores] = useState(initialScores)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleScoreChange = (key: string, score: number) => {
    setCategoryScores((prev) => ({
      ...prev,
      [key]: { score, direction: prev[key]?.direction ?? 0.5 },
    }))
  }

  const handleSubmit = async () => {
    const text = factText.trim()
    if (!text) return
    setSubmitting(true)
    setError(null)
    try {
      const nonZeroScores: Record<string, { score: number; direction: number }> = {}
      for (const { key } of categories) {
        const entry = categoryScores[key]
        if (entry && entry.score > 0) {
          nonZeroScores[key] = { score: entry.score, direction: entry.direction }
        }
      }
      const success = await onSubmit({
        factText: text,
        isPertinent,
        categoryScores: nonZeroScores,
      })
      if (!success) {
        setError('Failed to save fact')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="dv-modal-overlay" onClick={onClose}>
      <div className="dv-modal" onClick={(e) => e.stopPropagation()}>
        <h3 className="dv-modal-title">
          {mode === 'edit' ? 'Edit fact' : 'Add fact from selection'}
        </h3>
        {pageNumber != null && mode === 'add' && (
          <p className="dv-modal-page">From page {pageNumber}</p>
        )}
        <div className="dv-modal-field">
          <label htmlFor="dv-modal-fact-text">Fact text</label>
          <textarea
            id="dv-modal-fact-text"
            value={factText}
            onChange={(e) => setFactText(e.target.value)}
            rows={4}
            className="dv-modal-textarea"
          />
        </div>
        <div className="dv-modal-field">
          <label>Category relevance (sliding scale 0{'\u2013'}1)</label>
          <p className="dv-modal-category-hint">Set score per category; 0 = not relevant.</p>
          <div className="dv-modal-categories dv-modal-category-sliders">
            {categories.map(({ key, label }) => (
              <div key={key} className="dv-modal-category-row">
                <label className="dv-modal-category-label" htmlFor={`dv-cat-${key}`}>
                  {label}
                </label>
                <div className="dv-modal-category-slider-wrap">
                  <input
                    id={`dv-cat-${key}`}
                    type="range"
                    min={0}
                    max={1}
                    step={0.1}
                    value={categoryScores[key]?.score ?? 0}
                    onChange={(e) => handleScoreChange(key, parseFloat(e.target.value))}
                  />
                  <span className="dv-modal-category-value">
                    {(categoryScores[key]?.score ?? 0).toFixed(1)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="dv-modal-field">
          <label className="dv-modal-pertinent-check">
            <input
              type="checkbox"
              checked={isPertinent}
              onChange={(e) => setIsPertinent(e.target.checked)}
            />
            Pertinent to claims or members
          </label>
        </div>
        {error && <div className="dv-modal-error">{error}</div>}
        <div className="dv-modal-actions">
          <button type="button" className="dv-btn dv-btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="dv-btn dv-btn-primary"
            onClick={handleSubmit}
            disabled={submitting || !factText.trim()}
          >
            {submitting
              ? mode === 'edit'
                ? 'Saving...'
                : 'Adding...'
              : mode === 'edit'
                ? 'Save changes'
                : 'Add fact'}
          </button>
        </div>
      </div>
    </div>
  )
}
