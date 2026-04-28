import { useEffect, useRef } from 'react'

interface SplitterProps {
  /** Container element whose --repo-w CSS variable gets updated during drag. */
  containerRef: React.RefObject<HTMLDivElement | null>
  /** Cycle button click — parent decides next state. */
  onCycle: () => void
  /** Label for the cycle button (e.g. "‹‹" or "››"). */
  cycleLabel: string
  /** Persist callback fired when drag ends. */
  onPersist?: (widthPx: number) => void
}

/**
 * Vertical drag handle between Repository and Reader panes.
 *
 * Updates the parent container's `--repo-w` CSS variable directly (no
 * React state churn during drag). Clamps between 64px and (container -
 * 320px). On release, calls onPersist with the final px value.
 */
export function Splitter({ containerRef, onCycle, cycleLabel, onPersist }: SplitterProps) {
  const draggingRef = useRef(false)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const onMove = (e: MouseEvent) => {
      if (!draggingRef.current) return
      const rect = container.getBoundingClientRect()
      const offset = e.clientX - rect.left
      const min = 64
      const max = Math.max(min, rect.width - 320)
      const clamped = Math.min(max, Math.max(min, offset))
      container.style.setProperty('--repo-w', `${clamped}px`)
    }
    const onUp = () => {
      if (!draggingRef.current) return
      draggingRef.current = false
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      const w = container.style.getPropertyValue('--repo-w')
      const px = parseFloat(w) || 0
      if (px && onPersist) onPersist(px)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [containerRef, onPersist])

  const startDrag = (e: React.MouseEvent) => {
    e.preventDefault()
    draggingRef.current = true
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }

  return (
    <div
      className="repo-splitter"
      onMouseDown={startDrag}
      role="separator"
      aria-orientation="vertical"
    >
      <button
        type="button"
        className="repo-splitter-pill"
        onClick={(e) => {
          e.stopPropagation()
          onCycle()
        }}
        onMouseDown={(e) => e.stopPropagation()}
        title="Toggle Repository width"
        aria-label="Toggle Repository width"
      >
        {cycleLabel}
      </button>
    </div>
  )
}
