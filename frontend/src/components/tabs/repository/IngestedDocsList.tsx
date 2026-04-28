interface DocLike {
  id: string
  filename: string
  display_name?: string | null
  payer?: string | null
  state?: string | null
  extraction_status?: string
  chunking_status?: string | null
  embedding_status?: string | null
  published_at?: string | null
  source_metadata?: { source_url?: string | null } | null
  source_url?: string | null
}

interface Props {
  docs: DocLike[]
  selectedDocumentId?: string | null
  onSelect: (documentId: string) => void
  /** Optional map of docId → topic tags (bridged from discovered_sources). */
  tagLabels?: Map<string, string[]>
}

export function IngestedDocsList({ docs, selectedDocumentId, onSelect, tagLabels }: Props) {
  if (docs.length === 0) {
    return (
      <div className="ingested-docs-empty">
        No documents from this entity in the corpus yet.
      </div>
    )
  }
  return (
    <ul className="ingested-docs-list">
      {docs.map((d) => {
        const name = (d.display_name && d.display_name.trim()) || d.filename
        const status = d.published_at
          ? 'published'
          : d.embedding_status === 'failed' || d.chunking_status === 'failed'
            ? 'failed'
            : d.embedding_status === 'completed'
              ? 'embedded'
              : d.chunking_status === 'completed'
                ? 'chunked'
                : d.extraction_status === 'completed'
                  ? 'extracted'
                  : (d.extraction_status || 'pending')

        // Dominant tag = first tag (already sorted by frequency at the parent level)
        const tags = tagLabels?.get(d.id) ?? []
        const dominantTag = tags[0] ?? null

        return (
          <li
            key={d.id}
            className={`ingested-doc-row ${selectedDocumentId === d.id ? 'selected' : ''}`}
          >
            <button
              type="button"
              className="ingested-doc-btn"
              onClick={() => onSelect(d.id)}
              title={name}
            >
              <span className="ingested-doc-name">{name}</span>
              {dominantTag && (
                <span className="ingested-doc-tag" title={tags.join(', ')}>
                  {dominantTag}
                </span>
              )}
              <span className={`ingested-doc-status status-${status}`}>{status}</span>
            </button>
          </li>
        )
      })}
    </ul>
  )
}
