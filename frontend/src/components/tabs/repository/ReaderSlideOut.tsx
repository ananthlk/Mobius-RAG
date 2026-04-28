import { DocumentReaderTab } from '../DocumentReaderTab'

interface DocLike {
  id: string
  filename: string
  display_name?: string | null
  file_path?: string | null
}

interface NavigateToRead {
  documentId: string
  pageNumber?: number
  factId?: string
  citeText?: string
}

interface UrlPreview {
  url: string
  ingested: boolean
}

interface Props {
  documents: DocLike[]
  selectedDocumentId: string | null
  navigateToRead: NavigateToRead | null
  onNavigateToReadConsumed: () => void
  onDocumentSelect: (documentId: string) => void
  /** Optional: if set, render a URL preview pane instead of the document reader. */
  urlPreview?: UrlPreview | null
  /** Trigger ingest for a URL preview that isn't yet in the corpus. */
  onIngestUrl?: (url: string) => void
  ingestingUrl?: boolean
  /** Close the slide-out (× button). */
  onClose: () => void
  /** Whether the sections sidebar inside the reader starts open. @default false */
  initialSidebarOpen?: boolean
}

/**
 * Right-side slide-out that wraps DocumentReaderTab.
 *
 * For URL-leaf rows that aren't yet ingested, renders a lightweight
 * preview pane with an Ingest button instead of the full Reader.
 */
export function ReaderSlideOut({
  documents,
  selectedDocumentId,
  navigateToRead,
  onNavigateToReadConsumed,
  onDocumentSelect,
  urlPreview,
  onIngestUrl,
  ingestingUrl,
  onClose,
  initialSidebarOpen = false,
}: Props) {
  // Resolve display name for header
  const selectedDoc = documents.find((d) => d.id === selectedDocumentId)
  const docTitle = urlPreview
    ? urlPreview.url.replace(/^https?:\/\//, '')
    : (selectedDoc?.display_name?.trim() || selectedDoc?.filename || null)

  return (
    <div className="reader-slideout">
      <div className="reader-slideout-header">
        <span className="reader-slideout-title" title={docTitle ?? undefined}>
          {docTitle ?? 'Reader'}
        </span>
        <button
          type="button"
          className="reader-slideout-close"
          onClick={onClose}
          aria-label="Close reader"
          title="Close reader"
        >
          ×
        </button>
      </div>
      <div className="reader-slideout-body">
        {urlPreview ? (
          <div className="reader-url-preview">
            <a
              href={urlPreview.url}
              target="_blank"
              rel="noopener noreferrer"
              className="reader-url-preview-link"
            >
              {urlPreview.url}
            </a>
            <div className="reader-url-preview-actions">
              {!urlPreview.ingested && onIngestUrl && (
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={() => onIngestUrl(urlPreview.url)}
                  disabled={ingestingUrl}
                >
                  {ingestingUrl ? 'Ingesting…' : 'Ingest →'}
                </button>
              )}
              {urlPreview.ingested && (
                <span className="reader-url-preview-already">✓ Already in corpus</span>
              )}
            </div>
            <p className="reader-url-preview-hint">
              Once ingested, the document will appear in the Documents list.
            </p>
          </div>
        ) : (
          <DocumentReaderTab
            documents={documents}
            selectedDocumentId={selectedDocumentId}
            navigateToRead={navigateToRead}
            onNavigateToReadConsumed={onNavigateToReadConsumed}
            onDocumentSelect={onDocumentSelect}
            initialSidebarOpen={initialSidebarOpen}
            hideDocumentSelector
          />
        )}
      </div>
    </div>
  )
}
