import { useEffect, useRef } from 'react'
import './LiveUpdatesTab.css'

interface ActiveJob {
  document_id: string
  filename: string
  status: string
  progress: number
  start_time: string
  current_page?: number
  total_pages?: number
  current_paragraph?: string
  total_paragraphs?: number
  completed_paragraphs?: number
  processing_stage?: 'extracting' | 'critiquing' | 'retrying' | 'persisting' | 'idle'
}

// Utility function to parse paragraph ID and extract page/paragraph numbers
function parseParagraphId(paraId: string): { page: number; paragraph: number } | null {
  const parts = paraId.split('_')
  if (parts.length >= 2) {
    const page = parseInt(parts[0], 10)
    const paragraph = parseInt(parts[1], 10)
    if (!isNaN(page) && !isNaN(paragraph)) {
      return { page, paragraph }
    }
  }
  return null
}

interface LiveUpdatesTabProps {
  activeJobs: ActiveJob[]
  onJobSelect: (documentId: string) => void
  selectedJobId: string | null
  streamingOutput: string
  isActive?: boolean
}

// Export the interface for use in App.tsx
export type { ActiveJob }

export function LiveUpdatesTab({ 
  activeJobs, 
  onJobSelect, 
  selectedJobId,
  streamingOutput,
  isActive = true
}: LiveUpdatesTabProps) {
  const streamRef = useRef<HTMLDivElement>(null)

  // Auto-select first active job when tab becomes active and no job is selected
  useEffect(() => {
    if (isActive && !selectedJobId && activeJobs.length > 0) {
      // Select the first active job
      onJobSelect(activeJobs[0].document_id)
    }
  }, [isActive, selectedJobId, activeJobs, onJobSelect])

  // Limit output to last 2500 characters to keep it compact
  // Show a truncation indicator if we're limiting
  const MAX_OUTPUT_LENGTH = 2500
  const isTruncated = streamingOutput.length > MAX_OUTPUT_LENGTH
  const limitedOutput = isTruncated 
    ? streamingOutput.slice(-MAX_OUTPUT_LENGTH) 
    : streamingOutput

  useEffect(() => {
    if (streamRef.current) {
      // Scroll to bottom to show latest content (since we're showing the tail)
      streamRef.current.scrollTop = streamRef.current.scrollHeight
    }
  }, [streamingOutput])

  const getJobStatusBadge = (status: string) => {
    const s = status || 'idle'
    const statusClass = `job-status-badge status-${s}`
    const statusLabel = s.charAt(0).toUpperCase() + s.slice(1).replace(/_/g, ' ')
    return <span className={statusClass}>{statusLabel}</span>
  }

  /** Normalize progress to 0–100 for display; always show a number. */
  const progressPercent = (job: ActiveJob) => {
    const p = typeof job.progress === 'number' && !Number.isNaN(job.progress) ? job.progress : 0
    return Math.min(100, Math.max(0, p))
  }

  const selectedJob = activeJobs.find(job => job.document_id === selectedJobId)

  return (
    <div className="live-updates-tab">
      <div className="live-updates-layout">
        {/* Left Sidebar - Active Jobs List */}
        <div className="jobs-sidebar">
          <h3 className="sidebar-title">Jobs</h3>
          {activeJobs.length === 0 ? (
            <div className="empty-jobs">
              <p>No jobs</p>
              <p className="empty-hint">Start a chunking job from the Document Status tab. Completed and in-progress jobs appear here.</p>
            </div>
          ) : (
            <div className="jobs-list">
              {activeJobs.map((job) => (
                <div
                  key={job.document_id}
                  className={`job-item ${selectedJobId === job.document_id ? 'active' : ''}`}
                  onClick={() => onJobSelect(job.document_id)}
                >
                  <div className="job-header">
                    <div className="job-name">{job.filename}</div>
                    {getJobStatusBadge(job.status)}
                  </div>
                  <div className="job-progress">
                    <div className="progress-bar">
                      <div 
                        className="progress-fill" 
                        style={{ width: `${progressPercent(job)}%` }}
                      />
                      {job.completed_paragraphs !== undefined && job.total_paragraphs != null && (
                        <span className="progress-text-overlay">
                          {job.completed_paragraphs}/{job.total_paragraphs}
                        </span>
                      )}
                    </div>
                    <div className="progress-details">
                      <span className="progress-text">{progressPercent(job)}%</span>
                      {job.current_page != null && job.total_pages != null && (
                        <span className="progress-page-info">
                          Page {job.current_page}/{job.total_pages}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="job-time">
                    Started: {new Date(job.start_time).toLocaleTimeString()}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right Panel - Job Details & Streaming */}
        <div className="job-details-panel">
          {selectedJob ? (
            <>
              <div className="job-metadata-card">
                <h3 className="metadata-title">Document Information</h3>
                <div className="metadata-grid">
                  <div className="metadata-item">
                    <span className="metadata-label">Filename:</span>
                    <span className="metadata-value">{selectedJob.filename}</span>
                  </div>
                  <div className="metadata-item">
                    <span className="metadata-label">Status:</span>
                    {getJobStatusBadge(selectedJob.status)}
                  </div>
                  {selectedJob.current_page && selectedJob.total_pages && (
                    <div className="metadata-item">
                      <span className="metadata-label">Current Page:</span>
                      <span className="metadata-value">
                        Page {selectedJob.current_page} of {selectedJob.total_pages}
                      </span>
                    </div>
                  )}
                  {selectedJob.current_paragraph && (() => {
                    const paraInfo = parseParagraphId(selectedJob.current_paragraph)
                    return paraInfo && (
                      <div className="metadata-item">
                        <span className="metadata-label">Current Paragraph:</span>
                        <span className="metadata-value">
                          Paragraph {paraInfo.paragraph + 1} on page {paraInfo.page}
                        </span>
                      </div>
                    )
                  })()}
                  <div className="metadata-item">
                    <span className="metadata-label">Progress:</span>
                    <span className="metadata-value">
                      {selectedJob.completed_paragraphs !== undefined && selectedJob.total_paragraphs != null
                        ? `${selectedJob.completed_paragraphs} of ${selectedJob.total_paragraphs} paragraphs (${progressPercent(selectedJob)}%)`
                        : `${progressPercent(selectedJob)}%`}
                    </span>
                  </div>
                  {selectedJob.processing_stage && selectedJob.processing_stage !== 'idle' && (
                    <div className="metadata-item">
                      <span className="metadata-label">Stage:</span>
                      <span className={`stage-badge stage-${selectedJob.processing_stage}`}>
                        {selectedJob.processing_stage.charAt(0).toUpperCase() + selectedJob.processing_stage.slice(1)}
                      </span>
                    </div>
                  )}
                  <div className="metadata-item">
                    <span className="metadata-label">Started:</span>
                    <span className="metadata-value">
                      {new Date(selectedJob.start_time).toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>

              <div className="streaming-container">
                <div className="streaming-header">
                  <h3 className="streaming-title">Live Output</h3>
                  <div className="streaming-indicator">
                    <span className="streaming-dot"></span>
                    Streaming...
                  </div>
                </div>
                <div className="streaming-output" ref={streamRef}>
                  {streamingOutput ? (
                    <pre className="streaming-pre">
                      {isTruncated && (
                        <span className="output-truncated">
                          {'... (showing last 2500 characters, older content hidden) ...\n\n'}
                        </span>
                      )}
                      {limitedOutput}
                    </pre>
                  ) : (
                    <div className="streaming-empty">
                      Waiting for output...
                    </div>
                  )}
                </div>
              </div>
            </>
          ) : (
            <div className="no-job-selected">
              <p>Select a job from the left to view live updates</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
