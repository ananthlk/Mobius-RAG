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

  const jobProgressPercent = (job: ActiveJob) => {
    if (job.completed_paragraphs != null && job.total_paragraphs != null && job.total_paragraphs > 0) {
      return Math.min(100, Math.round((job.completed_paragraphs / job.total_paragraphs) * 100))
    }
    const p = typeof job.progress === 'number' && !Number.isNaN(job.progress) ? job.progress : 0
    return Math.min(100, Math.max(0, p))
  }

  return (
    <div className="live-updates-tab">
      <div className="live-updates-table-wrap">
        {activeJobs.length === 0 ? (
          <div className="empty-jobs">
            <p>No jobs</p>
            <p className="empty-hint">Start a chunking job from the Document Status tab. Jobs appear here with live output.</p>
          </div>
        ) : (
          <table className="live-updates-table">
            <thead>
              <tr>
                <th className="col-doc-name">Document</th>
                <th className="col-started">Started</th>
                <th className="col-status">Status</th>
                <th className="col-output">Live output</th>
              </tr>
            </thead>
            <tbody>
              {activeJobs.map((job) => {
                const isStreamingJob = selectedJobId === job.document_id
                return (
                  <tr
                    key={job.document_id}
                    className={`live-job-row ${isStreamingJob ? 'streaming' : ''}`}
                    onClick={() => onJobSelect(job.document_id)}
                  >
                    <td className="col-doc-name">
                      <span className="job-doc-name" title={job.filename}>{job.filename}</span>
                    </td>
                    <td className="col-started">
                      {new Date(job.start_time).toLocaleString()}
                    </td>
                    <td className="col-status">
                      <div className="status-cell">
                        <div className="status-cell-top">
                          {getJobStatusBadge(job.status)}
                          {job.completed_paragraphs != null && job.total_paragraphs != null && job.total_paragraphs > 0 && (
                            <span className="job-progress-hint">
                              {job.completed_paragraphs}/{job.total_paragraphs} done
                              {job.current_paragraph && (() => {
                                const p = parseParagraphId(job.current_paragraph)
                                return p ? (
                                  <span className="job-current-para" title="Paragraph ID = page_paragraphIndex (e.g. 24_0 = page 24, 1st paragraph). We have completed 12 so far; this is the one we're working on now.">
                                    {' · '}now page {p.page}
                                  </span>
                                ) : null
                              })()}
                            </span>
                          )}
                        </div>
                        <div className="status-progress-bar">
                          <div
                            className="status-progress-fill"
                            style={{ width: `${jobProgressPercent(job)}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="col-output">
                      {isStreamingJob ? (
                        <div className="streaming-cell">
                          <div className="streaming-output" ref={streamRef}>
                            {streamingOutput ? (
                              <pre className="streaming-pre">
                                {isTruncated && (
                                  <span className="output-truncated">
                                    {'... (last 2500 chars) ...\n\n'}
                                  </span>
                                )}
                                {limitedOutput}
                              </pre>
                            ) : (
                              <span className="streaming-empty">Waiting for output...</span>
                            )}
                          </div>
                        </div>
                      ) : (
                        <span className="output-placeholder">—</span>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
