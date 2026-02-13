import { useState, useEffect, useMemo, useRef } from 'react'
import { API_BASE } from './config'
import './App.css'
import { Header } from './components/Header'
import { Tabs, TabList, Tab, TabPanels, TabPanel } from './components/Tabs'
import { DocumentInputTab } from './components/tabs/DocumentInputTab'
import { DocumentStatusTab, type ChunkingOptions } from './components/tabs/DocumentStatusTab'
import { LiveUpdatesTab } from './components/tabs/LiveUpdatesTab'
import { ReadDocumentTab } from './components/tabs/ReadDocumentTab'
import { ReviewFactsTab } from './components/tabs/ReviewFactsTab'
import { DatabaseLayerTab } from './components/tabs/DatabaseLayerTab'
import { ErrorReviewTab } from './components/tabs/ErrorReviewTab'
import { DocumentDetailTab } from './components/tabs/DocumentDetailTab'

interface UploadResponse {
  filename: string
  content_type: string
  size: number
  gcs_path: string
  document_id: string
  status: string
}

interface StatusResponse {
  document_id: string
  filename: string
  status: string
  pages_extracted: number
  pages_summary?: {
    total: number
    successful: number
    failed: number
    empty: number
  }
  problematic_pages?: Array<{
    page_number: number
    status: string
    error: string | null
    text_length: number
  }>
  created_at: string
}

type CategoryAssessment = Record<string, { score: number; note: string | null }>

interface RetryEntry {
  retry_count: number
  feedback: string | null
  extraction: { summary: string; facts: any[] }
  critique: { pass: boolean; score: number; category_assessment?: CategoryAssessment; feedback: string | null; issues: any[] } | null
}

interface ParagraphState {
  paragraph_id: string
  paragraph_text: string | null
  raw_llm_output: string
  is_streaming: boolean
  extraction_complete: boolean
  summary: string | null
  facts: any[]
  critique_status: "pending" | "reviewing" | "complete"
  critique_result: {
    pass: boolean
    score: number
    category_assessment?: CategoryAssessment
    feedback: string | null
    issues: any[]
  } | null
  retries: RetryEntry[]
  final_status: "pending" | "passed" | "failed"
  needs_human_review: boolean
}

function App() {
  // Tab state
  const [activeTab, setActiveTab] = useState<'input' | 'status' | 'live' | 'read' | 'review' | 'detail' | 'database' | 'errors'>('input')
  const [detailDocumentId, setDetailDocumentId] = useState<string | null>(null)
  
  // Upload state
  const [_file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState<UploadResponse | null>(null)
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  
  // Document viewing state
  const [_pages, _setPages] = useState<any[]>([])
  const [_pageZoom, _setPageZoom] = useState(1.0)
  
  // Document list state
  const [documents, setDocuments] = useState<any[]>([])
  const [_loadingDocuments, setLoadingDocuments] = useState(false)
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null)
  const [_selectedDocument, setSelectedDocument] = useState<any | null>(null)
  const [navigateToRead, setNavigateToRead] = useState<{ documentId: string; pageNumber?: number; factId?: string } | null>(null)
  
  // Chunking state
  const [chunkingActive, setChunkingActive] = useState(false)
  const [chunkingComplete, setChunkingComplete] = useState(false)
  const [_totalParagraphs, setTotalParagraphs] = useState<number | null>(null)
  const [paragraphs, setParagraphs] = useState<Map<string, ParagraphState>>(new Map())
  const [defaultLlmConfigVersion, setDefaultLlmConfigVersion] = useState<string | null>(null)
  const [selectedParagraphId, setSelectedParagraphId] = useState<string | null>(null)
  const [processingParagraphId, setProcessingParagraphId] = useState<string | null>(null)
  const [processingStage, setProcessingStage] = useState<'idle' | 'extracting' | 'critiquing' | 'retrying'>('idle')
  const [_processingRetryCount, setProcessingRetryCount] = useState(0)
  const [_chunkingStatus, setChunkingStatus] = useState<'idle' | 'in_progress' | 'completed' | 'stopped' | 'failed'>('idle')
  const streamContainerRef = useRef<HTMLDivElement>(null)
  const eventSourceRef = useRef<EventSource | null>(null)
  const [sseConnected, setSseConnected] = useState(false)

  // Live updates state
  const [activeJobs, setActiveJobs] = useState<Array<{
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
  }>>([])
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null)
  const [streamingOutput, setStreamingOutput] = useState('')
  
  // Chunking UI state (for old chunking view - will be removed)
  const [pageFilter, _setPageFilter] = useState<number | 'all'>('all')
  const [summarySearch, _setSummarySearch] = useState<string>('')

  // Poll extraction status only when actively extracting
  useEffect(() => {
    if (!result?.document_id) return
    
    let intervalId: ReturnType<typeof setInterval> | null = null
    
    const pollStatus = async () => {
      try {
        const response = await fetch(`${API_BASE}/documents/${result.document_id}/status`)
        if (response.ok) {
          const statusData: StatusResponse = await response.json()
          setStatus(statusData)
          
          // Stop polling if completed or failed
          if (statusData.status === 'completed' || statusData.status === 'failed') {
            if (intervalId) {
              clearInterval(intervalId)
            }
            return
          }
          
          // Only continue polling if status is "extracting"
          if (statusData.status !== 'extracting') {
            if (intervalId) {
              clearInterval(intervalId)
            }
            return
          }
        }
      } catch (err) {
        console.error('Failed to fetch status:', err)
      }
    }
    
    // Only start polling if status is extracting, otherwise just fetch once
    if (status?.status === 'extracting' || result?.status === 'extracting') {
      // Poll immediately, then every 3 seconds (increased from 2s)
      pollStatus()
      intervalId = setInterval(pollStatus, 3000)
    } else {
      // Just fetch once if not extracting
      pollStatus()
    }
    
    return () => {
      if (intervalId) {
        clearInterval(intervalId)
      }
    }
  }, [result?.document_id, status?.status, result?.status])
  
  


  const loadDocuments = async () => {
    setLoadingDocuments(true)
    try {
      const response = await fetch(`${API_BASE}/documents`)
      if (response.ok) {
        const data = await response.json()
        setDocuments(data.documents || [])
      }
    } catch (err) {
      setError('Failed to load documents')
    } finally {
      setLoadingDocuments(false)
    }
  }

  // Load documents on mount
  useEffect(() => {
    loadDocuments()
  }, [])

  // Refresh documents when chunking is active or Live tab is visible so status and progress stay correct
  useEffect(() => {
    if (!chunkingActive && activeTab !== 'live') return
    const interval = setInterval(loadDocuments, 3000)
    return () => clearInterval(interval)
  }, [chunkingActive, activeTab])

  const _restartExtraction = async (documentId: string) => {
    try {
      const response = await fetch(`${API_BASE}/documents/${documentId}/extract/restart`, {
        method: 'POST'
      })
      if (response.ok) {
        await loadDocuments() // Refresh list
      } else {
        const errorData = await response.json().catch(() => null)
        setError(errorData?.detail || 'Failed to restart extraction')
      }
    } catch (err) {
      setError('Failed to restart extraction')
    }
  }

  const _startChunkingForDocument = async (documentId: string) => {
    // Find the document in the list
    const doc = documents.find(d => d.id === documentId)
    if (doc) {
      // Select the document first to load its data
      await selectDocument(documentId, doc)
    }
    // Use existing startChunking function
    await startChunking(documentId)
    await loadDocuments() // Refresh list
  }

  const restartChunkingForDocument = async (documentId: string, options?: ChunkingOptions) => {
    try {
      const opts = options ?? { threshold: 0.6, critiqueEnabled: true, maxRetries: 2 }
      const restartBody: Record<string, unknown> = {
        threshold: opts.threshold,
        critique_enabled: opts.critiqueEnabled,
        max_retries: Math.max(0, opts.maxRetries),
      }
      if (defaultLlmConfigVersion) restartBody.llm_config_version = defaultLlmConfigVersion
      if (opts.promptVersions && Object.keys(opts.promptVersions).length > 0) restartBody.prompt_versions = opts.promptVersions
      const response = await fetch(
        `${API_BASE}/documents/${documentId}/chunking/restart`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(restartBody),
        }
      )
      if (response.ok) {
        // Connect to live stream if not already connected
        setChunkingActive(true)
        setChunkingComplete(false)
        setStreamingOutput('')
        // SSE will reconnect automatically when selectedJobId / activeTab triggers the effect
        await loadDocuments()
      } else {
        const errorData = await response.json().catch(() => null)
        setError(errorData?.detail || 'Failed to restart chunking')
      }
    } catch (err) {
      setError('Failed to restart chunking')
    }
  }

  const deleteDocument = async (documentId: string) => {
    if (!confirm('Are you sure you want to delete this document and all its data?')) {
      return
    }
    try {
      const response = await fetch(`${API_BASE}/admin/db/documents/${documentId}/delete-cascade`, {
        method: 'POST'
      })
      if (response.ok) {
        await loadDocuments() // Refresh list
        // Clear selection if deleted document was selected
        if (selectedDocumentId === documentId) {
          setSelectedDocumentId(null)
          setSelectedDocument(null)
          setResult(null)
          setStatus(null)
          setChunkingActive(false)
          setChunkingComplete(false)
          setParagraphs(new Map())
          // SSE will close when selectedJobId changes via the useEffect cleanup
          if (eventSourceRef.current) {
            eventSourceRef.current.close()
            eventSourceRef.current = null
          }
        }
      } else {
        const errorData = await response.json().catch(() => null)
        setError(errorData?.detail || 'Failed to delete document')
      }
    } catch (err) {
      setError('Failed to delete document')
    }
  }

  const selectDocument = async (documentId: string, documentData: any) => {
    // If clicking the same document, deselect it
    if (selectedDocumentId === documentId) {
      setSelectedDocumentId(null)
      setSelectedDocument(null)
      setResult(null)
      setStatus(null)
      setChunkingActive(false)
      setChunkingComplete(false)
      setParagraphs(new Map())
      if (eventSourceRef.current) {
        eventSourceRef.current.close()
        eventSourceRef.current = null
      }
      return
    }

    setSelectedDocumentId(documentId)
    setSelectedDocument(documentData)
    // NOTE: Do NOT reset selectedPage here - preserve user's page selection in View Documents section
    
    // Create result-like object for View Documents section
    setResult({
      filename: documentData.display_name?.trim() || documentData.filename,
      content_type: 'application/pdf', // Default, could be fetched if needed
      size: 0, // Could be fetched if needed
      gcs_path: documentData.gcs_path || '',
      document_id: documentId,
      status: documentData.extraction_status || 'uploaded'
    })
    
    // Load extraction status
    try {
      const statusResponse = await fetch(`${API_BASE}/documents/${documentId}/status`)
      if (statusResponse.ok) {
        const statusData: StatusResponse = await statusResponse.json()
        setStatus(statusData)
      }
    } catch (err) {
      console.error('Failed to load extraction status:', err)
    }
    
    // Always try to load chunks from PostgreSQL when selecting a document
    await loadChunkingResults(documentId, false)
    // When user opens Live tab and selects this job, polling will load events from the API
  }


  // Handle stream events from SSE
  const handleStreamEvent = (eventData: { event: string; data: any }) => {
    const { event, data } = eventData
    
    switch (event) {
      case 'llm_stream':
        // Update raw_llm_output for the current paragraph (stored in paragraphs Map)
        setParagraphs(prev => {
          const newMap = new Map(prev)
          const paraId = data.paragraph_id || processingParagraphId
          if (paraId) {
            const current = newMap.get(paraId) || {
              paragraph_id: paraId,
              paragraph_text: null,
              raw_llm_output: '',
              is_streaming: true,
              extraction_complete: false,
              summary: null,
              facts: [],
              critique_status: 'pending' as const,
              critique_result: null,
              retries: [],
              final_status: 'pending' as const,
              needs_human_review: false
            }
            const newOutput = current.raw_llm_output + (data.chunk || '')
            newMap.set(paraId, {
              ...current,
              raw_llm_output: newOutput,
              is_streaming: true
            })
          }
          return newMap
        })
        break
        
      case 'paragraph_start':
        setProcessingParagraphId(data.paragraph_id)
        setSelectedParagraphId(data.paragraph_id)
        setProcessingStage('extracting')
        
        // Update activeJobs with progress info from paragraph_start event
        if (data.total_paragraphs !== undefined && data.completed_paragraphs !== undefined) {
          setActiveJobs(prev => prev.map(job => 
            job.document_id === selectedJobId
              ? {
                  ...job,
                  current_page: data.page_number,
                  total_pages: data.total_pages,
                  current_paragraph: data.paragraph_id,
                  total_paragraphs: data.total_paragraphs,
                  completed_paragraphs: data.completed_paragraphs,
                  progress: data.progress_percent || 0,
                  processing_stage: 'extracting'
                }
              : job
          ))
        }
        setParagraphs(prev => {
          const newMap = new Map(prev)
          newMap.set(data.paragraph_id, {
            paragraph_id: data.paragraph_id,
            paragraph_text: data.paragraph_text,
            raw_llm_output: '',
            is_streaming: true,
            extraction_complete: false,
            summary: null,
            facts: [],
            critique_status: 'pending' as const,
            critique_result: null,
            retries: [],
            final_status: 'pending' as const,
            needs_human_review: false
          })
          return newMap
        })
        // Clear streaming output when starting a new paragraph (will be filled by llm_stream events)
        if (selectedJobId) {
          setStreamingOutput('')
        }
        break
        
      case 'extraction_complete':
        setParagraphs(prev => {
          const newMap = new Map(prev)
          const current = newMap.get(data.paragraph_id)
          if (current) {
            newMap.set(data.paragraph_id, {
              ...current,
              is_streaming: false,
              extraction_complete: true,
              summary: data.summary,
              facts: data.facts || []
            })
          }
          return newMap
        })
        break
        
      case 'critique_start':
        setProcessingStage('critiquing')
        // Update activeJobs with critiquing stage
        setActiveJobs(prev => prev.map(job => 
          job.document_id === selectedJobId
            ? { ...job, processing_stage: 'critiquing' }
            : job
        ))
        setParagraphs(prev => {
          const newMap = new Map(prev)
          const current = newMap.get(data.paragraph_id)
          if (current) {
            newMap.set(data.paragraph_id, {
              ...current,
              critique_status: 'reviewing' as const
            })
          }
          return newMap
        })
        break
        
      case 'critique_complete':
        setParagraphs(prev => {
          const newMap = new Map(prev)
          const current = newMap.get(data.paragraph_id)
          if (current) {
            newMap.set(data.paragraph_id, {
              ...current,
              critique_status: 'complete' as const,
              critique_result: {
                pass: data.pass,
                score: data.score,
                category_assessment: data.category_assessment || {},
                feedback: data.feedback,
                issues: data.issues || []
              }
            })
          }
          return newMap
        })
        // Update activeJobs - critique complete, moving to next stage or idle
        setActiveJobs(prev => prev.map(job => 
          job.document_id === selectedJobId
            ? { ...job, processing_stage: 'idle' }
            : job
        ))
        break
        
      case 'retry_start':
        setProcessingStage('retrying')
        setProcessingRetryCount(data.retry_count)
        // Update activeJobs with retrying stage
        setActiveJobs(prev => prev.map(job => 
          job.document_id === selectedJobId
            ? { ...job, processing_stage: 'retrying' }
            : job
        ))
        setParagraphs(prev => {
          const newMap = new Map(prev)
          const current = newMap.get(data.paragraph_id)
          if (current) {
            const retryEntry = {
              retry_count: data.retry_count,
              feedback: data.feedback,
              extraction: { summary: '', facts: [] },
              critique: null
            }
            newMap.set(data.paragraph_id, {
              ...current,
              retries: [...(current.retries || []), retryEntry]
            })
          }
          return newMap
        })
        break
        
      case 'retry_extraction_complete':
        setParagraphs(prev => {
          const newMap = new Map(prev)
          const current = newMap.get(data.paragraph_id)
          if (current && current.retries && current.retries.length > 0) {
            const lastRetry = current.retries[current.retries.length - 1]
            lastRetry.extraction = {
              summary: data.summary,
              facts: data.facts || []
            }
            newMap.set(data.paragraph_id, {
              ...current,
              retries: [...current.retries]
            })
          }
          return newMap
        })
        break
        
      case 'paragraph_complete':
        setProcessingStage('idle')
        setParagraphs(prev => {
          const newMap = new Map(prev)
          const current = newMap.get(data.paragraph_id)
          if (current) {
            newMap.set(data.paragraph_id, {
              ...current,
              final_status: data.status as 'passed' | 'failed',
              needs_human_review: data.needs_human_review || false
            })
          }
          return newMap
        })
        // Update activeJobs with progress from event + idle stage
        if (selectedJobId) {
          setActiveJobs(prev => prev.map(job =>
            job.document_id === selectedJobId
              ? {
                  ...job,
                  processing_stage: 'idle',
                  ...(data.completed_paragraphs != null ? { completed_paragraphs: data.completed_paragraphs } : {}),
                  ...(data.total_paragraphs != null ? { total_paragraphs: data.total_paragraphs } : {}),
                  ...(data.progress_percent != null ? { progress: Math.round(data.progress_percent) } : {}),
                }
              : job
          ))
        }
        break
        
      case 'paragraph_persisted':
        // Update the paragraph data in state directly from the event, avoiding full reload
        // This is more efficient and doesn't disrupt the user's view
        if (data.paragraph_id) {
          setParagraphs(prev => {
            const newMap = new Map(prev)
            const existing = newMap.get(data.paragraph_id)
            if (existing) {
              // Update the paragraph with persisted data, but preserve streaming state
              newMap.set(data.paragraph_id, {
                ...existing,
                // Facts will be updated when user views the paragraph or on next full load
                // We don't reload here to avoid disrupting the user's view
              })
            }
            return newMap
          })
        }
        // Update activeJobs with persisting stage
        setActiveJobs(prev => prev.map(job => 
          job.document_id === selectedJobId
            ? { ...job, processing_stage: 'persisting' }
            : job
        ))
        // Optionally reload from DB in the background (debounced) to get full facts
        // But don't do it immediately to avoid disrupting user's view
        break
        
      case 'job_start':
        // Mark job as in_progress as soon as the worker picks it up
        setChunkingActive(true)
        setChunkingComplete(false)
        if (selectedJobId) {
          setActiveJobs(prev => prev.map(job =>
            job.document_id === selectedJobId
              ? { ...job, status: 'in_progress', total_paragraphs: data.total_paragraphs, total_pages: data.total_pages }
              : job
          ))
        }
        break

      case 'status_message':
        // State updates only – log output is handled by the SSE event handler
        break

      case 'progress_update':
        // Update activeJobs with progress information
        setActiveJobs(prev => prev.map(job => 
          job.document_id === selectedJobId
            ? {
                ...job,
                current_page: data.current_page,
                total_pages: data.total_pages,
                current_paragraph: data.current_paragraph,
                total_paragraphs: data.total_paragraphs,
                completed_paragraphs: data.completed_paragraphs,
                progress: data.progress_percent
              }
            : job
        ))
        break
      
      case 'chunking_complete':
        setChunkingActive(false)
        setChunkingComplete(true)
        // Update status to completed (will switch to embedding status if embedding starts)
        if (selectedJobId) {
          setActiveJobs(prev => prev.map(job =>
            job.document_id === selectedJobId
              ? { ...job, status: 'completed', progress: 100 }
              : job
          ))
        }
        break

      case 'job_failed':
      case 'job_timeout':
        setChunkingActive(false)
        if (selectedJobId) {
          setActiveJobs(prev => prev.map(job =>
            job.document_id === selectedJobId
              ? { ...job, status: 'failed' }
              : job
          ))
        }
        break

      case 'embedding_start':
        // Switch status to show embedding is happening
        if (selectedJobId) {
          setActiveJobs(prev => prev.map(job =>
            job.document_id === selectedJobId
              ? {
                  ...job,
                  status: 'processing',
                  total_paragraphs: data?.total_items ?? job.total_paragraphs,
                  completed_paragraphs: 0,
                  progress: 0,
                }
              : job
          ))
        }
        break

      case 'embedding_progress':
        if (data?.completed_items != null && data?.total_items != null && selectedJobId) {
          setActiveJobs(prev => prev.map(job =>
            job.document_id === selectedJobId
              ? {
                  ...job,
                  total_paragraphs: data.total_items,
                  completed_paragraphs: data.completed_items,
                  progress: data.total_items > 0 ? Math.round((data.completed_items / data.total_items) * 100) : 0,
                }
              : job
          ))
        }
        break

      case 'embedding_complete':
      case 'embedding_error':
        if (selectedJobId) {
          setActiveJobs(prev => prev.map(job =>
            job.document_id === selectedJobId
              ? { ...job, status: event === 'embedding_complete' ? 'completed' : 'failed' }
              : job
          ))
        }
        break

      case 'stream_end': {
        const reason = data?.reason
        if (reason === 'chunking_complete') {
          setChunkingActive(false)
          setChunkingComplete(true)
        }
        break
      }
    }
  }

  // Keep a ref to the latest handleStreamEvent so SSE callbacks avoid stale closures
  const handleStreamEventRef = useRef(handleStreamEvent)
  useEffect(() => { handleStreamEventRef.current = handleStreamEvent })

  // Convert an SSE event into a human-readable log line (or null to skip).
  // Prefers `user_message` (set by the worker for user-facing text) over raw `message`.
  const eventToLogLine = (event: string, data: any): string | null => {
    // Most events carry a user_message — use it directly when available
    const userMsg = data?.user_message ? String(data.user_message).trim() : ''

    switch (event) {
      case 'status_message':
        return userMsg || (data?.message ? String(data.message).trim() : null)

      case 'job_start':
        return userMsg || 'Starting document analysis...'

      case 'paragraph_start':
        return userMsg || (data?.paragraph_id
          ? `Analyzing section ${data.paragraph_id} (page ${data.page_number ?? '?'})...`
          : null)

      case 'paragraph_complete':
        return userMsg || (data?.paragraph_id
          ? `Section ${data.paragraph_id} complete.`
          : null)

      case 'chunking_complete':
        return userMsg || 'Document analysis complete.'

      case 'job_failed':
      case 'job_timeout':
        return userMsg || (data?.message ? String(data.message).trim() : 'Processing failed.')

      case 'embedding_start':
      case 'embedding_progress':
      case 'embedding_complete':
        return data?.message ? String(data.message) : null

      case 'embedding_error':
        return data?.message ? `Error: ${data.message}` : null

      case 'stream_end':
        return data?.reason ? `Stream ended: ${data.reason}` : null

      default:
        // Unknown event — surface user_message if present
        return userMsg || null
    }
  }

  // Load chunking results from DB and update state
  const loadChunkingResults = async (documentId: string, _mergeWithExisting: boolean = false) => {
    try {
      const response = await fetch(`${API_BASE}/documents/${documentId}/chunking/results`)
      if (!response.ok) {
        return
      }
      const data = await response.json()
      const metadata = data.metadata || {}
      const results = data.results || {}
      const paragraphsData = results.paragraphs || {}
      
      // Update status
      const status = metadata.status || 'idle'
      setChunkingStatus(status as 'idle' | 'in_progress' | 'completed' | 'stopped' | 'failed')
      setTotalParagraphs(metadata.total_paragraphs || null)
      
      // Update chunking active/complete flags
      if (status === 'in_progress') {
        setChunkingActive(true)
        setChunkingComplete(false)
      } else if (status === 'completed') {
        setChunkingActive(false)
        setChunkingComplete(true)
      } else if (status === 'stopped' || status === 'failed') {
        setChunkingActive(false)
        setChunkingComplete(false)
      }
      
      // When loading from DB, replace local state (worker writes to PostgreSQL; we poll events)
      const shouldMerge = false
      
      if (shouldMerge) {
        // Merge: only update paragraphs that don't exist or are complete (not streaming)
        setParagraphs(prev => {
          const newMap = new Map(prev)
          for (const [paraId, paraData] of Object.entries(paragraphsData)) {
            const p = paraData as any
            const existing = newMap.get(paraId)
            // Only replace if paragraph is complete (not streaming) or doesn't exist
            if (!existing || existing.final_status !== 'pending') {
              newMap.set(paraId, {
                paragraph_id: paraId,
                paragraph_text: p.paragraph_text || null,
                raw_llm_output: existing?.raw_llm_output || "", // Preserve streaming output if exists
                is_streaming: existing?.is_streaming || false, // Preserve streaming state
                extraction_complete: !!p.summary,
                summary: p.summary || null,
                facts: p.facts || [],
                critique_status: p.critique_result ? "complete" : (existing?.critique_status || "pending"),
                critique_result: p.critique_result ? {
                  pass: p.critique_result.pass || false,
                  score: p.critique_result.score || 0.5,
                  category_assessment: p.critique_result.category_assessment || {},
                  feedback: p.critique_result.feedback || null,
                  issues: p.critique_result.issues || []
                } : (existing?.critique_result || null),
                retries: p.retries || [],
                final_status: p.final_status || "pending",
                needs_human_review: p.needs_human_review || false
              })
            }
          }
          return newMap
        })
      } else {
        // Replace: full load from DB (initial load or SSE not active)
        const newParagraphs = new Map<string, ParagraphState>()
        for (const [paraId, paraData] of Object.entries(paragraphsData)) {
          const p = paraData as any
          newParagraphs.set(paraId, {
            paragraph_id: paraId,
            paragraph_text: p.paragraph_text || null,
            raw_llm_output: "", // Not stored in DB
            is_streaming: false,
            extraction_complete: !!p.summary,
            summary: p.summary || null,
            facts: p.facts || [],
            critique_status: p.critique_result ? "complete" : "pending",
            critique_result: p.critique_result ? {
              pass: p.critique_result.pass || false,
              score: p.critique_result.score || 0.5,
              category_assessment: p.critique_result.category_assessment || {},
              feedback: p.critique_result.feedback || null,
              issues: p.critique_result.issues || []
            } : null,
            retries: p.retries || [],
            final_status: p.final_status || "pending",
            needs_human_review: p.needs_human_review || false
          })
        }
        setParagraphs(newParagraphs)
      }
      
      // Set processing state if in progress (only if not merging to avoid overwriting SSE state)
      // NOTE: This only affects Chunking section state, NOT View Documents section's selectedPage
      // IMPORTANT: When merging (shouldMerge=true), NEVER modify any selection state to preserve user's view
      if (!shouldMerge && status === 'in_progress' && Object.keys(paragraphsData).length > 0) {
        // Find the last paragraph (likely the one being processed)
        const paraIds = Object.keys(paragraphsData).sort()
        const lastParaId = paraIds[paraIds.length - 1]
        // Only set if not already set (preserve user's selection)
        if (!processingParagraphId) {
          setProcessingParagraphId(lastParaId)
        }
        if (!selectedParagraphId) {
          setSelectedParagraphId(lastParaId)
        }
        setProcessingStage('extracting')
      } else if (!shouldMerge) {
        // Only reset if not merging (SSE will handle state during streaming)
        // Don't reset if user has a selection
        if (!selectedParagraphId) {
          setProcessingParagraphId(null)
        }
        if (!processingParagraphId) {
          setProcessingStage('idle')
        }
      }
      // IMPORTANT: Do NOT modify selectedPage here - it's independent Extract section state
      // IMPORTANT: When shouldMerge=true, do NOT modify ANY selection state (selectedPage, selectedParagraphId, processingParagraphId)
    } catch (err) {
      console.error("Failed to load chunking results:", err)
    }
  }

  const startChunking = async (documentId: string, options?: ChunkingOptions) => {
    setError(null)
    const opts = options ?? { generatorId: 'A', threshold: 0.6, critiqueEnabled: true, maxRetries: 2, extractionEnabled: true }
    console.log(`Starting chunking for document ${documentId}...`)
    
    try {
      const gen = opts.generatorId ?? 'A'
      const body: Record<string, unknown> = {
        threshold: opts.threshold,
        critique_enabled: opts.critiqueEnabled,
        max_retries: Math.max(0, opts.maxRetries),
        generator_id: gen,
        extraction_enabled: gen === 'B' ? false : (opts.extractionEnabled ?? true),
      }
      if (defaultLlmConfigVersion) body.llm_config_version = defaultLlmConfigVersion
      if (opts.promptVersions && Object.keys(opts.promptVersions).length > 0) body.prompt_versions = opts.promptVersions
      const response = await fetch(
        `${API_BASE}/documents/${documentId}/chunking/start?threshold=${encodeURIComponent(opts.threshold)}&critique_enabled=${opts.critiqueEnabled}&max_retries=${opts.maxRetries}&generator_id=${encodeURIComponent(gen)}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        }
      )
      
      console.log(`Start chunking response status: ${response.status}`)
      
      if (!response.ok) {
        if (response.status === 409) {
          setError('Chunking is already in progress for this document.')
        } else {
          const errorData = await response.json().catch(() => null)
          const errorMsg = errorData?.detail || `Failed to start chunking (${response.status})`
          console.error('Start chunking error:', errorMsg)
          setError(errorMsg)
        }
        return
      }
      
      const data = await response.json()
      console.log('Start chunking response:', data)
      
      if (data.status === 'started' || data.status === 'queued') {
        // Set chunking as active immediately
        setChunkingActive(true)
        setChunkingComplete(false)
        
        // Load initial state from DB once (full replace, not merge)
        await loadChunkingResults(documentId, false)
        // Live updates: when user opens Live tab and selects this job, polling will start
      }
    } catch (err) {
      console.error('Start chunking exception:', err)
      setError(err instanceof Error ? err.message : 'Failed to start chunking')
    }
  }

  const stopChunking = async (documentId: string) => {
    try {
      const response = await fetch(
        `${API_BASE}/documents/${documentId}/chunking/stop`,
        { method: 'POST' }
      )
      if (response.ok) {
        if (eventSourceRef.current) {
          eventSourceRef.current.close()
          eventSourceRef.current = null
        }
        await loadChunkingResults(documentId)
      }
    } catch (err) {
      console.error("Failed to stop chunking:", err)
    }
  }

  // Load results when chunking section is opened (worker writes to PostgreSQL)
  useEffect(() => {
    if (result?.document_id && (chunkingActive || chunkingComplete)) {
      loadChunkingResults(result.document_id, false)
    }
  }, [result?.document_id, chunkingActive, chunkingComplete])

  // Autoscroll for live stream container (when raw_llm_output changes)
  useEffect(() => {
    if (streamContainerRef.current && processingParagraphId) {
      const cur = paragraphs.get(processingParagraphId)
      const output = cur?.raw_llm_output || ""
      if (output) {
        // Use requestAnimationFrame to ensure DOM has updated
        requestAnimationFrame(() => {
          if (streamContainerRef.current) {
            streamContainerRef.current.scrollTo({
              top: streamContainerRef.current.scrollHeight,
              behavior: 'smooth'
            })
          }
        })
      }
    }
  }, [processingParagraphId, paragraphs])

  const paragraphsByPage = useMemo(() => {
    const map = new Map<number, ParagraphState[]>()
    for (const p of paragraphs.values()) {
      const [pageStr] = p.paragraph_id.split('_')
      const page = parseInt(pageStr, 10)
      if (!map.has(page)) map.set(page, [])
      map.get(page)!.push(p)
    }
    const entries = Array.from(map.entries()).sort((a, b) => a[0] - b[0])
    entries.forEach(([, arr]) =>
      arr.sort((a, b) => {
        const iA = parseInt(a.paragraph_id.split('_')[1], 10)
        const iB = parseInt(b.paragraph_id.split('_')[1], 10)
        return iA - iB
      })
    )
    return entries
  }, [paragraphs])

  const _availablePages = useMemo(() => {
    return [...new Set(paragraphsByPage.map(([p]) => p))].sort((a, b) => a - b)
  }, [paragraphsByPage])

  const _filteredParagraphsByPage = useMemo(() => {
    let filtered = paragraphsByPage

    // Apply page filter
    if (pageFilter !== 'all') {
      filtered = filtered.filter(([pageNum]) => pageNum === pageFilter)
    }

    // Apply summary search filter
    if (summarySearch.trim()) {
      const searchLower = summarySearch.toLowerCase().trim()
      filtered = filtered
        .map(([pageNum, paras]) => {
          const matchingParas = paras.filter((para) => {
            const summary = para.summary || ''
            return summary.toLowerCase().includes(searchLower)
          })
          return [pageNum, matchingParas] as [number, ParagraphState[]]
        })
        .filter(([, paras]) => paras.length > 0) // Remove pages with no matching paragraphs
    }

    return filtered
  }, [paragraphsByPage, pageFilter, summarySearch])

  const _completedCount = useMemo(
    () => [...paragraphs.values()].filter((p) => p.final_status !== 'pending').length,
    [paragraphs]
  )

  const _closeChunking = () => {
    setChunkingActive(false)
    setChunkingComplete(false)
    setTotalParagraphs(null)
    setParagraphs(new Map())
    setSelectedParagraphId(null)
    setProcessingParagraphId(null)
    setProcessingStage('idle')
    setProcessingRetryCount(0)
  }

  // Reference unused values so strict noUnusedLocals passes (kept for future use)
  void [_restartExtraction, _startChunkingForDocument, _availablePages, _filteredParagraphsByPage, _completedCount, _closeChunking]

  // Live Updates list: show all documents (processing logs). Filter by job can be added later.
  useEffect(() => {
    const totalParagraphs = (d: { chunking_total_paragraphs?: number }) => d.chunking_total_paragraphs ?? 0
    const completedParagraphs = (d: { chunking_completed_paragraphs?: number }) => d.chunking_completed_paragraphs ?? 0
    const jobs = documents.map(doc => {
      const total = totalParagraphs(doc)
      const completed = completedParagraphs(doc)
      const progress = total > 0 ? Math.min(100, Math.round((completed / total) * 100)) : 0
      const status = (doc.embedding_status === 'processing' || doc.embedding_status === 'pending')
        ? doc.embedding_status
        : (doc.chunking_status || 'idle')
      return {
        document_id: doc.id,
        filename: doc.display_name?.trim() || doc.filename,
        status,
        progress,
        start_time: doc.created_at,
        current_page: doc.chunking_current_page ?? undefined,
        total_pages: doc.chunking_total_pages ?? undefined,
        current_paragraph: doc.chunking_current_paragraph ?? (selectedJobId === doc.id ? processingParagraphId ?? undefined : undefined),
        total_paragraphs: total || undefined,
        completed_paragraphs: total ? completed : undefined,
        processing_stage: (doc.chunking_processing_stage ?? (selectedJobId === doc.id ? processingStage : 'idle')) as 'extracting' | 'critiquing' | 'retrying' | 'persisting' | 'idle'
      }
    })
    jobs.sort((a, b) => new Date(b.start_time).getTime() - new Date(a.start_time).getTime())
    setActiveJobs(jobs)
  }, [documents, selectedJobId, processingParagraphId, processingStage])

  // Only overwrite streamingOutput from paragraph raw_llm_output when chunking is active.
  // When job is completed/idle, the events effect owns streamingOutput (status_message log).
  useEffect(() => {
    if (!selectedJobId) {
      setStreamingOutput('')
      return
    }
    if (!chunkingActive) return // Leave output to events effect (processing log)
    if (processingParagraphId) {
      const para = paragraphs.get(processingParagraphId)
      if (para?.raw_llm_output) {
        setStreamingOutput(para.raw_llm_output)
      }
      return
    }
    const allOutput: string[] = []
    paragraphs.forEach((para) => {
      if (para.raw_llm_output && para.raw_llm_output.length > 0) {
        allOutput.push(para.raw_llm_output)
      }
    })
    if (allOutput.length > 0) {
      setStreamingOutput(allOutput[allOutput.length - 1])
    }
  }, [selectedJobId, chunkingActive, processingParagraphId, paragraphs])

  const handleUpload = async (file: File) => {
    setFile(file)
    setError(null)
    setUploading(true)
    
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => null)
        if (errorData?.detail?.error === 'duplicate_file') {
          setError(
            `This file has already been uploaded as "${errorData.detail.original_filename}".\n\n${errorData.detail.help}`
          )
        } else if (errorData?.detail) {
          const msg = typeof errorData.detail === 'string'
            ? errorData.detail
            : errorData.detail?.message || JSON.stringify(errorData.detail)
          setError(msg)
        } else {
          setError(`Upload failed: ${response.status} ${response.statusText}`)
        }
        return
      }

      const data: UploadResponse = await response.json()
      setResult(data)
      // Refresh document list
      await loadDocuments()
      // Switch to status tab to see the new document
      setActiveTab('status')
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Upload failed. Please try again.'
      setError(
        msg.includes('fetch') || msg.includes('Failed') || msg.includes('NetworkError')
          ? `${msg} Is the backend running at ${API_BASE}?`
          : msg
      )
    } finally {
      setUploading(false)
    }
  }

  const handleStartChunking = async (documentId: string, options: ChunkingOptions) => {
    await startChunking(documentId, options)
    await loadDocuments()
    // Switch to live updates tab
    setActiveTab('live')
    setSelectedJobId(documentId)
  }

  const handleStopChunking = async (documentId: string) => {
    await stopChunking(documentId)
    await loadDocuments()
  }

  const handleRestartChunking = async (documentId: string, options: ChunkingOptions) => {
    await restartChunkingForDocument(documentId, options)
    await loadDocuments()
    // Switch to live updates tab to see the restart
    setActiveTab('live')
    setSelectedJobId(documentId)
  }

  const handleStartEmbedding = async (documentId: string) => {
    try {
      const res = await fetch(`${API_BASE}/documents/${documentId}/embedding/start`, { method: 'POST' })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        setError(data?.detail || `Failed to start embedding (${res.status})`)
        return
      }
      setError(null)
      await loadDocuments()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start embedding')
    }
  }

  const handleResetEmbedding = async (documentId: string) => {
    try {
      const res = await fetch(`${API_BASE}/documents/${documentId}/embedding/reset`, { method: 'POST' })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        setError(data?.detail || `Failed to reset embedding (${res.status})`)
        return
      }
      setError(null)
      await loadDocuments()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reset embedding')
    }
  }

  const handleViewDocument = (documentId: string, options?: { pageNumber?: number; factId?: string }) => {
    setSelectedDocumentId(documentId)
    setNavigateToRead({
      documentId,
      pageNumber: options?.pageNumber,
      factId: options?.factId,
    })
    setActiveTab('read')
    const doc = documents.find(d => d.id === documentId)
    if (doc) {
      selectDocument(documentId, doc)
    }
  }

  const handleViewDocumentDetail = (documentId: string) => {
    setDetailDocumentId(documentId)
    setActiveTab('detail')
  }

  const handleJobSelect = async (documentId: string) => {
    setSelectedJobId(documentId)
    const doc = documents.find(d => d.id === documentId)
    if (doc) {
      // Set selectedDocumentId to match selectedJobId so streaming works
      setSelectedDocumentId(documentId)
      await selectDocument(documentId, doc)
      // When Live tab is active, the effect below will load events and start polling
    }
  }

  // SSE live-event stream (replaces the old polling useEffect)
  // The backend endpoint replays all existing events on connect, then pushes
  // new events in real-time via PostgreSQL LISTEN/NOTIFY.
  useEffect(() => {
    // Close any previous connection
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
      setSseConnected(false)
    }

    if (activeTab !== 'live' || !selectedJobId) {
      setStreamingOutput('')
      return
    }

    const jobId = selectedJobId
    setStreamingOutput('')  // clear before replay

    const es = new EventSource(`${API_BASE}/documents/${jobId}/chunking/events/stream`)
    eventSourceRef.current = es

    es.onopen = () => {
      setSseConnected(true)
    }

    es.onmessage = (msg) => {
      try {
        const parsed = JSON.parse(msg.data)
        const { event, data } = parsed

        // Update component state (paragraphs, activeJobs, flags, …)
        handleStreamEventRef.current({ event, data })

        // Append a human-readable log line (if the event is loggable)
        const line = eventToLogLine(event, data)
        if (line) {
          setStreamingOutput(prev => prev + line + '\n')
        }
      } catch (err) {
        console.error('[SSE] parse error:', err)
      }
    }

    es.onerror = () => {
      // EventSource auto-reconnects with exponential backoff.
      // The server replays all events on reconnect, so no data is lost.
      setSseConnected(false)
    }

    return () => {
      es.close()
      eventSourceRef.current = null
      setSseConnected(false)
    }
  }, [activeTab, selectedJobId])

  // Periodically refresh chunking results while a job is in progress
  // (independent of SSE — detects completion for the status badge / results view)
  useEffect(() => {
    if (activeTab !== 'live' || !selectedJobId) return
    const doc = documents.find(d => d.id === selectedJobId)
    if (!doc) return
    const isInProgress =
      doc.chunking_status === 'in_progress' || doc.chunking_status === 'pending' ||
      doc.embedding_status === 'processing' || doc.embedding_status === 'pending'
    if (!isInProgress) return

    const interval = setInterval(() => {
      loadChunkingResults(selectedJobId, false)
    }, 5000)
    return () => clearInterval(interval)
  }, [activeTab, selectedJobId, documents])

  return (
    <div className="app">
      <Header
        defaultLlmConfigVersion={defaultLlmConfigVersion}
        onDefaultLlmConfigVersionChange={setDefaultLlmConfigVersion}
      />
      
      <Tabs
        activeTab={activeTab}
        onTabChange={(tabId: string) => setActiveTab(tabId as typeof activeTab)}
      >
        <TabList>
          <Tab id="input" isActive={activeTab === 'input'} onClick={() => setActiveTab('input')}>
            Document Input
          </Tab>
          <Tab id="status" isActive={activeTab === 'status'} onClick={() => setActiveTab('status')}>
            Document Status
          </Tab>
          <Tab id="live" isActive={activeTab === 'live'} onClick={() => setActiveTab('live')}>
            Live Updates
          </Tab>
          <Tab id="read" isActive={activeTab === 'read'} onClick={() => setActiveTab('read')}>
            Read Document
          </Tab>
          <Tab id="review" isActive={activeTab === 'review'} onClick={() => setActiveTab('review')}>
            Review Facts
          </Tab>
          <Tab id="detail" isActive={activeTab === 'detail'} onClick={() => setActiveTab('detail')}>
            Document detail
          </Tab>
          <Tab id="database" isActive={activeTab === 'database'} onClick={() => setActiveTab('database')}>
            Database Layer
          </Tab>
          <Tab id="errors" isActive={activeTab === 'errors'} onClick={() => setActiveTab('errors')}>
            Error Review
          </Tab>
        </TabList>

        <TabPanels>
          <TabPanel id="input" isActive={activeTab === 'input'}>
            <DocumentInputTab
              onUpload={handleUpload}
              uploading={uploading}
              error={error}
            />
          </TabPanel>

          <TabPanel id="status" isActive={activeTab === 'status'}>
            <DocumentStatusTab
              onStartChunking={handleStartChunking}
              onStopChunking={handleStopChunking}
              onViewDocument={handleViewDocument}
              onViewDocumentDetail={handleViewDocumentDetail}
              onDeleteDocument={deleteDocument}
              onRestartChunking={handleRestartChunking}
              onStartEmbedding={handleStartEmbedding}
              onResetEmbedding={handleResetEmbedding}
            />
          </TabPanel>

          <TabPanel id="live" isActive={activeTab === 'live'}>
            <LiveUpdatesTab
              activeJobs={activeJobs}
              onJobSelect={handleJobSelect}
              selectedJobId={selectedJobId}
              streamingOutput={streamingOutput}
              isActive={activeTab === 'live'}
              sseConnected={sseConnected}
            />
          </TabPanel>

          <TabPanel id="read" isActive={activeTab === 'read'}>
            <ReadDocumentTab
              documents={documents}
              selectedDocumentId={selectedDocumentId}
              navigateToRead={navigateToRead}
              onNavigateToReadConsumed={() => setNavigateToRead(null)}
              onDocumentSelect={(docId: string) => {
                const doc = documents.find((d: { id: string }) => d.id === docId)
                if (doc) {
                  selectDocument(docId, doc)
                }
              }}
            />
          </TabPanel>

          <TabPanel id="review" isActive={activeTab === 'review'}>
            <ReviewFactsTab
              onViewDocument={handleViewDocument}
            />
          </TabPanel>

          <TabPanel id="detail" isActive={activeTab === 'detail'}>
            <DocumentDetailTab
              documentId={detailDocumentId}
              onViewDocument={handleViewDocument}
              onViewErrors={() => setActiveTab('errors')}
              onViewFacts={(id) => { setDetailDocumentId(id); setActiveTab('review') }}
              onPublishSuccess={() => loadDocuments()}
            />
          </TabPanel>

          <TabPanel id="database" isActive={activeTab === 'database'}>
            <DatabaseLayerTab isActive={activeTab === 'database'} />
          </TabPanel>

          <TabPanel id="errors" isActive={activeTab === 'errors'}>
            <ErrorReviewTab />
          </TabPanel>
        </TabPanels>
      </Tabs>

      {error && activeTab !== 'input' && (
        <div className="error-banner">
          {error}
        </div>
      )}
    </div>
  )
}

export default App
