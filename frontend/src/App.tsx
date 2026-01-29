import { useState, useEffect, useMemo, useRef } from 'react'
import './App.css'
import { Header } from './components/Header'
import { Tabs, TabList, Tab, TabPanels, TabPanel } from './components/Tabs'
import { DocumentInputTab } from './components/tabs/DocumentInputTab'
import { DocumentStatusTab } from './components/tabs/DocumentStatusTab'
import { LiveUpdatesTab } from './components/tabs/LiveUpdatesTab'
import { ReadDocumentTab } from './components/tabs/ReadDocumentTab'
import { ReviewFactsTab } from './components/tabs/ReviewFactsTab'
import { DatabaseLayerTab } from './components/tabs/DatabaseLayerTab'
import { ErrorReviewTab } from './components/tabs/ErrorReviewTab'

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
  const [activeTab, setActiveTab] = useState<'input' | 'status' | 'live' | 'read' | 'review' | 'database' | 'errors'>('input')
  
  // Upload state
  const [_file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState<UploadResponse | null>(null)
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  
  // Document viewing state
  const [_pages, _setPages] = useState<any[]>([])
  const [selectedPage, setSelectedPage] = useState<number | null>(null)
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
  const [retryThreshold, setRetryThreshold] = useState(0.6)
  const [critiqueEnabled, setCritiqueEnabled] = useState(true)
  const [maxRetries, setMaxRetries] = useState(2)
  const [defaultLlmConfigVersion, setDefaultLlmConfigVersion] = useState<string | null>(null)
  const [defaultPromptVersions, setDefaultPromptVersions] = useState<Record<string, string> | null>(null)
  const [selectedParagraphId, setSelectedParagraphId] = useState<string | null>(null)
  const [processingParagraphId, setProcessingParagraphId] = useState<string | null>(null)
  const [processingStage, setProcessingStage] = useState<'idle' | 'extracting' | 'critiquing' | 'retrying'>('idle')
  const [_processingRetryCount, setProcessingRetryCount] = useState(0)
  const [_chunkingStatus, setChunkingStatus] = useState<'idle' | 'in_progress' | 'completed' | 'stopped' | 'failed'>('idle')
  const streamContainerRef = useRef<HTMLDivElement>(null)
  const [eventSource, setEventSource] = useState<EventSource | null>(null)
  const lastEventTimeRef = useRef<number>(Date.now())
  const closedDueToStreamEndRef = useRef<boolean>(false)
  
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
        const response = await fetch(`http://localhost:8000/documents/${result.document_id}/status`)
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
      const response = await fetch('http://localhost:8000/documents')
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
      const response = await fetch(`http://localhost:8000/documents/${documentId}/extract/restart`, {
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

  const restartChunkingForDocument = async (documentId: string) => {
    try {
      const restartBody: Record<string, unknown> = {
        threshold: retryThreshold,
        critique_enabled: critiqueEnabled,
        max_retries: Math.max(0, maxRetries),
      }
      if (defaultLlmConfigVersion) restartBody.llm_config_version = defaultLlmConfigVersion
      if (defaultPromptVersions && Object.keys(defaultPromptVersions).length > 0) restartBody.prompt_versions = defaultPromptVersions
      const response = await fetch(
        `http://localhost:8000/documents/${documentId}/chunking/restart?threshold=${encodeURIComponent(retryThreshold)}&critique_enabled=${critiqueEnabled}&max_retries=${maxRetries}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(restartBody),
        }
      )
      if (response.ok) {
        // Connect to live stream if not already connected
        if (!eventSource) {
          closedDueToStreamEndRef.current = false
          const es = new EventSource(`http://localhost:8000/documents/${documentId}/chunking/live`)
          setEventSource(es)
          setStreamingOutput('') // Fresh status log
          es.onmessage = (event) => {
            try {
              const eventData = JSON.parse(event.data)
              handleStreamEvent(eventData)
            } catch (err) {
              console.error('Failed to parse stream event:', err)
            }
          }
        }
        setChunkingActive(true)
        await loadDocuments() // Refresh list
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
      const response = await fetch(`http://localhost:8000/admin/db/documents/${documentId}/delete-cascade`, {
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
          if (eventSource) {
            eventSource.close()
            setEventSource(null)
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
      if (eventSource) {
        eventSource.close()
        setEventSource(null)
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
      const statusResponse = await fetch(`http://localhost:8000/documents/${documentId}/status`)
      if (statusResponse.ok) {
        const statusData: StatusResponse = await statusResponse.json()
        setStatus(statusData)
      }
    } catch (err) {
      console.error('Failed to load extraction status:', err)
    }
    
    // Close existing event source if any
    if (eventSource) {
      eventSource.close()
      setEventSource(null)
    }
    
    // Always try to load chunks from PostgreSQL when selecting a document
    // loadChunkingResults handles empty results gracefully (returns early if no data)
    await loadChunkingResults(documentId, false)
    
    // Connect to live stream only if chunking is actively in progress
    // The loadChunkingResults function already sets chunkingActive/chunkingComplete flags
    // based on the status from the DB, but we check documentData.chunking_status here
    // to determine if we should connect to the live SSE stream
    if (documentData.chunking_status === 'in_progress') {
      // Connect to live event stream - this will be the only source of updates
      connectToChunkingStream(documentId)
    }
    // Note: For all other statuses (completed, stopped, failed, null/undefined),
    // loadChunkingResults has already set the appropriate state flags based on DB data
  }


  // Handle stream events from SSE
  const handleStreamEvent = (eventData: { event: string; data: any }) => {
    const { event, data } = eventData
    
    switch (event) {
      case 'llm_stream':
        // Update raw_llm_output for the current paragraph
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
            
            // Update streaming output immediately if this is the selected job
            // Show output if: selectedJobId is set AND (paraId matches processingParagraphId OR processingParagraphId isn't set yet)
            // This ensures we show output as soon as events arrive, even if processingParagraphId hasn't been set yet
            if (selectedJobId) {
              // If processingParagraphId is set, only show if it matches
              // If not set yet, show any output (will be refined when paragraph_start event arrives)
              if (!processingParagraphId || paraId === processingParagraphId) {
                setStreamingOutput(newOutput)
              }
            }
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
        // Update activeJobs - paragraph complete, stage is now idle (will update to persisting when persisted)
        setActiveJobs(prev => prev.map(job => 
          job.document_id === selectedJobId
            ? { ...job, processing_stage: 'idle' }
            : job
        ))
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
        
      case 'status_message':
        // Live "thinking" stream: paragraph started, facts found, critique passed/failed, etc.
        if (selectedJobId && data?.message != null) {
          const msg = String(data.message).trim()
          if (msg) setStreamingOutput(prev => prev + msg + '\n')
        }
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
        closedDueToStreamEndRef.current = true
        if (eventSource) {
          eventSource.close()
          setEventSource(null)
        }
        break
        
      case 'stream_end': {
        const reason = data?.reason
        if (reason === 'chunking_complete') {
          setChunkingActive(false)
          setChunkingComplete(true)
        }
        closedDueToStreamEndRef.current = true
        if (eventSource) {
          eventSource.close()
          setEventSource(null)
        }
        break
      }
    }
  }

  // Load chunking results from DB and update state
  const loadChunkingResults = async (documentId: string, mergeWithExisting: boolean = false) => {
    try {
      const response = await fetch(`http://localhost:8000/documents/${documentId}/chunking/results`)
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
      
      // Determine if we should merge (SSE is active and merge flag is set)
      const shouldMerge = mergeWithExisting && eventSource
      
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

  // Helper function to establish SSE connection with proper tracking
  const connectToChunkingStream = (documentId: string) => {
    // Close existing connection if any
    if (eventSource) {
      eventSource.close()
      setEventSource(null)
    }
    
    closedDueToStreamEndRef.current = false
    const es = new EventSource(`http://localhost:8000/documents/${documentId}/chunking/live`)
    setEventSource(es)
    setStreamingOutput('') // Fresh status log for this job
    lastEventTimeRef.current = Date.now()
    
    es.onopen = () => {
      console.log('SSE connection opened for chunking, document:', documentId)
      lastEventTimeRef.current = Date.now()
    }
    
    es.onmessage = (event) => {
      try {
        if (event.data.trim() === ': heartbeat') {
          lastEventTimeRef.current = Date.now()
          return
        }
        const eventData = JSON.parse(event.data)
        lastEventTimeRef.current = Date.now()
        const paraId = eventData.data?.paragraph_id
        const logLabel = eventData.event === 'stream_end' ? `stream_end (${eventData.data?.reason ?? 'end'})` : `${eventData.event}${paraId ? ` ${paraId}` : ''}`
        console.log('SSE event received:', logLabel)
        handleStreamEvent(eventData)
      } catch (err) {
        console.error('Failed to parse stream event:', err)
      }
    }
    
    es.onerror = () => {
      // If we closed due to stream_end, don't log reconnection noise
      if (closedDueToStreamEndRef.current) return
      if (es.readyState === EventSource.CLOSED) {
        console.log('SSE connection closed for document:', documentId)
      } else if (es.readyState === EventSource.CONNECTING) {
        console.log('SSE connection reconnecting for document:', documentId)
      }
    }
    
    return es
  }

  const startChunking = async (documentId: string) => {
    setError(null)
    console.log(`Starting chunking for document ${documentId}...`)
    
    try {
      const body: Record<string, unknown> = {
        threshold: retryThreshold,
        critique_enabled: critiqueEnabled,
        max_retries: Math.max(0, maxRetries),
      }
      if (defaultLlmConfigVersion) body.llm_config_version = defaultLlmConfigVersion
      if (defaultPromptVersions && Object.keys(defaultPromptVersions).length > 0) body.prompt_versions = defaultPromptVersions
      const response = await fetch(
        `http://localhost:8000/documents/${documentId}/chunking/start?threshold=${encodeURIComponent(retryThreshold)}&critique_enabled=${critiqueEnabled}&max_retries=${maxRetries}`,
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
        
        // Connect to live event stream - this will be the only source of real-time updates
        connectToChunkingStream(documentId)
      }
    } catch (err) {
      console.error('Start chunking exception:', err)
      setError(err instanceof Error ? err.message : 'Failed to start chunking')
    }
  }

  const stopChunking = async (documentId: string) => {
    try {
      const response = await fetch(
        `http://localhost:8000/documents/${documentId}/chunking/stop`,
        { method: 'POST' }
      )
      if (response.ok) {
        // Close event source
        if (eventSource) {
          eventSource.close()
          setEventSource(null)
        }
        // Fetch results once to update state
        await loadChunkingResults(documentId)
      }
    } catch (err) {
      console.error("Failed to stop chunking:", err)
    }
  }

  // Load results when chunking section is opened (only if SSE not active)
  useEffect(() => {
    if (result?.document_id && (chunkingActive || chunkingComplete)) {
      // Only load from DB if SSE is not active (SSE handles updates during streaming)
      if (!eventSource) {
        loadChunkingResults(result.document_id, false)
      }
    }
  }, [result?.document_id, eventSource])

  // Close event source when chunking completes
  useEffect(() => {
    if (chunkingComplete && eventSource) {
      eventSource.close()
      setEventSource(null)
    }
  }, [chunkingComplete, eventSource])

  // Clean up event source on unmount
  useEffect(() => {
    return () => {
      if (eventSource) {
        eventSource.close()
      }
    }
  }, [eventSource])


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

  // Monitor SSE connection health and reload from DB if stuck
  useEffect(() => {
    if (!chunkingActive || !selectedDocumentId || !eventSource) return
    
    let lastReloadTime = 0
    const RELOAD_DEBOUNCE_MS = 3000 // Only reload at most once every 3 seconds
    
    const checkInterval = setInterval(() => {
      const timeSinceLastEvent = Date.now() - lastEventTimeRef.current
      const connectionState = eventSource.readyState
      const now = Date.now()
      
      // If no events for 10 seconds and connection is open, reload from DB as fallback
      // Use merge=true to preserve user's current view and selections
      // Debounce reloads to avoid excessive requests
      if (timeSinceLastEvent > 10000 && connectionState === EventSource.OPEN && (now - lastReloadTime) > RELOAD_DEBOUNCE_MS) {
        console.log('SSE appears stuck, reloading from DB as fallback')
        lastReloadTime = now
        // Store current selections to preserve them
        const currentSelectedPage = selectedPage
        const currentSelectedParagraphId = selectedParagraphId
        const currentProcessingParagraphId = processingParagraphId
        
        loadChunkingResults(selectedDocumentId, true).then(() => {
          // Restore selections if they were changed (shouldn't happen with merge=true, but safety check)
          if (currentSelectedPage !== null && selectedPage !== currentSelectedPage) {
            setSelectedPage(currentSelectedPage)
          }
          if (currentSelectedParagraphId !== null && selectedParagraphId !== currentSelectedParagraphId) {
            setSelectedParagraphId(currentSelectedParagraphId)
          }
          if (currentProcessingParagraphId !== null && processingParagraphId !== currentProcessingParagraphId) {
            setProcessingParagraphId(currentProcessingParagraphId)
          }
        })
      }
      
      // If connection is closed, try to reconnect
      if (connectionState === EventSource.CLOSED) {
        console.log('SSE connection closed, attempting to reconnect')
        connectToChunkingStream(selectedDocumentId)
      }
    }, 5000) // Check every 5 seconds
    
    return () => clearInterval(checkInterval)
  }, [chunkingActive, selectedDocumentId, eventSource, selectedPage, selectedParagraphId, processingParagraphId])

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

  // Track active/recent jobs for Live Updates tab (in_progress, queued, or completed so user can view events)
  useEffect(() => {
    const updateActiveJobs = async () => {
      const activeDocs = documents.filter(doc =>
        doc.chunking_status === 'in_progress' ||
        doc.chunking_status === 'queued' ||
        doc.chunking_status === 'completed' ||
        doc.chunking_status === 'failed' ||
        doc.chunking_status === 'stopped'
      )
      
      const jobs = await Promise.all(
        activeDocs.map(async (doc) => {
          // Fetch chunking results to get progress
          let progress = 0
          let totalParagraphs = 0
          let completedParagraphs = 0
          let currentPage = 0
          let totalPages = 0
          
          try {
            const resultsResponse = await fetch(
              `http://localhost:8000/documents/${doc.id}/chunking/results`
            )
            
            if (resultsResponse.ok) {
              const resultsData = await resultsResponse.json()
              const metadata = resultsData.metadata || {}
              totalParagraphs = metadata.total_paragraphs || 0
              completedParagraphs = metadata.completed_count || 0
              progress = totalParagraphs > 0 
                ? Math.round((completedParagraphs / totalParagraphs) * 100) 
                : 0
              
              // Extract current page from processingParagraphId
              if (processingParagraphId && selectedJobId === doc.id) {
                const [pageNum] = processingParagraphId.split('_')
                currentPage = parseInt(pageNum, 10) || 0
              }
              
              // Get total pages from document pages
              try {
                const pagesResponse = await fetch(
                  `http://localhost:8000/documents/${doc.id}/pages`
                )
                if (pagesResponse.ok) {
                  const pagesData = await pagesResponse.json()
                  totalPages = pagesData.pages?.length || 0
                }
              } catch (pagesErr) {
                // Ignore pages fetch errors
              }
            }
          } catch (err) {
            // If fetch fails, use defaults (progress = 0)
            console.error(`Failed to fetch progress for ${doc.id}:`, err)
          }
          
          return {
            document_id: doc.id,
            filename: doc.display_name?.trim() || doc.filename,
            status: doc.chunking_status || 'idle',
            progress,
            start_time: doc.created_at,
            current_page: currentPage || undefined,
            total_pages: totalPages || undefined,
            current_paragraph: (selectedJobId === doc.id && processingParagraphId) || undefined,
            total_paragraphs: totalParagraphs || undefined,
            completed_paragraphs: completedParagraphs || undefined,
            processing_stage: (selectedJobId === doc.id ? processingStage : 'idle') as 'extracting' | 'critiquing' | 'retrying' | 'persisting' | 'idle'
          }
        })
      )
      // Order by entry date (newest first) so the Live Updates list is never jumbled
      jobs.sort((a, b) => new Date(b.start_time).getTime() - new Date(a.start_time).getTime())
      // Merge with current state so SSE-driven progress is not overwritten by stale poll
      setActiveJobs(prev => jobs.map(j => {
        const existing = prev.find(p => p.document_id === j.document_id)
        if (existing && (existing.progress > j.progress || (existing.completed_paragraphs != null && j.completed_paragraphs != null && existing.completed_paragraphs > j.completed_paragraphs))) {
          return { ...j, progress: existing.progress, completed_paragraphs: existing.completed_paragraphs, total_paragraphs: existing.total_paragraphs ?? j.total_paragraphs, current_page: existing.current_page ?? j.current_page, total_pages: existing.total_pages ?? j.total_pages }
        }
        return j
      }))
    }
    updateActiveJobs()
    const interval = setInterval(updateActiveJobs, 2000)
    return () => clearInterval(interval)
  }, [documents])

  // Update streaming output for selected job - show current paragraph's output
  useEffect(() => {
    if (selectedJobId && processingParagraphId) {
      const para = paragraphs.get(processingParagraphId)
      if (para?.raw_llm_output) {
        setStreamingOutput(para.raw_llm_output)
      } else {
        // If paragraph exists but no output yet, show empty (will be filled by llm_stream events)
        setStreamingOutput('')
      }
    } else if (selectedJobId) {
      // Job selected but no current paragraph - check if we have any paragraphs with output
      const allOutput: string[] = []
      paragraphs.forEach((para) => {
        if (para.raw_llm_output && para.raw_llm_output.length > 0) {
          allOutput.push(para.raw_llm_output)
        }
      })
      if (allOutput.length > 0) {
        // Show the most recent paragraph's output (last in array)
        setStreamingOutput(allOutput[allOutput.length - 1])
      } else {
        setStreamingOutput('')
      }
    } else {
      setStreamingOutput('')
    }
  }, [selectedJobId, processingParagraphId, paragraphs])

  const handleUpload = async (file: File) => {
    setFile(file)
    setError(null)
    setUploading(true)
    
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => null)
        
        if (errorData?.detail?.error === 'duplicate_file') {
          setError(
            `This file has already been uploaded as "${errorData.detail.original_filename}".\n\n${errorData.detail.help}`
          )
        } else if (errorData?.detail?.message) {
          setError(errorData.detail.message)
        } else {
          setError(`Upload failed: ${response.statusText}`)
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
      setError(err instanceof Error ? err.message : 'Upload failed. Please try again.')
    } finally {
      setUploading(false)
    }
  }

  const handleStartChunking = async (documentId: string) => {
    await startChunking(documentId)
    await loadDocuments()
    // Switch to live updates tab
    setActiveTab('live')
    setSelectedJobId(documentId)
  }

  const handleStopChunking = async (documentId: string) => {
    await stopChunking(documentId)
    await loadDocuments()
  }

  const handleRestartChunking = async (documentId: string) => {
    await restartChunkingForDocument(documentId)
    await loadDocuments()
    // Switch to live updates tab to see the restart
    setActiveTab('live')
    setSelectedJobId(documentId)
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

  const handleJobSelect = async (documentId: string) => {
    setSelectedJobId(documentId)
    const doc = documents.find(d => d.id === documentId)
    if (doc) {
      // Set selectedDocumentId to match selectedJobId so streaming works
      setSelectedDocumentId(documentId)
      await selectDocument(documentId, doc)
      // Ensure SSE connection is established if job is in progress
      // This will reconnect and receive all buffered events
      if (doc.chunking_status === 'in_progress') {
        // Close existing connection if it's for a different document
        if (eventSource) {
          eventSource.close()
          setEventSource(null)
        }
        // Small delay to ensure previous connection is closed
        setTimeout(() => {
          connectToChunkingStream(documentId)
        }, 100)
      }
    }
  }

  // Load existing events and connect to SSE when Live Updates tab becomes active
  useEffect(() => {
    if (activeTab === 'live' && selectedJobId) {
      const doc = documents.find(d => d.id === selectedJobId)
      if (doc) {
        // Load existing events from database
        const loadExistingEvents = async () => {
          try {
            const response = await fetch(`http://localhost:8000/documents/${selectedJobId}/chunking/events`)
            if (response.ok) {
              const data = await response.json()
              if (data.events && data.events.length > 0) {
                console.log(`Loading ${data.events.length} existing events for document ${selectedJobId}`)
                // Process existing events in order - only show status messages in output
                let output = ''
                for (const event of data.events) {
                  if (event.event === 'status_message' && event.data?.message) {
                    const msg = String(event.data.message).trim()
                    if (msg) {
                      output += msg + '\n'
                    }
                  }
                  // Note: We don't call handleStreamEvent for historical events to avoid
                  // triggering state updates that are meant for live events
                }
                if (output) {
                  setStreamingOutput(output) // Replace, not append, since these are historical
                }
              }
            }
          } catch (err) {
            console.error('Failed to load existing events:', err)
          }
        }
        
        loadExistingEvents()
        
        // Connect to SSE stream for new events (if job is in progress or we want to see new events)
        if (doc.chunking_status === 'in_progress' || doc.chunking_status === 'pending') {
          // Check if we already have an active connection for this document
          if (!eventSource || (eventSource && eventSource.readyState !== EventSource.OPEN)) {
            // Reconnect to get new events
            if (eventSource) {
              eventSource.close()
              setEventSource(null)
            }
            setTimeout(() => {
              connectToChunkingStream(selectedJobId)
            }, 100)
          }
        }
      }
    }
  }, [activeTab, selectedJobId, documents])

  return (
    <div className="app">
      <Header
        chunkingOptions={{
          threshold: retryThreshold,
          critiqueEnabled,
          maxRetries,
        }}
        onChunkingOptionsChange={{
          setThreshold: setRetryThreshold,
          setCritiqueEnabled,
          setMaxRetries,
        }}
        defaultLlmConfigVersion={defaultLlmConfigVersion}
        onDefaultLlmConfigVersionChange={setDefaultLlmConfigVersion}
        defaultPromptVersions={defaultPromptVersions}
        onDefaultPromptVersionsChange={setDefaultPromptVersions}
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
              onDeleteDocument={deleteDocument}
              onRestartChunking={handleRestartChunking}
              chunkingOptions={{
                threshold: retryThreshold,
                critiqueEnabled,
                maxRetries,
              }}
              onChunkingOptionsChange={{
                setThreshold: setRetryThreshold,
                setCritiqueEnabled,
                setMaxRetries,
              }}
            />
          </TabPanel>

          <TabPanel id="live" isActive={activeTab === 'live'}>
            <LiveUpdatesTab
              activeJobs={activeJobs}
              onJobSelect={handleJobSelect}
              selectedJobId={selectedJobId}
              streamingOutput={streamingOutput}
              isActive={activeTab === 'live'}
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
