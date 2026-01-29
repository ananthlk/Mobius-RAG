# Step 7: Atomic Chunking + LLM Extraction + QA/Critique

## Scope: Atomic Chunking Only

**Step 7 focuses ONLY on hierarchical/atomic chunking for fact extraction.**

**What we're NOT doing in Step 7:**
- ❌ Semantic chunking (for chat/retrieval) - separate step later
- ❌ Embeddings generation - separate step later

**What we ARE doing in Step 7:**
- ✅ Split pages into paragraphs (hierarchical structure)
- ✅ Extract summary + all facts from each paragraph
- ✅ Answer 5 prescriptive questions for each fact
- ✅ Auto-qualify verification status (explicit vs inferred)
- ✅ QA/Critique agent with retry logic

---

## What We're Building

### Hierarchical/Atomic Chunking
**Purpose**: Extract precise, atomic eligibility facts from paragraphs

**Characteristics**:
- Respects document structure: section → chapter → paragraph
- Paragraph-level chunks
- Each chunk is a discrete unit

**For Each Paragraph, Extract:**
1. Summary of the paragraph
2. All discrete facts
3. For each fact: Answer 5 prescriptive questions
4. Auto-qualify verification status (explicit vs inferred)
5. QA/Critique agent review with retry logic

---

## Dependencies

**Required for Local Development (Ollama)**:
- `aiohttp` - For async HTTP requests to Ollama API

**Required for Production (Vertex AI)**:
- `google-cloud-aiplatform` - Vertex AI SDK (optional, only if using Vertex AI)

**Configuration**:
- Set `LLM_PROVIDER=ollama` for local development
- Set `LLM_PROVIDER=vertex` for production
- Provider abstraction allows easy switching between LLM providers

---

## Implementation Plan

**What We Extract for Each Paragraph:**
1. **Summary**: Brief summary of what the paragraph states
2. **All Facts**: Every discrete fact mentioned in the paragraph
3. **For Each Fact**: Answer 5 prescriptive questions:
   - WHO is eligible?
   - HOW is eligibility verified?
   - WHAT happens in conflicts?
   - WHEN does eligibility apply?
   - LIMITATIONS: What restrictions apply?
4. **Auto-Qualify Verification**: Mark each fact as explicitly stated (verified) or inferred
5. **QA/Critique Agent**: Review extraction quality, retry with feedback if needed

**Quality Assurance:**
- Critique agent evaluates extraction accuracy
- Checks for missing facts, hallucinated facts, incorrect answers
- If critique fails → retry extraction with feedback (max 2 retries)
- Ensures high-quality fact extraction

**Implementation:**

1. Create hierarchical chunking service
   - Parse document structure (sections, chapters)
   - Split into paragraphs
   - Preserve hierarchy metadata

2. LLM Extraction for Each Paragraph
   ```
   For each paragraph, extract:
   
   1. Summary: Brief summary of what this paragraph states
   2. All Facts: List of all discrete facts mentioned
   3. For Each Fact: Answer prescriptive questions
   
   Initial Extraction Prompt:
   "Analyze this paragraph from a healthcare provider manual.
   
   Extract:
   1. A brief summary of this paragraph
   2. All discrete facts stated in this paragraph
   3. For each fact, answer these questions:
      - WHO is eligible? (qualifying criteria, who qualifies)
      - HOW is eligibility verified? (verification methods, processes)
      - WHAT happens in conflicts? (conflict resolution, disputes)
      - WHEN does eligibility apply? (effective dates, retroactive rules)
      - LIMITATIONS: What restrictions or limitations apply?
   
   Return JSON:
   {
     "summary": "Brief summary of paragraph",
     "facts": [
       {
         "fact_text": "Exact statement of the fact",
         "who_eligible": "Answer or null",
         "how_verified": "Answer or null",
         "conflict_resolution": "Answer or null",
         "when_applies": "Answer or null",
         "limitations": "Answer or null",
         "is_eligibility_related": true/false,
         "fact_type": "who_eligible|verification_method|conflict|effective_date|limitation|other",
         "is_verified": true/false  // Auto-qualify: true if fact is explicitly stated, false if inferred
       }
     ],
     "overall_is_eligibility_related": true/false,
     "confidence": 0.0-1.0
   }"
   ```

3. QA/Critique Agent
   ```
   After extraction, critique agent reviews:
   
   Critique Prompt:
   "Review this extracted summary and facts from a healthcare provider manual paragraph.
   
   Original paragraph:
   {paragraph_text}
   
   Extracted summary: {summary}
   Extracted facts: {facts}
   
   Evaluate:
   1. Is the summary accurate and complete?
   2. Are all facts correctly extracted (not missing, not hallucinated)?
   3. Are the prescriptive question answers accurate?
   4. Is the verification status (is_verified) correct?
   
   Return JSON:
   {
     "pass": true/false,
     "feedback": "Detailed feedback if fail",
     "issues": [
       {
         "type": "missing_fact|hallucinated_fact|incorrect_answer|verification_error",
         "description": "...",
         "suggestion": "..."
       }
     ],
     "confidence": 0.0-1.0
   }"
   ```

4. Retry Logic
   ```
   If critique.pass == false:
     - Retry extraction with feedback
     - Include critique feedback in retry prompt
     - Max 2 retries per paragraph
     - Track retry count
   
   Retry Prompt:
   "Previous extraction had issues. Please re-extract with this feedback:
   {critique.feedback}
   
   Original paragraph: {paragraph_text}
   
   Address these specific issues:
   {critique.issues}
   
   [Same extraction format as before]"
   ```

5. Fact Verification Auto-Qualification
   ```
   For each extracted fact:
   - is_verified = true if:
     * Fact is explicitly stated in paragraph
     * Can be directly quoted or closely paraphrased
   - is_verified = false if:
     * Fact is inferred or implied
     * Requires interpretation
     * Not directly stated
   
   LLM determines this during extraction based on:
   "Is this fact explicitly stated in the paragraph, or is it inferred?"
   ```
   
   **Key Points**:
   - Extract ALL facts from paragraph (not just eligibility)
   - For each fact, answer the 5 prescriptive questions
   - Auto-qualify verification status (explicit vs inferred)
   - QA/Critique agent reviews extraction quality
   - Retry with feedback if critique fails
   - Store summary + all facts + question answers + verification status

### Step 7c: Database Schema

**Hierarchical Chunks Table** (Atomic Chunking):
- id, document_id, page_number, paragraph_index
- section_path, chapter_path (hierarchy)
- text, text_length
- summary (LLM-extracted summary of paragraph)
- is_eligibility_related, confidence, reasoning
- extraction_status (pending, extracted, failed)
- critique_status (pending, passed, failed, retrying)
- critique_feedback (feedback from QA agent)
- retry_count (number of retries attempted, max 2)
- needs_human_review (boolean, true if failed after max retries)
- created_at

**Facts Table** (linked to hierarchical chunks):
- id, hierarchical_chunk_id (FK)
- fact_text (the extracted fact statement)
- fact_type (who_eligible, verification_method, conflict, effective_date, limitation, other)
- is_eligibility_related (true if any prescriptive question answered)
- is_verified (true if explicitly stated, false if inferred)
- who_eligible (answer to WHO question)
- how_verified (answer to HOW question)
- conflict_resolution (answer to WHAT/conflict question)
- when_applies (answer to WHEN question)
- limitations (answer to LIMITATIONS question)
- confidence
- created_at

### Step 7d: Live Streaming Architecture

**Backend: Server-Sent Events (SSE)**
- Stream LLM responses in real-time
- Send events for each stage of processing
- Minimize perceived lag

**Event Types**:
```typescript
// Raw LLM output (streaming)
{ type: "llm_stream", chunk: "raw text chunk..." }

// Initial extraction complete
{ type: "extraction_complete", data: { summary, facts, ... } }

// Critique agent reviewing
{ type: "critique_start", message: "Reviewing extraction quality..." }

// Critique results
{ type: "critique_complete", data: { pass: true/false, feedback, issues } }

// Retry triggered
{ type: "retry_start", retry_count: 1, feedback: "..." }

// Retry extraction complete
{ type: "retry_complete", data: { summary, facts, ... } }

// Final status
{ type: "paragraph_complete", status: "passed|failed", chunk_id: "...", needs_human_review: true/false }
```

**Frontend Component Structure**:
```typescript
interface ParagraphProcessingState {
  paragraph_id: string;
  page_number: number;
  paragraph_index: number;
  
  // Stage 1: Raw LLM streaming
  raw_llm_output: string;  // Accumulated streaming text
  is_streaming: boolean;
  
  // Stage 2: Parsed extraction
  extraction_complete: boolean;
  summary: string | null;
  facts: ExtractedFact[];
  
  // Stage 3: Critique agent
  critique_status: "pending" | "reviewing" | "complete";
  critique_result: {
    pass: boolean;
    feedback: string | null;
    issues: CritiqueIssue[];
  } | null;
  
  // Stage 4: Retry (if needed)
  retry_count: number;
  retry_feedback: string | null;
  retry_extraction: {
    summary: string;
    facts: ExtractedFact[];
  } | null;
  
  // Final status
  final_status: "pending" | "passed" | "failed";
  needs_human_review: boolean;
}
```

**UI Display Structure**:
```
┌─────────────────────────────────────┐
│ Paragraph 2 (Page 45)               │
├─────────────────────────────────────┤
│ [Streaming...]                       │
│ Raw LLM Output:                      │
│ "Analyzing paragraph... Extracting  │
│  summary... Found 3 facts..."       │
├─────────────────────────────────────┤
│ ✓ Extraction Complete               │
│ Summary: "..."                       │
│ Facts:                               │
│   • Fact 1: ...                      │
│   • Fact 2: ...                      │
├─────────────────────────────────────┤
│ 🔍 Critique Agent: Reviewing...     │
│ ✓ Passed                             │
│ (or)                                 │
│ ✗ Failed - Retrying...              │
│ Feedback: "Missing fact about..."  │
├─────────────────────────────────────┤
│ [If Retry]                           │
│ Retry #1 with feedback               │
│ New extraction results...            │
├─────────────────────────────────────┤
│ [If Failed After Retries]           │
│ ⚠️ Flagged for Human Review         │
│ Extraction failed after 2 retries    │
└─────────────────────────────────────┘
```

**LLM Provider Abstraction**:

```python
# app/services/llm_provider.py
from abc import ABC, abstractmethod
from typing import AsyncIterator

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream LLM response tokens."""
        pass
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate complete LLM response (non-streaming)."""
        pass

# Ollama implementation (for local development)
class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url
        self.model = model
    
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    **kwargs
                }
            ) as response:
                async for line in response.content:
                    if line:
                        import json
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
    
    async def generate(self, prompt: str, **kwargs) -> str:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    **kwargs
                }
            ) as response:
                data = await response.json()
                return data.get("response", "")

# Vertex AI implementation (for production)
class VertexAIProvider(LLMProvider):
    def __init__(self, project_id: str, location: str = "us-central1", model: str = "gemini-1.5-pro"):
        from google.cloud import aiplatform
        import vertexai
        vertexai.init(project=project_id, location=location)
        self.model = model
    
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        from vertexai.generative_models import GenerativeModel
        model = GenerativeModel(self.model)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.1, **kwargs},
            stream=True
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
    
    async def generate(self, prompt: str, **kwargs) -> str:
        from vertexai.generative_models import GenerativeModel
        model = GenerativeModel(self.model)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.1, **kwargs}
        )
        return response.text

# Factory function
def get_llm_provider() -> LLMProvider:
    """Get LLM provider based on configuration."""
    from app.config import LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL, VERTEX_PROJECT_ID, VERTEX_LOCATION, VERTEX_MODEL
    
    if LLM_PROVIDER == "ollama":
        return OllamaProvider(
            base_url=OLLAMA_BASE_URL or "http://localhost:11434",
            model=OLLAMA_MODEL or "llama3.2"
        )
    elif LLM_PROVIDER == "vertex":
        return VertexAIProvider(
            project_id=VERTEX_PROJECT_ID,
            location=VERTEX_LOCATION or "us-central1",
            model=VERTEX_MODEL or "gemini-1.5-pro"
        )
    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")
```

**Configuration** (`app/config.py`):
```python
# LLM Provider Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "vertex"

# Ollama settings (for local development)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Vertex AI settings (for production)
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-1.5-pro")
```

**Backend Implementation**:
```python
from fastapi.responses import StreamingResponse
import json
from app.services.llm_provider import get_llm_provider

@app.get("/documents/{document_id}/chunking/stream")
async def stream_chunking_process(document_id: str):
    """Stream chunking and extraction process in real-time."""
    llm = get_llm_provider()  # Get configured LLM provider
    
    async def event_generator():
        # Get all pages for document
        pages = await get_document_pages(document_id)
        
        for page in pages:
            paragraphs = split_paragraphs(page.text)
            
            for para_idx, paragraph in enumerate(paragraphs):
                para_id = f"{page.page_number}_{para_idx}"
                
                # Stage 1: Stream raw LLM output (provider-agnostic)
                async def stream_llm_response(prompt):
                    async for chunk in llm.stream_generate(prompt):
                        yield f"data: {json.dumps({
                            'event': 'llm_stream',
                            'data': {
                                'paragraph_id': para_id,
                                'chunk': chunk
                            }
                        })}\n\n"
                
                # Stream extraction
                raw_output = ""
                async for event in stream_llm_response(extraction_prompt):
                    raw_output += event
                    yield event
                
                # Stage 2: Parse and send extraction results
                extraction = parse_extraction_result(raw_output)
                yield f"data: {json.dumps({
                    'event': 'extraction_complete',
                    'data': {
                        'paragraph_id': para_id,
                        'summary': extraction.summary,
                        'facts': extraction.facts
                    }
                })}\n\n"
                
                # Stage 3: Critique agent
                yield f"data: {json.dumps({
                    'event': 'critique_start',
                    'data': {'paragraph_id': para_id}
                })}\n\n"
                
                # Stream critique LLM response
                critique_raw = ""
                async for event in stream_llm_response(critique_prompt):
                    critique_raw += event
                    yield event
                
                critique = parse_critique_result(critique_raw)
                yield f"data: {json.dumps({
                    'event': 'critique_complete',
                    'data': {
                        'paragraph_id': para_id,
                        'pass': critique.pass,
                        'feedback': critique.feedback,
                        'issues': critique.issues
                    }
                })}\n\n"
                
                # Stage 4: Retry if needed
                retry_count = 0
                while not critique.pass and retry_count < 2:
                    retry_count += 1
                    yield f"data: {json.dumps({
                        'event': 'retry_start',
                        'data': {
                            'paragraph_id': para_id,
                            'retry_count': retry_count,
                            'feedback': critique.feedback
                        }
                    })}\n\n"
                    
                    # Retry extraction with streaming
                    retry_raw = ""
                    async for event in stream_llm_response(retry_prompt_with_feedback):
                        retry_raw += event
                        yield event
                    
                    retry_extraction = parse_extraction_result(retry_raw)
                    yield f"data: {json.dumps({
                        'event': 'retry_extraction_complete',
                        'data': {
                            'paragraph_id': para_id,
                            'summary': retry_extraction.summary,
                            'facts': retry_extraction.facts
                        }
                    })}\n\n"
                    
                    # Re-run critique
                    critique = await run_critique_agent(retry_extraction)
                    yield f"data: {json.dumps({
                        'event': 'critique_complete',
                        'data': {
                            'paragraph_id': para_id,
                            'pass': critique.pass,
                            'feedback': critique.feedback,
                            'issues': critique.issues
                        }
                    })}\n\n"
                
                # Final status - flag for human review if failed after retries
                needs_human_review = not critique.pass and retry_count >= 2
                yield f"data: {json.dumps({
                    'event': 'paragraph_complete',
                    'data': {
                        'paragraph_id': para_id,
                        'status': 'passed' if critique.pass else 'failed',
                        'needs_human_review': needs_human_review
                    }
                })}\n\n"
                
                # Store chunk with human review flag
                await store_hierarchical_chunk(
                    document_id=document_id,
                    page_number=page.page_number,
                    paragraph_index=para_idx,
                    text=paragraph,
                    summary=extraction.summary if critique.pass else (retry_extraction.summary if retry_extraction else None),
                    critique_status='passed' if critique.pass else 'failed',
                    retry_count=retry_count,
                    needs_human_review=needs_human_review
                )
    
    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
```

**LLM Provider Streaming**:
- **Ollama (Local Dev)**: Uses `/api/generate` with `stream: true` for real-time token generation
- **Vertex AI (Production)**: Uses Vertex AI's streaming API for real-time token generation
- Both providers stream tokens as they're generated (not waiting for full response)
- Parse JSON incrementally if possible, or parse at end
- Handle streaming errors gracefully
- Provider selected via `LLM_PROVIDER` environment variable

**Frontend Implementation**:
```typescript
const eventSource = new EventSource(`/documents/${docId}/chunking/stream`);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch (data.event) {
    case "llm_stream":
      // Append to raw_llm_output
      updateParagraphState(data.paragraph_id, {
        raw_llm_output: prev => prev + data.chunk,
        is_streaming: true
      });
      break;
      
    case "extraction_complete":
      // Update with parsed results
      updateParagraphState(data.paragraph_id, {
        is_streaming: false,
        extraction_complete: true,
        summary: data.summary,
        facts: data.facts
      });
      break;
      
    case "critique_complete":
      // Update critique status
      updateParagraphState(data.paragraph_id, {
        critique_status: "complete",
        critique_result: {
          pass: data.pass,
          feedback: data.feedback,
          issues: data.issues
        }
      });
      break;
      
    case "paragraph_complete":
      // Update final status and human review flag
      updateParagraphState(data.paragraph_id, {
        final_status: data.status,
        needs_human_review: data.needs_human_review || false
      });
      break;
      
    // ... handle other events
  }
};
```

### Step 7e: Processing Flow
```
Extract Pages
    ↓
Split into Paragraphs (hierarchical structure)
    ↓
For Each Paragraph (streamed to frontend):
    ↓
    [STREAM] LLM Extraction (raw output streaming):
    - Stream raw LLM tokens to frontend in real-time
    - Extract summary
    - Extract all facts
    - For each fact: Answer 5 prescriptive questions
    - Auto-qualify: is_verified (explicit vs inferred)
    ↓
    [STREAM] Send parsed extraction results to frontend
    ↓
    [STREAM] QA/Critique Agent:
    - Stream "Reviewing..." status to frontend
    - Review extraction quality
    - Check for missing/hallucinated facts
    - Verify question answers
    - Check verification status
    ↓
    [STREAM] Send critique results to frontend
    ↓
    If critique.pass == false:
        [STREAM] Send retry start event with feedback
        Retry extraction with feedback (max 2 retries)
        - Stream raw LLM output again
        - Stream new parsed results
        ↓
        [STREAM] Re-run critique and stream results
    ↓
    If critique.pass == true OR retry_count >= 2:
        [STREAM] Send final status (passed/failed)
        If critique.pass == false AND retry_count >= 2:
            Flag for human review (needs_human_review = true)
        Store results with needs_human_review flag
    ↓
Store:
- Hierarchical chunk (paragraph + summary + critique status + needs_human_review flag)
- All facts with question answers + verification status
    ↓
Tag chunk as eligibility-related if ANY fact is eligibility-related
```

---

## Benefits

1. **Atomic extraction**: Precise fact extraction from paragraphs
2. **QA/Critique agent**: Ensures high-quality extraction with retry logic
3. **Verification status**: Know if facts are explicitly stated or inferred
4. **Prescriptive questions**: Structured answers for eligibility facts
5. **Ready for Step 8**: Extracted facts ready for atomic fact storage

---

## Frontend Component Structure (Detailed)

**Real-time Display for Each Paragraph**:

```typescript
// Component: ParagraphChunkingView
interface ParagraphChunkingViewProps {
  paragraph_id: string;
  page_number: number;
  paragraph_index: number;
}

// State management per paragraph
const [paragraphState, setParagraphState] = useState<ParagraphProcessingState>({
  raw_llm_output: "",
  is_streaming: false,
  extraction_complete: false,
  summary: null,
  facts: [],
  critique_status: "pending",
  critique_result: null,
  retry_count: 0,
  retry_feedback: null,
  retry_extraction: null,
  final_status: "pending",
  needs_human_review: false
});

// Render stages
return (
  <div className="paragraph-processing">
    {/* Stage 1: Raw LLM Streaming */}
    {is_streaming && (
      <div className="streaming-output">
        <h4>Extracting...</h4>
        <pre>{raw_llm_output}</pre>
        <span className="streaming-indicator">●</span>
      </div>
    )}
    
    {/* Stage 2: Parsed Extraction */}
    {extraction_complete && (
      <div className="extraction-results">
        <h4>✓ Extraction Complete</h4>
        <p><strong>Summary:</strong> {summary}</p>
        <ul>
          {facts.map(fact => (
            <li key={fact.id}>
              <strong>{fact.fact_text}</strong>
              <div>Type: {fact.fact_type}</div>
              <div>Verified: {fact.is_verified ? "✓" : "?"}</div>
              {/* Show question answers */}
            </li>
          ))}
        </ul>
      </div>
    )}
    
    {/* Stage 3: Critique Agent */}
    {critique_status === "reviewing" && (
      <div className="critique-reviewing">
        <h4>🔍 Critique Agent: Reviewing...</h4>
      </div>
    )}
    
    {critique_result && (
      <div className={critique_result.pass ? "critique-passed" : "critique-failed"}>
        <h4>{critique_result.pass ? "✓ Passed" : "✗ Failed"}</h4>
        {critique_result.feedback && <p>{critique_result.feedback}</p>}
        {critique_result.issues.map(issue => (
          <div key={issue.type}>{issue.description}</div>
        ))}
      </div>
    )}
    
    {/* Stage 4: Retry (if applicable) */}
    {retry_count > 0 && (
      <div className="retry-section">
        <h4>Retry #{retry_count}</h4>
        <p>Feedback: {retry_feedback}</p>
        {retry_extraction && (
          <div>
            <p>New Summary: {retry_extraction.summary}</p>
            <p>New Facts: {retry_extraction.facts.length}</p>
          </div>
        )}
      </div>
    )}
    
    {/* Final Status */}
    {final_status !== "pending" && (
      <div className={`final-status ${final_status}`}>
        {final_status === "passed" ? "✓ Complete" : "✗ Failed"}
      </div>
    )}
    
    {/* Human Review Flag */}
    {needs_human_review && (
      <div className="human-review-flag">
        <h4>⚠️ Flagged for Human Review</h4>
        <p>Extraction failed after {retry_count} retries. Requires manual review.</p>
      </div>
    )}
  </div>
);
```

---

## Example Output Structure

**Hierarchical Chunk**:
```json
{
  "id": "...",
  "document_id": "...",
  "page_number": 45,
  "paragraph_index": 2,
  "section_path": "Section 3.2",
  "text": "Full paragraph text...",
  "summary": "This paragraph states that eligibility verification must be completed within 30 days...",
  "is_eligibility_related": true,
  "confidence": 0.95,
  "extraction_status": "extracted",
  "critique_status": "passed",
  "critique_feedback": null,
  "retry_count": 0,
  "needs_human_review": false
}
```

**Facts Extracted**:
```json
[
  {
    "fact_text": "Eligibility verification must be completed within 30 days of enrollment",
    "fact_type": "verification_method",
    "is_eligibility_related": true,
    "is_verified": true,
    "who_eligible": null,
    "how_verified": "Must be completed within 30 days of enrollment",
    "conflict_resolution": null,
    "when_applies": "At time of enrollment",
    "limitations": null,
    "confidence": 0.98
  },
  {
    "fact_text": "Retroactive eligibility may apply up to 90 days prior",
    "fact_type": "effective_date",
    "is_eligibility_related": true,
    "is_verified": true,
    "who_eligible": null,
    "how_verified": null,
    "conflict_resolution": null,
    "when_applies": "Up to 90 days prior to enrollment",
    "limitations": "May apply (not guaranteed)",
    "confidence": 0.92
  }
]
```

**Critique Agent Response** (if failed):
```json
{
  "pass": false,
  "feedback": "Missing fact: The paragraph also mentions that verification requires member ID submission, which was not extracted.",
  "issues": [
    {
      "type": "missing_fact",
      "description": "Fact about member ID requirement not extracted",
      "suggestion": "Add fact: 'Member ID must be submitted for verification'"
    }
  ],
  "confidence": 0.85
}
```

**Hierarchical Chunk** (failed after retries, flagged for human review):
```json
{
  "id": "...",
  "document_id": "...",
  "page_number": 45,
  "paragraph_index": 2,
  "section_path": "Section 3.2",
  "text": "Full paragraph text...",
  "summary": "Attempted extraction summary...",
  "is_eligibility_related": true,
  "confidence": 0.65,
  "extraction_status": "extracted",
  "critique_status": "failed",
  "critique_feedback": "Missing fact: The paragraph also mentions that verification requires member ID submission...",
  "retry_count": 2,
  "needs_human_review": true
}
```

---

## Next Steps After Step 7

- Step 8: Store extracted facts as atomic eligibility facts (with citations)
- Step 9: Semantic chunking (for chat/retrieval) - separate step
- Step 10: Query and retrieval system
