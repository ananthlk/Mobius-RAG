# Design: Markdown as Master (Regardless of Doc Type)

## Your questions, answered

### 1. Is the current structure “just rendering”?

**Partly.** Right now:

- **Display:** We derive structure (headers, paragraphs) in two places:
  - **When we have `text_markdown`** (no highlights): we render that markdown (SimpleMarkdown).
  - **When we have highlights or no markdown:** we build structure from **raw page text** (split by `\n\n`, detect section headers) for display and for highlight offsets.
- **LLM pipeline:** Uses **raw page text** only. Chunking (`split_paragraphs` in `app/services/chunking.py`) splits raw text into paragraphs and detects section headers, then sends the LLM `"## Section\n\nparagraph body"` for each chunk. So the LLM never sees `text_markdown`; it sees structure derived from the same raw text.

So: the “formatting” we fixed is **display-time** structure. The LLM already gets structured input, but that structure is computed again from raw text inside the chunking step, not from a single canonical format.

### 2. Would this work when passed to the LLM?

**Yes.** The pipeline already passes structured text to the LLM (paragraph + optional `## Section` context). It works today because chunking and extraction both use the same raw text. If we switched to “markdown as master” (below), we’d pass chunks from that markdown instead; LLMs handle markdown well.

### 3. What if the doc was PDF, Excel, or PowerPoint?

**Current state:**

- **PDF:** Supported. We extract raw text per page (PyMuPDF), then derive both display structure and chunking from that text. Logic assumes paragraph-like flow and `\n\n` boundaries.
- **Excel / PowerPoint:** **Not supported.** There is no extractor for `.xlsx` or `.pptx`. If we added them:
  - **Excel:** Extraction would give cells/tables, not “paragraphs” and “headers” in the same way. Our current “split by `\n\n`, detect short lines as headers” would not match the structure of a spreadsheet.
  - **PowerPoint:** We’d get slide-by-slide content (titles, bullets). Again, structure is different from “page of prose with double newlines.”

So the **current logic is PDF/text-oriented**. It would **not** generalize cleanly to Excel or PowerPoint without format-specific handling.

### 4. Should we convert everything to .md first and use that as the master?

**Yes.** Using **markdown as the single canonical format** is a good approach:

- **Regardless of source type** (PDF, Excel, PowerPoint, etc.), we:
  1. **Extract** content in a format-specific way (PDF text, Excel tables, PPT slides).
  2. **Convert** that content into a **canonical markdown** representation (per document or per page).
  3. **Use that .md as the only source** for:
     - Reader display (we already have markdown rendering).
     - Chunking and LLM (split the markdown into sections/paragraphs).
     - Storing highlights (character offsets in the markdown text).
     - Any search or RAG over the doc.

Then:

- **One format** to reason about: all downstream code (display, chunking, highlights, LLM) consumes markdown only.
- **LLMs** work well with markdown (headers, lists, tables).
- **Excel** can be converted to markdown tables and optional headers; **PowerPoint** to `## Slide N`, bullets, etc. The rest of the pipeline stays the same.

---

## Proposed architecture: MD as master

Pipeline: **Upload → Store → Convert to MD → Chunk**

- **Upload:** File stored (e.g. GCS).
- **Store:** Raw content extracted and stored per page (e.g. `document_pages.text`).
- **Convert to MD:** Raw content is converted to canonical markdown and stored (e.g. `document_pages.text_markdown`). This is its own step so the pipeline is explicit and future formats (Excel, PPT) plug in the same way.
- **Chunk:** Markdown is split into sections/paragraphs; chunking and LLM use only the markdown.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. Ingest (format-specific)                                            │
│  PDF  → extract text per page  → raw text per page                       │
│  XLSX → extract sheets/cells   → e.g. raw cells / structure              │
│  PPTX → extract slides         → e.g. slide titles, bullets              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  2. Store (raw)                                                          │
│  - Raw content per page in DB: document_pages.text                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  3. Convert to MD                                                        │
│  - One .md per page (or per document). Stored: document_pages.text_markdown │
│  - This .md is the ONLY source of truth for display, chunking, highlights │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  4. Chunk                                                                │
│  - Split markdown into sections/paragraphs → chunking, LLM, highlights   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Concrete steps

1. **Define “master” content**
   - Keep `text` as raw (Store step), add/use `text_markdown` (Convert to MD step). Use only `text_markdown` for display, chunking, and highlights.

2. **Pipeline: four steps**
   - **Upload:** Store file (e.g. GCS).
   - **Store:** Extract and save raw content per page (`document_pages.text`).
   - **Convert to MD:** Convert raw → canonical markdown and save (`document_pages.text_markdown`). Own step so the pipeline is explicit.
   - **Chunk:** Split markdown; run extraction/LLM and store offsets in markdown.

3. **PDF path**
   - Store: current extraction (raw text per page).
   - Convert to MD: one function (e.g. `raw_page_to_markdown`) writes `text_markdown`. All downstream code reads from this markdown.

4. **Excel / PowerPoint (future)**
   - Add extractors that output **markdown** (tables, slide headings, bullets), not free-form raw text.
   - Store that markdown the same way as for PDF. Chunking, display, and highlights then work the same.

5. **Chunking and LLM**
   - **Input:** Markdown (per page or per document).
   - **Process:** Split markdown by `##` and `\n\n` (or a small markdown-aware splitter) into sections/paragraphs. Send those chunks to the LLM. No more separate “raw + section detection” in chunking; the structure is already in the markdown.

6. **Highlights**
   - Store offsets in the **markdown** text (same as we do today for “normalized” text). Display uses the same markdown, so highlights stay aligned.

7. **Display**
   - Always render from the master markdown (with or without highlights). No dual path (raw vs markdown); one source, one renderer.

---

## Summary

| Question | Answer |
|----------|--------|
| Is this just rendering? | The formatting fix is display-time; LLM already gets structure from raw text via chunking. |
| Does it work for the LLM? | Yes; moving to “md as master” would still work well for the LLM. |
| PDF / Excel / PowerPoint? | Current logic is PDF-oriented; Excel/PPT need format-specific conversion. |
| Convert to .md first and use as master? | **Yes.** One canonical markdown per doc/page, then use it for display, chunking, highlights, and LLM. |

Implementing “md as master” would involve: (1) making markdown the single stored content for each page (or doc), (2) switching chunking and highlight offsets to that markdown, and (3) adding format-specific converters when you support Excel or PowerPoint.
