"""
table_capture — pure table detection/excision for the extract_text_from_gcs page loop.

    capture_page_tables(fitz_page, raw_text, page_number) -> (clean_text, list[table])

Operates on the LIVE fitz (PyMuPDF) page — it has the ruling lines and cell bboxes a
detector needs; the flattened string does not. PURE: no DB, no I/O, no network.
FAIL-OPEN: any error degrades to (raw_text, []) — ingest availability outranks table
quality. Master RAG owns the call site, the TABLE_CAPTURE flag, and persistence.

Each returned `table` is a plain dict, mapped to the live `document_tables` columns by
the call site at persist:
  {id, page_number, table_index, grid{header,rows}, anchor{section,page,bbox},
   coverage[codes], caption, strategy, n_rows, n_cols}  (+ `bbox`/`breadcrumb`, excision-only)
The breadcrumb left in clean_text is `[Table: <caption> · ->document_tables:<id>]`; a
retrieved chunk resolves its row by that uuid (DB owns the row; `id` is its PK).
"""
from __future__ import annotations
import logging
import re
import uuid

logger = logging.getLogger(__name__)

CODE = re.compile(r"[A-Z]\d{4}|(?<!\d)\d{5}(?!\d)")   # HCPCS Level-II + CPT-ish (IDs filtered later by registry)


def _clean(c) -> str:
    return (c or "").replace("\n", " ").replace("\xa0", " ").strip()


def _split_header(grid):
    """Header/data split by STRUCTURE, not content (no numeric assumption). Header =
    leading rows up to & incl. the first 'full' row (column labels, possibly under a
    sparse merged super-header), merged per column; data = the body below."""
    ncols = max(len(r) for r in grid)
    fill = [sum(1 for c in r if c) for r in grid]
    full = [i for i, f in enumerate(fill) if f >= 0.7 * ncols]
    ds = (full[0] + 1) if full else 1
    ds = max(1, min(ds, len(grid) - 1)) if len(grid) > 1 else 1
    header = [" ".join(grid[r][c] for r in range(ds) if c < len(grid[r]) and grid[r][c]).strip()
              for c in range(ncols)]
    return header, grid[ds:], ncols


def _is_table(data, ncols) -> bool:
    """A real table = a consistently-shaped body. Merged/imperfect header tolerated."""
    if len(data) < 2 or ncols < 2:
        return False
    bodyfill = [sum(1 for c in r if c) for r in data]
    mode = max(set(bodyfill), key=bodyfill.count)
    consistent = sum(1 for f in bodyfill if abs(f - mode) <= 1) / len(bodyfill)
    return mode >= 0.5 * ncols and consistent >= 0.6


def _caption(page, bbox) -> str:
    """Nearest text block just above the table (x-overlapping, smallest gap)."""
    x0, y0, x1, _ = bbox
    best, best_gap = "", 1e9
    try:
        for b in page.get_text("blocks"):
            bx0, by0, bx1, by1, txt = b[0], b[1], b[2], b[3], b[4]
            if by1 <= y0 and bx1 > x0 and bx0 < x1 and (y0 - by1) < best_gap and txt.strip():
                best_gap, best = (y0 - by1), " ".join(txt.split())[:80]
    except Exception:
        pass
    return best


def _find_tables(page):
    """Ruled detection first, fall back to text/whitespace. Returns (tables, strategy)."""
    for name, kwargs in (("lines", dict()),
                         ("text", dict(vertical_strategy="text", horizontal_strategy="text"))):
        try:
            tabs = page.find_tables(**kwargs).tables if kwargs else page.find_tables().tables
        except Exception:
            tabs = []
        if tabs:
            return tabs, name
    return [], "none"


def _excise(page, page_number, captured, raw_text):
    """Rebuild page text with each captured table's region replaced by one breadcrumb.
    Non-table blocks kept in reading order; table blocks dropped."""
    boxes = [(t["bbox"], t) for t in captured]
    def which(cx, cy):
        for i, (bb, _) in enumerate(boxes):
            if bb[0] - 1 <= cx <= bb[2] + 1 and bb[1] - 1 <= cy <= bb[3] + 1:
                return i
        return None
    items, placed = [], set()
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
        i = which((x0 + x1) / 2, (y0 + y1) / 2)
        if i is None:
            if txt.strip():
                items.append((y0, x0, txt.strip()))
        elif i not in placed:
            placed.add(i)
            bb, t = boxes[i]
            items.append((bb[1], bb[0], t["breadcrumb"]))
    if not placed:                     # nothing actually excised → don't claim tables
        return None
    items.sort(key=lambda z: (round(z[0], 1), round(z[1], 1)))
    return "\n".join(t for _, _, t in items)


def capture_page_tables(fitz_page, raw_text, page_number):
    try:
        tabs, strategy = _find_tables(fitz_page)
        if not tabs:
            return raw_text, []
        captured = []
        for idx, t in enumerate(tabs):
            try:
                grid = [[_clean(c) for c in r] for r in t.extract()]
                grid = [r for r in grid if any(r)]
                if not grid:
                    continue
                header, data, ncols = _split_header(grid)
                if not _is_table(data, ncols):
                    continue                       # low-confidence → leave in prose, don't excise
                bbox = tuple(round(v, 1) for v in t.bbox)
                codes = sorted(set(CODE.findall(" ".join(c for r in grid for c in r))))
                cap = _caption(fitz_page, bbox)
                table_id = str(uuid.uuid4())       # breadcrumb ↔ document_tables.id link (DB owns the row)
                captured.append({
                    # keys mapped to document_tables columns by Master RAG at persist:
                    "id": table_id,
                    "page_number": page_number,
                    "table_index": idx,                              # stable natural-key part → UNIQUE(doc,page,table_index)
                    "grid": {"header": header, "rows": data},        # jsonb; shape varies by table
                    "anchor": {"section": cap, "page": page_number, "bbox": bbox},
                    "coverage": codes,                               # text[]; codes pre-registry-validation
                    "caption": cap,
                    "strategy": strategy,                            # lines | text
                    "n_rows": len(data), "n_cols": ncols,
                    # excision-only (not persisted):
                    "bbox": bbox,
                    "breadcrumb": f"[Table: {cap or 'data table'} · →document_tables:{table_id}]",
                })
            except Exception as e:
                logger.warning("table_capture: table %s on page %s skipped: %s", idx, page_number, e)
                continue
        # Detection-recall telemetry for Eval (stage 4): document_tables holds only the
        # CAPTURED rows, so the 'found' count lives here. found = raw detected; captured = kept.
        logger.info("table_capture_telemetry page=%s found=%s captured=%s strategy=%s",
                    page_number, len(tabs), len(captured), strategy)
        if not captured:
            return raw_text, []
        clean = _excise(fitz_page, page_number, captured, raw_text)
        if clean is None:                          # excision produced nothing to replace → degrade
            return raw_text, []
        return clean, captured
    except Exception as e:                          # FAIL-OPEN: never take ingest down
        logger.warning("table_capture: page %s failed, passing through: %s", page_number, e)
        return raw_text, []
