"""
table_capture — pure table detection/excision for the extract_text_from_gcs page loop.

    capture_page_tables(fitz_page, raw_text, page_number) -> (clean_text, list[table])

Operates on the LIVE fitz (PyMuPDF) page — it has the ruling lines and cell bboxes a
detector needs; the flattened string does not. PURE: no DB, no I/O, no network.
FAIL-OPEN: any error degrades to (raw_text, []) — ingest availability outranks table
quality. Master RAG owns the call site, the TABLE_CAPTURE flag, and persistence.

Detection routes on ACCEPTANCE, not first-detection: every strategy is run, each grid
is gated, and the accepted tables are merged (dedup by bbox overlap, larger kept) so a
rejected `lines` grid never masks a good `text` grid. Excision is rotation-aware
(bboxes derotated into the get_text frame) — the flattened LIP models are 90°-rotated.

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

_STRATEGIES = [
    ("lines", None),
    ("text", dict(vertical_strategy="text", horizontal_strategy="text")),
    ("text_relaxed", dict(vertical_strategy="text", horizontal_strategy="text",
                          min_words_vertical=2, min_words_horizontal=1, snap_tolerance=6)),
]


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
    """A real table = a consistently-shaped body. Tuned to accept WIDE SPARSE grids
    (financial tables with many spacer columns) that a 0.5*ncols floor wrongly rejects:
    require an absolute floor of filled cells + a modest fill fraction + row consistency."""
    if len(data) < 2 or ncols < 2:
        return False
    bodyfill = [sum(1 for c in r if c) for r in data]
    mode = max(set(bodyfill), key=bodyfill.count)
    consistent = sum(1 for f in bodyfill if abs(f - mode) <= 1) / len(bodyfill)
    return mode >= 2 and (mode / ncols) >= 0.30 and consistent >= 0.6


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


def _area(bb):
    return max(0.0, bb[2] - bb[0]) * max(0.0, bb[3] - bb[1])


def _overlap(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    aa = _area(a)
    return (ix * iy / aa) if aa else 0.0


def _build_one(t, page, page_number, strategy):
    """Extract + gate one fitz Table → a table dict, or None if it fails the gate."""
    grid = [[_clean(c) for c in r] for r in t.extract()]
    grid = [r for r in grid if any(r)]
    if not grid:
        return None
    header, data, ncols = _split_header(grid)
    if not _is_table(data, ncols):
        return None
    bbox = tuple(round(v, 1) for v in t.bbox)
    codes = sorted(set(CODE.findall(" ".join(c for r in grid for c in r))))
    cap = _caption(page, bbox)
    table_id = str(uuid.uuid4())
    return {
        "id": table_id,
        "page_number": page_number,
        "grid": {"header": header, "rows": data},
        "anchor": {"section": cap, "page": page_number, "bbox": bbox},
        "coverage": codes,
        "caption": cap,
        "strategy": strategy,
        "n_rows": len(data), "n_cols": ncols,
        "bbox": bbox,                       # excision-only
        "breadcrumb": f"[Table: {cap or 'data table'} · →document_tables:{table_id}]",
    }


def _derotate_bbox(page, bb):
    """Map a find_tables bbox into the get_text/blocks frame (identity when unrotated)."""
    m = page.derotation_matrix
    pts = [(bb[0], bb[1]), (bb[2], bb[1]), (bb[2], bb[3]), (bb[0], bb[3])]
    xs = [x * m.a + y * m.c + m.e for x, y in pts]
    ys = [x * m.b + y * m.d + m.f for x, y in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _excise(page, page_number, captured, raw_text):
    """Rebuild page text with each captured table's region replaced by one breadcrumb.
    Table bboxes are derotated into the block frame so containment holds on rotated pages."""
    boxes = [(_derotate_bbox(page, t["bbox"]), t) for t in captured]
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
    if not placed:
        return None
    items.sort(key=lambda z: (round(z[0], 1), round(z[1], 1)))
    return "\n".join(t for _, _, t in items)


def capture_page_tables(fitz_page, raw_text, page_number):
    try:
        # Route on ACCEPTANCE: run every strategy, gate each grid, keep all accepted.
        cand = []
        for strat, kw in _STRATEGIES:
            try:
                tabs = fitz_page.find_tables(**kw).tables if kw else fitz_page.find_tables().tables
            except Exception:
                tabs = []
            for t in tabs:
                try:
                    built = _build_one(t, fitz_page, page_number, strat)
                except Exception as e:
                    logger.warning("table_capture: build failed page %s: %s", page_number, e)
                    built = None
                if built:
                    cand.append(built)
        # Merge across strategies: dedup overlapping regions, keep the larger.
        cand.sort(key=lambda c: -_area(c["bbox"]))
        kept = []
        for c in cand:
            if not any(_overlap(c["bbox"], k["bbox"]) > 0.5 for k in kept):
                kept.append(c)
        # Stable table_index in reading order.
        kept.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))
        for i, c in enumerate(kept):
            c["table_index"] = i
        logger.info("table_capture_telemetry page=%s captured=%s strategies=%s",
                    page_number, len(kept), sorted({c["strategy"] for c in kept}))
        if not kept:
            return raw_text, []
        clean = _excise(fitz_page, page_number, kept, raw_text)
        if clean is None:
            return raw_text, []
        return clean, kept
    except Exception as e:                          # FAIL-OPEN: never take ingest down
        logger.warning("table_capture: page %s failed, passing through: %s", page_number, e)
        return raw_text, []
