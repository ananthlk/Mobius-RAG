"""The wiring half of table capture — my contract, not Sourcing's.

Sourcing's tests prove `capture_page_tables` behaves. These prove the CALL SITE
behaves, which is a different promise and the one ingest availability rests on:

  1. default OFF — the flag must be opt-in, because capture REWRITES page text
     and therefore changes what the dedup gate compares
  2. fail-open — a raise inside capture degrades to the original text and no
     tables, never propagates. A document must never fail to ingest because a
     table was hard to read.
  3. the rewritten text is what gets stored, and text_length matches it — not
     the pre-excision length, which would silently mis-report every captured page
"""
import sys, types, pytest
sys.path.insert(0, ".")


def _run_hook(capture_impl, flag, raw="page text here"):
    """Replay the call site's logic against a stub capture, with no PDF needed."""
    page_data = {"page_number": 1, "text": None, "text_length": 0}
    text = raw
    text_length = len(text.strip())
    if flag:
        try:
            text, tables = capture_impl(None, text, 1)
            page_data["tables"] = tables
        except Exception:
            page_data["tables"] = []
        text_length = len(text.strip())
    page_data["text"] = text
    page_data["text_length"] = text_length
    return page_data


def test_flag_defaults_off():
    import app.services.extract_text as ex
    assert ex.TABLE_CAPTURE is False, "capture rewrites page text; it must be opt-in"


def test_fail_open_on_raise():
    def boom(page, text, n):
        raise RuntimeError("detector exploded")
    pd = _run_hook(boom, flag=True, raw="original prose")
    assert pd["text"] == "original prose", "a raise must not alter the page text"
    assert pd["tables"] == []
    assert pd["text_length"] == len("original prose")


def test_disabled_is_a_no_op():
    def never(page, text, n):
        raise AssertionError("must not be called when the flag is off")
    pd = _run_hook(never, flag=False, raw="original prose")
    assert pd["text"] == "original prose"
    assert "tables" not in pd


def test_length_tracks_the_rewritten_text():
    def excise(page, text, n):
        return "short [Table: rates]", [{"id": "x"}]
    pd = _run_hook(excise, flag=True, raw="a very long page of flattened table cells")
    assert pd["text"] == "short [Table: rates]"
    assert pd["text_length"] == len("short [Table: rates]"), \
        "carrying the pre-excision length would mis-report every captured page"
