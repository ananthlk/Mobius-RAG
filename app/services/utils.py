"""Utility functions for parsing LLM responses and text normalization."""
import json
import logging
import re

logger = logging.getLogger(__name__)


def normalize_page_text(text: str) -> str:
    """Normalize page text to match frontend ReadDocumentTab (normalizeText / PageTextWithHighlights).
    Used so backend-stored offsets match the normalized string the frontend displays."""
    if not text:
        return ""
    lines = text.split("\n")
    normalized_lines = [re.sub(r"\s+", " ", line).strip() for line in lines]
    normalized = "\n".join(normalized_lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    return normalized


def raw_span_to_normalized_span(
    page_text_raw: str, raw_start: int, raw_end: int
) -> tuple[int | None, int | None]:
    """Convert a raw page text span [raw_start, raw_end) to normalized page text offsets.
    Returns (norm_start, norm_end) or (None, None) if the span cannot be located."""
    normalized_page = normalize_page_text(page_text_raw)
    if raw_start >= raw_end or raw_start < 0 or raw_end > len(page_text_raw):
        return (None, None)
    span_raw = page_text_raw[raw_start:raw_end]
    span_norm = normalize_page_text(span_raw)
    if not span_norm:
        return (None, None)
    idx = normalized_page.find(span_norm)
    if idx < 0:
        return (None, None)
    return (idx, idx + len(span_norm))


def _preprocess_response(response: str) -> str:
    """Strip markdown, preamble, and return content from first '{'."""
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()
    idx = response.find("{")
    if idx >= 0:
        response = response[idx:]
    return response


def _normalize_extraction(obj: dict) -> dict:
    """Ensure extraction result has summary and facts; default if missing."""
    if not isinstance(obj, dict):
        return obj
    if "summary" not in obj:
        obj["summary"] = ""
    if "facts" not in obj or not isinstance(obj["facts"], list):
        obj["facts"] = []
    return obj


def _try_close_truncated_json(s: str) -> str:
    """If string looks truncated (ends with comma or incomplete key), append closing brackets."""
    s = s.rstrip()
    if not s or s[-1] == "}":
        return s
    # Ended mid-key (opening quote only): close key with null, then object/array/root
    if s[-1] == '"':
        return s + '": null}]}'
    # Ended mid-value after comma: close value, then object/array/root
    if s[-1] == ",":
        return s + " null}]}"
    # Ended after colon: add null and close
    if s[-1] == ":":
        return s + " null}]}"
    return s


def parse_json_response(response: str) -> dict:
    """Parse JSON from LLM response, handling markdown, preamble, and trailing text (extra data).
    Tries: 1) strict parse, 2) truncate at last '}', 3) json_repair on full, 4) json_repair on truncated, 5) close truncated and repair.
    Returns normalized dict with at least summary and facts (so caller can proceed); raises only if nothing usable."""
    preprocessed = _preprocess_response(response)
    first_error = None

    try:
        dec = json.JSONDecoder()
        obj, _ = dec.raw_decode(preprocessed)
        return _normalize_extraction(obj)
    except json.JSONDecodeError as e:
        first_error = e
        logger.error(f"Failed to parse JSON: {e}")
        logger.error(f"Response was: {response[:500]}")

    # Try to recover partial JSON by truncating at last complete '}'
    try:
        last_brace = preprocessed.rfind("}")
        if last_brace > 0:
            partial = preprocessed[: last_brace + 1]
            if partial.count("{") == partial.count("}"):
                dec = json.JSONDecoder()
                obj, _ = dec.raw_decode(partial)
                logger.warning("Recovered partial JSON by truncating at last complete brace")
                return _normalize_extraction(obj)
    except Exception:
        pass

    try:
        import json_repair
    except ImportError:
        json_repair = None

    # Try json_repair on full preprocessed string
    if json_repair:
        try:
            obj = json_repair.loads(preprocessed)
            if isinstance(obj, dict):
                logger.warning("Recovered JSON using json_repair after strict parse failed")
                return _normalize_extraction(obj)
        except Exception as repair_err:
            logger.debug(f"json_repair on full failed: {repair_err}")

    # Try json_repair on truncated (balanced braces) string
    if json_repair:
        try:
            last_brace = preprocessed.rfind("}")
            if last_brace > 0:
                partial = preprocessed[: last_brace + 1]
                if partial.count("{") == partial.count("}"):
                    obj = json_repair.loads(partial)
                    if isinstance(obj, dict):
                        logger.warning("Recovered JSON using json_repair on truncated string")
                        return _normalize_extraction(obj)
        except Exception:
            pass

    # Try closing truncated JSON then repair (e.g. ends with "is_verified": true,\n  ")
    if json_repair:
        try:
            closed = _try_close_truncated_json(preprocessed)
            if closed != preprocessed:
                obj = json_repair.loads(closed)
                if isinstance(obj, dict):
                    logger.warning("Recovered JSON by closing truncated string and using json_repair")
                    return _normalize_extraction(obj)
        except Exception:
            pass

    if first_error is not None:
        raise first_error
    raise ValueError("Failed to parse JSON")


def parse_json_response_best_effort(response: str) -> dict | None:
    """Try to get any usable extraction from malformed/truncated JSON. Returns dict with summary and facts, or None."""
    preprocessed = _preprocess_response(response)
    try:
        import json_repair
    except ImportError:
        json_repair = None
    if not json_repair:
        return None
    # Try closed truncated + repair
    for candidate in [preprocessed, _try_close_truncated_json(preprocessed)]:
        try:
            obj = json_repair.loads(candidate)
            if isinstance(obj, dict) and ("summary" in obj or "facts" in obj):
                return _normalize_extraction(obj)
        except Exception:
            pass
    return None
