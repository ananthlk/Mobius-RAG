#!/usr/bin/env python3
"""Verify that policy_lines have p/d/j tags loaded for a Path B document.

Calls GET /documents/{id}/policy/lines and reports how many lines have tags,
plus a few sample lines with non-empty p_tags, d_tags, or j_tags.

Usage:
  # By document ID (e.g. from chunking log)
  python3 scripts/check_policy_lines_tags.py --document-id e123a670-8be6-4d60-8ad1-132d0847e929

  # By filename (first match)
  python3 scripts/check_policy_lines_tags.py --filename "MMA-LTC-Member-Handbook.pdf"

  # Custom base URL
  MOBIUS_API_URL=http://localhost:8001 python3 scripts/check_policy_lines_tags.py --document-id <uuid>
"""
import argparse
import json
import os
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

DEFAULT_BASE = "http://localhost:8001"
DEFAULT_FILENAME = "MMA-LTC-Member-Handbook.pdf"


def find_document_by_filename(base: str, filename_sub: str) -> str:
    req = Request(f"{base}/documents?limit=500", method="GET", headers={"Accept": "application/json"})
    with urlopen(req, timeout=15) as r:
        data = json.loads(r.read().decode())
    for doc in data.get("documents") or []:
        fn = doc.get("filename") or ""
        if filename_sub in fn or fn == filename_sub:
            return doc.get("id") or str(doc.get("id"))
    raise SystemExit(f"No document found for filename containing: {filename_sub}")


def fetch_policy_lines(base: str, document_id: str, limit: int = 5000) -> dict:
    url = f"{base}/documents/{document_id}/policy/lines?limit={limit}"
    req = Request(url, method="GET", headers={"Accept": "application/json"})
    with urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def main():
    parser = argparse.ArgumentParser(description="Check policy_lines p/d/j tags for a document")
    parser.add_argument("--document-id", type=str, help="Document UUID")
    parser.add_argument("--filename", type=str, default=DEFAULT_FILENAME, help="Filename substring if no --document-id")
    parser.add_argument("--base-url", type=str, default=os.environ.get("MOBIUS_API_URL", DEFAULT_BASE))
    parser.add_argument("--samples", type=int, default=5, help="Number of sample lines with tags to print")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    document_id = args.document_id
    if not document_id:
        try:
            document_id = find_document_by_filename(base, args.filename)
        except SystemExit as e:
            print(e, file=sys.stderr)
            sys.exit(1)
    print(f"Document ID: {document_id}")
    print(f"Fetching policy lines from {base}/documents/{document_id}/policy/lines ...")

    try:
        data = fetch_policy_lines(base, document_id)
    except HTTPError as e:
        print(f"HTTP error: {e.code} {e.reason}", file=sys.stderr)
        if e.read:
            try:
                print(e.read().decode()[:500], file=sys.stderr)
            except Exception:
                pass
        sys.exit(1)

    lines = data.get("lines") or []
    total = data.get("total") or len(lines)
    print(f"Total lines (returned): {len(lines)} (total in DB: {total})")

    with_p = sum(1 for ln in lines if ln.get("p_tags"))
    with_d = sum(1 for ln in lines if ln.get("d_tags"))
    with_j = sum(1 for ln in lines if ln.get("j_tags"))
    with_any = sum(1 for ln in lines if ln.get("p_tags") or ln.get("d_tags") or ln.get("j_tags"))

    print(f"Lines with p_tags: {with_p}")
    print(f"Lines with d_tags: {with_d}")
    print(f"Lines with j_tags: {with_j}")
    print(f"Lines with at least one tag: {with_any}")

    if with_any == 0:
        print("\nNo tags found on policy lines. Check that Path B ran with a loaded lexicon (see worker log for 'loaded lexicon with N phrases').")
        sys.exit(0)

    # Sample lines that have tags
    tagged = [ln for ln in lines if ln.get("p_tags") or ln.get("d_tags") or ln.get("j_tags")]
    n_show = min(args.samples, len(tagged))
    print(f"\n--- Sample of {n_show} lines with tags ---")
    for i, ln in enumerate(tagged[:n_show]):
        text = (ln.get("text") or "")[:80]
        if len((ln.get("text") or "")) > 80:
            text += "..."
        print(f"  [{i+1}] p_tags={ln.get('p_tags')} d_tags={ln.get('d_tags')} j_tags={ln.get('j_tags')}")
        print(f"      text: {text!r}")
    print("Tags are loaded correctly.")


if __name__ == "__main__":
    main()
