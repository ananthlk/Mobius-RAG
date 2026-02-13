#!/usr/bin/env python3
"""Start Path B chunking for a document and stream chunking events to the terminal.

The document MMA-LTC-Member-Handbook.pdf is already uploaded; this script finds it by
filename, starts a Path B chunking job, then polls the chunking-events API and prints
each event so you can watch the log here.

Usage:
  # Default: find "MMA-LTC-Member-Handbook.pdf", start Path B, stream events
  python3 scripts/run_chunking_and_stream_log.py

  # By document ID
  python3 scripts/run_chunking_and_stream_log.py --document-id <uuid>

  # By filename substring (first match)
  python3 scripts/run_chunking_and_stream_log.py --filename "MMA-LTC"

  # Custom base URL (default: http://localhost:8001 for RAG backend)
  MOBIUS_API_URL=http://localhost:8001 python3 scripts/run_chunking_and_stream_log.py

Requires the Mobius RAG API and worker to be running (same DB). Events are written by
the worker to the chunking_events table; this script polls GET /documents/{id}/chunking/events.
"""
import argparse
import json
import os
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

# RAG backend (mstart: mobius-rag runs on 8001; mobius-chat is 8000)
DEFAULT_BASE = "http://localhost:8001"
DEFAULT_FILENAME = "MMA-LTC-Member-Handbook.pdf"
POLL_INTERVAL = 1.0
TIMEOUT_SECONDS = 3600  # 1 hour max wait for completion


def main():
    parser = argparse.ArgumentParser(description="Start Path B chunking and stream events to terminal")
    parser.add_argument("--document-id", type=str, help="Document UUID (skip lookup)")
    parser.add_argument("--filename", type=str, default=DEFAULT_FILENAME, help=f"Filename to search (default: {DEFAULT_FILENAME})")
    parser.add_argument("--base-url", type=str, default=os.environ.get("MOBIUS_API_URL", DEFAULT_BASE), help="API base URL")
    parser.add_argument("--no-start", action="store_true", help="Only stream existing events (do not start a new job)")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    document_id = args.document_id
    if not document_id:
        document_id = find_document_by_filename(base, args.filename)
        if not document_id:
            print(f"No document found with filename matching {args.filename!r}", file=sys.stderr)
            sys.exit(1)
        print(f"Found document: {document_id}")

    if not args.no_start:
        start_path_b_chunking(base, document_id)
        print("Path B chunking started. Streaming events...\n")
    else:
        print("Streaming existing events (no new job started)...\n")

    last_id = None
    start = time.time()
    while True:
        events, last_id, is_asc = fetch_events(base, document_id, after_id=last_id)
        # API returns desc on initial load (newest first); with after_id returns asc (oldest first). Print chronological.
        to_print = list(reversed(events)) if not is_asc and events else events
        for ev in to_print:
            print_event(ev)
            if ev.get("event") == "chunking_complete":
                print("\nChunking complete.")
                return
        if not events and last_id is None:
            # No events yet; keep polling with after_id=None until we see events
            last_id = None
        if time.time() - start > TIMEOUT_SECONDS:
            print("\nTimeout waiting for chunking_complete.", file=sys.stderr)
            sys.exit(1)
        time.sleep(POLL_INTERVAL)


def find_document_by_filename(base: str, filename_substring: str) -> str | None:
    url = f"{base}/documents?limit=500"
    data = get_json(url)
    for doc in data.get("documents") or []:
        fn = doc.get("filename") or ""
        if filename_substring in fn or fn == filename_substring:
            return doc.get("id")
    return None


def start_path_b_chunking(base: str, document_id: str) -> None:
    url = f"{base}/documents/{document_id}/chunking/start"
    body = {
        "generator_id": "B",
        "extraction_enabled": False,
        "critique_enabled": False,
    }
    req = Request(url, data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=30) as resp:
            if resp.status not in (200, 201, 202):
                print(f"Start chunking returned {resp.status}", file=sys.stderr)
                sys.exit(1)
    except HTTPError as e:
        if e.code == 409:
            print("A chunking job is already queued or in progress for this document.", file=sys.stderr)
        else:
            print(f"Failed to start chunking: {e.code} {e.reason}", file=sys.stderr)
        sys.exit(1)


def fetch_events(base: str, document_id: str, after_id: str | None, limit: int = 500) -> tuple[list, str | None, bool]:
    if after_id:
        url = f"{base}/documents/{document_id}/chunking/events?after_id={after_id}&limit={limit}"
        is_asc = True  # API returns asc when after_id is set
    else:
        url = f"{base}/documents/{document_id}/chunking/events?limit={limit}"
        is_asc = False  # API returns desc on initial load
    data = get_json(url)
    events = data.get("events") or []
    # For next poll: use id of the newest event we have (first in desc, last in asc)
    if events:
        last_id = events[-1].get("id") if is_asc else events[0].get("id")
    else:
        last_id = after_id  # keep same after_id so we don't re-fetch old events
    return events, last_id, is_asc


def get_json(url: str) -> dict:
    req = Request(url)
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def print_event(ev: dict) -> None:
    ts = ev.get("timestamp", "")[:19] if ev.get("timestamp") else ""
    typ = ev.get("event", "")
    data = ev.get("data") or {}
    msg = data.get("message") if isinstance(data.get("message"), str) else None
    if msg:
        print(f"[{ts}] {typ}: {msg}")
    elif typ == "progress_update":
        done = data.get("completed_paragraphs", data.get("paragraph_index"))
        total = data.get("total_paragraphs")
        stage = data.get("stage") or ""
        para = data.get("current_paragraph", "")
        if total is not None:
            print(f"[{ts}] {typ}: {done}/{total} {stage} {para}".strip())
        else:
            print(f"[{ts}] {typ}: {data}")
    elif typ == "chunking_complete":
        print(f"[{ts}] {typ}: {data}")
    else:
        print(f"[{ts}] {typ}: {data}")


if __name__ == "__main__":
    main()
