"""
Google Drive API integration for OAuth-based document import.

Lists files, downloads PDFs and exports Google Docs as PDF.
"""
from __future__ import annotations

import io
import logging
import re
from datetime import datetime, timezone
from typing import Any

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import (
    DRIVE_API_ENABLED,
    GOOGLE_DRIVE_CLIENT_ID,
    GOOGLE_DRIVE_CLIENT_SECRET,
)
from app.models import DriveConnection

logger = logging.getLogger(__name__)

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
TOKEN_URI = "https://oauth2.googleapis.com/token"

# Supported MIME types: PDF (direct download), Google Docs (export as PDF)
SUPPORTED_MIMES = {
    "application/pdf",
    "application/vnd.google-apps.document",
}


def parse_folder_id(link_or_id: str) -> str | None:
    """Extract folder ID from Drive URL, return 'root' for My Drive, or raw ID."""
    if link_or_id is None:
        return None
    s = link_or_id.strip()
    if not s or s.lower() == "root":
        return "root"
    # Match https://drive.google.com/drive/folders/xxx or /file/d/xxx
    m = re.search(r"[/]folders[/]([a-zA-Z0-9_-]+)", s)
    if m:
        return m.group(1)
    # Raw ID: alphanumeric, underscore, hyphen
    if re.match(r"^[a-zA-Z0-9_-]+$", s):
        return s
    return None


async def get_credentials(
    session_id: str,
    db: AsyncSession,
) -> Credentials | None:
    """
    Load tokens from drive_connections, refresh if expired.
    Returns Credentials or None if no connection.
    """
    if not DRIVE_API_ENABLED or not GOOGLE_DRIVE_CLIENT_ID or not GOOGLE_DRIVE_CLIENT_SECRET:
        return None

    result = await db.execute(
        select(DriveConnection).where(DriveConnection.session_id == session_id)
    )
    conn = result.scalar_one_or_none()
    if not conn:
        return None

    creds = Credentials(
        token=conn.access_token,
        refresh_token=conn.refresh_token,
        token_uri=TOKEN_URI,
        client_id=GOOGLE_DRIVE_CLIENT_ID,
        client_secret=GOOGLE_DRIVE_CLIENT_SECRET,
        scopes=DRIVE_SCOPES,
        expiry=conn.expires_at,
    )

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            conn.access_token = creds.token
            conn.expires_at = creds.expiry
            await db.commit()
        except Exception as e:
            logger.warning("Drive token refresh failed: %s", e)
            return None

    return creds


FOLDER_MIME = "application/vnd.google-apps.folder"


def list_folder_files(creds: Credentials, folder_id: str) -> list[dict[str, Any]]:
    """
    List PDF and Google Doc files in a Drive folder.
    Returns list of {id, name, mimeType, size, webViewLink}.
    """
    contents = list_folder_contents(creds, folder_id)
    return contents["files"]


def list_folder_contents(creds: Credentials, folder_id: str) -> dict[str, Any]:
    """
    List folders and supported files in a Drive folder. Use 'root' for My Drive.
    Returns {folders: [...], files: [...], parent_id: str | null, folder_name: str}.
    """
    service = build("drive", "v3", credentials=creds)
    fid = folder_id if folder_id != "root" else "root"
    parent_id: str | None = None
    folder_name = "My Drive" if fid == "root" else ""

    if fid != "root":
        try:
            meta = service.files().get(fileId=fid, fields="name, parents").execute()
            folder_name = meta.get("name", "")
            parents = meta.get("parents") or []
            parent_id = parents[0] if parents else None
        except Exception as e:
            logger.warning("Drive folder meta failed for %s: %s", fid, e)

    query = f"'{fid}' in parents and trashed = false"
    items: list[dict[str, Any]] = []
    page_token = None
    while True:
        resp = (
            service.files()
            .list(
                q=query,
                pageSize=100,
                pageToken=page_token,
                fields="nextPageToken, files(id, name, mimeType, size, webViewLink)",
            )
            .execute()
        )
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    folders: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for it in items:
        mime = it.get("mimeType", "")
        if mime == FOLDER_MIME:
            folders.append(it)
        elif mime in SUPPORTED_MIMES:
            files.append(it)

    return {"folders": folders, "files": files, "parent_id": parent_id, "folder_name": folder_name}


def download_file(creds: Credentials, file_id: str, mime_type: str) -> bytes:
    """
    Download a file from Drive.
    - PDF: direct download via get_media
    - Google Docs: export as PDF (10MB limit)
    """
    service = build("drive", "v3", credentials=creds)

    if mime_type == "application/pdf":
        request = service.files().get_media(fileId=file_id)
    elif mime_type == "application/vnd.google-apps.document":
        request = service.files().export_media(
            fileId=file_id,
            mimeType="application/pdf",
        )
    else:
        raise ValueError(f"Unsupported mimeType for download: {mime_type}")

    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    return buf.read()
