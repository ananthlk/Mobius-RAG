/**
 * Shared API helpers for fact actions (approve, edit, delete).
 * Used by ReviewFactsTab and DocumentReaderTab for consistent behavior.
 */

import { API_BASE } from '../config'

export async function approveFactApi(
  documentId: string,
  factId: string,
  baseUrl: string = API_BASE
): Promise<Response> {
  return fetch(`${baseUrl}/documents/${documentId}/facts/${factId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ verification_status: 'approved' }),
  })
}

export async function rejectFactApi(
  documentId: string,
  factId: string,
  baseUrl: string = API_BASE
): Promise<Response> {
  return fetch(`${baseUrl}/documents/${documentId}/facts/${factId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ verification_status: 'rejected' }),
  })
}

export async function deleteFactApi(
  documentId: string,
  factId: string,
  baseUrl: string = API_BASE
): Promise<Response> {
  return fetch(`${baseUrl}/documents/${documentId}/facts/${factId}`, {
    method: 'DELETE',
  })
}

export async function patchFactApi(
  documentId: string,
  factId: string,
  body: { fact_text?: string; verification_status?: string; [key: string]: unknown },
  baseUrl: string = API_BASE
): Promise<Response> {
  return fetch(`${baseUrl}/documents/${documentId}/facts/${factId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}
