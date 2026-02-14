/**
 * RAG-specific fact tooltip – renders inside the generic dv-tooltip shell
 * provided by @mobius/document-viewer.
 */

/** Short labels for compact tooltip display (supports both old flat codes and new dotted codes). */
const CATEGORY_SHORT: Record<string, string> = {
  // New dotted taxonomy
  'claims.submission': 'Claims submission',
  'claims.denial': 'Denial',
  'claims.appeals_grievances': 'Appeals',
  'claims.clean_claim': 'Clean claim',
  'claims.timely_filing': 'Timely filing',
  'claims.coordination_of_benefits': 'COB',
  'claims.electronic_claims': 'E-claims',
  'eligibility.verification': 'Elig. verification',
  'eligibility.enrollment': 'Enrollment',
  'utilization_management.prior_authorization': 'Prior auth',
  'utilization_management.referrals': 'Referrals',
  'credentialing.general': 'Credentialing',
  'compliance.fraud_waste_abuse': 'FWA',
  'compliance.hipaa': 'HIPAA',
  'provider.network': 'Provider network',
  'pharmacy.pharmacy_benefit': 'Pharmacy',
  'responsibilities.continuity_of_care': 'Continuity',
  'contact_information.phone': 'Phone',
  // Legacy flat codes (backward compat)
  contacting_marketing_members: 'Contacting members',
  member_eligibility_molina: 'Eligibility',
  benefit_access_limitations: 'Benefits / access',
  prior_authorization_required: 'Prior auth',
  claims_authorization_submissions: 'Claims / auth',
  compliant_claim_requirements: 'Compliant claims',
  claim_disputes: 'Disputes',
  credentialing: 'Credentialing',
  claim_submission_important: 'Claim submission',
  coordination_of_benefits: 'COB',
  other_important: 'Other',
}

function dirSymbol(d: number | null): string {
  if (d === 1) return '\u2191'   // ↑
  if (d === 0) return '\u2193'   // ↓
  return '\u2192'                // →
}

export interface FactTooltipData {
  type: 'fact'
  factText: string
  isPertinent: boolean
  verificationStatus?: string | null
  categoryScores?: Record<string, { score: number | null; direction: number | null }>
}

export function FactTooltipContent({ data }: { data: FactTooltipData }) {
  const pertinent = data.isPertinent
  const topCategories = data.categoryScores
    ? Object.entries(data.categoryScores)
        .filter(([, v]) => v && typeof v.score === 'number' && (v.score ?? 0) > 0)
        .sort(([, a], [, b]) => (b?.score ?? 0) - (a?.score ?? 0))
        .slice(0, 2)
        .map(([k, v]) => ({
          key: k,
          label: CATEGORY_SHORT[k] ?? k.replace(/_/g, ' '),
          score: v?.score ?? 0,
          direction: v?.direction ?? null,
        }))
    : []

  return (
    <>
      <div style={{ fontWeight: 600, color: pertinent ? '#10b981' : '#6b7280', marginBottom: topCategories.length > 0 ? '0.375rem' : 0 }}>
        {pertinent ? '\u2713' : '\u25cb'} {pertinent ? 'Pertinent' : 'Not pertinent'}
      </div>
      {topCategories.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
          {topCategories.map(({ key, label, score, direction }) => (
            <div key={key} style={{ display: 'flex', justifyContent: 'space-between', gap: '0.5rem', fontSize: '0.75rem' }}>
              <span style={{ color: '#374151' }}>{label}</span>
              <span style={{ fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>
                {((score ?? 0) * 100).toFixed(0)}% {dirSymbol(direction)}
              </span>
            </div>
          ))}
        </div>
      )}
    </>
  )
}

export interface TagTooltipData {
  type: 'tag'
  tag: string
  tagLabel: string
  taggedText: string
}

export function TagTooltipContent({ data }: { data: TagTooltipData }) {
  return (
    <div style={{ fontSize: '0.8125rem' }}>
      <div style={{ fontWeight: 600, color: '#1f2937', marginBottom: '0.25rem' }}>
        Tag: {data.tagLabel}
      </div>
      <div style={{ color: '#6b7280', fontSize: '0.75rem', maxWidth: 240, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {data.taggedText}
      </div>
    </div>
  )
}

/* ─── Policy-line tag tooltip (Path B pipeline tags) ─── */

const PTAG_LABELS: Record<string, string> = {
  appeal: 'Appeal', call: 'Call', contact: 'Contact', email: 'Email',
  resubmit: 'Resubmit', review: 'Review', submit: 'Submit', verify: 'Verify',
}

function formatTagKeys(tags: Record<string, number> | null | undefined, labels?: Record<string, string>): string[] {
  if (!tags) return []
  return Object.entries(tags)
    .filter(([, v]) => v != null && v > 0)
    .map(([k]) => {
      if (labels?.[k]) return labels[k]
      // Handle dotted codes: "claims.denial" -> "Claims > Denial"
      return k
        .replace(/\./g, ' > ')
        .replace(/_/g, ' ')
        .replace(/\b\w/g, (c) => c.toUpperCase())
    })
}

export function PolicyLineTagTooltipContent({ data }: { data: Record<string, unknown> }) {
  const pTags = data.p_tags as Record<string, number> | null
  const dTags = data.d_tags as Record<string, number> | null
  const jTags = data.j_tags as Record<string, number> | null

  const pLabels = formatTagKeys(pTags, PTAG_LABELS)
  const dLabels = formatTagKeys(dTags)
  const jLabels = formatTagKeys(jTags)

  return (
    <div style={{ fontSize: '0.8125rem', maxWidth: 280 }}>
      {pLabels.length > 0 && (
        <div style={{ marginBottom: '0.25rem' }}>
          <span style={{ fontWeight: 600, color: '#7c3aed', fontSize: '0.6875rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Process </span>
          <span style={{ color: '#374151' }}>{pLabels.join(', ')}</span>
        </div>
      )}
      {dLabels.length > 0 && (
        <div style={{ marginBottom: '0.25rem' }}>
          <span style={{ fontWeight: 600, color: '#0891b2', fontSize: '0.6875rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Domain </span>
          <span style={{ color: '#374151' }}>{dLabels.join(', ')}</span>
        </div>
      )}
      {jLabels.length > 0 && (
        <div style={{ marginBottom: '0.25rem' }}>
          <span style={{ fontWeight: 600, color: '#059669', fontSize: '0.6875rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Jurisdiction </span>
          <span style={{ color: '#374151' }}>{jLabels.join(', ')}</span>
        </div>
      )}
      {data.taggedText && (
        <div style={{ color: '#6b7280', fontSize: '0.75rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginTop: '0.125rem' }}>
          {data.taggedText as string}
        </div>
      )}
    </div>
  )
}
