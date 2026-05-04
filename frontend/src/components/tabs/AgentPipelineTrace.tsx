/**
 * AgentPipelineTrace — collapsible visualization of one
 * ``corpus_search_agent`` response.
 *
 * Renders every stage in order:
 *   ① Query
 *   ② Expected (only when supplied — eval mode)
 *   ③ Parser (classify_query)
 *   ④ Term partition (REQUIRED / BOOSTED / DROPPED)
 *   ⑤ Cascade pool
 *   ⑥ Router decision
 *   ⑦ Strategy execution (BM25 / vector arms + timing)
 *   ⑧a Themes (Strategy b)
 *   ⑨ Validated citations (Strategy c/d)
 *   ⑩ Fail-fast (Strategy e)
 *   ⑪ LLM answer (Strategy c/d)
 *   ⑫ Assembled chunks (final output)
 *   ⑬ Judge verdict (eval only)
 *
 * Used by EvalTab (pre-fetched) and TestTab (live invocation).
 */
import { useState } from 'react'

// ── Shared types — surface of corpus_search_agent response ─────────

export interface AgentChunk {
  id?: string
  text: string
  document_id?: string
  document_name?: string | null
  page_number?: number | null
  paragraph_index?: number | null
  source_type?: string
  similarity?: number
  rerank_score?: number
  confidence_label?: string
  retrieval_arms?: string[]
  authority_level?: string | null
  payer?: string | null
  state?: string | null
  // Where in the doc this chunk lives — for context-quality diagnostics.
  section_path?: string | null
  chapter_path?: string | null
  summary?: string | null
}

export interface AgentResponse {
  chunks?: AgentChunk[]
  confidence?: string
  strategy_used?: string
  query_profile?: {
    query_type?: string
    coverage?: number
    tag_matches?: string[]
    literal_anchors?: string[]
    untagged_meaningful_tokens?: string[]
    raw_query?: string
  }
  term_partition?: {
    required?: { term: string; kind?: string; code?: string; selectivity?: number }[]
    boosted?: { term: string; kind?: string; code?: string; selectivity?: number }[]
    dropped?: { term: string; kind?: string; code?: string; selectivity?: number }[]
  }
  candidate_pool?: {
    size?: number
    cascade_level?: string
    cascade_steps?: { level: string; result: number | string }[]
    intersect_codes?: string[]
    used_for_search?: boolean
    effective_pool_size?: number
    narrowed_via_bootstrap?: boolean
  }
  routing?: {
    strategy?: string
    executed_strategy?: string
    fallback?: string | null
    query_class?: string
    method?: string
    scores?: Record<string, number>
    self_assessments?: Record<string, { est_recall: number; static_recall?: number; reason: string }>
    withdrawn?: string[]
    prefs_resolved?: Record<string, unknown>
    priors_version?: string
    fail_fast_reason?: string | null
    score_breakdown?: Record<string, ScoreBreakdown>
  }
  strategies_tried?: {
    strategy: string
    query_used: string
    succeeded: boolean
    note: string
    n_chunks: number
    top_rerank: number
    elapsed_ms: number
    arms?: {
      bm25_pool_hits?: number
      vector_pool_hits?: number
      result_breakdown?: { bm25_only: number; vector_only: number; both: number }
      timing_ms?: { embed: number; bm25: number; vector: number; rerank: number }
    }
    scoring_trace?: ScoringTraceItem[]
  }[]
  themes?: { label: string; full_code: string; n_docs: number; n_chunks_seen: number; top_rerank: number }[]
  theme_diagnostic?: {
    n_themes: number
    n_wide_chunks: number
    dominant_theme_share: number
    narrower_than_expected: boolean
  }
  validated_citations?: {
    candidate?: { document_title?: string; page?: number; section?: string; url?: string; quote?: string }
    status?: string
    document_id?: string
    document_display_name?: string
    matched_page?: number
    discovered_source_url?: string
    notes?: string
  }[]
  fail_fast?: {
    reason?: string
    response_mode?: string
    user_message?: string
    options?: string[]
  }
  llm_answer?: string | null
  improvement_hint?: { suggestion?: string; would_reframing_help?: boolean } | null
  telemetry?: Record<string, unknown>
  queries_per_strategy?: {
    hybrid?: string
    phrase_strict?: string
    vector_broad?: string
  } | null
}

export interface ScoringTraceItem {
  rank?: number
  chunk_id?: string
  document_name?: string
  page_number?: number | null
  retrieval_arms?: string[]
  authority_level?: string | null
  rerank_signals?: {
    sim_raw?: number
    sim_weighted?: number
    authority_raw?: number
    authority_weighted?: number
    length_raw?: number
    length_weighted?: number
    jpd_raw?: number
    jpd_weighted?: number
    jpd_tags?: string[]
    tag_coverage_raw?: number
    tag_coverage_weighted?: number
    tag_coverage_present?: string[]
    tag_coverage_missing?: string[]
    total_raw?: number
    rerank_score?: number
    max_weight?: number
  }
}

export interface ScoreBreakdown {
  withdrawn?: boolean
  withdraw_reason?: string
  accuracy?: { prior: number; weight: number; contrib: number }
  recall?: { est: number; static: number; weight: number; contrib: number }
  speed?: { prior: number; weight: number; contrib: number }
  shape?: { match: number; weight: number; contrib: number }
  cost_per_call?: number
  total: number
}

export interface JudgeBlock {
  verdict: string | null
  score: number | null
  reasoning: string | null
  model: string | null
}


// ── Component ─────────────────────────────────────────────────────────

export interface AgentPipelineTraceProps {
  /** The full agent response. */
  response: AgentResponse
  /** Optional: the query (defaults to response.query_profile.raw_query). */
  query?: string
  /** Optional: the labeled expected outcome (eval mode). */
  expected?: Record<string, unknown> | null
  /** Optional: judge verdict + reasoning (eval mode). */
  judge?: JudgeBlock | null
}

const VERDICT_COLOR: Record<string, string> = {
  correct: 'verdict-correct',
  partial: 'verdict-partial',
  wrong: 'verdict-wrong',
  unable_to_verify: 'verdict-unable',
}

export function AgentPipelineTrace({ response, query, expected, judge }: AgentPipelineTraceProps) {
  const profile = response.query_profile
  const partition = response.term_partition
  const pool = response.candidate_pool
  const routing = response.routing
  const strategies = response.strategies_tried || []
  const themes = response.themes || []
  const themeDiag = response.theme_diagnostic
  const citations = response.validated_citations || []
  const failFast = response.fail_fast

  const queryText = query || profile?.raw_query || ''
  const stratExec = response.strategy_used || routing?.executed_strategy || routing?.strategy

  // Per-arm chunk distribution — used by both Rerank + Assembly views.
  const chunks = response.chunks || []
  const armCounts: Record<string, number> = {}
  const tierCounts: Record<string, number> = {}
  for (const c of chunks) {
    const arms = (c.retrieval_arms || []).join('+') || 'unknown'
    armCounts[arms] = (armCounts[arms] || 0) + 1
    const tier = c.authority_level || c.source_type || 'untagged'
    tierCounts[tier] = (tierCounts[tier] || 0) + 1
  }

  // Cleanup stats — derive from the parser's outputs vs the raw query.
  const rawTokens = (queryText.match(/\b\w+\b/g) || [])
  const tokens = profile
    ? [
        ...(profile.literal_anchors || []),
        ...((profile.tag_matches || []).flatMap((t) => t.split(/[:.]/).slice(1))),
        ...(profile.untagged_meaningful_tokens || []),
      ]
    : []
  const meaningfulTokens = new Set(tokens.map((t) => t.toLowerCase()))
  const droppedTokens = rawTokens.filter((t) => !meaningfulTokens.has(t.toLowerCase()) && t.length > 2)

  // Required anchors — combine literals + REQUIRED partition entries.
  const requiredAnchors = [
    ...(profile?.literal_anchors || []),
    ...((partition?.required || []).map((r) => r.term)),
  ]

  return (
    <div className="agent-trace">
      {/* ── Pipeline flow indicator (orientation) ── */}
      <PipelineFlow
        stages={[
          { id: 'query', label: 'Query', active: true, ok: true },
          { id: 'cleanup', label: 'Cleanup', active: !!profile, ok: !!profile, sub: profile ? `-${droppedTokens.length} noise` : '' },
          { id: 'rewrite', label: 'Rewrite', active: !!response.queries_per_strategy, ok: !!response.queries_per_strategy },
          { id: 'requirements', label: 'Anchors', active: !!profile, ok: requiredAnchors.length > 0, sub: requiredAnchors.length ? `${requiredAnchors.length} required` : '' },
          { id: 'selfassess', label: 'Self-assess', active: !!routing?.self_assessments, ok: !!routing?.self_assessments },
          { id: 'parser', label: 'Parser', active: !!profile, ok: !!profile, sub: profile ? `${profile.query_type}` : '' },
          { id: 'router', label: 'Router', active: !!routing, ok: !!routing, sub: routing ? `→ ${stratExec}` : '' },
          { id: 'strategy', label: 'Strategy', active: strategies.length > 0 || !!failFast, ok: strategies.some((s) => s.succeeded), sub: strategies.length ? `${strategies.length} ran` : (failFast ? 'fail-fast' : '') },
          { id: 'rerank', label: 'Rerank', active: chunks.length > 0, ok: chunks.length > 0, sub: chunks.length ? `top ${chunks[0]?.rerank_score?.toFixed(2)}` : '' },
          { id: 'assemble', label: 'Assemble', active: chunks.length > 0, ok: chunks.length > 0, sub: chunks.length ? `${chunks.length}/${(response.chunks || []).length}` : '' },
          { id: 'verdict', label: 'Verdict', active: !!judge?.verdict, ok: judge?.verdict === 'correct', sub: judge?.verdict || '' },
        ]}
      />

      {/* ── 1. Query ── */}
      <CollapsibleSection title="① Query (raw)" defaultOpen>
        <div className="block">{queryText}</div>
      </CollapsibleSection>

      {/* ── eval-only: Expected ── */}
      {expected && (
        <CollapsibleSection title="🎯 Expected (label)">
          <pre className="pre">{JSON.stringify(expected, null, 2)}</pre>
        </CollapsibleSection>
      )}

      {/* ── 2. Cleanup — what got stripped ── */}
      {profile && (
        <CollapsibleSection
          title="② Cleanup — non-required words removed"
          subtitle={`${rawTokens.length} tokens in → ${meaningfulTokens.size} kept · ${droppedTokens.length} dropped (noise/stopwords)`}
        >
          <div className="block kvgrid">
            <div className="full">
              <strong>All raw tokens:</strong>{' '}
              {rawTokens.map((t, i) => {
                const dropped = droppedTokens.includes(t)
                return (
                  <span key={i} className={`tag-pill ${dropped ? 'dim-pill' : 'ok-pill'}`} style={{ textDecoration: dropped ? 'line-through' : 'none' }}>
                    {t}
                  </span>
                )
              })}
            </div>
            <div className="full">
              <strong>Dropped:</strong>{' '}
              {droppedTokens.length === 0 && <span className="dim-text">none</span>}
              {droppedTokens.map((t, i) => (
                <span key={i} className="tag-pill dim-pill">{t}</span>
              ))}
            </div>
            <div className="full">
              <strong>Kept (meaningful):</strong>{' '}
              {Array.from(meaningfulTokens).map((t, i) => (
                <span key={i} className="tag-pill ok-pill">{t}</span>
              ))}
            </div>
          </div>
        </CollapsibleSection>
      )}

      {/* ── 3. Query Rewrite — per-strategy variants ── */}
      {response.queries_per_strategy && (
        <CollapsibleSection
          title="③ Query Rewrite — per-strategy variants"
          subtitle="what each strategy actually queries with"
          defaultOpen
        >
          <div className="block">
            <table className="trace-table">
              <thead><tr><th>strategy</th><th>rewritten query</th></tr></thead>
              <tbody>
                <tr>
                  <td className="mono">hybrid (a)</td>
                  <td><code>{response.queries_per_strategy.hybrid || '—'}</code></td>
                </tr>
                <tr>
                  <td className="mono">phrase_strict</td>
                  <td><code>{response.queries_per_strategy.phrase_strict || '—'}</code></td>
                </tr>
                <tr>
                  <td className="mono">vector_broad</td>
                  <td><code>{response.queries_per_strategy.vector_broad || '—'}</code></td>
                </tr>
              </tbody>
            </table>
          </div>
        </CollapsibleSection>
      )}

      {/* ── 4. Required anchors / key requirements ── */}
      {(profile || partition) && (
        <CollapsibleSection
          title="④ Key Requirements — must-haves"
          subtitle={`${requiredAnchors.length} anchors · ${(partition?.required || []).length} required tags · ${(partition?.boosted || []).length} boosted`}
          defaultOpen={requiredAnchors.length > 0}
        >
          <div className="block">
            {(profile?.literal_anchors || []).length > 0 && (
              <div className="kv-row">
                <strong>Literal anchors (hard-must):</strong>
                <div>
                  {(profile?.literal_anchors || []).map((t, i) => (
                    <span key={i} className="tag-pill ok-pill" style={{ fontFamily: 'ui-monospace, monospace' }}>{t}</span>
                  ))}
                </div>
              </div>
            )}
            {partition && (
              <>
                <PartitionBucket label="REQUIRED (sel ≥ 0.85)" items={partition.required || []} kind="ok" />
                <PartitionBucket label="BOOSTED (sel 0.40–0.85)" items={partition.boosted || []} kind="warn" />
                <PartitionBucket label="DROPPED (sel < 0.40)" items={partition.dropped || []} kind="dim" />
              </>
            )}
          </div>
        </CollapsibleSection>
      )}

      {/* ── 5. Self-assessment / internal confidence ── */}
      {routing?.self_assessments && (
        <CollapsibleSection
          title="⑤ Self-Assessment — internal confidence"
          subtitle={`${routing.withdrawn?.length || 0} withdrawn`}
          defaultOpen
        >
          <div className="block">
            <table className="trace-table">
              <thead>
                <tr><th>strategy</th><th>est_recall</th><th>static</th><th>delta</th><th>reason / penalty</th></tr>
              </thead>
              <tbody>
                {Object.entries(routing.self_assessments).map(([s, a]) => {
                  const sa = a as { est_recall?: number; static_recall?: number; reason?: string }
                  const delta = (sa?.est_recall ?? 0) - (sa?.static_recall ?? 0)
                  const withdrawn = routing.withdrawn?.includes(s)
                  return (
                    <tr key={s} style={withdrawn ? { opacity: 0.5 } : {}}>
                      <td className="mono">
                        {s}
                        {withdrawn && <span className="tag-pill warn-pill" style={{ marginLeft: 4 }}>withdrew</span>}
                      </td>
                      <td className="mono">{sa?.est_recall?.toFixed?.(2) ?? '—'}</td>
                      <td className="mono dim-text">{sa?.static_recall?.toFixed?.(2) ?? '—'}</td>
                      <td className={`mono ${delta < -0.1 ? 'warn-text' : delta > 0.1 ? 'ok-text' : 'dim-text'}`}>
                        {delta > 0 ? '+' : ''}{delta.toFixed(2)}
                      </td>
                      <td>{sa?.reason}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </CollapsibleSection>
      )}

      {/* ── 6. Parser summary ── */}
      {profile && (
        <CollapsibleSection
          title="⑥ Parser — classify_query"
          subtitle={`type=${profile.query_type} · coverage=${profile.coverage}`}
        >
          <div className="block kvgrid">
            <div><strong>query_type:</strong> {profile.query_type}</div>
            <div><strong>coverage:</strong> {profile.coverage}</div>
            <div className="full">
              <strong>tag_matches:</strong>{' '}
              {(profile.tag_matches || []).map((t, i) => (
                <span key={i} className="tag-pill">{t}</span>
              ))}
              {(!profile.tag_matches || profile.tag_matches.length === 0) && (
                <span className="dim-text">none</span>
              )}
            </div>
            <div className="full">
              <strong>literal_anchors:</strong>{' '}
              {(profile.literal_anchors || []).map((t, i) => (
                <span key={i} className="tag-pill">{t}</span>
              ))}
              {(!profile.literal_anchors || profile.literal_anchors.length === 0) && (
                <span className="dim-text">none</span>
              )}
            </div>
            <div className="full">
              <strong>untagged_meaningful:</strong>{' '}
              {(profile.untagged_meaningful_tokens || []).map((t, i) => (
                <span key={i} className="tag-pill">{t}</span>
              ))}
              {(!profile.untagged_meaningful_tokens || profile.untagged_meaningful_tokens.length === 0) && (
                <span className="dim-text">none</span>
              )}
            </div>
          </div>
        </CollapsibleSection>
      )}

      {/* ── Cascade pool (informational, between Parser and Router) ── */}
      {pool && (Object.keys(pool).length > 0) && (
        <CollapsibleSection
          title="⑥a Cascade Pool"
          subtitle={`${pool.cascade_level} · ${pool.size ?? 0} docs${pool.narrowed_via_bootstrap ? ' (bootstrap narrowed)' : ''}`}
        >
          <div className="block kvgrid">
            <div><strong>cascade_level:</strong> {pool.cascade_level}</div>
            <div><strong>pool size:</strong> {pool.size}</div>
            <div><strong>used_for_search:</strong> {String(pool.used_for_search)}</div>
            <div><strong>effective:</strong> {pool.effective_pool_size}</div>
            <div className="full">
              <strong>intersect_codes:</strong>{' '}
              {(pool.intersect_codes || []).map((c, i) => (
                <span key={i} className="tag-pill">{c}</span>
              ))}
            </div>
            {pool.cascade_steps && pool.cascade_steps.length > 0 && (
              <div className="full">
                <strong>cascade_steps:</strong>
                <ul className="cascade-steps">
                  {pool.cascade_steps.map((s, i) => (
                    <li key={i}>
                      <span className="step-level">{s.level}</span> →{' '}
                      <span className="step-result">{String(s.result)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </CollapsibleSection>
      )}

      {/* ── Router decision ── */}
      {routing && (
        <CollapsibleSection
          title="⑦ Router — strategy choice"
          subtitle={`picked ${routing.strategy} → executed ${stratExec}${routing.fallback ? ' (fb ' + routing.fallback + ')' : ''}`}
          defaultOpen
        >
          <div className="block kvgrid">
            <div>
              <strong>Picked:</strong> {routing.strategy} → executed {stratExec}
              {routing.fallback && (<> (fallback: {routing.fallback})</>)}
            </div>
            <div><strong>Confidence:</strong> {response.confidence}</div>
            <div><strong>Class:</strong> {routing.query_class}</div>
            <div><strong>Method:</strong> {routing.method}</div>
            <div><strong>Priors:</strong> {routing.priors_version}</div>
            {routing.scores && (
              <div className="full">
                <strong>Scores:</strong>{' '}
                {Object.entries(routing.scores).map(([k, v]) => (
                  <span
                    key={k}
                    className={`score-pill ${k === routing.strategy ? 'score-pill-winner' : ''}`}
                  >
                    {k}: {Number(v).toFixed(2)}
                  </span>
                ))}
              </div>
            )}
            {routing.score_breakdown && (
              <div className="full">
                <strong>Score breakdown — how each total was computed:</strong>
                <table className="trace-table breakdown-table">
                  <thead>
                    <tr>
                      <th>strat</th>
                      <th colSpan={2}>accuracy</th>
                      <th colSpan={2}>recall</th>
                      <th colSpan={2}>speed</th>
                      <th colSpan={2}>shape</th>
                      <th>total</th>
                    </tr>
                    <tr className="breakdown-subhead">
                      <th></th>
                      <th>prior × need</th><th>=</th>
                      <th>est × demand</th><th>=</th>
                      <th>prior × wt</th><th>=</th>
                      <th>match × wt</th><th>=</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(routing.score_breakdown).map(([s, br]) => {
                      const winner = s === routing.strategy
                      if (br.withdrawn) {
                        return (
                          <tr key={s} className="breakdown-row dim-text">
                            <td className="mono">{s}</td>
                            <td colSpan={9}>
                              <span className="tag-pill warn-pill">withdrew</span>
                              {' '}{br.withdraw_reason}
                            </td>
                          </tr>
                        )
                      }
                      return (
                        <tr key={s} className={`breakdown-row ${winner ? 'breakdown-winner' : ''}`}>
                          <td className="mono">{winner && '★ '}{s}</td>
                          <td className="mono dim-text">{br.accuracy?.prior.toFixed(2)} × {br.accuracy?.weight.toFixed(2)}</td>
                          <td className="mono">{br.accuracy?.contrib.toFixed(3)}</td>
                          <td className="mono dim-text">
                            {br.recall?.est.toFixed(2)} × {br.recall?.weight.toFixed(2)}
                            {br.recall && br.recall.est !== br.recall.static && (
                              <span className="dim-text"> (static {br.recall.static.toFixed(2)})</span>
                            )}
                          </td>
                          <td className="mono">{br.recall?.contrib.toFixed(3)}</td>
                          <td className="mono dim-text">{br.speed?.prior.toFixed(2)} × {br.speed?.weight.toFixed(2)}</td>
                          <td className="mono">{br.speed?.contrib.toFixed(3)}</td>
                          <td className="mono dim-text">{br.shape?.match.toFixed(2)} × {br.shape?.weight.toFixed(2)}</td>
                          <td className="mono">{br.shape?.contrib.toFixed(3)}</td>
                          <td className="mono"><strong>{br.total.toFixed(3)}</strong></td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
            {routing.self_assessments && (
              <div className="full">
                <strong>Self-assessments:</strong>
                <table className="trace-table">
                  <thead>
                    <tr><th>strat</th><th>est_recall</th><th>static</th><th>reason</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(routing.self_assessments).map(([s, a]) => {
                      const sa = a as { est_recall?: number; static_recall?: number; reason?: string }
                      return (
                        <tr key={s}>
                          <td className="mono">{s}</td>
                          <td className="mono">{sa?.est_recall?.toFixed?.(2) ?? '—'}</td>
                          <td className="mono dim-text">{sa?.static_recall?.toFixed?.(2) ?? '—'}</td>
                          <td>{sa?.reason}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
            {routing.withdrawn && routing.withdrawn.length > 0 && (
              <div className="full">
                <strong>Withdrawn:</strong>{' '}
                {routing.withdrawn.map((w, i) => (
                  <span key={i} className="tag-pill warn-pill">{w}</span>
                ))}
              </div>
            )}
          </div>
        </CollapsibleSection>
      )}

      {/* ── 7. Strategy execution ── */}
      {strategies.length > 0 && (
        <CollapsibleSection
          title="⑧ Strategy Execution"
          subtitle={`${strategies.length} ran`}
          defaultOpen
        >
          {strategies.map((s, i) => (
            <div key={i} className="strategy-card">
              <div className="strategy-header">
                <span className={`strategy-flag ${s.succeeded ? 'ok' : 'warn'}`}>
                  {s.succeeded ? '✓' : '✗'}
                </span>
                <span className="strategy-name">{s.strategy}</span>
                <span className="strategy-time">{s.elapsed_ms}ms</span>
                <span className="strategy-rerank">top rerank {Number(s.top_rerank).toFixed(2)}</span>
                <span className="strategy-chunks">{s.n_chunks} chunks</span>
              </div>
              <div className="strategy-query">
                <strong>query:</strong> <code>{s.query_used}</code>
              </div>
              <div className="strategy-note">
                <strong>note:</strong> {s.note}
              </div>
              {s.arms && (
                <div className="strategy-arms">
                  <strong>arms:</strong>
                  {' '}bm25={s.arms.bm25_pool_hits ?? 0}
                  {' · '}vector={s.arms.vector_pool_hits ?? 0}
                  {s.arms.timing_ms && (
                    <>
                      {' · '}embed {s.arms.timing_ms.embed}ms
                      {' / '}bm25 {s.arms.timing_ms.bm25}ms
                      {' / '}vec {s.arms.timing_ms.vector}ms
                      {' / '}rerank {s.arms.timing_ms.rerank}ms
                    </>
                  )}
                  {s.arms.result_breakdown && (
                    <>
                      {' · '}result_arms: bm25_only={s.arms.result_breakdown.bm25_only}
                      {' · '}vector_only={s.arms.result_breakdown.vector_only}
                      {' · '}both={s.arms.result_breakdown.both}
                    </>
                  )}
                </div>
              )}
            </div>
          ))}
        </CollapsibleSection>
      )}

      {/* ── 8. Strategy (b) themes ── */}
      {themes.length > 0 && (
        <CollapsibleSection
          title="⑧a Themes (Strategy b)"
          subtitle={
            themeDiag
              ? `${themeDiag.n_themes} themes · ${themeDiag.n_wide_chunks} wide chunks · dominance ${themeDiag.dominant_theme_share}${themeDiag.narrower_than_expected ? ' (narrower than expected)' : ''}`
              : `${themes.length} themes`
          }
        >
          <table className="trace-table">
            <thead>
              <tr><th>theme</th><th>full_code</th><th>n_docs</th><th>chunks_seen</th><th>top_rerank</th></tr>
            </thead>
            <tbody>
              {themes.map((th, i) => (
                <tr key={i}>
                  <td>{th.label}</td>
                  <td className="mono dim-text">{th.full_code}</td>
                  <td className="mono">{th.n_docs}</td>
                  <td className="mono">{th.n_chunks_seen}</td>
                  <td className="mono">{Number(th.top_rerank).toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </CollapsibleSection>
      )}

      {/* ── 9. Validated citations ── */}
      {citations.length > 0 && (
        <CollapsibleSection
          title="⑧b Validated Citations (Strategy c/d)"
          subtitle={`${citations.length} citations`}
        >
          {citations.map((cite, i) => {
            const cand = cite.candidate || {}
            return (
              <div key={i} className="citation-card">
                <div className="citation-header">
                  <span className={`status-pill status-${(cite.status || 'unknown').replace(/_/g, '-')}`}>
                    {cite.status}
                  </span>
                  <strong>{cand.document_title || cite.discovered_source_url}</strong>
                  {cand.page && <> p.{cand.page}</>}
                  {cand.section && <> · {cand.section}</>}
                </div>
                {cand.url && (
                  <div className="citation-url">
                    <a href={cand.url} target="_blank" rel="noopener noreferrer">{cand.url}</a>
                  </div>
                )}
                {cand.quote && (
                  <div className="citation-quote">
                    <em>LLM quote:</em> "{cand.quote}"
                  </div>
                )}
                {cite.notes && <div className="citation-notes">{cite.notes}</div>}
              </div>
            )
          })}
        </CollapsibleSection>
      )}

      {/* ── 10. Fail-fast ── */}
      {failFast && (
        <CollapsibleSection
          title="⑧c Fail-Fast (Strategy e)"
          subtitle={`${failFast.reason} — ${failFast.response_mode}`}
          defaultOpen
        >
          <div className="block">
            <div><strong>reason:</strong> {failFast.reason}</div>
            <div><strong>response_mode:</strong> {failFast.response_mode}</div>
            {failFast.user_message && (
              <div className="reasoning"><strong>user_message:</strong> {failFast.user_message}</div>
            )}
            {failFast.options && failFast.options.length > 0 && (
              <details>
                <summary>Available options ({failFast.options.length})</summary>
                <pre className="pre">{(failFast.options || []).join(', ')}</pre>
              </details>
            )}
          </div>
        </CollapsibleSection>
      )}

      {/* ── 11. LLM answer ── */}
      {response.llm_answer && (
        <CollapsibleSection title="⑧d LLM Answer (Strategy c/d)" defaultOpen>
          <div className="block">{response.llm_answer}</div>
        </CollapsibleSection>
      )}

      {/* ── 12. Rerank ── */}
      {chunks.length > 0 && (
        <CollapsibleSection
          title="⑨ Reranking"
          subtitle={`${chunks.length} chunks · top ${chunks[0]?.rerank_score?.toFixed?.(2) ?? '—'} · arm split: ${Object.entries(armCounts).map(([a, n]) => `${a}=${n}`).join(' / ')}`}
          defaultOpen
        >
          <RerankingPanel
            chunks={chunks}
            scoringTrace={(response.strategies_tried || []).flatMap((s) => s.scoring_trace || [])}
          />
        </CollapsibleSection>
      )}

      {/* ── 13. Assembly ── */}
      {chunks.length > 0 && (
        <AssemblyDiagnostics chunks={chunks} armCounts={armCounts} tierCounts={tierCounts} />
      )}

      {/* ── 14. Final chunks (what was sent) ── */}
      <CollapsibleSection
        title={`⑩a Sent to Caller (${response.chunks?.length || 0} chunks)`}
        subtitle="final output — body text + section context"
        defaultOpen
      >
        {(response.chunks || []).map((c, i) => {
          const len = (c.text || '').length
          const thin = len < 300
          return (
            <div key={i} className="chunk-card">
              <div className="chunk-header">
                <span className="mono dim-text">#{i + 1}</span>{' '}
                <strong>{c.document_name || 'Unknown doc'}</strong>
                {c.page_number != null && <> · p.{c.page_number}</>}
                {c.paragraph_index != null && <> · ¶{c.paragraph_index}</>}
                {c.rerank_score != null && (
                  <span className="rerank">rerank {Number(c.rerank_score).toFixed(2)}</span>
                )}
                {(c.retrieval_arms || []).map((a, j) => (
                  <span key={j} className={`tag-pill ${a === 'bm25' ? 'ok-pill' : a === 'vector' ? 'warn-pill' : ''}`}>
                    {a}
                  </span>
                ))}
                <span className={`tag-pill ${thin ? 'warn-pill' : 'dim-pill'}`}>
                  {len} chars{thin ? ' · thin' : ''}
                </span>
              </div>
              {(c.chapter_path || c.section_path) && (
                <div className="chunk-context">
                  {c.chapter_path && <span className="context-pill">📖 {c.chapter_path}</span>}
                  {c.section_path && <span className="context-pill">§ {c.section_path}</span>}
                </div>
              )}
              {c.summary && (
                <div className="chunk-summary">
                  <em>summary:</em> {c.summary}
                </div>
              )}
              <div className="chunk-text">{c.text}</div>
            </div>
          )
        })}
        {(!response.chunks || response.chunks.length === 0) && (
          <div className="eval-empty">No chunks were returned.</div>
        )}
      </CollapsibleSection>

      {/* ── 15. Neighborhood (the actual fix for thin chunks) ── */}
      <CollapsibleSection
        title="⑩b Neighborhood Expansion"
        subtitle="not yet wired — this is the lever for thin BM25 chunks"
      >
        <div className="block">
          <div style={{ marginBottom: 8 }}>
            <strong>What it would do:</strong>
          </div>
          <ul style={{ margin: '4px 0 8px 18px', padding: 0 }}>
            <li>For each top chunk, fetch the chunks at <code>paragraph_index ± 1</code> in the same doc.</li>
            <li>Inline them as preceding/following context so the LLM sees the rule + the exception, the question + the answer, the section header + the bullet.</li>
            <li>Adds ~200–600 chars of glue per top chunk; the BM25 ranking is unchanged.</li>
          </ul>
          <div style={{ marginBottom: 8 }}>
            <strong>Why it matters:</strong> the Assembly section above
            shows whether average chunk length is THIN / OK / GOOD. When
            it's THIN, single-paragraph BM25 hits often lack the
            surrounding context needed to answer. This section, once
            wired, shows exactly what neighbors were fetched per chunk.
          </div>
          <div className="dim-text">
            Wiring path: agent's ``_strategy_a`` already calls
            ``corpus_search`` which has access to ``paragraph_index``.
            One SQL per top-k chunk fetches neighbors. Cost is small
            (sub-100ms total) since we're picking by ``document_id ANY``
            + ``paragraph_index BETWEEN ...``.
          </div>
        </div>
      </CollapsibleSection>

      {/* ── 13. Judge (eval only) ── */}
      {judge && (
        <CollapsibleSection
          title="⑪ Judge Verdict"
          subtitle={`${judge.verdict} (score ${judge.score?.toFixed?.(2) ?? '—'})`}
          defaultOpen
        >
          <div className="block">
            <div>
              <strong>Verdict:</strong>{' '}
              <span className={`verdict ${VERDICT_COLOR[judge.verdict || 'unable_to_verify']}`}>
                {judge.verdict}
              </span>
            </div>
            {judge.score != null && (
              <div><strong>Score:</strong> {judge.score.toFixed(2)}</div>
            )}
            {judge.model && <div><strong>Model:</strong> {judge.model}</div>}
            {judge.reasoning && (
              <div className="reasoning">
                <strong>Reasoning:</strong> {judge.reasoning}
              </div>
            )}
          </div>
        </CollapsibleSection>
      )}
    </div>
  )
}


// ── Reranking panel — per-chunk signal breakdown ─────────────────────

function RerankingPanel({
  chunks,
  scoringTrace,
}: {
  chunks: AgentChunk[]
  scoringTrace: ScoringTraceItem[]
}) {
  // Index trace items by chunk_id so we can match them to chunk rows.
  const traceByChunkId = new Map<string, ScoringTraceItem>()
  for (const t of scoringTrace) {
    if (t.chunk_id) traceByChunkId.set(t.chunk_id, t)
  }
  const haveSignals = scoringTrace.some((t) => !!t.rerank_signals)

  if (!haveSignals) {
    // Fall back to the simple table if no signal trace was returned
    // (e.g. legacy responses or strategies that don't run the reranker).
    return (
      <table className="trace-table">
        <thead>
          <tr><th>#</th><th>arms</th><th>rerank</th><th>sim</th><th>auth</th><th>doc</th><th>page</th></tr>
        </thead>
        <tbody>
          {chunks.map((c, i) => (
            <tr key={i}>
              <td className="mono">{i + 1}</td>
              <td>
                {(c.retrieval_arms || []).map((a, j) => (
                  <span key={j} className={`tag-pill ${a === 'bm25' ? 'ok-pill' : a === 'vector' ? 'warn-pill' : 'dim-pill'}`}>{a}</span>
                ))}
              </td>
              <td className="mono">{c.rerank_score?.toFixed?.(3) ?? '—'}</td>
              <td className="mono dim-text">{c.similarity?.toFixed?.(3) ?? '—'}</td>
              <td className="mono dim-text">{c.authority_level || '—'}</td>
              <td className="dim-text" style={{ maxWidth: 240, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {c.document_name || '—'}
              </td>
              <td className="mono">{c.page_number ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    )
  }

  return (
    <>
      <div className="dim-text" style={{ marginBottom: 6, fontSize: 11 }}>
        Per-chunk score = (sim × W_sim) + (auth × W_auth) + (length × W_len) + (jpd × W_jpd) + (tag_coverage × W_tag) ÷ MAX_WEIGHT.
        The <strong>tag_coverage</strong> column shows what fraction of the REQUIRED tag phrases each chunk's body actually contains —
        chunks missing tags drop here.
      </div>
      <table className="trace-table breakdown-table">
        <thead>
          <tr>
            <th>#</th>
            <th>arms</th>
            <th>sim</th>
            <th>auth</th>
            <th>len</th>
            <th>jpd</th>
            <th>tag_cov</th>
            <th>missing tags</th>
            <th>total</th>
            <th>doc · p.</th>
          </tr>
        </thead>
        <tbody>
          {chunks.map((c, i) => {
            const t = c.id ? traceByChunkId.get(c.id) : undefined
            const sig = t?.rerank_signals
            const missing = sig?.tag_coverage_missing || []
            return (
              <tr key={i} className={i === 0 ? 'breakdown-winner' : ''}>
                <td className="mono">{i === 0 && '★ '}{i + 1}</td>
                <td>
                  {(c.retrieval_arms || []).map((a, j) => (
                    <span key={j} className={`tag-pill ${a === 'bm25' ? 'ok-pill' : a === 'vector' ? 'warn-pill' : 'dim-pill'}`}>{a}</span>
                  ))}
                </td>
                <td className="mono dim-text">
                  {sig ? (
                    <span title={`raw ${sig.sim_raw?.toFixed?.(2)} → weighted ${sig.sim_weighted?.toFixed?.(3)}`}>
                      {sig.sim_raw?.toFixed?.(2)}
                    </span>
                  ) : '—'}
                </td>
                <td className="mono dim-text">{sig?.authority_raw?.toFixed?.(2) ?? '—'}</td>
                <td className="mono dim-text">{sig?.length_raw?.toFixed?.(2) ?? '—'}</td>
                <td className="mono dim-text">{sig?.jpd_raw?.toFixed?.(2) ?? '—'}</td>
                <td className="mono">
                  {sig?.tag_coverage_raw !== undefined ? (
                    <span className={`tag-pill ${sig.tag_coverage_raw === 1 ? 'ok-pill' : sig.tag_coverage_raw >= 0.5 ? 'warn-pill' : 'dim-pill'}`}>
                      {sig.tag_coverage_raw.toFixed(2)}
                    </span>
                  ) : '—'}
                </td>
                <td>
                  {missing.length === 0 ? (
                    <span className="dim-text">—</span>
                  ) : (
                    missing.map((m, j) => (
                      <span key={j} className="tag-pill warn-pill" style={{ fontSize: 10 }}>
                        ✗ {m}
                      </span>
                    ))
                  )}
                </td>
                <td className="mono">
                  <strong>{c.rerank_score?.toFixed?.(3) ?? '—'}</strong>
                </td>
                <td className="dim-text" style={{ maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {c.document_name || '—'} · p.{c.page_number ?? '—'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </>
  )
}


// ── Assembly diagnostics ─────────────────────────────────────────────

function AssemblyDiagnostics({
  chunks,
  armCounts,
  tierCounts,
}: {
  chunks: AgentChunk[]
  armCounts: Record<string, number>
  tierCounts: Record<string, number>
}) {
  const lengths = chunks.map((c) => (c.text || '').length)
  const totalChars = lengths.reduce((a, b) => a + b, 0)
  // Crude token estimate — 1 token ≈ 4 chars for English prose.
  const estTokens = Math.round(totalChars / 4)
  const minLen = lengths.length ? Math.min(...lengths) : 0
  const maxLen = lengths.length ? Math.max(...lengths) : 0
  const avgLen = lengths.length ? Math.round(totalChars / lengths.length) : 0
  const thinCount = lengths.filter((l) => l < 300).length
  const sectionCovered = chunks.filter((c) => c.section_path || c.chapter_path).length
  const summaryCovered = chunks.filter((c) => c.summary).length

  // Adequacy heuristic — surface as a single line so user can tell at a
  // glance whether the LLM has enough to work with.
  const adequacy =
    chunks.length === 0
      ? { label: 'NONE', color: 'bad', note: 'no chunks were returned' }
      : avgLen < 250
      ? { label: 'THIN', color: 'bad', note: 'avg chunk <250 chars — likely missing context (single-paragraph hits without surrounding text)' }
      : avgLen < 500
      ? { label: 'OK', color: 'warn', note: 'avg chunk 250–500 chars — usable but could benefit from neighborhood expansion' }
      : { label: 'GOOD', color: 'ok', note: `avg ${avgLen} chars per chunk — substantial body context` }

  return (
    <CollapsibleSection
      title="⑩ Assembly"
      subtitle={`${chunks.length} chunks · ${totalChars.toLocaleString()} chars · ~${estTokens.toLocaleString()} tokens · ${adequacy.label}`}
      defaultOpen
    >
      <div className="block kvgrid">
        <div className="full" style={{ marginBottom: 6 }}>
          <span className={`tag-pill ${adequacy.color}-pill`} style={{ fontSize: 12, padding: '2px 10px' }}>
            CONTEXT {adequacy.label}
          </span>
          <span className="dim-text" style={{ marginLeft: 8 }}>{adequacy.note}</span>
        </div>
        <div><strong>Returned:</strong> {chunks.length} chunks</div>
        <div><strong>Total context:</strong> {totalChars.toLocaleString()} chars (~{estTokens.toLocaleString()} tokens)</div>
        <div><strong>Avg chunk:</strong> {avgLen} chars</div>
        <div><strong>Range:</strong> {minLen}–{maxLen} chars</div>
        <div><strong>Thin (&lt;300 chars):</strong>{' '}
          <span className={thinCount > 0 ? 'tag-pill warn-pill' : 'tag-pill ok-pill'}>
            {thinCount} of {chunks.length}
          </span>
        </div>
        <div><strong>With section path:</strong>{' '}
          <span className={`tag-pill ${sectionCovered === chunks.length ? 'ok-pill' : sectionCovered > 0 ? 'warn-pill' : 'dim-pill'}`}>
            {sectionCovered} / {chunks.length}
          </span>
        </div>
        <div><strong>With summary:</strong>{' '}
          <span className="tag-pill dim-pill">
            {summaryCovered} / {chunks.length}
          </span>
        </div>
        <div><strong>Top rerank:</strong> {chunks[0]?.rerank_score?.toFixed?.(3) ?? '—'}</div>
        <div className="full">
          <strong>Tier breakdown:</strong>{' '}
          {Object.entries(tierCounts).map(([t, n]) => (
            <span key={t} className="tag-pill">{t}: {n}</span>
          ))}
        </div>
        <div className="full">
          <strong>Retrieval arm split:</strong>{' '}
          {Object.entries(armCounts).map(([arms, n]) => (
            <span key={arms} className={`tag-pill ${arms === 'bm25' ? 'ok-pill' : arms === 'vector' ? 'warn-pill' : ''}`}>
              {arms}: {n}
            </span>
          ))}
        </div>
        <div className="full">
          <strong>Confidence labels:</strong>{' '}
          {Object.entries(
            chunks.reduce((acc: Record<string, number>, c) => {
              const k = c.confidence_label || 'unset'
              acc[k] = (acc[k] || 0) + 1
              return acc
            }, {})
          ).map(([k, n]) => (
            <span key={k} className={`tag-pill ${k === 'high' ? 'ok-pill' : k === 'low' ? 'warn-pill' : 'dim-pill'}`}>
              {k}: {n}
            </span>
          ))}
        </div>
      </div>
    </CollapsibleSection>
  )
}


// ── Helpers ───────────────────────────────────────────────────────────

interface FlowStage {
  id: string
  label: string
  active: boolean
  ok: boolean
  sub?: string
}

function PipelineFlow({ stages }: { stages: FlowStage[] }) {
  return (
    <div className="pipeline-flow">
      {stages.map((s, i) => (
        <div key={s.id} className="flow-step-wrap">
          <div
            className={`flow-step ${s.active ? 'active' : 'inactive'} ${s.ok ? 'ok' : 'warn'}`}
          >
            <div className="flow-step-label">{s.label}</div>
            {s.sub && <div className="flow-step-sub">{s.sub}</div>}
          </div>
          {i < stages.length - 1 && <div className="flow-arrow">▶</div>}
        </div>
      ))}
    </div>
  )
}

export function CollapsibleSection({
  title,
  subtitle,
  defaultOpen = true,
  children,
}: {
  title: string
  subtitle?: string
  defaultOpen?: boolean
  children: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className={`section collapsible ${open ? 'open' : 'closed'}`}>
      <div className="section-header" onClick={() => setOpen(!open)}>
        <span className="section-toggle">{open ? '▾' : '▸'}</span>
        <span className="section-title">{title}</span>
        {subtitle && <span className="section-subtitle">{subtitle}</span>}
      </div>
      {open && <div className="section-body">{children}</div>}
    </div>
  )
}

function PartitionBucket({
  label,
  items,
  kind,
}: {
  label: string
  items: { term: string; kind?: string; code?: string; selectivity?: number }[]
  kind: 'ok' | 'warn' | 'dim'
}) {
  if (!items.length) return null
  return (
    <div className={`partition-bucket bucket-${kind}`}>
      <strong>{label}</strong>
      <div>
        {items.map((it, i) => (
          <span key={i} className={`tag-pill ${kind}-pill`}>
            {it.term}
            {it.selectivity !== undefined && (
              <span className="dim-text"> ({it.selectivity.toFixed(2)})</span>
            )}
          </span>
        ))}
      </div>
    </div>
  )
}
