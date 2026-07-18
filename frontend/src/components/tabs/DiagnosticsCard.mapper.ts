/**
 * mapToTree — converts a real AgentResponse + rag_query_decisions row into a DiagnosticsTree.
 * Field→level mapping sourced from mobius-rag/docs/diagnostics-card-content-tree.md (EVAL agent).
 */
import type { AgentResponse, ScoringTraceItem } from './AgentPipelineTrace'
import type { DiagnosticsTree, TreeNode } from './DiagnosticsCard'

// Shape of one rag_query_decisions row passed in from the caller
export interface GradeRow {
  retrieval_grade?: number | null
  synthesis_grade?: number | null
  synthesis_gap?: number | null
  per_claim_ledger?: Array<{
    fact?: string
    status?: string       // 'validated' | 'contradicted' | 'unverified'
    chunk_id?: number | null
    support?: number | string
  }> | null
  leaf_key?: string | null
  feature_vector?: Record<string, number> | null
  strategy_scores?: Record<string, number> | null
  priors_version?: string | null
  fact_checker_version?: string | null
  corpus_version?: string | number | null
  is_prod?: boolean | null
  caller_id?: string | null   // GAP: not captured yet
}

// ── helpers ───────────────────────────────────────────────────────────────────

function obj(v: unknown): Record<string, unknown> {
  return v && typeof v === 'object' && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : {}
}

function trunc(s: string | null | undefined, n: number): string {
  if (!s) return ''
  return s.length > n ? s.slice(0, n) + '…' : s
}

function fmt2(n: number | null | undefined): string {
  return n != null ? n.toFixed(2) : '—'
}

function fmt3(n: number | null | undefined): string {
  return n != null ? n.toFixed(3) : '—'
}

// ── REASON: gate ──────────────────────────────────────────────────────────────

function buildGateNode(response: AgentResponse): TreeNode | null {
  if (!response.fail_fast) return null
  const fired = !!response.fail_fast.reason
  return {
    id: 'reason-gate',
    title: 'Gate',
    summary: fired
      ? `fast-exit: ${response.fail_fast.reason}`
      : 'passed — no fast-exit',
    status: fired ? 'warn' : 'ok',
    telemetry: {
      passed: !fired,
      reason: response.fail_fast.reason ?? null,
      response_mode: response.fail_fast.response_mode ?? null,
      user_message: response.fail_fast.user_message ?? null,
    },
  }
}

// ── REASON: cleanup ───────────────────────────────────────────────────────────

function buildCleanupNode(response: AgentResponse): TreeNode {
  const qp = response.query_profile ?? {}
  const tags = qp.tag_matches ?? []
  const lits = qp.literal_anchors ?? []
  const untagged = qp.untagged_meaningful_tokens ?? []
  const kept = tags.length + lits.length + untagged.length
  return {
    id: 'reason-cleanup',
    title: 'Cleanup',
    summary: `${kept} tokens kept`,
    status: 'ok',
    telemetry: {
      literal_anchors: lits.join(', ') || 'none',
      tag_matches: tags.join(', ') || 'none',
      untagged_meaningful: untagged.join(', ') || 'none',
      dropped: 'noise (not captured)',
    },
  }
}

// ── REASON: rewrite ───────────────────────────────────────────────────────────

function buildRewriteNode(response: AgentResponse): TreeNode {
  const qps = response.queries_per_strategy
  return {
    id: 'reason-rewrite',
    title: 'Rewrite',
    summary: '3 per-strategy variants',
    status: 'ok',
    telemetry: qps ? {
      hybrid: qps.hybrid ?? '—',
      phrase_strict: qps.phrase_strict ?? '—',
      vector_broad: qps.vector_broad ?? '—',
    } : { status: 'not captured yet' },
  }
}

// ── REASON: classify ──────────────────────────────────────────────────────────

function buildClassifyNode(response: AgentResponse): TreeNode {
  const qp = response.query_profile ?? {}
  const routing = response.routing ?? {}
  const cf = obj((routing as Record<string, unknown>).classify_flags)  // live rev 00423
  return {
    id: 'reason-classify',
    title: 'Classify',
    summary: [
      qp.query_type,
      qp.coverage != null ? `coverage ${fmt2(qp.coverage)}` : null,
    ].filter(Boolean).join(' · ') || 'query profile',
    status: 'ok',
    telemetry: {
      query_type: qp.query_type ?? '—',
      coverage: qp.coverage ?? '—',
      tag_matches: qp.tag_matches?.join(', ') || 'none',
      literal_anchors: qp.literal_anchors?.join(', ') || 'none',
      is_exploratory: cf.is_exploratory ?? '—',
      has_service_specificity: cf.has_service_specificity ?? '—',
    },
  }
}

// ── REASON: scorer ────────────────────────────────────────────────────────────

function buildScorerNode(response: AgentResponse): TreeNode {
  const routing = response.routing ?? {}
  const scores = (routing.scores ?? {}) as Record<string, number>
  const breakdown = obj(routing.score_breakdown ?? {})
  const fv = obj((routing as Record<string, unknown>).feature_vector ?? {})
  const sorted = Object.entries(scores).sort(([, a], [, b]) => b - a)
  const [topStrat, topScore] = sorted[0] ?? ['?', null]

  return {
    id: 'reason-scorer',
    title: 'Scorer',
    summary: topScore != null
      ? `${topStrat} wins ${fmt3(topScore)} argmax`
      : 'linear router',
    status: routing.fail_fast_reason ? 'warn' : 'ok',
    strategyScores: sorted.length > 0 ? scores : undefined,
    telemetry: {
      routing_method: routing.method ?? '—',
      withdrawn: (routing.withdrawn ?? []).join(', ') || 'none',
      fail_fast_reason: routing.fail_fast_reason ?? null,
      priors_version: routing.priors_version ?? '—',
      multi_invoke_considered: (routing as Record<string, unknown>).multi_invoke_considered ?? '—',
      ...(Object.keys(fv).length > 0 ? { feature_vector: fv } : {}),
      ...(Object.keys(breakdown).length > 0 ? { score_breakdown: breakdown } : {}),
      self_assessments: routing.self_assessments ?? '—',
    },
  }
}

// ── REASON branch ─────────────────────────────────────────────────────────────

function buildReasonNode(response: AgentResponse, grades: DiagnosticsTree['grades']): TreeNode {
  const qp = response.query_profile ?? {}
  const tel = obj(response.telemetry ?? {})
  const routing = response.routing ?? {}
  const scores = (routing.scores ?? {}) as Record<string, number>
  const sorted = Object.entries(scores).sort(([, a], [, b]) => b - a)
  const [topStrat, topScore] = sorted[0] ?? [null, null]

  const children: TreeNode[] = []
  const gate = buildGateNode(response)
  if (gate) children.push(gate)
  children.push(buildCleanupNode(response))
  children.push(buildRewriteNode(response))
  children.push(buildClassifyNode(response))
  children.push(buildScorerNode(response))

  // not-built planned leaves — gray
  for (const [id, title] of [
    ['reason-s', 's · structured-read'],
    ['reason-reformulate', 'reformulate · re-query'],
    ['reason-f', 'f · honest-floor'],
    ['reason-m', 'm · cached-replay'],
    ['reason-research', 'research · fanout'],
  ] as [string, string][]) {
    children.push({ id, title, summary: 'planned — not built', status: 'gray' })
  }

  const classifyMs = (tel.classify_ms as number) || 0
  const routeMs = (tel.route_ms as number) || 0

  return {
    id: 'reason',
    title: 'Reason',
    summary: [
      `cleanup → ${qp.query_type ?? '?'}`,
      topStrat ? `scored ${topStrat} ${fmt3(topScore)} argmax` : null,
      grades.gap != null ? `gap ${fmt2(grades.gap)}` : null,
    ].filter(Boolean).join(' · '),
    latencyMs: (classifyMs + routeMs) || undefined,
    status: 'ok',
    children,
  }
}

// ── ACT: retrieve per strategy ────────────────────────────────────────────────

function chunkItemStatus(item: ScoringTraceItem): TreeNode['status'] {
  const score = item.rerank_signals?.rerank_score ?? item.rerank_signals?.total_raw
  if (score == null) return 'gray'
  return score >= 0.35 ? 'ok' : 'warn'
}

function buildChunkNode(item: ScoringTraceItem, idx: number): TreeNode {
  const sig = item.rerank_signals ?? {}
  return {
    id: `chunk-${item.chunk_id ?? idx}`,
    title: `Chunk ${item.rank ?? idx + 1}`,
    summary: item.document_name
      ? `${item.document_name}${item.page_number != null ? ` p${item.page_number}` : ''}`
      : `chunk ${idx + 1}`,
    status: chunkItemStatus(item),
    telemetry: {
      sim_raw: sig.sim_raw ?? '—',
      authority_raw: sig.authority_raw ?? '—',
      length_raw: sig.length_raw ?? '—',
      jpd_raw: sig.jpd_raw ?? '—',
      tag_coverage_raw: sig.tag_coverage_raw ?? '—',
      tag_coverage_present: sig.tag_coverage_present?.join(', ') ?? '—',
      tag_coverage_missing: sig.tag_coverage_missing?.join(', ') ?? '—',
      rerank_score: sig.rerank_score ?? '—',
      authority_level: item.authority_level ?? '—',
      retrieval_arms: item.retrieval_arms?.join('+') ?? '—',
    },
  }
}

type StrategyAttempt = NonNullable<AgentResponse['strategies_tried']>[0]

function buildRetrieveNodeA(st: StrategyAttempt): TreeNode {
  const arms = st.arms ?? {}
  const timing = (arms.timing_ms ?? {}) as Partial<{ embed: number; bm25: number; vector: number; rerank: number }>
  const scoring = st.scoring_trace ?? []

  const armsNode: TreeNode = {
    id: 'retrieve-a-arms',
    title: 'Arms + timing',
    summary: [
      arms.bm25_pool_hits ? `bm25:${arms.bm25_pool_hits}` : null,
      arms.vector_pool_hits ? `vec:${arms.vector_pool_hits}` : null,
    ].filter(Boolean).join(' · ') || '—',
    status: 'ok',
    telemetry: {
      bm25_hits: arms.bm25_pool_hits ?? '—',
      vector_hits: arms.vector_pool_hits ?? '—',
      bm25_only: arms.result_breakdown?.bm25_only ?? '—',
      vector_only: arms.result_breakdown?.vector_only ?? '—',
      both: arms.result_breakdown?.both ?? '—',
      embed_ms: timing.embed ?? '—',
      bm25_ms: timing.bm25 ?? '—',
      vector_ms: timing.vector ?? '—',
      rerank_ms: timing.rerank ?? '—',
    },
  }

  const chunkChildren = scoring.slice(0, 10).map((item, i) => buildChunkNode(item, i))
  const chunksNode: TreeNode = {
    id: 'retrieve-a-chunks',
    title: `Chunk scores (${scoring.length})`,
    summary: `top rerank ${fmt3(st.top_rerank)} · showing top ${Math.min(10, scoring.length)}`,
    status: (st.top_rerank ?? 0) >= 0.35 ? 'ok' : 'warn',
    children: chunkChildren.length > 0 ? chunkChildren : undefined,
    telemetry: chunkChildren.length === 0 ? { status: 'no scoring trace' } : undefined,
  }

  return {
    id: 'retrieve-a',
    title: 'Retrieve · a · hybrid',
    summary: `${st.n_chunks} chunks · top ${fmt3(st.top_rerank)}`,
    latencyMs: st.elapsed_ms,
    status: st.succeeded ? ((st.top_rerank ?? 0) >= 0.35 ? 'ok' : 'warn') : 'warn',
    children: [armsNode, chunksNode],
  }
}

function buildRetrieveNodeB(st: StrategyAttempt, response: AgentResponse): TreeNode {
  const tel = obj(response.telemetry ?? {})
  const bTel = obj(tel.strategy_b as Record<string, unknown> ?? {})
  const themes = response.themes ?? []
  const td = (response.theme_diagnostic ?? {}) as Partial<{ n_themes: number; dominant_theme_share: number; narrower_than_expected: boolean }>

  return {
    id: 'retrieve-b',
    title: 'Retrieve · b · wide→themes',
    summary: `${themes.length} themes · ${st.n_chunks} chunks`,
    latencyMs: st.elapsed_ms,
    status: st.succeeded ? 'ok' : 'warn',
    telemetry: {
      wide_hits: bTel.wide_hits ?? '—',
      wide_ms: bTel.wide_ms ?? '—',
      themes_ms: bTel.themes_ms ?? '—',
      narrow_ms: bTel.narrow_ms ?? '—',
      n_themes: td.n_themes ?? themes.length,
      dominant_theme_share: td.dominant_theme_share ?? '—',
      narrower_than_expected: td.narrower_than_expected ?? '—',
      themes: themes
        .map(t => `${t.label} (${t.n_docs}d top:${t.top_rerank?.toFixed(2) ?? '?'})`)
        .join('; ') || '—',
    },
  }
}

function buildRetrieveNodeC(st: StrategyAttempt, response: AgentResponse): TreeNode {
  const cits = response.validated_citations ?? []
  return {
    id: 'retrieve-c',
    title: 'Retrieve · c · reverse-RAG',
    summary: `${cits.length} citations verified`,
    latencyMs: st.elapsed_ms,
    status: st.succeeded ? 'ok' : 'warn',
    telemetry: {
      n_citations: cits.length,
      citations: cits
        .map(c => `[${c.status ?? '?'}] ${c.document_display_name ?? c.candidate?.document_title ?? '?'}`)
        .join('; ') || '—',
    },
  }
}

function buildRetrieveNodeD(st: StrategyAttempt): TreeNode {
  return {
    id: 'retrieve-d',
    title: 'Retrieve · d · external',
    summary: `${st.n_chunks} results · top ${fmt3(st.top_rerank)}`,
    latencyMs: st.elapsed_ms,
    status: st.succeeded ? 'ok' : 'warn',
    telemetry: {
      note: st.note || '—',
      n_chunks: st.n_chunks,
      top_rerank: st.top_rerank ?? '—',
      fetch_tier_breakdown: 'not captured yet',
      per_url_breakdown: 'not captured yet',
    },
  }
}

function buildRetrieveNode(st: StrategyAttempt, response: AgentResponse): TreeNode {
  switch (st.strategy) {
    case 'a': return buildRetrieveNodeA(st)
    case 'b': return buildRetrieveNodeB(st, response)
    case 'c': return buildRetrieveNodeC(st, response)
    case 'd': return buildRetrieveNodeD(st)
    default:  return {
      id: `retrieve-${st.strategy}`,
      title: `Retrieve · ${st.strategy}`,
      summary: `${st.n_chunks} chunks · top ${fmt3(st.top_rerank)}`,
      latencyMs: st.elapsed_ms,
      status: st.succeeded ? 'ok' : 'warn',
    }
  }
}

// ── ACT: synthesize ───────────────────────────────────────────────────────────

function buildSynthesizeNode(response: AgentResponse, grades: DiagnosticsTree['grades']): TreeNode {
  const tel = obj(response.telemetry ?? {})
  const overallWarn = grades.synthesis != null && grades.synthesis < 0.5
  return {
    id: 'synthesize',
    title: 'Synthesize',
    summary: `answer built · ${response.confidence ?? '?'}`,
    latencyMs: (tel.llm_ms as number) ?? undefined,
    status: overallWarn ? 'warn' : 'ok',
    telemetry: {
      llm_ms: tel.llm_ms ?? '—',         // live rev 00423
      model: tel.model ?? '—',            // live rev 00423
      used_passages: tel.used_passages ?? '—',      // live rev 00423 — indices into per_claim_ledger
      n_passages_offered: tel.n_passages_offered ?? '—',  // live rev 00423
      finish_reason: tel.finish_reason ?? '—',
    },
  }
}

// ── ACT branch ────────────────────────────────────────────────────────────────

function buildActNode(response: AgentResponse, grades: DiagnosticsTree['grades']): TreeNode {
  const tried = response.strategies_tried ?? []
  const totalChunks = tried.reduce((s, t) => s + (t.n_chunks ?? 0), 0)
  const totalLatency = tried.reduce((s, t) => s + (t.elapsed_ms ?? 0), 0)

  const retrieveNodes = tried.map(st => buildRetrieveNode(st, response))

  const rerankNode: TreeNode = {
    id: 'rerank',
    title: 'Rerank',
    summary: tried.length > 0
      ? `top ${tried.map(s => fmt3(s.top_rerank)).join(' / ')}`
      : '—',
    status: tried.some(s => (s.top_rerank ?? 0) < 0.35) ? 'warn' : 'ok',
    telemetry: Object.fromEntries(tried.map(s => [`${s.strategy}_top`, s.top_rerank ?? '—'])),
  }

  const assembleNode: TreeNode = {
    id: 'assemble',
    title: 'Assemble',
    summary: `${totalChunks} chunks · page+content dedup`,
    status: 'ok',
    telemetry: { mode: 'page-dedup + content-dedup (sha/200c) + neighbors exempt', chunk_cap: 10 },
  }

  const synthNode = buildSynthesizeNode(response, grades)
  const answer = response.llm_answer ?? ''

  return {
    id: 'act',
    title: 'Act',
    summary: [
      tried.length > 0 ? `strategy ${tried.map(s => s.strategy).join('+')}` : null,
      totalChunks > 0 ? `${totalChunks} chunks` : null,
      answer.length > 0 ? `${answer.length}c answer` : null,
      response.confidence,
    ].filter(Boolean).join(' · '),
    latencyMs: totalLatency || undefined,
    status: retrieveNodes.some(n => n.status === 'warn') ? 'warn' : 'ok',
    children: [...retrieveNodes, rerankNode, assembleNode, synthNode],
  }
}

// ── OBSERVE branch ────────────────────────────────────────────────────────────

function buildObserveNode(row: GradeRow | null, grades: DiagnosticsTree['grades']): TreeNode {
  const ledger = row?.per_claim_ledger ?? []
  const anyContradicted = ledger.some(c => c.status === 'contradicted')

  const retGradeNode: TreeNode = {
    id: 'observe-retrieval-grade',
    title: 'retrieval_grade',
    summary: grades.retrieval != null
      ? `${(grades.retrieval * 100).toFixed(0)}%`
      : 'not graded in prod',
    status: grades.retrieval != null ? 'ok' : 'gray',
    telemetry: {
      grade: grades.retrieval ?? 'null — no gold set in prod',
      basis: 'chunk-only fact_check',
    },
  }

  const synGradeNode: TreeNode = {
    id: 'observe-synthesis-grade',
    title: 'synthesis_grade',
    summary: grades.synthesis != null
      ? `${(grades.synthesis * 100).toFixed(0)}% · gap ${fmt2(grades.gap)}`
      : '—',
    status: (grades.gap ?? 0) < 0 ? 'warn' : (grades.synthesis != null ? 'ok' : 'gray'),
    telemetry: {
      grade: grades.synthesis ?? '—',
      gap: grades.gap ?? '—',
    },
  }

  const claimNode: TreeNode = {
    id: 'observe-claims',
    title: 'per_claim_ledger',
    summary: ledger.length > 0
      ? `${ledger.filter(c => c.status === 'validated').length}/${ledger.length} validated`
      : 'no ledger',
    status: anyContradicted ? 'warn' : (ledger.length > 0 ? 'ok' : 'gray'),
    telemetry: Object.fromEntries(
      ledger.map((c, i) => [`claim_${i + 1}`, `[${c.status ?? '?'}] ${trunc(c.fact, 80)}`])
    ),
  }

  const decisionRow: TreeNode = {
    id: 'observe-decision-row',
    title: 'decision_row',
    summary: row?.leaf_key ?? '—',
    status: 'ok',
    telemetry: {
      leaf_key: row?.leaf_key ?? '—',
      priors_version: row?.priors_version ?? '—',
      fact_checker_version: row?.fact_checker_version ?? '—',
      corpus_version: row?.corpus_version != null
        ? String(row.corpus_version) + (row.corpus_version === 1 ? ' (bump not wired)' : '')
        : '—',
      is_prod: row?.is_prod ?? '—',
      caller_id: 'not captured yet',  // GAP: open
      feature_vector: row?.feature_vector ?? '—',
    },
  }

  return {
    id: 'observe',
    title: 'Observe',
    summary: [
      grades.retrieval != null
        ? `retrieval ${(grades.retrieval * 100).toFixed(0)}%`
        : 'retrieval ungraded',
      grades.synthesis != null
        ? `synthesis ${(grades.synthesis * 100).toFixed(0)}%`
        : null,
      ledger.length > 0
        ? `${ledger.filter(c => c.status === 'validated').length}/${ledger.length} validated`
        : null,
    ].filter(Boolean).join(' · '),
    status: anyContradicted || (grades.gap ?? 0) < 0 ? 'warn' : 'ok',
    children: [retGradeNode, synGradeNode, claimNode, decisionRow],
  }
}

// ── DECIDE branch ─────────────────────────────────────────────────────────────

function buildDecideNode(
  response: AgentResponse,
  row: GradeRow | null,
  grades: DiagnosticsTree['grades'],
): TreeNode {
  const routing = response.routing ?? {}
  const tel = obj(response.telemetry ?? {})

  const multiConsidered = (routing as Record<string, unknown>).multi_invoke_considered
  const multiInvokeNode: TreeNode = {
    id: 'decide-multi-invoke',
    title: 'multi_invoke',
    summary: multiConsidered ? 'considered' : 'not triggered',
    status: multiConsidered ? 'ok' : 'gray',
    telemetry: { multi_invoke_considered: multiConsidered ?? false },
  }

  const stratChain = (tel.strategy_chain as string[] | undefined) ?? []
  const escalated = !!(tel.escalated)
  const escalateNode: TreeNode = {
    id: 'decide-escalate',
    title: 'escalate',
    summary: escalated
      ? `escalated via ${stratChain.join('→')}`
      : stratChain.length > 0 ? stratChain.join('→') : 'single-pass',
    status: 'ok',
    telemetry: {
      strategy_chain: stratChain.join('→') || '—',
      escalated,
      inherited_boost: tel.inherited_boost ?? '—',
    },
  }

  const fastExit = obj(tel.fast_exit as Record<string, unknown> ?? {})
  const fastExitFired = !!(fastExit.fired)
  const fastExitNode: TreeNode = {
    id: 'decide-fast-exit',
    title: 'fast_exit',
    summary: fastExitFired ? `fired: ${fastExit.reason ?? '?'}` : 'no cache hit',
    status: fastExitFired ? 'warn' : 'ok',
    telemetry: { fired: fastExitFired, reason: fastExit.reason ?? null },
  }

  const banditNode: TreeNode = {
    id: 'decide-bandit',
    title: 'bandit',
    summary: 'not wired — loop open',
    status: 'gray',
    telemetry: {
      reward: grades.synthesis ?? 'null',
      would_update: `weights[${row?.leaf_key ?? '?'}]`,
      status: 'loop-open — not built',
    },
  }

  return {
    id: 'decide',
    title: 'Decide',
    summary: [
      grades.gap != null ? `gap ${fmt2(grades.gap)}` : null,
      escalated ? 'multi-attempt' : 'single-pass',
      fastExitFired ? `fast-exit: ${fastExit.reason}` : null,
      'bandit not wired',
    ].filter(Boolean).join(' · '),
    status: 'warn',   // always warn — bandit loop is open
    children: [multiInvokeNode, escalateNode, fastExitNode, banditNode],
  }
}

// ── Root ──────────────────────────────────────────────────────────────────────

function buildRoot(
  response: AgentResponse,
  row: GradeRow | null,
  partial: Omit<DiagnosticsTree, 'root'>,
): TreeNode {
  const { grades, claims } = partial
  const overallWarn =
    (grades.synthesis != null && grades.synthesis < 0.5) ||
    (grades.gap != null && grades.gap < 0)

  return {
    id: 'loop',
    title: 'Full query trace',
    summary: [
      `routed ${response.strategy_used ?? '?'}`,
      trunc(response.llm_answer, 60) || null,
      `retr ${fmt2(grades.retrieval)}/synth ${fmt2(grades.synthesis)}`,
      claims ? `${claims.passed}/${claims.total} ✓` : null,
    ].filter(Boolean).join(' · '),
    latencyMs: partial.latencyMs,
    status: overallWarn ? 'warn' : 'ok',
    children: [
      buildReasonNode(response, grades),
      buildActNode(response, grades),
      buildObserveNode(row, grades),
      buildDecideNode(response, row, grades),
    ],
  }
}

// ── Public entry point ────────────────────────────────────────────────────────

export function mapToTree(
  query: string,
  response: AgentResponse,
  row: GradeRow | null = null,
): DiagnosticsTree {
  const qp = response.query_profile ?? {}
  const tel = obj(response.telemetry ?? {})
  const ledger = row?.per_claim_ledger ?? []

  const grades: DiagnosticsTree['grades'] = {
    retrieval: row?.retrieval_grade ?? null,
    synthesis: row?.synthesis_grade ?? null,
    gap: row?.synthesis_gap ?? null,
  }

  const claims = ledger.length > 0
    ? { passed: ledger.filter(c => c.status === 'validated').length, total: ledger.length }
    : null

  const partial: Omit<DiagnosticsTree, 'root'> = {
    query,
    answer: response.llm_answer ?? '',
    route: {
      strategy: response.strategy_used ?? '?',
      confidence: (response.confidence ?? 'unknown') as DiagnosticsTree['route']['confidence'],
    },
    focusTags: qp.tag_matches ?? [],
    grades,
    claims,
    latencyMs: (tel.total_ms as number) ?? 0,
    decisionId: response.routing_decision_id ?? '',
  }

  return { ...partial, root: buildRoot(response, row, partial) }
}
