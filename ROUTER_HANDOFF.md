# Router Agent — Code Build Complete & Ready for Calibration

**Status:** ✅ Implementation complete, unit tests passing, all architects signed off. Awaiting real data (Fillers + Synthesis) for calibration.

**Last updated:** 2026-07-23

---

## What Shipped

### 1. Core Modules (9 files)
- `dispatch.py` — dispatch logic (calibration/forced/production paths)
- `allocation.py` — greedy-by-priority with expected-value fallback chains
- `priors.py` — corpus-depth bucketing (5 levels) + strategy profile lookup
- `persist.py` — ONE-WRITER rag_query_decisions (21-column schema)
- `decision.py` — decision types (RouterDecision, RoutingContext, ResourcePosture)
- `router.py` — main orchestrator (route() function)
- `__init__.py` — exports
- `test_router.py` — 32+ unit tests (dispatch, allocation, priors, persistence, integration)
- `test_fallback_chains.py` — expected-value fallback validation

### 2. Schema (21 columns, rag_query_decisions)
```sql
id, agent_id, query,                                    -- 3 identifiers
is_calibration, is_prod, eval_run_id,                  -- 3 mode flags
depth_bucket, strategy_chosen, strategy_sequence,      -- 3 Router outputs
gate_contour, gate_underspecified_kind,                -- 2 deferred (Gate)
reformat_posture, reformat_fanout_n,                   -- 2 deferred (Reformat)
feature_vector, strategy_scores, priors_version,       -- 3 feature context
confidence, accuracy_estimate, cost,                   -- 3 outcome metrics
total_ms, leaf_key                                     -- 2 audit trail
```

### 3. Algorithm
**Dispatch logic:**
- Calibration/forced: skip optimization, max_attempts=1, isolation measurement
- Production: full constrained optimization, parallel slots, weakest-link bottleneck

**Allocation (expected-value fallback chains):**
```python
# For each slot, allocate strategies until:
# P(success | [s1, s2, ..., sN]) >= confidence_bar
# where P(success) = P(s1) + P(!s1)*P(s2) + P(!s1)*P(!s2)*P(s3) + ...
# and time budget MAX(slot_times) <= speed_budget
```

**Corpus-depth bucketing:**
```
Bucket 0: tight (top_score >= 0.90, pool < 50)
Bucket 1: tight-moderate (top_score >= 0.75, pool < 200)
Bucket 2: moderate (top_score >= 0.50, pool < 500)
Bucket 3: broad-moderate (top_score >= 0.25, pool < 5000)
Bucket 4: broad (all others)
```

### 4. Test Coverage
- **Dispatch:** 3 paths (calibration, forced, production), precedence rules
- **Allocation:** priority ordering, time budgets, per-slot limits, feasibility
- **Bucketing:** all 5 buckets, missing data handling
- **Priors:** lookup (primary + fallback), invalid strategies
- **Persistence:** 21-column write, None handling, UUID generation
- **Fallback chains:** expected-value calculation, cumulative success probability
- **Integration:** full route() pipeline, error handling

---

## How It Works

### Entry Point
```python
async def route(db_session_factory, ctx: RoutingContext) -> RouterDecision:
    # 1. DISPATCH: which path (calibration/forced/production)?
    dispatch_decision = dispatch(is_calibration, forced_strategy, ...)
    
    # 2. ALLOCATE: if production, optimize strategy sequences
    routing_ladder = allocate_strategies(slots, pool_metadata, resource_posture)
    
    # 3. PERSIST: write rag_query_decisions row (ONE-WRITER)
    decision_id = await persist_decision(...)
    
    # 4. RETURN: RoutingLadder for Fillers to walk
    return RouterDecision(routing_ladder, decision_id, ...)
```

### Fallback Chain Example
**Query:** "What is the timely filing deadline?"
**Corpus:** Moderate (depth 2), pool_size=400
**Confidence bar:** 0.70

Router allocates: `["a", "b", "c"]`

```
Execution:
  Fillers tries strategy "a" (BM25)
  Observer measures confidence: 0.45 < 0.70 ❌
  
  Fillers tries strategy "b" (broad search)
  Observer measures confidence: 0.45 + 0.55*0.44 ≈ 0.69 ≈ 0.70 ✅
  
Result: slot satisfied with fallback chain [a→b]
Telemetry: depth_bucket=2, strategy_sequence=["a","b","c"], 
           confidence=0.69, latency=2000ms (both a and b attempted)
```

---

## What's Ready vs. What's Blocked

### ✅ Ready Now
- Router code compiles, unit tests pass
- Allocation logic validates against mocked priors
- Dispatch paths working (calibration/forced/production)
- ONE-WRITER rag_query_decisions schema defined
- 21-column telemetry ready to receive real data

### ⏳ Blocked (Waiting For)
- **Fillers:** Strategy execution (real measurements coming from BM25)
- **Synthesis:** Confidence score production (not yet built)
- **Real Pool output:** Corpus-depth signals flowing through Structure→Slots
- **Real Observer:** Confidence/speed verdicts per attempt

---

## Calibration Loop Integration (Week 1–3)

Once Fillers + Synthesis exist:

```
Week 1: Bootstrap with seed priors
  Fillers executes strategies on N queries
  Observer measures confidence per attempt
  rag_query_decisions rows written (21 columns)
  Eval ingests rows WHERE is_calibration=true
  
Week 2: Empirical priors emergence
  Eval groups: (depth_bucket, strategy) → [outcomes]
  Computes empirical {recall_lift, accuracy, latency, cost}
  Replaces seed with empirical in PRIORS_SEED_DEFAULTS
  priors_version tag incremented
  
Week 3+: Convergence
  Cells with N≥50 samples → stable empirical priors
  Bandit continues refining (per-cell updates)
  Router uses empirical priors for production allocation
```

**Key:** Router persists (depth_bucket, strategy_sequence) in every row.
Eval groups by depth_bucket + strategy_chosen → outcome to compute empirical profiles.

---

## How Priors Drive Allocation

**Seed priors (hand-set, conservative):**
- Depth 0 (tight): a high-accuracy, low-recall
- Depth 4 (broad): a high-recall, lower-accuracy

**Allocation response:**
- Tight corpus → prefer strategy a (high accuracy, fast)
- Broad corpus → allocate [a, b, c] for recall coverage

**Empirical priors (real measurements):**
- Week 2 onward, actual performance data replaces seeds
- If real-a outperforms seed-a at depth 2, allocation shifts
- If real-b underperforms, allocated less frequently

**Sensitivity:** Router's allocation directly responds to prior changes.
Running with different prior configs validates algorithm is truly optimizing, not just echoing values back.

---

## Known Limitations & Future Work

### Addressed
- ✅ Expected-value fallback chains (not naive sum)
- ✅ Time budget constraint on fallback depth
- ✅ Caller-mode-dependent tolerance bands (±15% real_time, ±25% background)
- ✅ One-writer enforcement on rag_query_decisions
- ✅ Accuracy-recall tradeoff across corpus depths

### Not Yet Built
- Orchestrator integration (walks Router's strategy_sequences)
- Observer implementation (measures confidence, gates attempts)
- Fillers code build (executes strategies)
- Synthesis module (produces confidence scores)
- Real calibration loop (Eval's data aggregation pipeline)

### Future Optimization Opportunities
- Dynamic strategy ordering (not fixed STRATEGY_PRIORITY_ORDER)
- Confidence calibration (ensure reported confidence matches real success rate)
- Cost-aware allocation (trade cost against confidence/time)
- Multi-objective optimization (Pareto frontier instead of greedy)

---

## File Locations

```
mobius-rag/app/services/router/
  __init__.py                  -- exports
  dispatch.py                  -- dispatch logic
  allocation.py                -- allocation algorithm (expected-value fallback chains)
  priors.py                    -- corpus-depth bucketing, priors lookup
  persist.py                   -- ONE-WRITER rag_query_decisions
  decision.py                  -- Router decision types
  router.py                    -- main orchestrator
  test_router.py               -- 32+ unit tests
  test_fallback_chains.py      -- expected-value validation
  test_priors_sensitivity.py   -- priors variant testing
```

---

## Next Steps (For Whoever Owns Fillers/Synthesis/Orchestrator)

1. **Fillers:** Execute strategies, feed results to Observer
2. **Synthesis:** Produce confidence scores per slot
3. **Orchestrator:** Walk Router's strategy_sequences, gate on Observer verdicts
4. **Eval:** Ingest (depth_bucket, strategy, outcome) tuples, compute empirical priors
5. **Monitoring:** Alert on allocation anomalies (all queries at confidence floor, etc.)

---

## Questions for Ananth

1. Should Router support dynamic strategy ordering (e.g., prioritize cache for real_time callers)?
2. How tight should accuracy-recall calibration be? (Currently using priors; could also measure against ground truth)
3. Should we log feature_vector + strategy_scores for offline analysis, or is depth_bucket + confidence enough?
4. Timeline: when will Fillers + Synthesis be ready to feed real data?

---

**Router is production-ready for code integration. Waiting for real data (Fillers + Synthesis) to begin real calibration.**
