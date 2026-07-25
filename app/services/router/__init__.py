"""Router — Step 4 of the RAG answer engine (constrained optimizer).

Runs ONCE upfront, before Fillers acts.
Dispatch → Allocate (expected-value fallback chains) → Persist → RoutingLadder.

Priors: file-loaded from eval/priors_bootstrap.yaml (Eval-owned; mtime-cached,
swap without redeploy). Hardcoded snapshot is a last-resort fallback only.

TECH gates:
  (b) No DB access during optimization (priors are file/cache reads)
  (d) Segments timed
  (f) ONE-WRITER: persist_decision() is the only rag_query_decisions writer
"""

from app.services.router.dispatch import dispatch, DispatchDecision
from app.services.router.allocation import (
    AnswerSlot,
    RoutingLadder,
    STRATEGY_PRIORITY_ORDER,
    allocate_strategies,
    chain_expected_accuracy,
    chain_success_probability,
    resolve_tolerance_pct,
)
from app.services.router.priors import (
    PRIORS_SEED_DEFAULTS,
    SEED_PSEUDO_COUNT,
    PriorsBundle,
    StrategyProfile,
    compute_depth_bucket,
    default_priors_path,
    get_priors_version,
    load_priors,
    lookup_priors,
    lookup_priors_qclass_fallback,
    wilson_lower_bound,
)
from app.services.router.persist import persist_decision
from app.services.router.decision import RouterDecision, RoutingContext, ResourcePosture
from app.services.router.router import route
from app.services.router.dispatch import stable_draw
from app.services.router.optimizer import optimize_allocation
from app.services.router.bayesian_optimizer import optimize_allocation_bayesian
from app.services.router.priors import beta_lower_bound
from app.services.router.tracing import DecisionTrace, SlotTrace, StrategyStep
from app.services.router.router_narrate import narrate

__all__ = [
    "dispatch",
    "DispatchDecision",
    "AnswerSlot",
    "RoutingLadder",
    "STRATEGY_PRIORITY_ORDER",
    "allocate_strategies",
    "chain_expected_accuracy",
    "chain_success_probability",
    "resolve_tolerance_pct",
    "PRIORS_SEED_DEFAULTS",
    "SEED_PSEUDO_COUNT",
    "wilson_lower_bound",
    "PriorsBundle",
    "StrategyProfile",
    "compute_depth_bucket",
    "default_priors_path",
    "get_priors_version",
    "load_priors",
    "lookup_priors",
    "lookup_priors_qclass_fallback",
    "persist_decision",
    "RouterDecision",
    "RoutingContext",
    "ResourcePosture",
    "route",
    "stable_draw",
    "optimize_allocation",
    "optimize_allocation_bayesian",
    "beta_lower_bound",
    "DecisionTrace",
    "SlotTrace",
    "StrategyStep",
    "narrate",
]
