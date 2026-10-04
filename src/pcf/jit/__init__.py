"""Experimental JIT cognitive execution optimizer."""

from .cache import CacheEnvelope, RiskAwareResultCache
from .fingerprint import FingerprintAudit, FingerprintAuditSummary, FingerprintObservation, TaskSignature
from .hotpath import CompileEconomics, HotPathDetector, HotPathStats
from .optimizer import JITOptimizer, NoAdmissibleRoute
from .pcf_cost import (
    PCFInputCostEstimate,
    apply_pcf_input_cost,
    estimate_pcf_input_cost,
    make_pcf_metered_shadow_runner,
)
from .promotion import PromotionDecision, PromotionEvaluator, PromotionEvidence, PromotionPolicy
from .registry import (
    InvalidRouteTransition,
    RegisteredRoute,
    RouteRegistry,
    RouteState,
    RouteTransition,
    StaleRouteGeneration,
)
from .shadow import (
    ReplayRouteStats,
    ShadowAssessment,
    ShadowExecutor,
    ShadowObservation,
    ShadowRequest,
    ShadowResult,
    ShadowRun,
    ShadowTarget,
    TraceReplaySummary,
    TraceReplayer,
)
from .types import CandidateScore, ExecutionPrimitive, OptimizerPolicy, RouteCandidate, RouteDecision

__all__ = [
    "CacheEnvelope",
    "CandidateScore",
    "CompileEconomics",
    "ExecutionPrimitive",
    "FingerprintAudit",
    "FingerprintAuditSummary",
    "FingerprintObservation",
    "HotPathDetector",
    "HotPathStats",
    "InvalidRouteTransition",
    "JITOptimizer",
    "NoAdmissibleRoute",
    "OptimizerPolicy",
    "PCFInputCostEstimate",
    "PromotionDecision",
    "PromotionEvaluator",
    "PromotionEvidence",
    "PromotionPolicy",
    "RegisteredRoute",
    "ReplayRouteStats",
    "RiskAwareResultCache",
    "RouteCandidate",
    "RouteDecision",
    "RouteRegistry",
    "RouteState",
    "RouteTransition",
    "ShadowAssessment",
    "ShadowExecutor",
    "ShadowObservation",
    "ShadowRequest",
    "ShadowResult",
    "ShadowRun",
    "ShadowTarget",
    "StaleRouteGeneration",
    "TaskSignature",
    "TraceReplaySummary",
    "TraceReplayer",
    "apply_pcf_input_cost",
    "estimate_pcf_input_cost",
    "make_pcf_metered_shadow_runner",
]
