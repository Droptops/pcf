"""Experimental JIT cognitive execution optimizer."""

from .cache import CacheEnvelope, RiskAwareResultCache
from .hotpath import CompileEconomics, HotPathDetector, HotPathStats
from .optimizer import JITOptimizer, NoAdmissibleRoute
from .promotion import PromotionDecision, PromotionEvaluator, PromotionEvidence, PromotionPolicy
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
    "HotPathDetector",
    "HotPathStats",
    "JITOptimizer",
    "NoAdmissibleRoute",
    "OptimizerPolicy",
    "PromotionDecision",
    "PromotionEvaluator",
    "PromotionEvidence",
    "PromotionPolicy",
    "ReplayRouteStats",
    "RiskAwareResultCache",
    "RouteCandidate",
    "RouteDecision",
    "ShadowAssessment",
    "ShadowExecutor",
    "ShadowObservation",
    "ShadowRequest",
    "ShadowResult",
    "ShadowRun",
    "ShadowTarget",
    "TraceReplaySummary",
    "TraceReplayer",
]
