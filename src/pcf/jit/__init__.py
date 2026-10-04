"""Experimental JIT cognitive execution optimizer."""

from .authority import (
    AuthorityEvidenceError,
    AuthorityGateEvidence,
    apply_faar_authority,
    normalize_faar_authority_decision,
)
from .cache import CacheEnvelope, RiskAwareResultCache
from .cost_calibration import CostCalibrationPolicy, CostCalibrationSnapshot, PCFCostCalibrator
from .fingerprint import FingerprintAudit, FingerprintAuditSummary, FingerprintObservation, TaskSignature
from .hotpath import CompileEconomics, CompileEligibility, HotPathDetector, HotPathStats
from .jev_openrouter import (
    DEFAULT_JEV_MODEL,
    DEFAULT_OPENROUTER_DECISIONS_URL,
    JevOpenRouterError,
    JevRouteAdvice,
    JevRouteBenchmarkStats,
    JevUsage,
    OpenRouterJevClient,
    annotate_candidates_with_jev,
)
from .optimizer import JITOptimizer, NoAdmissibleRoute
from .pcf_cost import (
    PCFInputCostEstimate,
    apply_pcf_input_cost,
    estimate_pcf_input_cost,
    make_pcf_metered_shadow_runner,
)
from .pcf_reconcile import (
    PCFInputCostReconciliation,
    PCFReconciliationStats,
    apply_pcf_reconciled_input_cost,
    make_pcf_reconciled_shadow_runner,
    reconcile_pcf_input_cost,
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
from .sqlite_registry import RegistryStorageError, SQLiteRouteRegistry
from .types import CandidateScore, ExecutionPrimitive, OptimizerPolicy, RouteCandidate, RouteDecision

__all__ = [
    "AuthorityEvidenceError",
    "AuthorityGateEvidence",
    "CacheEnvelope",
    "CandidateScore",
    "CompileEconomics",
    "CompileEligibility",
    "CostCalibrationPolicy",
    "CostCalibrationSnapshot",
    "DEFAULT_JEV_MODEL",
    "DEFAULT_OPENROUTER_DECISIONS_URL",
    "ExecutionPrimitive",
    "FingerprintAudit",
    "FingerprintAuditSummary",
    "FingerprintObservation",
    "HotPathDetector",
    "HotPathStats",
    "InvalidRouteTransition",
    "JITOptimizer",
    "JevOpenRouterError",
    "JevRouteAdvice",
    "JevRouteBenchmarkStats",
    "JevUsage",
    "NoAdmissibleRoute",
    "OpenRouterJevClient",
    "OptimizerPolicy",
    "PCFCostCalibrator",
    "PCFInputCostEstimate",
    "PCFInputCostReconciliation",
    "PCFReconciliationStats",
    "PromotionDecision",
    "PromotionEvaluator",
    "PromotionEvidence",
    "PromotionPolicy",
    "RegisteredRoute",
    "RegistryStorageError",
    "ReplayRouteStats",
    "RiskAwareResultCache",
    "RouteCandidate",
    "RouteDecision",
    "RouteRegistry",
    "RouteState",
    "RouteTransition",
    "SQLiteRouteRegistry",
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
    "annotate_candidates_with_jev",
    "apply_faar_authority",
    "apply_pcf_input_cost",
    "apply_pcf_reconciled_input_cost",
    "estimate_pcf_input_cost",
    "make_pcf_metered_shadow_runner",
    "make_pcf_reconciled_shadow_runner",
    "normalize_faar_authority_decision",
    "reconcile_pcf_input_cost",
]
