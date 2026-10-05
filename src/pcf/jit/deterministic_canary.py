"""Canary evidence binding for deterministic JIT activation.

The generic registry remains a low-level lifecycle primitive. This module is the
hardened deterministic path: activation evidence is generated only for a route that
is already in CANARY and is bound to that exact route generation, traffic fraction,
fingerprint, candidate key, and observation window.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .deterministic_pipeline import DeterministicCandidateBuild, DeterministicPipelineError
from .fingerprint import FingerprintAuditSummary
from .promotion import PromotionDecision, PromotionEvaluator
from .registry import RegisteredRoute, RouteRegistry, RouteState
from .shadow import TraceReplaySummary


@dataclass(frozen=True)
class DeterministicCanaryPolicy:
    min_attempted_requests: int = 100

    def __post_init__(self) -> None:
        if type(self.min_attempted_requests) is not int or self.min_attempted_requests <= 0:
            raise ValueError("min_attempted_requests must be a positive integer")


@dataclass(frozen=True)
class DeterministicCanaryEvidence:
    activate: bool
    reasons: tuple[str, ...]
    route_id: str
    route_generation: int
    candidate_key: str
    fingerprint: str
    traffic_fraction: float
    window_id: str
    evaluated_at: float
    attempted_requests: int
    promotion: PromotionDecision


def _require_route_matches_build(route: RegisteredRoute, build: DeterministicCandidateBuild) -> None:
    if not isinstance(route, RegisteredRoute):
        raise DeterministicPipelineError("route must be RegisteredRoute")
    if not isinstance(build, DeterministicCandidateBuild):
        raise DeterministicPipelineError("build must be DeterministicCandidateBuild")
    if route.candidate_key != build.candidate_key or route.task_fingerprint != build.fingerprint:
        raise DeterministicPipelineError("registered route does not match deterministic build")
    if route.metadata.get("artifact_sha256") != build.artifact.artifact_sha256:
        raise DeterministicPipelineError("registered artifact identity does not match build")


def start_deterministic_canary(
    registry: RouteRegistry,
    route: RegisteredRoute,
    build: DeterministicCandidateBuild,
    *,
    traffic_fraction: float,
    evidence_ref: str,
    now: float,
) -> RegisteredRoute:
    """Start canary only from the exact eligible deterministic artifact record."""
    if not isinstance(registry, RouteRegistry):
        raise DeterministicPipelineError("registry must be RouteRegistry")
    _require_route_matches_build(route, build)
    if route.state is not RouteState.ELIGIBLE:
        raise DeterministicPipelineError("route must be eligible before canary")
    return registry.start_canary(
        route.route_id,
        traffic_fraction=traffic_fraction,
        evidence_ref=evidence_ref,
        now=now,
        expected_generation=route.generation,
    )


def evaluate_deterministic_canary(
    route: RegisteredRoute,
    summary: TraceReplaySummary,
    *,
    baseline: str,
    projected_future_calls: int | None,
    window_id: str,
    evaluated_at: float,
    promotion_evaluator: PromotionEvaluator | None = None,
    policy: DeterministicCanaryPolicy | None = None,
) -> DeterministicCanaryEvidence:
    """Bind fresh paired canary telemetry to the exact current canary generation."""
    if not isinstance(route, RegisteredRoute) or route.state is not RouteState.CANARY:
        raise DeterministicPipelineError("route must be in canary state")
    if not isinstance(summary, TraceReplaySummary):
        raise DeterministicPipelineError("summary must be TraceReplaySummary")
    if not isinstance(window_id, str) or not window_id.strip():
        raise DeterministicPipelineError("window_id must be non-empty")
    if isinstance(evaluated_at, bool) or not isinstance(evaluated_at, (int, float)) or not math.isfinite(float(evaluated_at)):
        raise DeterministicPipelineError("evaluated_at must be finite")
    if evaluated_at < route.updated_at:
        raise DeterministicPipelineError("canary evidence predates the current canary generation")
    if any(result.fingerprint != route.task_fingerprint for result in summary.results):
        raise DeterministicPipelineError("canary summary contains a different task fingerprint")

    stats = summary.route_stats.get(route.candidate_key)
    if stats is None:
        raise DeterministicPipelineError("canary summary does not contain the candidate route")
    attempted = stats.successes + stats.errors
    evaluator = promotion_evaluator or PromotionEvaluator()
    promotion = evaluator.evaluate(
        summary,
        candidate=route.candidate_key,
        baseline=baseline,
        projected_future_calls=projected_future_calls,
    )

    canary_policy = policy or DeterministicCanaryPolicy()
    reasons: list[str] = []
    if attempted < canary_policy.min_attempted_requests:
        reasons.append("insufficient_canary_attempts")
    if not promotion.promote:
        reasons.extend(f"promotion:{reason}" for reason in promotion.reasons)

    return DeterministicCanaryEvidence(
        activate=not reasons,
        reasons=tuple(reasons),
        route_id=route.route_id,
        route_generation=route.generation,
        candidate_key=route.candidate_key,
        fingerprint=route.task_fingerprint,
        traffic_fraction=route.traffic_fraction,
        window_id=window_id,
        evaluated_at=float(evaluated_at),
        attempted_requests=attempted,
        promotion=promotion,
    )


def _require_canary_evidence(
    route: RegisteredRoute,
    evidence: DeterministicCanaryEvidence,
) -> None:
    if not isinstance(evidence, DeterministicCanaryEvidence):
        raise DeterministicPipelineError("canary evidence must be DeterministicCanaryEvidence")
    if evidence.route_id != route.route_id:
        raise DeterministicPipelineError("canary evidence is for a different route")
    if evidence.route_generation != route.generation:
        raise DeterministicPipelineError("canary evidence is stale for the current route generation")
    if evidence.candidate_key != route.candidate_key or evidence.fingerprint != route.task_fingerprint:
        raise DeterministicPipelineError("canary evidence identity does not match route")
    if evidence.traffic_fraction != route.traffic_fraction:
        raise DeterministicPipelineError("canary evidence traffic fraction does not match route")
    if evidence.evaluated_at < route.updated_at:
        raise DeterministicPipelineError("canary evidence predates current route state")


def activate_deterministic_candidate(
    registry: RouteRegistry,
    route: RegisteredRoute,
    build: DeterministicCandidateBuild,
    evidence: DeterministicCanaryEvidence,
    *,
    fingerprint_audit: FingerprintAuditSummary,
    evidence_ref: str,
    now: float,
    replace_active_route_id: str | None = None,
    expected_active_generation: int | None = None,
) -> RegisteredRoute:
    """Activate only with passing evidence bound to the current canary generation."""
    if not isinstance(registry, RouteRegistry):
        raise DeterministicPipelineError("registry must be RouteRegistry")
    _require_route_matches_build(route, build)
    if route.state is not RouteState.CANARY:
        raise DeterministicPipelineError("route must be in canary state")
    _require_canary_evidence(route, evidence)
    if not evidence.activate:
        raise DeterministicPipelineError("canary evidence did not pass activation gates")
    return registry.activate(
        route.route_id,
        promotion=evidence.promotion,
        fingerprint_audit=fingerprint_audit,
        evidence_ref=evidence_ref,
        now=now,
        expected_generation=route.generation,
        replace_active_route_id=replace_active_route_id,
        expected_active_generation=expected_active_generation,
    )


def rollback_failed_deterministic_canary(
    registry: RouteRegistry,
    route: RegisteredRoute,
    evidence: DeterministicCanaryEvidence,
    *,
    evidence_ref: str,
    now: float,
) -> RegisteredRoute:
    """Rollback a canary only when bound evidence says activation gates did not pass."""
    if not isinstance(registry, RouteRegistry):
        raise DeterministicPipelineError("registry must be RouteRegistry")
    if route.state is not RouteState.CANARY:
        raise DeterministicPipelineError("route must be in canary state")
    _require_canary_evidence(route, evidence)
    if evidence.activate:
        raise DeterministicPipelineError("passing canary evidence cannot be used as failed-evidence rollback")
    return registry.rollback_canary(
        route.route_id,
        evidence_ref=evidence_ref,
        now=now,
        expected_generation=route.generation,
    )
