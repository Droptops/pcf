"""Generation-bound monitoring and demotion for active deterministic JIT routes."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .deterministic_pipeline import DeterministicCandidateBuild, DeterministicPipelineError
from .promotion import PromotionDecision, PromotionEvaluator
from .registry import RegisteredRoute, RouteRegistry, RouteState
from .shadow import TraceReplaySummary


@dataclass(frozen=True)
class DeterministicMonitorPolicy:
    min_attempted_requests: int = 100
    demote_traffic_fraction: float = 0.1

    def __post_init__(self) -> None:
        if type(self.min_attempted_requests) is not int or self.min_attempted_requests <= 0:
            raise ValueError("min_attempted_requests must be a positive integer")
        if (
            isinstance(self.demote_traffic_fraction, bool)
            or not isinstance(self.demote_traffic_fraction, (int, float))
            or not math.isfinite(float(self.demote_traffic_fraction))
            or not 0 < self.demote_traffic_fraction < 1
        ):
            raise ValueError("demote_traffic_fraction must be finite and between 0 and 1")


@dataclass(frozen=True)
class DeterministicHealthEvidence:
    healthy: bool
    reasons: tuple[str, ...]
    route_id: str
    route_generation: int
    candidate_key: str
    fingerprint: str
    window_id: str
    evaluated_at: float
    attempted_requests: int
    promotion: PromotionDecision


def _require_active_route_matches_build(route: RegisteredRoute, build: DeterministicCandidateBuild) -> None:
    if not isinstance(route, RegisteredRoute) or route.state is not RouteState.ACTIVE:
        raise DeterministicPipelineError("route must be active")
    if not isinstance(build, DeterministicCandidateBuild):
        raise DeterministicPipelineError("build must be DeterministicCandidateBuild")
    if route.candidate_key != build.candidate_key or route.task_fingerprint != build.fingerprint:
        raise DeterministicPipelineError("active route does not match deterministic build")
    if route.metadata.get("artifact_sha256") != build.artifact.artifact_sha256:
        raise DeterministicPipelineError("active route artifact identity does not match build")


def evaluate_deterministic_active_health(
    route: RegisteredRoute,
    build: DeterministicCandidateBuild,
    summary: TraceReplaySummary,
    *,
    baseline: str,
    projected_future_calls: int | None,
    window_id: str,
    evaluated_at: float,
    promotion_evaluator: PromotionEvaluator | None = None,
    policy: DeterministicMonitorPolicy | None = None,
) -> DeterministicHealthEvidence:
    """Evaluate fresh paired telemetry for the exact current ACTIVE generation."""
    _require_active_route_matches_build(route, build)
    if not isinstance(summary, TraceReplaySummary):
        raise DeterministicPipelineError("summary must be TraceReplaySummary")
    if not isinstance(window_id, str) or not window_id.strip():
        raise DeterministicPipelineError("window_id must be non-empty")
    if (
        isinstance(evaluated_at, bool)
        or not isinstance(evaluated_at, (int, float))
        or not math.isfinite(float(evaluated_at))
    ):
        raise DeterministicPipelineError("evaluated_at must be finite")
    if evaluated_at < route.updated_at:
        raise DeterministicPipelineError("health evidence predates the current active generation")
    if any(result.fingerprint != route.task_fingerprint for result in summary.results):
        raise DeterministicPipelineError("health summary contains a different task fingerprint")

    stats = summary.route_stats.get(route.candidate_key)
    if stats is None:
        raise DeterministicPipelineError("health summary does not contain the active route")
    attempted = stats.successes + stats.errors
    evaluator = promotion_evaluator or PromotionEvaluator()
    promotion = evaluator.evaluate(
        summary,
        candidate=route.candidate_key,
        baseline=baseline,
        projected_future_calls=projected_future_calls,
    )
    monitor_policy = policy or DeterministicMonitorPolicy()
    reasons: list[str] = []
    if attempted < monitor_policy.min_attempted_requests:
        reasons.append("insufficient_monitor_attempts")
    if not promotion.promote:
        reasons.extend(f"health:{reason}" for reason in promotion.reasons)

    return DeterministicHealthEvidence(
        healthy=not reasons,
        reasons=tuple(reasons),
        route_id=route.route_id,
        route_generation=route.generation,
        candidate_key=route.candidate_key,
        fingerprint=route.task_fingerprint,
        window_id=window_id,
        evaluated_at=float(evaluated_at),
        attempted_requests=attempted,
        promotion=promotion,
    )


def _require_health_evidence(route: RegisteredRoute, evidence: DeterministicHealthEvidence) -> None:
    if not isinstance(evidence, DeterministicHealthEvidence):
        raise DeterministicPipelineError("health evidence must be DeterministicHealthEvidence")
    if evidence.route_id != route.route_id:
        raise DeterministicPipelineError("health evidence is for a different route")
    if evidence.route_generation != route.generation:
        raise DeterministicPipelineError("health evidence is stale for the current active generation")
    if evidence.candidate_key != route.candidate_key or evidence.fingerprint != route.task_fingerprint:
        raise DeterministicPipelineError("health evidence identity does not match route")
    if evidence.evaluated_at < route.updated_at:
        raise DeterministicPipelineError("health evidence predates current route state")


def demote_unhealthy_deterministic_active(
    registry: RouteRegistry,
    route: RegisteredRoute,
    build: DeterministicCandidateBuild,
    evidence: DeterministicHealthEvidence,
    *,
    evidence_ref: str,
    now: float,
    policy: DeterministicMonitorPolicy | None = None,
) -> RegisteredRoute:
    """Demote ACTIVE -> CANARY only with failing evidence bound to this generation."""
    if not isinstance(registry, RouteRegistry):
        raise DeterministicPipelineError("registry must be RouteRegistry")
    _require_active_route_matches_build(route, build)
    _require_health_evidence(route, evidence)
    if evidence.healthy:
        raise DeterministicPipelineError("healthy evidence cannot demote an active route")
    monitor_policy = policy or DeterministicMonitorPolicy()
    return registry.demote_active(
        route.route_id,
        traffic_fraction=monitor_policy.demote_traffic_fraction,
        evidence_ref=evidence_ref,
        now=now,
        expected_generation=route.generation,
    )
