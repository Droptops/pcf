from __future__ import annotations

from dataclasses import replace

import pytest

from pcf.jit.deterministic import compile_deterministic_artifact
from pcf.jit.deterministic_monitor import (
    DeterministicMonitorPolicy,
    demote_unhealthy_deterministic_active,
    evaluate_deterministic_active_health,
)
from pcf.jit.deterministic_pipeline import (
    DeterministicBuildEvidence,
    DeterministicCandidateBuild,
    DeterministicPipelineError,
    make_deterministic_shadow_target,
    register_deterministic_shadow_candidate,
)
from pcf.jit.fingerprint import FingerprintAudit
from pcf.jit.promotion import PromotionDecision, PromotionEvidence, PromotionEvaluator, PromotionPolicy
from pcf.jit.registry import RouteRegistry, RouteState
from pcf.jit.shadow import ShadowAssessment, ShadowExecutor, ShadowRequest, ShadowRun, ShadowTarget, TraceReplayer
from pcf.jit.types import ExecutionPrimitive


FINGERPRINT = "fingerprint-monitor"
CANDIDATE = "compiled-monitor:v1"
BASELINE = "frontier-monitor"


def _build() -> DeterministicCandidateBuild:
    artifact = compile_deterministic_artifact(
        name="normalize_monitor",
        version="1",
        expression={"op": "lower", "arg": {"op": "strip", "arg": {"op": "input", "path": "value"}}},
    )
    return DeterministicCandidateBuild(
        candidate_key=CANDIDATE,
        fingerprint=FINGERPRINT,
        artifact=artifact,
        evidence=DeterministicBuildEvidence(
            fingerprint=FINGERPRINT,
            calls=100,
            stability=1.0,
            audit_observations=20,
            validation_examples=20,
            validation_matches=20,
            projected_future_calls=1000,
            breakeven_calls=2.0,
            projected_net_savings_usd=5.0,
        ),
    )


def _audit():
    audit = FingerprintAudit()
    audit.observe_fingerprint(
        FINGERPRINT,
        semantic_id="normalize-monitor",
        behavior_id="normalize-monitor.v1",
        observed_at=1.0,
    )
    return audit.summary()


def _passing_promotion() -> PromotionDecision:
    evidence = PromotionEvidence(
        candidate=CANDIDATE,
        baseline=BASELINE,
        matched_samples=100,
        candidate_successes=100,
        candidate_errors=0,
        candidate_skipped=0,
        success_rate=1.0,
        success_rate_lower_bound=1.0,
        admissible_rate=1.0,
        admissible_rate_lower_bound=1.0,
        mean_quality_delta=0.0,
        quality_delta_lower_bound=0.0,
        mean_candidate_htokens=0.0,
        mean_htoken_delta=0.0,
        htoken_delta_upper_bound=0.0,
        mean_risk_adjusted_savings_usd_per_call=0.01,
        projected_future_calls=1000,
        projected_net_savings_usd=10.0,
    )
    return PromotionDecision(promote=True, reasons=(), evidence=evidence)


def _active_route():
    build = _build()
    registry = RouteRegistry()
    audit = _audit()
    route = register_deterministic_shadow_candidate(registry, build, now=1.0)
    route = registry.mark_eligible(
        route.route_id,
        promotion=_passing_promotion(),
        fingerprint_audit=audit,
        evidence_ref="shadow-pass",
        now=2.0,
        expected_generation=route.generation,
    )
    route = registry.start_canary(
        route.route_id,
        traffic_fraction=0.1,
        evidence_ref="canary-start",
        now=3.0,
        expected_generation=route.generation,
    )
    route = registry.activate(
        route.route_id,
        promotion=_passing_promotion(),
        fingerprint_audit=audit,
        evidence_ref="canary-pass",
        now=4.0,
        expected_generation=route.generation,
    )
    return registry, build, route


def _summary(build: DeterministicCandidateBuild, count: int, *, candidate_quality: float):
    candidate = make_deterministic_shadow_target(build)
    baseline = ShadowTarget(
        name=BASELINE,
        primitive=ExecutionPrimitive.FRONTIER_MODEL,
        shadow_safe=True,
        runner=lambda request: ShadowRun(
            output=request.payload["value"].strip().lower(),
            token_cost_usd=0.01,
            authority_ok=True,
            freshness_ok=True,
        ),
    )

    def assessor(request, target, run):
        return ShadowAssessment(
            expected_quality=candidate_quality if target.name == CANDIDATE else 1.0
        )

    requests = [
        ShadowRequest(
            request_id=f"monitor-{index}",
            fingerprint=FINGERPRINT,
            payload={"value": f" USER{index}@EXAMPLE.COM "},
            metadata={"authority_ok": True, "freshness_ok": True},
        )
        for index in range(count)
    ]
    return TraceReplayer(ShadowExecutor(assessor)).replay(requests, (candidate, baseline))


def _health_evaluator() -> PromotionEvaluator:
    return PromotionEvaluator(
        PromotionPolicy(
            min_matched_samples=5,
            min_success_rate=0.5,
            min_admissible_rate=0.5,
            max_quality_regression=0.0,
            max_mean_htokens=0.0,
            max_htoken_regression=0.0,
            min_risk_adjusted_savings_usd_per_call=0.0,
            min_projected_net_savings_usd=0.0,
            require_projection=True,
        )
    )


def test_healthy_active_window_is_retained() -> None:
    _, build, route = _active_route()
    evidence = evaluate_deterministic_active_health(
        route,
        build,
        _summary(build, 5, candidate_quality=1.0),
        baseline=BASELINE,
        projected_future_calls=1000,
        window_id="healthy-window",
        evaluated_at=5.0,
        promotion_evaluator=_health_evaluator(),
        policy=DeterministicMonitorPolicy(min_attempted_requests=5),
    )
    assert evidence.healthy is True
    assert evidence.demote is False
    assert evidence.reasons == ()


def test_quality_regression_produces_actionable_demotion() -> None:
    registry, build, route = _active_route()
    policy = DeterministicMonitorPolicy(min_attempted_requests=5, demote_traffic_fraction=0.05)
    evidence = evaluate_deterministic_active_health(
        route,
        build,
        _summary(build, 5, candidate_quality=0.0),
        baseline=BASELINE,
        projected_future_calls=1000,
        window_id="bad-window",
        evaluated_at=5.0,
        promotion_evaluator=_health_evaluator(),
        policy=policy,
    )
    assert evidence.healthy is False
    assert evidence.demote is True
    demoted = demote_unhealthy_deterministic_active(
        registry,
        route,
        build,
        evidence,
        evidence_ref="health-regression",
        now=6.0,
        policy=policy,
    )
    assert demoted.state is RouteState.CANARY
    assert demoted.traffic_fraction == pytest.approx(0.05)


def test_insufficient_monitor_window_holds_instead_of_demoting() -> None:
    registry, build, route = _active_route()
    policy = DeterministicMonitorPolicy(min_attempted_requests=5)
    evidence = evaluate_deterministic_active_health(
        route,
        build,
        _summary(build, 1, candidate_quality=0.0),
        baseline=BASELINE,
        projected_future_calls=1000,
        window_id="small-window",
        evaluated_at=5.0,
        promotion_evaluator=PromotionEvaluator(
            PromotionPolicy(
                min_matched_samples=1,
                min_success_rate=0.0,
                min_admissible_rate=0.0,
                max_quality_regression=0.0,
                max_mean_htokens=1.0,
                max_htoken_regression=1.0,
                min_risk_adjusted_savings_usd_per_call=0.0,
                require_projection=False,
            )
        ),
        policy=policy,
    )
    assert evidence.healthy is False
    assert evidence.demote is False
    assert evidence.reasons == ("insufficient_monitor_attempts",)
    with pytest.raises(DeterministicPipelineError, match="insufficient"):
        demote_unhealthy_deterministic_active(
            registry,
            route,
            build,
            evidence,
            evidence_ref="must-hold",
            now=6.0,
            policy=policy,
        )


def test_stale_health_evidence_cannot_demote_new_generation() -> None:
    registry, build, route = _active_route()
    policy = DeterministicMonitorPolicy(min_attempted_requests=5)
    evidence = evaluate_deterministic_active_health(
        route,
        build,
        _summary(build, 5, candidate_quality=0.0),
        baseline=BASELINE,
        projected_future_calls=1000,
        window_id="old-window",
        evaluated_at=5.0,
        promotion_evaluator=_health_evaluator(),
        policy=policy,
    )
    newer = replace(route, generation=route.generation + 1, updated_at=6.0)
    with pytest.raises(DeterministicPipelineError, match="stale"):
        demote_unhealthy_deterministic_active(
            registry,
            newer,
            build,
            evidence,
            evidence_ref="stale-health",
            now=7.0,
            policy=policy,
        )
