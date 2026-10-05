from __future__ import annotations

import pytest

from pcf.jit.deterministic import compile_deterministic_artifact
from pcf.jit.deterministic_canary import (
    DeterministicCanaryPolicy,
    activate_deterministic_candidate,
    evaluate_deterministic_canary,
    rollback_failed_deterministic_canary,
    start_deterministic_canary,
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


FINGERPRINT = "fingerprint-1"
CANDIDATE = "compiled:v1"
BASELINE = "frontier"


def _build() -> DeterministicCandidateBuild:
    artifact = compile_deterministic_artifact(
        name="normalize",
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
            breakeven_calls=2,
            projected_net_savings_usd=5.0,
        ),
    )


def _promotion(candidate: str = CANDIDATE) -> PromotionDecision:
    evidence = PromotionEvidence(
        candidate=candidate,
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


def _audit():
    audit = FingerprintAudit()
    audit.observe_fingerprint(
        FINGERPRINT,
        semantic_id="normalize",
        behavior_id="normalize.v1",
        observed_at=1.0,
    )
    return audit.summary()


def _canary_route():
    build = _build()
    audit = _audit()
    registry = RouteRegistry()
    route = register_deterministic_shadow_candidate(registry, build, now=1.0)
    route = registry.mark_eligible(
        route.route_id,
        promotion=_promotion(),
        fingerprint_audit=audit,
        evidence_ref="shadow-pass",
        now=2.0,
        expected_generation=route.generation,
    )
    route = start_deterministic_canary(
        registry,
        route,
        build,
        traffic_fraction=0.1,
        evidence_ref="start-canary",
        now=3.0,
    )
    return registry, build, audit, route


def _summary(build: DeterministicCandidateBuild, count: int):
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
        expected = request.payload["value"].strip().lower()
        return ShadowAssessment(expected_quality=1.0 if run.output == expected else 0.0)

    requests = [
        ShadowRequest(
            request_id=f"canary-{index}",
            fingerprint=FINGERPRINT,
            payload={"value": f" USER{index}@EXAMPLE.COM "},
            metadata={"authority_ok": True, "freshness_ok": True},
        )
        for index in range(count)
    ]
    return TraceReplayer(ShadowExecutor(assessor)).replay(requests, (candidate, baseline))


def _relaxed_evaluator() -> PromotionEvaluator:
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


def test_canary_evidence_activates_exact_bound_generation() -> None:
    registry, build, audit, route = _canary_route()
    evidence = evaluate_deterministic_canary(
        route,
        _summary(build, 5),
        baseline=BASELINE,
        projected_future_calls=1000,
        window_id="window-1",
        evaluated_at=4.0,
        promotion_evaluator=_relaxed_evaluator(),
        policy=DeterministicCanaryPolicy(min_attempted_requests=5),
    )
    assert evidence.activate is True
    assert evidence.route_generation == route.generation
    active = activate_deterministic_candidate(
        registry,
        route,
        build,
        evidence,
        fingerprint_audit=audit,
        evidence_ref="canary-pass-window-1",
        now=5.0,
    )
    assert active.state is RouteState.ACTIVE
    assert active.traffic_fraction == 1.0


def test_old_canary_evidence_cannot_activate_restarted_canary() -> None:
    registry, build, audit, route = _canary_route()
    evidence = evaluate_deterministic_canary(
        route,
        _summary(build, 5),
        baseline=BASELINE,
        projected_future_calls=1000,
        window_id="window-old",
        evaluated_at=4.0,
        promotion_evaluator=_relaxed_evaluator(),
        policy=DeterministicCanaryPolicy(min_attempted_requests=5),
    )
    eligible = registry.rollback_canary(
        route.route_id,
        evidence_ref="restart",
        now=5.0,
        expected_generation=route.generation,
    )
    restarted = start_deterministic_canary(
        registry,
        eligible,
        build,
        traffic_fraction=0.1,
        evidence_ref="restart-canary",
        now=6.0,
    )
    with pytest.raises(DeterministicPipelineError, match="stale"):
        activate_deterministic_candidate(
            registry,
            restarted,
            build,
            evidence,
            fingerprint_audit=audit,
            evidence_ref="must-not-activate",
            now=7.0,
        )


def test_insufficient_canary_exposure_rolls_back_instead_of_activating() -> None:
    registry, build, _, route = _canary_route()
    evidence = evaluate_deterministic_canary(
        route,
        _summary(build, 1),
        baseline=BASELINE,
        projected_future_calls=1000,
        window_id="window-small",
        evaluated_at=4.0,
        promotion_evaluator=PromotionEvaluator(
            PromotionPolicy(
                min_matched_samples=1,
                min_success_rate=0.0,
                min_admissible_rate=0.0,
                max_quality_regression=1.0,
                max_mean_htokens=1.0,
                max_htoken_regression=1.0,
                min_risk_adjusted_savings_usd_per_call=0.0,
                require_projection=False,
            )
        ),
        policy=DeterministicCanaryPolicy(min_attempted_requests=5),
    )
    assert evidence.activate is False
    assert "insufficient_canary_attempts" in evidence.reasons
    rolled_back = rollback_failed_deterministic_canary(
        registry,
        route,
        evidence,
        evidence_ref="canary-insufficient",
        now=5.0,
    )
    assert rolled_back.state is RouteState.ELIGIBLE


def test_canary_evidence_cannot_predate_canary_start() -> None:
    _, build, _, route = _canary_route()
    with pytest.raises(DeterministicPipelineError, match="predates"):
        evaluate_deterministic_canary(
            route,
            _summary(build, 5),
            baseline=BASELINE,
            projected_future_calls=1000,
            window_id="old-window",
            evaluated_at=2.5,
            promotion_evaluator=_relaxed_evaluator(),
            policy=DeterministicCanaryPolicy(min_attempted_requests=5),
        )
