from __future__ import annotations

from pcf.jit.deterministic_canary import (
    DeterministicCanaryPolicy,
    activate_deterministic_candidate,
    evaluate_deterministic_canary,
    start_deterministic_canary,
)
from pcf.jit.deterministic_monitor import (
    DeterministicMonitorPolicy,
    demote_unhealthy_deterministic_active,
    evaluate_deterministic_active_health,
)
from pcf.jit.deterministic_pipeline import (
    DeterministicBuildPolicy,
    DeterministicValidationExample,
    evaluate_deterministic_shadow,
    execute_verified_deterministic_artifact,
    make_deterministic_shadow_target,
    mark_deterministic_candidate_eligible,
    register_deterministic_shadow_candidate,
)
from pcf.jit.deterministic_store import (
    SQLiteDeterministicArtifactStore,
    resolve_active_deterministic_artifact,
)
from pcf.jit.deterministic_synthesis import (
    DeterministicSynthesisPolicy,
    synthesize_and_prepare_deterministic_candidate,
)
from pcf.jit.fingerprint import FingerprintAudit, TaskSignature
from pcf.jit.hotpath import CompileEconomics, HotPathDetector
from pcf.jit.promotion import PromotionEvaluator, PromotionPolicy
from pcf.jit.registry import RouteState
from pcf.jit.shadow import ShadowAssessment, ShadowExecutor, ShadowRequest, ShadowRun, ShadowTarget, TraceReplayer
from pcf.jit.sqlite_registry import SQLiteRouteRegistry
from pcf.jit.types import ExecutionPrimitive


BASELINE = "frontier-e2e"


def _example(case_id: str, email: str) -> DeterministicValidationExample:
    return DeterministicValidationExample(
        case_id=case_id,
        inputs={"customer": {"email": email}},
        expected_output={"email": email.strip().lower()},
    )


def _promotion_evaluator() -> PromotionEvaluator:
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


def _baseline() -> ShadowTarget:
    return ShadowTarget(
        name=BASELINE,
        primitive=ExecutionPrimitive.FRONTIER_MODEL,
        shadow_safe=True,
        runner=lambda request: ShadowRun(
            output={"email": request.payload["customer"]["email"].strip().lower()},
            token_cost_usd=0.01,
            authority_ok=True,
            freshness_ok=True,
        ),
    )


def _requests(fingerprint: str, prefix: str, count: int) -> list[ShadowRequest]:
    return [
        ShadowRequest(
            request_id=f"{prefix}-{index}",
            fingerprint=fingerprint,
            payload={"customer": {"email": f" USER{index}@EXAMPLE.COM "}},
            metadata={"authority_ok": True, "freshness_ok": True},
        )
        for index in range(count)
    ]


def _replay_with_quality(build, *, count: int, candidate_quality: float):
    candidate = make_deterministic_shadow_target(build)
    baseline = _baseline()

    def assessor(request, target, run):
        expected = {"email": request.payload["customer"]["email"].strip().lower()}
        quality = 1.0 if run.output == expected else 0.0
        if target.name == build.candidate_key:
            quality = min(quality, candidate_quality)
        return ShadowAssessment(expected_quality=quality)

    return TraceReplayer(ShadowExecutor(assessor)).replay(
        _requests(build.fingerprint, "replay", count),
        (candidate, baseline),
    )


def test_deterministic_jit_full_reference_lifecycle_survives_restarts(tmp_path) -> None:
    signature = TaskSignature(
        operation="normalize_customer_email",
        input_schema_id="customer.v1",
        output_schema_id="normalized-customer.v1",
        policy_version="policy.v1",
        authority_scope="customer.read",
    )
    detector = HotPathDetector(min_calls=3, min_stability=1.0, min_audit_observations=3)
    audit = FingerprintAudit()
    for index in range(3):
        detector.observe(signature.fingerprint, stable=True)
        audit.observe(
            signature,
            semantic_id="normalize_customer_email",
            behavior_id="normalize_customer_email.v1",
            observed_at=float(index + 1),
        )
    audit_summary = audit.summary()

    training = [
        _example("train-1", " A@EXAMPLE.COM "),
        _example("train-2", "B@Example.COM  "),
        _example("train-3", "  C@example.com"),
    ]
    validation = [
        _example("validation-1", " D@EXAMPLE.COM "),
        _example("validation-2", "E@Example.COM  "),
        _example("validation-3", "  F@example.com"),
    ]
    build, synthesis = synthesize_and_prepare_deterministic_candidate(
        candidate_key="normalize_customer_email:compiled:v1",
        artifact_name="normalize_customer_email",
        artifact_version="1",
        fingerprint=signature.fingerprint,
        detector=detector,
        audit=audit_summary,
        economics=CompileEconomics(
            general_cost_per_call_usd=0.01,
            compiled_cost_per_call_usd=0.001,
            build_cost_usd=0.01,
        ),
        projected_future_calls=1000,
        training_examples=training,
        validation_examples=validation,
        synthesis_policy=DeterministicSynthesisPolicy(min_training_examples=3),
        build_policy=DeterministicBuildPolicy(min_validation_examples=3),
    )
    assert synthesis.training_examples == 3
    assert build.evidence.validation_matches == 3

    artifact_path = tmp_path / "jit-artifacts.sqlite"
    route_path = tmp_path / "jit-routes.sqlite"
    artifact_store = SQLiteDeterministicArtifactStore(str(artifact_path))
    artifact_store.put(build.artifact, created_at=10.0)
    registry = SQLiteRouteRegistry(route_path)
    route = register_deterministic_shadow_candidate(registry, build, now=11.0)
    assert route.state is RouteState.SHADOW

    shadow = evaluate_deterministic_shadow(
        build,
        requests=_requests(build.fingerprint, "shadow", 5),
        baseline=_baseline(),
        assessor=lambda request, target, run: ShadowAssessment(
            expected_quality=1.0
            if run.output == {"email": request.payload["customer"]["email"].strip().lower()}
            else 0.0
        ),
        projected_future_calls=1000,
        promotion_evaluator=_promotion_evaluator(),
    )
    assert shadow.promotion.promote is True
    route = mark_deterministic_candidate_eligible(
        registry,
        route,
        build,
        shadow,
        fingerprint_audit=audit_summary,
        evidence_ref="shadow-pass-e2e",
        now=12.0,
    )
    assert route.state is RouteState.ELIGIBLE

    route = start_deterministic_canary(
        registry,
        route,
        build,
        traffic_fraction=0.1,
        evidence_ref="canary-start-e2e",
        now=13.0,
    )
    assert route.state is RouteState.CANARY

    canary_summary = _replay_with_quality(build, count=5, candidate_quality=1.0)
    canary_evidence = evaluate_deterministic_canary(
        route,
        canary_summary,
        baseline=BASELINE,
        projected_future_calls=1000,
        window_id="canary-window-e2e",
        evaluated_at=14.0,
        promotion_evaluator=_promotion_evaluator(),
        policy=DeterministicCanaryPolicy(min_attempted_requests=5),
    )
    assert canary_evidence.activate is True
    route = activate_deterministic_candidate(
        registry,
        route,
        build,
        canary_evidence,
        fingerprint_audit=audit_summary,
        evidence_ref="canary-pass-e2e",
        now=15.0,
    )
    assert route.state is RouteState.ACTIVE

    del registry
    del artifact_store
    restarted_registry = SQLiteRouteRegistry(route_path)
    restarted_store = SQLiteDeterministicArtifactStore(str(artifact_path))
    active_route, active_artifact = resolve_active_deterministic_artifact(
        restarted_registry,
        restarted_store,
        build.fingerprint,
    )
    assert active_route.route_id == route.route_id
    assert active_artifact.artifact_sha256 == build.artifact.artifact_sha256
    assert execute_verified_deterministic_artifact(
        active_artifact,
        {"customer": {"email": " RESTART@EXAMPLE.COM "}},
    ) == {"email": "restart@example.com"}

    bad_health_summary = _replay_with_quality(build, count=5, candidate_quality=0.0)
    monitor_policy = DeterministicMonitorPolicy(
        min_attempted_requests=5,
        demote_traffic_fraction=0.05,
    )
    health = evaluate_deterministic_active_health(
        active_route,
        build,
        bad_health_summary,
        baseline=BASELINE,
        projected_future_calls=1000,
        window_id="monitor-regression-e2e",
        evaluated_at=16.0,
        promotion_evaluator=_promotion_evaluator(),
        policy=monitor_policy,
    )
    assert health.demote is True
    demoted = demote_unhealthy_deterministic_active(
        restarted_registry,
        active_route,
        build,
        health,
        evidence_ref="monitor-regression-e2e",
        now=17.0,
        policy=monitor_policy,
    )
    assert demoted.state is RouteState.CANARY
    assert demoted.traffic_fraction == 0.05

    del restarted_registry
    final_registry = SQLiteRouteRegistry(route_path)
    assert final_registry.active(build.fingerprint) is None
    persisted = final_registry.get(demoted.route_id)
    assert persisted.state is RouteState.CANARY
    assert persisted.generation == demoted.generation
    assert persisted.traffic_fraction == 0.05
