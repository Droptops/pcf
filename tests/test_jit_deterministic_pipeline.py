from __future__ import annotations

import pytest

from pcf.jit.deterministic import DeterministicExecutionError, compile_deterministic_artifact
from pcf.jit.deterministic_pipeline import (
    DeterministicBuildPolicy,
    DeterministicPipelineError,
    DeterministicValidationExample,
    evaluate_deterministic_shadow,
    execute_verified_deterministic_artifact,
    mark_deterministic_candidate_eligible,
    prepare_deterministic_candidate,
    register_deterministic_shadow_candidate,
)
from pcf.jit.fingerprint import FingerprintAudit, TaskSignature
from pcf.jit.hotpath import CompileEconomics, HotPathDetector
from pcf.jit.promotion import PromotionEvaluator, PromotionPolicy
from pcf.jit.registry import RouteRegistry, RouteState
from pcf.jit.shadow import ShadowAssessment, ShadowRequest, ShadowRun, ShadowTarget
from pcf.jit.types import ExecutionPrimitive


def _ready_hot_path():
    signature = TaskSignature(
        operation="normalize_customer",
        input_schema_id="customer.v1",
        output_schema_id="normalized.v1",
        policy_version="policy.v1",
        authority_scope="customer.read",
    )
    detector = HotPathDetector(min_calls=3, min_stability=1.0, min_audit_observations=3)
    audit = FingerprintAudit()
    for index in range(3):
        detector.observe(signature.fingerprint, stable=True)
        audit.observe(
            signature,
            semantic_id="normalize_customer",
            behavior_id="normalize_customer.v1",
            observed_at=float(index + 1),
        )
    return signature, detector, audit.summary()


def _examples():
    return [
        DeterministicValidationExample(
            case_id=f"case-{index}",
            inputs={"customer": {"email": email}},
            expected_output={"email": email.strip().lower()},
        )
        for index, email in enumerate(
            [" A@EXAMPLE.COM ", "B@Example.com", "  c@example.com"],
            start=1,
        )
    ]


def _expression():
    return {
        "op": "object",
        "fields": {
            "email": {
                "op": "lower",
                "arg": {
                    "op": "strip",
                    "arg": {"op": "input", "path": "customer.email"},
                },
            }
        },
    }


def _build():
    signature, detector, audit = _ready_hot_path()
    build = prepare_deterministic_candidate(
        candidate_key="normalize_customer:deterministic:v1",
        artifact_name="normalize_customer",
        artifact_version="1",
        expression=_expression(),
        fingerprint=signature.fingerprint,
        detector=detector,
        audit=audit,
        economics=CompileEconomics(
            general_cost_per_call_usd=0.01,
            compiled_cost_per_call_usd=0.001,
            build_cost_usd=0.01,
        ),
        projected_future_calls=100,
        validation_examples=_examples(),
        build_policy=DeterministicBuildPolicy(min_validation_examples=3),
    )
    return build, audit


def test_prepare_requires_hot_path_audit_roi_and_exact_validation() -> None:
    build, _ = _build()
    assert build.evidence.validation_matches == 3
    assert build.evidence.projected_net_savings_usd == pytest.approx(0.89)
    assert build.artifact.artifact_sha256


def test_prepare_rejects_validation_mismatch() -> None:
    signature, detector, audit = _ready_hot_path()
    examples = _examples()
    examples[-1] = DeterministicValidationExample(
        case_id="bad",
        inputs={"customer": {"email": "C@EXAMPLE.COM"}},
        expected_output={"email": "wrong"},
    )
    with pytest.raises(DeterministicPipelineError, match="validation mismatch"):
        prepare_deterministic_candidate(
            candidate_key="candidate",
            artifact_name="normalize_customer",
            artifact_version="1",
            expression=_expression(),
            fingerprint=signature.fingerprint,
            detector=detector,
            audit=audit,
            economics=CompileEconomics(0.01, 0.001, 0.01),
            projected_future_calls=100,
            validation_examples=examples,
            build_policy=DeterministicBuildPolicy(min_validation_examples=3),
        )


def test_verified_execution_detects_post_compile_mutation() -> None:
    artifact = compile_deterministic_artifact(
        name="mutable_surface",
        version="1",
        expression={"op": "input", "path": "value"},
    )
    artifact.expression["path"] = "other"
    with pytest.raises(DeterministicExecutionError, match="integrity validation failed"):
        execute_verified_deterministic_artifact(artifact, {"value": 1, "other": 2})


def test_shadow_runner_does_not_self_authorize() -> None:
    build, _ = _build()

    baseline = ShadowTarget(
        name="frontier",
        primitive=ExecutionPrimitive.FRONTIER_MODEL,
        shadow_safe=True,
        runner=lambda request: ShadowRun(output={"email": request.payload["customer"]["email"].strip().lower()}, token_cost_usd=0.01),
    )

    def assessor(request, target, run):
        return ShadowAssessment(expected_quality=1.0)

    evaluation = evaluate_deterministic_shadow(
        build,
        requests=[
            ShadowRequest(
                request_id="missing-authority",
                fingerprint=build.fingerprint,
                payload={"customer": {"email": " X@EXAMPLE.COM "}},
                metadata={},
            )
        ],
        baseline=baseline,
        assessor=assessor,
        projected_future_calls=100,
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
    )
    candidate = evaluation.summary.results[0].observations[0].candidate
    assert candidate is not None
    assert candidate.authority_ok is False
    assert candidate.freshness_ok is False


def test_end_to_end_shadow_evidence_marks_route_eligible() -> None:
    build, audit = _build()
    registry = RouteRegistry()
    route = register_deterministic_shadow_candidate(registry, build, now=10.0)
    assert route.state is RouteState.SHADOW
    assert "validation_examples" in route.metadata
    assert "inputs" not in route.metadata

    baseline = ShadowTarget(
        name="frontier",
        primitive=ExecutionPrimitive.FRONTIER_MODEL,
        shadow_safe=True,
        runner=lambda request: ShadowRun(
            output={"email": request.payload["customer"]["email"].strip().lower()},
            token_cost_usd=0.01,
            authority_ok=True,
            freshness_ok=True,
        ),
    )

    def assessor(request, target, run):
        expected = {"email": request.payload["customer"]["email"].strip().lower()}
        return ShadowAssessment(expected_quality=1.0 if run.output == expected else 0.0)

    requests = [
        ShadowRequest(
            request_id=f"shadow-{index}",
            fingerprint=build.fingerprint,
            payload={"customer": {"email": f" USER{index}@EXAMPLE.COM "}},
            metadata={"authority_ok": True, "freshness_ok": True},
        )
        for index in range(5)
    ]
    evaluation = evaluate_deterministic_shadow(
        build,
        requests=requests,
        baseline=baseline,
        assessor=assessor,
        projected_future_calls=100,
        promotion_evaluator=PromotionEvaluator(
            PromotionPolicy(
                min_matched_samples=5,
                min_success_rate=0.5,
                min_admissible_rate=0.5,
                max_quality_regression=0.0,
                max_mean_htokens=0.0,
                max_htoken_regression=0.0,
                min_risk_adjusted_savings_usd_per_call=0.0,
                promotion_fixed_cost_usd=0.0,
                min_projected_net_savings_usd=0.0,
                require_projection=True,
            )
        ),
    )
    assert evaluation.promotion.promote is True
    eligible = mark_deterministic_candidate_eligible(
        registry,
        route,
        build,
        evaluation,
        fingerprint_audit=audit,
        evidence_ref="shadow-eval-1",
        now=20.0,
    )
    assert eligible.state is RouteState.ELIGIBLE
    assert eligible.generation == 1
