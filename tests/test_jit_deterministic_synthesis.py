from __future__ import annotations

import pytest

from pcf.jit.deterministic_pipeline import (
    DeterministicBuildPolicy,
    DeterministicPipelineError,
    DeterministicValidationExample,
    execute_verified_deterministic_artifact,
)
from pcf.jit.deterministic_synthesis import (
    SYNTHESIS_VERSION,
    DeterministicSynthesisError,
    DeterministicSynthesisPolicy,
    synthesize_and_prepare_deterministic_candidate,
    synthesize_deterministic_expression,
)
from pcf.jit.fingerprint import FingerprintAudit, TaskSignature
from pcf.jit.hotpath import CompileEconomics, HotPathDetector


def _example(case_id: str, email: str, expected: str) -> DeterministicValidationExample:
    return DeterministicValidationExample(
        case_id=case_id,
        inputs={"customer": {"email": email}},
        expected_output={"email": expected},
    )


def _hot_path():
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


def test_synthesizes_unique_strip_lower_object_projection() -> None:
    training = [
        _example("t1", " A@EXAMPLE.COM ", "a@example.com"),
        _example("t2", "B@Example.COM  ", "b@example.com"),
        _example("t3", "  C@example.com", "c@example.com"),
    ]
    result = synthesize_deterministic_expression(
        training,
        policy=DeterministicSynthesisPolicy(min_training_examples=3),
    )
    assert result.expression == {
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
    assert result.evidence.training_examples == 3
    assert result.evidence.synthesis_version == SYNTHESIS_VERSION


def test_synthesis_fails_on_equally_simple_ambiguous_paths() -> None:
    examples = [
        DeterministicValidationExample(
            case_id=f"a{index}",
            inputs={"left": value, "right": value},
            expected_output=value,
        )
        for index, value in enumerate(("x", "y", "z"), start=1)
    ]
    with pytest.raises(DeterministicSynthesisError, match="ambiguous"):
        synthesize_deterministic_expression(
            examples,
            policy=DeterministicSynthesisPolicy(min_training_examples=3),
        )


def test_synthesis_rejects_unsupported_transform_instead_of_inventing_code() -> None:
    examples = [
        DeterministicValidationExample(
            case_id=f"r{index}",
            inputs={"value": value},
            expected_output=value[::-1],
        )
        for index, value in enumerate(("abc", "def", "ghi"), start=1)
    ]
    with pytest.raises(DeterministicSynthesisError, match="no supported"):
        synthesize_deterministic_expression(
            examples,
            policy=DeterministicSynthesisPolicy(min_training_examples=3),
        )


def test_constants_are_disabled_by_default() -> None:
    examples = [
        DeterministicValidationExample(
            case_id=f"c{index}",
            inputs={"value": index},
            expected_output="fixed",
        )
        for index in range(3)
    ]
    with pytest.raises(DeterministicSynthesisError, match="no supported"):
        synthesize_deterministic_expression(
            examples,
            policy=DeterministicSynthesisPolicy(min_training_examples=3),
        )


def test_synthesize_then_prepare_uses_separate_held_out_validation() -> None:
    signature, detector, audit = _hot_path()
    training = [
        _example("t1", " A@EXAMPLE.COM ", "a@example.com"),
        _example("t2", "B@Example.COM  ", "b@example.com"),
        _example("t3", "  C@example.com", "c@example.com"),
    ]
    validation = [
        _example("v1", " D@EXAMPLE.COM ", "d@example.com"),
        _example("v2", "E@Example.COM  ", "e@example.com"),
        _example("v3", "  F@example.com", "f@example.com"),
    ]
    build, evidence = synthesize_and_prepare_deterministic_candidate(
        candidate_key="normalize_customer:compiled:v1",
        artifact_name="normalize_customer",
        artifact_version="1",
        fingerprint=signature.fingerprint,
        detector=detector,
        audit=audit,
        economics=CompileEconomics(0.01, 0.001, 0.01),
        projected_future_calls=100,
        training_examples=training,
        validation_examples=validation,
        synthesis_policy=DeterministicSynthesisPolicy(min_training_examples=3),
        build_policy=DeterministicBuildPolicy(min_validation_examples=3),
    )
    assert dict(build.artifact.dependencies)["synthesizer"] == SYNTHESIS_VERSION
    assert evidence.training_examples == 3
    assert execute_verified_deterministic_artifact(
        build.artifact,
        {"customer": {"email": " NEW@EXAMPLE.COM "}},
    ) == {"email": "new@example.com"}


def test_held_out_validation_rejects_training_only_constant_rule() -> None:
    signature, detector, audit = _hot_path()
    training = [
        DeterministicValidationExample(
            case_id=f"t{index}",
            inputs={"value": index},
            expected_output="fixed",
        )
        for index in range(3)
    ]
    validation = [
        DeterministicValidationExample(
            case_id=f"v{index}",
            inputs={"value": index + 10},
            expected_output="different",
        )
        for index in range(3)
    ]
    with pytest.raises(DeterministicPipelineError, match="validation mismatch"):
        synthesize_and_prepare_deterministic_candidate(
            candidate_key="bad-constant",
            artifact_name="bad-constant",
            artifact_version="1",
            fingerprint=signature.fingerprint,
            detector=detector,
            audit=audit,
            economics=CompileEconomics(0.01, 0.001, 0.01),
            projected_future_calls=100,
            training_examples=training,
            validation_examples=validation,
            synthesis_policy=DeterministicSynthesisPolicy(
                min_training_examples=3,
                allow_constants=True,
            ),
            build_policy=DeterministicBuildPolicy(min_validation_examples=3),
        )
