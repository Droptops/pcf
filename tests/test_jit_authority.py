from dataclasses import dataclass
from enum import StrEnum

import pytest

from pcf.jit import (
    AuthorityEvidenceError,
    ExecutionPrimitive,
    JITOptimizer,
    OptimizerPolicy,
    ShadowAssessment,
    ShadowExecutor,
    ShadowRequest,
    ShadowRun,
    ShadowTarget,
    apply_faar_authority,
    normalize_faar_authority_decision,
)


class Verdict(StrEnum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    DEFER = "DEFER"
    STOP = "STOP"


@dataclass(frozen=True)
class FaarDecisionShape:
    verdict: Verdict
    reason_codes: tuple[str, ...]
    layer: str = "authority"


def decision(verdict=Verdict.ALLOW, reasons=()):
    return FaarDecisionShape(verdict, tuple(reasons))


def test_only_explicit_allow_authorizes():
    assert normalize_faar_authority_decision(decision(Verdict.ALLOW)).authority_ok
    for verdict in (Verdict.DENY, Verdict.DEFER, Verdict.STOP):
        evidence = normalize_faar_authority_decision(decision(verdict, (f"AUTH_{verdict}",)))
        assert not evidence.authority_ok
        assert evidence.verdict == verdict.value


def test_mapping_shape_is_supported_without_faar_dependency():
    evidence = normalize_faar_authority_decision({
        "verdict": "ALLOW",
        "reason_codes": [],
        "layer": "authority",
    })
    assert evidence.authority_ok
    assert evidence.source == "faar"


def test_wrong_layer_unknown_verdict_and_missing_fields_fail_closed():
    with pytest.raises(AuthorityEvidenceError, match="authority layer"):
        normalize_faar_authority_decision({"verdict": "ALLOW", "reason_codes": (), "layer": "risk"})
    with pytest.raises(AuthorityEvidenceError, match="unsupported"):
        normalize_faar_authority_decision({"verdict": "MAYBE", "reason_codes": (), "layer": "authority"})
    with pytest.raises(AuthorityEvidenceError, match="missing reason_codes"):
        normalize_faar_authority_decision({"verdict": "ALLOW", "layer": "authority"})


def test_reason_codes_are_bounded_like_faar_public_contract():
    with pytest.raises(AuthorityEvidenceError, match="too many"):
        normalize_faar_authority_decision({
            "verdict": "DENY",
            "reason_codes": tuple(f"R{i}" for i in range(65)),
            "layer": "authority",
        })
    with pytest.raises(AuthorityEvidenceError, match="bounded"):
        normalize_faar_authority_decision({
            "verdict": "DENY",
            "reason_codes": ("x" * 257,),
            "layer": "authority",
        })


def test_apply_authority_preserves_cost_output_and_can_only_reduce_authority():
    run = ShadowRun(
        output="ok",
        token_cost_usd=0.02,
        memory_cost_usd=0.003,
        latency_ms=12,
        authority_ok=True,
        metadata={"trace": "t1"},
    )
    allowed = apply_faar_authority(run, decision(Verdict.ALLOW))
    denied = apply_faar_authority(run, decision(Verdict.DENY, ("AUTHORITY_STOP",)))
    already_denied = apply_faar_authority(
        ShadowRun(output="ok", authority_ok=False),
        decision(Verdict.ALLOW),
    )

    assert allowed.authority_ok
    assert not denied.authority_ok
    assert not already_denied.authority_ok
    assert denied.output == "ok"
    assert denied.token_cost_usd == 0.02
    assert denied.memory_cost_usd == 0.003
    assert denied.metadata["trace"] == "t1"
    assert denied.metadata["authority_verdict"] == "DENY"
    assert denied.metadata["authority_reason_codes"] == ("AUTHORITY_STOP",)


def test_reserved_authority_metadata_cannot_be_overwritten():
    run = ShadowRun(output="ok", metadata={"authority_verdict": "ALLOW"})
    with pytest.raises(AuthorityEvidenceError, match="collides"):
        apply_faar_authority(run, decision(Verdict.DENY))


def test_denied_faar_route_is_rejected_by_existing_jit_hard_gate_even_when_cheapest():
    def assessor(request, target, run):
        return ShadowAssessment(expected_quality=1.0)

    def denied_runner(request):
        return apply_faar_authority(
            ShadowRun(output="cheap", token_cost_usd=0.0001),
            decision(Verdict.DENY, ("AUTHORITY_POSTURE_NOT_EXECUTE",)),
        )

    def allowed_runner(request):
        return apply_faar_authority(
            ShadowRun(output="expensive", token_cost_usd=0.05),
            decision(Verdict.ALLOW),
        )

    executor = ShadowExecutor(
        assessor,
        optimizer=JITOptimizer(OptimizerPolicy(min_quality=0.9, max_htokens=1.0)),
    )
    result = executor.execute(
        ShadowRequest("req-1", "fp", {}),
        (
            ShadowTarget("denied", ExecutionPrimitive.SPECIALIST_MODEL, denied_runner, shadow_safe=True),
            ShadowTarget("allowed", ExecutionPrimitive.FRONTIER_MODEL, allowed_runner, shadow_safe=True),
        ),
    )

    assert result.decision is not None
    assert result.decision.chosen == "allowed"
    denied_report = next(report for report in result.decision.reports if report.name == "denied")
    assert not denied_report.admissible
    assert "authority" in denied_report.reject_reasons


def test_adapter_does_not_import_or_verify_faar_signatures():
    # This is a deliberate trust boundary: PCF only consumes a final authority-layer
    # decision shape. Attestation verification and posture/primitive policy stay in FAAR.
    evidence = normalize_faar_authority_decision(decision(Verdict.ALLOW))
    assert evidence.source == "faar"
