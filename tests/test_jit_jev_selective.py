from __future__ import annotations

from pcf.jit.jev_benchmark import JevBenchmarkObservation, JevBenchmarkSummary
from pcf.jit.jev_selective import JevSelectivePolicy, evaluate_jev_selective_evidence, wilson_lower_bound
from pcf.jit.types import ExecutionPrimitive


def _obs(
    *,
    confidence: float | None,
    matched: bool = True,
    error: bool = False,
) -> JevBenchmarkObservation:
    expected = ExecutionPrimitive.SMALL_MODEL
    observed = None if error else (expected if matched else ExecutionPrimitive.DETERMINISTIC_CODE)
    return JevBenchmarkObservation(
        case_name=f"case-{confidence}-{matched}-{error}",
        expected=expected,
        observed=observed,
        confidence=None if error else confidence,
        cost_usd=0.0,
        latency_ms=0.0,
        input_tokens=0,
        output_tokens=0,
        error_type="JevOpenRouterError" if error else None,
    )


def test_wilson_lower_bound_is_conservative_for_perfect_small_sample() -> None:
    bound = wilson_lower_bound(26, 26)
    assert 0.86 < bound < 0.89


def test_low_confidence_misses_abstain() -> None:
    observations = tuple(_obs(confidence=0.95) for _ in range(8)) + tuple(
        _obs(confidence=0.40, matched=False) for _ in range(2)
    )
    summary = JevBenchmarkSummary(observations)
    evidence = evaluate_jev_selective_evidence(
        summary,
        JevSelectivePolicy(
            min_confidence=0.8,
            min_selected_cases=8,
            min_accuracy_lcb=0.60,
            max_error_rate=0.0,
        ),
    )
    assert evidence.selected_cases == 8
    assert evidence.selected_matches == 8
    assert evidence.abstained_cases == 2
    assert evidence.coverage == 0.8
    assert evidence.selected_accuracy == 1.0
    assert evidence.eligible


def test_high_confidence_miss_counts_against_selected_accuracy() -> None:
    observations = tuple(_obs(confidence=0.95) for _ in range(9)) + (_obs(confidence=0.95, matched=False),)
    summary = JevBenchmarkSummary(observations)
    evidence = evaluate_jev_selective_evidence(
        summary,
        JevSelectivePolicy(
            min_confidence=0.8,
            min_selected_cases=10,
            min_accuracy_lcb=0.80,
            max_error_rate=0.0,
        ),
    )
    assert evidence.selected_cases == 10
    assert evidence.selected_matches == 9
    assert evidence.selected_accuracy == 0.9
    assert not evidence.eligible
    assert "accuracy_lcb_below_threshold" in evidence.reasons


def test_transport_or_schema_errors_block_evidence_gate() -> None:
    observations = tuple(_obs(confidence=0.99) for _ in range(9)) + (_obs(confidence=None, error=True),)
    summary = JevBenchmarkSummary(observations)
    evidence = evaluate_jev_selective_evidence(
        summary,
        JevSelectivePolicy(
            min_confidence=0.8,
            min_selected_cases=9,
            min_accuracy_lcb=0.60,
            max_error_rate=0.05,
        ),
    )
    assert evidence.error_rate == 0.1
    assert not evidence.eligible
    assert "error_rate_above_threshold" in evidence.reasons


def test_default_policy_rejects_small_perfect_slice() -> None:
    summary = JevBenchmarkSummary(tuple(_obs(confidence=0.99) for _ in range(26)))
    evidence = evaluate_jev_selective_evidence(summary)
    assert evidence.selected_accuracy == 1.0
    assert evidence.selected_cases == 26
    assert not evidence.eligible
    assert "insufficient_selected_cases" in evidence.reasons
    assert "accuracy_lcb_below_threshold" in evidence.reasons
