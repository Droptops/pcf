from __future__ import annotations

from dataclasses import dataclass

import pytest

from pcf.jit import ExecutionPrimitive
from pcf.jit.jev_benchmark import (
    JevBenchmarkCase,
    default_jev_benchmark_cases,
    run_jev_benchmark,
)
from pcf.jit.jev_openrouter import JevRouteAdvice, JevUsage


def _advice(primitive: ExecutionPrimitive, *, confidence: float = 0.9) -> JevRouteAdvice:
    probabilities = {item: 0.0 for item in ExecutionPrimitive}
    probabilities[primitive] = 1.0
    return JevRouteAdvice(
        requested_model="typesafe/jev-1.13",
        served_model="typesafe/jev-1.13-test",
        provider="TypeSafe",
        decision_id="decision-1",
        primitive=primitive,
        probabilities=probabilities,
        confidence=confidence,
        usage=JevUsage(input_tokens=10, output_tokens=1, cost_usd=0.00001),
        latency_ms=12.0,
    )


@dataclass
class SequenceRouter:
    outcomes: list[object]

    def route(self, state: object) -> JevRouteAdvice:
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, JevRouteAdvice)
        return outcome


def test_default_suite_balances_all_primitives() -> None:
    cases = default_jev_benchmark_cases()
    assert len(cases) == 15
    for primitive in ExecutionPrimitive:
        assert sum(case.expected is primitive for case in cases) == 3
    assert len({case.name for case in cases}) == len(cases)


def test_benchmark_counts_failures_as_misses() -> None:
    cases = (
        JevBenchmarkCase("a", ExecutionPrimitive.DETERMINISTIC_CODE, {"x": 1}),
        JevBenchmarkCase("b", ExecutionPrimitive.SMALL_MODEL, {"x": 2}),
        JevBenchmarkCase("c", ExecutionPrimitive.FRONTIER_MODEL, {"x": 3}),
    )
    router = SequenceRouter(
        [
            _advice(ExecutionPrimitive.DETERMINISTIC_CODE),
            RuntimeError("provider failed"),
            _advice(ExecutionPrimitive.SMALL_MODEL),
        ]
    )
    summary = run_jev_benchmark(router, cases)
    assert summary.total_cases == 3
    assert summary.successes == 2
    assert summary.errors == 1
    assert summary.matches == 1
    assert summary.accuracy == pytest.approx(1 / 3)
    assert summary.successful_accuracy == pytest.approx(1 / 2)


def test_results_do_not_retain_request_state() -> None:
    secret_marker = "do-not-retain-this-state"
    case = JevBenchmarkCase(
        "privacy",
        ExecutionPrimitive.RESULT_CACHE,
        {"payload": secret_marker},
    )
    summary = run_jev_benchmark(
        SequenceRouter([_advice(ExecutionPrimitive.RESULT_CACHE)]),
        (case,),
    )
    assert secret_marker not in repr(summary)
    observation = summary.observations[0]
    assert not hasattr(observation, "state")


def test_low_confidence_rate_uses_successful_calls_only() -> None:
    cases = (
        JevBenchmarkCase("a", ExecutionPrimitive.SMALL_MODEL, {}),
        JevBenchmarkCase("b", ExecutionPrimitive.SMALL_MODEL, {}),
        JevBenchmarkCase("c", ExecutionPrimitive.SMALL_MODEL, {}),
    )
    summary = run_jev_benchmark(
        SequenceRouter(
            [
                _advice(ExecutionPrimitive.SMALL_MODEL, confidence=0.95),
                _advice(ExecutionPrimitive.SMALL_MODEL, confidence=0.4),
                RuntimeError("down"),
            ]
        ),
        cases,
        min_confidence=0.8,
    )
    assert summary.low_confidence_rate == pytest.approx(0.5)


def test_confusion_matrix_and_per_class_accuracy() -> None:
    cases = (
        JevBenchmarkCase("a", ExecutionPrimitive.RESULT_CACHE, {}),
        JevBenchmarkCase("b", ExecutionPrimitive.RESULT_CACHE, {}),
        JevBenchmarkCase("c", ExecutionPrimitive.DETERMINISTIC_CODE, {}),
    )
    summary = run_jev_benchmark(
        SequenceRouter(
            [
                _advice(ExecutionPrimitive.RESULT_CACHE),
                _advice(ExecutionPrimitive.DETERMINISTIC_CODE),
                _advice(ExecutionPrimitive.DETERMINISTIC_CODE),
            ]
        ),
        cases,
    )
    matrix = summary.confusion_matrix()
    assert matrix[ExecutionPrimitive.RESULT_CACHE]["result_cache"] == 1
    assert matrix[ExecutionPrimitive.RESULT_CACHE]["deterministic_code"] == 1
    accuracy = summary.accuracy_by_expected()
    assert accuracy[ExecutionPrimitive.RESULT_CACHE] == pytest.approx(0.5)
    assert accuracy[ExecutionPrimitive.DETERMINISTIC_CODE] == pytest.approx(1.0)


def test_cost_and_latency_are_aggregated() -> None:
    cases = (
        JevBenchmarkCase("a", ExecutionPrimitive.SMALL_MODEL, {}),
        JevBenchmarkCase("b", ExecutionPrimitive.SMALL_MODEL, {}),
    )
    summary = run_jev_benchmark(
        SequenceRouter(
            [
                _advice(ExecutionPrimitive.SMALL_MODEL, confidence=0.8),
                _advice(ExecutionPrimitive.SMALL_MODEL, confidence=1.0),
            ]
        ),
        cases,
    )
    assert summary.total_cost_usd == pytest.approx(0.00002)
    assert summary.mean_latency_ms == pytest.approx(12.0)
    assert summary.mean_confidence == pytest.approx(0.9)


def test_duplicate_names_fail_closed() -> None:
    cases = (
        JevBenchmarkCase("same", ExecutionPrimitive.SMALL_MODEL, {}),
        JevBenchmarkCase("same", ExecutionPrimitive.FRONTIER_MODEL, {}),
    )
    with pytest.raises(ValueError, match="unique"):
        run_jev_benchmark(SequenceRouter([]), cases)


def test_empty_suite_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        run_jev_benchmark(SequenceRouter([]), ())
