"""Labeled Jev task-shape benchmark for the PCF JIT control plane.

The benchmark measures whether Jev agrees with adjudicated execution-primitive
labels. It is evidence only: benchmark accuracy does not itself authorize route
filtering, repricing, or promotion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .jev_openrouter import JevRouteAdvice
from .types import ExecutionPrimitive


class JevRouter(Protocol):
    def route(self, state: Any) -> JevRouteAdvice: ...


@dataclass(frozen=True)
class JevBenchmarkCase:
    name: str
    expected: ExecutionPrimitive
    state: Mapping[str, Any]
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("benchmark case name must be non-empty")
        if not isinstance(self.expected, ExecutionPrimitive):
            raise ValueError("expected must be an ExecutionPrimitive")
        if not isinstance(self.state, Mapping):
            raise ValueError("state must be a mapping")


@dataclass(frozen=True)
class JevBenchmarkObservation:
    case_name: str
    expected: ExecutionPrimitive
    observed: ExecutionPrimitive | None
    confidence: float | None
    cost_usd: float
    latency_ms: float
    input_tokens: int
    output_tokens: int
    error_type: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error_type is None and self.observed is not None

    @property
    def matched(self) -> bool:
        return self.succeeded and self.observed is self.expected


@dataclass(frozen=True)
class JevBenchmarkSummary:
    observations: tuple[JevBenchmarkObservation, ...]
    min_confidence: float = 0.8

    def __post_init__(self) -> None:
        if not self.observations:
            raise ValueError("benchmark summary requires at least one observation")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")

    @property
    def total_cases(self) -> int:
        return len(self.observations)

    @property
    def successes(self) -> int:
        return sum(item.succeeded for item in self.observations)

    @property
    def errors(self) -> int:
        return self.total_cases - self.successes

    @property
    def matches(self) -> int:
        return sum(item.matched for item in self.observations)

    @property
    def accuracy(self) -> float:
        """Overall accuracy, counting transport/schema failures as misses."""
        return self.matches / self.total_cases

    @property
    def successful_accuracy(self) -> float:
        return self.matches / self.successes if self.successes else 0.0

    @property
    def low_confidence_rate(self) -> float:
        successful = [item for item in self.observations if item.succeeded]
        if not successful:
            return 1.0
        low = sum((item.confidence or 0.0) < self.min_confidence for item in successful)
        return low / len(successful)

    @property
    def total_cost_usd(self) -> float:
        return sum(item.cost_usd for item in self.observations)

    @property
    def mean_latency_ms(self) -> float:
        successful = [item.latency_ms for item in self.observations if item.succeeded]
        return sum(successful) / len(successful) if successful else 0.0

    @property
    def mean_confidence(self) -> float:
        successful = [item.confidence for item in self.observations if item.succeeded and item.confidence is not None]
        return sum(successful) / len(successful) if successful else 0.0

    def accuracy_by_expected(self) -> dict[ExecutionPrimitive, float]:
        result: dict[ExecutionPrimitive, float] = {}
        for primitive in ExecutionPrimitive:
            bucket = [item for item in self.observations if item.expected is primitive]
            result[primitive] = sum(item.matched for item in bucket) / len(bucket) if bucket else 0.0
        return result

    def confusion_matrix(self) -> dict[ExecutionPrimitive, dict[str, int]]:
        matrix: dict[ExecutionPrimitive, dict[str, int]] = {}
        for expected in ExecutionPrimitive:
            row = {observed.value: 0 for observed in ExecutionPrimitive}
            row["error"] = 0
            for item in self.observations:
                if item.expected is not expected:
                    continue
                if item.observed is None:
                    row["error"] += 1
                else:
                    row[item.observed.value] += 1
            matrix[expected] = row
        return matrix


def run_jev_benchmark(
    router: JevRouter,
    cases: Sequence[JevBenchmarkCase],
    *,
    min_confidence: float = 0.8,
) -> JevBenchmarkSummary:
    """Run labeled cases without retaining request state in benchmark results."""
    if not cases:
        raise ValueError("benchmark requires at least one case")
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("benchmark case names must be unique")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")

    observations: list[JevBenchmarkObservation] = []
    for case in cases:
        try:
            advice = router.route(case.state)
        except Exception as exc:  # benchmark must measure route failures without retaining request data
            observations.append(
                JevBenchmarkObservation(
                    case_name=case.name,
                    expected=case.expected,
                    observed=None,
                    confidence=None,
                    cost_usd=0.0,
                    latency_ms=0.0,
                    input_tokens=0,
                    output_tokens=0,
                    error_type=type(exc).__name__,
                )
            )
            continue
        observations.append(
            JevBenchmarkObservation(
                case_name=case.name,
                expected=case.expected,
                observed=advice.primitive,
                confidence=advice.confidence,
                cost_usd=advice.usage.cost_usd,
                latency_ms=advice.latency_ms,
                input_tokens=advice.usage.input_tokens,
                output_tokens=advice.usage.output_tokens,
            )
        )
    return JevBenchmarkSummary(tuple(observations), min_confidence=min_confidence)


def default_jev_benchmark_cases() -> tuple[JevBenchmarkCase, ...]:
    """Synthetic, adjudicated task-shape cases spanning every PCF primitive.

    These labels are hypotheses to test, not universal semantic truth. The suite
    deliberately states the execution preconditions so the benchmark measures
    route-shape discrimination rather than hidden policy assumptions.
    """
    return (
        JevBenchmarkCase(
            "cache_verified_exact_reuse",
            ExecutionPrimitive.RESULT_CACHE,
            {
                "task": "return the previously verified answer for this exact request",
                "cache": "verified hit",
                "dependencies": "unchanged",
                "policy_version": "unchanged",
                "authority_scope": "unchanged",
            },
            ("reuse", "stable-state"),
        ),
        JevBenchmarkCase(
            "cache_semantic_reuse_stable_dependencies",
            ExecutionPrimitive.RESULT_CACHE,
            {
                "task": "answer a semantically equivalent request from a verified result",
                "semantic_match": "high and adjudicated",
                "dependencies": "unchanged",
                "freshness": "valid",
            },
            ("reuse", "semantic-cache"),
        ),
        JevBenchmarkCase(
            "cache_repeated_reference_lookup",
            ExecutionPrimitive.RESULT_CACHE,
            {
                "task": "reuse a verified reference response",
                "request_fingerprint": "same",
                "source_versions": "same",
                "cached_result": "present and valid",
            },
            ("reuse", "fingerprint"),
        ),
        JevBenchmarkCase(
            "code_unit_conversion",
            ExecutionPrimitive.DETERMINISTIC_CODE,
            {
                "task": "convert 37 degrees Celsius to Fahrenheit",
                "requirements": "exact deterministic arithmetic",
                "external_state": "none",
            },
            ("code", "arithmetic"),
        ),
        JevBenchmarkCase(
            "code_schema_validation",
            ExecutionPrimitive.DETERMINISTIC_CODE,
            {
                "task": "validate a JSON object against a fixed JSON Schema and return pass/fail",
                "schema": "fixed and available",
                "generation": "not required",
            },
            ("code", "validation"),
        ),
        JevBenchmarkCase(
            "code_exact_lookup",
            ExecutionPrimitive.DETERMINISTIC_CODE,
            {
                "task": "map a two-letter US state code to its full state name",
                "mapping": "fixed local table",
                "ambiguity": "none",
            },
            ("code", "lookup"),
        ),
        JevBenchmarkCase(
            "specialist_telecom_fault_taxonomy",
            ExecutionPrimitive.SPECIALIST_MODEL,
            {
                "task": "classify a carrier interconnect incident into a narrow telecom fault taxonomy",
                "input": "free-form NOC notes with SIP and routing jargon",
                "taxonomy": "bounded but domain-specific",
                "generation": "minimal",
            },
            ("specialist", "telecom"),
        ),
        JevBenchmarkCase(
            "specialist_financial_document_clause",
            ExecutionPrimitive.SPECIALIST_MODEL,
            {
                "task": "identify which predefined covenant category a complex loan-clause excerpt belongs to",
                "taxonomy": "fixed domain taxonomy",
                "input": "specialized financial/legal language",
                "open_ended": False,
            },
            ("specialist", "finance"),
        ),
        JevBenchmarkCase(
            "specialist_medical_coding_family",
            ExecutionPrimitive.SPECIALIST_MODEL,
            {
                "task": "map a short de-identified clinical description to one of a fixed set of coding families",
                "taxonomy": "bounded clinical coding families",
                "input": "domain terminology requiring specialist semantics",
                "treatment_decision": "none",
            },
            ("specialist", "clinical"),
        ),
        JevBenchmarkCase(
            "small_rewrite_for_tone",
            ExecutionPrimitive.SMALL_MODEL,
            {
                "task": "rewrite a short paragraph to sound concise and professional",
                "domain_expertise": "none",
                "reasoning_depth": "low",
                "generation": "bounded",
            },
            ("small-model", "rewrite"),
        ),
        JevBenchmarkCase(
            "small_plain_language_summary",
            ExecutionPrimitive.SMALL_MODEL,
            {
                "task": "summarize a straightforward internal update in three bullets",
                "ambiguity": "low",
                "domain_expertise": "none",
                "generation": "bounded",
            },
            ("small-model", "summary"),
        ),
        JevBenchmarkCase(
            "small_generic_extraction_with_paraphrase",
            ExecutionPrimitive.SMALL_MODEL,
            {
                "task": "extract three key points from an ordinary customer email and paraphrase them",
                "input": "general business language",
                "reasoning_depth": "low",
                "generation": "bounded",
            },
            ("small-model", "general-language"),
        ),
        JevBenchmarkCase(
            "frontier_ambiguous_strategy",
            ExecutionPrimitive.FRONTIER_MODEL,
            {
                "task": "design a strategy under conflicting goals, incomplete evidence, and multiple plausible interpretations",
                "constraints": "novel and interacting",
                "reasoning_depth": "high",
                "ambiguity": "high",
            },
            ("frontier", "strategy"),
        ),
        JevBenchmarkCase(
            "frontier_novel_architecture_debug",
            ExecutionPrimitive.FRONTIER_MODEL,
            {
                "task": "diagnose a novel distributed-system failure spanning cache invalidation, concurrency, and partial outages",
                "evidence": "incomplete traces and conflicting symptoms",
                "reasoning_depth": "multi-step",
                "known_playbook": "none",
            },
            ("frontier", "engineering"),
        ),
        JevBenchmarkCase(
            "frontier_research_synthesis",
            ExecutionPrimitive.FRONTIER_MODEL,
            {
                "task": "synthesize several conflicting research findings and propose what additional evidence would resolve them",
                "sources": "heterogeneous and uncertain",
                "reasoning_depth": "high",
                "output": "novel synthesis",
            },
            ("frontier", "research"),
        ),
    )
