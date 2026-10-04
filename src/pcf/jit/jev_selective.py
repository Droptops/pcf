"""Selective evidence gate for Jev advisory routing.

This module does not filter candidates. It evaluates whether benchmark evidence
is strong enough to justify considering a future candidate-narrowing policy.
Low-confidence decisions abstain instead of being counted as selected routes.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from .jev_benchmark import JevBenchmarkSummary


@dataclass(frozen=True)
class JevSelectivePolicy:
    """Evidence requirements before Jev may be considered for route narrowing."""

    min_confidence: float = 0.80
    min_selected_cases: int = 100
    min_accuracy_lcb: float = 0.98
    max_error_rate: float = 0.01
    z_score: float = 1.96

    def __post_init__(self) -> None:
        for name in ("min_confidence", "min_accuracy_lcb", "max_error_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if type(self.min_selected_cases) is not int or self.min_selected_cases <= 0:
            raise ValueError("min_selected_cases must be a positive integer")
        if isinstance(self.z_score, bool) or not isinstance(self.z_score, (int, float)) or self.z_score <= 0:
            raise ValueError("z_score must be positive")


@dataclass(frozen=True)
class JevSelectiveEvidence:
    total_cases: int
    selected_cases: int
    selected_matches: int
    abstained_cases: int
    coverage: float
    selected_accuracy: float
    accuracy_lcb: float
    error_rate: float
    eligible: bool
    reasons: tuple[str, ...]


def wilson_lower_bound(successes: int, trials: int, *, z_score: float = 1.96) -> float:
    """Wilson-score lower confidence bound for a binomial success rate."""
    if type(successes) is not int or type(trials) is not int:
        raise ValueError("successes and trials must be integers")
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    if isinstance(z_score, bool) or not isinstance(z_score, (int, float)) or z_score <= 0:
        raise ValueError("z_score must be positive")
    if trials == 0:
        return 0.0
    p_hat = successes / trials
    z2 = float(z_score) ** 2
    denominator = 1 + z2 / trials
    center = p_hat + z2 / (2 * trials)
    radius = float(z_score) * sqrt((p_hat * (1 - p_hat) + z2 / (4 * trials)) / trials)
    return max(0.0, (center - radius) / denominator)


def evaluate_jev_selective_evidence(
    summary: JevBenchmarkSummary,
    policy: JevSelectivePolicy | None = None,
) -> JevSelectiveEvidence:
    """Evaluate abstaining Jev evidence without changing runtime routing."""
    if not isinstance(summary, JevBenchmarkSummary):
        raise ValueError("summary must be JevBenchmarkSummary")
    policy = policy or JevSelectivePolicy()
    if not isinstance(policy, JevSelectivePolicy):
        raise ValueError("policy must be JevSelectivePolicy")

    selected = [
        item
        for item in summary.observations
        if item.succeeded and item.confidence is not None and item.confidence >= policy.min_confidence
    ]
    selected_cases = len(selected)
    selected_matches = sum(item.matched for item in selected)
    selected_accuracy = selected_matches / selected_cases if selected_cases else 0.0
    accuracy_lcb = wilson_lower_bound(selected_matches, selected_cases, z_score=policy.z_score)
    error_rate = summary.errors / summary.total_cases
    coverage = selected_cases / summary.total_cases

    reasons: list[str] = []
    if selected_cases < policy.min_selected_cases:
        reasons.append("insufficient_selected_cases")
    if accuracy_lcb < policy.min_accuracy_lcb:
        reasons.append("accuracy_lcb_below_threshold")
    if error_rate > policy.max_error_rate:
        reasons.append("error_rate_above_threshold")

    return JevSelectiveEvidence(
        total_cases=summary.total_cases,
        selected_cases=selected_cases,
        selected_matches=selected_matches,
        abstained_cases=summary.total_cases - selected_cases,
        coverage=coverage,
        selected_accuracy=selected_accuracy,
        accuracy_lcb=accuracy_lcb,
        error_rate=error_rate,
        eligible=not reasons,
        reasons=tuple(reasons),
    )
