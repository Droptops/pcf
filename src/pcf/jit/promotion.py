"""Statistically explicit promotion gates for JIT shadow candidates."""
from __future__ import annotations

from dataclasses import dataclass
from math import inf, sqrt
from statistics import NormalDist, mean, stdev

from .optimizer import JITOptimizer
from .shadow import ShadowObservation, TraceReplaySummary
from .types import OptimizerPolicy, RouteCandidate


@dataclass(frozen=True)
class PromotionPolicy:
    """Evidence thresholds a shadow route must clear before promotion is recommended."""

    min_matched_samples: int = 100
    confidence_level: float = 0.95
    min_success_rate: float = 0.99
    min_admissible_rate: float = 0.99
    max_quality_regression: float = 0.01
    max_mean_htokens: float = 0.05
    max_htoken_regression: float = 0.0
    min_risk_adjusted_savings_usd_per_call: float = 0.0
    promotion_fixed_cost_usd: float = 0.0
    min_projected_net_savings_usd: float = 0.0
    require_projection: bool = True

    def __post_init__(self) -> None:
        if self.min_matched_samples <= 0:
            raise ValueError("min_matched_samples must be positive")
        if not 0.5 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0.5 and 1")
        for name in ("min_success_rate", "min_admissible_rate"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in (
            "max_quality_regression",
            "max_mean_htokens",
            "max_htoken_regression",
            "min_risk_adjusted_savings_usd_per_call",
            "promotion_fixed_cost_usd",
            "min_projected_net_savings_usd",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if type(self.require_projection) is not bool:
            raise ValueError("require_projection must be boolean")


@dataclass(frozen=True)
class PromotionEvidence:
    candidate: str
    baseline: str
    matched_samples: int
    candidate_successes: int
    candidate_errors: int
    candidate_skipped: int
    success_rate: float
    success_rate_lower_bound: float
    admissible_rate: float
    admissible_rate_lower_bound: float
    mean_quality_delta: float
    quality_delta_lower_bound: float
    mean_candidate_htokens: float
    mean_htoken_delta: float
    htoken_delta_upper_bound: float
    mean_risk_adjusted_savings_usd_per_call: float
    projected_future_calls: int | None
    projected_net_savings_usd: float | None


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reasons: tuple[str, ...]
    evidence: PromotionEvidence
    decision_version: str = "0.1"


class PromotionEvaluator:
    """Compare a candidate with a baseline using paired shadow observations.

    Reliability and admissibility use Wilson lower confidence bounds. Quality and
    HToken deltas use one-sided normal-approximation bounds over paired samples.
    This is a promotion recommendation only; it does not mutate routing state.
    """

    def __init__(
        self,
        policy: PromotionPolicy | None = None,
        *,
        optimizer_policy: OptimizerPolicy | None = None,
    ) -> None:
        self.policy = policy or PromotionPolicy()
        self.optimizer = JITOptimizer(optimizer_policy)

    def evaluate(
        self,
        summary: TraceReplaySummary,
        *,
        candidate: str,
        baseline: str,
        projected_future_calls: int | None,
    ) -> PromotionDecision:
        if not candidate or not baseline:
            raise ValueError("candidate and baseline must be non-empty")
        if candidate == baseline:
            raise ValueError("candidate and baseline must differ")
        if projected_future_calls is not None and projected_future_calls < 0:
            raise ValueError("projected_future_calls must be non-negative")

        candidate_stats = summary.route_stats.get(candidate)
        baseline_stats = summary.route_stats.get(baseline)
        if candidate_stats is None:
            raise ValueError(f"candidate route not found: {candidate}")
        if baseline_stats is None:
            raise ValueError(f"baseline route not found: {baseline}")

        paired_candidate: list[RouteCandidate] = []
        paired_baseline: list[RouteCandidate] = []
        successful_candidates: list[RouteCandidate] = []

        for result in summary.results:
            by_name = {observation.target: observation for observation in result.observations}
            candidate_observation = by_name.get(candidate)
            baseline_observation = by_name.get(baseline)
            candidate_route = self._successful_candidate(candidate_observation)
            baseline_route = self._successful_candidate(baseline_observation)
            if candidate_route is not None:
                successful_candidates.append(candidate_route)
            if candidate_route is not None and baseline_route is not None:
                paired_candidate.append(candidate_route)
                paired_baseline.append(baseline_route)

        attempted = candidate_stats.successes + candidate_stats.errors
        success_rate = candidate_stats.successes / attempted if attempted else 0.0
        success_lower = _wilson_lower_bound(
            candidate_stats.successes,
            attempted,
            self.policy.confidence_level,
        )

        admissible = sum(self.optimizer.score(route).admissible for route in successful_candidates)
        admissible_total = len(successful_candidates)
        admissible_rate = admissible / admissible_total if admissible_total else 0.0
        admissible_lower = _wilson_lower_bound(admissible, admissible_total, self.policy.confidence_level)

        quality_deltas = [c.expected_quality - b.expected_quality for c, b in zip(paired_candidate, paired_baseline)]
        htoken_deltas = [c.htokens - b.htokens for c, b in zip(paired_candidate, paired_baseline)]
        candidate_htokens = [c.htokens for c in paired_candidate]
        risk_savings = [
            self.optimizer.score(b).total_cost_usd - self.optimizer.score(c).total_cost_usd
            for c, b in zip(paired_candidate, paired_baseline)
        ]

        mean_quality_delta = _safe_mean(quality_deltas)
        quality_lower = _mean_bound(quality_deltas, self.policy.confidence_level, upper=False)
        mean_candidate_htokens = _safe_mean(candidate_htokens)
        mean_htoken_delta = _safe_mean(htoken_deltas)
        htoken_upper = _mean_bound(htoken_deltas, self.policy.confidence_level, upper=True)
        mean_savings = _safe_mean(risk_savings)
        projected_net = None
        if projected_future_calls is not None:
            projected_net = mean_savings * projected_future_calls - self.policy.promotion_fixed_cost_usd

        evidence = PromotionEvidence(
            candidate=candidate,
            baseline=baseline,
            matched_samples=len(paired_candidate),
            candidate_successes=candidate_stats.successes,
            candidate_errors=candidate_stats.errors,
            candidate_skipped=candidate_stats.skipped,
            success_rate=success_rate,
            success_rate_lower_bound=success_lower,
            admissible_rate=admissible_rate,
            admissible_rate_lower_bound=admissible_lower,
            mean_quality_delta=mean_quality_delta,
            quality_delta_lower_bound=quality_lower,
            mean_candidate_htokens=mean_candidate_htokens,
            mean_htoken_delta=mean_htoken_delta,
            htoken_delta_upper_bound=htoken_upper,
            mean_risk_adjusted_savings_usd_per_call=mean_savings,
            projected_future_calls=projected_future_calls,
            projected_net_savings_usd=projected_net,
        )

        reasons: list[str] = []
        p = self.policy
        if candidate_stats.skipped:
            reasons.append("candidate_has_skipped_shadow_runs")
        if evidence.matched_samples < p.min_matched_samples:
            reasons.append("insufficient_matched_samples")
        if evidence.success_rate_lower_bound < p.min_success_rate:
            reasons.append("success_rate_below_gate")
        if evidence.admissible_rate_lower_bound < p.min_admissible_rate:
            reasons.append("admissible_rate_below_gate")
        if evidence.quality_delta_lower_bound < -p.max_quality_regression:
            reasons.append("quality_regression_exceeds_gate")
        if evidence.mean_candidate_htokens > p.max_mean_htokens:
            reasons.append("mean_htokens_exceed_gate")
        if evidence.htoken_delta_upper_bound > p.max_htoken_regression:
            reasons.append("htoken_regression_exceeds_gate")
        if evidence.mean_risk_adjusted_savings_usd_per_call <= p.min_risk_adjusted_savings_usd_per_call:
            reasons.append("savings_below_gate")
        if projected_future_calls is None:
            if p.require_projection:
                reasons.append("projection_required")
        elif evidence.projected_net_savings_usd is not None and evidence.projected_net_savings_usd < p.min_projected_net_savings_usd:
            reasons.append("projected_net_savings_below_gate")

        return PromotionDecision(promote=not reasons, reasons=tuple(reasons), evidence=evidence)

    @staticmethod
    def _successful_candidate(observation: ShadowObservation | None) -> RouteCandidate | None:
        if observation is None or observation.status != "success":
            return None
        return observation.candidate


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _wilson_lower_bound(successes: int, total: int, confidence_level: float) -> float:
    if total <= 0:
        return 0.0
    z = NormalDist().inv_cdf(confidence_level)
    phat = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    centre = phat + z2 / (2 * total)
    margin = z * sqrt((phat * (1 - phat) + z2 / (4 * total)) / total)
    return max(0.0, (centre - margin) / denominator)


def _mean_bound(values: list[float], confidence_level: float, *, upper: bool) -> float:
    if not values:
        return inf if upper else -inf
    sample_mean = mean(values)
    if len(values) == 1:
        return inf if upper else -inf
    standard_error = stdev(values) / sqrt(len(values))
    z = NormalDist().inv_cdf(confidence_level)
    delta = z * standard_error
    return sample_mean + delta if upper else sample_mean - delta
