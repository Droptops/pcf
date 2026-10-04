"""Risk-adjusted least-cost routing across cache, code, and model primitives."""
from __future__ import annotations

from .types import CandidateScore, OptimizerPolicy, RouteCandidate, RouteDecision


class NoAdmissibleRoute(RuntimeError):
    """Raised when every candidate violates a hard routing gate."""


class JITOptimizer:
    """Select the cheapest route *after* authority, freshness, harm, and quality gates."""

    def __init__(self, policy: OptimizerPolicy | None = None) -> None:
        self.policy = policy or OptimizerPolicy()

    def score(self, candidate: RouteCandidate) -> CandidateScore:
        p = self.policy
        reject: list[str] = []
        if not candidate.enabled:
            reject.append("disabled")
        if p.require_authority and not candidate.authority_ok:
            reject.append("authority")
        if p.require_freshness and not candidate.freshness_ok:
            reject.append("freshness")
        if candidate.expected_quality < p.min_quality:
            reject.append("quality")
        if candidate.htokens > p.max_htokens:
            reject.append("harm")

        harm_cost = candidate.htokens * p.usd_per_htoken
        latency_cost = candidate.latency_ms * p.usd_per_ms
        total = (
            candidate.token_cost_usd
            + candidate.memory_cost_usd
            + candidate.uncertainty_cost_usd
            + harm_cost
            + latency_cost
        )
        return CandidateScore(
            name=candidate.name,
            primitive=candidate.primitive,
            admissible=not reject,
            reject_reasons=tuple(reject),
            htokens=candidate.htokens,
            token_cost_usd=candidate.token_cost_usd,
            memory_cost_usd=candidate.memory_cost_usd,
            uncertainty_cost_usd=candidate.uncertainty_cost_usd,
            harm_cost_usd=harm_cost,
            latency_cost_usd=latency_cost,
            total_cost_usd=total,
            expected_quality=candidate.expected_quality,
        )

    def choose(self, candidates: list[RouteCandidate] | tuple[RouteCandidate, ...]) -> RouteDecision:
        if not candidates:
            raise ValueError("at least one route candidate is required")
        reports = tuple(self.score(candidate) for candidate in candidates)
        admissible = [report for report in reports if report.admissible]
        if not admissible:
            reasons = ", ".join(f"{r.name}:{'/'.join(r.reject_reasons)}" for r in reports)
            raise NoAdmissibleRoute(f"no admissible execution route ({reasons})")

        chosen = min(admissible, key=lambda r: (r.total_cost_usd, r.htokens, r.name))
        return RouteDecision(
            chosen=chosen.name,
            primitive=chosen.primitive,
            total_cost_usd=chosen.total_cost_usd,
            htokens=chosen.htokens,
            reports=reports,
            reason="lowest risk-adjusted cost among routes that passed hard gates",
        )
