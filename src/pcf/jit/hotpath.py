"""Hot-path and compile-economics helpers."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, inf

from .fingerprint import FingerprintAuditSummary


@dataclass(frozen=True)
class CompileEconomics:
    general_cost_per_call_usd: float
    compiled_cost_per_call_usd: float
    build_cost_usd: float
    evaluation_cost_usd: float = 0.0
    deployment_cost_usd: float = 0.0
    risk_reserve_usd: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def fixed_cost_usd(self) -> float:
        return self.build_cost_usd + self.evaluation_cost_usd + self.deployment_cost_usd + self.risk_reserve_usd

    @property
    def savings_per_call_usd(self) -> float:
        return self.general_cost_per_call_usd - self.compiled_cost_per_call_usd

    @property
    def breakeven_calls(self) -> float:
        savings = self.savings_per_call_usd
        if savings <= 0:
            return inf
        return ceil(self.fixed_cost_usd / savings)

    def should_compile(self, projected_future_calls: int) -> bool:
        if projected_future_calls < 0:
            raise ValueError("projected_future_calls must be non-negative")
        return self.savings_per_call_usd > 0 and projected_future_calls >= self.breakeven_calls


@dataclass
class HotPathStats:
    calls: int = 0
    stable_calls: int = 0

    @property
    def stability(self) -> float:
        return self.stable_calls / self.calls if self.calls else 0.0

    def observe(self, *, stable: bool) -> None:
        self.calls += 1
        self.stable_calls += int(stable)


@dataclass(frozen=True)
class CompileEligibility:
    """Evidence-backed decision about whether one hot path may enter compilation."""

    fingerprint: str
    eligible: bool
    reasons: tuple[str, ...]
    calls: int
    stability: float
    audit_observations: int


class HotPathDetector:
    """Identifies repetitive workloads and gates compilation on semantic audit evidence.

    ``is_hot`` remains a volume/stability signal. ``evaluate_compile_eligibility`` is
    the stronger gate that requires adjudicated fingerprint coverage and blocks
    collisions, semantic drift, and fingerprint splits.
    """

    def __init__(
        self,
        *,
        min_calls: int = 100,
        min_stability: float = 0.98,
        min_audit_observations: int = 20,
    ) -> None:
        if min_calls <= 0:
            raise ValueError("min_calls must be positive")
        if not 0 <= min_stability <= 1:
            raise ValueError("min_stability must be between 0 and 1")
        if min_audit_observations <= 0:
            raise ValueError("min_audit_observations must be positive")
        self.min_calls = min_calls
        self.min_stability = min_stability
        self.min_audit_observations = min_audit_observations
        self._stats: dict[str, HotPathStats] = {}

    def observe(self, fingerprint: str, *, stable: bool) -> HotPathStats:
        if not fingerprint:
            raise ValueError("fingerprint must be non-empty")
        stats = self._stats.setdefault(fingerprint, HotPathStats())
        stats.observe(stable=stable)
        return stats

    def is_hot(self, fingerprint: str) -> bool:
        stats = self._stats.get(fingerprint)
        return bool(stats and stats.calls >= self.min_calls and stats.stability >= self.min_stability)

    def evaluate_compile_eligibility(
        self,
        fingerprint: str,
        audit: FingerprintAuditSummary,
    ) -> CompileEligibility:
        if not fingerprint:
            raise ValueError("fingerprint must be non-empty")
        if not isinstance(audit, FingerprintAuditSummary):
            raise ValueError("audit must be FingerprintAuditSummary")

        stats = self._stats.get(fingerprint) or HotPathStats()
        reasons: list[str] = []
        if stats.calls < self.min_calls:
            reasons.append("insufficient_calls")
        if stats.stability < self.min_stability:
            reasons.append("insufficient_stability")

        audit_observations = audit.observations_for(fingerprint)
        if audit_observations == 0:
            reasons.append("audit_missing")
        elif audit_observations < self.min_audit_observations:
            reasons.append("insufficient_audit_observations")

        if fingerprint in audit.collision_examples:
            reasons.append("fingerprint_collision")
        if any(key.split("|", 1)[0] == fingerprint for key in audit.drift_examples):
            reasons.append("semantic_drift")
        if any(fingerprint in fingerprints for fingerprints in audit.split_examples.values()):
            reasons.append("fingerprint_split")

        return CompileEligibility(
            fingerprint=fingerprint,
            eligible=not reasons,
            reasons=tuple(reasons),
            calls=stats.calls,
            stability=stats.stability,
            audit_observations=audit_observations,
        )

    def is_compile_eligible(self, fingerprint: str, audit: FingerprintAuditSummary) -> bool:
        return self.evaluate_compile_eligibility(fingerprint, audit).eligible
