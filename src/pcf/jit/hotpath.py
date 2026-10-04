"""Hot-path and compile-economics helpers."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, inf


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


class HotPathDetector:
    """Identifies repetitive, semantically stable workloads worth considering for compilation."""

    def __init__(self, *, min_calls: int = 100, min_stability: float = 0.98) -> None:
        if min_calls <= 0:
            raise ValueError("min_calls must be positive")
        if not 0 <= min_stability <= 1:
            raise ValueError("min_stability must be between 0 and 1")
        self.min_calls = min_calls
        self.min_stability = min_stability
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
