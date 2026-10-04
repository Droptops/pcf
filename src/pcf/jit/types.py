"""Core types for the risk-adjusted JIT execution optimizer."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExecutionPrimitive(str, Enum):
    RESULT_CACHE = "result_cache"
    DETERMINISTIC_CODE = "deterministic_code"
    SPECIALIST_MODEL = "specialist_model"
    SMALL_MODEL = "small_model"
    FRONTIER_MODEL = "frontier_model"


@dataclass(frozen=True)
class RouteCandidate:
    """One admissible execution route for the same semantic task."""

    name: str
    primitive: ExecutionPrimitive
    token_cost_usd: float = 0.0
    memory_cost_usd: float = 0.0
    latency_ms: float = 0.0
    expected_quality: float = 1.0
    harm_probability: float = 0.0
    harm_severity: float = 0.0
    exposure: float = 1.0
    authority_ok: bool = True
    freshness_ok: bool = True
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("candidate name must be non-empty")
        for field_name in (
            "token_cost_usd",
            "memory_cost_usd",
            "latency_ms",
            "harm_probability",
            "harm_severity",
            "exposure",
        ):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if not 0 <= self.expected_quality <= 1:
            raise ValueError("expected_quality must be between 0 and 1")
        if not 0 <= self.harm_probability <= 1:
            raise ValueError("harm_probability must be between 0 and 1")

    @property
    def htokens(self) -> float:
        """Expected harm units: probability x severity x exposure."""
        return self.harm_probability * self.harm_severity * self.exposure


@dataclass(frozen=True)
class OptimizerPolicy:
    """Hard safety/quality gates followed by least-risk-adjusted-cost routing."""

    min_quality: float = 0.95
    max_htokens: float = 1.0
    usd_per_htoken: float = 1.0
    usd_per_ms: float = 0.0
    require_authority: bool = True
    require_freshness: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.min_quality <= 1:
            raise ValueError("min_quality must be between 0 and 1")
        for field_name in ("max_htokens", "usd_per_htoken", "usd_per_ms"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")


@dataclass(frozen=True)
class CandidateScore:
    name: str
    primitive: ExecutionPrimitive
    admissible: bool
    reject_reasons: tuple[str, ...]
    htokens: float
    token_cost_usd: float
    memory_cost_usd: float
    harm_cost_usd: float
    latency_cost_usd: float
    total_cost_usd: float
    expected_quality: float


@dataclass(frozen=True)
class RouteDecision:
    chosen: str
    primitive: ExecutionPrimitive
    total_cost_usd: float
    htokens: float
    reports: tuple[CandidateScore, ...]
    reason: str
