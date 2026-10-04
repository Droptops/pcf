"""Reconcile PCF preflight cache-cost estimates with post-execution provider usage."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from ..compiler import Usage
from ..router.base import Candidate
from ..segments import Context
from .pcf_cost import PCFInputCostEstimate, estimate_pcf_input_cost
from .shadow import ShadowRequest, ShadowRun


@dataclass(frozen=True)
class PCFInputCostReconciliation:
    """Estimated-versus-observed input/cache billing for one execution."""

    model_id: str
    estimated_cache_read_tokens: int
    actual_cache_read_tokens: int
    estimated_cache_creation_tokens: int
    actual_cache_creation_tokens: int
    estimated_uncached_tokens: int
    actual_uncached_tokens: int
    estimated_total_input_tokens: int
    actual_total_input_tokens: int
    estimated_token_cost_usd: float
    actual_token_cost_usd: float
    estimated_memory_cost_usd: float
    actual_memory_cost_usd: float
    estimated_total_input_cost_usd: float
    actual_total_input_cost_usd: float
    estimate_error_usd: float
    absolute_error_usd: float
    relative_error: float | None

    def __post_init__(self) -> None:
        for name in (
            "estimated_cache_read_tokens",
            "actual_cache_read_tokens",
            "estimated_cache_creation_tokens",
            "actual_cache_creation_tokens",
            "estimated_uncached_tokens",
            "actual_uncached_tokens",
            "estimated_total_input_tokens",
            "actual_total_input_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "estimated_token_cost_usd",
            "actual_token_cost_usd",
            "estimated_memory_cost_usd",
            "actual_memory_cost_usd",
            "estimated_total_input_cost_usd",
            "actual_total_input_cost_usd",
            "absolute_error_usd",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def cache_read_error_tokens(self) -> int:
        return self.estimated_cache_read_tokens - self.actual_cache_read_tokens

    @property
    def cache_creation_error_tokens(self) -> int:
        return self.estimated_cache_creation_tokens - self.actual_cache_creation_tokens

    @property
    def uncached_error_tokens(self) -> int:
        return self.estimated_uncached_tokens - self.actual_uncached_tokens

    @property
    def total_input_error_tokens(self) -> int:
        return self.estimated_total_input_tokens - self.actual_total_input_tokens

    def metadata(self) -> dict[str, object]:
        return {
            "pcf_usage_reconciled": True,
            "pcf_actual_cache_read_tokens": self.actual_cache_read_tokens,
            "pcf_actual_cache_creation_tokens": self.actual_cache_creation_tokens,
            "pcf_actual_uncached_tokens": self.actual_uncached_tokens,
            "pcf_actual_total_input_tokens": self.actual_total_input_tokens,
            "pcf_estimated_token_cost_usd": self.estimated_token_cost_usd,
            "pcf_actual_token_cost_usd": self.actual_token_cost_usd,
            "pcf_estimated_memory_cost_usd": self.estimated_memory_cost_usd,
            "pcf_actual_memory_cost_usd": self.actual_memory_cost_usd,
            "pcf_estimated_total_input_cost_usd": self.estimated_total_input_cost_usd,
            "pcf_actual_total_input_cost_usd": self.actual_total_input_cost_usd,
            "pcf_estimate_error_usd": self.estimate_error_usd,
            "pcf_estimate_absolute_error_usd": self.absolute_error_usd,
            "pcf_estimate_relative_error": self.relative_error,
            "pcf_cache_read_error_tokens": self.cache_read_error_tokens,
            "pcf_cache_creation_error_tokens": self.cache_creation_error_tokens,
            "pcf_uncached_error_tokens": self.uncached_error_tokens,
            "pcf_total_input_error_tokens": self.total_input_error_tokens,
        }


def reconcile_pcf_input_cost(
    estimate: PCFInputCostEstimate,
    usage: Usage,
    candidate: Candidate,
) -> PCFInputCostReconciliation:
    """Price normalized provider usage and compare it with the preflight estimate.

    Positive ``estimate_error_usd`` means the preflight estimate was higher than
    provider-reported input/cache billing. Output-token billing remains out of scope.
    """

    if not isinstance(estimate, PCFInputCostEstimate):
        raise ValueError("estimate must be PCFInputCostEstimate")
    if not isinstance(usage, Usage):
        raise ValueError("usage must be pcf.compiler.Usage")
    if not isinstance(candidate, Candidate):
        raise ValueError("candidate must be pcf.router.Candidate")
    if estimate.model_id != candidate.model_id:
        raise ValueError("estimate model_id does not match candidate")

    actual_token_cost = usage.input_tokens * candidate.input_price_per_mtok / 1e6
    actual_memory_cost = (
        usage.cache_creation_input_tokens * candidate.write_price
        + usage.cache_read_input_tokens * candidate.cache_read_price_per_mtok
    ) / 1e6
    actual_total_cost = actual_token_cost + actual_memory_cost
    error = estimate.total_input_cost_usd - actual_total_cost
    return PCFInputCostReconciliation(
        model_id=estimate.model_id,
        estimated_cache_read_tokens=estimate.warm_tokens,
        actual_cache_read_tokens=usage.cache_read_input_tokens,
        estimated_cache_creation_tokens=estimate.cache_creation_tokens,
        actual_cache_creation_tokens=usage.cache_creation_input_tokens,
        estimated_uncached_tokens=estimate.uncached_tokens,
        actual_uncached_tokens=usage.input_tokens,
        estimated_total_input_tokens=estimate.warm_tokens + estimate.cold_tokens,
        actual_total_input_tokens=usage.total_input_tokens,
        estimated_token_cost_usd=estimate.token_cost_usd,
        actual_token_cost_usd=actual_token_cost,
        estimated_memory_cost_usd=estimate.memory_cost_usd,
        actual_memory_cost_usd=actual_memory_cost,
        estimated_total_input_cost_usd=estimate.total_input_cost_usd,
        actual_total_input_cost_usd=actual_total_cost,
        estimate_error_usd=error,
        absolute_error_usd=abs(error),
        relative_error=(error / actual_total_cost if actual_total_cost > 0 else None),
    )


def apply_pcf_reconciled_input_cost(
    run: ShadowRun,
    estimate: PCFInputCostEstimate,
    reconciliation: PCFInputCostReconciliation,
) -> ShadowRun:
    """Attach estimate + actual billing metadata and meter the run with actual cost."""

    if not isinstance(run, ShadowRun):
        raise ValueError("run must be ShadowRun")
    if not isinstance(estimate, PCFInputCostEstimate):
        raise ValueError("estimate must be PCFInputCostEstimate")
    if not isinstance(reconciliation, PCFInputCostReconciliation):
        raise ValueError("reconciliation must be PCFInputCostReconciliation")
    if estimate.model_id != reconciliation.model_id:
        raise ValueError("estimate and reconciliation model_id must match")
    if run.token_cost_usd != 0 or run.memory_cost_usd != 0:
        raise ValueError("shadow run already contains token or memory cost")

    metadata = dict(run.metadata)
    reserved = {**estimate.metadata(), **reconciliation.metadata()}
    overlap = set(metadata).intersection(reserved)
    if overlap:
        raise ValueError(f"shadow run metadata collides with PCF reconciliation keys: {sorted(overlap)}")
    metadata.update(estimate.metadata())
    metadata.update(reconciliation.metadata())
    return replace(
        run,
        token_cost_usd=reconciliation.actual_token_cost_usd,
        memory_cost_usd=reconciliation.actual_memory_cost_usd,
        metadata=metadata,
    )


NormalizedUsageExtractor = Callable[[ShadowRequest, ShadowRun], Usage]


def make_pcf_reconciled_shadow_runner(
    candidate: Candidate,
    execute: Callable[[ShadowRequest], ShadowRun],
    *,
    context_of: Callable[[ShadowRequest], Context],
    now_of: Callable[[ShadowRequest], float],
    usage_of: NormalizedUsageExtractor,
) -> Callable[[ShadowRequest], ShadowRun]:
    """Meter a shadow route with provider-reported input/cache usage.

    The preflight estimate is captured before execution. After execution, ``usage_of``
    must return PCF's normalized ``Usage`` object from the provider response. Missing
    or malformed usage fails closed instead of silently mixing estimated and actual
    billing in one reconciliation series.
    """

    if not isinstance(candidate, Candidate):
        raise ValueError("candidate must be pcf.router.Candidate")
    for name, fn in (
        ("execute", execute),
        ("context_of", context_of),
        ("now_of", now_of),
        ("usage_of", usage_of),
    ):
        if not callable(fn):
            raise ValueError(f"{name} must be callable")

    def runner(request: ShadowRequest) -> ShadowRun:
        estimate = estimate_pcf_input_cost(context_of(request), candidate, now_of(request))
        run = execute(request)
        if not isinstance(run, ShadowRun):
            raise TypeError("execute must return ShadowRun")
        usage = usage_of(request, run)
        if not isinstance(usage, Usage):
            raise TypeError("usage_of must return pcf.compiler.Usage")
        reconciliation = reconcile_pcf_input_cost(estimate, usage, candidate)
        return apply_pcf_reconciled_input_cost(run, estimate, reconciliation)

    return runner


@dataclass
class PCFReconciliationStats:
    """Aggregate estimate error without retaining request payloads or provider responses."""

    observations: int = 0
    estimate_error_usd: float = 0.0
    absolute_error_usd: float = 0.0
    estimated_total_cost_usd: float = 0.0
    actual_total_cost_usd: float = 0.0
    cache_read_error_tokens: int = 0
    cache_creation_error_tokens: int = 0
    uncached_error_tokens: int = 0

    def observe(self, reconciliation: PCFInputCostReconciliation) -> None:
        if not isinstance(reconciliation, PCFInputCostReconciliation):
            raise ValueError("reconciliation must be PCFInputCostReconciliation")
        self.observations += 1
        self.estimate_error_usd += reconciliation.estimate_error_usd
        self.absolute_error_usd += reconciliation.absolute_error_usd
        self.estimated_total_cost_usd += reconciliation.estimated_total_input_cost_usd
        self.actual_total_cost_usd += reconciliation.actual_total_input_cost_usd
        self.cache_read_error_tokens += reconciliation.cache_read_error_tokens
        self.cache_creation_error_tokens += reconciliation.cache_creation_error_tokens
        self.uncached_error_tokens += reconciliation.uncached_error_tokens

    @property
    def mean_estimate_error_usd(self) -> float:
        return self.estimate_error_usd / self.observations if self.observations else 0.0

    @property
    def mean_absolute_error_usd(self) -> float:
        return self.absolute_error_usd / self.observations if self.observations else 0.0

    @property
    def cost_ratio_actual_to_estimated(self) -> float | None:
        if self.estimated_total_cost_usd <= 0:
            return None
        return self.actual_total_cost_usd / self.estimated_total_cost_usd
