"""Bridge PCF prompt-cache cost estimates into JIT shadow-run metering."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from ..router.base import Candidate
from ..segments import Context
from ..validation import number
from .shadow import ShadowRequest, ShadowRun


@dataclass(frozen=True)
class PCFInputCostEstimate:
    """Pre-execution input/cache cost estimate from PCF's compiler warmth model."""

    model_id: str
    warm_prefix_segments: int
    warm_tokens: int
    cold_tokens: int
    cache_creation_tokens: int
    uncached_tokens: int
    cache_state: str
    observed_at: float
    token_cost_usd: float
    memory_cost_usd: float
    total_input_cost_usd: float
    token_count_is_estimate: bool

    def __post_init__(self) -> None:
        for name in (
            "warm_prefix_segments",
            "warm_tokens",
            "cold_tokens",
            "cache_creation_tokens",
            "uncached_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("token_cost_usd", "memory_cost_usd", "total_input_cost_usd"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def metadata(self) -> dict[str, object]:
        return {
            "pcf_model_id": self.model_id,
            "pcf_warm_prefix_segments": self.warm_prefix_segments,
            "pcf_warm_tokens": self.warm_tokens,
            "pcf_cold_tokens": self.cold_tokens,
            "pcf_cache_creation_tokens": self.cache_creation_tokens,
            "pcf_uncached_tokens": self.uncached_tokens,
            "pcf_cache_state": self.cache_state,
            "pcf_cost_observed_at": self.observed_at,
            "pcf_token_count_is_estimate": self.token_count_is_estimate,
        }


def estimate_pcf_input_cost(ctx: Context, candidate: Candidate, now: float) -> PCFInputCostEstimate:
    """Estimate one candidate's input cost without mutating its cache metadata store.

    Uncached tokens are JIT ``token_cost_usd``. Cache reads and cache creation are
    JIT ``memory_cost_usd``. Their sum is identical to the existing PCF router's
    input-cost formula for the same candidate and cache state.
    """

    if not isinstance(ctx, Context):
        raise ValueError("ctx must be Context")
    if not isinstance(candidate, Candidate):
        raise ValueError("candidate must be pcf.router.Candidate")
    number(now, "now")
    warmth = candidate.compiler.warmth(ctx, candidate.cache, now)
    token_cost = warmth.uncached_tokens * candidate.input_price_per_mtok / 1e6
    memory_cost = (
        warmth.cache_creation_tokens * candidate.write_price
        + warmth.warm_tokens * candidate.cache_read_price_per_mtok
    ) / 1e6
    return PCFInputCostEstimate(
        model_id=candidate.model_id,
        warm_prefix_segments=warmth.warm_prefix_segments,
        warm_tokens=warmth.warm_tokens,
        cold_tokens=warmth.cold_tokens,
        cache_creation_tokens=warmth.cache_creation_tokens,
        uncached_tokens=warmth.uncached_tokens,
        cache_state=warmth.cache_state,
        observed_at=warmth.observed_at,
        token_cost_usd=token_cost,
        memory_cost_usd=memory_cost,
        total_input_cost_usd=token_cost + memory_cost,
        token_count_is_estimate=(
            candidate.compiler.tokenizer.is_estimate
            or candidate.compiler.descriptor.identity_kind != "simulated"
        ),
    )


def apply_pcf_input_cost(run: ShadowRun, estimate: PCFInputCostEstimate) -> ShadowRun:
    """Attach PCF input/cache economics to an otherwise unmetered shadow run.

    Reject pre-metered runs so token/cache cost cannot be accidentally double-counted.
    Output cost is intentionally out of scope because PCF warmth models input caching.
    """

    if not isinstance(run, ShadowRun):
        raise ValueError("run must be ShadowRun")
    if not isinstance(estimate, PCFInputCostEstimate):
        raise ValueError("estimate must be PCFInputCostEstimate")
    if run.token_cost_usd != 0 or run.memory_cost_usd != 0:
        raise ValueError("shadow run already contains token or memory cost")
    metadata = dict(run.metadata)
    overlap = set(metadata).intersection(estimate.metadata())
    if overlap:
        raise ValueError(f"shadow run metadata collides with PCF metering keys: {sorted(overlap)}")
    metadata.update(estimate.metadata())
    return replace(
        run,
        token_cost_usd=estimate.token_cost_usd,
        memory_cost_usd=estimate.memory_cost_usd,
        metadata=metadata,
    )


def make_pcf_metered_shadow_runner(
    candidate: Candidate,
    execute: Callable[[ShadowRequest], ShadowRun],
    *,
    context_of: Callable[[ShadowRequest], Context],
    now_of: Callable[[ShadowRequest], float],
) -> Callable[[ShadowRequest], ShadowRun]:
    """Wrap a shadow runner with a pre-execution PCF cache-cost snapshot.

    Cost is estimated before ``execute`` is called. This ordering prevents an
    execution that populates cache from making its own request appear warm.
    """

    if not isinstance(candidate, Candidate):
        raise ValueError("candidate must be pcf.router.Candidate")
    for name, fn in (("execute", execute), ("context_of", context_of), ("now_of", now_of)):
        if not callable(fn):
            raise ValueError(f"{name} must be callable")

    def runner(request: ShadowRequest) -> ShadowRun:
        ctx = context_of(request)
        now = now_of(request)
        estimate = estimate_pcf_input_cost(ctx, candidate, now)
        run = execute(request)
        return apply_pcf_input_cost(run, estimate)

    return runner
