"""Side-effect-free shadow execution and trace replay for JIT route evaluation."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping

from .optimizer import JITOptimizer, NoAdmissibleRoute
from .types import ExecutionPrimitive, RouteCandidate, RouteDecision


@dataclass(frozen=True)
class ShadowRequest:
    """One semantic request replayed against each shadow-safe route."""

    request_id: str
    fingerprint: str
    payload: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.fingerprint:
            raise ValueError("fingerprint must be non-empty")


@dataclass(frozen=True)
class ShadowRun:
    """Observed execution output and metering from one route."""

    output: Any
    token_cost_usd: float = 0.0
    memory_cost_usd: float = 0.0
    latency_ms: float | None = None
    authority_ok: bool = True
    freshness_ok: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("token_cost_usd", "memory_cost_usd"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")


@dataclass(frozen=True)
class ShadowAssessment:
    """Quality/risk adjudication for a completed route."""

    expected_quality: float
    harm_probability: float = 0.0
    harm_severity: float = 0.0
    exposure: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.expected_quality <= 1:
            raise ValueError("expected_quality must be between 0 and 1")
        if not 0 <= self.harm_probability <= 1:
            raise ValueError("harm_probability must be between 0 and 1")
        for name in ("harm_severity", "exposure"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


ShadowRunner = Callable[[ShadowRequest], ShadowRun]
ShadowAssessor = Callable[[ShadowRequest, "ShadowTarget", ShadowRun], ShadowAssessment]


@dataclass(frozen=True)
class ShadowTarget:
    """A route adapter eligible for replay only when explicitly declared shadow-safe."""

    name: str
    primitive: ExecutionPrimitive
    runner: ShadowRunner
    shadow_safe: bool = False
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("target name must be non-empty")
        if not callable(self.runner):
            raise ValueError("runner must be callable")


@dataclass(frozen=True)
class ShadowObservation:
    target: str
    primitive: ExecutionPrimitive
    status: str
    output: Any = None
    error: str | None = None
    latency_ms: float = 0.0
    candidate: RouteCandidate | None = None


@dataclass(frozen=True)
class ShadowResult:
    request_id: str
    fingerprint: str
    observations: tuple[ShadowObservation, ...]
    decision: RouteDecision | None
    decision_error: str | None = None


@dataclass
class ReplayRouteStats:
    attempts: int = 0
    successes: int = 0
    errors: int = 0
    skipped: int = 0
    admissible: int = 0
    chosen: int = 0
    token_cost_usd: float = 0.0
    memory_cost_usd: float = 0.0
    latency_ms: float = 0.0
    htokens: float = 0.0
    quality: float = 0.0

    @property
    def success_rate(self) -> float:
        attempted = self.successes + self.errors
        return self.successes / attempted if attempted else 0.0

    @property
    def mean_quality(self) -> float:
        return self.quality / self.successes if self.successes else 0.0

    @property
    def mean_htokens(self) -> float:
        return self.htokens / self.successes if self.successes else 0.0

    @property
    def mean_latency_ms(self) -> float:
        return self.latency_ms / self.successes if self.successes else 0.0


@dataclass(frozen=True)
class TraceReplaySummary:
    results: tuple[ShadowResult, ...]
    route_stats: Mapping[str, ReplayRouteStats]


class ShadowExecutor:
    """Run the same request against shadow-safe candidates without promoting any route."""

    def __init__(self, assessor: ShadowAssessor, optimizer: JITOptimizer | None = None) -> None:
        if not callable(assessor):
            raise ValueError("assessor must be callable")
        self.assessor = assessor
        self.optimizer = optimizer or JITOptimizer()

    def execute(
        self,
        request: ShadowRequest,
        targets: Iterable[ShadowTarget],
    ) -> ShadowResult:
        targets = tuple(targets)
        if not targets:
            raise ValueError("at least one shadow target is required")
        if len({target.name for target in targets}) != len(targets):
            raise ValueError("shadow target names must be unique")

        observations: list[ShadowObservation] = []
        candidates: list[RouteCandidate] = []

        for target in targets:
            if not target.enabled or not target.shadow_safe:
                observations.append(
                    ShadowObservation(
                        target=target.name,
                        primitive=target.primitive,
                        status="skipped",
                        error="disabled" if not target.enabled else "not_shadow_safe",
                    )
                )
                continue

            started = perf_counter()
            try:
                isolated_request = deepcopy(request)
                run = target.runner(isolated_request)
                if not isinstance(run, ShadowRun):
                    raise TypeError("runner must return ShadowRun")
                elapsed_ms = (perf_counter() - started) * 1000
                latency_ms = run.latency_ms if run.latency_ms is not None else elapsed_ms
                assessment = self.assessor(deepcopy(request), target, run)
                if not isinstance(assessment, ShadowAssessment):
                    raise TypeError("assessor must return ShadowAssessment")
                candidate = RouteCandidate(
                    name=target.name,
                    primitive=target.primitive,
                    token_cost_usd=run.token_cost_usd,
                    memory_cost_usd=run.memory_cost_usd,
                    latency_ms=latency_ms,
                    expected_quality=assessment.expected_quality,
                    harm_probability=assessment.harm_probability,
                    harm_severity=assessment.harm_severity,
                    exposure=assessment.exposure,
                    authority_ok=run.authority_ok,
                    freshness_ok=run.freshness_ok,
                    metadata=dict(run.metadata),
                )
                candidates.append(candidate)
                observations.append(
                    ShadowObservation(
                        target=target.name,
                        primitive=target.primitive,
                        status="success",
                        output=run.output,
                        latency_ms=latency_ms,
                        candidate=candidate,
                    )
                )
            except Exception as exc:
                observations.append(
                    ShadowObservation(
                        target=target.name,
                        primitive=target.primitive,
                        status="error",
                        error=f"{type(exc).__name__}: {exc}",
                        latency_ms=(perf_counter() - started) * 1000,
                    )
                )

        decision = None
        decision_error = None
        if candidates:
            try:
                decision = self.optimizer.choose(candidates)
            except NoAdmissibleRoute as exc:
                decision_error = str(exc)
        else:
            decision_error = "no successful shadow-safe routes"

        return ShadowResult(
            request_id=request.request_id,
            fingerprint=request.fingerprint,
            observations=tuple(observations),
            decision=decision,
            decision_error=decision_error,
        )


class TraceReplayer:
    """Replay a trace and aggregate observed route economics and risk."""

    def __init__(self, executor: ShadowExecutor) -> None:
        if not isinstance(executor, ShadowExecutor):
            raise ValueError("executor must be ShadowExecutor")
        self.executor = executor

    def replay(
        self,
        requests: Iterable[ShadowRequest],
        targets: Iterable[ShadowTarget],
    ) -> TraceReplaySummary:
        targets = tuple(targets)
        stats: dict[str, ReplayRouteStats] = defaultdict(ReplayRouteStats)
        results: list[ShadowResult] = []

        for request in requests:
            result = self.executor.execute(request, targets)
            results.append(result)
            chosen = result.decision.chosen if result.decision else None
            report_by_name = (
                {report.name: report for report in result.decision.reports}
                if result.decision is not None
                else {}
            )
            for observation in result.observations:
                route = stats[observation.target]
                route.attempts += 1
                if observation.status == "skipped":
                    route.skipped += 1
                    continue
                if observation.status == "error":
                    route.errors += 1
                    continue

                route.successes += 1
                candidate = observation.candidate
                if candidate is None:
                    continue
                route.token_cost_usd += candidate.token_cost_usd
                route.memory_cost_usd += candidate.memory_cost_usd
                route.latency_ms += candidate.latency_ms
                route.htokens += candidate.htokens
                route.quality += candidate.expected_quality
                report = report_by_name.get(observation.target)
                if report is not None and report.admissible:
                    route.admissible += 1
                if observation.target == chosen:
                    route.chosen += 1

        return TraceReplaySummary(results=tuple(results), route_stats=dict(stats))
