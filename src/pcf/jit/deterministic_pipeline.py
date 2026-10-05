"""End-to-end preparation and shadow evaluation for deterministic JIT candidates.

This module deliberately stops at the evidence boundary. It can prepare a verified
artifact, register it in SHADOW, replay it against a baseline, and mark it ELIGIBLE
when existing promotion/fingerprint gates pass. It does not send production traffic.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Iterable, Mapping

from .deterministic import (
    DeterministicArtifact,
    DeterministicExecutionError,
    DeterministicPolicy,
    compile_deterministic_artifact,
    execute_deterministic_artifact,
)
from .fingerprint import FingerprintAuditSummary
from .hotpath import CompileEconomics, CompileEligibility, HotPathDetector
from .promotion import PromotionDecision, PromotionEvaluator
from .registry import RegisteredRoute, RouteRegistry, RouteState
from .shadow import (
    ShadowAssessor,
    ShadowExecutor,
    ShadowRequest,
    ShadowRun,
    ShadowTarget,
    TraceReplaySummary,
    TraceReplayer,
)
from .types import ExecutionPrimitive


class DeterministicPipelineError(RuntimeError):
    """A deterministic candidate failed a JIT build or lifecycle precondition."""


@dataclass(frozen=True)
class DeterministicValidationExample:
    case_id: str
    inputs: Mapping[str, Any]
    expected_output: Any

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id must be non-empty")
        if not isinstance(self.inputs, Mapping):
            raise ValueError("inputs must be a mapping")


@dataclass(frozen=True)
class DeterministicBuildPolicy:
    min_validation_examples: int = 20

    def __post_init__(self) -> None:
        if type(self.min_validation_examples) is not int or self.min_validation_examples <= 0:
            raise ValueError("min_validation_examples must be a positive integer")


@dataclass(frozen=True)
class DeterministicBuildEvidence:
    fingerprint: str
    calls: int
    stability: float
    audit_observations: int
    validation_examples: int
    validation_matches: int
    projected_future_calls: int
    breakeven_calls: float
    projected_net_savings_usd: float


@dataclass(frozen=True)
class DeterministicCandidateBuild:
    candidate_key: str
    fingerprint: str
    artifact: DeterministicArtifact
    evidence: DeterministicBuildEvidence


@dataclass(frozen=True)
class DeterministicShadowEvaluation:
    summary: TraceReplaySummary
    promotion: PromotionDecision


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DeterministicPipelineError("validation value is not finite JSON") from exc


def verify_deterministic_artifact(
    artifact: DeterministicArtifact,
    *,
    policy: DeterministicPolicy | None = None,
) -> None:
    """Rebuild an artifact from exposed content and verify its frozen identity fields.

    ``DeterministicArtifact`` currently exposes a mapping for the expression. Rebuilding
    immediately before use detects post-compile mutation or hand-constructed/tampered
    artifacts without trusting the stored hash or structural counters.
    """
    if not isinstance(artifact, DeterministicArtifact):
        raise DeterministicExecutionError("artifact must be DeterministicArtifact")
    dependency_map = dict(artifact.dependencies)
    if len(dependency_map) != len(artifact.dependencies):
        raise DeterministicExecutionError("artifact contains duplicate dependency names")
    try:
        rebuilt = compile_deterministic_artifact(
            name=artifact.name,
            version=artifact.version,
            expression=artifact.expression,
            dependencies=dependency_map,
            policy=policy,
        )
    except Exception as exc:
        raise DeterministicExecutionError("artifact integrity validation failed") from exc
    if (
        rebuilt.artifact_sha256 != artifact.artifact_sha256
        or rebuilt.node_count != artifact.node_count
        or rebuilt.max_depth != artifact.max_depth
        or rebuilt.dependencies != artifact.dependencies
    ):
        raise DeterministicExecutionError("artifact integrity validation failed")


def execute_verified_deterministic_artifact(
    artifact: DeterministicArtifact,
    inputs: Mapping[str, Any],
    *,
    policy: DeterministicPolicy | None = None,
) -> Any:
    verify_deterministic_artifact(artifact, policy=policy)
    return execute_deterministic_artifact(artifact, inputs, policy=policy)


def prepare_deterministic_candidate(
    *,
    candidate_key: str,
    artifact_name: str,
    artifact_version: str,
    expression: Mapping[str, Any],
    fingerprint: str,
    detector: HotPathDetector,
    audit: FingerprintAuditSummary,
    economics: CompileEconomics,
    projected_future_calls: int,
    validation_examples: Iterable[DeterministicValidationExample],
    dependencies: Mapping[str, str] | None = None,
    build_policy: DeterministicBuildPolicy | None = None,
    artifact_policy: DeterministicPolicy | None = None,
) -> DeterministicCandidateBuild:
    """Build a deterministic candidate only after hot-path, audit, ROI, and exactness gates."""
    if not isinstance(candidate_key, str) or not candidate_key.strip():
        raise DeterministicPipelineError("candidate_key must be non-empty")
    if not isinstance(detector, HotPathDetector):
        raise DeterministicPipelineError("detector must be HotPathDetector")
    if not isinstance(audit, FingerprintAuditSummary):
        raise DeterministicPipelineError("audit must be FingerprintAuditSummary")
    if not isinstance(economics, CompileEconomics):
        raise DeterministicPipelineError("economics must be CompileEconomics")
    if type(projected_future_calls) is not int or projected_future_calls < 0:
        raise DeterministicPipelineError("projected_future_calls must be a non-negative integer")

    eligibility: CompileEligibility = detector.evaluate_compile_eligibility(fingerprint, audit)
    if not eligibility.eligible:
        raise DeterministicPipelineError(
            "hot path is not compile eligible: " + ",".join(eligibility.reasons)
        )
    if not economics.should_compile(projected_future_calls):
        raise DeterministicPipelineError("compile economics do not clear breakeven")

    artifact = compile_deterministic_artifact(
        name=artifact_name,
        version=artifact_version,
        expression=expression,
        dependencies=dependencies,
        policy=artifact_policy,
    )
    verify_deterministic_artifact(artifact, policy=artifact_policy)

    examples = tuple(validation_examples)
    policy = build_policy or DeterministicBuildPolicy()
    if len(examples) < policy.min_validation_examples:
        raise DeterministicPipelineError("insufficient deterministic validation examples")
    if len({example.case_id for example in examples}) != len(examples):
        raise DeterministicPipelineError("validation case_id values must be unique")

    matches = 0
    for example in examples:
        if not isinstance(example, DeterministicValidationExample):
            raise DeterministicPipelineError("validation_examples must contain DeterministicValidationExample")
        observed = execute_verified_deterministic_artifact(
            artifact,
            example.inputs,
            policy=artifact_policy,
        )
        if _canonical_json(observed) == _canonical_json(example.expected_output):
            matches += 1
    if matches != len(examples):
        raise DeterministicPipelineError(
            f"deterministic validation mismatch: {matches}/{len(examples)} exact matches"
        )

    projected_net = economics.savings_per_call_usd * projected_future_calls - economics.fixed_cost_usd
    if not math.isfinite(projected_net):
        raise DeterministicPipelineError("projected net savings must be finite")
    evidence = DeterministicBuildEvidence(
        fingerprint=fingerprint,
        calls=eligibility.calls,
        stability=eligibility.stability,
        audit_observations=eligibility.audit_observations,
        validation_examples=len(examples),
        validation_matches=matches,
        projected_future_calls=projected_future_calls,
        breakeven_calls=economics.breakeven_calls,
        projected_net_savings_usd=projected_net,
    )
    return DeterministicCandidateBuild(
        candidate_key=candidate_key,
        fingerprint=fingerprint,
        artifact=artifact,
        evidence=evidence,
    )


def make_deterministic_shadow_target(
    build: DeterministicCandidateBuild,
    *,
    token_cost_usd: float = 0.0,
    memory_cost_usd: float = 0.0,
    artifact_policy: DeterministicPolicy | None = None,
    authority_metadata_key: str = "authority_ok",
    freshness_metadata_key: str = "freshness_ok",
) -> ShadowTarget:
    """Adapt a verified deterministic artifact to shadow execution without self-authorizing."""
    if not isinstance(build, DeterministicCandidateBuild):
        raise DeterministicPipelineError("build must be DeterministicCandidateBuild")
    for name, value in (("token_cost_usd", token_cost_usd), ("memory_cost_usd", memory_cost_usd)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise DeterministicPipelineError(f"{name} must be finite and non-negative")

    def runner(request: ShadowRequest) -> ShadowRun:
        if request.fingerprint != build.fingerprint:
            raise DeterministicExecutionError("shadow request fingerprint does not match artifact")
        if not isinstance(request.payload, Mapping):
            raise DeterministicExecutionError("deterministic shadow payload must be a mapping")
        output = execute_verified_deterministic_artifact(
            build.artifact,
            request.payload,
            policy=artifact_policy,
        )
        return ShadowRun(
            output=output,
            token_cost_usd=float(token_cost_usd),
            memory_cost_usd=float(memory_cost_usd),
            authority_ok=request.metadata.get(authority_metadata_key) is True,
            freshness_ok=request.metadata.get(freshness_metadata_key) is True,
            metadata={
                "deterministic_verified": True,
                "artifact_sha256": build.artifact.artifact_sha256,
            },
        )

    return ShadowTarget(
        name=build.candidate_key,
        primitive=ExecutionPrimitive.DETERMINISTIC_CODE,
        runner=runner,
        shadow_safe=True,
    )


def register_deterministic_shadow_candidate(
    registry: RouteRegistry,
    build: DeterministicCandidateBuild,
    *,
    now: float,
) -> RegisteredRoute:
    """Register only non-sensitive build evidence; validation inputs/outputs are never stored."""
    if not isinstance(registry, RouteRegistry):
        raise DeterministicPipelineError("registry must be RouteRegistry")
    if not isinstance(build, DeterministicCandidateBuild):
        raise DeterministicPipelineError("build must be DeterministicCandidateBuild")
    return registry.register(
        name=build.artifact.name,
        route_version=build.artifact.version,
        task_fingerprint=build.fingerprint,
        primitive=ExecutionPrimitive.DETERMINISTIC_CODE,
        now=now,
        candidate_key=build.candidate_key,
        metadata={
            "artifact_sha256": build.artifact.artifact_sha256,
            "artifact_nodes": build.artifact.node_count,
            "artifact_depth": build.artifact.max_depth,
            "validation_examples": build.evidence.validation_examples,
            "projected_net_savings_usd": build.evidence.projected_net_savings_usd,
        },
    )


def evaluate_deterministic_shadow(
    build: DeterministicCandidateBuild,
    *,
    requests: Iterable[ShadowRequest],
    baseline: ShadowTarget,
    assessor: ShadowAssessor,
    projected_future_calls: int,
    promotion_evaluator: PromotionEvaluator | None = None,
    token_cost_usd: float = 0.0,
    memory_cost_usd: float = 0.0,
    artifact_policy: DeterministicPolicy | None = None,
) -> DeterministicShadowEvaluation:
    """Replay the deterministic candidate against a baseline and compute promotion evidence."""
    if baseline.name == build.candidate_key:
        raise DeterministicPipelineError("baseline name must differ from deterministic candidate_key")
    replay_requests = tuple(requests)
    if not replay_requests:
        raise DeterministicPipelineError("at least one shadow request is required")
    if any(request.fingerprint != build.fingerprint for request in replay_requests):
        raise DeterministicPipelineError("all shadow requests must match the candidate fingerprint")

    candidate = make_deterministic_shadow_target(
        build,
        token_cost_usd=token_cost_usd,
        memory_cost_usd=memory_cost_usd,
        artifact_policy=artifact_policy,
    )
    summary = TraceReplayer(ShadowExecutor(assessor)).replay(
        replay_requests,
        (candidate, baseline),
    )
    evaluator = promotion_evaluator or PromotionEvaluator()
    promotion = evaluator.evaluate(
        summary,
        candidate=build.candidate_key,
        baseline=baseline.name,
        projected_future_calls=projected_future_calls,
    )
    return DeterministicShadowEvaluation(summary=summary, promotion=promotion)


def mark_deterministic_candidate_eligible(
    registry: RouteRegistry,
    route: RegisteredRoute,
    build: DeterministicCandidateBuild,
    evaluation: DeterministicShadowEvaluation,
    *,
    fingerprint_audit: FingerprintAuditSummary,
    evidence_ref: str,
    now: float,
) -> RegisteredRoute:
    """Advance SHADOW -> ELIGIBLE only through existing registry evidence gates."""
    if route.state is not RouteState.SHADOW:
        raise DeterministicPipelineError("route must be in shadow state")
    if route.candidate_key != build.candidate_key or route.task_fingerprint != build.fingerprint:
        raise DeterministicPipelineError("registered route does not match deterministic build")
    if route.metadata.get("artifact_sha256") != build.artifact.artifact_sha256:
        raise DeterministicPipelineError("registered artifact identity does not match build")
    return registry.mark_eligible(
        route.route_id,
        promotion=evaluation.promotion,
        fingerprint_audit=fingerprint_audit,
        evidence_ref=evidence_ref,
        now=now,
        expected_generation=route.generation,
    )
