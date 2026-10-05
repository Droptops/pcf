"""Bounded synthesis of deterministic JIT expressions from adjudicated examples.

The synthesizer enumerates only a tiny declarative grammar. It never emits source
code and it fails on ambiguity instead of guessing between equally simple programs.
Training examples are used only to infer the expression; callers must still pass a
separate held-out validation set through ``prepare_deterministic_candidate``.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable, Mapping, Sequence

from .deterministic import compile_deterministic_artifact, execute_deterministic_artifact
from .deterministic_pipeline import (
    DeterministicBuildPolicy,
    DeterministicCandidateBuild,
    DeterministicPipelineError,
    DeterministicValidationExample,
    prepare_deterministic_candidate,
)
from .fingerprint import FingerprintAuditSummary
from .hotpath import CompileEconomics, HotPathDetector


SYNTHESIS_VERSION = "pcf.deterministic-synthesis.v1"
_FORBIDDEN_PATH_PARTS = {"", "__class__", "__dict__", "__globals__"}


class DeterministicSynthesisError(RuntimeError):
    """No unique safe program can be synthesized from the supplied examples."""


@dataclass(frozen=True)
class DeterministicSynthesisPolicy:
    min_training_examples: int = 20
    max_input_paths: int = 128
    max_object_fields: int = 64
    max_array_items: int = 64
    allow_constants: bool = False

    def __post_init__(self) -> None:
        for name in (
            "min_training_examples",
            "max_input_paths",
            "max_object_fields",
            "max_array_items",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.allow_constants) is not bool:
            raise ValueError("allow_constants must be boolean")


@dataclass(frozen=True)
class DeterministicSynthesisEvidence:
    synthesis_version: str
    training_examples: int
    input_paths_considered: int
    expression_nodes: int


@dataclass(frozen=True)
class DeterministicSynthesisResult:
    expression: Mapping[str, Any]
    evidence: DeterministicSynthesisEvidence


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DeterministicSynthesisError("examples must contain finite JSON values") from exc


def _flatten_inputs(value: Mapping[str, Any], *, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or "." in key or key in _FORBIDDEN_PATH_PARTS:
            raise DeterministicSynthesisError("input keys must be non-empty safe path components")
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            flattened.update(_flatten_inputs(item, prefix=path))
        else:
            flattened[path] = item
    return flattened


def _node_count(expr: Mapping[str, Any]) -> int:
    artifact = compile_deterministic_artifact(
        name="synthesis_probe",
        version="1",
        expression=expr,
        dependencies={"synthesizer": SYNTHESIS_VERSION},
    )
    return artifact.node_count


def _candidate_leaf_expressions(
    expected: Sequence[Any],
    values_by_path: Mapping[str, Sequence[Any]],
    *,
    policy: DeterministicSynthesisPolicy,
) -> list[Mapping[str, Any]]:
    candidates: list[Mapping[str, Any]] = []
    expected_canonical = [_canonical(value) for value in expected]

    if policy.allow_constants and len(set(expected_canonical)) == 1:
        candidates.append({"op": "const", "value": expected[0]})

    for path in sorted(values_by_path):
        values = values_by_path[path]
        if [_canonical(value) for value in values] == expected_canonical:
            candidates.append({"op": "input", "path": path})

        if all(isinstance(value, str) for value in values) and all(
            isinstance(value, str) for value in expected
        ):
            transforms = (
                ("strip", lambda value: value.strip()),
                ("lower", lambda value: value.lower()),
                ("upper", lambda value: value.upper()),
                ("strip_lower", lambda value: value.strip().lower()),
                ("strip_upper", lambda value: value.strip().upper()),
            )
            for name, transform in transforms:
                observed = [transform(value) for value in values]
                if [_canonical(value) for value in observed] != expected_canonical:
                    continue
                base: Mapping[str, Any] = {"op": "input", "path": path}
                if name == "strip":
                    expr = {"op": "strip", "arg": base}
                elif name == "lower":
                    expr = {"op": "lower", "arg": base}
                elif name == "upper":
                    expr = {"op": "upper", "arg": base}
                elif name == "strip_lower":
                    expr = {"op": "lower", "arg": {"op": "strip", "arg": base}}
                else:
                    expr = {"op": "upper", "arg": {"op": "strip", "arg": base}}
                candidates.append(expr)

    if not candidates:
        return []

    by_complexity: dict[int, list[Mapping[str, Any]]] = {}
    for candidate in candidates:
        by_complexity.setdefault(_node_count(candidate), []).append(candidate)
    simplest = by_complexity[min(by_complexity)]

    unique: dict[str, Mapping[str, Any]] = {}
    for candidate in simplest:
        unique[_canonical(candidate)] = candidate
    return [unique[key] for key in sorted(unique)]


def _synthesize_value(
    expected: Sequence[Any],
    values_by_path: Mapping[str, Sequence[Any]],
    *,
    policy: DeterministicSynthesisPolicy,
) -> Mapping[str, Any]:
    if all(isinstance(value, Mapping) for value in expected):
        key_sets = [set(value.keys()) for value in expected]
        if not key_sets or any(keys != key_sets[0] for keys in key_sets[1:]):
            raise DeterministicSynthesisError("output object shape is not stable")
        keys = sorted(key_sets[0])
        if len(keys) > policy.max_object_fields:
            raise DeterministicSynthesisError("output object exceeds synthesis field limit")
        fields = {
            key: _synthesize_value(
                [value[key] for value in expected],
                values_by_path,
                policy=policy,
            )
            for key in keys
        }
        return {"op": "object", "fields": fields}

    if all(isinstance(value, list) for value in expected):
        lengths = {len(value) for value in expected}
        if len(lengths) != 1:
            raise DeterministicSynthesisError("output array length is not stable")
        length = next(iter(lengths))
        if length > policy.max_array_items:
            raise DeterministicSynthesisError("output array exceeds synthesis item limit")
        items = [
            _synthesize_value(
                [value[index] for value in expected],
                values_by_path,
                policy=policy,
            )
            for index in range(length)
        ]
        return {"op": "array", "items": items}

    if any(isinstance(value, (Mapping, list)) for value in expected):
        raise DeterministicSynthesisError("output type is not stable")

    candidates = _candidate_leaf_expressions(expected, values_by_path, policy=policy)
    if not candidates:
        raise DeterministicSynthesisError("no supported deterministic expression fits examples")
    if len(candidates) > 1:
        raise DeterministicSynthesisError("ambiguous deterministic expression")
    return candidates[0]


def synthesize_deterministic_expression(
    training_examples: Iterable[DeterministicValidationExample],
    *,
    policy: DeterministicSynthesisPolicy | None = None,
) -> DeterministicSynthesisResult:
    """Infer the simplest unique expression in the bounded synthesis grammar."""
    policy = policy or DeterministicSynthesisPolicy()
    examples = tuple(training_examples)
    if len(examples) < policy.min_training_examples:
        raise DeterministicSynthesisError("insufficient synthesis training examples")
    if len({example.case_id for example in examples}) != len(examples):
        raise DeterministicSynthesisError("training case_id values must be unique")
    if not all(isinstance(example, DeterministicValidationExample) for example in examples):
        raise DeterministicSynthesisError("training examples have invalid type")

    flattened = [_flatten_inputs(example.inputs) for example in examples]
    common_paths = set(flattened[0])
    for item in flattened[1:]:
        common_paths &= set(item)
    if not common_paths:
        raise DeterministicSynthesisError("training examples have no common input paths")
    if len(common_paths) > policy.max_input_paths:
        raise DeterministicSynthesisError("common input paths exceed synthesis limit")

    values_by_path = {
        path: tuple(item[path] for item in flattened)
        for path in sorted(common_paths)
    }
    expected = tuple(example.expected_output for example in examples)
    expression = _synthesize_value(expected, values_by_path, policy=policy)

    probe = compile_deterministic_artifact(
        name="synthesis_probe",
        version="1",
        expression=expression,
        dependencies={"synthesizer": SYNTHESIS_VERSION},
    )
    for example in examples:
        observed = execute_deterministic_artifact(probe, example.inputs)
        if _canonical(observed) != _canonical(example.expected_output):
            raise DeterministicSynthesisError("synthesized expression failed training replay")

    return DeterministicSynthesisResult(
        expression=expression,
        evidence=DeterministicSynthesisEvidence(
            synthesis_version=SYNTHESIS_VERSION,
            training_examples=len(examples),
            input_paths_considered=len(common_paths),
            expression_nodes=probe.node_count,
        ),
    )


def synthesize_and_prepare_deterministic_candidate(
    *,
    candidate_key: str,
    artifact_name: str,
    artifact_version: str,
    fingerprint: str,
    detector: HotPathDetector,
    audit: FingerprintAuditSummary,
    economics: CompileEconomics,
    projected_future_calls: int,
    training_examples: Iterable[DeterministicValidationExample],
    validation_examples: Iterable[DeterministicValidationExample],
    synthesis_policy: DeterministicSynthesisPolicy | None = None,
    build_policy: DeterministicBuildPolicy | None = None,
    dependencies: Mapping[str, str] | None = None,
) -> tuple[DeterministicCandidateBuild, DeterministicSynthesisEvidence]:
    """Synthesize from training data, then require separate held-out pipeline validation."""
    synthesis = synthesize_deterministic_expression(
        training_examples,
        policy=synthesis_policy,
    )
    merged_dependencies = dict(dependencies or {})
    if "synthesizer" in merged_dependencies and merged_dependencies["synthesizer"] != SYNTHESIS_VERSION:
        raise DeterministicPipelineError("synthesizer dependency is reserved")
    merged_dependencies["synthesizer"] = SYNTHESIS_VERSION
    build = prepare_deterministic_candidate(
        candidate_key=candidate_key,
        artifact_name=artifact_name,
        artifact_version=artifact_version,
        expression=synthesis.expression,
        fingerprint=fingerprint,
        detector=detector,
        audit=audit,
        economics=economics,
        projected_future_calls=projected_future_calls,
        validation_examples=validation_examples,
        dependencies=merged_dependencies,
        build_policy=build_policy,
    )
    return build, synthesis.evidence
