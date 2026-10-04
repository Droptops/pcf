"""Task fingerprints plus collision and semantic-drift measurement for JIT routing."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..descriptor import hash_object
from ..validation import nonempty, number


@dataclass(frozen=True)
class TaskSignature:
    """Declared stable dimensions that define one semantic task class.

    Raw request values are intentionally absent. Callers must opt stable features in
    explicitly so a fingerprint does not accidentally become a hash of user data.
    """

    operation: str
    input_schema_id: str
    output_schema_id: str
    policy_version: str
    authority_scope: str
    tool_contract_id: str | None = None
    stable_features: Mapping[str, Any] = field(default_factory=dict)
    fingerprint_version: str = "0.1"

    def __post_init__(self) -> None:
        for name in ("operation", "input_schema_id", "output_schema_id", "policy_version", "authority_scope"):
            nonempty(getattr(self, name), name)
        if self.tool_contract_id is not None:
            nonempty(self.tool_contract_id, "tool_contract_id")
        if self.fingerprint_version != "0.1":
            raise ValueError("unsupported fingerprint_version")
        if not isinstance(self.stable_features, Mapping):
            raise ValueError("stable_features must be a mapping")

    @property
    def fingerprint(self) -> str:
        material = {
            "operation": self.operation,
            "input_schema_id": self.input_schema_id,
            "output_schema_id": self.output_schema_id,
            "policy_version": self.policy_version,
            "authority_scope": self.authority_scope,
            "tool_contract_id": self.tool_contract_id,
            "stable_features": dict(self.stable_features),
            "fingerprint_version": self.fingerprint_version,
        }
        return hash_object("pcf:jit-task-fingerprint:0.1", material)


@dataclass(frozen=True)
class FingerprintObservation:
    """Adjudicated ground truth used to audit a task fingerprint."""

    fingerprint: str
    semantic_id: str
    behavior_id: str
    observed_at: float

    def __post_init__(self) -> None:
        nonempty(self.fingerprint, "fingerprint")
        nonempty(self.semantic_id, "semantic_id")
        nonempty(self.behavior_id, "behavior_id")
        number(self.observed_at, "observed_at")


@dataclass(frozen=True)
class FingerprintAuditSummary:
    observations: int
    unique_fingerprints: int
    unique_semantics: int
    collision_fingerprints: int
    collision_rate: float
    drift_fingerprints: int
    drift_rate: float
    split_semantics: int
    split_rate: float
    collision_examples: Mapping[str, tuple[str, ...]]
    drift_examples: Mapping[str, tuple[str, ...]]
    split_examples: Mapping[str, tuple[str, ...]]

    @property
    def clean(self) -> bool:
        return self.collision_fingerprints == 0 and self.drift_fingerprints == 0


class FingerprintAudit:
    """Measure collision, drift, and over-splitting from adjudicated observations.

    Collision: one fingerprint maps to multiple semantic identities.
    Drift: one fingerprint/semantic identity maps to multiple behavior identities.
    Split: one semantic identity maps to multiple fingerprints.
    """

    def __init__(self) -> None:
        self._observations: list[FingerprintObservation] = []
        self._semantics_by_fingerprint: dict[str, set[str]] = defaultdict(set)
        self._behaviors_by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._fingerprints_by_semantic: dict[str, set[str]] = defaultdict(set)

    def observe(
        self,
        signature: TaskSignature,
        *,
        semantic_id: str,
        behavior_id: str,
        observed_at: float,
    ) -> FingerprintObservation:
        if not isinstance(signature, TaskSignature):
            raise ValueError("signature must be TaskSignature")
        return self.observe_fingerprint(
            signature.fingerprint,
            semantic_id=semantic_id,
            behavior_id=behavior_id,
            observed_at=observed_at,
        )

    def observe_fingerprint(
        self,
        fingerprint: str,
        *,
        semantic_id: str,
        behavior_id: str,
        observed_at: float,
    ) -> FingerprintObservation:
        observation = FingerprintObservation(fingerprint, semantic_id, behavior_id, observed_at)
        self._observations.append(observation)
        self._semantics_by_fingerprint[fingerprint].add(semantic_id)
        self._behaviors_by_pair[(fingerprint, semantic_id)].add(behavior_id)
        self._fingerprints_by_semantic[semantic_id].add(fingerprint)
        return observation

    def summary(self) -> FingerprintAuditSummary:
        collision_examples = {
            fingerprint: tuple(sorted(semantic_ids))
            for fingerprint, semantic_ids in self._semantics_by_fingerprint.items()
            if len(semantic_ids) > 1
        }

        drift_examples: dict[str, tuple[str, ...]] = {}
        drift_fingerprints: set[str] = set()
        for (fingerprint, semantic_id), behavior_ids in self._behaviors_by_pair.items():
            if len(behavior_ids) > 1:
                drift_fingerprints.add(fingerprint)
                drift_examples[f"{fingerprint}|{semantic_id}"] = tuple(sorted(behavior_ids))

        split_examples = {
            semantic_id: tuple(sorted(fingerprints))
            for semantic_id, fingerprints in self._fingerprints_by_semantic.items()
            if len(fingerprints) > 1
        }

        fingerprints = len(self._semantics_by_fingerprint)
        semantics = len(self._fingerprints_by_semantic)
        collisions = len(collision_examples)
        drifts = len(drift_fingerprints)
        splits = len(split_examples)
        return FingerprintAuditSummary(
            observations=len(self._observations),
            unique_fingerprints=fingerprints,
            unique_semantics=semantics,
            collision_fingerprints=collisions,
            collision_rate=collisions / fingerprints if fingerprints else 0.0,
            drift_fingerprints=drifts,
            drift_rate=drifts / fingerprints if fingerprints else 0.0,
            split_semantics=splits,
            split_rate=splits / semantics if semantics else 0.0,
            collision_examples=collision_examples,
            drift_examples=drift_examples,
            split_examples=split_examples,
        )
