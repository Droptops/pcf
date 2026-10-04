"""Risk-aware semantic/result cache primitives for the JIT optimizer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class CacheEnvelope:
    value: Any
    created_at: float
    max_age_seconds: float
    dependency_versions: Mapping[str, str] = field(default_factory=dict)
    policy_version: str = "v1"
    authority_scope: str = "default"
    expected_staleness_htokens: float = 0.0

    def valid_for(
        self,
        *,
        now: float,
        dependency_versions: Mapping[str, str],
        policy_version: str,
        authority_scope: str,
        max_staleness_htokens: float,
    ) -> bool:
        if now < self.created_at:
            return False
        if now - self.created_at > self.max_age_seconds:
            return False
        if dict(dependency_versions) != dict(self.dependency_versions):
            return False
        if policy_version != self.policy_version:
            return False
        if authority_scope != self.authority_scope:
            return False
        if self.expected_staleness_htokens > max_staleness_htokens:
            return False
        return True


class RiskAwareResultCache:
    """In-memory reference cache; production backends can implement the same contract."""

    def __init__(self) -> None:
        self._entries: dict[str, CacheEnvelope] = {}

    def put(self, key: str, envelope: CacheEnvelope) -> None:
        if not key:
            raise ValueError("cache key must be non-empty")
        self._entries[key] = envelope

    def get(
        self,
        key: str,
        *,
        now: float,
        dependency_versions: Mapping[str, str],
        policy_version: str,
        authority_scope: str,
        max_staleness_htokens: float,
    ) -> Any | None:
        envelope = self._entries.get(key)
        if envelope is None:
            return None
        if not envelope.valid_for(
            now=now,
            dependency_versions=dependency_versions,
            policy_version=policy_version,
            authority_scope=authority_scope,
            max_staleness_htokens=max_staleness_htokens,
        ):
            return None
        return envelope.value
