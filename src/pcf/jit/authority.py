"""Dependency-light FAAR authority verdict adapter for JIT hard admissibility gates."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from .shadow import ShadowRun

_FAAR_VERDICTS = frozenset({"ALLOW", "DENY", "DEFER", "STOP"})
_MAX_REASON_CODES = 64
_MAX_REASON_CODE_CHARS = 256
_RESERVED_METADATA_KEYS = frozenset({
    "authority_source",
    "authority_layer",
    "authority_verdict",
    "authority_reason_codes",
})


class AuthorityEvidenceError(ValueError):
    """External authority evidence is malformed or is not an authority-layer decision."""


@dataclass(frozen=True)
class AuthorityGateEvidence:
    """Normalized output of a trusted FAAR authority gate.

    PCF intentionally does not import FAAR. The integration boundary consumes the
    small public shape of FAAR's ``Decision``: verdict, reason_codes, and layer.
    Signature/attestation verification remains FAAR's responsibility.
    """

    verdict: str
    reason_codes: tuple[str, ...]
    layer: str = "authority"
    source: str = "faar"

    def __post_init__(self) -> None:
        if self.verdict not in _FAAR_VERDICTS:
            raise AuthorityEvidenceError(f"unsupported authority verdict: {self.verdict!r}")
        if self.layer != "authority":
            raise AuthorityEvidenceError("decision must come from the authority layer")
        if self.source != "faar":
            raise AuthorityEvidenceError("authority source must be faar")
        if not isinstance(self.reason_codes, tuple):
            raise AuthorityEvidenceError("reason_codes must be a tuple")
        if len(self.reason_codes) > _MAX_REASON_CODES:
            raise AuthorityEvidenceError("too many authority reason codes")
        for reason in self.reason_codes:
            if not isinstance(reason, str) or not reason or len(reason) > _MAX_REASON_CODE_CHARS:
                raise AuthorityEvidenceError("authority reason codes must be non-empty bounded strings")

    @property
    def authority_ok(self) -> bool:
        """Only an explicit FAAR ALLOW authorizes the route."""
        return self.verdict == "ALLOW"

    def metadata(self) -> dict[str, object]:
        return {
            "authority_source": self.source,
            "authority_layer": self.layer,
            "authority_verdict": self.verdict,
            "authority_reason_codes": self.reason_codes,
        }


def normalize_faar_authority_decision(decision: Any) -> AuthorityGateEvidence:
    """Normalize a FAAR ``Decision`` or equivalent mapping without importing FAAR.

    The caller must supply the result of FAAR's verified authority path. PCF does
    not verify FAAR attestations or reconstruct posture/primitive policy locally.
    Unknown/malformed inputs fail closed by raising ``AuthorityEvidenceError``.
    """

    verdict = _field(decision, "verdict")
    layer = _field(decision, "layer")
    reasons = _field(decision, "reason_codes")

    # FAAR uses StrEnum; str(StrEnumMember) is its value on supported Python, but
    # ``.value`` is used explicitly when present so foreign Enum reprs cannot leak in.
    verdict = getattr(verdict, "value", verdict)
    layer = getattr(layer, "value", layer)
    if not isinstance(verdict, str):
        raise AuthorityEvidenceError("authority verdict must be a string-like enum value")
    if not isinstance(layer, str):
        raise AuthorityEvidenceError("authority layer must be a string")
    if isinstance(reasons, list):
        reasons = tuple(reasons)
    if not isinstance(reasons, tuple):
        raise AuthorityEvidenceError("authority reason_codes must be a tuple or list")
    return AuthorityGateEvidence(
        verdict=verdict,
        reason_codes=reasons,
        layer=layer,
    )


def apply_faar_authority(run: ShadowRun, decision: Any) -> ShadowRun:
    """Bind FAAR authority evidence to a shadow run as a hard optimizer gate.

    This operation can only reduce authority. An existing ``authority_ok=False``
    remains false even when FAAR returns ALLOW. Reserved metadata collisions are
    rejected so upstream evidence cannot be silently overwritten.
    """

    if not isinstance(run, ShadowRun):
        raise AuthorityEvidenceError("run must be ShadowRun")
    evidence = normalize_faar_authority_decision(decision)
    metadata = dict(run.metadata)
    overlap = set(metadata).intersection(_RESERVED_METADATA_KEYS)
    if overlap:
        raise AuthorityEvidenceError(f"shadow run metadata collides with authority keys: {sorted(overlap)}")
    metadata.update(evidence.metadata())
    return replace(
        run,
        authority_ok=run.authority_ok and evidence.authority_ok,
        metadata=metadata,
    )


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            raise AuthorityEvidenceError(f"authority decision missing {name}")
        return value[name]
    try:
        return getattr(value, name)
    except (AttributeError, TypeError):
        raise AuthorityEvidenceError(f"authority decision missing {name}") from None
