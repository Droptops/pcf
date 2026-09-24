"""Model/rubric/version-bound held-out calibration reports."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from ..descriptor import hash_object
from ..validation import integer, nonempty, number
from .calibration import brier_score, expected_calibration_error


@dataclass(frozen=True)
class ValidationSample:
    context: object
    candidate: object
    label: int
    is_tail: bool

    def __post_init__(self):
        if type(self.label) is not int or self.label not in (0, 1) or type(self.is_tail) is not bool:
            raise ValueError("validation requires binary labels and an explicit tail flag")


@dataclass(frozen=True)
class CalibrationRecord:
    source_fingerprint: str
    candidate_fingerprint: str
    dataset_id: str
    rubric_id: str
    threshold: float
    n_samples: int
    n_tail: int
    n_selected: int
    n_tail_selected: int
    ece: float
    tail_ece: float
    brier: float
    selected_quality: float | None
    tail_selected_quality: float | None
    max_ece: float
    quality_floor: float
    passed: bool
    failures: tuple[str, ...]
    sample_digest: str

    @property
    def validation_id(self):
        return hash_object("pcf:validation:0.2", asdict(self))

    def to_json(self):
        import json
        from ..segments import canonical_bytes
        result = json.loads(canonical_bytes(asdict(self)))
        result["validation_id"] = self.validation_id
        return result


def validate_source(source, samples, *, dataset_id, threshold=.8, max_ece=.10,
                    quality_floor=.75, min_samples=100, min_tail_samples=30, min_selected_samples=20):
    samples = list(samples)
    nonempty(dataset_id, "dataset_id")
    for name, value in [("threshold", threshold), ("max_ece", max_ece), ("quality_floor", quality_floor)]:
        number(value, name, maximum=1)
    for name, value in [("min_samples", min_samples), ("min_tail_samples", min_tail_samples), ("min_selected_samples", min_selected_samples)]:
        integer(value, name, minimum=1)
    if not samples or any(not isinstance(s, ValidationSample) for s in samples):
        raise ValueError("validation needs ValidationSample rows")
    if len({s.candidate.fingerprint for s in samples}) != 1:
        raise ValueError("validate each candidate separately")
    candidate = samples[0].candidate
    if not source.can_validate(candidate):
        raise ValueError("source must be fitted/versioned for this candidate before validation")
    if any(s.context.prefix_chain()[-1] in getattr(source, "training_contexts", set()) for s in samples):
        raise ValueError("validation contexts overlap calibration fitting data")
    probs = [number(source.p_sufficient(s.context, s.candidate), "p_sufficient", maximum=1) for s in samples]
    labels = [s.label for s in samples]
    tail = [(p, s.label) for p, s in zip(probs, samples) if s.is_tail]
    if not tail:
        raise ValueError("held-out validation needs a tail slice")
    selected = [s.label for p, s in zip(probs, samples) if p >= threshold]
    tail_selected = [s.label for p, s in zip(probs, samples) if p >= threshold and s.is_tail]
    quality = sum(selected) / len(selected) if selected else None
    tail_quality = sum(tail_selected) / len(tail_selected) if tail_selected else None
    ece = expected_calibration_error(probs, labels)
    tail_ece = expected_calibration_error([p for p, _ in tail], [y for _, y in tail])
    failures = []
    if len(samples) < min_samples:
        failures.append("insufficient validation samples")
    if len(tail) < min_tail_samples:
        failures.append("insufficient tail samples")
    if ece > max_ece or tail_ece > max_ece:
        failures.append("calibration error exceeds bound")
    for name, rows, value in [("selected", selected, quality), ("tail-selected", tail_selected, tail_quality)]:
        if rows and (len(rows) < min_selected_samples or value < quality_floor):
            failures.append(f"{name} quality or sample count below requirement")
    record = CalibrationRecord(source.fingerprint, candidate.fingerprint, dataset_id, source.rubric_id,
                               threshold, len(samples), len(tail), len(selected), len(tail_selected),
                               ece, tail_ece, brier_score(probs, labels), quality, tail_quality,
                               max_ece, quality_floor, not failures, tuple(failures),
                               hash_object("pcf:validation-samples:0.2", [[s.context.prefix_chain()[-1], s.label, s.is_tail] for s in samples]))
    source._records[(candidate.fingerprint, threshold)] = record
    return record
