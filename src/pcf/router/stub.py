"""Offline scorers. Fitted parameters and held-out validation are distinct states."""
from __future__ import annotations

from ..descriptor import hash_object
from .base import ConfidenceSource, ConfidenceUnavailable
from .calibration import apply_platt, fit_platt


class PlattScaledSource(ConfidenceSource):
    name = "stub-platt-scaled"

    def __init__(self, raw, a=1.0, b=0.0, *, scorer_version="unversioned", rubric_id="task-sufficiency"):
        super().__init__(source_version=scorer_version, rubric_id=rubric_id)
        self._raw = raw
        self.a, self.b = a, b
        self.fitted_candidate = None
        self.training_contexts = set()
        self._generation = 0

    @property
    def raw(self):
        return self._raw

    @raw.setter
    def raw(self, value):
        self._raw = value
        self._generation += 1
        self._records.clear()
        self.fitted_candidate = None
        self.training_contexts.clear()

    @property
    def fingerprint(self):
        return hash_object("pcf:platt-source:0.2", {"base": super().fingerprint, "a": self.a, "b": self.b,
                           "generation": self._generation, "candidate": self.fitted_candidate})

    def fit(self, samples):
        samples = list(samples)
        self._records.clear()
        self.fitted_candidate = None
        self.training_contexts.clear()
        if not samples or len({cand.fingerprint for _, cand, _ in samples}) != 1:
            raise ValueError("fit requires samples for exactly one candidate")
        raw = [self.raw(ctx, cand) for ctx, cand, _ in samples]
        self.a, self.b = fit_platt(raw, [y for _, _, y in samples])
        self.fitted_candidate = samples[0][1].fingerprint
        self.training_contexts = {ctx.prefix_chain()[-1] for ctx, _, _ in samples}
        return self.a, self.b

    def can_validate(self, candidate):
        return super().can_validate(candidate) and self.fitted_candidate == candidate.fingerprint

    def p_sufficient(self, ctx, candidate):
        if self.fitted_candidate is not None and self.fitted_candidate != candidate.fingerprint:
            raise ConfidenceUnavailable("calibrator fitted for a different candidate")
        return apply_platt(self.raw(ctx, candidate), self.a, self.b)


class UncalibratedSource(ConfidenceSource):
    name = "uncalibrated"
    def __init__(self, raw):
        super().__init__()
        self.raw = raw
    def p_sufficient(self, ctx, candidate):
        return self.raw(ctx, candidate)
