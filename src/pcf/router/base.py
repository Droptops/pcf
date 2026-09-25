"""Validated routing policy with separate fallback eligibility and confidence."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

from ..cache import PrefixCache
from ..compiler import ContextCompiler, UnsupportedRequest
from ..descriptor import hash_object
from ..segments import Context
from ..validation import nonempty, number


class ConfidenceUnavailable(RuntimeError):
    """Transient scorer failure; the router may select its policy fallback."""


@dataclass(frozen=True)
class Candidate:
    compiler: ContextCompiler
    cache: PrefixCache
    input_price_per_mtok: float
    cache_read_price_per_mtok: float
    is_fallback: bool = False
    cache_write_price_per_mtok: float | None = None

    def __post_init__(self):
        number(self.input_price_per_mtok, "input_price_per_mtok")
        number(self.cache_read_price_per_mtok, "cache_read_price_per_mtok")
        if self.cache_write_price_per_mtok is not None:
            number(self.cache_write_price_per_mtok, "cache_write_price_per_mtok")
        if type(self.is_fallback) is not bool:
            raise ValueError("is_fallback must be boolean")
        if not isinstance(self.compiler, ContextCompiler) or not isinstance(self.cache, PrefixCache):
            raise ValueError("candidate needs a compiler and a cache metadata store")

    @property
    def model_id(self):
        return self.compiler.descriptor.model_id

    @property
    def fingerprint(self):
        identity = {"compiler": self.compiler.candidate_fingerprint,
                    "max_tokens": getattr(self.compiler, "max_tokens", None)}
        generation = self.compiler.generation_identity  # non-default reasoning settings change what the model does
        return hash_object("pcf:quality-candidate:0.2", {**identity, **({"generation": generation} if generation else {})})

    @property
    def write_price(self):
        return self.cache_write_price_per_mtok if self.cache_write_price_per_mtok is not None else (
            self.input_price_per_mtok * self.compiler.descriptor.cache_write_multiplier)


@dataclass(frozen=True)
class CandidateReport:
    family: str
    model_id: str
    compat_key: str
    warm_prefix_segments: int
    warm_tokens: int
    cold_tokens: int
    cache_creation_tokens: int
    uncached_tokens: int
    est_input_cost_usd: float
    p_sufficient: float | None
    score: float | None
    token_count_is_estimate: bool
    cache_state: str
    observed_at: float
    calibration_valid: bool
    validation_id: str | None
    confidence_error: str | None = None

    def to_json(self):
        return asdict(self)


@dataclass(frozen=True)
class RouteDecision:
    candidates: tuple[CandidateReport, ...]
    chosen: str
    confidence: float | None
    escalate: bool
    escalation_threshold: float
    confidence_source: str
    source_fingerprint: str
    reason: str
    request_id: str | None = None
    decision_version: str = "0.2"

    def to_json(self):
        result = asdict(self)
        result["candidates"] = [c.to_json() for c in self.candidates]
        if self.request_id is None:
            result.pop("request_id")
        return result


class ConfidenceSource(ABC):
    name = "custom"

    def __init__(self, *, source_version="unversioned", rubric_id="unspecified"):
        self.source_version = nonempty(source_version, "source_version")
        self.rubric_id = nonempty(rubric_id, "rubric_id")
        self._records = {}

    @property
    def fingerprint(self):
        return hash_object("pcf:confidence-source:0.2", {"name": self.name,
                           "version": getattr(self, "source_version", "unversioned"),
                           "rubric": getattr(self, "rubric_id", "unspecified")})

    def can_validate(self, candidate):
        return getattr(self, "source_version", "unversioned") != "unversioned" and self.name != "uncalibrated"

    def validate(self, samples, **kwargs):
        from .evaluation import validate_source
        if not hasattr(self, "_records"):
            self._records = {}
        return validate_source(self, samples, **kwargs)

    def context_key(self, ctx: Context) -> str:
        """Identity of what this source scores; held-out rows must differ in it to count as independent evidence."""
        return ctx.prefix_chain()[-1]

    def validation_for(self, candidate, threshold):
        record = getattr(self, "_records", {}).get((candidate.fingerprint, threshold))
        return record if record and record.passed and record.source_fingerprint == self.fingerprint else None

    @abstractmethod
    def p_sufficient(self, ctx: Context, candidate: Candidate) -> float: ...


class Router:
    def __init__(self, candidates, confidence: ConfidenceSource, threshold=.8, *, score_unvalidated: bool = False):
        """``score_unvalidated=True`` also queries the scorer for candidates without a passing validation record
        (the score is reported but never used to route), e.g. to collect calibration data; by default those
        candidates are not scored, since a paid confidence call cannot change the decision."""
        candidates = tuple(candidates)
        if not candidates or any(not isinstance(c, Candidate) for c in candidates):
            raise ValueError("router requires Candidate entries")
        if sum(c.is_fallback for c in candidates) != 1:
            raise ValueError("exactly one candidate must be marked is_fallback")
        if len({c.model_id for c in candidates}) != len(candidates):
            raise ValueError("candidate model identifiers must be unique")
        if not isinstance(confidence, ConfidenceSource):
            raise ValueError("confidence must be a ConfidenceSource")
        self.candidates = candidates
        self.confidence = confidence
        self.threshold = number(threshold, "threshold", maximum=1)
        if type(score_unvalidated) is not bool:
            raise ValueError("score_unvalidated must be boolean")
        self.score_unvalidated = score_unvalidated

    def route(self, ctx: Context, now: float, request_id=None) -> RouteDecision:
        number(now, "now")
        if request_id is not None:
            nonempty(request_id, "request_id")
        reports = []
        for cand in self.candidates:
            try:
                w = cand.compiler.warmth(ctx, cand.cache, now)
            except UnsupportedRequest:
                if cand.is_fallback:
                    raise
                continue  # this target cannot serve the context; the others still can
            cost = (w.uncached_tokens * cand.input_price_per_mtok + w.cache_creation_tokens * cand.write_price
                    + w.warm_tokens * cand.cache_read_price_per_mtok) / 1e6
            record = self.confidence.validation_for(cand, self.threshold)
            score, error = None, None
            if record is not None or self.score_unvalidated:
                try:
                    score = number(self.confidence.p_sufficient(ctx, cand), "p_sufficient", maximum=1)
                except ConfidenceUnavailable as exc:
                    record, error = None, str(exc)
            p = score if record is not None else None
            d = cand.compiler.descriptor
            report = CandidateReport(d.family, d.model_id, d.compat_key, w.warm_prefix_segments,
                                     w.warm_tokens, w.cold_tokens, w.cache_creation_tokens, w.uncached_tokens,
                                     cost, p, score, cand.compiler.tokenizer.is_estimate or d.identity_kind != "simulated",
                                     w.cache_state, w.observed_at, record is not None,
                                     record.validation_id if record else None, error)
            reports.append((cand, report))
        qualified = [(c, r) for c, r in reports if r.p_sufficient is not None and r.p_sufficient >= self.threshold]
        if qualified:
            _, selected = min(qualified, key=lambda cr: (cr[1].est_input_cost_usd, cr[1].model_id))
            escalate, reason = False, "lowest estimated input cost among validated candidates above threshold"
        else:
            _, selected = next((c, r) for c, r in reports if c.is_fallback)
            escalate, reason = True, "policy fallback: no validated candidate meets threshold; quality not guaranteed"
        return RouteDecision(tuple(r for _, r in reports), selected.model_id, selected.p_sufficient, escalate,
                             self.threshold, self.confidence.name, self.confidence.fingerprint, reason, request_id)
