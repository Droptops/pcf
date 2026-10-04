"""OpenRouter Decisions API adapter for advisory Jev route-shape classification."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .types import ExecutionPrimitive, RouteCandidate

DEFAULT_OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_JEV_MODEL = "typesafe/jev-1.13"

ROUTE_CRITERIA: Mapping[ExecutionPrimitive, str] = {
    ExecutionPrimitive.RESULT_CACHE: (
        "Reuse a previously verified result when the same semantic task and all relevant dependencies are unchanged."
    ),
    ExecutionPrimitive.DETERMINISTIC_CODE: (
        "A bounded deterministic transformation, lookup, validation, or rule can produce the answer without semantic "
        "generation."
    ),
    ExecutionPrimitive.SPECIALIST_MODEL: (
        "A narrow, stable domain task needs semantic judgment or bounded generation suited to a specialist model."
    ),
    ExecutionPrimitive.SMALL_MODEL: (
        "A general language task needs generation or reasoning but is bounded enough for a smaller general model."
    ),
    ExecutionPrimitive.FRONTIER_MODEL: (
        "The task is open-ended, ambiguous, out-of-distribution, or requires complex multi-step reasoning."
    ),
}


class JevOpenRouterError(RuntimeError):
    """The Jev/OpenRouter decision could not be trusted or completed."""


@dataclass(frozen=True)
class JevUsage:
    input_tokens: int
    output_tokens: int
    cost_usd: float

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.cost_usd < 0:
            raise ValueError("cost_usd must be non-negative")


@dataclass(frozen=True)
class JevRouteAdvice:
    requested_model: str
    served_model: str
    provider: str
    decision_id: str
    primitive: ExecutionPrimitive
    probabilities: Mapping[ExecutionPrimitive, float]
    confidence: float
    usage: JevUsage
    latency_ms: float

    def __post_init__(self) -> None:
        if not self.requested_model or not self.served_model or not self.provider or not self.decision_id:
            raise ValueError("Jev route advice identity fields must be non-empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        if self.primitive not in self.probabilities:
            raise ValueError("winning primitive must be present in probabilities")
        if set(self.probabilities) != set(ExecutionPrimitive):
            raise ValueError("probabilities must cover every execution primitive")
        total = 0.0
        for probability in self.probabilities.values():
            if not 0 <= probability <= 1:
                raise ValueError("Jev probabilities must be between 0 and 1")
            total += probability
        if abs(total - 1.0) > 1e-3:
            raise ValueError("Jev probabilities must sum to 1")


@dataclass
class JevRouteBenchmarkStats:
    """Aggregate route-shape accuracy without retaining request state."""

    observations: int = 0
    matches: int = 0
    low_confidence: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    total_confidence: float = 0.0
    min_confidence: float = 0.8

    def __post_init__(self) -> None:
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")

    def observe(self, advice: JevRouteAdvice, expected: ExecutionPrimitive) -> None:
        if not isinstance(advice, JevRouteAdvice):
            raise ValueError("advice must be JevRouteAdvice")
        if not isinstance(expected, ExecutionPrimitive):
            raise ValueError("expected must be ExecutionPrimitive")
        self.observations += 1
        self.matches += int(advice.primitive is expected)
        self.low_confidence += int(advice.confidence < self.min_confidence)
        self.total_cost_usd += advice.usage.cost_usd
        self.total_latency_ms += advice.latency_ms
        self.total_confidence += advice.confidence

    @property
    def accuracy(self) -> float:
        return self.matches / self.observations if self.observations else 0.0

    @property
    def low_confidence_rate(self) -> float:
        return self.low_confidence / self.observations if self.observations else 0.0

    @property
    def mean_latency_ms(self) -> float:
        return self.total_latency_ms / self.observations if self.observations else 0.0

    @property
    def mean_confidence(self) -> float:
        return self.total_confidence / self.observations if self.observations else 0.0


JevTransport = Callable[[str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]]


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JevOpenRouterError(f"{name} must be numeric")
    result = float(value)
    if result < 0:
        raise JevOpenRouterError(f"{name} must be non-negative")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise JevOpenRouterError(f"{name} must be a non-negative integer")
    return value


def _usage_field(usage: Mapping[str, Any], snake: str, camel: str, default: Any = None) -> Any:
    if snake in usage:
        return usage[snake]
    if camel in usage:
        return usage[camel]
    return default


def _default_transport(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
    request = Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            raw = response.read()
    except HTTPError as exc:
        raise JevOpenRouterError(f"OpenRouter Decisions API returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise JevOpenRouterError("OpenRouter Decisions API is unavailable") from exc
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise JevOpenRouterError("OpenRouter Decisions API returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise JevOpenRouterError("OpenRouter Decisions API response must be a JSON object")
    return decoded


class OpenRouterJevClient:
    """Minimal stdlib client for Jev route-shape decisions through OpenRouter."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_JEV_MODEL,
        endpoint: str = DEFAULT_OPENROUTER_DECISIONS_URL,
        timeout_seconds: float = 15.0,
        transport: JevTransport | None = None,
        app_title: str = "PCF JIT",
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be non-empty")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be non-empty")
        if not isinstance(endpoint, str) or not endpoint.startswith("https://openrouter.ai/"):
            raise ValueError("endpoint must be an https://openrouter.ai/ URL")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if transport is not None and not callable(transport):
            raise ValueError("transport must be callable")
        self._api_key = api_key.strip()
        self.model = model.strip()
        self.endpoint = endpoint
        self.timeout_seconds = float(timeout_seconds)
        self._transport = transport or _default_transport
        self.app_title = app_title

    def __repr__(self) -> str:
        return f"OpenRouterJevClient(model={self.model!r}, endpoint={self.endpoint!r})"

    @classmethod
    def from_env(cls, **kwargs: Any) -> OpenRouterJevClient:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise JevOpenRouterError("OPENROUTER_API_KEY is not configured")
        return cls(key, **kwargs)

    def route(self, state: Any) -> JevRouteAdvice:
        """Ask Jev for task-shape advice, excluding safety, authority, and economics."""
        try:
            json.dumps(state, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("state must be finite JSON-serializable data") from exc
        criteria = {primitive.value: description for primitive, description in ROUTE_CRITERIA.items()}
        payload = {
            "model": self.model,
            "state": state,
            "questions": {
                "execution_primitive": {
                    "type": "choice",
                    "instructions": (
                        "Which execution primitive best matches the semantic work required? Judge task shape only. "
                        "Do not consider authorization, safety, price, or current route availability."
                    ),
                    "criteria": criteria,
                }
            },
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Title": self.app_title,
        }
        started = time.perf_counter()
        response = self._transport(self.endpoint, headers, payload, self.timeout_seconds)
        return self._parse_route_response(response, (time.perf_counter() - started) * 1000)

    def _parse_route_response(self, response: Mapping[str, Any], latency_ms: float) -> JevRouteAdvice:
        if not isinstance(response, Mapping):
            raise JevOpenRouterError("OpenRouter Decisions response must be an object")
        answers = response.get("answers")
        answer = answers.get("execution_primitive") if isinstance(answers, Mapping) else None
        if not isinstance(answer, Mapping) or answer.get("type") != "choice":
            raise JevOpenRouterError("Jev execution_primitive answer must be a choice")
        choice = answer.get("choice")
        try:
            primitive = ExecutionPrimitive(choice)
        except (TypeError, ValueError) as exc:
            raise JevOpenRouterError(f"Jev returned unknown execution primitive: {choice!r}") from exc
        raw_probabilities = answer.get("probabilities")
        if not isinstance(raw_probabilities, Mapping):
            raise JevOpenRouterError("Jev choice answer is missing probabilities")
        expected_keys = {item.value for item in ExecutionPrimitive}
        if set(raw_probabilities) != expected_keys:
            raise JevOpenRouterError("Jev probabilities do not cover the expected execution primitives")
        probabilities: dict[ExecutionPrimitive, float] = {}
        for key, value in raw_probabilities.items():
            probability = _number(value, f"probability[{key}]")
            if probability > 1:
                raise JevOpenRouterError("Jev probabilities must be between 0 and 1")
            probabilities[ExecutionPrimitive(key)] = probability
        if abs(sum(probabilities.values()) - 1.0) > 1e-3:
            raise JevOpenRouterError("Jev probabilities must sum to 1")
        confidence = _number(answer.get("confidence"), "confidence")
        if confidence > 1:
            raise JevOpenRouterError("confidence must be between 0 and 1")

        usage = response.get("usage")
        if not isinstance(usage, Mapping):
            raise JevOpenRouterError("OpenRouter Decisions response is missing usage")
        normalized_usage = JevUsage(
            input_tokens=_integer(_usage_field(usage, "input_tokens", "inputTokens", 0), "input_tokens"),
            output_tokens=_integer(_usage_field(usage, "output_tokens", "outputTokens", 0), "output_tokens"),
            cost_usd=_number(usage.get("cost", 0.0), "usage.cost"),
        )
        served_model, provider, decision_id = response.get("model"), response.get("provider"), response.get("id")
        if not all(isinstance(item, str) and item for item in (served_model, provider, decision_id)):
            raise JevOpenRouterError("OpenRouter Decisions response is missing model/provider/id")
        return JevRouteAdvice(
            requested_model=self.model,
            served_model=served_model,
            provider=provider,
            decision_id=decision_id,
            primitive=primitive,
            probabilities=probabilities,
            confidence=confidence,
            usage=normalized_usage,
            latency_ms=latency_ms,
        )


_JEV_METADATA_KEYS = {
    "jev_choice",
    "jev_confidence",
    "jev_probability",
    "jev_requested_model",
    "jev_served_model",
    "jev_provider",
    "jev_decision_id",
    "jev_cost_usd",
    "jev_latency_ms",
}


def annotate_candidates_with_jev(
    candidates: Sequence[RouteCandidate], advice: JevRouteAdvice
) -> tuple[RouteCandidate, ...]:
    """Attach Jev evidence without changing route eligibility or economics."""
    if not isinstance(advice, JevRouteAdvice):
        raise ValueError("advice must be JevRouteAdvice")
    annotated: list[RouteCandidate] = []
    for candidate in candidates:
        if not isinstance(candidate, RouteCandidate):
            raise ValueError("candidates must contain RouteCandidate objects")
        metadata = dict(candidate.metadata)
        overlap = set(metadata).intersection(_JEV_METADATA_KEYS)
        if overlap:
            raise ValueError(f"candidate metadata collides with Jev keys: {sorted(overlap)}")
        metadata.update(
            {
                "jev_choice": advice.primitive.value,
                "jev_confidence": advice.confidence,
                "jev_probability": advice.probabilities[candidate.primitive],
                "jev_requested_model": advice.requested_model,
                "jev_served_model": advice.served_model,
                "jev_provider": advice.provider,
                "jev_decision_id": advice.decision_id,
                "jev_cost_usd": advice.usage.cost_usd,
                "jev_latency_ms": advice.latency_ms,
            }
        )
        annotated.append(replace(candidate, metadata=metadata))
    return tuple(annotated)
