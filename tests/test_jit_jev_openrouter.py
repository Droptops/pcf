import math

import pytest

from pcf.jit import ExecutionPrimitive, RouteCandidate
from pcf.jit.jev_openrouter import (
    DEFAULT_JEV_MODEL,
    DEFAULT_OPENROUTER_DECISIONS_URL,
    JevOpenRouterError,
    JevRouteBenchmarkStats,
    OpenRouterJevClient,
    annotate_candidates_with_jev,
)


def response_fixture(**overrides):
    value = {
        "model": "typesafe/jev-1.13-20260917",
        "answers": {
            "execution_primitive": {
                "type": "choice",
                "choice": "deterministic_code",
                "probabilities": {
                    "result_cache": 0.01,
                    "deterministic_code": 0.82,
                    "specialist_model": 0.10,
                    "small_model": 0.05,
                    "frontier_model": 0.02,
                },
                "confidence": 0.91,
            }
        },
        "usage": {"input_tokens": 357, "output_tokens": 38, "cost": 0.000014994},
        "id": "gen-dec-test",
        "provider": "TypeSafe",
    }
    value.update(overrides)
    return value


def test_route_uses_openrouter_decisions_contract_and_pinned_model():
    seen = {}

    def transport(url, headers, payload, timeout):
        seen.update(url=url, headers=dict(headers), payload=payload, timeout=timeout)
        return response_fixture()

    client = OpenRouterJevClient("sk-or-secret", transport=transport)
    advice = client.route({"operation": "normalize_phone", "input": "+1 (919) 555-0100"})

    assert seen["url"] == DEFAULT_OPENROUTER_DECISIONS_URL
    assert seen["headers"]["Authorization"] == "Bearer sk-or-secret"
    assert seen["headers"]["Content-Type"] == "application/json"
    assert seen["payload"]["model"] == DEFAULT_JEV_MODEL
    question = seen["payload"]["questions"]["execution_primitive"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {item.value for item in ExecutionPrimitive}
    assert "Do not consider authorization" in question["instructions"]
    assert advice.primitive is ExecutionPrimitive.DETERMINISTIC_CODE
    assert advice.probabilities[ExecutionPrimitive.DETERMINISTIC_CODE] == pytest.approx(0.82)
    assert advice.confidence == pytest.approx(0.91)
    assert advice.usage.input_tokens == 357
    assert advice.usage.cost_usd == pytest.approx(0.000014994)
    assert advice.latency_ms >= 0


def test_client_repr_does_not_expose_api_key():
    client = OpenRouterJevClient("sk-or-super-secret", transport=lambda *_: response_fixture())
    assert "super-secret" not in repr(client)
    assert DEFAULT_JEV_MODEL in repr(client)


def test_usage_parser_accepts_openrouter_camel_case_variant():
    response = response_fixture(usage={"inputTokens": 381, "outputTokens": 62, "cost": 0.000016002})
    client = OpenRouterJevClient("key", transport=lambda *_: response)
    advice = client.route("classify this")
    assert advice.usage.input_tokens == 381
    assert advice.usage.output_tokens == 62


def test_unknown_choice_fails_closed():
    response = response_fixture()
    response["answers"]["execution_primitive"]["choice"] = "magic_router"
    client = OpenRouterJevClient("key", transport=lambda *_: response)
    with pytest.raises(JevOpenRouterError, match="unknown execution primitive"):
        client.route("task")


def test_incomplete_or_invalid_probability_distribution_fails_closed():
    response = response_fixture()
    response["answers"]["execution_primitive"]["probabilities"].pop("frontier_model")
    client = OpenRouterJevClient("key", transport=lambda *_: response)
    with pytest.raises(JevOpenRouterError, match="do not cover"):
        client.route("task")

    response = response_fixture()
    response["answers"]["execution_primitive"]["probabilities"]["deterministic_code"] = 0.5
    client = OpenRouterJevClient("key", transport=lambda *_: response)
    with pytest.raises(JevOpenRouterError, match="sum to 1"):
        client.route("task")


def test_non_json_or_non_finite_state_is_rejected_before_transport():
    calls = 0

    def transport(*_):
        nonlocal calls
        calls += 1
        return response_fixture()

    client = OpenRouterJevClient("key", transport=transport)
    with pytest.raises(ValueError, match="JSON-serializable"):
        client.route({"bad": {1, 2, 3}})
    with pytest.raises(ValueError, match="JSON-serializable"):
        client.route({"bad": math.nan})
    assert calls == 0


def test_annotation_is_advisory_and_does_not_change_route_gates_or_costs():
    client = OpenRouterJevClient("key", transport=lambda *_: response_fixture())
    advice = client.route("bounded deterministic normalization")
    original = RouteCandidate(
        name="cheap-denied-code",
        primitive=ExecutionPrimitive.DETERMINISTIC_CODE,
        token_cost_usd=0.001,
        memory_cost_usd=0.002,
        latency_ms=4,
        expected_quality=0.99,
        harm_probability=0.01,
        harm_severity=2,
        exposure=3,
        authority_ok=False,
        freshness_ok=True,
        enabled=True,
        uncertainty_cost_usd=0.003,
        metadata={"source": "fixture"},
    )
    annotated = annotate_candidates_with_jev([original], advice)[0]

    assert annotated.authority_ok is False
    assert annotated.freshness_ok is True
    assert annotated.enabled is True
    assert annotated.token_cost_usd == original.token_cost_usd
    assert annotated.memory_cost_usd == original.memory_cost_usd
    assert annotated.uncertainty_cost_usd == original.uncertainty_cost_usd
    assert annotated.expected_quality == original.expected_quality
    assert annotated.harm_probability == original.harm_probability
    assert annotated.metadata["source"] == "fixture"
    assert annotated.metadata["jev_choice"] == "deterministic_code"
    assert annotated.metadata["jev_probability"] == pytest.approx(0.82)


def test_reserved_jev_metadata_collision_is_rejected():
    client = OpenRouterJevClient("key", transport=lambda *_: response_fixture())
    advice = client.route("task")
    candidate = RouteCandidate(
        name="code",
        primitive=ExecutionPrimitive.DETERMINISTIC_CODE,
        metadata={"jev_choice": "forged"},
    )
    with pytest.raises(ValueError, match="collides"):
        annotate_candidates_with_jev([candidate], advice)


def test_benchmark_stats_track_accuracy_confidence_cost_and_latency():
    client = OpenRouterJevClient("key", transport=lambda *_: response_fixture())
    advice = client.route("task")
    stats = JevRouteBenchmarkStats(min_confidence=0.95)
    stats.observe(advice, ExecutionPrimitive.DETERMINISTIC_CODE)
    stats.observe(advice, ExecutionPrimitive.SMALL_MODEL)

    assert stats.observations == 2
    assert stats.matches == 1
    assert stats.accuracy == pytest.approx(0.5)
    assert stats.low_confidence_rate == pytest.approx(1.0)
    assert stats.total_cost_usd == pytest.approx(advice.usage.cost_usd * 2)
    assert stats.mean_confidence == pytest.approx(0.91)
    assert stats.mean_latency_ms >= 0


def test_from_env_requires_openrouter_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(JevOpenRouterError, match="OPENROUTER_API_KEY"):
        OpenRouterJevClient.from_env()
