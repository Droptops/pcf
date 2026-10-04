from math import inf

import pytest

from pcf.jit import (
    CacheEnvelope,
    CompileEconomics,
    ExecutionPrimitive,
    HotPathDetector,
    JITOptimizer,
    NoAdmissibleRoute,
    OptimizerPolicy,
    RiskAwareResultCache,
    RouteCandidate,
)


def candidate(name, primitive, **kwargs):
    return RouteCandidate(name=name, primitive=primitive, **kwargs)


def test_cheapest_admissible_route_wins_after_hard_gates():
    router = JITOptimizer(OptimizerPolicy(min_quality=0.95, max_htokens=0.1, usd_per_htoken=10))
    decision = router.choose([
        candidate(
            "cheap-but-unauthorized",
            ExecutionPrimitive.DETERMINISTIC_CODE,
            token_cost_usd=0.0,
            authority_ok=False,
        ),
        candidate(
            "small",
            ExecutionPrimitive.SMALL_MODEL,
            token_cost_usd=0.002,
            expected_quality=0.97,
            harm_probability=0.001,
            harm_severity=1,
        ),
        candidate(
            "frontier",
            ExecutionPrimitive.FRONTIER_MODEL,
            token_cost_usd=0.05,
            expected_quality=0.999,
            harm_probability=0.0001,
            harm_severity=1,
        ),
    ])
    assert decision.chosen == "small"
    unauthorized = next(r for r in decision.reports if r.name == "cheap-but-unauthorized")
    assert unauthorized.reject_reasons == ("authority",)


def test_htokens_can_make_nominally_cheaper_model_more_expensive():
    router = JITOptimizer(OptimizerPolicy(min_quality=0.9, max_htokens=10, usd_per_htoken=100))
    decision = router.choose([
        candidate(
            "cheap-risky",
            ExecutionPrimitive.SMALL_MODEL,
            token_cost_usd=0.001,
            harm_probability=0.02,
            harm_severity=1,
        ),
        candidate(
            "expensive-safe",
            ExecutionPrimitive.FRONTIER_MODEL,
            token_cost_usd=0.05,
            harm_probability=0.0001,
            harm_severity=1,
        ),
    ])
    assert decision.chosen == "expensive-safe"


def test_quality_and_harm_are_hard_gates():
    router = JITOptimizer(OptimizerPolicy(min_quality=0.98, max_htokens=0.01))
    with pytest.raises(NoAdmissibleRoute):
        router.choose([
            candidate("low-quality", ExecutionPrimitive.SMALL_MODEL, expected_quality=0.9),
            candidate(
                "high-harm",
                ExecutionPrimitive.FRONTIER_MODEL,
                expected_quality=0.999,
                harm_probability=0.1,
                harm_severity=1,
            ),
        ])


def test_result_cache_requires_fresh_dependencies_policy_authority_and_harm_budget():
    cache = RiskAwareResultCache()
    cache.put(
        "territory:acme",
        CacheEnvelope(
            value="ENT-WEST",
            created_at=100,
            max_age_seconds=60,
            dependency_versions={"territory_map": "7"},
            policy_version="3",
            authority_scope="sales-read",
            expected_staleness_htokens=0.001,
        ),
    )
    kwargs = dict(
        dependency_versions={"territory_map": "7"},
        policy_version="3",
        authority_scope="sales-read",
        max_staleness_htokens=0.01,
    )
    assert cache.get("territory:acme", now=120, **kwargs) == "ENT-WEST"
    assert cache.get("territory:acme", now=161, **kwargs) is None
    assert cache.get("territory:acme", now=120, **{**kwargs, "policy_version": "4"}) is None
    assert cache.get(
        "territory:acme",
        now=120,
        **{**kwargs, "dependency_versions": {"territory_map": "8"}},
    ) is None
    assert cache.get("territory:acme", now=120, **{**kwargs, "max_staleness_htokens": 0.0001}) is None


def test_compile_economics_requires_projected_savings_to_clear_fixed_cost():
    economics = CompileEconomics(
        general_cost_per_call_usd=0.05,
        compiled_cost_per_call_usd=0.005,
        build_cost_usd=9,
    )
    assert economics.breakeven_calls == 200
    assert not economics.should_compile(199)
    assert economics.should_compile(200)


def test_compile_economics_never_compiles_when_unit_cost_does_not_improve():
    economics = CompileEconomics(0.01, 0.02, 1)
    assert economics.breakeven_calls == inf
    assert not economics.should_compile(1_000_000)


def test_hot_path_detector_needs_both_volume_and_stability():
    detector = HotPathDetector(min_calls=5, min_stability=0.8)
    for stable in [True, True, True, True]:
        detector.observe("lead-route", stable=stable)
    assert not detector.is_hot("lead-route")
    detector.observe("lead-route", stable=True)
    assert detector.is_hot("lead-route")


def test_deterministic_code_can_beat_models_when_admissible():
    router = JITOptimizer(OptimizerPolicy(min_quality=0.99, max_htokens=0.01, usd_per_htoken=100))
    decision = router.choose([
        candidate(
            "compiled-rule",
            ExecutionPrimitive.DETERMINISTIC_CODE,
            token_cost_usd=0,
            expected_quality=1,
            harm_probability=0,
        ),
        candidate(
            "frontier",
            ExecutionPrimitive.FRONTIER_MODEL,
            token_cost_usd=0.03,
            expected_quality=0.999,
            harm_probability=0.00001,
            harm_severity=1,
        ),
    ])
    assert decision.chosen == "compiled-rule"
