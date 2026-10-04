import pytest

from pcf import Context, Segment
from pcf.compiler import Usage
from pcf.families.sim import family_a, family_b
from pcf.jit import (
    PCFReconciliationStats,
    ShadowRequest,
    ShadowRun,
    apply_pcf_reconciled_input_cost,
    estimate_pcf_input_cost,
    make_pcf_reconciled_shadow_runner,
    reconcile_pcf_input_cost,
)
from pcf.router import Candidate


def context():
    return Context([
        Segment("s", "system", "stable system prompt " * 20),
        Segment("m", "memory", "stable account facts " * 10),
        Segment("u", "user", "what is the answer?", stable=False),
    ])


def candidate(engine=None, **kwargs):
    engine = engine or family_a(min_cacheable=1)
    return Candidate(
        engine.compiler,
        engine.cache,
        input_price_per_mtok=kwargs.pop("input_price_per_mtok", 2.0),
        cache_read_price_per_mtok=kwargs.pop("cache_read_price_per_mtok", 0.2),
        **kwargs,
    )


def test_reconciliation_prices_provider_usage_with_candidate_rates():
    cand = candidate(cache_write_price_per_mtok=7.0)
    estimate = estimate_pcf_input_cost(context(), cand, now=0)
    usage = Usage(cache_read_input_tokens=100, cache_creation_input_tokens=50, input_tokens=25)

    result = reconcile_pcf_input_cost(estimate, usage, cand)
    assert result.actual_token_cost_usd == pytest.approx(25 * 2.0 / 1e6)
    assert result.actual_memory_cost_usd == pytest.approx((50 * 7.0 + 100 * 0.2) / 1e6)
    assert result.actual_total_input_cost_usd == pytest.approx(
        result.actual_token_cost_usd + result.actual_memory_cost_usd
    )
    assert result.actual_total_input_tokens == 175
    assert result.estimate_error_usd == pytest.approx(
        estimate.total_input_cost_usd - result.actual_total_input_cost_usd
    )
    assert result.absolute_error_usd == pytest.approx(abs(result.estimate_error_usd))


def test_reconciliation_preserves_each_token_bucket_error():
    cand = candidate()
    estimate = estimate_pcf_input_cost(context(), cand, now=0)
    usage = Usage(
        cache_read_input_tokens=estimate.warm_tokens + 3,
        cache_creation_input_tokens=estimate.cache_creation_tokens + 5,
        input_tokens=estimate.uncached_tokens + 7,
    )

    result = reconcile_pcf_input_cost(estimate, usage, cand)
    assert result.cache_read_error_tokens == -3
    assert result.cache_creation_error_tokens == -5
    assert result.uncached_error_tokens == -7
    assert result.total_input_error_tokens == -15


def test_reconciled_run_is_metered_with_actual_not_estimated_cost():
    cand = candidate()
    estimate = estimate_pcf_input_cost(context(), cand, now=0)
    usage = Usage(cache_read_input_tokens=10, cache_creation_input_tokens=20, input_tokens=30)
    reconciliation = reconcile_pcf_input_cost(estimate, usage, cand)
    run = ShadowRun(output={"answer": 42}, latency_ms=9, metadata={"trace": "abc"})

    metered = apply_pcf_reconciled_input_cost(run, estimate, reconciliation)
    assert metered.token_cost_usd == reconciliation.actual_token_cost_usd
    assert metered.memory_cost_usd == reconciliation.actual_memory_cost_usd
    assert metered.metadata["trace"] == "abc"
    assert metered.metadata["pcf_usage_reconciled"] is True
    assert metered.metadata["pcf_warm_tokens"] == estimate.warm_tokens
    assert metered.metadata["pcf_actual_cache_read_tokens"] == usage.cache_read_input_tokens
    assert metered.metadata["pcf_estimated_total_input_cost_usd"] == estimate.total_input_cost_usd
    assert metered.metadata["pcf_actual_total_input_cost_usd"] == reconciliation.actual_total_input_cost_usd


def test_reconciled_runner_snapshots_estimate_before_execution():
    cand = candidate()
    ctx = context()
    seen = []

    def execute(request):
        seen.append("execute")
        return ShadowRun(
            output="ok",
            metadata={"provider_usage": Usage(5, 7, 11)},
        )

    def context_of(request):
        seen.append("context")
        return request.payload

    def now_of(request):
        seen.append("now")
        return request.metadata["now"]

    def usage_of(request, run):
        seen.append("usage")
        return run.metadata["provider_usage"]

    runner = make_pcf_reconciled_shadow_runner(
        cand,
        execute,
        context_of=context_of,
        now_of=now_of,
        usage_of=usage_of,
    )
    result = runner(ShadowRequest("r1", "fp", ctx, {"now": 0.0}))

    assert seen == ["context", "now", "execute", "usage"]
    assert result.metadata["pcf_usage_reconciled"] is True
    assert result.metadata["pcf_actual_uncached_tokens"] == 11


def test_reconciled_runner_fails_closed_when_usage_is_missing_or_wrong_type():
    cand = candidate()
    runner = make_pcf_reconciled_shadow_runner(
        cand,
        lambda request: ShadowRun(output="ok"),
        context_of=lambda request: request.payload,
        now_of=lambda request: 0.0,
        usage_of=lambda request, run: None,
    )
    with pytest.raises(TypeError, match="Usage"):
        runner(ShadowRequest("r1", "fp", context()))


def test_reconciliation_rejects_candidate_model_mismatch():
    first = candidate(family_a(min_cacheable=1))
    second = candidate(family_b(min_cacheable=1))
    estimate = estimate_pcf_input_cost(context(), first, now=0)

    with pytest.raises(ValueError, match="model_id"):
        reconcile_pcf_input_cost(estimate, Usage(0, 0, 1), second)


def test_reconciled_apply_rejects_double_metering_and_reserved_metadata_collision():
    cand = candidate()
    estimate = estimate_pcf_input_cost(context(), cand, now=0)
    reconciliation = reconcile_pcf_input_cost(estimate, Usage(0, 0, 1), cand)

    with pytest.raises(ValueError, match="already contains"):
        apply_pcf_reconciled_input_cost(
            ShadowRun(output="x", token_cost_usd=0.1),
            estimate,
            reconciliation,
        )
    with pytest.raises(ValueError, match="collides"):
        apply_pcf_reconciled_input_cost(
            ShadowRun(output="x", metadata={"pcf_actual_uncached_tokens": 3}),
            estimate,
            reconciliation,
        )


def test_reconciliation_stats_aggregate_bias_without_payloads():
    cand = candidate()
    estimate = estimate_pcf_input_cost(context(), cand, now=0)
    first = reconcile_pcf_input_cost(estimate, Usage(0, 0, 10), cand)
    second = reconcile_pcf_input_cost(estimate, Usage(5, 5, 10), cand)
    stats = PCFReconciliationStats()
    stats.observe(first)
    stats.observe(second)

    assert stats.observations == 2
    assert stats.mean_estimate_error_usd == pytest.approx(
        (first.estimate_error_usd + second.estimate_error_usd) / 2
    )
    assert stats.mean_absolute_error_usd == pytest.approx(
        (first.absolute_error_usd + second.absolute_error_usd) / 2
    )
    assert stats.actual_total_cost_usd == pytest.approx(
        first.actual_total_input_cost_usd + second.actual_total_input_cost_usd
    )
    assert stats.cost_ratio_actual_to_estimated == pytest.approx(
        stats.actual_total_cost_usd / stats.estimated_total_cost_usd
    )


def test_zero_actual_cost_has_no_relative_error_denominator():
    cand = candidate()
    estimate = estimate_pcf_input_cost(context(), cand, now=0)
    result = reconcile_pcf_input_cost(estimate, Usage(0, 0, 0), cand)
    assert result.relative_error is None
