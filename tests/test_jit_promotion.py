import pytest

from pcf.jit import (
    ExecutionPrimitive,
    OptimizerPolicy,
    PromotionEvaluator,
    PromotionPolicy,
    ShadowAssessment,
    ShadowExecutor,
    ShadowRequest,
    ShadowRun,
    ShadowTarget,
    TraceReplayer,
)


def _summary(count=200, *, candidate_quality=0.99, candidate_harm=0.001, candidate_cost=0.01, fail_every=None):
    def baseline_runner(request):
        return ShadowRun(output="ok", token_cost_usd=0.05, latency_ms=20)

    def candidate_runner(request):
        if fail_every is not None and int(request.request_id.split("-")[-1]) % fail_every == 0:
            raise RuntimeError("candidate failed")
        return ShadowRun(output="ok", token_cost_usd=candidate_cost, latency_ms=5)

    def assessor(request, target, run):
        if target.name == "candidate":
            return ShadowAssessment(
                expected_quality=candidate_quality,
                harm_probability=candidate_harm,
                harm_severity=1,
            )
        return ShadowAssessment(expected_quality=1.0, harm_probability=0.002, harm_severity=1)

    executor = ShadowExecutor(
        assessor,
        optimizer=None,
    )
    targets = (
        ShadowTarget("candidate", ExecutionPrimitive.SPECIALIST_MODEL, candidate_runner, shadow_safe=True),
        ShadowTarget("baseline", ExecutionPrimitive.FRONTIER_MODEL, baseline_runner, shadow_safe=True),
    )
    requests = [ShadowRequest(f"req-{i}", "task", {"i": i}) for i in range(1, count + 1)]
    return TraceReplayer(executor).replay(requests, targets)


def _evaluator(**kwargs):
    policy = PromotionPolicy(
        min_matched_samples=kwargs.pop("min_matched_samples", 100),
        min_success_rate=kwargs.pop("min_success_rate", 0.95),
        min_admissible_rate=kwargs.pop("min_admissible_rate", 0.95),
        max_quality_regression=kwargs.pop("max_quality_regression", 0.02),
        max_mean_htokens=kwargs.pop("max_mean_htokens", 0.01),
        max_htoken_regression=kwargs.pop("max_htoken_regression", 0.0),
        promotion_fixed_cost_usd=kwargs.pop("promotion_fixed_cost_usd", 1.0),
        min_projected_net_savings_usd=kwargs.pop("min_projected_net_savings_usd", 0.0),
        **kwargs,
    )
    return PromotionEvaluator(
        policy,
        optimizer_policy=OptimizerPolicy(min_quality=0.95, max_htokens=0.05, usd_per_htoken=1),
    )


def test_promotion_passes_only_after_paired_statistical_and_economic_gates():
    decision = _evaluator().evaluate(
        _summary(),
        candidate="candidate",
        baseline="baseline",
        projected_future_calls=1_000,
    )
    assert decision.promote
    assert decision.reasons == ()
    assert decision.evidence.matched_samples == 200
    assert decision.evidence.quality_delta_lower_bound >= -0.02
    assert decision.evidence.htoken_delta_upper_bound <= 0
    assert decision.evidence.projected_net_savings_usd > 0


def test_insufficient_matched_samples_blocks_promotion():
    decision = _evaluator(min_matched_samples=100).evaluate(
        _summary(count=20),
        candidate="candidate",
        baseline="baseline",
        projected_future_calls=1_000,
    )
    assert not decision.promote
    assert "insufficient_matched_samples" in decision.reasons


def test_quality_regression_blocks_promotion():
    decision = _evaluator(max_quality_regression=0.01).evaluate(
        _summary(candidate_quality=0.95),
        candidate="candidate",
        baseline="baseline",
        projected_future_calls=1_000,
    )
    assert not decision.promote
    assert "quality_regression_exceeds_gate" in decision.reasons


def test_htoken_regression_blocks_promotion_even_when_candidate_is_cheaper():
    decision = _evaluator(max_htoken_regression=0.0, max_mean_htokens=0.05).evaluate(
        _summary(candidate_harm=0.01, candidate_cost=0.001),
        candidate="candidate",
        baseline="baseline",
        projected_future_calls=1_000,
    )
    assert not decision.promote
    assert "htoken_regression_exceeds_gate" in decision.reasons


def test_reliability_uses_lower_confidence_bound_not_point_estimate():
    decision = _evaluator(min_matched_samples=50, min_success_rate=0.95).evaluate(
        _summary(count=100, fail_every=20),
        candidate="candidate",
        baseline="baseline",
        projected_future_calls=1_000,
    )
    assert decision.evidence.success_rate == pytest.approx(0.95)
    assert decision.evidence.success_rate_lower_bound < 0.95
    assert not decision.promote
    assert "success_rate_below_gate" in decision.reasons


def test_projection_is_required_by_default():
    decision = _evaluator().evaluate(
        _summary(),
        candidate="candidate",
        baseline="baseline",
        projected_future_calls=None,
    )
    assert not decision.promote
    assert "projection_required" in decision.reasons


def test_projected_savings_must_clear_fixed_promotion_cost():
    decision = _evaluator(promotion_fixed_cost_usd=100).evaluate(
        _summary(),
        candidate="candidate",
        baseline="baseline",
        projected_future_calls=100,
    )
    assert not decision.promote
    assert "projected_net_savings_below_gate" in decision.reasons


def test_missing_route_and_same_route_are_rejected():
    summary = _summary()
    evaluator = _evaluator()
    with pytest.raises(ValueError, match="must differ"):
        evaluator.evaluate(summary, candidate="candidate", baseline="candidate", projected_future_calls=100)
    with pytest.raises(ValueError, match="route not found"):
        evaluator.evaluate(summary, candidate="missing", baseline="baseline", projected_future_calls=100)
