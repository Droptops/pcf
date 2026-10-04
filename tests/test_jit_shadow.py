import pytest

from pcf.jit import ExecutionPrimitive, JITOptimizer, OptimizerPolicy
from pcf.jit.shadow import (
    ShadowAssessment,
    ShadowExecutor,
    ShadowRequest,
    ShadowRun,
    ShadowTarget,
    TraceReplayer,
)


def assess(request, target, run):
    quality = 1.0 if run.output == request.metadata.get("expected") else 0.5
    risk = 0.02 if target.name == "risky" else 0.0
    return ShadowAssessment(quality, harm_probability=risk, harm_severity=1)


def test_unsafe_target_is_skipped():
    calls = []
    target = ShadowTarget(
        "unsafe",
        ExecutionPrimitive.DETERMINISTIC_CODE,
        lambda req: calls.append(1),
        shadow_safe=False,
    )
    result = ShadowExecutor(assess).execute(
        ShadowRequest("1", "fp", {}, {"expected": "x"}),
        [target],
    )
    assert calls == []
    assert result.observations[0].status == "skipped"
    assert result.decision is None


def test_request_is_deepcopied_per_target():
    seen = []

    def mutator(req):
        req.payload["x"].append(2)
        return ShadowRun("ok")

    def reader(req):
        seen.append(list(req.payload["x"]))
        return ShadowRun("ok")

    targets = [
        ShadowTarget("a", ExecutionPrimitive.DETERMINISTIC_CODE, mutator, shadow_safe=True),
        ShadowTarget("b", ExecutionPrimitive.SMALL_MODEL, reader, shadow_safe=True),
    ]
    req = ShadowRequest("1", "fp", {"x": [1]}, {"expected": "ok"})
    ShadowExecutor(assess).execute(req, targets)
    assert seen == [[1]]
    assert req.payload == {"x": [1]}


def test_failures_are_isolated():
    def fail(_):
        raise RuntimeError("boom")

    targets = [
        ShadowTarget("bad", ExecutionPrimitive.SMALL_MODEL, fail, shadow_safe=True),
        ShadowTarget(
            "good",
            ExecutionPrimitive.FRONTIER_MODEL,
            lambda _: ShadowRun("ok", token_cost_usd=0.1),
            shadow_safe=True,
        ),
    ]
    result = ShadowExecutor(assess).execute(
        ShadowRequest("1", "fp", {}, {"expected": "ok"}),
        targets,
    )
    assert [o.status for o in result.observations] == ["error", "success"]
    assert result.decision.chosen == "good"


def test_optimizer_uses_observed_metrics_and_assessment():
    targets = [
        ShadowTarget(
            "cheap",
            ExecutionPrimitive.SMALL_MODEL,
            lambda _: ShadowRun("ok", token_cost_usd=0.01),
            shadow_safe=True,
        ),
        ShadowTarget(
            "frontier",
            ExecutionPrimitive.FRONTIER_MODEL,
            lambda _: ShadowRun("ok", token_cost_usd=0.1),
            shadow_safe=True,
        ),
    ]
    result = ShadowExecutor(assess, JITOptimizer(OptimizerPolicy(min_quality=0.9))).execute(
        ShadowRequest("1", "fp", {}, {"expected": "ok"}),
        targets,
    )
    assert result.decision.chosen == "cheap"


def test_harm_gate_can_reject_cheapest_shadow_route():
    targets = [
        ShadowTarget(
            "risky",
            ExecutionPrimitive.SMALL_MODEL,
            lambda _: ShadowRun("ok", token_cost_usd=0.001),
            shadow_safe=True,
        ),
        ShadowTarget(
            "safe",
            ExecutionPrimitive.FRONTIER_MODEL,
            lambda _: ShadowRun("ok", token_cost_usd=0.1),
            shadow_safe=True,
        ),
    ]
    executor = ShadowExecutor(
        assess,
        JITOptimizer(OptimizerPolicy(min_quality=0.9, max_htokens=0.01)),
    )
    result = executor.execute(
        ShadowRequest("1", "fp", {}, {"expected": "ok"}),
        targets,
    )
    assert result.decision.chosen == "safe"
    risky = next(r for r in result.decision.reports if r.name == "risky")
    assert risky.reject_reasons == ("harm",)


def test_trace_replay_aggregates_route_statistics():
    targets = [
        ShadowTarget(
            "small",
            ExecutionPrimitive.SMALL_MODEL,
            lambda _: ShadowRun("ok", token_cost_usd=0.01, latency_ms=3),
            shadow_safe=True,
        ),
        ShadowTarget(
            "frontier",
            ExecutionPrimitive.FRONTIER_MODEL,
            lambda _: ShadowRun("ok", token_cost_usd=0.1, latency_ms=8),
            shadow_safe=True,
        ),
    ]
    requests = [ShadowRequest(str(i), "fp", {}, {"expected": "ok"}) for i in range(3)]
    summary = TraceReplayer(ShadowExecutor(assess)).replay(requests, targets)
    small = summary.route_stats["small"]
    assert small.attempts == 3 and small.successes == 3 and small.chosen == 3
    assert small.token_cost_usd == pytest.approx(0.03)
    assert small.mean_latency_ms == pytest.approx(3)
    assert small.mean_quality == pytest.approx(1)


def test_no_admissible_route_is_recorded_not_raised():
    target = ShadowTarget(
        "bad",
        ExecutionPrimitive.SMALL_MODEL,
        lambda _: ShadowRun("wrong"),
        shadow_safe=True,
    )
    result = ShadowExecutor(assess, JITOptimizer(OptimizerPolicy(min_quality=0.9))).execute(
        ShadowRequest("1", "fp", {}, {"expected": "ok"}),
        [target],
    )
    assert result.decision is None
    assert result.decision_error is not None


def test_duplicate_target_names_are_rejected():
    def target(name):
        return ShadowTarget(
            name,
            ExecutionPrimitive.SMALL_MODEL,
            lambda _: ShadowRun("ok"),
            shadow_safe=True,
        )

    with pytest.raises(ValueError, match="unique"):
        ShadowExecutor(assess).execute(
            ShadowRequest("1", "fp", {}, {"expected": "ok"}),
            [target("x"), target("x")],
        )
