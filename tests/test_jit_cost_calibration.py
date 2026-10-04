import pytest

from pcf.jit import ExecutionPrimitive, JITOptimizer, RouteCandidate
from pcf.jit.cost_calibration import CostCalibrationPolicy, PCFCostCalibrator
from pcf.jit.pcf_reconcile import PCFInputCostReconciliation


def reconciliation(*, model="m", estimated=0.01, actual=0.01):
    return PCFInputCostReconciliation(
        model_id=model,
        estimated_cache_read_tokens=0,
        actual_cache_read_tokens=0,
        estimated_cache_creation_tokens=0,
        actual_cache_creation_tokens=0,
        estimated_uncached_tokens=100,
        actual_uncached_tokens=100,
        estimated_total_input_tokens=100,
        actual_total_input_tokens=100,
        estimated_token_cost_usd=estimated,
        actual_token_cost_usd=actual,
        estimated_memory_cost_usd=0.0,
        actual_memory_cost_usd=0.0,
        estimated_total_input_cost_usd=estimated,
        actual_total_input_cost_usd=actual,
        estimate_error_usd=estimated - actual,
        absolute_error_usd=abs(estimated - actual),
        relative_error=((estimated - actual) / actual if actual > 0 else None),
    )


def test_uncalibrated_route_gets_explicit_safety_margin():
    calibrator = PCFCostCalibrator(
        CostCalibrationPolicy(min_observations=3, uncalibrated_relative_margin=0.25)
    )
    route = RouteCandidate(
        name="small",
        primitive=ExecutionPrimitive.SMALL_MODEL,
        token_cost_usd=0.008,
        memory_cost_usd=0.002,
    )
    adjusted = calibrator.apply(route, model_id="m")
    assert adjusted.token_cost_usd == route.token_cost_usd
    assert adjusted.memory_cost_usd == route.memory_cost_usd
    assert adjusted.uncertainty_cost_usd == pytest.approx(0.0025)
    assert JITOptimizer().score(adjusted).total_cost_usd == pytest.approx(0.0125)
    assert adjusted.metadata["pcf_cost_calibration_calibrated"] is False


def test_historical_overestimates_never_discount_future_route_cost():
    calibrator = PCFCostCalibrator(
        CostCalibrationPolicy(
            min_observations=3,
            quantile=1.0,
            calibrated_relative_floor=0.05,
            uncalibrated_relative_margin=0.25,
        )
    )
    for _ in range(3):
        calibrator.observe(reconciliation(estimated=0.02, actual=0.01))
    snapshot = calibrator.snapshot("m")
    assert snapshot.calibrated
    assert snapshot.overrun_ratio_quantile == 0
    assert snapshot.relative_margin == pytest.approx(0.05)
    assert snapshot.upper_bound_usd(0.02) == pytest.approx(0.021)


def test_empirical_overrun_quantile_sets_conservative_bound():
    calibrator = PCFCostCalibrator(
        CostCalibrationPolicy(min_observations=3, quantile=1.0, calibrated_relative_floor=0.0)
    )
    calibrator.observe(reconciliation(estimated=0.01, actual=0.011))
    calibrator.observe(reconciliation(estimated=0.01, actual=0.015))
    calibrator.observe(reconciliation(estimated=0.01, actual=0.012))
    snapshot = calibrator.snapshot("m")
    assert snapshot.overrun_ratio_quantile == pytest.approx(0.5)
    assert snapshot.max_overrun_ratio == pytest.approx(0.5)
    assert snapshot.upper_bound_usd(0.02) == pytest.approx(0.03)


def test_positive_actual_cost_after_zero_estimate_fails_closed():
    calibrator = PCFCostCalibrator(CostCalibrationPolicy(min_observations=1))
    calibrator.observe(reconciliation(estimated=0.0, actual=0.01))
    snapshot = calibrator.snapshot("m")
    assert not snapshot.usable
    assert snapshot.zero_estimate_overruns == 1
    with pytest.raises(ValueError, match="unusable"):
        snapshot.upper_bound_usd(0.01)


def test_uncertainty_margin_can_change_least_cost_route_choice():
    calibrator = PCFCostCalibrator(
        CostCalibrationPolicy(min_observations=1, quantile=1.0, calibrated_relative_floor=0.0)
    )
    calibrator.observe(reconciliation(estimated=0.01, actual=0.03))
    cheap = RouteCandidate(
        name="apparently-cheap",
        primitive=ExecutionPrimitive.SMALL_MODEL,
        token_cost_usd=0.01,
    )
    safer = RouteCandidate(
        name="stable-cost",
        primitive=ExecutionPrimitive.FRONTIER_MODEL,
        token_cost_usd=0.02,
    )
    adjusted = calibrator.apply(cheap, model_id="m")
    decision = JITOptimizer().choose([adjusted, safer])
    assert adjusted.uncertainty_cost_usd == pytest.approx(0.02)
    assert decision.chosen == "stable-cost"


def test_calibration_metadata_collisions_fail_closed():
    calibrator = PCFCostCalibrator()
    route = RouteCandidate(
        name="small",
        primitive=ExecutionPrimitive.SMALL_MODEL,
        token_cost_usd=0.01,
        metadata={"pcf_cost_calibration_model_id": "spoof"},
    )
    with pytest.raises(ValueError, match="collides"):
        calibrator.apply(route, model_id="m")
