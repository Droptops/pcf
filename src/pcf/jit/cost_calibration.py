"""Conservative calibration of JIT input/cache cost estimates."""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil

from .pcf_reconcile import PCFInputCostReconciliation
from .types import RouteCandidate


@dataclass(frozen=True)
class CostCalibrationPolicy:
    """Policy for converting reconciliation history into an upper cost bound.

    Calibration is intentionally one-way: it may add uncertainty cost, but never
    reduce the current preflight estimate. Before enough evidence exists, the
    configured uncalibrated margin is used instead of pretending the estimate is
    precise.
    """

    min_observations: int = 50
    quantile: float = 0.99
    uncalibrated_relative_margin: float = 0.25
    calibrated_relative_floor: float = 0.05
    absolute_margin_floor_usd: float = 0.0

    def __post_init__(self) -> None:
        if type(self.min_observations) is not int or self.min_observations <= 0:
            raise ValueError("min_observations must be a positive integer")
        if not 0 < self.quantile <= 1:
            raise ValueError("quantile must be in (0, 1]")
        for name in (
            "uncalibrated_relative_margin",
            "calibrated_relative_floor",
            "absolute_margin_floor_usd",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class CostCalibrationSnapshot:
    model_id: str
    observations: int
    calibrated: bool
    usable: bool
    quantile: float
    overrun_ratio_quantile: float
    max_overrun_ratio: float
    zero_estimate_overruns: int
    relative_margin: float
    absolute_margin_floor_usd: float
    reasons: tuple[str, ...]

    def upper_bound_usd(self, estimated_total_input_cost_usd: float) -> float:
        if estimated_total_input_cost_usd < 0:
            raise ValueError("estimated_total_input_cost_usd must be non-negative")
        if not self.usable:
            raise ValueError("cost calibration is unusable because positive actual cost followed a zero estimate")
        relative = estimated_total_input_cost_usd * (1.0 + self.relative_margin)
        absolute = estimated_total_input_cost_usd + self.absolute_margin_floor_usd
        return max(estimated_total_input_cost_usd, relative, absolute)


class PCFCostCalibrator:
    """Collect normalized provider reconciliation and produce safe route-cost bounds.

    Only cost-overrun evidence matters for routing conservatism. Historical cases
    where PCF overestimated provider billing never decrease the future bound.
    """

    def __init__(self, policy: CostCalibrationPolicy | None = None) -> None:
        self.policy = policy or CostCalibrationPolicy()
        self._overrun_ratios: dict[str, list[float]] = {}
        self._observations: dict[str, int] = {}
        self._zero_estimate_overruns: dict[str, int] = {}

    def observe(self, reconciliation: PCFInputCostReconciliation) -> None:
        if not isinstance(reconciliation, PCFInputCostReconciliation):
            raise ValueError("reconciliation must be PCFInputCostReconciliation")
        model_id = reconciliation.model_id
        self._observations[model_id] = self._observations.get(model_id, 0) + 1
        estimated = reconciliation.estimated_total_input_cost_usd
        actual = reconciliation.actual_total_input_cost_usd
        if estimated == 0:
            if actual > 0:
                self._zero_estimate_overruns[model_id] = self._zero_estimate_overruns.get(model_id, 0) + 1
            return
        overrun_ratio = max(actual / estimated - 1.0, 0.0)
        self._overrun_ratios.setdefault(model_id, []).append(overrun_ratio)

    @staticmethod
    def _nearest_rank(values: list[float], quantile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        rank = max(1, ceil(quantile * len(ordered)))
        return ordered[rank - 1]

    def snapshot(self, model_id: str) -> CostCalibrationSnapshot:
        if not model_id:
            raise ValueError("model_id must be non-empty")
        p = self.policy
        observations = self._observations.get(model_id, 0)
        ratios = self._overrun_ratios.get(model_id, [])
        zero_overruns = self._zero_estimate_overruns.get(model_id, 0)
        calibrated = observations >= p.min_observations
        usable = zero_overruns == 0
        q = self._nearest_rank(ratios, p.quantile)
        maximum = max(ratios, default=0.0)
        reasons: list[str] = []
        if not calibrated:
            reasons.append("insufficient_observations")
        if zero_overruns:
            reasons.append("positive_actual_cost_after_zero_estimate")
        if calibrated:
            margin = max(p.calibrated_relative_floor, q)
        else:
            margin = p.uncalibrated_relative_margin
        return CostCalibrationSnapshot(
            model_id=model_id,
            observations=observations,
            calibrated=calibrated,
            usable=usable,
            quantile=p.quantile,
            overrun_ratio_quantile=q,
            max_overrun_ratio=maximum,
            zero_estimate_overruns=zero_overruns,
            relative_margin=margin,
            absolute_margin_floor_usd=p.absolute_margin_floor_usd,
            reasons=tuple(reasons),
        )

    def apply(self, candidate: RouteCandidate, *, model_id: str) -> RouteCandidate:
        """Add conservative cost uncertainty to a JIT route candidate.

        The existing token + memory estimate remains untouched. The difference to
        the safe upper bound is carried in ``uncertainty_cost_usd`` so accounting
        stays auditable instead of disguising calibration margin as token/cache cost.
        """

        if not isinstance(candidate, RouteCandidate):
            raise ValueError("candidate must be RouteCandidate")
        snapshot = self.snapshot(model_id)
        estimated = candidate.token_cost_usd + candidate.memory_cost_usd
        upper = snapshot.upper_bound_usd(estimated)
        margin = upper - estimated
        metadata = dict(candidate.metadata)
        reserved = {
            "pcf_cost_calibration_model_id",
            "pcf_cost_calibration_observations",
            "pcf_cost_calibration_calibrated",
            "pcf_cost_calibration_relative_margin",
            "pcf_cost_calibration_upper_bound_usd",
        }
        overlap = reserved.intersection(metadata)
        if overlap:
            raise ValueError(f"candidate metadata collides with calibration keys: {sorted(overlap)}")
        metadata.update(
            {
                "pcf_cost_calibration_model_id": model_id,
                "pcf_cost_calibration_observations": snapshot.observations,
                "pcf_cost_calibration_calibrated": snapshot.calibrated,
                "pcf_cost_calibration_relative_margin": snapshot.relative_margin,
                "pcf_cost_calibration_upper_bound_usd": upper,
            }
        )
        return replace(candidate, uncertainty_cost_usd=margin, metadata=metadata)
