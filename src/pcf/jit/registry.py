"""Evidence-gated staged route registry for JIT execution candidates."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

from ..descriptor import hash_object
from ..validation import nonempty, number
from .fingerprint import FingerprintAuditSummary
from .promotion import PromotionDecision
from .types import ExecutionPrimitive


class RouteState(str, Enum):
    SHADOW = "shadow"
    ELIGIBLE = "eligible"
    CANARY = "canary"
    ACTIVE = "active"
    RETIRED = "retired"


class InvalidRouteTransition(RuntimeError):
    """The requested lifecycle transition violates registry policy."""


class StaleRouteGeneration(InvalidRouteTransition):
    """Optimistic-concurrency generation did not match the current route."""


@dataclass(frozen=True)
class RegisteredRoute:
    route_id: str
    name: str
    route_version: str
    candidate_key: str
    task_fingerprint: str
    primitive: ExecutionPrimitive
    state: RouteState
    generation: int
    traffic_fraction: float
    created_at: float
    updated_at: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteTransition:
    route_id: str
    from_state: RouteState
    to_state: RouteState
    generation: int
    at: float
    evidence_ref: str


class RouteRegistry:
    """Reference in-memory registry with explicit, evidence-gated lifecycle transitions.

    The registry changes routing metadata only. It does not send production traffic,
    deploy models, or execute tools.
    """

    def __init__(self) -> None:
        self._routes: dict[str, RegisteredRoute] = {}
        self._history: list[RouteTransition] = []

    def register(
        self,
        *,
        name: str,
        route_version: str,
        task_fingerprint: str,
        primitive: ExecutionPrimitive,
        now: float,
        candidate_key: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RegisteredRoute:
        nonempty(name, "name")
        nonempty(route_version, "route_version")
        nonempty(task_fingerprint, "task_fingerprint")
        number(now, "now")
        if not isinstance(primitive, ExecutionPrimitive):
            raise ValueError("primitive must be ExecutionPrimitive")
        key = candidate_key or name
        nonempty(key, "candidate_key")
        if any(
            route.task_fingerprint == task_fingerprint
            and route.candidate_key == key
            and route.state is not RouteState.RETIRED
            for route in self._routes.values()
        ):
            raise ValueError("candidate_key must be unique among non-retired routes for a task fingerprint")
        route_id = hash_object(
            "pcf:jit-route-record:0.1",
            {
                "name": name,
                "route_version": route_version,
                "candidate_key": key,
                "task_fingerprint": task_fingerprint,
                "primitive": primitive.value,
            },
        )
        if route_id in self._routes:
            raise ValueError("route is already registered")
        route = RegisteredRoute(
            route_id=route_id,
            name=name,
            route_version=route_version,
            candidate_key=key,
            task_fingerprint=task_fingerprint,
            primitive=primitive,
            state=RouteState.SHADOW,
            generation=0,
            traffic_fraction=0.0,
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
        )
        self._routes[route_id] = route
        return route

    def get(self, route_id: str) -> RegisteredRoute:
        nonempty(route_id, "route_id")
        try:
            return self._routes[route_id]
        except KeyError:
            raise KeyError(f"unknown route: {route_id}") from None

    def routes_for_fingerprint(self, task_fingerprint: str) -> tuple[RegisteredRoute, ...]:
        nonempty(task_fingerprint, "task_fingerprint")
        return tuple(
            sorted(
                (route for route in self._routes.values() if route.task_fingerprint == task_fingerprint),
                key=lambda route: (route.created_at, route.route_id),
            )
        )

    def active(self, task_fingerprint: str) -> RegisteredRoute | None:
        active = [
            route
            for route in self.routes_for_fingerprint(task_fingerprint)
            if route.state is RouteState.ACTIVE
        ]
        if len(active) > 1:
            raise RuntimeError("registry invariant violated: multiple active routes for one task fingerprint")
        return active[0] if active else None

    def mark_eligible(
        self,
        route_id: str,
        *,
        promotion: PromotionDecision,
        fingerprint_audit: FingerprintAuditSummary,
        evidence_ref: str,
        now: float,
        expected_generation: int,
    ) -> RegisteredRoute:
        route = self._prepare(route_id, RouteState.SHADOW, now, expected_generation)
        self._require_promotion(route, promotion)
        self._require_clean_fingerprint(route, fingerprint_audit)
        return self._transition(route, RouteState.ELIGIBLE, now=now, traffic_fraction=0.0, evidence_ref=evidence_ref)

    def start_canary(
        self,
        route_id: str,
        *,
        traffic_fraction: float,
        evidence_ref: str,
        now: float,
        expected_generation: int,
    ) -> RegisteredRoute:
        route = self._prepare(route_id, RouteState.ELIGIBLE, now, expected_generation)
        if not 0 < traffic_fraction < 1:
            raise InvalidRouteTransition("canary traffic_fraction must be between 0 and 1")
        return self._transition(
            route,
            RouteState.CANARY,
            now=now,
            traffic_fraction=traffic_fraction,
            evidence_ref=evidence_ref,
        )

    def activate(
        self,
        route_id: str,
        *,
        promotion: PromotionDecision,
        fingerprint_audit: FingerprintAuditSummary,
        evidence_ref: str,
        now: float,
        expected_generation: int,
        replace_active_route_id: str | None = None,
        expected_active_generation: int | None = None,
    ) -> RegisteredRoute:
        route = self._prepare(route_id, RouteState.CANARY, now, expected_generation)
        self._require_promotion(route, promotion)
        self._require_clean_fingerprint(route, fingerprint_audit)

        current_active = self.active(route.task_fingerprint)
        replaced = None
        if current_active is not None and current_active.route_id != route.route_id:
            if replace_active_route_id != current_active.route_id:
                raise InvalidRouteTransition("activation would create multiple active routes; explicit replacement required")
            if expected_active_generation is None:
                raise InvalidRouteTransition("expected_active_generation is required when replacing an active route")
            replaced = self._prepare(
                current_active.route_id,
                RouteState.ACTIVE,
                now,
                expected_active_generation,
            )
        elif replace_active_route_id is not None:
            raise InvalidRouteTransition("replace_active_route_id supplied but no different active route exists")

        if replaced is not None:
            self._transition(
                replaced,
                RouteState.RETIRED,
                now=now,
                traffic_fraction=0.0,
                evidence_ref=f"{evidence_ref}:replaced",
            )
        return self._transition(route, RouteState.ACTIVE, now=now, traffic_fraction=1.0, evidence_ref=evidence_ref)

    def rollback_canary(
        self,
        route_id: str,
        *,
        evidence_ref: str,
        now: float,
        expected_generation: int,
    ) -> RegisteredRoute:
        route = self._prepare(route_id, RouteState.CANARY, now, expected_generation)
        return self._transition(route, RouteState.ELIGIBLE, now=now, traffic_fraction=0.0, evidence_ref=evidence_ref)

    def demote_active(
        self,
        route_id: str,
        *,
        traffic_fraction: float,
        evidence_ref: str,
        now: float,
        expected_generation: int,
    ) -> RegisteredRoute:
        route = self._prepare(route_id, RouteState.ACTIVE, now, expected_generation)
        if not 0 < traffic_fraction < 1:
            raise InvalidRouteTransition("canary traffic_fraction must be between 0 and 1")
        return self._transition(
            route,
            RouteState.CANARY,
            now=now,
            traffic_fraction=traffic_fraction,
            evidence_ref=evidence_ref,
        )

    def retire(
        self,
        route_id: str,
        *,
        evidence_ref: str,
        now: float,
        expected_generation: int,
    ) -> RegisteredRoute:
        route = self.get(route_id)
        self._require_generation(route, expected_generation)
        self._require_time(route, now)
        if route.state is RouteState.RETIRED:
            raise InvalidRouteTransition("retired routes are terminal")
        return self._transition(route, RouteState.RETIRED, now=now, traffic_fraction=0.0, evidence_ref=evidence_ref)

    def history(self, route_id: str | None = None) -> tuple[RouteTransition, ...]:
        if route_id is None:
            return tuple(self._history)
        nonempty(route_id, "route_id")
        return tuple(item for item in self._history if item.route_id == route_id)

    def _prepare(
        self,
        route_id: str,
        required_state: RouteState,
        now: float,
        expected_generation: int,
    ) -> RegisteredRoute:
        route = self.get(route_id)
        self._require_generation(route, expected_generation)
        self._require_time(route, now)
        if route.state is not required_state:
            raise InvalidRouteTransition(f"route must be {required_state.value}, got {route.state.value}")
        return route

    @staticmethod
    def _require_generation(route: RegisteredRoute, expected_generation: int) -> None:
        if type(expected_generation) is not int or expected_generation < 0:
            raise ValueError("expected_generation must be a non-negative integer")
        if route.generation != expected_generation:
            raise StaleRouteGeneration(
                f"stale route generation: expected {expected_generation}, current {route.generation}"
            )

    @staticmethod
    def _require_time(route: RegisteredRoute, now: float) -> None:
        number(now, "now")
        if now < route.updated_at:
            raise InvalidRouteTransition("transition time cannot move backwards")

    @staticmethod
    def _require_promotion(route: RegisteredRoute, promotion: PromotionDecision) -> None:
        if not isinstance(promotion, PromotionDecision):
            raise InvalidRouteTransition("promotion evidence must be PromotionDecision")
        if not promotion.promote:
            raise InvalidRouteTransition("promotion evidence did not pass")
        if promotion.evidence.candidate != route.candidate_key:
            raise InvalidRouteTransition("promotion candidate does not match registered candidate_key")

    @staticmethod
    def _require_clean_fingerprint(route: RegisteredRoute, summary: FingerprintAuditSummary) -> None:
        if not isinstance(summary, FingerprintAuditSummary):
            raise InvalidRouteTransition("fingerprint audit must be FingerprintAuditSummary")
        if route.task_fingerprint not in summary.observed_fingerprints:
            raise InvalidRouteTransition("fingerprint audit has no observations for this task fingerprint")
        if route.task_fingerprint in summary.collision_examples:
            raise InvalidRouteTransition("task fingerprint has an adjudicated semantic collision")
        prefix = f"{route.task_fingerprint}|"
        if any(key.startswith(prefix) for key in summary.drift_examples):
            raise InvalidRouteTransition("task fingerprint has adjudicated semantic drift")

    def _transition(
        self,
        route: RegisteredRoute,
        state: RouteState,
        *,
        now: float,
        traffic_fraction: float,
        evidence_ref: str,
    ) -> RegisteredRoute:
        nonempty(evidence_ref, "evidence_ref")
        updated = replace(
            route,
            state=state,
            generation=route.generation + 1,
            traffic_fraction=traffic_fraction,
            updated_at=now,
        )
        self._routes[route.route_id] = updated
        self._history.append(
            RouteTransition(
                route_id=route.route_id,
                from_state=route.state,
                to_state=state,
                generation=updated.generation,
                at=now,
                evidence_ref=evidence_ref,
            )
        )
        return updated
