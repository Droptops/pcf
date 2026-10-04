import pytest

from pcf.jit import (
    ExecutionPrimitive,
    FingerprintAudit,
    InvalidRouteTransition,
    PromotionDecision,
    PromotionEvidence,
    RouteRegistry,
    RouteState,
    StaleRouteGeneration,
)


FINGERPRINT = "sha256:task-a"


def passed_promotion(candidate_key, *, promote=True):
    evidence = PromotionEvidence(
        candidate=candidate_key,
        baseline="frontier",
        matched_samples=200,
        candidate_successes=200,
        candidate_errors=0,
        candidate_skipped=0,
        success_rate=1.0,
        success_rate_lower_bound=0.99,
        admissible_rate=1.0,
        admissible_rate_lower_bound=0.99,
        mean_quality_delta=0.0,
        quality_delta_lower_bound=-0.001,
        mean_candidate_htokens=0.001,
        mean_htoken_delta=-0.001,
        htoken_delta_upper_bound=-0.0005,
        mean_risk_adjusted_savings_usd_per_call=0.02,
        projected_future_calls=10_000,
        projected_net_savings_usd=199.0,
    )
    return PromotionDecision(promote=promote, reasons=() if promote else ("failed",), evidence=evidence)


def clean_audit(fingerprint=FINGERPRINT):
    audit = FingerprintAudit()
    audit.observe_fingerprint(
        fingerprint,
        semantic_id="territory-classification",
        behavior_id="rules-v1",
        observed_at=1,
    )
    return audit.summary()


def register(registry, *, name="specialist", version="v1", candidate_key="specialist-v1", now=1):
    return registry.register(
        name=name,
        route_version=version,
        candidate_key=candidate_key,
        task_fingerprint=FINGERPRINT,
        primitive=ExecutionPrimitive.SPECIALIST_MODEL,
        now=now,
    )


def advance_to_active(registry, route, *, start=2):
    route = registry.mark_eligible(
        route.route_id,
        promotion=passed_promotion(route.candidate_key),
        fingerprint_audit=clean_audit(route.task_fingerprint),
        evidence_ref="shadow-evidence",
        now=start,
        expected_generation=route.generation,
    )
    route = registry.start_canary(
        route.route_id,
        traffic_fraction=0.1,
        evidence_ref="canary-start",
        now=start + 1,
        expected_generation=route.generation,
    )
    return registry.activate(
        route.route_id,
        promotion=passed_promotion(route.candidate_key),
        fingerprint_audit=clean_audit(route.task_fingerprint),
        evidence_ref="canary-evidence",
        now=start + 2,
        expected_generation=route.generation,
    )


def test_route_moves_through_explicit_evidence_gated_states():
    registry = RouteRegistry()
    route = register(registry)
    assert route.state is RouteState.SHADOW
    assert route.generation == 0
    assert route.traffic_fraction == 0

    eligible = registry.mark_eligible(
        route.route_id,
        promotion=passed_promotion(route.candidate_key),
        fingerprint_audit=clean_audit(),
        evidence_ref="shadow-pass",
        now=2,
        expected_generation=0,
    )
    assert eligible.state is RouteState.ELIGIBLE
    assert eligible.generation == 1

    canary = registry.start_canary(
        route.route_id,
        traffic_fraction=0.05,
        evidence_ref="operator-canary",
        now=3,
        expected_generation=1,
    )
    assert canary.state is RouteState.CANARY
    assert canary.traffic_fraction == 0.05

    active = registry.activate(
        route.route_id,
        promotion=passed_promotion(route.candidate_key),
        fingerprint_audit=clean_audit(),
        evidence_ref="canary-pass",
        now=4,
        expected_generation=2,
    )
    assert active.state is RouteState.ACTIVE
    assert active.traffic_fraction == 1
    assert active.generation == 3
    assert registry.active(FINGERPRINT).route_id == active.route_id
    assert [item.to_state for item in registry.history(route.route_id)] == [
        RouteState.ELIGIBLE,
        RouteState.CANARY,
        RouteState.ACTIVE,
    ]


def test_failed_promotion_cannot_mark_shadow_route_eligible():
    registry = RouteRegistry()
    route = register(registry)
    with pytest.raises(InvalidRouteTransition, match="did not pass"):
        registry.mark_eligible(
            route.route_id,
            promotion=passed_promotion(route.candidate_key, promote=False),
            fingerprint_audit=clean_audit(),
            evidence_ref="failed",
            now=2,
            expected_generation=0,
        )


def test_fingerprint_collision_and_drift_block_forward_transition():
    for mode in ("collision", "drift"):
        registry = RouteRegistry()
        route = register(registry)
        audit = FingerprintAudit()
        audit.observe_fingerprint(
            FINGERPRINT,
            semantic_id="territory",
            behavior_id="v1",
            observed_at=1,
        )
        if mode == "collision":
            audit.observe_fingerprint(
                FINGERPRINT,
                semantic_id="ownership",
                behavior_id="v1",
                observed_at=2,
            )
        else:
            audit.observe_fingerprint(
                FINGERPRINT,
                semantic_id="territory",
                behavior_id="v2",
                observed_at=2,
            )
        with pytest.raises(InvalidRouteTransition, match="collision|drift"):
            registry.mark_eligible(
                route.route_id,
                promotion=passed_promotion(route.candidate_key),
                fingerprint_audit=audit.summary(),
                evidence_ref="bad-fingerprint",
                now=3,
                expected_generation=0,
            )


def test_fingerprint_audit_must_cover_the_registered_task():
    registry = RouteRegistry()
    route = register(registry)
    with pytest.raises(InvalidRouteTransition, match="no observations"):
        registry.mark_eligible(
            route.route_id,
            promotion=passed_promotion(route.candidate_key),
            fingerprint_audit=clean_audit("sha256:other"),
            evidence_ref="wrong-audit",
            now=2,
            expected_generation=0,
        )


def test_generation_and_time_checks_prevent_stale_transitions():
    registry = RouteRegistry()
    route = register(registry, now=10)
    eligible = registry.mark_eligible(
        route.route_id,
        promotion=passed_promotion(route.candidate_key),
        fingerprint_audit=clean_audit(),
        evidence_ref="pass",
        now=11,
        expected_generation=0,
    )
    with pytest.raises(StaleRouteGeneration):
        registry.start_canary(
            route.route_id,
            traffic_fraction=0.1,
            evidence_ref="stale",
            now=12,
            expected_generation=0,
        )
    with pytest.raises(InvalidRouteTransition, match="backwards"):
        registry.start_canary(
            route.route_id,
            traffic_fraction=0.1,
            evidence_ref="backwards",
            now=10.5,
            expected_generation=eligible.generation,
        )


def test_canary_fraction_is_bounded_and_canary_can_roll_back():
    registry = RouteRegistry()
    route = register(registry)
    eligible = registry.mark_eligible(
        route.route_id,
        promotion=passed_promotion(route.candidate_key),
        fingerprint_audit=clean_audit(),
        evidence_ref="pass",
        now=2,
        expected_generation=0,
    )
    with pytest.raises(InvalidRouteTransition, match="traffic_fraction"):
        registry.start_canary(
            route.route_id,
            traffic_fraction=1.0,
            evidence_ref="bad-canary",
            now=3,
            expected_generation=eligible.generation,
        )
    canary = registry.start_canary(
        route.route_id,
        traffic_fraction=0.2,
        evidence_ref="canary",
        now=3,
        expected_generation=eligible.generation,
    )
    rolled_back = registry.rollback_canary(
        route.route_id,
        evidence_ref="canary-regression",
        now=4,
        expected_generation=canary.generation,
    )
    assert rolled_back.state is RouteState.ELIGIBLE
    assert rolled_back.traffic_fraction == 0


def test_activation_requires_explicit_atomic_replacement_of_existing_active_route():
    registry = RouteRegistry()
    first = advance_to_active(registry, register(registry, candidate_key="specialist-v1"))

    second = register(
        registry,
        name="specialist-2",
        version="v2",
        candidate_key="specialist-v2",
        now=5,
    )
    second = registry.mark_eligible(
        second.route_id,
        promotion=passed_promotion(second.candidate_key),
        fingerprint_audit=clean_audit(),
        evidence_ref="shadow-2",
        now=6,
        expected_generation=0,
    )
    second = registry.start_canary(
        second.route_id,
        traffic_fraction=0.1,
        evidence_ref="canary-2",
        now=7,
        expected_generation=second.generation,
    )

    with pytest.raises(InvalidRouteTransition, match="explicit replacement"):
        registry.activate(
            second.route_id,
            promotion=passed_promotion(second.candidate_key),
            fingerprint_audit=clean_audit(),
            evidence_ref="activate-2",
            now=8,
            expected_generation=second.generation,
        )

    activated = registry.activate(
        second.route_id,
        promotion=passed_promotion(second.candidate_key),
        fingerprint_audit=clean_audit(),
        evidence_ref="activate-2",
        now=8,
        expected_generation=second.generation,
        replace_active_route_id=first.route_id,
        expected_active_generation=first.generation,
    )
    assert activated.state is RouteState.ACTIVE
    assert registry.get(first.route_id).state is RouteState.RETIRED
    assert registry.active(FINGERPRINT).route_id == activated.route_id


def test_active_route_can_be_demoted_and_retired_is_terminal():
    registry = RouteRegistry()
    active = advance_to_active(registry, register(registry))
    demoted = registry.demote_active(
        active.route_id,
        traffic_fraction=0.05,
        evidence_ref="rollback",
        now=5,
        expected_generation=active.generation,
    )
    assert demoted.state is RouteState.CANARY
    retired = registry.retire(
        demoted.route_id,
        evidence_ref="retire",
        now=6,
        expected_generation=demoted.generation,
    )
    assert retired.state is RouteState.RETIRED
    with pytest.raises(InvalidRouteTransition, match="terminal"):
        registry.retire(
            retired.route_id,
            evidence_ref="again",
            now=7,
            expected_generation=retired.generation,
        )


def test_candidate_key_is_unique_while_route_is_live():
    registry = RouteRegistry()
    first = register(registry)
    with pytest.raises(ValueError, match="candidate_key"):
        register(registry, name="duplicate", version="v2", candidate_key=first.candidate_key, now=2)
