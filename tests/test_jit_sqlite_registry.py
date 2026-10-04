import sqlite3

import pytest

from pcf.jit import (
    ExecutionPrimitive,
    FingerprintAudit,
    InvalidRouteTransition,
    PromotionDecision,
    PromotionEvidence,
    RegistryStorageError,
    RouteState,
    SQLiteRouteRegistry,
    StaleRouteGeneration,
)


FINGERPRINT = "sha256:sqlite-task"


def passed_promotion(candidate_key):
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
    return PromotionDecision(promote=True, reasons=(), evidence=evidence)


def clean_audit():
    audit = FingerprintAudit()
    for i in range(20):
        audit.observe_fingerprint(
            FINGERPRINT,
            semantic_id="territory-classification",
            behavior_id="rules-v1",
            observed_at=float(i),
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
        metadata={"owner": "jit", "revision": 1},
    )


def advance_to_canary(registry, route, *, start=2):
    route = registry.mark_eligible(
        route.route_id,
        promotion=passed_promotion(route.candidate_key),
        fingerprint_audit=clean_audit(),
        evidence_ref="shadow-pass",
        now=start,
        expected_generation=route.generation,
    )
    return registry.start_canary(
        route.route_id,
        traffic_fraction=0.1,
        evidence_ref="canary-start",
        now=start + 1,
        expected_generation=route.generation,
    )


def advance_to_active(registry, route, *, start=2):
    route = advance_to_canary(registry, route, start=start)
    return registry.activate(
        route.route_id,
        promotion=passed_promotion(route.candidate_key),
        fingerprint_audit=clean_audit(),
        evidence_ref="canary-pass",
        now=start + 2,
        expected_generation=route.generation,
    )


def test_state_and_history_survive_new_registry_instance(tmp_path):
    path = tmp_path / "routes.sqlite3"
    first = SQLiteRouteRegistry(path)
    route = advance_to_active(first, register(first))

    reopened = SQLiteRouteRegistry(path)
    loaded = reopened.get(route.route_id)
    assert loaded == route
    assert loaded.metadata == {"owner": "jit", "revision": 1}
    assert reopened.active(FINGERPRINT).route_id == route.route_id
    assert [item.to_state for item in reopened.history(route.route_id)] == [
        RouteState.ELIGIBLE,
        RouteState.CANARY,
        RouteState.ACTIVE,
    ]


def test_two_instances_observe_durable_generation_cas(tmp_path):
    path = tmp_path / "routes.sqlite3"
    first = SQLiteRouteRegistry(path)
    second = SQLiteRouteRegistry(path)
    route = register(first)

    eligible = first.mark_eligible(
        route.route_id,
        promotion=passed_promotion(route.candidate_key),
        fingerprint_audit=clean_audit(),
        evidence_ref="first-writer",
        now=2,
        expected_generation=0,
    )
    assert second.get(route.route_id).generation == eligible.generation

    with pytest.raises(StaleRouteGeneration):
        second.start_canary(
            route.route_id,
            traffic_fraction=0.1,
            evidence_ref="stale-writer",
            now=3,
            expected_generation=0,
        )

    assert SQLiteRouteRegistry(path).get(route.route_id).state is RouteState.ELIGIBLE
    assert len(SQLiteRouteRegistry(path).history(route.route_id)) == 1


def test_failed_mutation_rolls_back_route_and_history_together(tmp_path):
    path = tmp_path / "routes.sqlite3"
    registry = SQLiteRouteRegistry(path)
    route = register(registry)

    with pytest.raises(InvalidRouteTransition, match="route must be eligible"):
        registry.start_canary(
            route.route_id,
            traffic_fraction=0.1,
            evidence_ref="invalid",
            now=2,
            expected_generation=0,
        )

    reopened = SQLiteRouteRegistry(path)
    assert reopened.get(route.route_id).state is RouteState.SHADOW
    assert reopened.get(route.route_id).generation == 0
    assert reopened.history(route.route_id) == ()


def test_active_replacement_is_one_durable_transaction(tmp_path):
    path = tmp_path / "routes.sqlite3"
    registry = SQLiteRouteRegistry(path)
    first = advance_to_active(registry, register(registry, candidate_key="specialist-v1"))

    second = register(
        registry,
        name="specialist-2",
        version="v2",
        candidate_key="specialist-v2",
        now=5,
    )
    second = advance_to_canary(registry, second, start=6)
    second = registry.activate(
        second.route_id,
        promotion=passed_promotion(second.candidate_key),
        fingerprint_audit=clean_audit(),
        evidence_ref="replace-active",
        now=8,
        expected_generation=second.generation,
        replace_active_route_id=first.route_id,
        expected_active_generation=first.generation,
    )

    reopened = SQLiteRouteRegistry(path)
    assert reopened.active(FINGERPRINT).route_id == second.route_id
    assert reopened.get(first.route_id).state is RouteState.RETIRED
    assert reopened.get(second.route_id).state is RouteState.ACTIVE
    assert [item.to_state for item in reopened.history()[-2:]] == [
        RouteState.RETIRED,
        RouteState.ACTIVE,
    ]


def test_other_instance_reads_new_writes_without_reconstruction(tmp_path):
    path = tmp_path / "routes.sqlite3"
    writer = SQLiteRouteRegistry(path)
    reader = SQLiteRouteRegistry(path)
    route = register(writer)

    assert reader.get(route.route_id).route_id == route.route_id
    retired = writer.retire(
        route.route_id,
        evidence_ref="stop",
        now=2,
        expected_generation=0,
    )
    assert reader.get(route.route_id).generation == retired.generation
    assert reader.get(route.route_id).state is RouteState.RETIRED


def test_non_json_metadata_fails_without_leaving_partial_route(tmp_path):
    path = tmp_path / "routes.sqlite3"
    registry = SQLiteRouteRegistry(path)

    with pytest.raises(ValueError, match="JSON-serializable"):
        registry.register(
            name="bad",
            route_version="v1",
            candidate_key="bad-v1",
            task_fingerprint=FINGERPRINT,
            primitive=ExecutionPrimitive.DETERMINISTIC_CODE,
            now=1,
            metadata={"bad": object()},
        )

    assert SQLiteRouteRegistry(path).routes_for_fingerprint(FINGERPRINT) == ()


def test_database_unique_index_defends_single_active_invariant(tmp_path):
    path = tmp_path / "routes.sqlite3"
    registry = SQLiteRouteRegistry(path)
    first = advance_to_active(registry, register(registry, candidate_key="specialist-v1"))
    second = register(
        registry,
        name="specialist-2",
        version="v2",
        candidate_key="specialist-v2",
        now=5,
    )

    connection = sqlite3.connect(path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE jit_routes SET state = 'active', traffic_fraction = 1 WHERE route_id = ?",
                (second.route_id,),
            )
        connection.rollback()
    finally:
        connection.close()

    assert SQLiteRouteRegistry(path).active(FINGERPRINT).route_id == first.route_id


def test_schema_version_mismatch_fails_closed(tmp_path):
    path = tmp_path / "routes.sqlite3"
    SQLiteRouteRegistry(path)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "UPDATE jit_registry_meta SET value = '999' WHERE key = 'schema_version'"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RegistryStorageError, match="schema version"):
        SQLiteRouteRegistry(path)
