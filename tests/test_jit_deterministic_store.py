from __future__ import annotations

import json
import sqlite3

import pytest

from pcf.jit.deterministic import compile_deterministic_artifact
from pcf.jit.deterministic_store import (
    DeterministicArtifactStoreError,
    SQLiteDeterministicArtifactStore,
    resolve_registered_deterministic_artifact,
)
from pcf.jit.registry import RegisteredRoute, RouteState
from pcf.jit.types import ExecutionPrimitive


def _artifact():
    return compile_deterministic_artifact(
        name="normalize_email",
        version="1",
        dependencies={"synthesizer": "pcf.deterministic-synthesis.v1"},
        expression={
            "op": "lower",
            "arg": {"op": "strip", "arg": {"op": "input", "path": "email"}},
        },
    )


def _route(artifact, *, primitive=ExecutionPrimitive.DETERMINISTIC_CODE, metadata=None):
    return RegisteredRoute(
        route_id="route-1",
        name=artifact.name,
        route_version=artifact.version,
        candidate_key="normalize_email:compiled:v1",
        task_fingerprint="fingerprint-1",
        primitive=primitive,
        state=RouteState.ACTIVE,
        generation=3,
        traffic_fraction=1.0,
        created_at=1.0,
        updated_at=4.0,
        metadata=metadata if metadata is not None else {"artifact_sha256": artifact.artifact_sha256},
    )


def test_store_survives_restart_and_reverifies_artifact(tmp_path) -> None:
    path = tmp_path / "artifacts.sqlite"
    artifact = _artifact()
    first = SQLiteDeterministicArtifactStore(str(path))
    stored = first.put(artifact, created_at=10.0)
    assert stored.artifact_sha256 == artifact.artifact_sha256

    restarted = SQLiteDeterministicArtifactStore(str(path))
    loaded = restarted.get(artifact.artifact_sha256)
    assert loaded.artifact_sha256 == artifact.artifact_sha256
    assert loaded.expression == artifact.expression
    assert loaded.dependencies == artifact.dependencies
    assert restarted.contains(artifact.artifact_sha256)


def test_put_is_idempotent_and_preserves_original_created_at(tmp_path) -> None:
    path = tmp_path / "artifacts.sqlite"
    artifact = _artifact()
    store = SQLiteDeterministicArtifactStore(str(path))
    first = store.put(artifact, created_at=10.0)
    second = store.put(artifact, created_at=99.0)
    assert first == second
    assert second.created_at == 10.0


def test_corrupted_payload_fails_closed_on_read(tmp_path) -> None:
    path = tmp_path / "artifacts.sqlite"
    artifact = _artifact()
    store = SQLiteDeterministicArtifactStore(str(path))
    store.put(artifact, created_at=10.0)

    connection = sqlite3.connect(path)
    row = connection.execute(
        "SELECT payload_json FROM deterministic_artifacts WHERE artifact_sha256 = ?",
        (artifact.artifact_sha256,),
    ).fetchone()
    payload = json.loads(row[0])
    payload["expression"] = {"op": "const", "value": "tampered"}
    connection.execute(
        "UPDATE deterministic_artifacts SET payload_json = ? WHERE artifact_sha256 = ?",
        (json.dumps(payload, sort_keys=True, separators=(",", ":")), artifact.artifact_sha256),
    )
    connection.commit()
    connection.close()

    with pytest.raises(DeterministicArtifactStoreError, match="corrupt"):
        store.get(artifact.artifact_sha256)


def test_unknown_artifact_and_bad_hash_fail_cleanly(tmp_path) -> None:
    store = SQLiteDeterministicArtifactStore(str(tmp_path / "artifacts.sqlite"))
    with pytest.raises(ValueError, match="64-character"):
        store.get("short")
    with pytest.raises(KeyError, match="unknown deterministic artifact"):
        store.get("0" * 64)
    assert store.contains("short") is False


def test_registered_route_resolves_only_matching_deterministic_artifact(tmp_path) -> None:
    artifact = _artifact()
    store = SQLiteDeterministicArtifactStore(str(tmp_path / "artifacts.sqlite"))
    store.put(artifact, created_at=10.0)

    loaded = resolve_registered_deterministic_artifact(_route(artifact), store)
    assert loaded.artifact_sha256 == artifact.artifact_sha256

    with pytest.raises(DeterministicArtifactStoreError, match="not deterministic"):
        resolve_registered_deterministic_artifact(
            _route(artifact, primitive=ExecutionPrimitive.FRONTIER_MODEL),
            store,
        )
    with pytest.raises(DeterministicArtifactStoreError, match="no valid"):
        resolve_registered_deterministic_artifact(
            _route(artifact, metadata={}),
            store,
        )


def test_route_name_or_version_mismatch_fails_closed(tmp_path) -> None:
    artifact = _artifact()
    store = SQLiteDeterministicArtifactStore(str(tmp_path / "artifacts.sqlite"))
    store.put(artifact, created_at=10.0)
    route = _route(artifact)
    wrong = RegisteredRoute(**{**route.__dict__, "name": "different"})
    with pytest.raises(DeterministicArtifactStoreError, match="metadata does not match"):
        resolve_registered_deterministic_artifact(wrong, store)


def test_store_rejects_invalid_schema_version_on_restart(tmp_path) -> None:
    path = tmp_path / "artifacts.sqlite"
    SQLiteDeterministicArtifactStore(str(path))
    connection = sqlite3.connect(path)
    connection.execute("UPDATE deterministic_store_meta SET schema_version = 999 WHERE singleton = 1")
    connection.commit()
    connection.close()
    with pytest.raises(DeterministicArtifactStoreError, match="unsupported"):
        SQLiteDeterministicArtifactStore(str(path))
