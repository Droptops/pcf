"""Durable immutable storage for verified deterministic JIT artifacts."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import sqlite3
from threading import RLock
from typing import Any

from .deterministic import DeterministicArtifact, compile_deterministic_artifact
from .deterministic_pipeline import DeterministicPipelineError, verify_deterministic_artifact
from .registry import RegisteredRoute, RouteRegistry
from .types import ExecutionPrimitive


_STORE_SCHEMA_VERSION = 1
_ARTIFACT_PAYLOAD_SCHEMA = "pcf.deterministic-stored.v1"


class DeterministicArtifactStoreError(RuntimeError):
    """The deterministic artifact store is unavailable, corrupt, or inconsistent."""


@dataclass(frozen=True)
class StoredDeterministicArtifact:
    artifact_sha256: str
    created_at: float


class SQLiteDeterministicArtifactStore:
    """Immutable content-addressed SQLite store for deterministic artifacts.

    Rows are keyed by the verified artifact SHA-256. A repeated put of identical
    content is idempotent; the same key with different serialized content fails
    closed. Every read recompiles and re-verifies the artifact before returning it.
    """

    def __init__(self, path: str, *, timeout_seconds: float = 5.0) -> None:
        if not isinstance(path, str) or not path.strip():
            raise ValueError("path must be non-empty")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        self.path = path
        self.timeout_seconds = float(timeout_seconds)
        self._lock = RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=self.timeout_seconds, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS deterministic_store_meta(
                        singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                        schema_version INTEGER NOT NULL
                    )
                    """
                )
                row = connection.execute(
                    "SELECT schema_version FROM deterministic_store_meta WHERE singleton = 1"
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO deterministic_store_meta(singleton, schema_version) VALUES(1, ?)",
                        (_STORE_SCHEMA_VERSION,),
                    )
                elif int(row["schema_version"]) != _STORE_SCHEMA_VERSION:
                    raise DeterministicArtifactStoreError("unsupported deterministic artifact store schema")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS deterministic_artifacts(
                        artifact_sha256 TEXT PRIMARY KEY,
                        payload_json TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                    """
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    @staticmethod
    def _payload(artifact: DeterministicArtifact) -> dict[str, Any]:
        return {
            "schema": _ARTIFACT_PAYLOAD_SCHEMA,
            "name": artifact.name,
            "version": artifact.version,
            "dependencies": [list(item) for item in artifact.dependencies],
            "expression": artifact.expression,
            "artifact_sha256": artifact.artifact_sha256,
            "node_count": artifact.node_count,
            "max_depth": artifact.max_depth,
        }

    @staticmethod
    def _encode_payload(artifact: DeterministicArtifact) -> str:
        verify_deterministic_artifact(artifact)
        try:
            return json.dumps(
                SQLiteDeterministicArtifactStore._payload(artifact),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise DeterministicArtifactStoreError("artifact cannot be serialized") from exc

    def put(self, artifact: DeterministicArtifact, *, created_at: float) -> StoredDeterministicArtifact:
        if (
            isinstance(created_at, bool)
            or not isinstance(created_at, (int, float))
            or not math.isfinite(float(created_at))
            or created_at < 0
        ):
            raise ValueError("created_at must be finite and non-negative")
        payload = self._encode_payload(artifact)
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT payload_json, created_at FROM deterministic_artifacts WHERE artifact_sha256 = ?",
                    (artifact.artifact_sha256,),
                ).fetchone()
                if existing is not None:
                    if existing["payload_json"] != payload:
                        raise DeterministicArtifactStoreError("artifact hash maps to different stored content")
                    connection.commit()
                    return StoredDeterministicArtifact(
                        artifact_sha256=artifact.artifact_sha256,
                        created_at=float(existing["created_at"]),
                    )
                connection.execute(
                    "INSERT INTO deterministic_artifacts(artifact_sha256, payload_json, created_at) VALUES(?, ?, ?)",
                    (artifact.artifact_sha256, payload, float(created_at)),
                )
                connection.commit()
                return StoredDeterministicArtifact(artifact.artifact_sha256, float(created_at))
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    @staticmethod
    def _decode_payload(artifact_sha256: str, payload_json: str) -> DeterministicArtifact:
        try:
            payload = json.loads(payload_json)
            if not isinstance(payload, dict) or payload.get("schema") != _ARTIFACT_PAYLOAD_SCHEMA:
                raise ValueError("invalid artifact payload schema")
            dependencies_raw = payload["dependencies"]
            if not isinstance(dependencies_raw, list):
                raise ValueError("dependencies must be a list")
            dependencies: dict[str, str] = {}
            for item in dependencies_raw:
                if not isinstance(item, list) or len(item) != 2:
                    raise ValueError("invalid dependency entry")
                key, value = item
                if not isinstance(key, str) or not isinstance(value, str) or key in dependencies:
                    raise ValueError("invalid or duplicate dependency")
                dependencies[key] = value
            artifact = compile_deterministic_artifact(
                name=payload["name"],
                version=payload["version"],
                expression=payload["expression"],
                dependencies=dependencies,
            )
            if payload.get("artifact_sha256") != artifact_sha256:
                raise ValueError("stored artifact identity does not match row key")
            if artifact.artifact_sha256 != artifact_sha256:
                raise ValueError("recompiled artifact identity does not match row key")
            if payload.get("node_count") != artifact.node_count or payload.get("max_depth") != artifact.max_depth:
                raise ValueError("stored artifact structure counters do not match")
            verify_deterministic_artifact(artifact)
            return artifact
        except Exception as exc:
            if isinstance(exc, DeterministicArtifactStoreError):
                raise
            raise DeterministicArtifactStoreError("stored deterministic artifact is corrupt") from exc

    def get(self, artifact_sha256: str) -> DeterministicArtifact:
        if not isinstance(artifact_sha256, str) or len(artifact_sha256) != 64:
            raise ValueError("artifact_sha256 must be a 64-character string")
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    "SELECT payload_json FROM deterministic_artifacts WHERE artifact_sha256 = ?",
                    (artifact_sha256,),
                ).fetchone()
            finally:
                connection.close()
        if row is None:
            raise KeyError(f"unknown deterministic artifact: {artifact_sha256}")
        return self._decode_payload(artifact_sha256, row["payload_json"])

    def contains(self, artifact_sha256: str) -> bool:
        if not isinstance(artifact_sha256, str) or len(artifact_sha256) != 64:
            return False
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    "SELECT 1 FROM deterministic_artifacts WHERE artifact_sha256 = ?",
                    (artifact_sha256,),
                ).fetchone()
                return row is not None
            finally:
                connection.close()


def resolve_registered_deterministic_artifact(
    route: RegisteredRoute,
    store: SQLiteDeterministicArtifactStore,
) -> DeterministicArtifact:
    """Resolve and cross-check the immutable artifact referenced by a registry row."""
    if not isinstance(route, RegisteredRoute):
        raise DeterministicArtifactStoreError("route must be RegisteredRoute")
    if route.primitive is not ExecutionPrimitive.DETERMINISTIC_CODE:
        raise DeterministicArtifactStoreError("route is not deterministic code")
    artifact_sha256 = route.metadata.get("artifact_sha256")
    if not isinstance(artifact_sha256, str) or len(artifact_sha256) != 64:
        raise DeterministicArtifactStoreError("route has no valid deterministic artifact identity")
    artifact = store.get(artifact_sha256)
    if artifact.name != route.name or artifact.version != route.route_version:
        raise DeterministicArtifactStoreError("stored artifact metadata does not match registered route")
    return artifact


def resolve_active_deterministic_artifact(
    registry: RouteRegistry,
    store: SQLiteDeterministicArtifactStore,
    task_fingerprint: str,
) -> tuple[RegisteredRoute, DeterministicArtifact]:
    """Resolve the current active deterministic route and its verified durable artifact."""
    if not isinstance(registry, RouteRegistry):
        raise DeterministicArtifactStoreError("registry must be RouteRegistry")
    active = registry.active(task_fingerprint)
    if active is None:
        raise KeyError(f"no active route for task fingerprint: {task_fingerprint}")
    return active, resolve_registered_deterministic_artifact(active, store)
