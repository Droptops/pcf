"""Durable SQLite-backed JIT route registry with transactional generation CAS."""
from __future__ import annotations

import json
import os
import sqlite3
from threading import RLock
from typing import Any, Callable, Mapping, TypeVar

from ..validation import number
from .fingerprint import FingerprintAuditSummary
from .promotion import PromotionDecision
from .registry import (
    RegisteredRoute,
    RouteRegistry,
    RouteState,
    RouteTransition,
    StaleRouteGeneration,
)
from .types import ExecutionPrimitive

T = TypeVar("T")


class RegistryStorageError(RuntimeError):
    """Persistent registry storage is unreadable, incompatible, or violates invariants."""


class SQLiteRouteRegistry(RouteRegistry):
    """SQLite persistence for the evidence-gated route lifecycle.

    Every mutation runs under ``BEGIN IMMEDIATE`` and reloads the durable state before
    applying the existing ``RouteRegistry`` policy. Updates also include a SQL
    ``WHERE generation = ?`` compare-and-swap guard. This provides durable,
    multi-process coordination for one SQLite database file; it is not a distributed
    consensus or high-availability datastore.
    """

    SCHEMA_VERSION = "1"

    def __init__(self, path: str | os.PathLike[str], *, busy_timeout_seconds: float = 5.0) -> None:
        super().__init__()
        self._path = os.fspath(path)
        if not self._path:
            raise ValueError("path must be non-empty")
        if self._path == ":memory:":
            raise ValueError("SQLiteRouteRegistry requires a durable database path, not :memory:")
        number(busy_timeout_seconds, "busy_timeout_seconds")
        if busy_timeout_seconds < 0:
            raise ValueError("busy_timeout_seconds must be non-negative")
        self._busy_timeout_seconds = float(busy_timeout_seconds)
        self._lock = RLock()
        self._in_write = False
        self._initialize()
        self._refresh()

    @property
    def path(self) -> str:
        return self._path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._path,
            timeout=self._busy_timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {int(self._busy_timeout_seconds * 1000)}")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jit_registry_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS jit_routes (
                    route_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    route_version TEXT NOT NULL,
                    candidate_key TEXT NOT NULL,
                    task_fingerprint TEXT NOT NULL,
                    primitive TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('shadow', 'eligible', 'canary', 'active', 'retired')),
                    generation INTEGER NOT NULL CHECK (generation >= 0),
                    traffic_fraction REAL NOT NULL CHECK (traffic_fraction >= 0 AND traffic_fraction <= 1),
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS jit_routes_live_candidate_unique
                    ON jit_routes(task_fingerprint, candidate_key)
                    WHERE state <> 'retired';

                CREATE UNIQUE INDEX IF NOT EXISTS jit_routes_one_active_per_task
                    ON jit_routes(task_fingerprint)
                    WHERE state = 'active';

                CREATE TABLE IF NOT EXISTS jit_route_history (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    route_id TEXT NOT NULL,
                    from_state TEXT NOT NULL,
                    to_state TEXT NOT NULL,
                    generation INTEGER NOT NULL CHECK (generation >= 1),
                    at REAL NOT NULL,
                    evidence_ref TEXT NOT NULL,
                    FOREIGN KEY(route_id) REFERENCES jit_routes(route_id)
                );

                CREATE INDEX IF NOT EXISTS jit_route_history_route_sequence
                    ON jit_route_history(route_id, sequence);
                """
            )
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT value FROM jit_registry_meta WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO jit_registry_meta(key, value) VALUES('schema_version', ?)",
                    (self.SCHEMA_VERSION,),
                )
            elif row["value"] != self.SCHEMA_VERSION:
                raise RegistryStorageError(
                    f"unsupported JIT registry schema version: {row['value']}"
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _encode_metadata(metadata: Mapping[str, Any]) -> str:
        try:
            encoded = json.dumps(
                dict(metadata),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            decoded = json.loads(encoded)
        except (TypeError, ValueError) as exc:
            raise ValueError("route metadata must be finite JSON-serializable data") from exc
        if not isinstance(decoded, dict):
            raise ValueError("route metadata must encode to a JSON object")
        return encoded

    @staticmethod
    def _decode_metadata(encoded: str) -> dict[str, Any]:
        try:
            value = json.loads(encoded)
        except (TypeError, ValueError) as exc:
            raise RegistryStorageError("stored route metadata is not valid JSON") from exc
        if not isinstance(value, dict):
            raise RegistryStorageError("stored route metadata must be a JSON object")
        return value

    def _load_locked(self, connection: sqlite3.Connection) -> None:
        routes: dict[str, RegisteredRoute] = {}
        try:
            for row in connection.execute(
                """
                SELECT route_id, name, route_version, candidate_key, task_fingerprint,
                       primitive, state, generation, traffic_fraction, created_at,
                       updated_at, metadata_json
                  FROM jit_routes
                """
            ):
                route = RegisteredRoute(
                    route_id=row["route_id"],
                    name=row["name"],
                    route_version=row["route_version"],
                    candidate_key=row["candidate_key"],
                    task_fingerprint=row["task_fingerprint"],
                    primitive=ExecutionPrimitive(row["primitive"]),
                    state=RouteState(row["state"]),
                    generation=int(row["generation"]),
                    traffic_fraction=float(row["traffic_fraction"]),
                    created_at=float(row["created_at"]),
                    updated_at=float(row["updated_at"]),
                    metadata=self._decode_metadata(row["metadata_json"]),
                )
                routes[route.route_id] = route

            history = [
                RouteTransition(
                    route_id=row["route_id"],
                    from_state=RouteState(row["from_state"]),
                    to_state=RouteState(row["to_state"]),
                    generation=int(row["generation"]),
                    at=float(row["at"]),
                    evidence_ref=row["evidence_ref"],
                )
                for row in connection.execute(
                    """
                    SELECT route_id, from_state, to_state, generation, at, evidence_ref
                      FROM jit_route_history
                     ORDER BY sequence
                    """
                )
            ]
        except (ValueError, KeyError, TypeError) as exc:
            raise RegistryStorageError("stored JIT registry data is invalid") from exc
        self._routes = routes
        self._history = history

    def _refresh_locked(self) -> None:
        if self._in_write:
            return
        connection = self._connect()
        try:
            self._load_locked(connection)
        finally:
            connection.close()

    def _refresh(self) -> None:
        with self._lock:
            self._refresh_locked()

    @staticmethod
    def _route_values(route: RegisteredRoute) -> tuple[Any, ...]:
        return (
            route.name,
            route.route_version,
            route.candidate_key,
            route.task_fingerprint,
            route.primitive.value,
            route.state.value,
            route.generation,
            route.traffic_fraction,
            route.created_at,
            route.updated_at,
            SQLiteRouteRegistry._encode_metadata(route.metadata),
        )

    def _persist_route_changes(
        self,
        connection: sqlite3.Connection,
        before: Mapping[str, RegisteredRoute],
    ) -> None:
        changed = [
            (route_id, before.get(route_id), route)
            for route_id, route in self._routes.items()
            if before.get(route_id) != route
        ]
        # Active replacement must retire the previous route before the new active row
        # is written, otherwise the durable unique index correctly rejects the transient
        # two-active state inside this transaction.
        changed.sort(key=lambda item: 0 if item[2].state is RouteState.RETIRED else 1)

        for route_id, previous, route in changed:
            values = self._route_values(route)
            if previous is None:
                connection.execute(
                    """
                    INSERT INTO jit_routes(
                        route_id, name, route_version, candidate_key, task_fingerprint,
                        primitive, state, generation, traffic_fraction, created_at,
                        updated_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (route_id, *values),
                )
                continue

            cursor = connection.execute(
                """
                UPDATE jit_routes
                   SET name = ?, route_version = ?, candidate_key = ?, task_fingerprint = ?,
                       primitive = ?, state = ?, generation = ?, traffic_fraction = ?,
                       created_at = ?, updated_at = ?, metadata_json = ?
                 WHERE route_id = ? AND generation = ?
                """,
                (*values, route_id, previous.generation),
            )
            if cursor.rowcount != 1:
                raise StaleRouteGeneration(
                    f"durable route generation changed while updating {route_id}"
                )

    def _persist_history(self, connection: sqlite3.Connection, start: int) -> None:
        for transition in self._history[start:]:
            connection.execute(
                """
                INSERT INTO jit_route_history(
                    route_id, from_state, to_state, generation, at, evidence_ref
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    transition.route_id,
                    transition.from_state.value,
                    transition.to_state.value,
                    transition.generation,
                    transition.at,
                    transition.evidence_ref,
                ),
            )

    def _mutate(self, operation: Callable[[], T]) -> T:
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                self._in_write = True
                self._load_locked(connection)
                before_routes = dict(self._routes)
                before_history = len(self._history)
                result = operation()
                self._persist_route_changes(connection, before_routes)
                self._persist_history(connection, before_history)
                connection.commit()
                return result
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                self._load_locked(connection)
                raise RegistryStorageError("durable registry invariant rejected the mutation") from exc
            except Exception:
                connection.rollback()
                self._load_locked(connection)
                raise
            finally:
                self._in_write = False
                connection.close()

    def get(self, route_id: str) -> RegisteredRoute:
        with self._lock:
            self._refresh_locked()
            return RouteRegistry.get(self, route_id)

    def routes_for_fingerprint(self, task_fingerprint: str) -> tuple[RegisteredRoute, ...]:
        with self._lock:
            self._refresh_locked()
            return RouteRegistry.routes_for_fingerprint(self, task_fingerprint)

    def active(self, task_fingerprint: str) -> RegisteredRoute | None:
        with self._lock:
            self._refresh_locked()
            active = [
                route
                for route in RouteRegistry.routes_for_fingerprint(self, task_fingerprint)
                if route.state is RouteState.ACTIVE
            ]
            if len(active) > 1:
                raise RegistryStorageError("durable registry has multiple active routes for one task fingerprint")
            return active[0] if active else None

    def history(self, route_id: str | None = None) -> tuple[RouteTransition, ...]:
        with self._lock:
            self._refresh_locked()
            return RouteRegistry.history(self, route_id)

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
        return self._mutate(
            lambda: RouteRegistry.register(
                self,
                name=name,
                route_version=route_version,
                task_fingerprint=task_fingerprint,
                primitive=primitive,
                now=now,
                candidate_key=candidate_key,
                metadata=metadata,
            )
        )

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
        return self._mutate(
            lambda: RouteRegistry.mark_eligible(
                self,
                route_id,
                promotion=promotion,
                fingerprint_audit=fingerprint_audit,
                evidence_ref=evidence_ref,
                now=now,
                expected_generation=expected_generation,
            )
        )

    def start_canary(
        self,
        route_id: str,
        *,
        traffic_fraction: float,
        evidence_ref: str,
        now: float,
        expected_generation: int,
    ) -> RegisteredRoute:
        return self._mutate(
            lambda: RouteRegistry.start_canary(
                self,
                route_id,
                traffic_fraction=traffic_fraction,
                evidence_ref=evidence_ref,
                now=now,
                expected_generation=expected_generation,
            )
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
        return self._mutate(
            lambda: RouteRegistry.activate(
                self,
                route_id,
                promotion=promotion,
                fingerprint_audit=fingerprint_audit,
                evidence_ref=evidence_ref,
                now=now,
                expected_generation=expected_generation,
                replace_active_route_id=replace_active_route_id,
                expected_active_generation=expected_active_generation,
            )
        )

    def rollback_canary(
        self,
        route_id: str,
        *,
        evidence_ref: str,
        now: float,
        expected_generation: int,
    ) -> RegisteredRoute:
        return self._mutate(
            lambda: RouteRegistry.rollback_canary(
                self,
                route_id,
                evidence_ref=evidence_ref,
                now=now,
                expected_generation=expected_generation,
            )
        )

    def demote_active(
        self,
        route_id: str,
        *,
        traffic_fraction: float,
        evidence_ref: str,
        now: float,
        expected_generation: int,
    ) -> RegisteredRoute:
        return self._mutate(
            lambda: RouteRegistry.demote_active(
                self,
                route_id,
                traffic_fraction=traffic_fraction,
                evidence_ref=evidence_ref,
                now=now,
                expected_generation=expected_generation,
            )
        )

    def retire(
        self,
        route_id: str,
        *,
        evidence_ref: str,
        now: float,
        expected_generation: int,
    ) -> RegisteredRoute:
        return self._mutate(
            lambda: RouteRegistry.retire(
                self,
                route_id,
                evidence_ref=evidence_ref,
                now=now,
                expected_generation=expected_generation,
            )
        )
