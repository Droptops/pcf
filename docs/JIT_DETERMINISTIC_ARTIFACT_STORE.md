# Deterministic JIT artifact store

A registry row is not enough to execute a compiled route after restart: it contains the artifact identity, not the executable declarative expression. `SQLiteDeterministicArtifactStore` provides an immutable, content-addressed reference store for the compiled deterministic artifact itself.

## Invariants

- The verified artifact SHA-256 is the primary key.
- Repeated writes of identical content are idempotent.
- A hash that maps to different serialized content fails closed.
- Store schema version is explicit and incompatible versions are rejected.
- Reads parse, recompile, re-hash and re-check structural counters before returning an artifact.
- Route resolution requires `DETERMINISTIC_CODE`, a valid `artifact_sha256`, and matching artifact name/version.

## Restart flow

1. Compile/synthesize the deterministic artifact.
2. Store it with `SQLiteDeterministicArtifactStore.put()` before depending on the route record.
3. Register/promote the route normally; registry metadata contains the artifact SHA-256.
4. After restart, reopen the artifact store and route registry.
5. Resolve the active route with `resolve_active_deterministic_artifact()`.
6. Execute through the verified deterministic execution path.

## Scope

Like `SQLiteRouteRegistry`, this is a single-database durable reference implementation, not a distributed artifact service or HA consensus system. Production multi-host deployments should back the same immutable/content-addressed contract with a networked durable store.
