# Deterministic JIT reference lifecycle

The deterministic JIT path now has a complete single-node reference lifecycle:

`observe -> fingerprint -> hot/audited -> synthesize -> held-out validate -> persist artifact -> shadow -> eligible -> canary -> active -> restart resolve -> monitor -> demote`

The end-to-end test uses the real `SQLiteRouteRegistry` and `SQLiteDeterministicArtifactStore`, closes/reopens both durable stores after activation, resolves the active artifact by its verified SHA-256 identity, executes it after restart, then applies generation-bound monitoring evidence and verifies demotion persists through another registry reopen.

## What this proves

- hot-path volume/stability and fingerprint audit gates are enforced before compilation;
- bounded synthesis emits only the restricted deterministic IR;
- training and held-out validation are separate;
- compile economics must clear breakeven;
- artifacts are integrity-checked and durably content-addressed;
- shadow execution cannot self-authorize;
- SHADOW -> ELIGIBLE uses statistical promotion and clean-fingerprint evidence;
- CANARY -> ACTIVE requires fresh evidence bound to the exact canary generation;
- active routes can be reconstructed after process restart;
- ACTIVE -> CANARY demotion requires actionable health evidence bound to the exact active generation;
- stale evidence cannot operate on a newer generation.

## Completion boundary

This is a complete **single-node deterministic reference implementation**, not a claim of distributed production readiness. The SQLite registry and artifact store deliberately do not provide multi-host HA or consensus. Live traffic sampling/dispatch remains an embedding concern: production callers must supply authority/freshness evidence and paired telemetry to the same lifecycle APIs. Other compile targets such as learned classifiers or distilled specialist models are separate migration-ladder extensions, not required to validate the deterministic reference path.
