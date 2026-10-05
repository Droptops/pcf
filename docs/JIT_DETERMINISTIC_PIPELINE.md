# Deterministic JIT pipeline

The deterministic target is now wired into the existing JIT evidence plane.

## Lifecycle

1. Observe a stable task fingerprint.
2. Require hot-path volume/stability and a clean adjudicated fingerprint audit.
3. Require compile economics to clear breakeven for projected future calls.
4. Compile the restricted declarative deterministic artifact.
5. Rebuild and verify the artifact identity immediately before execution.
6. Require exact held-out validation over at least the configured sample count.
7. Register the candidate in `SHADOW` with only non-sensitive build evidence.
8. Replay the deterministic route beside a baseline through `ShadowExecutor`.
9. Evaluate reliability, admissibility, quality, HTokens and savings with `PromotionEvaluator`.
10. Move `SHADOW -> ELIGIBLE` only through the existing registry promotion and fingerprint gates.

This module intentionally stops before production traffic. Existing canary/activation APIs remain separate so build-time evidence cannot silently become production authority.

## Authority boundary

The deterministic shadow adapter does not infer or grant authority. It reads explicit boolean authority and freshness evidence from the `ShadowRequest` metadata and fails closed when either is absent. FAAR remains authoritative.

## Artifact integrity

`execute_verified_deterministic_artifact()` recompiles the exposed artifact content and compares its SHA-256 identity, dependency set and structural counters before execution. This detects post-compile mutation or hand-constructed artifacts whose stored identity no longer matches their executable expression.

## Privacy boundary

Validation examples are used transiently for exact-match gating. Registration metadata stores counts, structural metrics, artifact identity and projected economics only; it does not persist validation inputs or outputs.
