# Deterministic JIT canary activation

The hardened deterministic lifecycle binds activation evidence to the exact canary generation that produced it.

## Flow

`ELIGIBLE -> CANARY -> ACTIVE`

1. `start_deterministic_canary()` checks that the eligible registry row matches the compiled artifact identity.
2. Canary telemetry is evaluated with the existing paired `PromotionEvaluator` plus a minimum live-attempt count.
3. `DeterministicCanaryEvidence` binds the result to route ID, route generation, candidate key, task fingerprint, traffic fraction, observation window ID, and evaluation time.
4. `activate_deterministic_candidate()` refuses evidence from a different or older canary generation, a different traffic fraction, candidate, fingerprint, or artifact.
5. Failed/insufficient evidence can be passed to `rollback_failed_deterministic_canary()` to return the route to `ELIGIBLE`.

If a canary is rolled back and restarted, its generation changes. Evidence from the old window is therefore stale and cannot activate the restarted route.

## Evidence boundary

The generic `RouteRegistry` is a low-level lifecycle primitive. Deterministic JIT callers should use the canary-bound functions in this module for activation so shadow promotion evidence cannot be reused as if it were fresh canary evidence.

FAAR/fingerprint gates remain unchanged. Activation still invokes the registry's clean-fingerprint and promotion checks after the canary binding checks pass.
