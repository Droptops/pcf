# JIT cognitive execution optimizer (experimental)

This experiment treats model selection like telecom least-cost routing, but the route inventory is broader than models:

`result cache -> deterministic code -> specialist model -> small model -> frontier model`

The optimizer applies hard gates first, then minimizes risk-adjusted cost.

## Objective

For an execution route `r`:

```text
HTokens(r) = P(harm | r) * severity(r) * exposure(r)

risk_adjusted_cost(r) =
    token_cost
  + memory/cache_cost
  + HTokens * USD_per_HToken
  + latency_ms * USD_per_ms
```

A route is ineligible when it violates configured authority, freshness, minimum-quality, or maximum-HToken gates. The
optimizer never trades away those hard constraints for a lower aggregate score.

## Why HTokens

HTokens are not literal harmful output tokens. They are a normalized expected-harm unit that allows the runtime to price
risk without pretending every failure is equivalent. The conversion from HTokens to dollars is a policy parameter, not
a universal constant.

## Cache is an execution primitive

A semantic/result cache is treated as a route, but cache reuse must be invalidated by:

- TTL expiry
- dependency-version changes
- policy-version changes
- authority-scope changes
- an HToken budget for staleness

Prompt/KV cache economics remain the responsibility of PCF's existing cache-aware compiler. This layer decides whether
inference is needed at all and, if so, what execution primitive should serve it.

## Shadow runtime

`pcf.jit.shadow` replays one semantic request against multiple candidate routes and feeds the observed economics and an
external quality/risk adjudication back into `JITOptimizer`.

Shadow execution fails closed:

- a `ShadowTarget` defaults to `shadow_safe=False`
- disabled or non-shadow-safe routes are recorded as skipped and are never invoked
- each route receives a deep-copied request so in-process mutations do not leak to later candidates
- one route failure does not prevent other candidates from completing
- the runtime returns a recommendation only; it does not promote a route or perform production side effects

`shadow_safe=True` is an adapter contract, not a sandbox. A caller must only mark runners safe when the underlying route
cannot mutate external state. Tool execution and write-capable adapters remain outside this experiment.

A `TraceReplayer` aggregates empirical route statistics across requests: successes/errors, admissibility, selection
count, token cost, memory cost, latency, mean quality, and mean HTokens. This creates the evidence layer needed before
any route can be promoted or compiled.

## Promotion evidence

`pcf.jit.promotion` compares a candidate route with a named baseline on matched shadow observations. Aggregate averages
alone are not sufficient for promotion.

The default promotion policy requires:

- at least 100 matched successful observations
- a Wilson lower confidence bound on route success of at least 0.99
- a Wilson lower confidence bound on hard-gate admissibility of at least 0.99
- a one-sided paired quality bound no worse than 0.01 below the baseline
- mean candidate HTokens no greater than 0.05
- a one-sided paired HToken bound showing no positive HToken regression
- positive mean risk-adjusted savings per call
- an explicit future-call projection whose net savings clear promotion fixed cost

Quality and HToken bounds are currently normal-approximation bounds over paired samples; reliability/admissibility use
Wilson bounds. These are engineering promotion gates, not claims of universal statistical validity. Heavy-tailed or
high-consequence workloads should replace the approximate bounds with a domain-appropriate adjudication protocol.

Promotion remains advisory: `PromotionEvaluator` returns evidence plus pass/fail reasons and never changes routing state.
A skipped shadow run is itself disqualifying because it means the route has incomplete evidence under the replay set.

## Task fingerprints and drift audit

`pcf.jit.fingerprint.TaskSignature` hashes only declared stable task dimensions: operation, input/output schema identity,
policy version, authority scope, optional tool contract, and caller-declared stable features. Raw request values are
intentionally not part of the API so a task fingerprint does not silently become a user-data hash.

A fingerprint is only a routing key. It is not evidence that two requests mean the same thing. `FingerprintAudit` takes
adjudicated observations and measures three distinct failure modes:

- **collision**: one fingerprint maps to more than one semantic identity
- **semantic drift**: one fingerprint + semantic identity maps to more than one behavior identity
- **split**: one semantic identity maps to more than one fingerprint, indicating an unstable or over-specific key

The audit reports rates plus concrete examples. A clean hash function is not enough; the task-signature design must be
validated on production-like traces before its fingerprints are allowed to drive hot-path compilation or route reuse.

## JIT compilation trigger

Compilation is justified only when projected future savings exceed build, evaluation, deployment, and risk-reserve costs:

```text
projected_calls * (general_cost_per_call - compiled_cost_per_call)
    >= build_cost + evaluation_cost + deployment_cost + risk_reserve
```

A hot-path detector separately requires enough observations and semantic stability before a workload is considered a
compile candidate.

## Current boundary

The runtime can score, fingerprint, shadow-run, replay, audit task-key collisions/drift, and statistically gate candidate
routes, but it deliberately does **not** train models, synthesize executable code, invoke write-capable tools, or
automatically mutate production routing.

Next experiments:

1. Add a route registry with explicit staged states: shadow -> eligible -> canary -> active -> retired.
2. Connect PCF prompt-cache cost estimates to shadow-run token/memory measurements.
3. Connect FAAR/AAR authority decisions as a hard admissibility gate rather than another weighted feature.
4. Feed fingerprint collision/drift gates into hot-path eligibility.
5. Add compile targets in order: deterministic rules, classifier, fine-tuned specialist, then distilled small LLM.
