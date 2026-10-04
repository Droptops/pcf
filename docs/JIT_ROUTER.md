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

## JIT compilation trigger

Compilation is justified only when projected future savings exceed build, evaluation, deployment, and risk-reserve costs:

```text
projected_calls * (general_cost_per_call - compiled_cost_per_call)
    >= build_cost + evaluation_cost + deployment_cost + risk_reserve
```

A hot-path detector separately requires enough observations and semantic stability before a workload is considered a
compile candidate.

## Current boundary

The runtime can now score, shadow-run, and replay execution routes, but it deliberately does **not** train models,
synthesize executable code, invoke write-capable tools, or auto-promote a candidate. Those capabilities require
empirical gates built from shadow evidence first.

Next experiments:

1. Add a task-fingerprint implementation and measure collision/semantic-drift rates.
2. Connect PCF prompt-cache cost estimates to shadow-run token/memory measurements.
3. Connect FAAR/AAR authority decisions as a hard admissibility gate rather than another weighted feature.
4. Define promotion gates from replay evidence: sample size, quality delta, HToken bound, failure rate, and savings.
5. Add compile targets in order: deterministic rules, classifier, fine-tuned specialist, then distilled small LLM.
