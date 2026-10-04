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

## JIT compilation trigger

Compilation is justified only when projected future savings exceed build, evaluation, deployment, and risk-reserve costs:

```text
projected_calls * (general_cost_per_call - compiled_cost_per_call)
    >= build_cost + evaluation_cost + deployment_cost + risk_reserve
```

A hot-path detector separately requires enough observations and semantic stability before a workload is considered a
compile candidate.

## v0 boundaries

This patch deliberately does **not** train models, synthesize code, or execute tools. It establishes the decision
contract and safety/economic invariants needed before adding those capabilities.

Next experiments:

1. Replay production-like traces and estimate route-level HTokens from adjudicated failures.
2. Add a shadow-mode route executor and compare cache/code/specialist/frontier candidates on identical requests.
3. Learn a task fingerprint and measure collision/semantic-drift rates.
4. Connect PCF prompt-cache cost estimates to the model-route cost term.
5. Connect FAAR/AAR authority decisions as a hard admissibility gate rather than another weighted feature.
6. Add compile targets in order: deterministic rules, classifier, fine-tuned specialist, then distilled small LLM.
