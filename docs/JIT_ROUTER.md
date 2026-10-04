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

## Authority is a hard gate

`pcf.jit.authority` consumes the public shape of FAAR's final authority-layer `Decision` without making PCF depend on the
FAAR package. The adapter accepts the FAAR verdict vocabulary (`ALLOW`, `DENY`, `DEFER`, `STOP`), bounded reason codes,
and requires `layer="authority"`.

Only an explicit `ALLOW` sets the FAAR evidence to authorized. `DENY`, `DEFER`, and `STOP` all set the JIT route's
`authority_ok` to false, which the existing optimizer rejects before economic comparison. A cheaper route therefore
cannot buy its way past authority.

The adapter is monotone: applying an `ALLOW` to a run that is already unauthorized cannot elevate it back to authorized.
Authority verdict/layer/reason metadata uses reserved keys and collisions fail closed.

PCF deliberately does **not** verify FAAR attestations, reimplement FAAR posture/primitive rules, or inspect permits. The
caller must supply the result of FAAR's verified authority path. In FAAR itself, authority execution requires the
`EXECUTE` posture with the `EXECUTE_ACTION` primitive; that policy remains owned and tested by FAAR rather than copied
into PCF.

## Cache is an execution primitive

A semantic/result cache is treated as a route, but cache reuse must be invalidated by:

- TTL expiry
- dependency-version changes
- policy-version changes
- authority-scope changes
- an HToken budget for staleness

Prompt/KV cache economics remain the responsibility of PCF's existing cache-aware compiler. This layer decides whether
inference is needed at all and, if so, what execution primitive should serve it.

## PCF prompt-cache economics

`pcf.jit.pcf_cost` bridges the existing PCF compiler/router cache model into JIT shadow metering rather than reimplementing
provider tokenization or prompt-cache behavior.

For a PCF `Candidate`, the bridge asks the candidate compiler for its current `warmth()` and splits the existing router's
input-cost equation into the two JIT economic buckets:

```text
token_cost = uncached_tokens * input_price

memory_cost =
    cache_creation_tokens * cache_write_price
  + warm_tokens * cache_read_price
```

The sum is exactly the PCF router input-cost estimate for that candidate and cache state. The shadow metadata records the
warm/cold/cache-creation/uncached token counts, cache state, model id, observation time, and whether token counts are
estimated.

The metered shadow-run wrapper takes the cache-cost snapshot **before** route execution. A route that populates cache while
it runs therefore cannot make its own current request appear warm. The bridge rejects already-metered runs and reserved
metadata collisions to avoid silent double counting.

For simulated PCF families, locally observed cache metadata can produce warm-read estimates. For closed providers, PCF's
existing contract remains intact: provider cache state is `unknown` preflight and local cache metadata is never promoted
into knowledge of the provider's live cache. Output-token cost is not included by this bridge because PCF `warmth()`
models input caching only.

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

The audit reports rates, concrete examples, the fingerprints covered by adjudication, and per-fingerprint observation
counts. A clean hash function is not enough; the task-signature design must be validated on production-like traces before
its fingerprints are allowed to drive hot-path compilation or route reuse.

## Audited hot-path compilation gate

`HotPathDetector.is_hot()` remains a descriptive volume/stability signal. It does **not** authorize compilation.
`evaluate_compile_eligibility()` is the stronger gate. By default, a fingerprint needs at least 100 calls, 98% observed
stability, and 20 adjudicated fingerprint observations before it can become a compile candidate.

Compilation is blocked when the exact fingerprint is missing from audit coverage, is under-sampled, participates in a
semantic collision, shows behavior drift, or participates in a semantic split across multiple fingerprints. This stops a
frequent but incorrectly keyed workload from being compiled merely because its surface-level traffic looks repetitive.
The result is a structured `CompileEligibility` record with explicit failure reasons; it still does not build or deploy
anything.

## Staged route registry

`pcf.jit.registry.RouteRegistry` turns promotion evidence into an explicit lifecycle without touching production traffic:

```text
shadow -> eligible -> canary -> active -> retired
              ^         |
              |---------|  rollback
```

Forward transitions are evidence-gated:

- `shadow -> eligible` requires a passing `PromotionDecision` and collision/drift-free audit coverage for that exact task fingerprint
- `eligible -> canary` requires an explicit traffic fraction strictly between 0 and 1
- `canary -> active` requires fresh passing promotion evidence and clean fingerprint audit coverage again
- replacement of an existing active route must be explicit and retires the old route in the same registry operation

Every record has a monotonically increasing generation. Mutations require the caller's expected generation, so stale
operators cannot overwrite a newer lifecycle decision. Transition timestamps cannot move backwards, every transition
carries an `evidence_ref`, and retired routes are terminal. There can be at most one active route per task fingerprint.

Rollback is explicit: a canary can return to eligible, and an active route can be demoted back to canary. The registry is
an in-memory reference implementation; it changes route metadata only and does not deploy models or send traffic.

## JIT compilation trigger

Compilation is justified only when projected future savings exceed build, evaluation, deployment, and risk-reserve costs:

```text
projected_calls * (general_cost_per_call - compiled_cost_per_call)
    >= build_cost + evaluation_cost + deployment_cost + risk_reserve
```

The economic test is necessary but not sufficient: the audited hot-path gate must also pass before a workload is eligible
to enter a compilation pipeline.

## Current boundary

The runtime can score, fingerprint, shadow-run, bind FAAR authority as a hard gate, meter PCF prompt-cache economics,
replay, audit task-key collisions/drift, gate hot-path compilation eligibility, statistically gate candidates, and manage
an evidence-backed route lifecycle. It deliberately does **not** verify FAAR signatures, train models, synthesize
executable code, invoke write-capable tools, or send production traffic.

Next experiments:

1. Add durable registry persistence + transactional compare-and-swap semantics.
2. Add post-execution provider usage reconciliation so estimates can be compared with billed cache usage.
3. Add compile targets in order: deterministic rules, classifier, fine-tuned specialist, then distilled small LLM.
