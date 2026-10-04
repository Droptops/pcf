# JIT provider usage reconciliation

The JIT optimizer estimates input and prompt-cache cost before execution so it can compare routes. Closed providers do not expose their live cache inventory before a request, so those estimates must not be treated as billed truth.

`pcf.jit.pcf_reconcile` closes that measurement loop after execution without changing provider request semantics.

## Measurement flow

For a PCF router `Candidate`:

1. capture `PCFInputCostEstimate` before the route executes;
2. execute the shadow-safe route;
3. normalize the provider response through the existing compiler adapter into `pcf.compiler.Usage`;
4. price observed uncached, cache-creation, and cache-read tokens using the same candidate prices as the preflight estimate;
5. attach estimated and observed fields to the shadow run;
6. meter the JIT route with the observed input/cache cost.

OpenAI and Anthropic response parsing remains in their existing compiler adapters. The JIT reconciliation layer consumes only normalized `Usage` and therefore does not duplicate provider-specific billing-field logic.

## Error convention

Signed cost error is:

```text
estimate_error_usd = estimated_input_cache_cost - observed_input_cache_cost
```

Positive values mean the preflight estimate was high. Negative values mean it underestimated provider-reported input/cache billing.

The reconciliation also records signed token-bucket errors for cache reads, cache creation, uncached input, and total input. Absolute cost error is retained separately so over- and under-estimates cannot cancel in accuracy reporting.

## Runtime boundary

`make_pcf_reconciled_shadow_runner()` captures the estimate before execution and requires a normalized `Usage` object afterward. Missing or malformed usage fails closed for the reconciled path instead of silently mixing estimated and observed billing.

A reconciled `ShadowRun` uses observed input/cache cost for JIT economic comparison. The original estimate and its error remain in metadata for audit and calibration.

Already-metered runs and reserved reconciliation metadata collisions are rejected.

## What this does not claim

- Provider-reported usage is treated as the billing observation supplied by the provider adapter; this layer does not independently audit provider invoices.
- Output-token billing is not reconciled here because PCF's `warmth()` model concerns input and prompt-cache economics.
- Reconciliation does not automatically modify future route estimates.
- A small reconciliation sample is not evidence that a preflight estimator is calibrated.

`PCFReconciliationStats` aggregates count, signed cost error, absolute error, estimated/observed cost totals, and token-bucket error without retaining request payloads or raw provider responses. A later calibration gate can use that evidence to decide whether an estimator is reliable enough for economic routing.
