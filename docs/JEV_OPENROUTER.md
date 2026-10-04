# Jev through OpenRouter

PCF can use TypeSafe Jev as a fast **advisory task-shape classifier** through OpenRouter's Decisions API.

The pinned default is `typesafe/jev-1.13` and the endpoint is:

```text
POST https://openrouter.ai/api/alpha/decisions
```

Authentication comes from `OPENROUTER_API_KEY`. PCF never requires a TypeSafe-specific credential for this path.

## What Jev decides

The adapter asks one `choice` question over the five PCF execution primitives:

```text
result_cache
  -> deterministic_code
  -> specialist_model
  -> small_model
  -> frontier_model
```

The question explicitly asks Jev to classify **semantic task shape only**. It must not reason about authorization, safety,
price, or current route availability.

That separation is deliberate. Jev evidence is metadata, not authority.

## What Jev cannot change

`annotate_candidates_with_jev()` preserves every routing field that matters to PCF policy:

- `authority_ok`
- `freshness_ok`
- `enabled`
- token/cache/uncertainty cost
- expected quality
- HToken inputs
- latency

FAAR remains the authority boundary. PCF still applies authority, freshness, quality, and harm hard gates before minimizing
risk-adjusted cost.

## Response validation

The adapter fails closed unless the Decisions response contains:

- a `choice` answer for `execution_primitive`
- exactly one probability for every PCF execution primitive
- probabilities in `[0, 1]` that sum to one within a small floating-point tolerance
- confidence in `[0, 1]`
- provider usage/cost telemetry
- model, provider, and decision identifiers

Both snake-case and camel-case OpenRouter usage token fields are accepted because current examples expose both forms.

## Credential boundary

The default client uses only Python's standard library. A supplied OpenRouter API key may only be sent to an
`https://openrouter.ai/...` endpoint. Client `repr()` does not contain the key, and HTTP error messages do not include the
response body.

The live smoke script reads the credential from the environment and prints only sanitized decision telemetry:

```bash
OPENROUTER_API_KEY=... python scripts/live_jev_smoke.py
```

Do not commit or paste the key into source, tests, issues, or PR comments.

## Benchmarking

`JevRouteBenchmarkStats` tracks labeled route-shape accuracy, low-confidence rate, mean confidence, decision cost, and
latency without retaining request state.

The first live benchmark should compare Jev's primitive classification with adjudicated labels. Only after that evidence
is collected should PCF test any policy that uses Jev probabilities to reduce the candidate set. The v1 adapter does not
filter candidates automatically.
