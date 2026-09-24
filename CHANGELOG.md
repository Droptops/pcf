# Changelog

## Unreleased

Fixes from a coding-discipline review. Hashes and wire format of valid 0.2 documents are unchanged.

- OpenAI: assistant history is sent as plain text (Responses rejects `input_text` on assistant turns) and carries
  no cache marker; explicit mode is sent only when a marker was placed (explicit with none disables caching);
  parallel `function_call` items normalize into one assistant turn; unsupported boundaries no longer consume the
  breakpoint budget.
- Response normalization rejects what it cannot represent instead of dropping it: Anthropic text after `tool_use`
  or with citations, OpenAI `output_text` annotations and non-text `function_call_output` parts.
- Held-out validation fails when no sample reaches the threshold; `fit_platt` stops on the Newton decrement instead
  of stalling; malformed or truncated Jev responses fall back instead of raising; `number()` raises `ValueError` for
  huge ints.
- Empty `tools`/`history` segments bill zero tokens; mixed 0.1/0.2 tool results that would flip `is_error` are rejected.
- Schema: tool results require `is_error` (the hash covers it). SPEC now states the exact hash, prefix-chain, cache
  liveness and validation rules; dangling `SPEC.md §N` citations and the demo's calibrated-routing claim are corrected.

## 0.2.0

- Added immutable snapshots, RFC 8785 canonicalization, authority/provenance, and strict schemas.
- Preserved structured tool-call/result history across provider adapters.
- Corrected provider markers/profiles, TTL semantics, read-only inspection, and cache-write pricing.
- Added conservative descriptor identity and held-out calibration gates.
- Breaking: hashes, schemas, route decision fields, and history wire format changed from 0.1.
