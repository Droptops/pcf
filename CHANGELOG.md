# Changelog

## Unreleased

Fixes from coding reviews. Portable segment hashes and PCF document wire format remain unchanged; compiler cache keys, simulated billing counts, candidate fingerprints and calibration record identities change as described below.

- Jev's HTTPS transport rejects all redirects so bearer credentials cannot follow a different origin or an HTTP downgrade. Redirects trigger confidence fallback.
- Held-out validation rejects repeated context hashes and requires enough selected tail examples even when no tail score reaches the threshold. All sample minimums are included in validation record identity. Existing scorers with only below-threshold tail evidence now fall back.
- Token accounting uses cumulative canonical native input, matching cache identity. Splitting or combining history segments cannot change billing for the same native prefix. The accounting version in compiler cache keys invalidates older metadata and candidate validations. The character-chunk tokenizer's identity now includes its validated configuration.
- Added regression coverage for redirects, evidence minimums, duplicate samples, regrouped history, and tokenizer identity; CI covers Python 3.11–3.14, lint, offline smoke checks, wheel installation, and session-link prevention in tracked text and new commits.

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
- Anthropic compilation rejects an empty message list and, on the Claude 4.6+ model IDs it lists, a final assistant
  turn (prefill). These raise `UnsupportedRequest`; the router leaves such a candidate out, and raises only when it is
  the fallback.
- `PrefixCache` keeps one store-wide event clock: `write`/`touch`/`prune` reject earlier times and `peek` rejects a
  time before the latest event, so warmth can no longer report a hit before the entry existed.
- OpenAI requests in implicit mode now estimate the provider's own breakpoint as a write of the whole prompt, so
  route costs include the cache-write premium. Caller-supplied implicit-only profiles default to free writes (1x),
  like the built-in pre-5.6 profiles; explicit profiles keep 1.25x.
- Schemas now reject what the runtime rejects: empty or whitespace-only strings (whitespace as Python's `str.strip()`
  defines it, spelled out so ECMA-262 validators agree), empty history turns, duplicate axes, and `byte_compat_key`
  without a complete manifest; `$` anchors no longer accept a trailing newline under Python's `re`. Serialized
  documents may no longer carry the 0.1 assistant `tool_use` history form (constructors still migrate it).
- Added tests for route cost math, TTL refresh, cheapest-validated routing, eviction, namespace isolation, usage
  parsing, Jev guards, validation revocation and `byte_compat_key` gating.

## 0.2.0

- Added immutable snapshots, RFC 8785 canonicalization, authority/provenance, and strict schemas.
- Preserved structured tool-call/result history across provider adapters.
- Corrected provider markers/profiles, TTL semantics, read-only inspection, and cache-write pricing.
- Added conservative descriptor identity and held-out calibration gates.
- Breaking: hashes, schemas, route decision fields, and history wire format changed from 0.1.
