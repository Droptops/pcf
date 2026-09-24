# PCF 0.2 specification

## Document

Documents contain ordered `tools`, `system`, `memory`/`document`, `history`, and `user` segments. IDs are unique. Instructions precede data and are never silently reordered. `authority` is explicit: tools/system require `instruction`; history/user require `data`. `stable` is a breakpoint hint only.

Segment hash = `"sha256:" + hex(SHA-256("pcf:segment:0.2" || 0x00 || RFC8785({"kind", "authority", "provenance", "content"})))`, using the resolved authority, `provenance` null when absent, and normalized content (tool results always carry `is_error`); IDs and stability are excluded. Prefix chain: `c_0 = SHA-256("pcf:0.2")`, `c_i = SHA-256(c_{i-1} || h_i)` over raw 32-byte digests; the seed `c_0` is not emitted. Caller-owned objects are snapshotted.

## History and providers

Neutral history has user/assistant turns, assistant `tool_calls` (`id`, `name`, object `arguments`), and tool results (`call_id`, content, `is_error`). Calls must be resolved before compilation. OpenAI Responses and Anthropic Messages adapters preserve native IDs and reject unknown response blocks. External documents remain data messages.

Anthropic compilation rejects an empty message list and, for the Claude 4.6+ model IDs it lists as rejecting prefill, a final assistant turn. OpenAI explicit markers are content-block fields for supported modern profiles. Anthropic markers are native `cache_control` fields with provider-specific lookback. Provider profiles are explicit and versioned; unknown model IDs require a caller-supplied profile.

## Cache and routing

`PrefixCache.peek` is read-only. An entry is live while `now < expires_at`; `touch` after actual reuse refreshes it. Simulated engines write at breakpoints past the hit whose cumulative tokens reach `min_cacheable_tokens`. `write`, `touch` and `prune` advance one store-wide event clock and reject earlier times; `peek` rejects a time before that clock, because later events may already have refreshed or pruned the state it asks about. TTLs are validated. Closed providers report `cache_state="unknown"` preflight. Costs split warm reads, cache creation (including write multiplier), and uncached input. When the provider places its own breakpoint (OpenAI implicit mode), the estimate assumes it may write the whole prompt.

For each segment prefix, compilation renders the native input without cache markers or generation controls. Its digest names the cached prefix. The cumulative token count is the counter applied to RFC 8785 JSON of that same input, omitting empty top-level values; empty input counts zero. Segment counts are successive differences of those cumulative counts. Counters must be deterministic and monotonic on native prefixes. Regrouping identical rendered history into different PCF segments therefore preserves total counts and cache accounting. Simulated billing uses these exact units; closed-provider counts remain estimates. Compiler cache keys include `accounting="native-input:1"`, and the simulated character counter hashes its configuration.

`compat_key` is computation grouping only; engine name/version and block size are outside it. Opaque provider descriptors carry placeholder `layout` values because providers do not publish them. `byte_compat_key` exists only for complete manifest descriptors and still requires external attestation/transport policy. No raw KV exchange is implemented.

Routing has exactly one fallback. A candidate whose API rejects the request shape is left out of the decision; if that candidate is the fallback, routing raises. Confidence is nullable and eligible only after versioned held-out validation for the same candidate, rubric, threshold, and scorer fingerprint. Validation requires unique context prefix hashes disjoint from fitting data, a tail slice (head calibration is not evidence of tail calibration), calibration-error bounds on the whole set and on the tail, and at least `min_selected_samples` examples at or above the threshold in both the whole set and the tail. Empty selected-tail evidence fails even if aggregate calibration is perfect. Repeated rows, including contexts differing only in segment IDs, stability or session IDs, cannot count toward the sample minimums. Validation records include all sample minimums in their identity. Otherwise routing falls back with an explicit quality-unknown reason. The current gate has no per-request tail classifier, so a candidate lacking selected-tail evidence remains globally ineligible.

The Jev HTTP transport requires HTTPS and rejects all redirects, including same-origin redirects. It never forwards credentials to a redirect target. A redirect is a confidence transport failure and activates the router's fallback policy.

## Validation and compatibility

Numbers are finite and booleans are not numbers. Schemas are strict and runtime constructors enforce core constraints. Synthetic tests demonstrate invariants; they do not establish provider cache hit rates or production model quality.

PCF 0.2 changes segment hash domains, authority fields, neutral tool history, schemas, and route-decision nullability. Reserialize 0.1 inputs before use; do not mix hash versions in one cache namespace.
