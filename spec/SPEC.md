# PCF 0.2 specification

## Document

Documents contain ordered `tools`, `system`, `memory`/`document`, `history`, and `user` segments. IDs are unique. Instructions precede data and are never silently reordered. `authority` is explicit: tools/system require `instruction`; history/user require `data`. `stable` is a breakpoint hint only.

Segment hashes use `sha256("pcf:segment:0.2\\0" || RFC8785(content record))`; IDs and stability are excluded. Context prefix chains use domain-separated raw digest bytes. Caller-owned objects are snapshotted.

## History and providers

Neutral history has user/assistant turns, assistant `tool_calls` (`id`, `name`, object `arguments`), and tool results (`call_id`, content, `is_error`). Calls must be resolved before compilation. OpenAI Responses and Anthropic Messages adapters preserve native IDs and reject unknown response blocks. External documents remain data messages.

OpenAI explicit markers are content-block fields for supported modern profiles. Anthropic markers are native `cache_control` fields with provider-specific lookback. Provider profiles are explicit and versioned; unknown model IDs require a caller-supplied profile.

## Cache and routing

`PrefixCache.peek` is read-only. `touch` and `write` validate monotonic event time and TTL. Closed providers report `cache_state="unknown"` preflight. Costs split warm reads, cache creation (including write multiplier), and uncached input.

`compat_key` is computation grouping only. `byte_compat_key` exists only for complete manifest descriptors and still requires external attestation/transport policy. No raw KV exchange is implemented.

Routing has exactly one fallback. Confidence is nullable and eligible only after versioned held-out validation for the same candidate, rubric, threshold, and scorer fingerprint. Otherwise routing falls back with an explicit quality-unknown reason.

## Validation and compatibility

Numbers are finite and booleans are not numbers. Schemas are strict and runtime constructors enforce core constraints. Synthetic tests demonstrate invariants; they do not establish provider cache hit rates or production model quality.

PCF 0.2 changes segment hash domains, authority fields, neutral tool history, schemas, and route-decision nullability. Reserialize 0.1 inputs before use; do not mix hash versions in one cache namespace.
