# Changelog

## Unreleased

Fixes from coding reviews. Portable segment hashes remain unchanged. Document acceptance changes: tool results require `is_error`, memory may follow history (tail memory), and 0.1-style `tool_use` blocks are rejected, so a 0.2.0 reader rejects some documents this version writes (see SPEC "Validation and compatibility"). Compiler cache keys, simulated billing counts, candidate fingerprints and calibration record identities change as described below.

- Jev's HTTPS transport rejects all redirects so bearer credentials cannot follow a different origin or an HTTP downgrade. Redirects trigger confidence fallback.
- Held-out validation rejects repeated context hashes and requires enough selected tail examples even when no tail score reaches the threshold. All sample minimums are included in validation record identity. Existing scorers with only below-threshold tail evidence now fall back.
- Token accounting uses cumulative canonical native input, matching cache identity. Splitting or combining history segments cannot change billing for the same native prefix. The accounting version in compiler cache keys invalidates older metadata and candidate validations. The character-chunk tokenizer's identity now includes its validated configuration.
- Added regression coverage for redirects, evidence minimums, duplicate samples, regrouped history, and tokenizer identity; CI covers Python 3.11–3.14, lint, offline smoke checks, wheel installation, and session-link prevention in tracked text and new commits.

- Breakpoints: memory segments sharing a `provenance` form a module with its own anchor, so editing a later
  module keeps earlier ones warm. Memory without provenance keeps the shared anchor; hashes are unchanged.
- Ordering: `memory` may follow history when only `user` segments come after it (tail memory), so editing
  volatile memory re-bills only the tail instead of the whole conversation.
- `pcf.placement.MemoryPlacer`: each memory module picks front or tail placement from its own observed change
  rate (tail when p·(m+H) > m); modules are stable until a change is seen. Optional; the compiler does not use it.
  The tail is sticky while the cache is warm (moving back re-bills m+H); with `split(..., cold=True)` modules are
  re-placed from a decayed change rate (`decay=0.7`), so a module that went quiet returns to the front.
- Jev via OpenRouter: `openrouter_transport()` posts the unchanged Jev body to OpenRouter's Decisions API
  (`/api/alpha/decisions`) with `OPENROUTER_API_KEY`. Pin the dated snapshot (`model="typesafe/jev-1.13-20260917"`): replies
  name it, so the `typesafe/jev-1.13` alias fails the pin check and the router falls back.
- OpenAI: assistant history is sent as plain text (Responses rejects `input_text` on assistant turns) and carries
  no cache marker; explicit mode is sent only when a marker was placed (explicit with none disables caching);
  parallel `function_call` items normalize into one assistant turn; unsupported boundaries no longer consume the
  breakpoint budget.
- OpenAI history caching (checked live on gpt-5.6, explicit mode): a history segment ending in assistant text is
  marked on its last user/tool item, since a marker on assistant `output_text` is accepted but writes nothing.
  Reads only hit markers present in the request, so earlier history endpoints stay marked.
- OpenAI write estimates stop at the history marker: `ContextCompiler.covered_tokens()` reports how much of the
  prefix a marker caches, and the OpenAI adapter excludes assistant turns after a history segment's last user/tool
  item, so `warmth()` and router cost no longer count them as written.
- `Router(..., score_unvalidated=False)`: candidates without a passing validation record are no longer scored by
  default (a paid confidence call that cannot change the decision); their report carries `score=None`. Pass
  `score_unvalidated=True` to collect scores anyway. SPEC documents closed-provider router costs as cold
  estimates, the breakpoint and OpenAI marker rules, `covered_tokens`, and the compatibility change.
- `CLAUDE.md` records the repository rules for agents: no private session links or session trailers, run the CI
  checks before pushing, paid live runs only on request, raw results under `results/`.
- `scripts/live_memory_placement.py` grades strictly (the reply's lead value must equal the expected value; the old
  substring check passed copied templates whose order numbers contained the count), records output tokens (and
  OpenAI reasoning tokens / Anthropic thinking blocks), prices output with `--output-multiplier`, flags format
  violations (copied history filler or overlong replies), records the compiler's per-turn token and write
  estimates, and writes run metadata (date, commit, settings). New: `--history-style template|varied`, a
  `placed-spacer` arm, `--thinking`, `--effort`, `--workers`, and `--regrade FILE` for offline re-grading. Offline
  runs of both providers are covered by the test suite. Raw results behind the README live under `results/`.
- `scripts/live_memory_placement.py --provider anthropic` sends the Anthropic adapter's Messages request unchanged to
  OpenRouter's Anthropic-compatible `/api/v1/messages`, pinned to Anthropic upstream (no fallbacks); prompt caching
  was checked live through it (6,012 tokens written, then read).
- `MemoryPlacer.for_candidate(candidate)` prices placement from a router `Candidate` (its tokenizer, write and
  cache-read prices). README documents memory placement with the measured live results.
- `MemoryPlacer(write_multiplier=, read_multiplier=)`: the tail rule weighs cache prices,
  p·(w−r)·(m+H) > (1−r)·m. Defaults (w=1, r=0) keep the plain token rule.
- `scripts/live_memory_placement.py` compares three arms (front, tail, placed) over four modules with different
  change rates, repeats runs, and scores stale-history traps separately.
- Breakpoints over budget keep the last anchor, plus the last anchor of the leading tools/system run when the
  compiler's `history_slots` endpoints still fit beside both (Anthropic and sim: 2 of 4). OpenAI needs 3 history
  endpoints (reads only hit markers present in the request, and a tool round appends two markable segments), so
  it keeps the last anchor and 3 endpoints and does not protect the system prompt from a volatile module left
  `stable`; mark such modules `stable=False` or place them after history. This replaces an interim rule that kept
  the first anchor, which on Anthropic protected `tools` instead of `system` and on OpenAI made per-message tool
  loops re-bill all history every request.
- `scripts/live_tool_loop.py`: a scripted tool loop ([user, call] and [tool] segments, four tail memory modules)
  that checks each request reads back the previous request's cached prefix. Live on 2026-09-25 every request
  after the first read back 100% on gpt-5.6 and on claude-sonnet-5 via OpenRouter (`results/2026-09-25/`).
- Provider reasoning state in history: assistant turns may carry `provider_blocks` (Anthropic thinking and
  redacted_thinking blocks, OpenAI reasoning items), which `history_from_response` now keeps instead of rejecting.
  The matching adapter replays them unchanged before the turn's text and calls (`AnthropicCompiler(thinking_blocks=)`,
  `OpenAICompiler(reasoning_items=)`, "replay" or "drop"); other adapters never send them. `AnthropicCompiler(thinking=)`
  sets the request's thinking configuration. Checked live 2026-09-25 on claude-sonnet-5 (via OpenRouter) and gpt-5.6:
  a tool round replays the captured block and completes, in both modes. Documents without the field hash as before.
- Tail memory (memory after history) never takes a breakpoint, whatever its `stable` flag: an entry there would be
  rewritten every turn and never read, and several tail modules could crowd history out of the budget.
- `MemoryPlacer.split`: memory with instruction authority always stays in front, and tail copies keep their
  authority. A change to a front module is priced against everything behind it (later front modules plus
  history), not history alone; list modules stable-first.
- OpenAI `covered_tokens()` counts the covered prefix in native units by rendering the history segment cut after its
  last user/tool item (the chars/4 trim of neutral JSON was up to 38% high for escape-heavy tool calls).
- Anthropic profiles: `claude-fable-5`, `claude-mythos-5`, `claude-mythos-5-1` (512) and `claude-mythos-preview`
  (2048) gain minimum cacheable lengths; `claude-mythos-5` rejects prefill. Every prefill-rejecting model now has a
  cache profile.
- `scripts/live_memory_placement.py`: low reasoning effort and 512 output tokens (64 returned empty answers), and a
  `billed_input_units` summary under the write multiplier and an assumed `--read-multiplier`.
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
