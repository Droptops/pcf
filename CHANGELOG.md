# Changelog

## Unreleased

- The domain grader reads negative numbers and amounts (`-3`, `-$605.63`, `$-605.63`), which the scenarios produce
  only after turn 24; 24-turn results are unchanged. Add live 60-turn runs on both providers.
- Calibration error and Brier score use correctly rounded summation (`math.fsum`), so metrics and the validation
  IDs hashed from them no longer differ between Python 3.11 and 3.12+. Records computed on 3.11 can change in the
  last digits and get new validation IDs; regenerate them.
- Treat invalid raw probabilities from a fitted Platt scorer as per-request confidence unavailability, allowing
  routing to fall back. Scorer programming errors and invalid calibration coefficients remain visible.
- Replace the Jev acceptance-v2 filler/length proxy with question-specific full-answer formats (acceptance-v3),
  regenerate labels and held-out validation offline, and retain both earlier result files. The gate still fails.
- Pool compatible domain result files before finalizing summaries, preserving repeated scenarios and paired
  counts. Reject incompatible settings, duplicate file paths and unpaired turns; expose pooled totals and cost
  denominators explicitly.
- Align placement advice and pilot effect sizes with the tuned-baseline results and clarify the synthetic
  evidence, cost denominators and value-only accuracy boundary.
- Portable hashes, wire format and compiler cache identity are unchanged. Calibration dataset IDs now include
  the labeling version; regenerate evaluation records that used the superseded acceptance labels.

## 0.3.0a1 (2026-09-25)

### Compatibility

Portable segment hashes are unchanged, and documents without the new fields hash as before. Document acceptance
changes: tool results require `is_error` (which the segment hash covers), memory may follow history (tail memory), assistant turns may carry
`provider_blocks`, and 0.1-style `tool_use` blocks are rejected, so a 0.2.0 reader rejects some documents this
version writes (see SPEC "Validation and compatibility"). Compiler cache keys (a new accounting version), simulated
billing counts, candidate fingerprints and calibration record identities change: regenerate old cache metadata and
validation records.

### Added

- Tail memory: `memory` may follow history when only `user` segments come after it, so editing volatile memory
  re-bills only the tail instead of the whole conversation. Tail memory never takes a breakpoint, whatever its
  `stable` flag.
- Memory modules: memory segments sharing a `provenance` form a module with its own anchor, so editing a later
  module keeps earlier ones warm. Memory without provenance keeps the shared anchor.
- `pcf.placement.MemoryPlacer`: each memory module picks front or tail placement from its observed change rate and
  cache prices, moving to the tail when p·(w−r)·(m+H) > (1−r)·m (defaults w=1, r=0 give p·(m+H) > m). H counts
  history and the later front modules; list modules stable-first. Modules are stable until a change is seen; the
  tail is sticky while the cache is warm, and `split(..., cold=True)` re-places modules from a decayed change rate
  (`decay=0.7`). Memory with instruction authority always stays in front. `MemoryPlacer.for_candidate(candidate)`
  takes the tokenizer and prices from a router `Candidate`. Optional; the compiler does not use it.
- Provider reasoning state in history: assistant turns may carry `provider_blocks` (Anthropic thinking and
  redacted_thinking blocks, OpenAI reasoning items), which `history_from_response` keeps. The matching adapter
  replays them before the turn's text and calls, or drops them (`AnthropicCompiler(thinking_blocks=)`,
  `OpenAICompiler(reasoning_items=)`); other adapters, Jev and the simulator never see them. Cache markers are
  rejected at any depth, as are non-string block types and empty `provider_blocks` lists (runtime and schema agree). `AnthropicCompiler(thinking=)` sets the request's thinking configuration. Checked live on
  claude-sonnet-5 and gpt-5.6: a tool round replays the captured block and completes, in both modes.
- `ContextCompiler.covered_tokens()` reports how much of the prefix a marker caches; `warmth()` and router cost use
  it. The OpenAI adapter counts the covered prefix in native units, excluding assistant turns after a history
  segment's last user or tool item.
- `Router(..., score_unvalidated=False)`: candidates without a passing validation record are no longer scored by
  default (a paid confidence call that cannot change the decision); their report carries `score=None`.
- `ConfidenceSource.context_key`: held-out uniqueness is keyed on what the confidence source scores, so contexts
  differing only in reasoning state are not independent evidence.
- Jev via OpenRouter: `openrouter_transport()` posts the unchanged Jev body to OpenRouter's Decisions API with
  `OPENROUTER_API_KEY`. Pin the dated snapshot (`model="typesafe/jev-1.13-20260917"`); undated aliases fail the pin
  check and the router falls back.
- `ScaledTokenizer(base, factor)` (in `pcf.families.anthropic_adapter`): a deterministic fixed-factor correction of
  a token counter with its own identity. In the 2026-09-25 placement runs (`results/2026-09-25/e1-*.json`, billed
  input over estimate per run), chars/4 undercounts claude-sonnet-5 by 1.21-1.42x and overcounts gpt-5.6 at
  0.86-0.98x. Defaults are unchanged.
- Anthropic profiles: `claude-fable-5`, `claude-mythos-5`, `claude-mythos-5-1` (512) and `claude-mythos-preview`
  (2048) gain minimum cacheable lengths; `claude-mythos-5` rejects prefill.

### Changed

- Breakpoints over budget keep the last anchor, then the last anchor of the leading tools/system run when the
  history endpoints (3 on OpenAI, 2 on Anthropic and the simulator, never more than the history available) still
  fit beside both anchors, then those history endpoints, then the newest remaining candidates. A history boundary
  that leaves a tool call waiting for its result takes no slot, unsupported boundaries use no budget, stable user
  turns after history remain candidates, and a budget of 2 no longer returns every candidate. OpenAI tool loops stay flat instead of re-billing all
  history, and Anthropic protects `system`, not `tools`. On OpenAI, once a request carries 3 or more history
  candidates, the system prompt is not protected from a volatile module left `stable`, and a stable user segment
  after history is not marked: mark such modules `stable=False` or place them after history, and send the latest
  user turn as the last history segment to cache it.
- OpenAI: assistant history is sent as plain text and carries no cache marker; a history segment ending in
  assistant text is marked on its last user or tool item (a marker on assistant `output_text` is accepted but
  writes nothing). Explicit mode is sent only when a marker was placed. Parallel `function_call` items normalize
  into one assistant turn. Implicit mode estimates the provider's own breakpoint as a write of the whole prompt, so
  route costs include the write premium; caller-supplied implicit-only profiles default to free writes (1x).
- Token accounting uses cumulative canonical native input, matching cache identity, so splitting or combining
  history segments cannot change billing for the same native prefix. The character-chunk tokenizer's identity
  includes its configuration. Empty `tools`/`history` segments bill zero tokens.
- Schemas reject what the runtime rejects: empty or whitespace-only strings (whitespace as Python's `str.strip()`
  defines it, spelled out so ECMA-262 validators agree), empty history turns, duplicate axes,
  `byte_compat_key` without a complete manifest, and trailing newlines under `$` anchors. Serialized documents may
  no longer carry the 0.1 assistant `tool_use` history form (constructors still migrate it).
- Anthropic compilation rejects an empty message list and, on the Claude 4.6+ model IDs it lists, a final assistant
  turn (prefill), raising `UnsupportedRequest`; the router leaves such a candidate out, and raises only when it is
  the fallback.
- `PrefixCache` keeps one store-wide event clock: `write`/`touch`/`prune` reject earlier times and `peek` rejects a
  time before the latest event, so warmth can no longer report a hit before the entry existed.

### Fixed

- Router: a validated confidence source that returns NaN, infinity, an out-of-range or non-numeric score for one
  request no longer aborts routing; the candidate is treated as confidence unavailable for that request
  (`confidence_error` records why) and the router picks another candidate or the fallback.
- OpenAI `history_from_response` rejects text, or a new call turn, that arrives before earlier calls' outputs,
  which it previously accepted as history `Context` then rejected. Every accepted conversion of up to five output
  items (and of Anthropic content blocks) is tested to form valid history once pending calls are resolved.
- Simulated caching follows each marker's actual prefix: `CompiledPrompt.marker_prefixes` records the native
  hash and token count a marker covers (an OpenAI history marker stops before the segment's trailing assistant
  items), and `SimEngine` and `warmth()` read and write that prefix. Previously a simulator run with an OpenAI
  compiler wrote and matched the whole segment. `covered_tokens()` now comes from `marker_prefix()`.
- Compilation no longer re-renders and re-encodes every prefix: the built-in compilers render all prefixes in one
  pass (`render_prefixes`), and their canonical JSON and hashes reuse the shared prefix. Hashes, token counts and
  requests are unchanged (tested against rendering every prefix from scratch); 400 history segments compile in
  about 0.1 s instead of about 4 s. Other compilers keep the exact per-prefix path.

- Jev's HTTPS transport rejects all redirects, so bearer credentials cannot follow a different origin or an HTTP
  downgrade; redirects trigger confidence fallback. Malformed or truncated Jev responses fall back instead of
  raising.
- Held-out validation rejects repeated contexts, fails when no sample reaches the threshold, and requires enough
  selected tail examples even when no tail score reaches the threshold; all sample minimums are part of the
  validation record identity. Existing scorers with only below-threshold tail evidence now fall back.
- `fit_platt` stops on the Newton decrement instead of stalling; `number()` raises `ValueError` for huge ints.
- Response normalization rejects what it cannot represent instead of dropping it: Anthropic text after `tool_use`
  or with citations, thinking after `tool_use`, OpenAI `output_text` annotations and non-text
  `function_call_output` parts. Mixed 0.1/0.2 tool results that would flip `is_error` are rejected.

### Scripts and results

- `scripts/live_memory_placement.py`: live comparison of memory layouts (front, tail, `MemoryPlacer`, and
  `MemoryPlacer` with a spacer) with value grading, output-token pricing, format-violation flags, repeated runs,
  and offline `--regrade`. `--provider anthropic` sends the Anthropic adapter's request unchanged to OpenRouter's
  Anthropic-compatible endpoint, pinned to Anthropic upstream.
- `scripts/live_tool_loop.py` checks that each tool-loop request reads back the previous request's cached prefix.
- `scripts/live_question_bank.py` asks paired conflict, cross-module and far-memory questions under each layout,
  with an exact McNemar test.
- `scripts/live_jev_calibration.py` validates Jev for one candidate on harness contexts and retests score noise.
  Rows carry `value_correct`, `instruction_compliant` and `label` (both, `acceptance-v2`), matching the rubric Jev
  is asked about; `--relabel FILE` relabels a saved question-bank run and re-validates from its recorded scores.
- `scripts/live_domain_sessions.py` runs synthetic healthcare, government and enterprise assistant sessions
  (`scripts/domain_scenarios.py`) under front, tuned-front, tail and `MemoryPlacer` layouts; `--analyze` pairs
  layouts per turn with an exact McNemar test, `--regrade` re-grades offline, and a failed session no longer loses
  the completed ones.
- Raw results of every run quoted in the README are under `results/`; README summarizes them.
- CI covers Python 3.11-3.14, lint, offline smoke checks, wheel installation, and a private-session-link check on
  tracked text, new commits and the pull request title and description (re-run when the description is edited).

## 0.2.0

- Added immutable snapshots, RFC 8785 canonicalization, authority/provenance, and strict schemas.
- Preserved structured tool-call/result history across provider adapters.
- Corrected provider markers/profiles, TTL semantics, read-only inspection, and cache-write pricing.
- Added conservative descriptor identity and held-out calibration gates.
- Breaking: hashes, schemas, route decision fields, and history wire format changed from 0.1.
