# PCF: Portable Context Format

[![CI](https://github.com/Droptops/pcf/actions/workflows/ci.yml/badge.svg)](https://github.com/Droptops/pcf/actions/workflows/ci.yml)

LLM providers bill a repeated prompt prefix at a steep discount through prompt caching, but only while the start of
the prompt stays identical from one request to the next. Most assistants put memory (a customer record, preferences,
account state) near the top of the prompt, so every memory update invalidates the cache for the whole conversation
after it, and the provider bills that history again on every turn.

PCF describes a conversation once, in a provider-neutral format, and compiles it into native OpenAI or Anthropic
requests with cache markers placed deliberately. It accounts for what each request will cost, including cache
writes and reads, decides where each memory module should sit, and routes to a cheaper model only when a
confidence scorer has passed a held-out calibration gate. It is a Python reference implementation and an
offline-tested prototype.

## Headline result

Measured live on gpt-5.6 and claude-sonnet-5 (raw data in [`results/`](results/)):

- **Memory placed after the conversation history cuts input cost 1.4-3.9x** against the usual layout with memory in
  front, and the gap grows with conversation length (5.3x at 60 turns on gpt-5.6).
- **It also answers better.** With memory in front and varied history, models sometimes repeated an older value
  stated earlier in the conversation: 6 of 55 such questions on gpt-5.6 and 12 of 55 on Claude. With memory after
  the history, none.
  In long sessions (60-100 turns), Claude recalled facts 42/50 times with memory in front and 50/50 after the
  history (exact McNemar p = 0.008).
- **Multi-step tool loops reuse 100% of the previous request's cache** on both providers.

Recommended default: on Claude, put all changing memory after the history; on gpt-5.6, let `MemoryPlacer` decide
per module (a further 7% input saving). For small models, see the caveat under [Limits](#limits-of-the-evidence).

## How it works

```
Memory in front:        [system][memory][history ......................][question]
                                   ^ a change here re-bills everything to its right
Memory after history:   [system][history ......................][memory][question]
                                                           a change here re-bills only memory ^
```

A provider caches a prompt from its start up to the first byte that changed. Moving memory that changes to just
before the question means an update re-bills only the memory itself, and the model sees the current value next to
the question instead of an older value repeated in the history.

PCF covers the pieces this needs:

1. **A portable document** (`pcf.Context`): ordered `tools`, `system`, `memory`/`document`, `history` and `user`
   segments with content hashes, authority (instruction or data) and provenance. JSON schemas define the wire
   format.
2. **Provider compilers** (`pcf.families`): render the same document as an OpenAI Responses or Anthropic Messages
   request, choosing which segment boundaries get the provider's limited cache markers.
3. **Cache accounting** (`pcf.cache`, `pcf.compiler`): predicted writes, reads and cost per request, reconciled
   with the usage the provider returns.
4. **Memory placement** (`pcf.placement.MemoryPlacer`): keeps rarely changing modules in front, where they stay
   cached, and moves frequently changing ones after the history.
5. **Conservative routing** (`pcf.router`): routes to a cheaper model only when a confidence scorer has passed a
   held-out calibration gate; otherwise it falls back to the default model.

## Evidence

One scripted support session per cell, 20 turns × 5 repeats. Four memory modules change at different rates (never,
every 6th turn, every 3rd, every turn). Costs are in uncached-input-token units: cache writes 1.25×, cache reads
0.1× (an assumed price; `--read-multiplier`), output 5× (`--output-multiplier`).

Terms used below:

- **Front**: all memory before the history. **Tail**: all memory after the history. **`MemoryPlacer`**: chosen
  per module.
- **Repetitive history** repeats one long reply pattern every turn; **varied history** uses short replies.
- **Stale-history checks**: turns where an earlier reply in the history states a value that memory has since
  changed.
- **Violations**: replies that copy the history's filler or run long.

| gpt-5.6 | Input | Input + output | Correct (stale-history checks) | Violations |
|---|---|---|---|---|
| Front, repetitive | 119.9k | 124.3k | 100/100 (55/55) | 0 |
| Tail, repetitive | 30.7k | 37.8k | 100/100 (55/55) | 0 |
| `MemoryPlacer`, repetitive | 28.4k | 33.2k | 100/100 (55/55) | 0 |
| Front, varied | 66.5k | 72.0k | 94/100 (49/55) | 32 |
| Tail, varied | 17.7k | 24.6k | 100/100 (55/55) | 0 |
| `MemoryPlacer`, varied | 16.4k | 22.9k | 100/100 (55/55) | 0 |

| claude-sonnet-5 (via OpenRouter, pinned to Anthropic) | Input | Input + output | Correct (stale-history checks) | Violations |
|---|---|---|---|---|
| Front, repetitive | 115.9k | 126.4k | 100/100 (55/55) | 17 |
| Tail, repetitive | 38.7k | 43.3k | 100/100 (55/55) | 10 |
| `MemoryPlacer`, repetitive | 36.3k | 54.6k | 100/100 (55/55) | 46 |
| `MemoryPlacer` + spacer, repetitive | 43.3k | 48.3k | 100/100 (55/55) | 9 |
| Front, varied | 34.4k | 46.5k | 87/100 (43/55) | 50 |
| Tail, varied | 24.7k | 26.9k | 100/100 (55/55) | 16 |
| `MemoryPlacer`, varied | 19.1k | 26.4k | 98/100 (53/55) | 41 |

Takeaways:

- **Cost.** Tail memory cuts input cost 3.0-3.9x with repetitive history and 1.4-3.8x with varied history. On a
  60-turn gpt-5.6 session: 640k front, 121k tail, 112k `MemoryPlacer` (`results/2026-09-24/`).
- **Accuracy.** With varied history, front memory missed 6 (gpt-5.6) and 12 (Claude) of 55 stale-history checks;
  tail memory missed none.
- **Which layout.** On gpt-5.6, `MemoryPlacer` is cheapest. On Claude, all-tail is cheapest or level once output is
  counted: with little between the last reply and the question, Claude copies the history's reply pattern more and
  thinks more.

A paired question bank (`scripts/live_question_bank.py`, 50 distinct items per type, each asked under every layout)
points the same way:

| Question type | gpt-5.6 front / tail / `MemoryPlacer` | claude-sonnet-5 front / tail / `MemoryPlacer` |
|---|---|---|
| Far memory (fact from memory at the end of 60-100 turns) | 48 / 50 / 50 | 42 / 50 / 50 (p = 0.008) |
| Cross-module (combines two memory modules) | 50 / 50 / 50 | 48 / 50 / 50 |
| Conflict (customer asked earlier to change a value memory still holds) | 50 / 50 / 50 | 50 / 50 / 50 |

Every far-memory miss repeated a value stated earlier in the history. Both cross-module misses gave wrong ticket
counts, one also the wrong plan. On conflict items every layout followed the customer's request: memory next to the
question did not override the conversation.

<details>
<summary>Details and caveats</summary>

- A neutral spacer after tail memory (about 280 estimated tokens, 354 billed on Claude) removes most of the reply
  copying: with repetitive history it lowers `MemoryPlacer` cost from 54.6k to 48.3k, still above all-tail (43.3k),
  and with varied history it raises cost. `MemoryPlacer` produced 35-79 thinking blocks per 100 Claude turns
  against 18 for all-tail.
- The front rows keep every module `stable`, as a naive layout would. Under the breakpoint rule these runs used
  (commit c95184f), gpt-5.6 then spent no marker on the system prompt, which is why front costs more here than in
  the 2026-09-24 runs (80.2k). The current rule also marks it while a request has fewer than 3 history candidates
  (see Breakpoints in `spec/SPEC.md`).
- The advantage depends on the read price: at reads of 0.1x-0.5x, tail costs 0.26-0.52 of front (gpt-5.6,
  repetitive), and `MemoryPlacer`'s input edge over all-tail falls from 7% to 2%.
- The 2026-09-25 placement runs capped output at 512 tokens, which ended 2 `MemoryPlacer` Claude turns inside
  thinking (the limit is now 4096).

</details>

## Routing safety: the calibration gate

PCF routes a request to a cheaper model only when an external confidence scorer, validated on held-out contexts,
says the cheaper model will answer well. We tested the third-party scorer Jev (`typesafe/jev-1.13-20260917` via
OpenRouter) with `scripts/live_jev_calibration.py`, and the gate correctly refused it.

On 450 distinct question-bank contexts, claude-haiku-4-5 answered 439 (97.6%) correctly. Jev scored the 369
contexts it could score 0.19-0.55 (mean 0.30), and scored the 11 failures 0.28-0.38, no lower than the successes
(AUC 0.33). No context reaches 0.7 and calibration error is about 0.67, so validation fails at every threshold and a
Jev-gated router falls back to the default model. Jev rejects requests above about 32.8k of its input tokens
(`max_tokens_exceeded`; here, compiled prompts above about 18.7k estimated tokens, 81 of 450), which the router
treats as confidence unavailable. Score noise is small (retest SD about 0.011, no decision flips at 0.8). The
default model id `jev-latest` is never eligible for validation; pass a dated snapshot id. The Jev transport requires
a direct HTTPS endpoint and rejects redirects.

## Quick start

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python examples/demo.py            # offline end-to-end demo: compile, account, route
python scripts/live_smoke.py       # offline request-shape check
```

## Reproducing the results

Every number above comes from a JSON file under [`results/`](results/); [`results/README.md`](results/README.md)
maps each file to its run. Re-grade a placement run offline:

```bash
python scripts/live_memory_placement.py --regrade results/2026-09-25/e1-openai-template.json
```

Live runs make paid API calls and need the provider SDKs (`python -m pip install -e '.[live]'`). The live scripts
are offline by default; `--run --provider openai` reads `OPENAI_API_KEY`, and `--run --provider anthropic` reads
`OPENROUTER_API_KEY` and sends the Anthropic adapter's request to OpenRouter's Anthropic-compatible endpoint, pinned
to Anthropic upstream.

## Repository layout

| Path | Contents |
|---|---|
| `src/pcf/` | The library: segments and hashing, compiler, cache, `placement`, `router`, provider `families` |
| `spec/SPEC.md` | The specification: document rules, hashing, breakpoints, accounting, routing, compatibility |
| `src/pcf/schemas/` | JSON schemas for documents, cache descriptors and route decisions (mirrored in `spec/schemas/`) |
| `examples/demo.py` | Offline demo of compilation, cost accounting and routing |
| `scripts/` | Offline checks and opt-in paid live experiments |
| `results/` | Raw JSON behind every published number |
| `tests/` | Offline test suite |

## Using MemoryPlacer

`MemoryPlacer` decides per module. Stable modules stay in front, where they stay cached. A module moves after the
history once p·(w−r)·(m+H) > (1−r)·m, where p is its observed change rate, m its size, H the history and later
front modules behind it, and w and r the cache write and read price multipliers. Get the prices from
`MemoryPlacer.for_candidate(candidate)`. List modules stable-first; memory with instruction authority never moves.
Tail memory never takes a cache marker.

Front memory that changes should be marked `stable=False`, or its marker is rewritten every turn (and on OpenAI,
whose budget keeps three history endpoints, it can take the system prompt's slot). Pass `split(..., cold=True)`
when the provider cache has expired, predicted from time (`now - last_request >= descriptor.ttl_seconds`), not from
a zero cache read: by the time usage shows a miss, that request has already rewritten the cache in the old layout.

## Limits of the evidence

- One scripted support workload with simple lookup questions graded by value: a cost and regression check, not a
  general quality evaluation.
- Cache prices are assumed multipliers, not billed dollars. Runs were back to back, so no cache entry expired
  between turns; idle gaps past the cache lifetime shrink the savings, and `MemoryPlacer`'s edge further.
- Small models: on conflict items, claude-haiku-4-5 followed the customer's request 50/50 with front memory but
  43/50 with `MemoryPlacer` and 49/50 with all-tail (`MemoryPlacer` against front: exact McNemar p = 0.016). Every
  miss gave the stale record's value. Where users change preferences in conversation, update the memory record, or
  keep that module in front, for small models.

## Status and design limits

- OpenAI and Anthropic adapters report cache warmth as `unknown` until a real response supplies usage; only the
  simulated engine knows warmth before execution. Provider billing is an estimate until usage is returned.
- Calibration is a versioned, held-out empirical gate. A fitted scorer is not automatically eligible to route.
  Validation requires unique contexts disjoint from fitting data, a tail slice, error bounds, and quality and sample
  thresholds on both all selected examples and selected tail examples. Zero selected tail examples does not qualify
  a candidate. These are operational checks, not a guarantee of future quality.
- `compat_key` groups computation identity; it does not authorize KV-byte transfer, and no cross-family KV
  transport is implemented.
- External documents remain data, tool IDs and arguments are preserved, cache inspection is read-only, costs
  include cache-write premiums, and unknown provider models require an explicit profile.

## Wire format and compatibility

Hashes use RFC 8785 JCS and tagged SHA-256 domains. PCF 0.2 is a breaking wire and hash change from 0.1. Cache
identity and cumulative token counts use the same rendered input; simulated billing counts canonical native JSON.
Changes to compiler cache keys and candidate fingerprints mean old cache metadata and validation records must not
be reused. See [`CHANGELOG.md`](CHANGELOG.md) for upgrade notes and `spec/SPEC.md` for the exact rules.

## Development

CI runs the offline tests on Python 3.11-3.14, lint, the demo and offline smoke check, and a wheel installation
check. Contributor rules are in `CLAUDE.md`.

License: MIT.
