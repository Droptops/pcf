# PCF 0.2

PCF is a reference implementation for portable context structure, provider-native prompt compilation, cache accounting, and conservative model routing.

## Status

This is an offline-tested prototype. The simulated engine can know cache warmth before execution. OpenAI and Anthropic adapters report provider warmth as `unknown` until a real response supplies usage. `compat_key` groups computation identity; it does not authorize KV-byte transfer, and no cross-family KV transport is implemented.

Calibration is a versioned, held-out empirical gate. A fitted scorer is not automatically eligible to route. Validation requires unique contexts disjoint from fitting data, a tail slice, error bounds, and quality/sample thresholds on both all selected examples and selected tail examples. Zero selected tail examples does not qualify a candidate. These are operational checks, not a guarantee of future quality.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
ruff check .
python examples/demo.py
python scripts/live_smoke.py       # offline request-shape check
python scripts/live_smoke.py --run # paid calls only when keys are set
```

Live scripts need the provider SDKs: `python -m pip install -e '.[live]'`. `scripts/live_memory_placement.py` is
offline by default; `--run --provider openai` reads `OPENAI_API_KEY`, and `--run --provider anthropic` reads
`OPENROUTER_API_KEY` and sends the Anthropic adapter's request to OpenRouter's Anthropic-compatible endpoint,
pinned to Anthropic upstream.

Schemas live under `pcf.schemas`. Hashes use RFC 8785 JCS and tagged SHA-256 domains. PCF 0.2 is a breaking wire/hash change from 0.1; see `CHANGELOG.md`.

External documents remain data, tool IDs/arguments are preserved, cache inspection is read-only, costs include write premiums, and unknown provider models require an explicit profile.

Cache identity and cumulative token counts use the same rendered input. Simulated billing counts canonical native JSON; provider billing remains an estimate until usage is returned. The revised accounting changes compiler cache keys and candidate fingerprints: old cache metadata and validation records must not be reused.

## Memory placement

Memory that changes between turns should follow history (tail memory), not precede it: a change to front memory re-bills every history token after it. Tail memory never takes a breakpoint. Front memory that changes should be marked `stable=False`, or its marker is rewritten every turn (and on OpenAI, whose budget keeps three history endpoints, it can take the system prompt's slot). `MemoryPlacer` decides per module: stable modules stay in front (cached), a module moves to the tail once p·(w−r)·(m+H) > (1−r)·m, with prices from `MemoryPlacer.for_candidate(candidate)`. H counts history and the front modules after it, so list modules stable-first; memory with instruction authority never moves. Pass `split(..., cold=True)` when the provider cache has expired, predicted from time (`now - last_request >= descriptor.ttl_seconds`), not from a zero cache read: by the time usage shows a miss, that request has already rewritten the cache in the old layout.

Measured live with `scripts/live_memory_placement.py --run`: one scripted support session per cell, four memory
modules changing never / every 6th / every 3rd / every turn, 20 turns × 5 repeats (raw data in `results/`). Costs
are in uncached-input-token units: cache writes 1.25×, reads 0.1× (assumed; `--read-multiplier`), output 5×
(`--output-multiplier`). "Template" history repeats one long reply pattern every turn; "varied" uses short replies.
Correct counts every answer, traps the turns where history holds a stale value; violations are replies that copy
the history's filler or run long.

| gpt-5.6 | Input | Input + output | Correct (traps) | Violations |
|---|---|---|---|---|
| Front, template | 119.9k | 124.3k | 100/100 (55/55) | 0 |
| Tail, template | 30.7k | 37.8k | 100/100 (55/55) | 0 |
| `MemoryPlacer`, template | 28.4k | 33.2k | 100/100 (55/55) | 0 |
| Front, varied | 66.5k | 72.0k | 94/100 (49/55) | 32 |
| Tail, varied | 17.7k | 24.6k | 100/100 (55/55) | 0 |
| `MemoryPlacer`, varied | 16.4k | 22.9k | 100/100 (55/55) | 0 |

| claude-sonnet-5 (OpenRouter, pinned to Anthropic) | Input | Input + output | Correct (traps) | Violations |
|---|---|---|---|---|
| Front, template | 115.9k | 126.4k | 100/100 (55/55) | 17 |
| Tail, template | 38.7k | 43.3k | 100/100 (55/55) | 10 |
| `MemoryPlacer`, template | 36.3k | 54.6k | 100/100 (55/55) | 46 |
| `MemoryPlacer` + 200-token spacer, template | 43.3k | 48.3k | 100/100 (55/55) | 9 |
| Front, varied | 34.4k | 46.5k | 86/100 (42/55) | 50 |
| Tail, varied | 24.7k | 26.9k | 100/100 (55/55) | 16 |
| `MemoryPlacer`, varied | 19.1k | 26.4k | 98/100 (53/55) | 41 |

What this shows, for this scripted workload:

- Memory after history cuts input cost against the front layout: 3.0-3.9x with template history, 1.4-3.8x with
  the shorter varied history, and more as history grows (a 60-turn gpt-5.6 run: 640k front, 121k tail, 112k
  placed; `results/2026-09-24/`).
- It also answers better. With varied history, front memory missed 6 (gpt-5.6) and 13 (Claude) of 55 stale-history
  traps; tail memory missed none.
- On gpt-5.6, `MemoryPlacer` saves a further 7% of input over all-tail (7-12% counting output). On Claude it saves
  6-22% of input, but Claude copies the history's reply pattern more when little sits between the last reply and
  the question, and thinks more (35-79 thinking blocks per 100 turns against 18 for all-tail). Counting output,
  all-tail is cheapest with template history (43.3k against 54.6k) and the two are level with varied history
  (26.9k, 26.4k). A ~200-token neutral spacer after tail memory removes most of the copying but costs more input
  than it saves. All-tail is the safer default for Claude; `MemoryPlacer` for gpt-5.6.
- The front rows keep every module `stable`, as a naive layout would. On gpt-5.6 the breakpoint budget then spends
  no marker on the system prompt (see Breakpoints in SPEC), which is why front costs more than in the 2026-09-24
  runs (80.2k).
- The input advantage depends on the read price: at reads of 0.1x-0.5x, tail costs 0.26-0.52 of front (gpt-5.6,
  template), and the placer's input edge over all-tail falls from 7% to 2%. Runs were back to back, so no cache
  entry expired between turns; with idle gaps past the TTL the placer's edge shrinks further. The E1 runs capped
  output at 512 tokens, which ended 2 placed Claude turns inside thinking (now 4096). Questions are simple lookups
  graded by value, so this is a cost and regression check, not a general quality evaluation.

The Jev transport requires a direct HTTPS endpoint and rejects redirects. CI runs offline tests, lint, examples, a wheel installation check, and a private-session-link check on tracked text and new commit messages. The metadata check does not remove links from existing Git history or GitHub PR descriptions.

See `spec/SPEC.md` and `spec/REMEDIATION.md`.

License: MIT.
