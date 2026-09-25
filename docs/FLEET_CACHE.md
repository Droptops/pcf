# Fleet cache test: shared prefix, long sessions, idle gaps

The domain runs (`results/2026-09-25/domain-*.json`) compare layouts inside one session. Each session puts a unique
nonce in the system segment, turns run back to back, and sessions stop at 24 turns. That measures the layout rule.
It does not measure the two things that decide whether the rule is a large cut in a real bill: a prefix reused
across sessions, and a cache that is still there on the next turn.

This protocol measures both, on the same synthetic scenarios, before anyone spends a production pilot on it.
`docs/PILOT.md` stays the test for a real assistant. History summarization and model routing are out of scope here.

## Question

On the synthetic domain scenarios, with system text and the reference module byte-identical across sessions:

1. Does `MemoryPlacer` cut billed input to 0.60 or less of a tuned front layout at 60 turns, while the cache stays warm?
2. Does session 2 onward read that shared prefix on the first turn, instead of writing it again?
3. Does a gap longer than the cache lifetime remove that cut?

## Arms

Same content in every arm. Only order, cache markers, prefix identity, and the gap between turns change.

- **Tuned, private.** Reference and records before history. Changing modules marked `stable=False`. A unique session
  nonce is prepended to the system segment, as `scripts/live_domain_sessions.py` does today.
- **Tuned, shared.** The same layout. System text and the reference module are byte-identical across sessions of
  this arm. The nonce, the time, and the arm name stay out of every segment before the per-session records.
- **Placed, shared.** `MemoryPlacer` with the same shared prefix. One placer per session, starting from a stable
  prior. The fleet shares the prefix bytes, not the placer's rate estimates.
- **Placed, shared, cold.** The placed shared arm, with a sleep of `ttl_seconds + 30` between turns.

All four arms share one `cache_namespace` per arm for the whole fleet. The OpenAI adapter derives
`prompt_cache_key` from that namespace (`OpenAICompiler._partition`). A fresh namespace per session isolates the
fleet even when the bytes match. Anthropic hits on prefix bytes and breakpoint placement; a nonce in the system
segment is enough to miss.

Profiles and lifetimes already in the adapters: gpt-5.6 explicit cache, retention 30m (`ttl_seconds` 1800, write
1.25×); claude-sonnet-5 `ttl="5m"` (300s, write 1.25×). A Claude `ttl="1h"` cell is optional and is not part of
the pass rule: its write multiplier is 2×, so report it separately.

## Workload

Reuse `scripts/domain_scenarios.py`. Pre-register two scenarios and run only those in the paid claim:

- `benefits`, a short volatile tail.
- `clinical`, the fat tail. At 24 turns on gpt-5.6 this was the cell where placed input sat 6% above tuned front.

Turns: 24 and 60. The 24-turn cell is a bridge to the committed domain totals, not the claim. History in those
scenarios grows about 31 tokens per turn on gpt-5.6 and about 35 on Claude; do not lengthen the scripted replies
in this run. A longer-reply cell is a separate factor and would move the result for a reason this protocol does
not claim.

Sessions: 1 warmup + 8 measured, sequential, per arm per scenario per model. Score only the measured 8. The
warmup exists so a shared-prefix miss on session 1 is not counted as a fleet failure.

Grading stays the domain grader (first asserted value, stale-history checks). Output limits stay at 4096.

## Order of work

1. **Offline.** Drive the four arms through `SimEngine` with one `PrefixCache` per arm and a clock. Warm arms
   advance the clock by 1 second per turn. The cold arm jumps `ttl_seconds + 30`. Apply the pass rules below to
   simulated usage before any paid call.
2. **Paid probe.** `benefits` only, 2 sessions × 8 turns, both models, tuned-private and placed-shared, no sleep.
   Session 2 turn 0 must read the shared prefix (rule 2, on this smaller sample). Stop if it does not. Fix
   markers or the cache key before the claim run.
3. **Paid claim.** 24- and 60-turn warm cells, both scenarios, both models, 1 + 8 sessions. Then the cold cell:
   Claude 5m only, `benefits`, 60 turns, 1 + 4 sessions. OpenAI's 30m lifetime makes a full cold cell a
   wall-clock problem; one 60-turn OpenAI cold session is enough to see a miss, and it is not part of the
   three-rule pass.

## Metrics

Primary: billed input per measured session, in the same units as the domain runs
(`uncached + 1.25 * written + 0.1 * cached`). The claim ratio is the mean of placed-shared over the mean of
tuned-shared, at 60 turns, pooled across the two scenarios, separately per model.

Secondary, reported and not folded into the ratio:

- Turn-0 cached tokens on measured sessions, divided by the estimated token count of system + reference.
- Output tokens per session. The domain runs showed a large Claude output drop; this protocol does not require
  it to recur.
- Correct and stale-history counts.
- Latency to full response, p50 and p90.
- On the cold arm, the fraction of turns with `cached == 0`.

## Pass and fail

Set these before reading the paid JSON.

1. **Long session.** Warm 60-turn placed-shared billed input ≤ 0.60 × warm 60-turn tuned-shared, on each model.
   Warm 60-turn sessions with private prefixes measured 0.49 (gpt-5.6) and 0.53 (Claude)
   (`results/2026-09-25/domain-*-60turn.json`); the offline model in `docs/SCALING_VALIDATION.md` predicts 0.43 and
   0.40 and overpredicts. 0.60 leaves room for the one-time rewrite when a module moves to the tail.
2. **Fleet.** On both shared arms, the median turn-0 read share of system + reference, over measured sessions, is
   ≥ 0.80. On tuned-private it is < 0.20. A pass of rule 1 with a fail of rule 2 means the layout saving is real
   and the fleet saving is not.
3. **Gap.** Claude cold placed-shared billed input > 0.85 × Claude warm tuned-shared, on `benefits` at 60 turns.
   A cold arm that still saves means the request is not using the lifetime we think it is.

The 24-turn placed/tuned input ratio should land near the committed means (0.86 on gpt-5.6, 0.88 on Claude,
six-scenario pool). A 24-turn cell far from that range stops the run: the harness is not the one already
measured. `clinical` may sit above 1.0 at 24 turns and still pass rule 1 at 60; that is expected if the late-turn
slope is about 3 units per turn against about 40.

## What a pass is worth

A pass of all three says the measured 60-turn result holds with a shared prefix: about half the
tuned input bill at 60 warm turns of this length, a shared prefix the fleet actually hits, and no saving once the
gap exceeds the lifetime. A fail of rule 2, with rule 1 passing, limits the claim to a single long conversation.
A fail of rule 3 means the cold model is wrong and `split(..., cold=True)` is being aimed at the wrong signal.

## Recording

One JSON file per paid run under `results/`, with `meta` (date, git sha, model, ttl, `ttl_seconds`, arms,
scenarios, turns, sessions, sleep, write and read multipliers) and per-turn `cached`, `written`, `uncached`,
`output_tokens`, and `latency_s`. Quote the file from `results/README.md` only after the run exists. Raw results
of a run that is quoted in docs are committed, per `CLAUDE.md`.
