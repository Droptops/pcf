# PCF: Portable Context Format

[![CI](https://github.com/Droptops/pcf/actions/workflows/ci.yml/badge.svg)](https://github.com/Droptops/pcf/actions/workflows/ci.yml)

LLM providers bill a repeated prompt prefix at a steep discount through prompt caching, but only while the start of
the prompt stays identical from one request to the next. Most assistants put memory (a customer record, preferences,
account state) near the top of the prompt, so every memory update invalidates the cache for the whole conversation
after it, and the provider bills that history again on every turn.

**The durable finding is a prompt convention:** keep stable text first, and put the memory that changes after the
conversation history. Both vendors already advise stable-first; this repository measures what the second half is
worth, and where it is not worth anything. A team can apply it by ordering blocks and setting cache markers in the
provider SDK. What the repository adds:

- **Evidence**, live on gpt-5.6 and claude-sonnet-5, with the raw results and the harnesses that produced them.
- **A small layout helper**, `MemoryPlacer` in Python and [TypeScript](ts/), which decides per module from its change
  rate and the cache prices, and sets cache markers. Its marker logic has had several bugs found only by running the
  providers (see the changelog); treat it as a tested helper, not a guarantee.
- **A cache audit** ([`scripts/cache_audit.py`](docs/CACHE_AUDIT.md)) that reads an assistant's existing request logs
  and names the fields whose changes cost the most cache.

Two parts are experiments, not products. The portable context format has one writer and no other reader; segment
hashes are not provider cache keys and do not transfer cache state. The model router has one empirical result: the
only scorer tested ranked rejected answers above accepted ones (AUC 0.328), so the gate has never admitted a
cheaper model and always falls back.

## Headline result

Measured live on gpt-5.6 and claude-sonnet-5 on synthetic sessions, with assumed cache prices (raw data in
[`results/`](results/)). The baseline is the best front layout, with the changing modules marked uncacheable, not
the naive one:

| Placed memory against the tuned front layout | gpt-5.6 | claude-sonnet-5 |
| --- | ---: | ---: |
| Input cost, 24 turns (six scenarios) | 0.86 | 0.88 |
| Input cost, 60 turns (six scenarios) | **0.46** | **0.49** |
| Input cost, 60 turns, prefix shared across users | **0.53** | **0.50** |
| Correct at 60 turns, tuned front → placed | 696 → 718 of 720 | 650 → 719 of 720 |
| Median latency at 60 turns, tuned front → placed | 2.00 s → 1.91 s | 6.01 s → 3.73 s |
| **Echo-all** at 60 turns: input cost, correct | 0.47, 719 of 720 | 0.51, 719 of 720 |

- **The 60-turn rows are from the current library** (`domain-*-60turn-after-fixes.json`), after two cache-marker
  fixes found by the fleet test and the cache audit; the run before them gave 0.49 and 0.53, with 703 and 652 correct
  for tuned front and 720 and 719 for placed. Two of placed's three misses answered a negative balance as "a credit
  of $145.47", which the grader counts wrong.
- **A simpler layout does as well.** *Echo-all* leaves the memory in front unchanged all session (the cached prefix
  never changes) and repeats every module that changes just before the question. In the same 60-turn runs it matched
  `MemoryPlacer` within noise on cost and accuracy (placed 0.46 and 0.49, 717 and 718 correct). The accuracy gain over
  tuned front is recency: the current value next to the question. Echoing only the field the question needs costs
  less (0.36 and 0.38) but needs to know that field, and on Claude the stale values left in front drew replies about
  three times longer that flagged the inconsistency. See [the echo baseline](#the-echo-baseline).
- **When it does not pay.** Turns that arrive after the cache lifetime (5 minutes on Claude's default) read nothing:
  in a Claude run with every turn past it, no turn read the cache, so casework with pauses between turns keeps
  little of the 60-turn figure. At 24 turns the input saving is 12-14%. Structured outputs or tool calls will not
  reproduce the shorter replies seen on Claude, so quote input cost, not total. Tail memory that changes between
  turns conflicts with replaying extended-thinking blocks bound to the earlier prefix. Sessions that share a prefix
  were tested with synchronized users only.
- **Front layouts gave out-of-date values.** Most of their wrong answers repeated an older value instead of the
  current record; placed memory sits next to the question. The scripted replies state old values in the history on
  purpose, so this measures recency against a planted stale value, not answer quality in general. The exception is a small model with a record
  that was not updated: see [Limits](#limits-of-the-evidence).
- **Against the naive front layout** (every module cacheable), the domain workloads cost 3.7-4.1x more than placed
  memory at 24 turns (input plus output). That comparison flatters the method; the tuned numbers above are the ones to quote.
- **Multi-step tool loops reuse 100% of the previous request's cache** on both providers.

Recommended starting point: keep large stable reference modules in front and let `MemoryPlacer` place changing
modules. All-tail worked well on the smaller support workload below, but was substantially more expensive on the
domain workloads with large stable references, including Claude. Compare layouts on your workload. For small
models, see the caveat under [Limits](#limits-of-the-evidence).

## The echo baseline

A reviewer asked for the arm the experiments lacked: leave the cached prefix alone and repeat the current values next
to the question. `scripts/live_domain_sessions.py` has two such arms. `echo` keeps the memory as it was on turn 0 in
front and repeats the current version of the module the question asks about, marked current; `echo-all` repeats
every module that changes. Six scenarios, 60 turns, 2 repeats, both providers, in the same runs as tuned front and
`MemoryPlacer` (`results/2026-09-25/domain-*-60turn-echo.json`):

| Against tuned front, 60 turns | gpt-5.6 input | total | correct | claude-sonnet-5 input | total | correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tuned front | 1 | 1 | 703/720 | 1 | 1 | 652/720 |
| `MemoryPlacer` | 0.456 | 0.484 | 717/720 | 0.486 | 0.405 | 718/720 |
| echo (the asked module) | 0.356 | 0.399 | 720/720 | 0.383 | 0.568 | 715/720 |
| echo-all (every changing module) | 0.465 | 0.498 | 719/720 | 0.508 | 0.439 | 719/720 |

Paired turn by turn, neither echo arm differs from `MemoryPlacer` in accuracy (discordant pairs 3/0, 2/0, 2/5 and
2/1; exact McNemar p ≥ 0.25). Both beat tuned front (p < 0.001). The conclusion: put the current values next to the
question. Moving them there and repeating them there cost about the same; repeating only what is asked costs less
when the application knows it. Leaving stale copies in front has a cost of its own on Claude: with only the asked
module current, Claude often noticed the contradiction ("I need to stop here and flag an issue with this session"),
answered at about three times the length (277 against 96 output tokens), and 4 of its 5 wrong answers quoted the
frozen record. Echo-all, with every changing value current, showed none of this.

## With the model's own replies

The runs above replay scripted assistant turns, some of which state old values on purpose. The run below feeds each
layout its own visible replies instead (`--history-mode model-text`), and adds `fixed-tail`: every declared changing
module after the history from turn zero. Six scenarios, 60 turns, 2 repeats, back to back
(`results/2026-09-26/domain-*-60turn-model-text.json`):

| Against tuned front, 60 turns | gpt-5.6 input | total | correct | claude-sonnet-5 input | total | correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tuned front | 1 | 1 | 702/720 | 1 | 1 | 667/720 |
| `MemoryPlacer` | 0.472 | 0.477 | 720/720 | 0.196 | 0.253 | 715/720 |
| echo-all | 0.483 | 0.496 | 720/720 | 0.171 | 0.199 | 703/720 |
| fixed-tail | 0.458 | 0.471 | 710/720 | 0.165 | 0.194 | 715/720 |

- **Each arm's replies make its own history, so history length differs by arm.** On Claude, tuned front answered at
  302 output tokens per turn against 107-174 for the others; its prompt reached 19.2k tokens at turn 60 against
  11.9-15.9k, and only 44% of its input was read from cache against 93%. Claude's 0.17-0.20 therefore combines layout
  and reply length and is not comparable with the scripted 0.49. Billed input per token sent, which removes history
  length, is 0.22-0.23 of tuned front for all three arms. On gpt-5.6 replies are short in every arm (17-32 tokens) and
  the ratios, 0.46-0.48, match the scripted runs.
- **Placing the current values after the history holds with real replies.** All three arms beat tuned front in
  accuracy on both models (paired discordant turns 0/18 on gpt-5.6 and 1/49 on Claude for `MemoryPlacer`).
- **The simple arms were not free.** Echo-all on Claude was less accurate than `MemoryPlacer` (5/17 discordant
  pairs, exact McNemar p = 0.017): 14 of its 17 misses gave a stale value, and in 11 of 14 turns after a wrong reply
  it repeated that error. Fixed-tail on gpt-5.6 missed 10 turns `MemoryPlacer` answered (0/10, p = 0.002), 4 of them
  stale. Fixed-tail matched `MemoryPlacer` on Claude (5/5), and echo-all matched it on gpt-5.6 (0/0).
- Cost is still in assumed cache multipliers, and the sessions had no idle gaps.

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
0.1× (an assumed price; `--read-multiplier`), output 5× (`--output-multiplier`). Cost columns are per-session
means over repetitions; correctness and violation counts pool all repetitions.

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

## Synthetic domain workloads

`scripts/live_domain_sessions.py` runs six synthetic assistant sessions closer to production prompts
(`scripts/domain_scenarios.py`; every record is invented): an inpatient nurse copilot and health plan member services
(healthcare), benefits casework and taxpayer assistance (government), and an IT service desk and sales CRM copilot
(enterprise). Each prompt carries a policy, a large stable reference module (formulary, benefit grid, program rules,
knowledge base) and four records changing never / every 6th turn / every 3rd / every turn: about 4-6k tokens before
history, 24 turns per session, 3 repeats. Four layouts: memory in front with every module `stable` (naive), memory in
front with the changing modules marked `stable=False` (tuned), all memory after history (tail), and `MemoryPlacer`.

Cost columns below sum the six per-scenario means over three repetitions (144 turns per repetition); correctness
and stale-history counts pool all three repetitions (432 answers per layout). These are assumed-price token
units, not invoice dollars. The analyzer also reports `totals_all_repeats` with cost and counts on the same
pooled denominator.

claude-sonnet-5 (`results/2026-09-25/domain-claude-sonnet-5.json`):

| Layout | Input | Input + output | Correct | Stale-history checks |
|---|---|---|---|---|
| Front, naive | 882.5k | 1,097.8k | 409/432 | 215/234 |
| Front, tuned | 250.7k | 463.2k | 408/432 | 213/234 |
| Tail | 842.1k | 916.5k | 432/432 | 234/234 |
| `MemoryPlacer` | 219.9k | 293.9k | 432/432 | 234/234 |

gpt-5.6 (`results/2026-09-25/domain-gpt-5.6.json`), same sessions:

| Layout | Input | Input + output | Correct | Stale-history checks | Latency p50 / p90 |
|---|---|---|---|---|---|
| Front, naive | 715.1k | 741.6k | 422/432 | 226/234 | 1.74 / 3.23 s |
| Front, tuned | 186.6k | 213.8k | 424/432 | 229/234 | 1.85 / 3.13 s |
| Tail | 592.8k | 620.1k | 432/432 | 234/234 | 1.62 / 2.98 s |
| `MemoryPlacer` | 160.7k | 182.7k | 432/432 | 234/234 | 1.65 / 3.02 s |

- **Cost.** `MemoryPlacer` was cheapest overall on both models: 3.7x (Claude) and 4.1x (gpt-5.6) below the naive
  front layout and 1.6x and 1.2x below the tuned one, counting output. It was cheapest in every scenario on Claude and
  in five of six on gpt-5.6 (clinical: 1% above tuned front).
- **Accuracy.** It answered every question on both models. These are extracted-value scores, not full instruction-compliance or safety grades. Paired by turn, it was right where tuned front memory was
  wrong 24 times on Claude and 8 on gpt-5.6, never the reverse (exact McNemar p < 1e-6 and p = 0.008). Tuning cache
  markers reduced front memory's cost without removing its value errors. The turn-level McNemar values are
  descriptive for these scripted sessions; broader inference needs conversation/scenario-level uncertainty analysis.
- **All-tail is the wrong lesson.** It was as accurate but cost 3.1-3.4x more than `MemoryPlacer`: with a large
  stable reference module, moving it after history bills it uncached every turn. Keep stable memory in front and
  move only what changes.
- **Latency.** On gpt-5.6 the layouts were within 0.2 s at the median. On Claude, a full replication with latency
  recording (`domain-claude-sonnet-5-replication.json`) gave the same result: `MemoryPlacer` 294.2k and 432/432
  (first run 293.9k), against 1,102.0k and 408/432 naive and 474.2k and 416/432 tuned, and a median response time of
  3.2 s against 5.4-5.5 s for either front layout, which produced about three times as many output tokens per turn.
- `domain-gpt-5.6-run1.json` is an earlier gpt-5.6 run whose answers are confounded (a third asked for identity
  verification before the scenarios recorded it); its input costs match this run's.

## Routing safety: the calibration gate (experimental; has never routed)

PCF routes a request to a cheaper model only when an external confidence scorer, validated on held-out contexts,
says the cheaper model will answer well. We tested the third-party scorer Jev (`typesafe/jev-1.13-20260917` via
OpenRouter) with `scripts/live_jev_calibration.py`, and the gate correctly refused it.

Jev is asked whether an answer is acceptable: correct, following the application instructions and satisfying the
request. On 450 distinct question-bank contexts, claude-haiku-4-5 gave the right value 439 times (97.6%), but only
145 answers (32.2%) met both value and question-specific answer-format requirements. `acceptance-v3` matches
whole answers: one word for contact/language, digits for ticket counts, a known plan name for plan lookups, or
`<number>, <plan>` for cross-module questions. Normalization allows outer whitespace, case differences, one
terminal period and one enclosing bold pair; extra prose is rejected. This is an operational rubric for these
lookup questions, not a general quality judge.

The saved answers and scores were relabeled offline with `--relabel`. Across the 369 scorable contexts, Jev scored
0.19-0.55 and scored rejected answers higher on average (0.31 versus 0.28; AUC 0.328). ECE is about 0.176 and no
score reaches 0.7, so every tested validation threshold still fails and the router falls back. Raw observations
are preserved. The earlier value-only report (AUC 0.33, ECE about 0.67) and acceptance-v2 report (AUC 0.38, ECE
about 0.32) are archived; v2's filler/length heuristic still accepted answers that violated explicit formats.
Jev rejects requests above about 32.8k of its input tokens
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
`PCF_ANTHROPIC_API_KEY` (or `ANTHROPIC_API_KEY`) and sends the Anthropic adapter's request to Anthropic's API.
Claude results published up to 2026-09-26 went through OpenRouter, pinned to Anthropic upstream; `--anthropic-route
openrouter` (with `OPENROUTER_API_KEY`) reproduces them. OpenRouter is otherwise used only for the Jev router.

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
| `docs/PILOT.md` | Protocol for testing PCF in one production assistant |

## Using MemoryPlacer

`MemoryPlacer` decides per module. Stable modules stay in front, where they stay cached. A module moves after the
history once p·(w−r)·(m+H) > (1−r)·m, where p is its observed change rate, m its size, H the history and later
front modules behind it, and w and r are the cache write and read price multipliers. Use
`MemoryPlacer.for_compiler(compiler)` or `MemoryPlacer.for_candidate(candidate)` so prices and the provider's
minimum cacheable length stay together. List modules stable-first; memory with instruction authority never moves.
Tail memory never takes a cache marker.

Front memory that changes should be marked `stable=False`, or its marker is rewritten every turn (and on OpenAI,
whose budget keeps three history endpoints, it can take the system prompt's slot). Pass `split(..., cold=True)`
when the provider cache has expired, predicted from time (`now - last_request >= descriptor.ttl_seconds`), not from
a zero cache read: by the time usage shows a miss, that request has already rewritten the cache in the old layout.

`MemoryPlacer` is stateful. In production, give every logical turn a durable id, restore the conversation's state,
and save the new state with a compare-and-set on its revision:

```python
placer = MemoryPlacer.for_compiler(compiler, expected_turns=40)
if saved_state is not None:
    placer.restore_state(saved_state)
revision = placer.revision
front, tail = placer.split(memory, history, turn_id=request_id, expected_revision=revision)
state_store.compare_and_set(conversation_id, revision, placer.export_state())
```

A retry with the same `turn_id` and inputs returns the recorded decision without updating the change-rate estimate.
Reusing the id with different inputs is rejected. Two workers planning new turns from the same revision cannot both
commit: one receives `ConcurrentPlacementUpdate`; the state store's compare-and-set provides the same guarantee
across processes. State documents are versioned and contain module hashes and decisions, not module contents.

## TypeScript

[`ts/`](ts/) is a dependency-free TypeScript port of `MemoryPlacer` with request layouts for Anthropic Messages and
OpenAI chat, for applications that keep their own message list. Its decisions match the Python placer turn for
turn on the domain scenarios and on random sessions near the decision threshold.

## Auditing an existing assistant

`scripts/cache_audit.py` reads the requests an assistant already sends, with the usage each response returned, and
names the fields whose changes cost the most cache: for example, on the tuned front layout at 60 turns,
`memory 'claims' amount_owed` re-billed the history behind it on 80 requests. It needs no PCF types. On PCF's own
logs it found the first-request anchor bug and a full prefix rewrite on OpenAI whenever `MemoryPlacer` moved a
module, both now fixed; see [`docs/CACHE_AUDIT.md`](docs/CACHE_AUDIT.md).

## Limits of the evidence

- Scripted support and six synthetic domain workloads with lookup questions graded by value. Histories use
  predetermined replies rather than each model's actual prior answer, so these runs do not test error propagation
  through real conversations. Value correctness does not imply instruction compliance or domain safety.
- Cache prices are assumed multipliers, not billed dollars. Runs were back to back, so no cache entry expired
  between turns; idle gaps past the cache lifetime shrink the savings, and `MemoryPlacer`'s edge further.
- Small models: on conflict items, claude-haiku-4-5 followed the customer's request 50/50 with front memory but
  43/50 with `MemoryPlacer` and 49/50 with all-tail (`MemoryPlacer` against front: exact McNemar p = 0.016). Every
  miss gave the stale record's value. Where users change preferences in conversation, update the memory record, or
  keep that module in front, for small models.

## Next step: a production pilot

The evaluation kit now supports `--history-mode model-text`, which feeds each layout's actual visible replies
into its own subsequent requests, and a `fixed-tail` baseline that places declared volatile records after history
from turn zero. One live run uses both; see [With the model's own replies](#with-the-models-own-replies). `--capture-requests` records outbound
SDK arguments and timestamps for auditing. The [four-arm pilot](docs/PILOT.md) compares tuned front, echo-all,
fixed-tail and MemoryPlacer with conversation-level assignment and descriptive bootstrap comparisons.

The audit separates usage-derived cost from heuristic miss attribution, checks log integrity and cache scopes,
and can score independently annotated logs with `--validate-labels`. No production-log validation or realized
savings claim has been added; see the [validation workflow](docs/CACHE_AUDIT.md#production-log-contract-and-validation-workflow).

The evidence here is scripted or synthetic. The last synthetic test, [`docs/FLEET_CACHE.md`](docs/FLEET_CACHE.md),
passed all its pre-registered rules on both models: at 60 turns with a prefix shared across sessions, placement
billed 0.53 (gpt-5.6) and 0.50 (Claude) of the tuned layout's input, a shared prefix cut a conversation's first
request to under a fifth, and turns after the cache lifetime read nothing. [`docs/PILOT.md`](docs/PILOT.md) is the
next test, on one real assistant, measuring billed cost, latency and answer quality; `scripts/pilot_analysis.py`
assigns conversations and analyzes the usage log.

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
