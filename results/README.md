# Live results

Raw output of the live scripts (`scripts/live_memory_placement.py`, `scripts/live_tool_loop.py`,
`scripts/live_question_bank.py`, `scripts/live_jev_calibration.py`), one JSON file per run, with the settings in
`meta`. Every number quoted in the top-level README comes from a file here. Re-grade a placement file offline with
`python scripts/live_memory_placement.py --regrade FILE`.

## Memory placement (`live_memory_placement.py`)

| File | Provider / model | Turns × repeats | Notes |
|---|---|---|---|
| `2026-09-24/openai-gpt-5.6-20turns-x3.json` | OpenAI gpt-5.6, explicit cache mode | 20 × 3 | output tokens not recorded |
| `2026-09-24/openai-gpt-5.6-60turns.json` | OpenAI gpt-5.6, explicit cache mode | 60 × 1 | output tokens not recorded |
| `2026-09-24/anthropic-claude-sonnet-5-20turns.json` | claude-sonnet-5 via OpenRouter, pinned to Anthropic | 20 × 1 | output tokens not recorded; thinking omitted |
| `2026-09-25/e1-openai-template.json` | OpenAI gpt-5.6 | 20 × 5, arms front / tail / placed / placed-spacer | template history |
| `2026-09-25/e1-openai-varied.json` | OpenAI gpt-5.6 | 20 × 5, same arms | varied history |
| `2026-09-25/e1-anthropic-template.json` | claude-sonnet-5 via OpenRouter, pinned to Anthropic | 20 × 5, same arms | template history; thinking omitted (adaptive) |
| `2026-09-25/e1-anthropic-varied.json` | claude-sonnet-5 via OpenRouter, pinned to Anthropic | 20 × 5, same arms | varied history; thinking omitted (adaptive) |

The 2026-09-24 runs predate output-token recording. All placement runs ran back to back with no idle time, so no
cache entry expired between turns. The 2026-09-25 placement runs (`e1-*`) used a 512-token output limit (now 4096) and the breakpoint rule at
commit c95184f; all placement files are re-graded with the current grader.

## Tool loops (`live_tool_loop.py`)

| File | Provider / model | Requests | Notes |
|---|---|---|---|
| `2026-09-25/tool-loop-openai.json` | OpenAI gpt-5.6 | 8 tool rounds, 16 requests | reuse and coverage checks pass |
| `2026-09-25/tool-loop-anthropic.json` | claude-sonnet-5 via OpenRouter | 8 tool rounds, 16 requests | thinking disabled; checks pass |

## Question bank (`live_question_bank.py`)

| File | Provider / model | Items | Notes |
|---|---|---|---|
| `2026-09-25/question-bank-gpt-5.6.json` | OpenAI gpt-5.6 | 150 distinct items × 3 arms | conflict, cross-module, far-memory |
| `2026-09-25/question-bank-claude-sonnet-5.json` | claude-sonnet-5 via OpenRouter | same items | |

Each row records its item (type, final turn, history style). The committed items were selected at commit 4211554
(`meta.git_sha`); `item_specs` later changed to mix history styles for any item count, so a new run picks different
items.

## Jev calibration (`live_jev_calibration.py`)

| File | Scorer / candidate | Contexts | Notes |
|---|---|---|---|
| `2026-09-25/jev-calibration-haiku-4-5-question-bank.json` | Jev `typesafe/jev-1.13-20260917` / claude-haiku-4-5 | 450 distinct (369 scorable) + 60 × 5 retest | conflict items as tail; per-row Jev errors recorded |
| `2026-09-25/jev-calibration-haiku-4-5-question-bank-acceptance-v2.json` | same | same 450 | archived: filler/length compliance proxy; superseded by acceptance-v3 |
| `2026-09-25/jev-calibration-haiku-4-5-question-bank-acceptance-v3.json` | same | same 450 | question-specific full-answer formats; 145 accepted; recorded answers and scores unchanged; offline validation still fails |
| `2026-09-25/jev-calibration-haiku-4-5.json` | same | 240 placement contexts, 117 distinct | superseded: session tags made repeats look unique (see `meta.note`) |
| `2026-09-25/jev-input-limit-probe.json` | Jev via OpenRouter | 10 probes | the input limit behind "about 32.8k tokens" |

## Synthetic domain workloads (`live_domain_sessions.py`)

| File | Provider / model | Sessions | Notes |
|---|---|---|---|
| `2026-09-25/domain-claude-sonnet-5.json` | claude-sonnet-5 via OpenRouter, pinned to Anthropic | 6 scenarios × 4 layouts × 3 repeats × 24 turns | re-graded offline with the leading-value grader (`meta.regrade_changes`) |
| `2026-09-25/domain-gpt-5.6.json` | OpenAI gpt-5.6 | 6 scenarios × 4 layouts × 3 repeats × 24 turns | corrected scenarios; latency and compile time recorded |
| `2026-09-25/domain-claude-sonnet-5-replication.json` | claude-sonnet-5 via OpenRouter | same | replication with latency recorded; claims rerun separately after credits ran out (`meta.note`) |
| `2026-09-25/domain-gpt-5.6-run1.json` | OpenAI gpt-5.6 | same | first run: costs valid, answers confounded by verification requests (see `meta.note`) |
| `2026-09-25/domain-gpt-5.6-60turn.json` | OpenAI gpt-5.6 | 6 scenarios × front-tuned, placed × 2 repeats × 60 turns | warm sessions; re-graded offline after the grader learned negative values (`meta.regrade_changes`) |
| `2026-09-25/domain-claude-sonnet-5-60turn.json` | claude-sonnet-5 via OpenRouter | same | same |
| `2026-09-25/domain-gpt-5.6-60turn-after-fixes.json` | OpenAI gpt-5.6 | same | rerun on the current library, after the first-request anchor and move-turn fixes |
| `2026-09-25/domain-claude-sonnet-5-60turn-after-fixes.json` | claude-sonnet-5 via OpenRouter | same | same; see `meta.note` for the recorded `git_sha` |
| `2026-09-25/domain-gpt-5.6-60turn-echo.json` | OpenAI gpt-5.6 | 6 scenarios × front-tuned, placed, echo, echo-all × 2 repeats × 60 turns | the echo baseline; see `meta.note` |
| `2026-09-25/domain-claude-sonnet-5-60turn-echo.json` | claude-sonnet-5 via OpenRouter | same | same |

Domain table costs in the main README sum per-scenario means over repetitions; accuracy counts cover all
repetitions. To pool the two compatible Claude runs, preserving all six repetitions per scenario:

```bash
python scripts/live_domain_sessions.py --analyze \
  results/2026-09-25/domain-claude-sonnet-5.json \
  results/2026-09-25/domain-claude-sonnet-5-replication.json
```

The output includes explicit cost-basis metadata, repetition counts, latency observation counts, and
`totals_all_repeats` for pooled cost and accuracy on the same denominator. Incompatible model settings or price
multipliers and duplicate input paths are rejected. Do not pool the confounded GPT run with the corrected run.

Reproduce the current calibration labels without paid calls:

```bash
python scripts/live_jev_calibration.py --relabel \
  results/2026-09-25/jev-calibration-haiku-4-5-question-bank.json
```

The v3 artifact records its normalization policy and the relabel script's SHA-256. Relabel timestamps may differ;
answers, Jev scores, labels, metrics and validation records are reproducible from the original file on any supported
Python version (the metrics use correctly rounded summation).

## Offline scaling experiment

`2026-09-25/offline-scaling-validation.json.gz` contains 168 deterministic simulated sessions plus 180 fleet
first requests; `offline-scaling-summary.json` is the compact summary. These are not live provider observations.
See [`docs/SCALING_VALIDATION.md`](../docs/SCALING_VALIDATION.md) for controls, reproduction, the mismatch with
the recorded 24-turn live savings and the live 60-turn check.

## Fleet cache test (`live_fleet_sessions.py`)

| File | Provider / model | Sessions | Notes |
|---|---|---|---|
| `2026-09-25/fleet-probe-openai-before-anchor-fix.json`, `fleet-probe-anthropic-before-anchor-fix.json` | gpt-5.6; claude-sonnet-5 via OpenRouter | `benefits`, tuned-private, tuned-shared, placed-shared, 1 warmup + 1 measured × 8 turns | paid probe: placed-shared read nothing of the shared prefix on the second session's first turn |
| `2026-09-25/fleet-probe-openai.json`, `fleet-probe-anthropic.json` | same | tuned-private, placed-shared | the same probe after the first-request anchor fix: the second session reads the shared prefix |
| `2026-09-25/fleet-gpt-5.6.json`, `fleet-claude-sonnet-5.json` | same | `benefits`, `clinical` × tuned-private, tuned-shared, placed-shared × (1 warmup + 8) × 60 turns | the claim run; `live_fleet_sessions.py --analyze` applies the pass rules |
| `2026-09-25/fleet-cold-claude-sonnet-5.json` | claude-sonnet-5 via OpenRouter | `benefits`, placed-cold × 4 × 12 turns, 330 s gaps | rule 3; analyze together with `fleet-claude-sonnet-5.json` |
| `2026-09-25/move-turn-gpt-5.6.json` | OpenAI gpt-5.6 | 6 scenarios × placed × 12 turns | move-turn marker fix: each move turn reads the reference prefix (`docs/CACHE_AUDIT.md`) |
