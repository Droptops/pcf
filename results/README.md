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
cache entry expired between turns. The E1 runs used a 512-token output limit (now 4096) and the breakpoint rule at
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
| `2026-09-25/jev-calibration-haiku-4-5.json` | same | 240 placement contexts, 117 distinct | superseded: session tags made repeats look unique (see `meta.note`) |
| `2026-09-25/jev-input-limit-probe.json` | Jev via OpenRouter | 10 probes | the input limit behind "about 32.8k tokens" |
