# Live results

Raw output of `scripts/live_memory_placement.py --run`, one JSON file per run, with the settings in `meta`.
Every number quoted in the top-level README comes from a file here. Re-grade any file offline with
`python scripts/live_memory_placement.py --regrade FILE`.

| File | Provider / model | Turns × repeats | Notes |
|---|---|---|---|
| `2026-09-24/openai-gpt-5.6-20turns-x3.json` | OpenAI gpt-5.6, explicit cache mode | 20 × 3 | output tokens not recorded |
| `2026-09-24/openai-gpt-5.6-60turns.json` | OpenAI gpt-5.6, explicit cache mode | 60 × 1 | output tokens not recorded |
| `2026-09-24/anthropic-claude-sonnet-5-20turns.json` | claude-sonnet-5 via OpenRouter, pinned to Anthropic | 20 × 1 | output tokens not recorded; thinking omitted |

The 2026-09-24 runs predate output-token recording, and ran back to back with no idle time, so no cache entry
expired between turns.

2026-09-25 runs (on the fixed breakpoint rules; output tokens recorded):

| File | Provider / model | Turns × repeats | Notes |
|---|---|---|---|
| `2026-09-25/e1-openai-template.json` | OpenAI gpt-5.6 | 20 × 5, arms front / tail / placed / placed-spacer | template history |
| `2026-09-25/e1-openai-varied.json` | OpenAI gpt-5.6 | 20 × 5, same arms | varied history |
| `2026-09-25/e1-anthropic-template.json` | claude-sonnet-5 via OpenRouter, pinned to Anthropic | 20 × 5, same arms | template history; thinking omitted (adaptive) |
| `2026-09-25/e1-anthropic-varied.json` | claude-sonnet-5 via OpenRouter, pinned to Anthropic | 20 × 5, same arms | varied history; thinking omitted (adaptive) |
| `2026-09-25/tool-loop-openai.json` | OpenAI gpt-5.6 | 8 tool rounds | `scripts/live_tool_loop.py` |
| `2026-09-25/tool-loop-anthropic.json` | claude-sonnet-5 via OpenRouter | 8 tool rounds | thinking disabled |

The E1 runs used a 512-token output limit (now 4096) and were re-graded with the answer-set grader; each file's
`meta.note` records how many grades changed.
