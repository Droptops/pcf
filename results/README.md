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
