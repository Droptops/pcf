# Grok long-session test report

Date: 2026-09-25

## Result

The long-session experiment was run offline through the repository's native OpenAI and Anthropic compilers and
`SimEngine`. It covered all six synthetic domain scenarios, 168 sessions, 12,240 requests, and 180 synthetic
fleet requests. No provider API call was made because this environment has no OpenAI, OpenRouter, or Anthropic
credentials.

The simulation supports the direction of Grok's claim: cache placement saves more as a warm conversation grows,
and the benefit collapses when every turn arrives after the cache TTL. It does **not** validate Grok's numerical
projection, because the same simulation overpredicts the savings in the already-recorded 24-turn live runs.

## Calibration against recorded live runs

The existing live runs are the check on the simulator. The percentage is input cost reduction for MemoryPlacer
against tuned front placement.

| Provider | Recorded live saving | Offline model saving |
| --- | ---: | ---: |
| GPT-5.6 | 13.90% | 20.84% |
| Claude Sonnet 5, five-minute profile | 12.27% | 26.48% |

The model therefore overpredicts known savings by 6.94 and 14.21 percentage points. The 60- and 120-turn
percentages below are model projections, not measured provider billing.

## Warm-session projection

| Profile | 24 turns | 60 turns | 120 turns |
| --- | ---: | ---: | ---: |
| GPT-5.6, 30-minute TTL | 20.84% | 56.72% | 72.77% |
| Claude Sonnet 5, five-minute TTL | 26.48% | 60.46% | 75.10% |
| Claude Sonnet 5, one-hour TTL | 38.84% | 70.72% | 82.74% |

The one-hour profile uses a 2x cache-write price, so its percentages are relative to its own more expensive
baseline. They should not be read as proof that one-hour retention is cheaper for every workload.

## Expiry sensitivity at 60 turns

| Expiry schedule | GPT-5.6 | Claude, five-minute TTL |
| --- | ---: | ---: |
| No expiry | 56.72% | 60.46% |
| Every 10th turn | 45.22% | 48.56% |
| Every fourth turn | 30.44% | 33.04% |
| Every second turn | 15.60% | 16.83% |
| Every follow-up turn | 1.02% | 0.92% |

These are deterministic schedules using virtual time, not random production miss rates. A cold turn still has
layout-dependent writes, so the final row is close to zero rather than exactly zero.

## Shared-prefix control

Twenty sequential first requests used a common synthetic reference and distinct user records.

| Profile | User-specific system text | Isolated namespaces | Shared prefix and namespace |
| --- | ---: | ---: | ---: |
| GPT-5.6 | 92,608 | 92,543 | 12,638 |
| Claude, five-minute TTL | 92,078 | 92,005 | 12,274 |
| Claude, one-hour TTL | 146,858 | 146,740 | 15,011 |

The shared-prefix result is a positive control for the local cache model. It does not prove provider-side routing,
cross-workspace reuse, or concurrent cold-start behavior.

## Reproduce

```bash
python scripts/offline_scaling_validation.py \
  --output results/2026-09-25/offline-scaling-validation.json.gz
python -m pytest -q
python -m ruff check .
```

The full experiment output is in
[`results/2026-09-25/offline-scaling-validation.json.gz`](../results/2026-09-25/offline-scaling-validation.json.gz),
with a compact extract in
[`results/2026-09-25/offline-scaling-summary.json`](../results/2026-09-25/offline-scaling-summary.json).
The longer methodology and the live-run instructions are in
[`SCALING_VALIDATION.md`](SCALING_VALIDATION.md).

## Verdict

Grok's mechanism is supported: warm, long sessions and shared byte-identical prefixes are where placement can
matter. Its exact scaling percentages remain a hypothesis until the same 24/60/120-turn schedules, idle gaps,
and fleet-prefix controls are run against provider usage with credentials and real production prompts.
