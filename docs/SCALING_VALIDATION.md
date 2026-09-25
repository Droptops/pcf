# Offline scaling experiment — September 25, 2026

This report simulates longer sessions, cache expiry and shared prefixes offline, then checks the 60-turn
projection against live runs. The simulations are deterministic and make no provider calls; the live check at
the end does.

**Result:** placement saves more as a warm session grows, and the saving collapses when every turn arrives after
the cache lifetime. The simulator overpredicts measured savings at both 24 and 60 turns, so its 60- and 120-turn
percentages are an upper bound, not a forecast.

## Design

The native OpenAI and Anthropic compilers were executed through the repository's `SimEngine`, using all six
existing synthetic domain scenarios and their scripted short replies. Compare tuned front with MemoryPlacer.
Warm sessions run for 120 turns, with cumulative checkpoints at 24, 60 and 120. Normal requests are 30 virtual
seconds apart. Separate 60-turn sessions expire the cache every 10th, 4th, 2nd or every subsequent turn. Another
schedule inserts a 600-second gap every fourth turn, identically across profiles. The placer receives
`cold=True` when the interval reaches the profile TTL. No miss-rate multiplier substitutes for execution.

Profiles: GPT-5.6 / 30 minutes / 1.25x writes; Claude Sonnet 5 / five minutes / 1.25x writes; Claude Sonnet 5 /
one hour / 2x writes. Reads cost 0.1x in all cases. Costs are canonical-native-JSON heuristic token units.
Summaries sum scenario costs over six sessions per arm. There is one deterministic execution per cell; repeats
would add no independent evidence. Output generation, output cost and real latency are not modeled.

168 sessions (12,240 requests), plus 180 sequential synthetic fleet first requests, were executed offline.

## Calibration against the already-recorded 24-turn live runs

| Profile | Recorded input saving | Offline modeled input saving |
| --- | ---: | ---: |
| GPT-5.6 | 13.90% | 20.84% |
| Claude Sonnet 5, five minutes | 12.27% | 26.48% |

**The model overpredicts the known savings. Its 60/120-turn percentages therefore do not validate the proposed
live scaling curve.** Native JSON token estimates and the simplified local cache are not provider tokenization
or cache behavior. The experiment establishes what this implementation predicts, with the mismatch visible.
The exact source of the mismatch has not been isolated; no scalar correction has been fitted to hide it.

## Warm-session simulation

Input-cost reduction against tuned front within the same provider/TTL profile:

| Profile | 24 turns | 60 turns | 120 turns |
| --- | ---: | ---: | ---: |
| GPT-5.6, 30 minutes | 20.84% | 56.72% | 72.77% |
| Claude Sonnet 5, five minutes | 26.48% | 60.46% | 75.10% |
| Claude Sonnet 5, one hour | 38.84% | 70.72% | 82.74% |

The one-hour percentage uses its own more expensive 2x-write baseline. It does not mean one-hour retention is
cheaper for traffic that is already warm: at 60 warm turns, placed Claude costs about 72,868 modeled units per
session with one-hour retention versus 65,212 with five-minute retention.

## Expiry sensitivity at 60 turns

The schedules are deterministic, not random production miss rates. For example, every-fourth-turn expiry gives
14 long gaps across the 59 follow-up intervals, rather than an exact 25% of all 60 requests.

| Expired gap schedule | GPT-5.6 input saving | Claude five-minute input saving |
| --- | ---: | ---: |
| None | 56.72% | 60.46% |
| Every 10th turn | 45.22% | 48.56% |
| Every fourth turn | 30.44% | 33.04% |
| Every second turn | 15.60% | 16.83% |
| Every follow-up turn | 1.02% | 0.92% |

Every-cold requests have zero reads. Their small remaining cost difference comes from layout-dependent writes
versus uncached input; a cold request is not necessarily billed entirely as a cache write.

For the same 600-second gaps every fourth turn, modeled savings remain 56.72% on GPT's 30-minute profile and
70.72% on Claude's one-hour profile, but fall to 33.04% on Claude's five-minute profile. Comparing the treatment
cost itself on this gapped trace, Claude one-hour placement is 72,868 units per session versus 153,085 with
five-minute placement, approximately 52.4% lower despite the more expensive writes. This is a simulated tradeoff,
not a measured retention recommendation.

## Shared-prefix controls

Twenty sequential first requests use a large common synthetic claims reference and distinct synthetic user
records. Three controls separate prefix text from namespace partitioning: a session ID in the system text;
stable system text but isolated namespaces; and stable text with one authorized shared-workflow namespace.
User records remain outside the shared cached reference. All requests are inside the modeled TTL.

| Profile | Unique system text | Isolated namespaces | Shared prefix and namespace |
| --- | ---: | ---: | ---: |
| GPT-5.6 | 92,608 | 92,543 | 12,638 |
| Claude, five minutes | 92,078 | 92,005 | 12,274 |
| Claude, one hour | 146,858 | 146,740 | 15,011 |

Values are total modeled input units for 20 requests. This is a positive control in an ideal shared store;
it does not prove real provider routing, concurrent cold-start handling, or cross-workspace reuse. A tuned
production baseline that already shares this prefix will already receive that benefit.

## Reproduce

```bash
python scripts/offline_scaling_validation.py \
  --output results/2026-09-25/offline-scaling-validation.json.gz
```

The compressed file contains every simulated request, usage category, input cost, gap and placement. Its metadata
includes the source revision and script hash. `offline-scaling-summary.json` is a compact extract from that run.
The synthetic replies are fixed; no model is asked a question, and no API credential is used by this script.

## Live check at 60 turns

The paid domain harness ran all six scenarios for 60 warm turns, tuned front against placed, 2 repeats per
scenario, on both providers
([`domain-gpt-5.6-60turn.json`](../results/2026-09-25/domain-gpt-5.6-60turn.json),
[`domain-claude-sonnet-5-60turn.json`](../results/2026-09-25/domain-claude-sonnet-5-60turn.json)):

```bash
python scripts/live_domain_sessions.py --run --provider openai --turns 60 --repeats 2 --arms front-tuned placed
python scripts/live_domain_sessions.py --run --provider anthropic --turns 60 --repeats 2 --arms front-tuned placed
```

| Profile | Recorded input saving, 24 turns | Recorded input saving, 60 turns | Offline model, 60 turns |
| --- | ---: | ---: | ---: |
| GPT-5.6 | 13.90% | 50.68% | 56.72% |
| Claude Sonnet 5, five minutes | 12.27% | 46.61% | 60.46% |

Savings are input cost over the sums of per-scenario means, at the same assumed multipliers. The gap between
model and measurement narrows from 6.94 and 14.21 points at 24 turns to 6.04 and 13.85 at 60, so the direction
holds and the model stays optimistic. Placed answered 720/720 (GPT) and 719/720 (Claude) record lookups
correctly, against 703 and 652 for tuned front.

Still unmeasured live: 120 turns, idle gaps past the cache lifetime, and shared prefixes across a fleet. The
last two need timed requests and shared-prefix arms that the live harness does not implement; do not substitute
virtual-time results for them.
