# Offline scaling experiment — September 25, 2026

**Live validation is blocked: the execution environment has no OpenAI, OpenRouter, or Anthropic API credentials.**
This report contains deterministic simulations, not new provider calls, billed dollars, or model quality results.

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

## Live work still required

With runtime credentials configured, the existing paid domain harness can measure warm sessions independently:

```bash
python scripts/live_domain_sessions.py --run --provider openai --model gpt-5.6 \
  --turns 60 --repeats 3 --arms front-tuned placed
```

Repeat at 24 and 120 turns and with `--provider anthropic --model claude-sonnet-5`; capture each JSON output to
a separate result file. Actual idle-gap and fleet experiments need timed provider requests and controlled shared
prefixes, which the existing live harness does not yet implement. Do not substitute virtual-time results for
those experiments. Live quality, total cost including output/retries, and latency remain unvalidated here.

The current conclusion is directional support from a cache model, plus evidence that the model needs better
calibration. The proposed half-cost result at 60 turns is still a hypothesis.
