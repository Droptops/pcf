# Pilot protocol: PCF in one production assistant

The committed evidence comes from scripted and synthetic sessions with assumed cache prices. This protocol tests
whether the advantage holds in one real assistant: billed cost, latency and answer quality against a well-configured
baseline, on real traffic.

## Question

For assistant *X*, does placing memory with `MemoryPlacer` lower billed cost per conversation without lowering
answer quality or raising latency, compared with the best configuration of the current prompt?

## Arms

- **Baseline:** the current prompt, tuned before the pilot starts. Stable content (system prompt, tools, reference
  documents) comes first, provider prompt caching is on, and on Anthropic the cache markers sit on the stable
  blocks. This is the "front, tuned" layout in the synthetic runs. A pilot against an untuned baseline would
  overstate the saving.
- **Treatment:** the same content as PCF segments, compiled by the provider adapter. `MemoryPlacer` places each
  memory module per turn; reference documents are memory modules with their own `provenance`.

Content, model, temperature, tools and output limits are identical in both arms. Only order and cache markers differ.

## Assignment

Assign whole conversations, never single turns: cache state belongs to a conversation, so mixing arms inside one
destroys both. Hash the conversation id into two buckets (50/50, or 10/90 for a cautious start). Keep a
conversation in its bucket for its whole life, including after an idle gap.

## Metrics

Primary:

- **Billed input cost per conversation**, from provider usage (`usage_from_response`: cached, written and uncached
  input tokens) priced with the current price sheet, not the assumed multipliers of the synthetic runs.

Secondary:

- **Output cost** per conversation (placement can change reply length).
- **Latency:** time to first token and to the full response, p50 and p90, per turn.
- **Answer quality:**
  - Record lookups: an automatic check of each answer against the record the assistant was given, for the
    questions where the source of truth is known.
  - A blind human review of a random sample from both arms.
  - Escalation, retry and complaint rates.
- **Cache behaviour:** read share of input tokens, and requests that arrive after the cache lifetime (use
  `split(..., cold=True)` for those, predicted from time since the last request).

## Sample size

The recorded input-plus-output reductions against tuned front were 14.5% on GPT and 36.6-38.0% on Claude
(about 1.17-1.61x). They used assumed prices. Estimate cost variability from representative conversations, choose a
minimum worthwhile saving, and calculate the required conversation count before starting; a fixed few hundred
conversations is not guaranteed to settle cost. Quality needs its own sample-size calculation. Detecting an error-rate change from 5% to 3% on record lookups at 80% power
(two-sided α = 0.05) takes about 1,500 checked answers per arm. For non-inferiority with a 1-point margin, plan for
several thousand independent observations. Answers within a conversation are correlated: account for clustering
when sizing and analyzing the pilot. Set the margin, analysis method and sample size with the assistant's owner
before starting.

## Stop and ship rules

- **Stop early** if the treatment's checked error rate exceeds the baseline's by more than the margin at any weekly
  look, or if p90 latency rises by more than an agreed amount.
- **Ship** at the prespecified final analysis if billed cost per conversation falls by at least the agreed amount,
  quality is non-inferior within the margin using a conversation-level confidence interval, and latency is no
  worse. Weekly checks are harm-monitoring rules, not repeated opportunities to declare success.
- **Report either way**, with the raw per-conversation usage and the quality sample, as for the synthetic runs.

## Known risks to check

- **Small models and in-conversation changes.** When a user changes a preference and the record is not updated,
  claude-haiku-4-5 followed the stale record near the question more often. Update the record when users change
  something, or keep that module in front.
- **Idle gaps.** Real conversations pause past the cache lifetime, which shrinks every layout's savings and
  `MemoryPlacer`'s edge most.
- **Tool loops.** The adapters keep tool loops flat (100% reuse in the scripted loops); confirm it on the
  assistant's real tools.
- **Latency of compilation.** Compiling adds local time per request (about 0.1 s at 400 history segments); measure
  it in the pilot's environment.

## Integration sketch

```python
from pcf import Context, Segment
from pcf.families.anthropic_adapter import AnthropicCompiler
from pcf.placement import MemoryPlacer

compiler = AnthropicCompiler("claude-sonnet-5")
placer = MemoryPlacer(compiler.tokenizer, write_multiplier=1.25, read_multiplier=0.1)  # or for_candidate()

def request(system, reference, records, history, question, cold=False):
    memory = [Segment("reference", "memory", reference, provenance="reference"),
              *[Segment(name, "memory", data, provenance=name) for name, data in records.items()]]
    front, tail = placer.split(memory, history, cold=cold)
    ctx = Context([Segment("s", "system", system), *front, *history, *tail,
                   Segment("u", "user", question, stable=False)])
    return compiler.compile(ctx).request  # send with the existing client; log compiler.usage_from_response(...)
```

`scripts/live_domain_sessions.py` is a working example of this loop, including usage and latency recording.
