# Pilot protocol: four layouts in one production assistant

The existing results are synthetic, with assumed cache prices. This protocol measures actual usage, latency and
quality on one real assistant. It separates the value of placing current records near the question from the
incremental value of adaptive placement. No production pilot has been completed in this repository.

## Four arms

| Arm | Layout | What it tests |
| --- | --- | --- |
| `front-tuned` | Stable system/tools/reference first; current records before history, changing records unmarked | Well-configured existing baseline |
| `echo-all` | Freeze initial records in the front prefix; repeat all potentially changing records, marked current, before each question | Simple repetition without changing the cached prefix |
| `fixed-tail` | Stable references and records first; all potentially changing records after history from turn zero | Simple placement without adaptive decisions or duplicate records |
| `placed` | `MemoryPlacer` chooses front/tail from observed changes and cache prices | Incremental value of adaptation |

Declare potentially changing modules from the application schema before assignment. Echo-all and fixed-tail use
that declaration; do not infer it from future observations or select only the field a question needs. The synthetic
harness uses scenario metadata for this declaration. Keep model, system instructions, record sources, tools,
temperature and output limits fixed across arms. Each arm retains its actual responses in its own conversation
history. Echo-all's stale initial copy is intentional; label every repeated record as current.

## Assignment and analysis plan

Assign **whole conversations**, with equal allocation across the four arms:

```bash
python scripts/pilot_analysis.py --assign CONVERSATION_ID --salt PILOT_SPECIFIC_SALT
```

The matching Python function is `assign_multi(conversation_id, salt)`. Keep the salt, arm mapping and assignment
for the life of the pilot. Use one `MemoryPlacer` per placed conversation. A paused conversation stays in its arm;
report a cold cache from elapsed time before compiling, not after a zero-read response. Keep cache namespaces
separate across arms so one arm does not warm another. Document shared-prefix routing within an arm.

Before looking at results, freeze the model/provider, price ratios, eligibility rules, minimum worthwhile cost
saving, quality non-inferiority margin, latency ceiling, sample size and end date. Predeclare these contrasts:

1. `placed` versus `fixed-tail`: primary test of whether adaptation earns its complexity.
2. `placed` versus `echo-all`: comparison with simple repetition.
3. Each alternative versus `front-tuned`: benefit relative to the current tuned baseline.

The analyzer reports all six pairwise comparisons with **unadjusted, descriptive** 95% conversation-bootstrap
intervals. It does not decide whether to ship. Use a prespecified multiplicity procedure for confirmatory secondary
claims; do not pick a winning arm after inspecting six unadjusted intervals. Size quality and cost separately,
accounting for clustering within conversations. The synthetic runs are not a sample-size justification.

## Log contract

One JSON line per completed provider request:

```json
{"conversation":"c-123","request_id":"req-1","arm":"fixed-tail","cached":1200,"written":0,"uncached":200,"output_tokens":60,"latency_s":1.9,"ttft_s":0.4,"success":true,"correct":true,"blind_acceptable":true}
```

- Token buckets must be disjoint, nonnegative provider-usage counts. Record retries as separate billable requests;
  preserve failures separately so missing completions do not hide failure rates.
- `correct` is a boolean, or null/absent when unknown. Use record checks plus blind human review for tasks where
  lookup correctness misses instruction, tool or safety failures. Record escalation and complaint rates separately.
- `success` records operational completion; do not omit failed attempts. `blind_acceptable` is null until the
  arm-hidden reviews are adjudicated. The analyzer reports conversation-bootstrap differences for automatic error,
  blind-review error and operational failure rates.
- `latency_s` is full response latency; `ttft_s` is optional. Include compilation in an additional end-to-end timing
  measurement when evaluating production latency.
- Record model, provider, prices, deployment revision and evaluation rubric in the pilot manifest. Analyze homogeneous
  pricing cohorts separately. Do not mix synthetic exports into the production dataset.
- Keep a distinct conversation ID per assigned conversation. Duplicate request IDs, invalid counts and conversations
  appearing in multiple arms are rejected. Preserve timestamps and exact requests in the audit log described below.

```bash
python scripts/pilot_analysis.py pilot.jsonl --write 1.25 --read 0.1 --output 5 --samples 5000
```

Replace those illustrative ratios with the actual model's prices. Costs are usage-derived uncached-input-token
units; multiply by the actual uncached input price per token to obtain dollars. Include other billed fees separately.
The report includes per-arm cost per conversation, error rates, p50/p90 latency, and pairwise cost ratios and
quality/latency differences. A zero denominator produces a null ratio, not a spurious saving.

Legacy `baseline`/`treatment` logs retain their old analysis. Use `--legacy-two-arm` for old CLI assignment and
`assign()` for the old Python API. A new four-arm log must contain all four arms; missing arms are an error.

## Actual-response synthetic evaluation

Before production, exercise the four layouts on the same scripted question and record schedule while feeding each
model's own visible replies into subsequent turns:

```bash
python scripts/live_domain_sessions.py --run --provider openai --model gpt-5.6 \
  --arms front-tuned echo-all fixed-tail placed --history-mode model-text \
  --capture-requests --turns 60 --repeats 2 > model-history.json
python scripts/live_domain_sessions.py --analyze model-history.json
python scripts/pilot_analysis.py --from-domain model-history.json --four-arm > synthetic-pilot.jsonl
python scripts/pilot_analysis.py synthetic-pilot.jsonl
```

The first command makes paid calls. `model-text` requires responses; offline mode refuses to invent them. Existing
runs default to `scripted` and retain their interpretation. Analysis refuses to pool the two history modes. Actual
history is still a synthetic conversation with predetermined user questions and records, not a production agent
benchmark. Visible answer text is replayed; hidden reasoning, tool execution and tool-result state are not replayed
by this domain harness. Use the separate tool-loop harness for those provider behaviors.

The report counts prior-error exposures and repeated prior errors for repeated questions. These are diagnostic
counts, not causal proof of error propagation: a model can independently make the same mistake. Stale-history
subsets may differ by arm; pairwise stale comparisons use only turns where both histories expose a stale value.
Turn-level significance remains descriptive; the conversation is the inferential unit.

## Audit and validation

Use `--capture-requests` to preserve exact outbound SDK arguments and request timestamps. Export these with:

```bash
python scripts/cache_audit.py --from-domain model-history.json --arm placed > synthetic-audit.jsonl
python scripts/cache_audit.py synthetic-audit.jsonl --ttl 300 --json
```

Production instrumentation should capture the actual outbound request and returned usage, with cache scope and
request identity, using [CACHE_AUDIT.md](CACHE_AUDIT.md). Validate suspected causes against independent annotations,
then validate realized cost improvement using randomized pilot outcomes. The audit's heuristic opportunity estimate
must not be reported as measured savings.

## Stop and ship rules

Monitor quality, failures and latency for harm using predeclared rules. Weekly looks are not opportunities to declare
success. At the frozen final analysis, require worthwhile cost reduction, quality non-inferiority and acceptable
latency under the prespecified statistical plan. Prefer a simpler layout if adaptation does not demonstrate an
additional benefit. Publish the outcome even if no layout improves on baseline; publish only consented, sanitized
artifacts rather than raw production prompts.

Check realistic idle gaps, short sessions, record changes followed by quiet periods, changing module sizes,
contradictory user preferences and extended-thinking/tool-loop compatibility. Keep old records current when users
change preferences: moving stale records near a question can worsen answers, especially on smaller models.

## Release handoff

The public repository does not contain production prompts or review exports. After the frozen analysis and blind
review pass, store those artifacts in the approved private evidence system and record their SHA-256 identities in
`release/production-pilot.json`; see [`release/README.md`](../release/README.md). The release gate requires actual
provider prices, all four arm counts, every preregistered gate, arm-hidden independent review with adjudication, and
distinct pilot-owner and quality-reviewer approvals. A synthetic run, an unfinished pilot, or a handwritten claim
without bound artifact identities cannot cut an RC or stable tag.
