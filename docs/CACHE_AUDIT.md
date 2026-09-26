# Cache audit: what broke the cache, from existing logs

`scripts/cache_audit.py` reads a log of provider requests, exactly as sent, with the usage each response returned. It
reports which fields are suspected of disrupting cache reuse. It needs no PCF types and no change to the request path. It is
a prototype.

```bash
python scripts/cache_audit.py LOG.jsonl --ttl 300
```

Each line of the log is `{"conversation", "ts", "request", "usage": {"cached", "written", "uncached"}}`. For each
request, the audit finds the longest prefix it shares with the conversation's previous request. For a
conversation's first request it uses the best earlier compatible match only within an explicit cache scope. It
classifies diagnostic estimates into three kinds:

- **changed:** a field changed, potentially invalidating the previous prompt after it. The field is named
  from the request itself; a memory block written as JSON with a `source` reads as `memory 'balance' balance_due`.
- **unread:** an estimated reusable prefix was not read. Missing markers and provider minimums are possible
  explanations; eviction, routing and other provider behavior can also cause misses.
- **expired:** the gap since the previous request reached `--ttl`; this is a timing hypothesis.

Token positions are estimated from the provider's input total in proportion to bytes. The diagnostic opportunity
uses write/read price differences and is capped against the request's non-read spending. It is not measured savings.

## On the committed runs

Older committed runs store usage but not requests. `--from-domain` and `--from-fleet` rebuild each request with the
harness that sent it; the content is the same apart from the session nonce. They then pair each request with the
usage the provider actually returned. New domain request captures are exported directly. The following table is
the historical prototype output before capped attribution and scope checks; do not treat its amounts as savings.

Claude, 60 turns, six scenarios, 2 repeats (`domain-claude-sonnet-5-60turn.json`):

| Layout | Read from cache | Cost attributed to misses | Top cause |
| --- | ---: | ---: | --- |
| tuned front | 77% | 1.09M units | each volatile module's field, about 115-135k units each (`claims` `amount_owed`, `vitals` `bp`, ...) |
| placed | 91% | 0.30M units | 14 unread prefixes, 87k units |

gpt-5.6 is similar: 73% and 0.82M units for tuned front, and 90% and 0.22M for placed.

## What it found in PCF itself

- **The first-request anchor bug.** Rebuilt from the paid probe before the fix, the second conversation's first
  request shows about 4.7k tokens of reusable prefix and 0 read. After the fix there are none
  (`tests/test_16_cache_audit.py`).
- **A one-time rewrite when a module moves, on Claude, before the fix.** In every placed Claude session of the
  60-turn domain run, the turn where `MemoryPlacer` first moved a module after the history (turn 6; also turn 9 in
  `clinical`) wrote the whole 4.6-7k-token prefix again with nothing read. The breakpoints before the moved module
  had never been written. The first-request anchor fix writes the reference's breakpoint on turn 0. In the
  post-fix Claude fleet run (1,080 requests) the audit finds no unread prefix, which explains most of the
  anchor fix's in-session saving on Claude.
- **The same rewrite on OpenAI, now fixed.** In the post-fix gpt-5.6 fleet run, placed sessions still showed 2
  unread prefixes each (36 in all, about 3.2k tokens each), about 176k units or 16% of the run's billed input.
  OpenAI reads only at markers present in the request. Once history exists, the budget of 4 goes to the last
  module anchor and 3 history endpoints, so no request after the first carried the reference's marker.
  `MemoryPlacer` now returns the front modules after the first one unstable on a warm turn where a module moves, so
  the first front module takes the anchor on that turn. In simulation the 60-turn placed cost on gpt-5.6 falls 7%
  and Claude is unchanged. A paid gpt-5.6 check (6 scenarios, 12 turns, `move-turn-gpt-5.6.json`) read the reference
  prefix on all 18 move turns, 3,022-4,092 tokens, and wrote 182-642. One request, `clinical` turn 3, read
  nothing on a turn without a move. Every other scenario read on that turn, and the audit cannot explain it from
  the request; it is most likely a provider-side miss.

## Limits

The prototype reads Anthropic Messages and OpenAI Responses shapes. It attributes each loss to the first changed
leaf only. Its token estimate is proportional to bytes. It has been run only on the committed synthetic logs, never
on a production log.

## Production log contract and validation workflow

No production-log validation has been completed. The audit accepts captured logs now; the workflow below measures
whether its diagnostic labels work on a particular assistant before using them to prioritize changes.

```json
{"conversation":"c-1","request_id":"req-1","cache_scope":"provider-project-route-A","ts":1800000000,"source":"production-captured","request":{"model":"model-id","instructions":"policy","input":"question"},"usage":{"cached":0,"written":0,"uncached":1200}}
```

`cache_scope` is a nonsecret identifier for the actual cache-sharing boundary (provider, account/project and route).
Cross-conversation matching is disabled without it. Different model IDs, prompt cache keys and provider-routing
settings are never matched. Within-conversation request order must be chronological; use chronological input overall
for cross-conversation analysis. Missing timestamps are reported, and cannot establish expiry. The configured TTL is
a hypothesis about retention; a gap beyond it does not prove why a provider missed. Identical prefixes may miss due
to routing, eviction or other provider behavior.

The audit validates usage as disjoint nonnegative integer token buckets, rejects duplicate request IDs and reports
missing timestamps, scope and request identities. It includes the write premium in `billed_input_units`: this is
observed usage priced at the ratios supplied, not an invoice total. Analyze different pricing cohorts separately.

Attribution still estimates token positions from bytes. It reports every differing previous leaf in `changed_fields`
and assigns the diagnostic estimate only once, to the first divergent field. Later changed fields are suspects,
not additive savings. `estimated_miss_opportunity_units` is capped against measured non-read spending per request;
it is **not recoverable savings**, a guaranteed upper bound on realizable savings, or a causal estimate. Individual
event/cause `lost_units` is retained as a legacy field name with that same diagnostic meaning. Existing consumers
should replace `estimated_lost_units` and `billed_input_units_approx` with the new totals fields.

1. Capture representative traffic, including short sessions, tool rounds, idle gaps and shared prefixes. Keep
   production records private and apply consistent redaction if exporting; redaction can change measured prefixes.
2. Select an independent sample before viewing audit predictions. Have an operator or controlled instrumentation
   label cache-miss categories using exact requests, marker configuration, timing and provider evidence. Include
   negative examples. Leave genuinely unknown cases unannotated; report their count.
3. Add annotations to those rows. Kinds may be `changed`, `unread`, `expired`, or an empty list for no detected
   category. For `changed`, also supply the expected first field label:

```json
{"annotation":{"kinds":["changed"],"first_changed_field":"memory 'account' balance"}}
```

4. Run the diagnostic validation:

```bash
python scripts/cache_audit.py captured.jsonl --ttl 300 --validate-labels
```

The report gives per-class TP/FP/FN, precision/recall, first-field matches and annotation coverage. It refuses to
claim validation when no annotations exist. Labels supplied by the same heuristic are not independent validation;
the tool cannot authenticate annotator provenance. Predefine acceptance thresholds with the assistant owner.
5. Test the proposed repair with the randomized [four-arm pilot](PILOT.md). That comparison measures realized cost,
   quality and latency. Diagnostic-label agreement alone does not validate estimated dollar savings.

New domain runs captured with `--capture-requests` export recorded requests and timestamps, tagged
`captured-synthetic`. Old scripted runs are tagged `reconstructed-synthetic`; their reconstructed timing and requests
must not establish production validity. Model-text runs without captured requests cannot be reconstructed by
substituting scripted answers and are rejected.
