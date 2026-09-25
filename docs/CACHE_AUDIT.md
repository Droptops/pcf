# Cache audit: what broke the cache, from existing logs

`scripts/cache_audit.py` reads a log of provider requests, exactly as sent, with the usage each response returned. It
reports which field's changes cost the most cache. It needs no PCF types and no change to the request path. It is
a prototype.

```bash
python scripts/cache_audit.py LOG.jsonl --ttl 300
```

Each line of the log is `{"conversation", "ts", "request", "usage": {"cached", "written", "uncached"}}`. For each
request, the audit finds the longest prefix it shares with the conversation's previous request. For a
conversation's first request it uses the best match among earlier requests. It then sorts the loss into three kinds:

- **changed:** a field changed, and the rest of the previous prompt after it was billed again. The field is named
  from the request itself; a memory block written as JSON with a `source` reads as `memory 'balance' balance_due`.
- **unread:** a prefix that could have been reused was not read, because no cache marker covered it or it was
  below the provider minimum.
- **expired:** the gap since the previous request was longer than `--ttl`.

Token counts are the provider's input total, split in proportion to bytes. The loss is priced at the write price
minus the read price. It is an upper bound on what a better layout could save, since data that really changed has
to be billed again anyway.

## On the committed runs

The committed runs store usage but not requests. `--from-domain` and `--from-fleet` rebuild each request with the
harness that sent it; the content is the same apart from the session nonce. They then pair each request with the
usage the provider actually returned.

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
