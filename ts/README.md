# pcf-memory-placer

A TypeScript port of PCF's `MemoryPlacer`, for assistants that already build their own message list. It decides,
turn by turn, which memory modules stay in front of the conversation history (cached) and which go after it (re-sent
each turn), from each module's observed change rate and the provider's cache prices. It places the modules into an
Anthropic Messages request with cache markers, or into OpenAI chat messages. It has no dependencies and needs
Node 22.18 or later, which runs TypeScript directly.

```ts
import { MemoryPlacer, layoutAnthropic } from "pcf-memory-placer";

const placer = new MemoryPlacer({ countTokens: (t) => Math.ceil(t.length / 4), writeMultiplier: 1.25,
                                  readMultiplier: 0.1, minCacheableTokens: 1024, expectedTurns: 40 });

// every turn: memory modules in a fixed order, stable ones first
const placement = placer.split([
  { id: "reference", content: formularyText },
  { id: "chart", content: JSON.stringify(patientChart) },
], historyTokens, { cold: secondsSinceLastRequest >= 300 });
const request = layoutAnthropic({ model, max_tokens: 1024, system, messages: [...history, userTurn] }, placement);
```

Use one placer per conversation. Prices default to a 1.25 write and a 0.1 read; set `minCacheableTokens` to the
provider's minimum and `expectedTurns` to your typical conversation length, which decides whether a module that has
gone quiet is worth moving back in front while the cache is warm. Keep module order and content deterministic: any byte that changes counts as a
change. The decisions match the Python implementation turn for turn on the repository's domain scenarios and on
random sessions near the decision threshold (`test/placer.test.ts`; regenerate the fixture with
`python scripts/export_placer_fixture.py`).

What this port does not include: the Python compiler's full breakpoint budgeting, OpenAI explicit cache breakpoints,
tool definitions, usage accounting and the router. `layoutAnthropic` sets at most three markers: the first front
module, the last stable front module, and the end of the history.

```bash
npm test            # node --test
npm run typecheck   # tsc
```
