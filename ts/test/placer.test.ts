// The TypeScript placer reaches the Python MemoryPlacer's decisions on every turn of the domain scenarios.
// Regenerate the fixture with `python scripts/export_placer_fixture.py`.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { MemoryPlacer } from "../src/index.ts";

const fixture = JSON.parse(readFileSync(new URL("./fixtures/placer.json", import.meta.url), "utf8"));

for (const session of fixture.sessions) {
  test(`matches Python on ${session.scenario}, cold every ${session.coldEvery}`, () => {
    const tokens = new Map<string, number>();
    const placer = new MemoryPlacer({ countTokens: (text) => tokens.get(text)!, writeMultiplier: fixture.writeMultiplier,
                                      readMultiplier: fixture.readMultiplier, decay: fixture.decay });
    session.turns.forEach((turn: any, i: number) => {
      for (const m of turn.modules) tokens.set(m.content, m.tokens);
      const { front, tail } = placer.split(turn.modules, turn.historyTokens, { cold: turn.cold });
      assert.deepEqual(front.map((m) => [m.id, m.stable]), turn.front, `turn ${i} front`);
      assert.deepEqual(tail.map((m) => m.id), turn.tail, `turn ${i} tail`);
    });
  });
}

test("rejects prices that make caching pointless", () => {
  assert.throws(() => new MemoryPlacer({ countTokens: () => 1, writeMultiplier: 0.1, readMultiplier: 0.1 }), RangeError);
});
