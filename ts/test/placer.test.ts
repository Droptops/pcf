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

test("keeps the last front module's marker when modules are only appended", () => {
  const placer = new MemoryPlacer({ countTokens: (text) => text.length, writeMultiplier: 1.25, readMultiplier: 0.1 });
  const base = [{ id: "ref", content: "reference ".repeat(800) }, { id: "plan", content: "plan basic" }];
  placer.split(base, 0);
  const grown = [...base, { id: "new", content: "new module" }];
  assert.deepEqual(placer.split(grown, 0).front.map((m) => [m.id, m.stable]),
                   [["ref", true], ["plan", true], ["new", false]]);
  assert.ok(placer.split(grown, 0).front.every((m) => m.stable));
});
