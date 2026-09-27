// The TypeScript placer reaches the Python MemoryPlacer's decisions on every turn of the domain scenarios.
// Regenerate the fixture with `python scripts/export_placer_fixture.py`.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { ConcurrentPlacementUpdate, MemoryPlacer } from "../src/index.ts";

const fixture = JSON.parse(readFileSync(new URL("./fixtures/placer.json", import.meta.url), "utf8"));

for (const session of fixture.sessions) {
  test(`matches Python on ${session.scenario}, cold every ${session.coldEvery}`, () => {
    const tokens = new Map<string, number>();
    const options = session.options ?? {};
    const placer = new MemoryPlacer({ countTokens: (text) => tokens.get(text)!, writeMultiplier: fixture.writeMultiplier,
                                      readMultiplier: fixture.readMultiplier, decay: fixture.decay,
                                      expectedTurns: options.expected_turns ?? undefined,
                                      minCacheableTokens: options.min_cacheable_tokens ?? 0 });
    session.turns.forEach((turn: any, i: number) => {
      for (const m of turn.modules) tokens.set(m.content, m.tokens);
      const { front, tail } = placer.split(turn.modules, turn.historyTokens,
                                           { cold: turn.cold, prefixTokens: turn.prefixTokens ?? 0 });
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

test("durable turn ids are idempotent across restart", () => {
  const options = { countTokens: (text: string) => text.length, tokenizerId: "chars-v1", expectedTurns: 20 };
  const placer = new MemoryPlacer(options);
  const memory = [{ id: "record", content: "v0" }];
  const first = placer.split(memory, 100, { turnId: "request-0", expectedRevision: 0 });
  const before = placer.exportState();
  assert.deepEqual(placer.split(memory, 100, { turnId: "request-0", expectedRevision: 0 }), first);
  assert.deepEqual(placer.exportState(), before);
  const restarted = new MemoryPlacer(options);
  restarted.restoreState(JSON.parse(JSON.stringify(before)));
  assert.deepEqual(restarted.split(memory, 100, { turnId: "request-0" }), first);
  assert.equal(restarted.revision, 1);
  assert.throws(() => restarted.split([{ id: "record", content: "v1" }], 100, { turnId: "request-0" }),
                /different inputs/);
});

test("stale concurrent revisions and configuration drift are rejected", () => {
  const options = { countTokens: (text: string) => text.length, tokenizerId: "chars-v1" };
  const placer = new MemoryPlacer(options);
  placer.split([{ id: "record", content: "v0" }], 100, { turnId: "request-0", expectedRevision: 0 });
  assert.throws(() => placer.split([{ id: "record", content: "v1" }], 100,
                                   { turnId: "request-1", expectedRevision: 0 }), ConcurrentPlacementUpdate);
  const other = new MemoryPlacer({ ...options, tokenizerId: "chars-v2" });
  assert.throws(() => other.restoreState(placer.exportState()), /configuration/);
});


test("minimum cacheability counts the lead before memory", () => {
  const placer = new MemoryPlacer({ countTokens: (text) => [...text].filter((c) => c === "x").length,
                                    minCacheableTokens: 1000 });
  placer.split([{ id: "m", content: "x".repeat(100) + "a" }], 90, { prefixTokens: 950 });
  assert.deepEqual(placer.split([{ id: "m", content: "x".repeat(100) + "b" }], 90,
                                { prefixTokens: 950 }).tail.map((m) => m.id), ["m"]);
});

test("minimum cacheability counts the history a change re-bills", () => {
  // 300 + 50k history is far above the minimum: a module that changes every turn must not pin the history
  const placer = new MemoryPlacer({ countTokens: (text) => [...text].filter((c) => c === "x").length,
                                    minCacheableTokens: 1024 });
  const tails = [0, 1, 2, 3].map((turn) =>
    placer.split([{ id: "m", content: "x".repeat(300) + turn }], 50_000).tail.map((m) => m.id));
  assert.deepEqual(tails, [[], ["m"], ["m"], ["m"]]);
});

test("cache minimum band keeps a module in front when the tail leaves the prompt uncached", () => {
  // 2500 + 1200 < 4096 <= 2500 + 1200 + 600: in the tail nothing is cached and the lead, history and module are
  // billed uncached every turn; in front the prompt is cached, and a change every 3rd turn costs less
  const placer = new MemoryPlacer({ countTokens: (text) => [...text].filter((c) => c === "x").length,
                                    minCacheableTokens: 4096 });
  const tails = Array.from({ length: 30 }, (_, turn) =>
    placer.split([{ id: "m", content: "x".repeat(600) + Math.floor(turn / 3) }], 1200, { prefixTokens: 2500 })
      .tail.map((m) => m.id));
  assert.deepEqual(tails, Array.from({ length: 30 }, () => []));
});

test("cache minimum band lets a module that changes every turn go to the tail", () => {
  // once its decayed rate p has p * (w - r) > 1 - r, cache writes cost more than the uncached prompt
  const placer = new MemoryPlacer({ countTokens: (text) => [...text].filter((c) => c === "x").length,
                                    minCacheableTokens: 4096 });
  const tails: string[][] = [];
  const rates: number[] = [];
  for (let turn = 0; turn < 10; turn++) {
    tails.push(placer.split([{ id: "m", content: "x".repeat(600) + turn }], 1200, { prefixTokens: 2500 })
      .tail.map((m) => m.id));
    rates.push(placer.exportState().seen.m.rate);
  }
  assert.deepEqual(tails, [[], [], [], [], [], ["m"], ["m"], ["m"], ["m"], ["m"]]);
  assert.ok(rates[4] * (1.25 - 0.1) <= 1 - 0.1 && 1 - 0.1 < rates[5] * (1.25 - 0.1));
});

test("cache minimum keeps a module outside the band in the tail in either order", () => {
  // i (1500 tokens) changes every turn, j (1700) every other turn; history 1400, minimum 4096. In one pass with i
  // first, i went back to the front because 1400 + 1500 stays below the minimum, then j lifted the prompt to 4600
  // and i's changes rewrote it every turn; with j first, i stayed in the tail. Now j goes back first, i would lift
  // 1400 + 1700 to the minimum, and i stays in the tail in either order once its rate leaves the band.
  const tails = ["ij", "ji"].map((order) => {
    const placer = new MemoryPlacer({ countTokens: (text) => [...text].filter((c) => c === "x").length,
                                      minCacheableTokens: 4096 });
    return Array.from({ length: 20 }, (_, turn) => {
      const i = { id: "i", content: "x".repeat(1500) + turn };
      const j = { id: "j", content: "x".repeat(1700) + Math.floor(turn / 2) };
      return placer.split(order === "ij" ? [i, j] : [j, i], 1400).tail.map((m) => m.id);
    });
  });
  const expected = Array.from({ length: 20 }, (_, turn) => (turn < 5 ? [] : ["i"]));
  assert.deepEqual(tails, [expected, expected]);
});

test("durable retry state is bounded and snapshot is exact", () => {
  const placer = new MemoryPlacer({ countTokens: (text) => text.length, tokenizerId: "chars",
                                    maxIdempotencyEntries: 2 });
  const memory = [{ id: "m", content: "v" }];
  const first = placer.splitAndSnapshot(memory, 10, { turnId: "t0", expectedRevision: 0 });
  placer.split(memory, 10, { turnId: "t1", expectedRevision: 1 });
  placer.split(memory, 10, { turnId: "t2", expectedRevision: 2 });
  assert.equal(first.state.revision, 1);
  assert.deepEqual(Object.keys(first.state.decisions), ["t0"]);
  assert.deepEqual(Object.keys(placer.exportState().decisions), ["t1", "t2"]);
  assert.throws(() => placer.split(memory, 10, { turnId: "t0" }), /requires expectedRevision/);
});

test("retry eviction after restore removes the oldest decision", () => {
  const options = { countTokens: (text: string) => text.length, tokenizerId: "chars", maxIdempotencyEntries: 3 };
  const placer = new MemoryPlacer(options);
  const memory = [{ id: "m", content: "v" }];
  // age order differs from both string order and the integer-key order Object.fromEntries imposes
  const first = new Map(["11", "10", "9"].map((id, n) =>
    [id, placer.split(memory, 10, { turnId: id, expectedRevision: n })]));
  const restored = new MemoryPlacer(options);
  restored.restoreState(JSON.parse(JSON.stringify(placer.exportState())));
  restored.split(memory, 10, { turnId: "12", expectedRevision: 3 });
  assert.deepEqual(Object.keys(restored.exportState().decisions).sort(), ["10", "12", "9"]);
  const before = restored.exportState();
  assert.deepEqual(restored.split(memory, 10, { turnId: "10" }), first.get("10"));
  assert.deepEqual(restored.exportState(), before);
  assert.throws(() => restored.split(memory, 10, { turnId: "11", expectedRevision: 0 }), ConcurrentPlacementUpdate);
});

test("restore rejects missing or duplicate decision revisions", () => {
  const options = { countTokens: (text: string) => text.length, tokenizerId: "chars" };
  const placer = new MemoryPlacer(options);
  const memory = [{ id: "m", content: "v" }];
  placer.split(memory, 10, { turnId: "t0", expectedRevision: 0 });
  placer.split(memory, 10, { turnId: "t1", expectedRevision: 1 });
  const missing = JSON.parse(JSON.stringify(placer.exportState()));
  delete missing.decisions.t0.revision;
  const duplicate = JSON.parse(JSON.stringify(placer.exportState()));
  duplicate.decisions.t0.revision = duplicate.decisions.t1.revision;
  assert.throws(() => new MemoryPlacer(options).restoreState(missing), /idempotency state is invalid/);
  assert.throws(() => new MemoryPlacer(options).restoreState(duplicate), /unique/);
});

test("restore rejects more decisions than the cap", () => {
  const options = { countTokens: (text: string) => text.length, tokenizerId: "chars", maxIdempotencyEntries: 2 };
  const placer = new MemoryPlacer(options);
  const memory = [{ id: "m", content: "v" }];
  for (let n = 0; n < 3; n++) placer.split(memory, 10, { turnId: `t${n}`, expectedRevision: n });
  const state = JSON.parse(JSON.stringify(placer.exportState()));
  new MemoryPlacer(options).restoreState(state);
  state.decisions.t0 = { ...state.decisions.t1, revision: 1 };  // the decision eviction removed
  assert.throws(() => new MemoryPlacer(options).restoreState(state), /more decisions than maxIdempotencyEntries/);
});

test("restore rejects malformed decisions, observed state and duplicate front ids, as Python does", () => {
  const options = { countTokens: (text: string) => text.length, tokenizerId: "chars" };
  const placer = new MemoryPlacer(options);
  placer.split([{ id: "a", content: "v" }, { id: "b", content: "w" }], 10, { turnId: "t0", expectedRevision: 0 });
  const exported = JSON.stringify(placer.exportState());
  new MemoryPlacer(options).restoreState(JSON.parse(exported));
  const invalid = /idempotency state is invalid/;
  const cases: Array<[(state: any) => void, RegExp]> = [
    [(state) => state.decisions.t0.front.push({ id: "a", stable: false }), invalid],
    [(state) => state.previousFront.push("a"), /previousFront is invalid/],
    [(state) => { state.decisions.t0 = null; }, invalid],
    [(state) => { state.decisions.t0.extra = true; }, invalid],
    [(state) => { state.decisions.t0.inputHash = "sha256:" + "A".repeat(64); }, invalid],
    [(state) => { state.decisions.t0.inputHash = state.decisions.t0.inputHash.slice(0, -1); }, invalid],
    [(state) => { state.decisions.t0.front[0].extra = 1; }, invalid],
    [(state) => { state.decisions.t0.front[0].id = " "; }, invalid],
    [(state) => { state.decisions.t0.front[0].id = 5; }, invalid],
    [(state) => { state.seen.a.last = "v"; }, /observed state is invalid/],
    [(state) => { state.seen.a.extra = 1; }, /observed state is invalid/],
    [(state) => { state.quiet[" "] = 0; }, /quiet state is invalid/],
  ];
  for (const [corrupt, message] of cases) {
    const state = JSON.parse(exported);
    corrupt(state);
    assert.throws(() => new MemoryPlacer(options).restoreState(state), { name: "RangeError", message });
  }
});
