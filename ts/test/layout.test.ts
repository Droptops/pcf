import { test } from "node:test";
import assert from "node:assert/strict";
import { layoutAnthropic, layoutOpenAIChat } from "../src/index.ts";
import type { Placement } from "../src/index.ts";

const placement: Placement = {
  front: [{ id: "ref", content: "REFERENCE", stable: true }, { id: "plan", content: "PLAN", stable: true }],
  tail: [{ id: "bal", content: "BALANCE", stable: false }],
};
const history = [{ role: "user" as const, content: "q1" }, { role: "assistant" as const, content: "a1" }];

test("anthropic: front before history, tail with the question, three markers", () => {
  const out = layoutAnthropic({ model: "m", system: "policy", messages: [...history, { role: "user", content: "q2" }] },
                              placement);
  assert.equal(out.model, "m");
  const texts = out.messages.map((m) => (m.content as any[]).map((b) => b.text));
  assert.deepEqual(texts, [["REFERENCE", "PLAN"], ["q1"], ["a1"], ["BALANCE", "q2"]]);
  const marked = out.messages.flatMap((m) => (m.content as any[]).filter((b) => b.cache_control).map((b) => b.text));
  assert.deepEqual(marked, ["REFERENCE", "PLAN", "a1"]);
  assert.equal((out.system as any[])[0].cache_control, undefined);
});

test("anthropic: on a move turn only the first front module is marked", () => {
  const moving: Placement = { ...placement, front: [placement.front[0], { ...placement.front[1], stable: false }] };
  const out = layoutAnthropic({ messages: [...history, { role: "user", content: "q2" }] }, moving, { ttl: "1h" });
  const marked = out.messages.flatMap((m) => (m.content as any[]).filter((b) => b.cache_control));
  assert.deepEqual(marked.map((b) => b.text), ["REFERENCE", "a1"]);
  assert.deepEqual(marked[0].cache_control, { type: "ephemeral", ttl: "1h" });
});

test("anthropic: existing markers are replaced and inputs are not mutated", () => {
  const request = { system: [{ type: "text", text: "policy", cache_control: { type: "ephemeral" } }],
                    messages: [{ role: "user" as const, content: "q" }] };
  const out = layoutAnthropic(request, { front: [], tail: [] });
  assert.deepEqual(request.system[0].cache_control, { type: "ephemeral" });
  assert.deepEqual((out.system as any[])[0].cache_control, { type: "ephemeral" });
  assert.throws(() => layoutAnthropic({ messages: history }, placement));
});

test("openai chat: system first, then front, history, tail and question", () => {
  const out = layoutOpenAIChat([{ role: "system", content: "policy" }, ...history, { role: "user", content: "q2" }],
                               placement);
  assert.deepEqual(out.map((m) => m.role), ["system", "user", "user", "assistant", "user"]);
  assert.deepEqual((out[4].content as any[]).map((p) => p.text), ["BALANCE", "q2"]);
});
