#!/usr/bin/env python3
"""Write ts/test/fixtures/placer.json: the Python MemoryPlacer's decisions on the domain scenarios.

The TypeScript port (ts/) replays the same inputs and must reach the same decisions. Module content is replaced
by its hash and token count, which is all the placer reads. Each scenario runs twice: with every 17th turn
reported cold, and with every 4th turn cold. Seeded random sessions then put module sizes, change rates and cold
turns near the decision threshold, so a small difference in the rule changes a decision; in half of them modules
join partway through, appended or inserted.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "src"), HERE]
from pcf import Segment  # noqa: E402
from pcf.compiler import segment_text  # noqa: E402
from pcf.families.openai_adapter import OpenAICompiler  # noqa: E402
from pcf.placement import MemoryPlacer  # noqa: E402
from domain_scenarios import SCENARIOS  # noqa: E402
from live_domain_sessions import memory  # noqa: E402

OUT = os.path.join(HERE, "..", "ts", "test", "fixtures", "placer.json")
TURNS, COLD_EVERY = 60, (17, 4)


def fixture() -> dict:
    tokenizer = OpenAICompiler("gpt-5.6").tokenizer
    sessions = []
    for key, every in ((k, e) for e in COLD_EVERY for k in sorted(SCENARIOS)):
        scenario = SCENARIOS[key]
        placer = MemoryPlacer(tokenizer, write_multiplier=1.25, read_multiplier=0.1)
        history, turns = [], []
        for turn in range(TURNS):
            mem = memory(scenario, turn)
            cold = turn > 0 and turn % every == 0
            front, tail = placer.split(mem, history, cold=cold)
            turns.append({
                "modules": [{"id": s.id, "content": hashlib.sha256(segment_text(s).encode()).hexdigest()[:16],
                             "tokens": tokenizer.count(segment_text(s))} for s in mem],
                "historyTokens": sum(tokenizer.count(segment_text(h)) for h in history),
                "cold": cold,
                "front": [[s.id, s.stable] for s in front], "tail": [s.id for s in tail]})
            ask, expected = scenario.question(turn)
            history.append(Segment(f"h{turn}", "history", [{"role": "user", "content": ask.text},
                                                            {"role": "assistant",
                                                             "content": scenario.reply(turn, expected)}]))
        sessions.append({"scenario": key, "coldEvery": every, "turns": turns})
    rng = random.Random(20260925)
    for n in range(8):
        counts: dict[str, int] = {}

        class Lookup:
            def count(self, text: str) -> int:
                return counts[text]
        placer = MemoryPlacer(Lookup(), write_multiplier=1.25, read_multiplier=0.1)
        spec = [(f"m{k}", rng.uniform(0.02, 0.7), rng.randint(20, 3000)) for k in range(rng.randint(2, 7))]
        # from session 4 on, modules after the first join partway through: appended at the end or inserted
        start = {name: 0 if n < 4 or k == 0 else rng.randint(0, 40) for k, (name, _, _) in enumerate(spec)}
        version = {name: 0 for name, _, _ in spec}
        history, history_tokens, turns = [], 0, []
        for turn in range(80):
            mem = []
            for name, p, size in spec:
                version[name] += rng.random() < p
                if turn < start[name]:
                    continue
                text = f"{name} v{version[name]}"
                counts[text] = size
                mem.append(Segment(name, "memory", text, provenance=name))
            cold = rng.random() < 0.15
            front, tail = placer.split(mem, history, cold=cold)
            turns.append({"modules": [{"id": seg.id, "content": seg.content, "tokens": counts[seg.content]}
                                      for seg in mem],
                          "historyTokens": history_tokens, "cold": cold,
                          "front": [[seg.id, seg.stable] for seg in front], "tail": [seg.id for seg in tail]})
            step = rng.randint(10, 400)
            entry = Segment(f"h{turn}", "history", [{"role": "user", "content": f"h{n}-{turn}"}])
            counts[segment_text(entry)] = step
            history.append(entry)
            history_tokens += step
        sessions.append({"scenario": f"random-{n}", "coldEvery": "random", "turns": turns})
    return {"writeMultiplier": 1.25, "readMultiplier": 0.1, "decay": 0.7, "sessions": sessions}


if __name__ == "__main__":
    with open(OUT, "w") as f:
        json.dump(fixture(), f, separators=(",", ":"))
        f.write("\n")
    print(OUT)
