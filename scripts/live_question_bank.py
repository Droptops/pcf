#!/usr/bin/env python3
"""Paired quality check of memory placement on harder questions. Offline by default; paid calls need --run.

Each item is one question asked at the end of a scripted session built with the placement harness
(scripts/live_memory_placement.py); earlier turns are scripted, so only the final question is sent, once per arm.
MemoryPlacer's state is replayed over the earlier turns offline. Item types:
  conflict - a history turn asks to switch contact channel; memory still holds the old one, and the system policy
             says the customer's latest request wins
  cross    - one answer needs two modules: "<open tickets>, <plan>"
  far      - a standard lookup at the end of a 60-100 turn session, so front memory sits far from the question
Arms are compared pairwise per type and model with an exact McNemar test on the paired outcomes.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pcf import Context, Segment  # noqa: E402

spec = importlib.util.spec_from_file_location("placement", os.path.join(os.path.dirname(__file__),
                                                                        "live_memory_placement.py"))
placement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(placement)

ARMS = ("front", "tail", "placed")
POLICY = ("Policy: when the customer's most recent request in this conversation conflicts with the account record, "
          "follow the customer's request.")


def item_specs(per_type: int):
    """(type, final turn, history style) for every item; deterministic."""
    specs = []
    for k in range(per_type):
        final = 12 + (k * 7) % 29
        while final % 6 < 2:  # the record must not change between the customer's request and the question
            final += 1
        specs.append(("conflict", final, "varied" if k % 2 else "template"))
        specs.append(("cross", 10 + (k * 5) % 31, "varied" if k % 2 else "template"))
        specs.append(("far", 60 + (k * 13) % 41, "template" if k % 3 else "varied"))
    return specs


def build(kind: str, final: int, style: str, arm: str, compiler, tag: str):
    """The context for one item under one arm, and the expected answer."""
    placer = placement.MemoryPlacer(compiler.tokenizer, write_multiplier=compiler.descriptor.cache_write_multiplier,
                                    read_multiplier=0.1)
    system = Segment("s", "system", f"Session {tag}-{arm}.\n" + "\n".join([*placement.POLICIES, POLICY]))
    history, front, tail = [], [], []
    switch_at = final - 2
    for turn in range(final + 1):
        front, tail = placement.arrange(arm, placer, placement.memory(turn), history)
        if turn == final:
            break
        if kind == "conflict" and turn == switch_at:
            old = placement.preferences(final)["contact_channel"]  # what the record still says at the question
            new = next(c for c in placement.CHANNELS if c != old)
            history.append(Segment(f"h{turn}", "history", [
                {"role": "user", "content": f"Please stop contacting me by {old}. Use {new} from now on."},
                {"role": "assistant", "content": f"Understood, I will use {new} from now on."}]))
            continue
        text, expected = placement.question(turn)
        history.append(placement.history_turn(turn, text, expected, style))
    if kind == "conflict":
        question = "How should you contact me? Answer with one word."
        expected = new
    elif kind == "cross":
        question = "How many open tickets do I have, and which plan am I on? Answer as: <number>, <plan>."
        expected = f"{placement.account(final)['open_tickets']}, {placement.notes(final)['current_plan']}"
    else:
        question, expected = placement.question(final)
    ctx = Context([system, *front, *history, *tail, Segment("u", "user", question, stable=False)])
    return ctx, expected


def grade(kind: str, final: int, answer: str, expected: str) -> bool:
    if kind == "conflict":
        return placement.answer_value(answer, 2) == expected.lower()
    if kind == "cross":
        number, plan = expected.split(", ")
        return placement.answer_value(answer, 0) == number and placement.answer_value(answer, 1) == plan.lower()
    return placement.answer_value(answer, final) == expected.lower()


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n = only_a + only_b
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n) if n else 1.0
    return {"only_first_correct": only_a, "only_second_correct": only_b, "p_exact": round(p, 4)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--provider", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--model")
    parser.add_argument("--per-type", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    model = args.model or ("gpt-5.6" if args.provider == "openai" else "claude-sonnet-5")
    compiler = placement.make_compiler(args.provider, model)
    jobs = [(i, kind, final, style, arm) for i, (kind, final, style) in enumerate(item_specs(args.per_type))
            for arm in ARMS]
    built = {(i, arm): build(kind, final, style, arm, compiler, f"qb{i}") for i, kind, final, style, arm in jobs}
    if not args.run:
        sizes = [compiler.compile(ctx).total_tokens for ctx, _ in built.values()]
        print(json.dumps({"offline": True, "items": len(jobs) // len(ARMS), "calls": len(jobs),
                          "est_tokens_mean": round(sum(sizes) / len(sizes)), "est_tokens_max": max(sizes)}, indent=1))
        raise SystemExit
    key = "OPENAI_API_KEY" if args.provider == "openai" else "OPENROUTER_API_KEY"
    if not os.environ.get(key):
        raise SystemExit(f"--run --provider {args.provider} requires {key}")
    if args.provider == "openai":
        import openai
        client = openai.OpenAI()
    else:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ[key], base_url="https://openrouter.ai/api")

    class Cfg:
        effort, thinking = "low", "default"

    def ask(job):
        i, kind, final, style, arm = job
        ctx, expected = built[(i, arm)]
        _, answer, out = placement.call(args.provider, client, compiler.compile(ctx).request, Cfg)
        answer = answer.strip()
        return {"item": i, "type": kind, "final_turn": final, "style": style, "arm": arm, "expected": expected,
                "answer": answer, "correct": grade(kind, final, answer, expected), **out}

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(ask, jobs))
    table, tests = {}, {}
    for kind in ("conflict", "cross", "far"):
        by_arm = {arm: [r["correct"] for r in sorted((r for r in rows if r["type"] == kind and r["arm"] == arm),
                                                      key=lambda r: r["item"])] for arm in ARMS}
        table[kind] = {arm: f"{sum(v)}/{len(v)}" for arm, v in by_arm.items()}
        tests[kind] = {f"{a} vs {b}": mcnemar(by_arm[a], by_arm[b])
                       for a, b in (("tail", "front"), ("placed", "front"), ("tail", "placed"))}
    print(json.dumps({"meta": {"date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                               "git_sha": placement.git_sha(), "provider": args.provider, "model": model,
                               "per_type": args.per_type},
                      "correct": table, "mcnemar": tests, "rows": rows}, indent=1))
