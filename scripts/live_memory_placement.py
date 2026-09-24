#!/usr/bin/env python3
"""Live A/B of memory placement: all memory in front vs MemoryPlacer. Offline by default; paid calls need --run.

One scripted support session is run twice on the OpenAI Responses API, once per arm, each in its own cache
namespace. A stable "profile" module never changes; an "account" module changes every turn. History is
scripted and identical in both arms so cache costs compare; the model's answers are recorded and checked
against the memory the turn was asked about (a crude correctness check, not a quality evaluation).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pcf import Context, Segment  # noqa: E402
from pcf.families.openai_adapter import OpenAICompiler  # noqa: E402
from pcf.placement import MemoryPlacer  # noqa: E402

POLICIES = [f"Policy {i}: when a customer asks about topic {i}, check the account record, cite the relevant "
            f"order or ticket id, and keep the reply under three sentences." for i in range(60)]
PROFILE = {"name": "Dana", "preferred_language": "Spanish", "plan": "Unlimited Plus"}


def account(turn: int) -> dict:
    return {"open_tickets": turn + 2, "latest_ticket": f"T-{4100 + turn}"}


def question(turn: int) -> tuple[str, str]:
    """(question, substring a correct answer must contain)"""
    if turn % 2:
        return "What is my preferred language? Answer with one word.", PROFILE["preferred_language"]
    return "How many open tickets do I have right now? Answer with just the number.", str(account(turn)["open_tickets"])


def history_turn(turn: int, text: str) -> Segment:
    reply = f"Answered question {turn}. " + " ".join(f"Order {9000 + turn * 10 + k} is on schedule." for k in range(30))
    return Segment(f"h{turn}", "history", [{"role": "user", "content": text}, {"role": "assistant", "content": reply}])


def session(arm: str, turns: int, model: str, nonce: str, client=None) -> list[dict]:
    compiler = OpenAICompiler(model)
    placer = MemoryPlacer(compiler.tokenizer)
    system = Segment("s", "system", f"Session {nonce}.\n" + "\n".join(POLICIES))
    history, rows = [], []
    for turn in range(turns):
        memory = [Segment("profile", "memory", PROFILE), Segment("account", "memory", account(turn))]
        front, tail = placer.split(memory, history) if arm == "placed" else (memory, [])
        text, expected = question(turn)
        ctx = Context([system, *front, *history, *tail, Segment("u", "user", text, stable=False)],
                      cache_namespace=f"{arm}-{nonce}")
        request = {**compiler.compile(ctx).request, "max_output_tokens": 64}
        row = {"turn": turn, "tail": [s.id for s in tail], "input_items": len(request["input"])}
        if client is not None:
            response = client.responses.create(**request)
            usage = compiler.usage_from_response(response)
            answer = (getattr(response, "output_text", "") or "").strip()
            row.update(cached=usage.cache_read_input_tokens, written=usage.cache_creation_input_tokens,
                       uncached=usage.input_tokens, answer=answer, correct=expected.lower() in answer.lower())
        rows.append(row)
        history.append(history_turn(turn, text))
    return rows


def summarize(rows: list[dict]) -> dict:
    keys = ("cached", "written", "uncached")
    out = {k: sum(r[k] for r in rows) for k in keys if all(k in r for r in rows)}
    if "correct" in rows[0]:
        out["correct"] = f"{sum(r['correct'] for r in rows)}/{len(rows)}"
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="make paid OpenAI calls (needs OPENAI_API_KEY)")
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--model", default="gpt-5.6")
    args = parser.parse_args()
    client, nonce = None, "offline"
    if args.run:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("--run requires OPENAI_API_KEY")
        import openai
        client, nonce = openai.OpenAI(), uuid.uuid4().hex[:12]  # fresh prefix: both arms start cold
    result = {arm: session(arm, args.turns, args.model, nonce, client) for arm in ("front", "placed")}
    result["summary"] = {arm: summarize(rows) for arm, rows in result.items()}
    print(json.dumps(result, indent=2))
