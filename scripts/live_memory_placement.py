#!/usr/bin/env python3
"""Live A/B/C of memory placement. Offline by default; paid calls need --run.

--provider openai uses the OpenAI Responses API. --provider anthropic sends the Anthropic adapter's Messages
request unchanged to OpenRouter's Anthropic-compatible endpoint, pinned to Anthropic as the upstream provider
(no fallbacks) so prompt caching is Anthropic's own.

One scripted support session is run per arm, each in its own cache namespace, --repeats times:
  front  - all memory before history, most volatile module last (the best hand ordering)
  tail   - all memory after history, just before the user turn
  placed - MemoryPlacer picks front or tail per module from observed change rates and cache prices
Four memory modules change at different rates (never, every 6th turn, every 3rd, every turn). History is
scripted and identical in every arm, and each scripted reply restates the value it answered, so after a module
changes the history holds a stale answer: a correct reply must follow memory, not the transcript. Answers are
checked by substring (a crude check, not a quality evaluation); stale-trap turns are reported separately.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pcf import Context, Segment  # noqa: E402
from pcf.families.anthropic_adapter import AnthropicCompiler  # noqa: E402
from pcf.families.openai_adapter import OpenAICompiler  # noqa: E402
from pcf.placement import MemoryPlacer  # noqa: E402

ARMS = ("front", "tail", "placed")
POLICIES = [f"Policy {i}: when a customer asks about topic {i}, check the account record, cite the relevant "
            f"order or ticket id, and keep the reply under three sentences." for i in range(60)]
CHANNELS = ["email", "SMS", "phone", "chat"]
PLANS = ["Basic", "Unlimited", "Unlimited Plus", "Premium", "Family", "Business"]


def profile() -> dict:  # never changes; large enough that re-billing it every turn is visible
    return {"name": "Dana Ruiz", "preferred_language": "Spanish", "member_since": 2019, "region": "Northwest",
            "devices": [f"Device {k}: model X{k}, purchased 20{15 + k}, warranty active" for k in range(8)],
            "addresses": [f"{100 + k} Main Street, Unit {k}, Springfield" for k in range(4)]}


def preferences(turn: int) -> dict:  # changes every 6th turn
    return {"contact_channel": CHANNELS[(turn // 6) % len(CHANNELS)], "quiet_hours": "21:00-08:00"}


def notes(turn: int) -> dict:  # changes every 3rd turn
    version = turn // 3
    return {"agent_notes": [f"Note {k}: follow up on shipment S-{700 + k}." for k in range(12)],
            "current_plan": PLANS[version % len(PLANS)]}


def account(turn: int) -> dict:  # changes every turn
    return {"open_tickets": turn + 2, "latest_ticket": f"T-{4100 + turn}"}


def memory(turn: int) -> list[Segment]:
    # Each module has its own provenance, so it gets its own breakpoint anchor.
    modules = [("profile", profile()), ("preferences", preferences(turn)), ("notes", notes(turn)),
               ("account", account(turn))]
    return [Segment(name, "memory", data, provenance=name) for name, data in modules]


QUESTIONS = [
    ("How many open tickets do I have right now? Answer with just the number.",
     lambda t: str(account(t)["open_tickets"])),
    ("Which plan am I on right now? Answer with just the plan name.", lambda t: notes(t)["current_plan"]),
    ("How should you contact me? Answer with one word.", lambda t: preferences(t)["contact_channel"]),
    ("What is my preferred language? Answer in English with one word.", lambda t: profile()["preferred_language"]),
]


def question(turn: int) -> tuple[str, str]:
    """(question, substring a correct answer must contain)"""
    text, expected = QUESTIONS[turn % len(QUESTIONS)]
    return text, expected(turn)


def history_turn(turn: int, text: str, answer: str) -> Segment:
    reply = f"The answer is {answer}. " + " ".join(f"Order {9000 + turn * 10 + k} is on schedule." for k in range(30))
    return Segment(f"h{turn}", "history", [{"role": "user", "content": text}, {"role": "assistant", "content": reply}])


def make_compiler(provider: str, model: str):
    return OpenAICompiler(model) if provider == "openai" else AnthropicCompiler(model, max_tokens=512)


def call(provider: str, client, request: dict) -> tuple[object, str]:
    """(response, answer text) for one compiled request."""
    if provider == "openai":
        # Low effort and room to answer: at 64 tokens reasoning models can return nothing visible.
        response = client.responses.create(**request, max_output_tokens=512, reasoning={"effort": "low"})
        return response, getattr(response, "output_text", "") or ""
    response = client.messages.create(**{**request, "model": "anthropic/" + request["model"]},
                                      extra_body={"provider": {"order": ["Anthropic"], "allow_fallbacks": False}})
    return response, "".join(getattr(block, "text", "") for block in response.content)


def session(arm: str, turns: int, provider: str, model: str, nonce: str, client=None) -> list[dict]:
    compiler = make_compiler(provider, model)
    placer = MemoryPlacer(compiler.tokenizer, write_multiplier=compiler.descriptor.cache_write_multiplier,
                          read_multiplier=READ_MULTIPLIER)
    system = Segment("s", "system", f"Session {nonce}.\n" + "\n".join(POLICIES))
    history, rows, said = [], [], {}
    for turn in range(turns):
        mem = memory(turn)
        if arm == "placed":
            front, tail = placer.split(mem, history)
        elif arm == "tail":
            front, tail = [], [Segment(s.id, "memory", s.content, False, provenance=s.provenance) for s in mem]
        else:
            front, tail = mem, []
        text, expected = question(turn)
        ctx = Context([system, *front, *history, *tail, Segment("u", "user", text, stable=False)],
                      cache_namespace=f"{arm}-{nonce}")
        request = compiler.compile(ctx).request
        stale = said.get(text)
        row = {"turn": turn, "tail": [s.id for s in tail], "stale_trap": stale is not None and stale != expected}
        if client is not None:
            response, answer = call(provider, client, request)
            usage, answer = compiler.usage_from_response(response), answer.strip()
            row.update(cached=usage.cache_read_input_tokens, written=usage.cache_creation_input_tokens,
                       uncached=usage.input_tokens, answer=answer, correct=expected.lower() in answer.lower())
        rows.append(row)
        said[text] = expected
        history.append(history_turn(turn, text, expected))
    return rows


def summarize(rows: list[dict], write_multiplier: float) -> dict:
    keys = ("cached", "written", "uncached")
    out = {k: sum(r[k] for r in rows) for k in keys if all(k in r for r in rows)}
    if len(out) == 3:  # input cost in uncached-token units, under the stated multipliers
        out["billed_input_units"] = round(out["uncached"] + write_multiplier * out["written"]
                                          + READ_MULTIPLIER * out["cached"])
    if "correct" in rows[0]:
        traps = [r for r in rows if r["stale_trap"]]
        out["correct"] = f"{sum(r['correct'] for r in rows)}/{len(rows)}"
        out["correct_on_stale_traps"] = f"{sum(r['correct'] for r in traps)}/{len(traps)}"
    return out


def aggregate(runs: list[dict]) -> dict:
    out = {}
    for arm in ARMS:
        billed = [run["summary"][arm]["billed_input_units"] for run in runs if "billed_input_units" in run["summary"][arm]]
        if billed:
            out[arm] = {"billed_mean": round(statistics.mean(billed)), "billed_min": min(billed),
                        "billed_max": max(billed)}
            for key in ("correct", "correct_on_stale_traps"):
                pairs = [run["summary"][arm][key].split("/") for run in runs]
                out[arm][key] = f"{sum(int(a) for a, _ in pairs)}/{sum(int(b) for _, b in pairs)}"
    return out


READ_MULTIPLIER = 0.1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="make paid calls (needs OPENAI_API_KEY or "
                        "OPENROUTER_API_KEY; any value works when a proxy injects the real key)")
    parser.add_argument("--provider", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--model", help="default: gpt-5.6 (openai) or claude-sonnet-5 (anthropic)")
    parser.add_argument("--read-multiplier", type=float, default=READ_MULTIPLIER,
                        help="assumed price of a cached input token relative to uncached (check current pricing)")
    args = parser.parse_args()
    READ_MULTIPLIER = args.read_multiplier
    model = args.model or ("gpt-5.6" if args.provider == "openai" else "claude-sonnet-5")
    client = None
    if args.run:
        key = "OPENAI_API_KEY" if args.provider == "openai" else "OPENROUTER_API_KEY"
        if not os.environ.get(key):
            raise SystemExit(f"--run --provider {args.provider} requires {key}")
        if args.provider == "openai":
            import openai
            client = openai.OpenAI()
        else:
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ[key], base_url="https://openrouter.ai/api")
    writes = make_compiler(args.provider, model).descriptor.cache_write_multiplier
    runs = []
    for _ in range(args.repeats if client else 1):
        nonce = uuid.uuid4().hex[:12] if client else "offline"  # fresh prefix: every arm starts cold
        run = {arm: session(arm, args.turns, args.provider, model, nonce, client) for arm in ARMS}
        run["summary"] = {arm: summarize(run[arm], writes) for arm in ARMS}
        runs.append(run)
    print(json.dumps({"runs": runs, "aggregate": aggregate(runs)}, indent=2))
