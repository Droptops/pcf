#!/usr/bin/env python3
"""Live comparison of memory placement. Offline by default; paid calls need --run.

--provider openai uses the OpenAI Responses API. --provider anthropic sends the Anthropic adapter's Messages
request unchanged to OpenRouter's Anthropic-compatible endpoint, pinned to Anthropic as the upstream provider
(no fallbacks) so prompt caching is Anthropic's own.

One scripted support session is run per arm, each in its own cache namespace, --repeats times:
  front         - all memory before history, most volatile module last (the best hand ordering)
  tail          - all memory after history, just before the user turn
  placed        - MemoryPlacer picks front or tail per module from observed change rates and cache prices
  placed-spacer - placed, plus a ~200-token neutral note between tail memory and the question
Four memory modules change at different rates (never, every 6th turn, every 3rd, every turn). History is
scripted and identical in every arm; each scripted reply states the value it answered, so after a module changes
the history holds a stale answer. --history-style template repeats one long filler pattern in every reply;
varied uses short replies with turn-specific filler.

Grading is strict: the reply's lead value (after "The answer is", else up to the first separator) must equal the
expected value, case-insensitively. A reply is a format violation when it copies history filler or runs past
--violation-tokens (estimated from its visible text). Output tokens are recorded (and OpenAI reasoning tokens;
Anthropic thinking is counted as blocks, since its usage folds thinking into output). Billed units price cache
writes at the descriptor multiplier, reads at --read-multiplier and output at --output-multiplier, all relative to
one uncached input token. --regrade FILE re-grades a saved result with the strict grader, offline.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import statistics
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pcf import Context, Segment  # noqa: E402
from pcf.cache import PrefixCache  # noqa: E402
from pcf.families.anthropic_adapter import AnthropicCompiler  # noqa: E402
from pcf.families.openai_adapter import OpenAICompiler  # noqa: E402
from pcf.placement import MemoryPlacer  # noqa: E402

ARMS = ("front", "tail", "placed", "placed-spacer")
DEFAULT_ARMS = ("front", "tail", "placed")
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
    """(question, the value a correct reply leads with)"""
    text, expected = QUESTIONS[turn % len(QUESTIONS)]
    return text, expected(turn)


TEMPLATE_FILLER = "is on schedule."
VARIED_FILLER = ["The courier scanned parcel {n} at the depot.", "Refund {n} cleared on the second attempt.",
                 "Survey {n} was sent after the last call.", "Warranty case {n} is waiting on a photo."]


def history_turn(turn: int, text: str, answer: str, style: str) -> Segment:
    if style == "template":
        reply = f"The answer is {answer}. " + " ".join(f"Order {9000 + turn * 10 + k} {TEMPLATE_FILLER}" for k in range(30))
    else:
        reply = f"The answer is {answer}. " + VARIED_FILLER[turn % len(VARIED_FILLER)].format(n=7000 + turn)
    return Segment(f"h{turn}", "history", [{"role": "user", "content": text}, {"role": "assistant", "content": reply}])


SPACER = Segment("notice", "memory", {"notice": " ".join(
    f"Service notice {k}: branch hours and holiday schedules are posted on the help centre." for k in range(12))},
    False, provenance="notice")
SEPARATORS = re.compile(r"\s+[-\u2013\u2014]\s+|[(\u2013\u2014\n.;:,!]")


def lead_value(answer: str) -> str:
    lower = answer.lower()
    if "the answer is" in lower:
        answer = answer[lower.index("the answer is") + len("the answer is"):]
    return SEPARATORS.split(answer.strip(), 1)[0].strip(" *\"'`").lower()


def grade(answer: str, expected: str, stale: str | None, style: str, violation_tokens: int) -> dict:
    lead = lead_value(answer)
    fillers = [TEMPLATE_FILLER] if style == "template" else [f.split("{n}")[1].strip() for f in VARIED_FILLER]
    copied = any(f.lower() in answer.lower() for f in fillers)
    trap = stale is not None and stale != expected
    return {"lead": lead, "correct": lead == expected.lower(), "gave_stale": trap and lead == stale.lower(),
            "violation": copied or math.ceil(len(answer) / 4) > violation_tokens}


def make_compiler(provider: str, model: str):
    return OpenAICompiler(model) if provider == "openai" else AnthropicCompiler(model, max_tokens=512)


def call(provider: str, client, request: dict, cfg) -> tuple[object, str, dict]:
    """(response, answer text, output usage) for one compiled request."""
    if provider == "openai":
        # Low effort and room to answer: at 64 tokens reasoning models can return nothing visible.
        response = client.responses.create(**request, max_output_tokens=512, reasoning={"effort": cfg.effort})
        details = getattr(response.usage, "output_tokens_details", None)
        out = {"output_tokens": response.usage.output_tokens,
               "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0, "served_model": response.model}
        return response, getattr(response, "output_text", "") or "", out
    extra = {"provider": {"order": ["Anthropic"], "allow_fallbacks": False}}
    thinking = {"thinking": {"type": "disabled"}} if cfg.thinking == "disabled" else {}
    response = client.messages.create(**{**request, "model": "anthropic/" + request["model"], **thinking},
                                      extra_body=extra)
    out = {"output_tokens": response.usage.output_tokens, "served_model": response.model,
           "thinking_blocks": sum(getattr(b, "type", "") == "thinking" for b in response.content)}
    return response, "".join(getattr(block, "text", "") for block in response.content), out


def arrange(arm: str, placer: MemoryPlacer, mem: list[Segment], history: list[Segment]):
    if arm in {"placed", "placed-spacer"}:
        front, tail = placer.split(mem, history)
        return front, [*tail, SPACER] if arm == "placed-spacer" else tail
    if arm == "tail":
        return [], [Segment(s.id, "memory", s.content, False, provenance=s.provenance) for s in mem]
    return mem, []


def session(arm: str, nonce: str, cfg, client=None) -> list[dict]:
    compiler = make_compiler(cfg.provider, cfg.model)
    placer = MemoryPlacer(compiler.tokenizer, write_multiplier=compiler.descriptor.cache_write_multiplier,
                          read_multiplier=cfg.read_multiplier)
    system = Segment("s", "system", f"Session {nonce}-{arm}.\n" + "\n".join(POLICIES))
    history, rows, said = [], [], {}
    for turn in range(cfg.turns):
        front, tail = arrange(arm, placer, memory(turn), history)
        text, expected = question(turn)
        ctx = Context([system, *front, *history, *tail, Segment("u", "user", text, stable=False)],
                      cache_namespace=f"{arm}-{nonce}")
        compiled = compiler.compile(ctx)
        estimate = compiler.warmth(ctx, PrefixCache(compiler.descriptor.ttl_seconds), 0.0)
        stale = said.get(text)
        row = {"turn": turn, "tail": [s.id for s in tail], "expected": expected, "stale": stale,
               "stale_trap": stale is not None and stale != expected, "est_tokens": compiled.total_tokens,
               "est_written": estimate.cache_creation_tokens}
        if client is not None:
            response, answer, out = call(cfg.provider, client, compiled.request, cfg)
            usage, answer = compiler.usage_from_response(response), answer.strip()
            row.update(cached=usage.cache_read_input_tokens, written=usage.cache_creation_input_tokens,
                       uncached=usage.input_tokens, answer=answer, **out,
                       **grade(answer, expected, stale, cfg.history_style, cfg.violation_tokens))
        rows.append(row)
        said[text] = expected
        history.append(history_turn(turn, text, expected, cfg.history_style))
    return rows


def summarize(rows: list[dict], cfg, write_multiplier: float) -> dict:
    keys = ("cached", "written", "uncached", "output_tokens")
    out = {k: sum(r[k] for r in rows) for k in keys if all(k in r for r in rows)}
    if len(out) == 4:  # costs in uncached-input-token units, under the stated multipliers
        billed = out["uncached"] + write_multiplier * out["written"] + cfg.read_multiplier * out["cached"]
        out["billed_input_units"] = round(billed)
        out["billed_total_units"] = round(billed + cfg.output_multiplier * out["output_tokens"])
    if "correct" in rows[0]:
        traps = [r for r in rows if r["stale_trap"]]
        out["correct"] = f"{sum(r['correct'] for r in rows)}/{len(rows)}"
        out["correct_on_stale_traps"] = f"{sum(r['correct'] for r in traps)}/{len(traps)}"
        out["gave_stale"] = sum(r["gave_stale"] for r in rows)
        out["violations"] = f"{sum(r['violation'] for r in rows)}/{len(rows)}"
    return out


def aggregate(runs: list[dict], arms) -> dict:
    out = {}
    for arm in arms:
        summaries = [run["summary"][arm] for run in runs]
        if "billed_input_units" not in summaries[0]:
            continue
        out[arm] = {}
        for key in ("billed_input_units", "billed_total_units"):
            values = [s[key] for s in summaries]
            out[arm][key] = {"mean": round(statistics.mean(values)), "min": min(values), "max": max(values)}
        for key in ("correct", "correct_on_stale_traps", "violations"):
            pairs = [s[key].split("/") for s in summaries]
            out[arm][key] = f"{sum(int(a) for a, _ in pairs)}/{sum(int(b) for _, b in pairs)}"
        out[arm]["gave_stale"] = sum(s["gave_stale"] for s in summaries)
    return out


def regrade(path: str, violation_tokens: int) -> dict:
    """Re-grade a saved result offline: expected and stale values are recomputed from the scripted turns."""
    saved = json.load(open(path))
    style = saved.get("meta", {}).get("history_style", "template")
    changes = []
    for i, run in enumerate(saved["runs"]):
        for arm, rows in run.items():
            if arm == "summary" or not rows or "answer" not in rows[0]:
                continue
            said = {}
            for row in rows:
                text, expected = question(row["turn"])
                old = row.get("correct")
                row.update(expected=expected, stale=said.get(text), stale_trap=said.get(text) not in (None, expected),
                           **grade(row["answer"], expected, said.get(text), style, violation_tokens))
                if old is not None and old != row["correct"]:
                    changes.append({"run": i, "arm": arm, "turn": row["turn"], "answer": row["answer"][:80],
                                    "was": old, "now": row["correct"]})
                said[text] = expected
    return {"file": path, "grade_changes": changes, "runs": saved["runs"]}


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(__file__), check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="make paid calls (needs OPENAI_API_KEY or "
                        "OPENROUTER_API_KEY; any value works when a proxy injects the real key)")
    parser.add_argument("--provider", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(DEFAULT_ARMS))
    parser.add_argument("--model", help="default: gpt-5.6 (openai) or claude-sonnet-5 (anthropic)")
    parser.add_argument("--history-style", choices=("template", "varied"), default="template")
    parser.add_argument("--thinking", choices=("default", "disabled"), default="default",
                        help="anthropic only: send thinking disabled, or omit the parameter")
    parser.add_argument("--effort", default="low", help="openai reasoning effort")
    parser.add_argument("--read-multiplier", type=float, default=0.1,
                        help="price of a cached input token relative to uncached (check current pricing)")
    parser.add_argument("--output-multiplier", type=float, default=5.0,
                        help="price of an output token relative to an uncached input token (check current pricing)")
    parser.add_argument("--violation-tokens", type=int, default=60)
    parser.add_argument("--workers", type=int, default=1, help="sessions run concurrently (paid runs only)")
    parser.add_argument("--regrade", metavar="FILE", help="re-grade a saved result offline and print the changes")
    cfg = parser.parse_args()
    if cfg.regrade:
        print(json.dumps(regrade(cfg.regrade, cfg.violation_tokens), indent=2))
        raise SystemExit
    cfg.model = cfg.model or ("gpt-5.6" if cfg.provider == "openai" else "claude-sonnet-5")
    client = None
    if cfg.run:
        key = "OPENAI_API_KEY" if cfg.provider == "openai" else "OPENROUTER_API_KEY"
        if not os.environ.get(key):
            raise SystemExit(f"--run --provider {cfg.provider} requires {key}")
        if cfg.provider == "openai":
            import openai
            client = openai.OpenAI()
        else:
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ[key], base_url="https://openrouter.ai/api")
    writes = make_compiler(cfg.provider, cfg.model).descriptor.cache_write_multiplier
    nonces = [uuid.uuid4().hex[:12] if client else "offline" for _ in range(cfg.repeats if client else 1)]
    jobs = [(i, arm) for i in range(len(nonces)) for arm in cfg.arms]  # fresh prefix per repeat: arms start cold
    with ThreadPoolExecutor(max(1, cfg.workers if client else 1)) as pool:
        done = dict(zip(jobs, pool.map(lambda job: session(job[1], nonces[job[0]], cfg, client), jobs)))
    runs = []
    for i in range(len(nonces)):
        run = {arm: done[(i, arm)] for arm in cfg.arms}
        run["summary"] = {arm: summarize(run[arm], cfg, writes) for arm in cfg.arms}
        runs.append(run)
    meta = {"date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "git_sha": git_sha(),
            "provider": cfg.provider, "model": cfg.model, "turns": cfg.turns, "repeats": len(nonces),
            "arms": cfg.arms, "history_style": cfg.history_style, "thinking": cfg.thinking, "effort": cfg.effort,
            "write_multiplier": writes, "read_multiplier": cfg.read_multiplier,
            "output_multiplier": cfg.output_multiplier, "paid": client is not None}
    print(json.dumps({"meta": meta, "runs": runs, "aggregate": aggregate(runs, cfg.arms)}, indent=2))
