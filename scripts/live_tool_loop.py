#!/usr/bin/env python3
"""Live check that tool loops keep hitting the provider cache. Offline by default; paid calls need --run.

A scripted agent loop, appended as [user, call] and [tool] history segments (the per-message shape that re-billed
all history under an earlier breakpoint rule), with four tail memory modules after history. Every request after
the first should read back everything the previous request read or wrote, because history is append-only and tail
memory is never marked. The model's replies are ignored: history is scripted so every run sends the same prefixes.

--provider openai uses gpt-5.6 in explicit cache mode; --provider anthropic sends the Anthropic adapter's request
to OpenRouter's Anthropic-compatible endpoint pinned to Anthropic, with thinking disabled so the request carries no
thinking blocks. Exit status is 1 when any request reads back less than --min-reuse of the previous request's
cached prefix, when a request's cached + written tokens fall short of --min-coverage of the estimated prefix up to its
last marker (history never cached), or when nothing after the first request is read.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pcf import Context, Segment  # noqa: E402
from pcf.families.anthropic_adapter import AnthropicCompiler  # noqa: E402
from pcf.families.openai_adapter import OpenAICompiler  # noqa: E402

POLICIES = [f"Policy {i}: look up the order before answering, cite its id, and keep replies to one sentence."
            for i in range(80)]
TOOLS = [{"name": "lookup_order", "description": "Look up an order by id.",
          "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}]


def tail_memory(step: int) -> list[Segment]:
    modules = {"profile": {"name": "Dana Ruiz", "tier": "gold"}, "cart": {"items": step + 1},
               "session": {"step": step}, "notes": {"last_lookup": f"O-{500 + step}"}}
    return [Segment(name, "memory", data, False, provenance=name) for name, data in modules.items()]


def requests(iterations: int, nonce: str):
    """Yield (label, Context) for each request of the scripted loop."""
    base = [Segment("t", "tools", TOOLS), Segment("s", "system", f"Session {nonce}.\n" + "\n".join(POLICIES))]
    history = []
    for k in range(iterations):
        question = f"Where is order O-{500 + k}?"
        yield f"q{k}", Context([*base, *history, *tail_memory(2 * k), Segment("u", "user", question, stable=False)])
        call = {"id": f"call_{k}", "name": "lookup_order", "arguments": {"order_id": f"O-{500 + k}"}}
        history.append(Segment(f"q{k}", "history", [{"role": "user", "content": question},
                                                   {"role": "assistant", "content": "", "tool_calls": [call]}]))
        history.append(Segment(f"r{k}", "history", [{"role": "tool", "call_id": f"call_{k}", "is_error": False,
                                                   "content": f"Order O-{500 + k} shipped; tracking T{9000 + k}. "
                                                              + "Scan history: depot, hub, van. " * 20}]))
        yield f"r{k}", Context([*base, *history, *tail_memory(2 * k + 1)])
        history.append(Segment(f"a{k}", "history", [{"role": "assistant",
                                                   "content": f"Order O-{500 + k} shipped, tracking T{9000 + k}."}]))


def send(provider: str, client, request: dict):
    if provider == "openai":
        return client.responses.create(**request, max_output_tokens=64, reasoning={"effort": "low"})
    model = "anthropic/" + re.sub(r"-(\d+)-(\d+)$", r"-\1.\2", request["model"])  # OpenRouter's model name
    return client.messages.create(**{**request, "model": model, "max_tokens": 64,
                                     "thinking": {"type": "disabled"}},
                                  extra_body={"provider": {"order": ["Anthropic"], "allow_fallbacks": False}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--provider", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--model", help="default: gpt-5.6 (openai) or claude-sonnet-5 (anthropic)")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--min-reuse", type=float, default=0.95)
    parser.add_argument("--min-coverage", type=float, default=0.7,
                        help="cached + written must reach this share of the estimated prefix up to the last marker")
    args = parser.parse_args()
    model = args.model or ("gpt-5.6" if args.provider == "openai" else "claude-sonnet-5")
    compiler = OpenAICompiler(model) if args.provider == "openai" else AnthropicCompiler(model)
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
    rows, previous, failed = [], None, False
    for label, ctx in requests(args.iterations, uuid.uuid4().hex[:12] if client else "offline"):
        compiled = compiler.compile(ctx)
        cum = compiled.cum_tokens
        covered = compiler.covered_tokens(ctx, compiled.breakpoints[-1], cum) if compiled.breakpoints else 0
        row = {"request": label, "marked": [ctx.segments[i].id for i in compiled.breakpoints],
               "est_tokens": compiled.total_tokens, "est_covered": covered}
        if client is not None:
            usage = compiler.usage_from_response(send(args.provider, client, compiled.request))
            row.update(cached=usage.cache_read_input_tokens, written=usage.cache_creation_input_tokens,
                       uncached=usage.input_tokens)
            if previous is not None:
                expected = previous["cached"] + previous["written"]
                row["reuse"] = round(row["cached"] / expected, 3) if expected else None
                failed |= expected > 0 and row["cached"] < args.min_reuse * expected
            # The cache must reach this request's last marker (estimates run 0.86-1.4x provider counts).
            row["reaches_marker"] = row["cached"] + row["written"] >= args.min_coverage * covered
            failed |= not row["reaches_marker"]
            previous = row
        rows.append(row)
    if client is not None and not any(r["cached"] for r in rows[1:]):
        failed = True  # nothing after the first request was ever read
    print(json.dumps({"provider": args.provider, "model": model, "paid": client is not None, "rows": rows,
                      "ok": not failed}, indent=1))
    raise SystemExit(1 if failed else 0)
