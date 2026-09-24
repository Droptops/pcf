#!/usr/bin/env python3
"""Safe provider smoke test: offline by default; paid calls require --run."""
from __future__ import annotations
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pcf import Context, Segment  # noqa: E402
from pcf.families.anthropic_adapter import AnthropicCompiler  # noqa: E402
from pcf.families.openai_adapter import OpenAICompiler  # noqa: E402

def context() -> Context:
    return Context([Segment("s", "system", "You are a concise support assistant."),
                    Segment("u", "user", "Return the word READY.", stable=False)])

def offline() -> dict:
    ctx = context()
    a = AnthropicCompiler("claude-sonnet-5").compile(ctx).request
    o = OpenAICompiler("gpt-5.6").compile(ctx).request
    assert a["messages"] and o["input"]
    return {"offline": True, "anthropic": {"model": a["model"], "message_count": len(a["messages"])},
            "openai": {"model": o["model"], "input_count": len(o["input"])} }

def run_live() -> dict:
    result = offline()
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        comp = AnthropicCompiler("claude-sonnet-5")
        response = anthropic.Anthropic().messages.create(**comp.compile(context()).request)
        result["anthropic_live"] = comp.usage_from_response(response).__dict__
    if os.environ.get("OPENAI_API_KEY"):
        import openai
        comp = OpenAICompiler("gpt-5.6")
        response = openai.OpenAI().responses.create(**comp.compile(context()).request)
        result["openai_live"] = comp.usage_from_response(response).__dict__
    if not any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")):
        raise RuntimeError("--run requires ANTHROPIC_API_KEY or OPENAI_API_KEY")
    result["offline"] = False
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="make paid provider calls")
    args = parser.parse_args()
    print(json.dumps(run_live() if args.run else offline(), indent=2, sort_keys=True))
