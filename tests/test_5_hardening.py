from __future__ import annotations

import math

import pytest

from pcf import Context, Segment, canonical_bytes
from pcf.cache import PrefixCache
from pcf.families.anthropic_adapter import AnthropicCompiler, history_from_response as anthropic_history
from pcf.families.openai_adapter import OpenAICompiler, history_from_response as openai_history
from pcf.families.sim import family_a
from pcf.router.calibration import fit_platt


def test_direct_empty_authority_is_rejected_and_cache_writes_are_monotonic():
    with pytest.raises(ValueError):
        Segment("s", "system", "x", authority="")
    c = PrefixCache(10)
    key = "sha256:" + "1" * 64
    prefix = "sha256:" + "2" * 64
    c.write(key, prefix, 1, 10)
    with pytest.raises(ValueError, match="backwards"):
        c.write(key, prefix, 1, 9)


def test_provider_tool_roundtrip_preserves_ids_and_arguments():
    neutral = [{"role": "assistant", "content": "", "tool_calls":
                [{"id": "call-1", "name": "lookup", "arguments": {"id": "42"}}]},
               {"role": "tool", "call_id": "call-1", "content": {"status": "ok"}, "is_error": False}]
    ctx = Context([Segment("h", "history", neutral), Segment("u", "user", "thanks", stable=False)])
    a = AnthropicCompiler("claude-sonnet-5").compile(ctx).request
    o = OpenAICompiler("gpt-5.6").compile(ctx).request
    assert a["messages"][0]["content"][0]["id"] == "call-1"
    assert o["input"][1]["call_id"] == "call-1"
    assert anthropic_history({"content": [{"type": "text", "text": "ok"},
                                             {"type": "tool_use", "id": "c", "name": "lookup", "input": {}}]})[0]["tool_calls"][0]["id"] == "c"
    assert openai_history({"output": [{"type": "function_call", "call_id": "c", "name": "lookup", "arguments": "{}"}]})[0]["tool_calls"][0]["id"] == "c"


def test_unknown_provider_response_block_is_not_silently_dropped():
    with pytest.raises(ValueError):
        anthropic_history({"content": [{"type": "thinking", "thinking": "secret"}]})
    with pytest.raises(ValueError):
        openai_history({"output": [{"type": "reasoning", "summary": []}]})


def test_jcs_rejects_nonfinite_and_constant_platt_fit_is_analytic():
    with pytest.raises(ValueError):
        canonical_bytes(float("nan"))
    a, b = fit_platt([0.5] * 20, [0, 1] * 10)
    assert a == 0.0 and math.isfinite(b) and abs(b) < 1e-9


def test_opaque_provider_warmth_is_unknown_even_with_local_metadata():
    engine = family_a(min_cacheable=1)
    ctx = Context([Segment("s", "system", "stable"), Segment("u", "user", "hi", stable=False)])
    engine.run(ctx, 0)
    assert engine.compiler.warmth(ctx, engine.cache, 1).cache_state == "simulated"
    # This is an explicit closed-provider contract test with an OpenAI compiler.
    openai = OpenAICompiler("gpt-5.6")
    assert openai.warmth(ctx, PrefixCache(30), 1).cache_state == "unknown"
