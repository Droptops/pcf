"""Provider reasoning state (Anthropic thinking, OpenAI reasoning items) survives the round trip through history."""
from __future__ import annotations

import pytest

from pcf import Context, Segment
from pcf.families.anthropic_adapter import AnthropicCompiler, history_from_response as anthropic_history
from pcf.families.openai_adapter import OpenAICompiler, history_from_response as openai_history
from pcf.schemas import validate

THINKING = {"type": "thinking", "thinking": "", "signature": "sig-abc"}
SONNET_RESPONSE = {"content": [THINKING, {"type": "text", "text": "Checking."},
                               {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}}]}
REASONING = {"type": "reasoning", "id": "rs_1", "summary": []}
GPT_RESPONSE = {"output": [REASONING, {"type": "function_call", "call_id": "call_1", "name": "lookup",
                                       "arguments": '{"q": "x"}'}]}
TOOLS = Segment("t", "tools", [{"name": "lookup", "parameters": {"type": "object"}}])


def _round(turns, call_id):
    result = {"role": "tool", "call_id": call_id, "content": "found", "is_error": False}
    return Context([TOOLS, Segment("s", "system", "sys"),
                    Segment("h", "history", [{"role": "user", "content": "find x"}, *turns, result])])


def test_anthropic_thinking_is_kept_and_replayed_first_or_dropped():
    turns = anthropic_history(SONNET_RESPONSE)
    assert turns[0]["provider_blocks"] == [{"provider": "anthropic", "block": THINKING}]
    ctx = _round(turns, "toolu_1")
    replayed = AnthropicCompiler("claude-sonnet-5").compile(ctx).request["messages"][1]["content"]
    assert [b["type"] for b in replayed] == ["thinking", "text", "tool_use"] and replayed[0] == THINKING
    dropped = AnthropicCompiler("claude-sonnet-5", thinking_blocks="drop").compile(ctx).request["messages"][1]
    assert [b["type"] for b in dropped["content"]] == ["text", "tool_use"]
    # Another provider never sees Anthropic's blocks.
    assert all(item.get("type") != "thinking" for item in OpenAICompiler("gpt-5.6").compile(ctx).request["input"])


def test_openai_reasoning_is_kept_and_replayed_before_its_turn_or_dropped():
    turns = openai_history(GPT_RESPONSE)
    assert turns[0]["provider_blocks"] == [{"provider": "openai", "block": REASONING}]
    ctx = _round(turns, "call_1")
    items = OpenAICompiler("gpt-5.6").compile(ctx).request["input"]
    assert [i.get("type", i.get("role")) for i in items] == ["developer", "user", "reasoning", "function_call",
                                                              "function_call_output"]
    dropped = OpenAICompiler("gpt-5.6", reasoning_items="drop").compile(ctx).request["input"]
    assert "reasoning" not in [i.get("type") for i in dropped]
    request = AnthropicCompiler("claude-sonnet-5").compile(ctx).request
    assert all(b["type"] != "reasoning" for m in request["messages"] for b in m["content"])


@pytest.mark.parametrize("response, error", [
    ({"content": [{"type": "text", "text": "a"}, THINKING]}, "thinking after text"),
    ({"content": [THINKING]}, "thinking without text"),
])
def test_anthropic_thinking_that_cannot_be_replayed_in_order_is_rejected(response, error):
    with pytest.raises(ValueError, match=error):
        anthropic_history(response)


@pytest.mark.parametrize("output, error", [
    ([REASONING], "reasoning without"),
    ([{"type": "message", "content": [{"type": "output_text", "text": "a"}]}, REASONING,
      {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"}], "between output items"),
])
def test_openai_reasoning_that_cannot_be_replayed_in_order_is_rejected(output, error):
    with pytest.raises(ValueError, match=error):
        openai_history({"output": output})


def test_provider_blocks_are_validated_hashed_and_schema_valid():
    turn = anthropic_history(SONNET_RESPONSE)[0]
    plain = {k: v for k, v in turn.items() if k != "provider_blocks"}
    assert Segment("h", "history", [turn]).hash != Segment("h", "history", [plain]).hash
    for bad in ([{"provider": "gemini", "block": {"type": "x"}}], [{"provider": "anthropic", "block": {}}],
                [{"provider": "anthropic"}]):
        with pytest.raises(ValueError):
            Segment("h", "history", [{**plain, "provider_blocks": bad}])
    with pytest.raises(ValueError, match="only assistant"):
        Segment("h", "history", [{"role": "user", "content": "q", "provider_blocks": turn["provider_blocks"]}])
    validate("pcf", _round([turn], "toolu_1").to_json())


def test_anthropic_thinking_config_is_sent_and_checked():
    ctx = Context([Segment("s", "system", "sys"), Segment("u", "user", "hi", stable=False)])
    assert AnthropicCompiler("claude-sonnet-5", thinking={"type": "disabled"}).compile(ctx).request["thinking"] == {
        "type": "disabled"}
    assert "thinking" not in AnthropicCompiler("claude-sonnet-5").compile(ctx).request
    for bad in ({"thinking": "adaptive"}, {"thinking_blocks": "keep"}):
        with pytest.raises(ValueError):
            AnthropicCompiler("claude-sonnet-5", **bad)
    with pytest.raises(ValueError):
        OpenAICompiler("gpt-5.6", reasoning_items="keep")
